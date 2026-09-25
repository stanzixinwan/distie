"""HuggingFace causal-LM engine with optional paged KV.

Without a PagedKvCache, BlockPool is occupancy-only and HuggingFace owns
past_key_values. With one, C++ Block IDs are the page table and this engine
scatter/gathers KV so the HF cache is no longer the source of truth.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Sequence

from inference.engine import (
    BlockAllocator,
    GenerateRequest,
    TokenEvent,
    clamp_max_tokens,
    release_blocks,
    reserve_blocks,
)
from inference.kv_cache import KvShape, PagedKvCache

_log = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def resolve_device(requested: str) -> str:
    device = requested.strip().lower()
    if device not in {"cuda", "cpu"}:
        raise ValueError("WORKER_DEVICE must be 'cuda' or 'cpu'")
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "WORKER_DEVICE=cuda but torch.cuda.is_available() is False. "
            "Install a CUDA build of torch or set WORKER_DEVICE=cpu."
        )
    return device


class TorchEngine:
    def __init__(
        self,
        model,
        tokenizer,
        device: str,
        pool: BlockAllocator | None = None,
        kv_cache: PagedKvCache | None = None,
    ) -> None:
        if kv_cache is not None and pool is None:
            raise ValueError("PagedKvCache requires a BlockAllocator")
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._pool = pool
        self._kv = kv_cache

    @classmethod
    def load(
        cls,
        model_id: str,
        device: str = "cuda",
        pool: BlockAllocator | None = None,
    ) -> TorchEngine:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        resolved = resolve_device(device)
        dtype = torch.float16 if resolved == "cuda" else torch.float32
        _log.info("loading torch model id=%s device=%s dtype=%s", model_id, resolved, dtype)

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        model.to(resolved)
        model.eval()
        kv_cache = _maybe_paged_cache(model, pool, resolved)
        _log.info("torch model ready id=%s device=%s paged_kv=%s", model_id, resolved, kv_cache is not None)
        return cls(model, tokenizer, resolved, pool, kv_cache)

    async def generate(self, req: GenerateRequest) -> AsyncIterator[TokenEvent]:
        started = time.perf_counter()
        cap = clamp_max_tokens(req.max_tokens)
        prompt_ids = self._encode(req.prompt)
        prompt_tokens = len(prompt_ids)
        block_ids: list[int] = []
        ttft_ms = 0.0

        try:
            block_ids = self._reserve(req.request_id, prompt_tokens, cap)
            if not prompt_ids:
                now = time.perf_counter()
                yield TokenEvent(
                    token="",
                    finished=True,
                    prompt_tokens=0,
                    completion_tokens=0,
                    time_to_first_token_ms=0.0,
                    total_latency_ms=_elapsed_ms(started, now),
                )
                return

            loop = asyncio.get_running_loop()
            current = prompt_ids
            past = None
            seq_len = 0
            eos_id = self._tokenizer.eos_token_id

            for step in range(cap):
                logits, past, seq_len = await loop.run_in_executor(
                    None, self._forward_step, current, past, block_ids, seq_len
                )
                token_id = _pick_token(logits, req.temperature)
                now = time.perf_counter()
                if step == 0:
                    ttft_ms = _elapsed_ms(started, now)
                text = self._tokenizer.decode(
                    [token_id], skip_special_tokens=True
                )
                finished = token_id == eos_id or step == cap - 1
                yield TokenEvent(
                    token=text,
                    finished=finished,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=step + 1,
                    time_to_first_token_ms=ttft_ms if finished else 0.0,
                    total_latency_ms=_elapsed_ms(started, now) if finished else 0.0,
                )
                if token_id == eos_id:
                    return
                current = [token_id]
        finally:
            release_blocks(self._pool, req.request_id, block_ids)

    def _reserve(self, request_id: str, prompt_tokens: int, cap: int) -> list[int]:
        if self._kv is None:
            return reserve_blocks(self._pool, request_id, cap)
        # Pages must cover prompt + new tokens; occupancy-only reserved cap.
        return reserve_blocks(
            self._pool,
            request_id,
            prompt_tokens + cap,
            tokens_per_block=self._kv.page_size,
        )

    def _encode(self, prompt: str) -> list[int]:
        if getattr(self._tokenizer, "chat_template", None):
            return self._tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=True,
            )
        return self._tokenizer.encode(prompt, add_special_tokens=True)

    def _forward_step(
        self,
        input_ids: list[int],
        past,
        block_ids: Sequence[int],
        seq_len: int,
    ):
        import torch

        with torch.inference_mode():
            hf_past = past
            if self._kv is not None:
                gathered = self._kv.gather(block_ids, seq_len)
                hf_past = None if gathered is None else _to_hf_past(*gathered)

            tensor = torch.tensor([input_ids], dtype=torch.long, device=self._device)
            out = self._model(
                input_ids=tensor,
                past_key_values=hf_past,
                use_cache=True,
            )
            logits = out.logits[0, -1, :].float().cpu()
            new_seq = seq_len + len(input_ids)
            if self._kv is not None:
                keys, values = _from_hf_cache(out.past_key_values)
                self._kv.scatter(block_ids, seq_len, keys, values)
                return logits, None, new_seq
            return logits, out.past_key_values, new_seq


def _maybe_paged_cache(model, pool: BlockAllocator | None, device: str) -> PagedKvCache | None:
    if pool is None:
        return None
    num_blocks = getattr(pool, "num_blocks", None)
    config = getattr(model, "config", None)
    if num_blocks is None or config is None:
        _log.warning("skipping paged KV: pool.num_blocks or model.config missing")
        return None
    import torch

    dtype = next(model.parameters()).dtype
    shape = KvShape.from_hf_config(config)
    return PagedKvCache(shape, num_blocks=int(num_blocks), device=device, dtype=dtype)


def _from_hf_cache(cache) -> tuple[list, list]:
    if hasattr(cache, "to_legacy_cache"):
        cache = cache.to_legacy_cache()
    keys = [layer[0] for layer in cache]
    values = [layer[1] for layer in cache]
    return keys, values


def _to_hf_past(keys: Sequence, values: Sequence):
    legacy = tuple(zip(keys, values, strict=True))
    try:
        from transformers.cache_utils import DynamicCache

        if hasattr(DynamicCache, "from_legacy_cache"):
            return DynamicCache.from_legacy_cache(legacy)
    except Exception:
        pass
    return legacy


def _pick_token(logits, temperature: float) -> int:
    import torch

    if temperature <= 0:
        return int(torch.argmax(logits).item())
    probs = torch.softmax(logits / temperature, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def _elapsed_ms(started: float, now: float) -> float:
    return (now - started) * 1000.0
