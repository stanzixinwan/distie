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
from dataclasses import dataclass
from typing import Any

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


_DTYPES = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}


def resolve_dtype(requested: str, device: str):
    """'auto' keeps the serving default: fp16 on CUDA, fp32 on CPU."""
    import torch

    name = requested.strip().lower()
    if name == "auto":
        return torch.float16 if device == "cuda" else torch.float32
    if name not in _DTYPES:
        raise ValueError(f"dtype must be one of auto, {', '.join(_DTYPES)}")
    return getattr(torch, _DTYPES[name])


@dataclass(frozen=True)
class Trace:
    """Per-step output of one sequence through the serving path.

    token_ids[t] is the argmax at step t; logits is [steps, vocab] float32 on CPU.
    """

    token_ids: list[int]
    logits: Any


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

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def paged(self) -> bool:
        return self._kv is not None

    @classmethod
    def load(
        cls,
        model_id: str,
        device: str = "cuda",
        pool: BlockAllocator | None = None,
        dtype: str = "auto",
    ) -> TorchEngine:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        resolved = resolve_device(device)
        torch_dtype = resolve_dtype(dtype, resolved)
        _log.info("loading torch model id=%s device=%s dtype=%s", model_id, resolved, torch_dtype)

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
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
        prompt_ids = self.encode(req.prompt)
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

    def trace(
        self,
        prompt_ids: Sequence[int],
        max_new_tokens: int,
        forced_ids: Sequence[int] | None = None,
    ) -> Trace:
        """Run one sequence synchronously through the same KV path as generate.

        Free-running (forced_ids=None) feeds back the greedy pick and stops
        after EOS. Teacher-forced feeds forced_ids so logits can be compared
        step-by-step against a reference that saw the same prefix.
        """
        import torch

        if not prompt_ids:
            raise ValueError("prompt_ids must be non-empty")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        if forced_ids is not None and len(forced_ids) < max_new_tokens:
            raise ValueError("forced_ids shorter than max_new_tokens")

        request_id = "trace"
        block_ids = self._reserve(request_id, len(prompt_ids), max_new_tokens)
        eos_id = self._tokenizer.eos_token_id
        picks: list[int] = []
        rows: list = []
        try:
            current = list(prompt_ids)
            past = None
            seq_len = 0
            for step in range(max_new_tokens):
                logits, past, seq_len = self._forward_step(current, past, block_ids, seq_len)
                pick = int(torch.argmax(logits).item())
                picks.append(pick)
                rows.append(logits)
                if forced_ids is None:
                    if pick == eos_id:
                        break
                    current = [pick]
                else:
                    current = [int(forced_ids[step])]
        finally:
            release_blocks(self._pool, request_id, block_ids)
        return Trace(token_ids=picks, logits=torch.stack(rows))

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

    def encode(self, prompt: str) -> list[int]:
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
    keys = [layer.keys for layer in cache.layers]
    values = [layer.values for layer in cache.layers]
    return keys, values


def _to_hf_past(keys: Sequence, values: Sequence):
    from transformers.cache_utils import DynamicCache

    return DynamicCache(list(zip(keys, values, strict=True)))


def _pick_token(logits, temperature: float) -> int:
    import torch

    if temperature <= 0:
        return int(torch.argmax(logits).item())
    probs = torch.softmax(logits / temperature, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def _elapsed_ms(started: float, now: float) -> float:
    return (now - started) * 1000.0
