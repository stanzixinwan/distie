"""Causal-LM engine.

With a PagedKvCache the model is the in-house Qwen2, which reads and writes
the slab directly, and generate() joins one shared BatchLoop: concurrent
requests are batched step by step instead of running one after another.
Without one, a caller-supplied module (tests, or a plain HuggingFace model)
still owns its own past_key_values and runs one request at a time.
"""

from __future__ import annotations

import asyncio
import logging
import os
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
from inference.batch_loop import BatchLoop
from inference.kv_cache import KvShape, PagedKvCache, pages_needed
from inference.model_runner import ModelRunner
from inference.qwen2.decode_graph import DecodeGraph
from inference.qwen2.paged_attn import flash_enabled, serving_page_size
from inference.qwen2 import Qwen2CausalLM
from inference.scheduler import DEFAULT_MAX_SEQS, DEFAULT_TOKEN_BUDGET, Scheduler, Sequence as SchedSeq

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
        self._decode_graph: DecodeGraph | None = None
        self._decode_graph_key: int | None = None
        self._decode_graph_failed = False
        self._loop: BatchLoop | None = None
        self._next_seq = 0

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
        resolved = resolve_device(device)
        torch_dtype = resolve_dtype(dtype, resolved)
        _log.info("loading torch model id=%s device=%s dtype=%s", model_id, resolved, torch_dtype)

        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        hf_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch_dtype,
            trust_remote_code=True,
        )
        hf_model.to(resolved)
        hf_model.eval()
        model = Qwen2CausalLM.from_hf(hf_model)
        del hf_model
        if resolved == "cuda":
            import torch

            torch.cuda.empty_cache()
        kv_cache = _maybe_paged_cache(model, pool, resolved)
        _log.info("torch model ready id=%s device=%s paged_kv=%s", model_id, resolved, kv_cache is not None)
        return cls(model, tokenizer, resolved, pool, kv_cache)

    async def generate(self, req: GenerateRequest) -> AsyncIterator[TokenEvent]:
        inner = self._generate_single(req) if self._kv is None else self._generate_batched(req)
        try:
            async for event in inner:
                yield event
        finally:
            # Closing this generator does not close `inner`; its finally releases blocks.
            await inner.aclose()

    async def close(self) -> None:
        if self._loop is not None:
            await self._loop.close()
            self._loop = None

    async def _generate_batched(self, req: GenerateRequest) -> AsyncIterator[TokenEvent]:
        """Join the shared scheduler loop; this coroutine only waits on its own queue."""
        started = time.perf_counter()
        cap = clamp_max_tokens(req.max_tokens)
        prompt_ids = self.encode(req.prompt)
        if not prompt_ids:
            yield TokenEvent(token="", finished=True, total_latency_ms=_elapsed_ms(started, time.perf_counter()))
            return
        self._next_seq += 1
        # Callers may reuse request ids; the scheduler needs a unique key.
        key = f"{req.request_id}#{self._next_seq}"
        seq = SchedSeq(
            request_id=key,
            prompt_ids=list(prompt_ids),
            max_new_tokens=cap,
            temperature=req.temperature,
            eos_id=self._tokenizer.eos_token_id,
        )
        loop = self._batch_loop()
        queue = loop.submit(seq)
        ttft_ms = 0.0
        produced = 0
        try:
            while True:
                item = await queue.get()
                if isinstance(item, BaseException):
                    raise item
                produced += 1
                now = time.perf_counter()
                if produced == 1:
                    ttft_ms = _elapsed_ms(started, now)
                yield TokenEvent(
                    token=self._tokenizer.decode([item.token_id], skip_special_tokens=True),
                    finished=item.finished,
                    prompt_tokens=len(prompt_ids),
                    completion_tokens=produced,
                    time_to_first_token_ms=ttft_ms if item.finished else 0.0,
                    total_latency_ms=_elapsed_ms(started, now) if item.finished else 0.0,
                )
                if item.finished:
                    return
        finally:
            loop.abort(key)

    def _batch_loop(self) -> BatchLoop:
        if self._loop is None:
            scheduler = Scheduler(
                self._pool,
                page_size=self._kv.page_size,
                max_seqs=_env_int("WORKER_MAX_SEQS", DEFAULT_MAX_SEQS),
                token_budget=_env_int("WORKER_TOKEN_BUDGET", DEFAULT_TOKEN_BUDGET),
            )
            self._loop = BatchLoop(scheduler, ModelRunner(self._model, self._kv, self._device))
        return self._loop

    async def _generate_single(self, req: GenerateRequest) -> AsyncIterator[TokenEvent]:
        """Models without a paged cache (test stubs) keep their own past_key_values."""
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
                return_dict=False,
            )
        return self._tokenizer.encode(prompt, add_special_tokens=True)

    def _graphed_decode_active(self) -> bool:
        if self._decode_graph_failed or self._device != "cuda" or self._kv is None:
            return False
        # Token ids are int64. The kernel decision follows the KV dtype.
        return flash_enabled(self._kv.layer_kv(0)[0].dtype)

    def _graphed_decode(self, token_id: int, block_ids: list[int], seq_len: int):
        """Replay a captured flash decode step. None means the caller should run eager."""
        pages = pages_needed(seq_len + 1, self._kv.page_size)
        try:
            if self._decode_graph is None or self._decode_graph_key != pages:
                graph = DecodeGraph(self._model, self._kv, pages)
                logits = graph.capture(token_id, seq_len, block_ids)
                self._decode_graph = graph
                self._decode_graph_key = pages
                return logits
            return self._decode_graph.replay(token_id, seq_len, block_ids)
        except Exception:
            _log.exception("flash decode graph failed; falling back to eager paged attention")
            self._decode_graph = None
            self._decode_graph_key = None
            self._decode_graph_failed = True
            return None

    def _forward_step(
        self,
        input_ids: list[int],
        past,
        block_ids: Sequence[int],
        seq_len: int,
    ):
        import torch

        with torch.inference_mode():
            tensor = torch.tensor([input_ids], dtype=torch.long, device=self._device)
            if self._kv is not None:
                if not hasattr(self._model, "forward_paged"):
                    raise RuntimeError("paged KV requires Qwen2CausalLM")
                if self._graphed_decode_active() and len(input_ids) == 1:
                    logits = self._graphed_decode(input_ids[0], list(block_ids), seq_len)
                    if logits is not None:
                        return logits[0, -1, :].float().cpu(), None, seq_len + 1
                logits = self._model.forward_paged(tensor, self._kv, list(block_ids), seq_len)
                new_seq = seq_len + len(input_ids)
                return logits[0, -1, :].float().cpu(), None, new_seq

            out = self._model(
                input_ids=tensor,
                past_key_values=past,
                use_cache=True,
            )
            logits = out.logits[0, -1, :].float().cpu()
            return logits, out.past_key_values, seq_len + len(input_ids)


def _maybe_paged_cache(model, pool: BlockAllocator | None, device: str) -> PagedKvCache | None:
    if pool is None:
        return None
    num_blocks = getattr(pool, "num_blocks", None)
    if num_blocks is None:
        _log.warning("skipping paged KV: pool.num_blocks missing")
        return None
    dims = getattr(model, "dims", None)
    if dims is not None:
        shape = KvShape(
            num_layers=dims.num_hidden_layers,
            num_kv_heads=dims.num_key_value_heads,
            head_dim=dims.head_dim,
        )
    else:
        config = getattr(model, "config", None)
        if config is None:
            _log.warning("skipping paged KV: model has no dims or config")
            return None
        shape = KvShape.from_hf_config(config)
    import torch

    dtype = next(model.parameters()).dtype
    return PagedKvCache(
        shape,
        num_blocks=int(num_blocks),
        device=device,
        dtype=dtype,
        page_size=serving_page_size(dtype),
    )


def _pick_token(logits, temperature: float) -> int:
    import torch

    if temperature <= 0:
        return int(torch.argmax(logits).item())
    probs = torch.softmax(logits / temperature, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def _elapsed_ms(started: float, now: float) -> float:
    return (now - started) * 1000.0


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{key} must be an integer") from exc
    if value < 1:
        raise ValueError(f"{key} must be >= 1")
    return value
