"""GPU side of continuous batching: one StepPlan in, one token per sequence out.

Runs only on the engine's single GPU thread. The scheduler decides who runs;
this module turns that plan into one forward and samples. A step where every
row decodes one token replays a CUDA graph (BatchDecodeGraphs); a step with
any prefill runs the flattened varlen forward eagerly.

With measure_kernel set (BatchLoop does this when step stats are on), the
forward is bracketed by CUDA events and last_kernel_ms holds its GPU time.
For a graph step that is pure kernel time; for an eager step it also
includes the GPU idling between kernels while Python launches the next one.
"""

from __future__ import annotations

import logging
import time

import torch

from inference.kv_cache import PagedKvCache
from inference.qwen2.batch_decode import BatchDecodeGraphs, DecodeRow
from inference.qwen2.paged_attn import BatchSeq, build_batch
from inference.scheduler import StepPlan

_log = logging.getLogger(__name__)


class ModelRunner:
    def __init__(
        self,
        model,
        cache: PagedKvCache,
        device: str,
        decode: BatchDecodeGraphs | None = None,
    ) -> None:
        """decode=None keeps every step on the eager varlen path."""
        if not hasattr(model, "forward_batch"):
            raise TypeError("ModelRunner needs a model with forward_batch (Qwen2CausalLM)")
        self._model = model
        self._cache = cache
        self._device = device
        self._decode = decode
        self.measure_kernel = False
        self.last_kernel_ms: float | None = None

    @property
    def graphs_active(self) -> bool:
        return self._decode is not None

    def step(self, plan: StepPlan) -> list[int]:
        if not plan.items:
            raise ValueError("empty step plan")
        timer = _KernelTimer(self._device) if self.measure_kernel else None
        with torch.inference_mode():
            logits = self._decode_step(plan, timer)
            if logits is None:
                logits = self._forward(plan, timer)
            picks = _sample(logits, [item.seq.temperature for item in plan.items])
        # _sample ends with a host sync, so both events have completed.
        self.last_kernel_ms = timer.elapsed_ms() if timer is not None else None
        return picks

    def _forward(self, plan: StepPlan, timer: _KernelTimer | None = None) -> torch.Tensor:
        seqs = [BatchSeq(tuple(it.seq.block_ids), it.past_len, len(it.token_ids)) for it in plan.items]
        meta = build_batch(seqs, self._cache, self._device)
        flat = [tok for it in plan.items for tok in it.token_ids]
        ids = torch.tensor(flat, dtype=torch.long, device=self._device)
        if timer is not None:
            timer.start()
        logits = self._model.forward_batch(ids, self._cache, meta)
        if timer is not None:
            timer.stop()
        return logits

    def _decode_step(self, plan: StepPlan, timer: _KernelTimer | None = None) -> torch.Tensor | None:
        """Pure-decode steps go through the bucketed graphs. None means run eager."""
        if self._decode is None or len(plan.items) > self._decode.max_batch:
            return None
        if any(len(item.token_ids) != 1 for item in plan.items):
            return None
        rows = [DecodeRow(item.token_ids[0], item.past_len, item.seq.block_ids) for item in plan.items]
        try:
            return self._decode.run(rows, timer)
        except Exception:
            # The eager forward rewrites the same K/V slots, so retrying this step is safe.
            _log.exception("batch decode graph failed; all steps stay eager from now on")
            self._decode = None
            return None


class _KernelTimer:
    """CUDA events around one forward; on CPU, plain wall time (CPU kernels are synchronous)."""

    def __init__(self, device: str) -> None:
        self._cuda = device == "cuda"
        self._events = None
        self._t0 = self._t1 = 0.0

    def start(self) -> None:
        if self._cuda:
            self._events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            self._events[0].record()
        else:
            self._t0 = time.perf_counter()

    def stop(self) -> None:
        if self._cuda:
            self._events[1].record()
        else:
            self._t1 = time.perf_counter()

    def elapsed_ms(self) -> float | None:
        if self._cuda:
            if self._events is None:
                return None
            self._events[1].synchronize()
            return self._events[0].elapsed_time(self._events[1])
        return (self._t1 - self._t0) * 1e3 if self._t1 else None


def _sample(logits: torch.Tensor, temperatures: list[float]) -> list[int]:
    """logits [B, vocab]. Greedy rows take argmax; others sample at their temperature."""
    logits = logits.float()
    picks = logits.argmax(dim=-1)
    for row, temp in enumerate(temperatures):
        if temp > 0:
            probs = torch.softmax(logits[row] / temp, dim=-1)
            picks[row] = torch.multinomial(probs, num_samples=1)[0]
    return picks.tolist()
