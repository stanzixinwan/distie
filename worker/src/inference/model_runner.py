"""GPU side of continuous batching: one StepPlan in, one token per sequence out.

Runs only on the engine's single GPU thread. The scheduler decides who runs;
this module turns that plan into one forward and samples. A step where every
row decodes one token replays a CUDA graph (BatchDecodeGraphs); a step with
any prefill runs the flattened varlen forward eagerly.
"""

from __future__ import annotations

import logging

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

    @property
    def graphs_active(self) -> bool:
        return self._decode is not None

    def step(self, plan: StepPlan) -> list[int]:
        if not plan.items:
            raise ValueError("empty step plan")
        with torch.inference_mode():
            logits = self._decode_step(plan)
            if logits is None:
                logits = self._forward(plan)
            return _sample(logits, [item.seq.temperature for item in plan.items])

    def _forward(self, plan: StepPlan) -> torch.Tensor:
        seqs = [BatchSeq(tuple(it.seq.block_ids), it.past_len, len(it.token_ids)) for it in plan.items]
        meta = build_batch(seqs, self._cache, self._device)
        flat = [tok for it in plan.items for tok in it.token_ids]
        ids = torch.tensor(flat, dtype=torch.long, device=self._device)
        return self._model.forward_batch(ids, self._cache, meta)

    def _decode_step(self, plan: StepPlan) -> torch.Tensor | None:
        """Pure-decode steps go through the bucketed graphs. None means run eager."""
        if self._decode is None or len(plan.items) > self._decode.max_batch:
            return None
        if any(len(item.token_ids) != 1 for item in plan.items):
            return None
        rows = [DecodeRow(item.token_ids[0], item.past_len, item.seq.block_ids) for item in plan.items]
        try:
            return self._decode.run(rows)
        except Exception:
            # The eager forward rewrites the same K/V slots, so retrying this step is safe.
            _log.exception("batch decode graph failed; all steps stay eager from now on")
            self._decode = None
            return None


def _sample(logits: torch.Tensor, temperatures: list[float]) -> list[int]:
    """logits [B, vocab]. Greedy rows take argmax; others sample at their temperature."""
    logits = logits.float()
    picks = logits.argmax(dim=-1)
    for row, temp in enumerate(temperatures):
        if temp > 0:
            probs = torch.softmax(logits[row] / temp, dim=-1)
            picks[row] = torch.multinomial(probs, num_samples=1)[0]
    return picks.tolist()
