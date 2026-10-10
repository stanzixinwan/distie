"""GPU side of continuous batching: one StepPlan in, one token per sequence out.

Runs only on the engine's single GPU thread. The scheduler decides who runs;
this module turns that plan into one flattened forward and samples.
"""

from __future__ import annotations

import logging

import torch

from inference.kv_cache import PagedKvCache, pages_needed
from inference.qwen2.decode_graph import DecodeGraph
from inference.qwen2.paged_attn import BatchSeq, build_batch, flash_enabled
from inference.scheduler import StepPlan

_log = logging.getLogger(__name__)


class ModelRunner:
    def __init__(self, model, cache: PagedKvCache, device: str) -> None:
        if not hasattr(model, "forward_batch"):
            raise TypeError("ModelRunner needs a model with forward_batch (Qwen2CausalLM)")
        self._model = model
        self._cache = cache
        self._device = device
        self._graphs: dict[int, DecodeGraph] = {}
        self._graphs_failed = False

    def step(self, plan: StepPlan) -> list[int]:
        if not plan.items:
            raise ValueError("empty step plan")
        with torch.inference_mode():
            logits = self._single_decode_graphed(plan)
            if logits is None:
                logits = self._forward(plan)
            return _sample(logits, [item.seq.temperature for item in plan.items])

    def _forward(self, plan: StepPlan) -> torch.Tensor:
        seqs = [BatchSeq(tuple(it.seq.block_ids), it.past_len, len(it.token_ids)) for it in plan.items]
        meta = build_batch(seqs, self._cache, self._device)
        flat = [tok for it in plan.items for tok in it.token_ids]
        ids = torch.tensor(flat, dtype=torch.long, device=self._device)
        return self._model.forward_batch(ids, self._cache, meta)

    def _single_decode_graphed(self, plan: StepPlan) -> torch.Tensor | None:
        """A lone decode row replays a CUDA graph; batched rows run eager."""
        if len(plan.items) != 1 or self._graphs_failed or self._device != "cuda":
            return None
        item = plan.items[0]
        if len(item.token_ids) != 1 or not flash_enabled(self._cache.dtype):
            return None
        pages = pages_needed(item.past_len + 1, self._cache.page_size)
        block_ids = item.seq.block_ids
        try:
            graph = self._graphs.get(pages)
            if graph is None:
                graph = DecodeGraph(self._model, self._cache, pages)
                logits = graph.capture(item.token_ids[0], item.past_len, block_ids)
                self._graphs[pages] = graph
            else:
                logits = graph.replay(item.token_ids[0], item.past_len, block_ids)
            return logits[:, -1, :]
        except Exception:
            _log.exception("flash decode graph failed; batched steps stay eager")
            self._graphs.clear()
            self._graphs_failed = True
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
