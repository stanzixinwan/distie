"""CUDA graph for one flash decode step.

flash_attn_with_kvcache itself is a short kernel. The gap versus HuggingFace
generate is the Python between the 28 layers: each step launches hundreds
of small kernels from Python and the GPU waits on it. Capturing the step
once and replaying it keeps the token, position, length, and block table in
device tensors that are updated in place, so one graph serves any sequence
whose block table has the same number of pages.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import torch

from inference.kv_cache import PagedKvCache

_log = logging.getLogger(__name__)


class DecodeGraph:
    """One captured q_len=1 step for block tables of a fixed width."""

    def __init__(self, model, cache: PagedKvCache, num_pages: int) -> None:
        if num_pages < 1:
            raise ValueError("decode graph needs at least one page")
        device = cache.layer_kv(0)[0].device
        if device.type != "cuda":
            raise ValueError("decode graph requires a CUDA KV cache")
        self.model = model
        self.cache = cache
        self.num_pages = num_pages
        self.token = torch.zeros(1, 1, dtype=torch.long, device=device)
        self.position = torch.zeros(1, 1, dtype=torch.long, device=device)
        self.seqlen = torch.zeros(1, dtype=torch.int32, device=device)
        self.table = torch.zeros(1, num_pages, dtype=torch.int32, device=device)
        self.graph = torch.cuda.CUDAGraph()
        self.logits: torch.Tensor | None = None

    def capture(self, token_id: int, seq_len: int, block_ids: Sequence[int]) -> torch.Tensor:
        """Record the graph, then replay it for this token.

        Kernels inside torch.cuda.graph() are recorded, not run, so the
        replay is what writes this token's K/V and fills the logits.
        """
        self._fill(token_id, seq_len, block_ids)
        torch.cuda.synchronize()
        with torch.cuda.graph(self.graph):
            self.logits = self.model.forward_decode(
                self.token, self.cache, self.table, self.position, self.seqlen
            )
        self.graph.replay()
        _log.info("captured flash decode graph pages=%s", self.num_pages)
        return self.logits

    def replay(self, token_id: int, seq_len: int, block_ids: Sequence[int]) -> torch.Tensor:
        self._fill(token_id, seq_len, block_ids)
        self.graph.replay()
        return self.logits

    def _fill(self, token_id: int, seq_len: int, block_ids: Sequence[int]) -> None:
        table = list(block_ids[: self.num_pages])
        if len(table) != self.num_pages:
            raise ValueError(f"block table has {len(table)} pages, graph expects {self.num_pages}")
        self.cache.check_table(table, seq_len + 1)
        self.token.fill_(token_id)
        self.position.fill_(seq_len)
        self.seqlen.fill_(seq_len)
        self.table.copy_(torch.tensor([table], dtype=torch.int32), non_blocking=True)
