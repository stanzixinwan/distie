"""CUDA graph for one flash decode step.

flash_attn_with_kvcache itself is a short kernel. The gap versus HuggingFace
generate is the Python between the 28 layers: each step rebuilds the block
table and stalls the GPU. Capturing the step once and replaying it keeps
the block table and the sequence-length tensor in place, so only the new
token id and the length change between tokens.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import torch

from inference.kv_cache import PagedKvCache

_log = logging.getLogger(__name__)


class DecodeGraph:
    """One captured q_len=1 step for a fixed block table."""

    def __init__(self, model, cache: PagedKvCache, block_ids: Sequence[int]) -> None:
        if not block_ids:
            raise ValueError("decode graph needs at least one block")
        device = cache.layer_kv(0)[0].device
        if device.type != "cuda":
            raise ValueError("decode graph requires a CUDA KV cache")
        self.model = model
        self.cache = cache
        self.block_ids = list(block_ids)
        self.token = torch.zeros(1, 1, dtype=torch.long, device=device)
        self.position = torch.zeros(1, 1, dtype=torch.long, device=device)
        self.seqlen = torch.zeros(1, dtype=torch.int32, device=device)
        self.table = torch.tensor([self.block_ids], dtype=torch.int32, device=device)
        self.graph = torch.cuda.CUDAGraph()
        self.logits: torch.Tensor | None = None

    def capture(self, token_id: int, seq_len: int) -> torch.Tensor:
        """Run this token while recording the graph. The kernels execute once."""
        self.cache.check_table(self.block_ids, seq_len + 1)
        self._fill(token_id, seq_len)
        torch.cuda.synchronize()
        with torch.cuda.graph(self.graph):
            self.logits = self.model.forward_decode(
                self.token, self.cache, self.table, self.position, self.seqlen
            )
        _log.info("captured flash decode graph pages=%s seq_len=%s", len(self.block_ids), seq_len)
        return self.logits

    def replay(self, token_id: int, seq_len: int) -> torch.Tensor:
        self.cache.check_table(self.block_ids, seq_len + 1)
        self._fill(token_id, seq_len)
        self.graph.replay()
        return self.logits

    def _fill(self, token_id: int, seq_len: int) -> None:
        self.token.fill_(token_id)
        self.position.fill_(seq_len)
        self.seqlen.fill_(seq_len)
