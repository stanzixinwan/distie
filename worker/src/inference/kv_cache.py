"""Paged KV-cache: C++ Block IDs are the page table, torch holds the slabs.

Why two stores?
  C++ BlockPool decides which physical pages are free (no Python GC churn).
  A pre-sized torch tensor on CPU/GPU holds the actual K/V. Same integer ID
  indexes both. Writing K/V into the C++ CPU arena would force a DtoH copy
  every token; that is the wrong place for GPU-resident cache.

This is vLLM's software paging without the CUDA attention kernel: we
gather pages into a contiguous past_key_values for standard SDPA, then
scatter new tokens back. Gather is the cost to delete next.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass

_log = logging.getLogger(__name__)

# vLLM default page size. FakeEngine keeps TOKENS_PER_BLOCK=4 (occupancy toy).
DEFAULT_PAGE_SIZE = 16


def pages_needed(token_count: int, page_size: int) -> int:
    if page_size < 1:
        raise ValueError("page_size must be >= 1")
    if token_count <= 0:
        return 0
    return (token_count + page_size - 1) // page_size


@dataclass(frozen=True)
class KvShape:
    """Per-token KV layout derived from a HuggingFace config."""

    num_layers: int
    num_kv_heads: int
    head_dim: int

    @classmethod
    def from_hf_config(cls, config) -> KvShape:
        layers = int(config.num_hidden_layers)
        heads = int(config.num_attention_heads)
        kv_heads = int(getattr(config, "num_key_value_heads", heads))
        if hasattr(config, "head_dim") and config.head_dim:
            head_dim = int(config.head_dim)
        else:
            hidden = int(config.hidden_size)
            if heads < 1 or hidden % heads != 0:
                raise ValueError("cannot derive head_dim from HuggingFace config")
            head_dim = hidden // heads
        if layers < 1 or kv_heads < 1 or head_dim < 1:
            raise ValueError("invalid KV shape from HuggingFace config")
        return cls(num_layers=layers, num_kv_heads=kv_heads, head_dim=head_dim)


class PagedKvCache:
    """Physical KV slabs indexed by BlockPool IDs.

    Layout (matches HuggingFace BHSD: batch, heads, seq, dim after gather):
      k, v : [num_layers, num_blocks, num_kv_heads, page_size, head_dim]

    Logical token `pos` of one request maps to:
      page   = pos // page_size          # index into that request's block table
      offset = pos %  page_size
      slot   = block_ids[page]           # physical BlockId from C++
      k[layer, slot, :, offset, :]
    """

    def __init__(
        self,
        shape: KvShape,
        num_blocks: int,
        device: str,
        dtype,
        page_size: int = DEFAULT_PAGE_SIZE,
    ) -> None:
        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        if page_size < 1:
            raise ValueError("page_size must be >= 1")

        import torch

        self.shape = shape
        self.num_blocks = num_blocks
        self.page_size = page_size
        self.device = device
        self.dtype = dtype
        # Two tensors (K and V), pre-allocated once. allocate() only hands out IDs.
        slot_shape = (
            shape.num_layers,
            num_blocks,
            shape.num_kv_heads,
            page_size,
            shape.head_dim,
        )
        self._k = torch.zeros(slot_shape, device=device, dtype=dtype)
        self._v = torch.zeros(slot_shape, device=device, dtype=dtype)
        bytes_ = 2 * math.prod(slot_shape) * self._k.element_size()
        _log.info(
            "paged kv cache layers=%s kv_heads=%s head_dim=%s blocks=%s "
            "page_size=%s device=%s dtype=%s bytes=%s",
            shape.num_layers,
            shape.num_kv_heads,
            shape.head_dim,
            num_blocks,
            page_size,
            device,
            dtype,
            bytes_,
        )

    # ------------------------------------------------------------------
    # ALGORITHM: scatter — write a contiguous HF cache slice into pages.
    # ------------------------------------------------------------------
    def scatter(
        self,
        block_ids: Sequence[int],
        start_pos: int,
        keys: Sequence,
        values: Sequence,
    ) -> None:
        """Copy keys/values[:, start_pos:] into the page table.

        keys[layer] shape: [1, num_kv_heads, seq_len, head_dim]
        Only the suffix starting at start_pos is written, so decode (1 new
        token) does not rewrite the prompt pages.
        """
        if start_pos < 0:
            raise ValueError("start_pos must be >= 0")
        seq_len = _seq_len(keys)
        if start_pos >= seq_len:
            return
        self._check_table(block_ids, seq_len)
        self._check_layer_shapes(keys, values, seq_len)

        # Walk the suffix page by page so a full page is one copy, not one
        # copy per token. Still Python-side; vectorize further with a
        # block-table tensor + index_copy when profiling says so.
        pos = start_pos
        while pos < seq_len:
            page = pos // self.page_size
            offset = pos % self.page_size
            slot = int(block_ids[page])
            n = min(self.page_size - offset, seq_len - pos)
            for layer in range(self.shape.num_layers):
                # dest: [H, n, D]  src: [1, H, n, D] squeezed at batch.
                self._k[layer, slot, :, offset : offset + n, :].copy_(
                    keys[layer][0, :, pos : pos + n, :]
                )
                self._v[layer, slot, :, offset : offset + n, :].copy_(
                    values[layer][0, :, pos : pos + n, :]
                )
            pos += n

    # ------------------------------------------------------------------
    # ALGORITHM: gather — rebuild contiguous past_key_values from pages.
    # ------------------------------------------------------------------
    def gather(
        self, block_ids: Sequence[int], seq_len: int
    ) -> tuple[list, list] | None:
        """Return (keys, values) with each tensor [1, H, seq_len, D].

        This concatenation is what a PagedAttention kernel avoids: it would
        index k[layer, block_ids[page], :, offset, :] inside the GEMM instead
        of copying pages into a dense [H, S, D] tensor every decode step.
        """
        if seq_len < 0:
            raise ValueError("seq_len must be >= 0")
        if seq_len == 0:
            return None
        import torch

        self._check_table(block_ids, seq_len)
        n_pages = pages_needed(seq_len, self.page_size)
        k_pieces: list = []
        v_pieces: list = []
        remaining = seq_len
        for page in range(n_pages):
            slot = int(block_ids[page])
            n = min(self.page_size, remaining)
            # [L, H, n, D]
            k_pieces.append(self._k[:, slot, :, :n, :])
            v_pieces.append(self._v[:, slot, :, :n, :])
            remaining -= n

        # cat along seq -> [L, H, S, D], then split per layer with batch dim.
        k_all = torch.cat(k_pieces, dim=2)
        v_all = torch.cat(v_pieces, dim=2)
        keys = [k_all[layer].unsqueeze(0) for layer in range(self.shape.num_layers)]
        values = [v_all[layer].unsqueeze(0) for layer in range(self.shape.num_layers)]
        return keys, values

    def _check_table(self, block_ids: Sequence[int], seq_len: int) -> None:
        need = pages_needed(seq_len, self.page_size)
        if len(block_ids) < need:
            raise ValueError(
                f"block table too short: have {len(block_ids)} pages, need {need}"
            )
        for slot in block_ids[:need]:
            idx = int(slot)
            if idx < 0 or idx >= self.num_blocks:
                raise ValueError(f"block id out of range: {idx}")

    def _check_layer_shapes(
        self, keys: Sequence, values: Sequence, seq_len: int
    ) -> None:
        if len(keys) != self.shape.num_layers or len(values) != self.shape.num_layers:
            raise ValueError("KV layer count does not match cache")
        expected = (
            1,
            self.shape.num_kv_heads,
            seq_len,
            self.shape.head_dim,
        )
        if tuple(keys[0].shape) != expected or tuple(values[0].shape) != expected:
            raise ValueError(
                f"expected KV shape {expected} (BHSD, batch=1), "
                f"got k={tuple(keys[0].shape)} v={tuple(values[0].shape)}"
            )


def _seq_len(keys: Sequence) -> int:
    if not keys:
        raise ValueError("empty KV")
    if keys[0].dim() != 4:
        raise ValueError("KV tensors must be rank-4 [batch, heads, seq, dim]")
    if int(keys[0].shape[0]) != 1:
        raise ValueError("PagedKvCache supports batch=1; continuous batching is next")
    return int(keys[0].shape[2])
