"""Paged attention for one sequence.

The torch backend writes new K/V straight into flash-layout pages, then
indexes that sequence's pages into a dense tensor for SDPA. DISTIE_ATTN=flash
instead calls flash_attn_with_kvcache, which reads the block table inside
the kernel. fp32 stays on the torch path: the kernel is fp16/bf16 only.
Batch stays 1 until the scheduler loop owns the GPU.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence

import torch
from torch.nn import functional as F

from inference.kv_cache import DEFAULT_PAGE_SIZE, PagedKvCache, pages_needed

# flash_attn_with_kvcache rejects any paged block that is not a multiple of 256.
FLASH_PAGE_SIZE = 256

_log = logging.getLogger(__name__)
_flash_import_failed = False
_flash_dtype_failed = False


def attend(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cache: PagedKvCache,
    layer: int,
    block_ids: Sequence[int],
    seq_len: int,
    n_rep: int,
) -> torch.Tensor:
    """query/key/value are RoPE'd, shapes [1, heads, q_len, dim] and [1, kv_heads, q_len, dim].

    seq_len is how many tokens are already in the cache. Returns [1, heads, q_len, dim].
    """
    if query.ndim != 4 or query.shape[0] != 1:
        raise ValueError("paged attention is batch=1 [1, heads, q_len, dim]")
    if key.shape[0] != 1 or value.shape[:2] != key.shape[:2] or key.shape[2] != query.shape[2]:
        raise ValueError("key/value must be [1, kv_heads, q_len, dim] for these queries")
    if seq_len < 0:
        raise ValueError("seq_len must be >= 0")
    q_len = int(query.shape[2])
    cache.check_table(block_ids, seq_len + q_len)
    if _backend(query.dtype) == "flash":
        return _flash_attend(query, key, value, cache, layer, block_ids, seq_len)
    cache_k, cache_v = cache.layer_kv(layer)
    _write(cache_k, block_ids, seq_len, key[0], cache.page_size)
    _write(cache_v, block_ids, seq_len, value[0], cache.page_size)
    k, v = read_kv(cache_k, cache_v, block_ids, seq_len + q_len, cache.page_size)
    k = _repeat_kv(k, n_rep)
    v = _repeat_kv(v, n_rep)
    return _sdpa(query, k.unsqueeze(0), v.unsqueeze(0))


def read_kv(
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    block_ids: Sequence[int],
    seq_len: int,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Index pages into [kv_heads, seq_len, dim]."""
    return _read(cache_k, block_ids, seq_len, page_size), _read(cache_v, block_ids, seq_len, page_size)


def _write(
    cache: torch.Tensor,
    block_ids: Sequence[int],
    start: int,
    tokens: torch.Tensor,
    page_size: int,
) -> None:
    # tokens: [kv_heads, q_len, dim]. cache slot: [page, kv_heads, dim].
    q_len = int(tokens.shape[1])
    pos = 0
    while pos < q_len:
        abs_pos = start + pos
        page = abs_pos // page_size
        offset = abs_pos % page_size
        n = min(page_size - offset, q_len - pos)
        slot = int(block_ids[page])
        cache[slot, offset : offset + n].copy_(tokens[:, pos : pos + n, :].transpose(0, 1))
        pos += n


def _read(
    cache: torch.Tensor,
    block_ids: Sequence[int],
    seq_len: int,
    page_size: int,
) -> torch.Tensor:
    pieces = []
    remaining = seq_len
    for page in range(pages_needed(seq_len, page_size)):
        n = min(page_size, remaining)
        slot = int(block_ids[page])
        # [n, H, D] -> [H, n, D]
        pieces.append(cache[slot, :n].permute(1, 0, 2))
        remaining -= n
    return torch.cat(pieces, dim=1)


def _repeat_kv(hidden: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden
    kv_heads, seq, dim = hidden.shape
    hidden = hidden[:, None, :, :].expand(kv_heads, n_rep, seq, dim)
    return hidden.reshape(kv_heads * n_rep, seq, dim)


def _sdpa(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    q_len = int(query.shape[-2])
    kv_len = int(key.shape[-2])
    if q_len == kv_len:
        return F.scaled_dot_product_attention(query, key, value, is_causal=True)
    if q_len == 1:
        return F.scaled_dot_product_attention(query, key, value, is_causal=False)
    past = kv_len - q_len
    q_idx = torch.arange(q_len, device=query.device)[:, None]
    k_idx = torch.arange(kv_len, device=query.device)[None, :]
    blocked = k_idx > (q_idx + past)
    mask = torch.zeros(q_len, kv_len, device=query.device, dtype=query.dtype)
    mask = mask.masked_fill(blocked, torch.finfo(query.dtype).min)
    return F.scaled_dot_product_attention(query, key, value, attn_mask=mask)


def _backend(dtype: torch.dtype) -> str:
    """'flash' only when requested, installed, and the dtype is half precision."""
    global _flash_import_failed, _flash_dtype_failed
    name = os.environ.get("DISTIE_ATTN", "torch").strip().lower()
    if name in {"", "torch"}:
        return "torch"
    if name != "flash":
        raise ValueError("DISTIE_ATTN must be 'torch' or 'flash'")
    if dtype not in {torch.float16, torch.bfloat16}:
        if not _flash_dtype_failed:
            _log.error("DISTIE_ATTN=flash does not support %s; using torch indexing", dtype)
            _flash_dtype_failed = True
        return "torch"
    try:
        from flash_attn import flash_attn_with_kvcache  # noqa: F401
    except ImportError:
        if not _flash_import_failed:
            _log.error("DISTIE_ATTN=flash but flash_attn is not installed; using torch indexing")
            _flash_import_failed = True
        return "torch"
    return "flash"


def flash_enabled(dtype: torch.dtype) -> bool:
    """True when this dtype will actually run flash_attn_with_kvcache."""
    return _backend(dtype) == "flash"


def serving_page_size(dtype: torch.dtype) -> int:
    """Page size for a newly allocated slab. The torch path keeps the small default."""
    if _backend(dtype) == "flash":
        return FLASH_PAGE_SIZE
    return DEFAULT_PAGE_SIZE


def _flash_attend(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cache: PagedKvCache,
    layer: int,
    block_ids: Sequence[int],
    seq_len: int,
) -> torch.Tensor:
    """query is [1, heads, q_len, dim]. The kernel appends key/value into the slab."""
    if cache.page_size % FLASH_PAGE_SIZE != 0:
        raise ValueError(
            "flash_attn_with_kvcache requires page_size to be a multiple of "
            f"{FLASH_PAGE_SIZE}, got {cache.page_size}"
        )
    from flash_attn import flash_attn_with_kvcache

    cache_k, cache_v = cache.layer_kv(layer)
    q_len = int(query.shape[2])
    needed = pages_needed(seq_len + q_len, cache.page_size)
    table = torch.tensor([list(block_ids[:needed])], dtype=torch.int32, device=query.device)
    seqlens = torch.tensor([seq_len], dtype=torch.int32, device=query.device)
    out = flash_attn_with_kvcache(
        query.transpose(1, 2).contiguous(),
        cache_k,
        cache_v,
        k=key.transpose(1, 2).contiguous(),
        v=value.transpose(1, 2).contiguous(),
        cache_seqlens=seqlens,
        block_table=table,
        causal=True,
    )
    return out.transpose(1, 2).contiguous()


def flash_decode(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cache: PagedKvCache,
    layer: int,
    block_table: torch.Tensor,
    seqlen: torch.Tensor,
) -> torch.Tensor:
    """q_len=1. block_table and seqlen are caller-owned so a CUDA graph can replay them.

    seqlen is tokens already in the cache, shape [1] int32. The kernel appends key/value.
    """
    if cache.page_size % FLASH_PAGE_SIZE != 0:
        raise ValueError(
            "flash_attn_with_kvcache requires page_size to be a multiple of "
            f"{FLASH_PAGE_SIZE}, got {cache.page_size}"
        )
    from flash_attn import flash_attn_with_kvcache

    cache_k, cache_v = cache.layer_kv(layer)
    out = flash_attn_with_kvcache(
        query.transpose(1, 2).contiguous(),
        cache_k,
        cache_v,
        k=key.transpose(1, 2).contiguous(),
        v=value.transpose(1, 2).contiguous(),
        cache_seqlens=seqlen,
        block_table=block_table,
        causal=True,
    )
    return out.transpose(1, 2).contiguous()
