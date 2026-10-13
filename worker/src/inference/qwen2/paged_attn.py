"""Paged attention over flash-layout KV pages.

The torch backend writes new K/V straight into flash-layout pages, then
indexes each sequence's pages into a dense tensor for SDPA. DISTIE_ATTN=flash
instead calls a flash-attn kernel that reads the block table itself.
fp32 stays on the torch path: the kernel is fp16/bf16 only.

attend() serves one sequence. attend_batch() serves a flattened varlen
batch: sequences packed back to back, cu_seqlens marking the boundaries,
so prefill and decode rows share one forward without padding.
decode_attend() serves a pure-decode batch from fixed-shape device tensors,
which is what a CUDA graph needs.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass

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


def decode_attend(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cache: PagedKvCache,
    layer: int,
    block_table: torch.Tensor,
    seqlen: torch.Tensor,
    n_rep: int,
) -> torch.Tensor:
    """One new token per row. query [B, heads, 1, dim], key/value [B, kv_heads, 1, dim].

    block_table [B, W] int32 and seqlen [B] int32 (tokens already stored) are
    device tensors, never Python ints, so a CUDA graph can replay this with
    them updated in place. Columns past a row's last page are ignored.
    """
    if query.ndim != 4 or query.shape[2] != 1:
        raise ValueError("decode_attend expects query [B, heads, 1, dim]")
    if _backend(query.dtype) == "flash":
        return flash_decode(query, key, value, cache, layer, block_table, seqlen)
    return _torch_decode(query, key, value, cache, layer, block_table, seqlen, n_rep)


def _torch_decode(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cache: PagedKvCache,
    layer: int,
    block_table: torch.Tensor,
    seqlen: torch.Tensor,
    n_rep: int,
) -> torch.Tensor:
    """Reference for decode_attend: tensor ops only, reads every column of the table.

    Exact enough for fp32 parity tests on CPU. Serving uses the flash kernel,
    which stops at each row's seqlen instead of masking a full-width read.
    """
    cache_k, cache_v = cache.layer_kv(layer)
    page = cache.page_size
    batch, width = int(block_table.shape[0]), int(block_table.shape[1])
    past = seqlen.long()
    table = block_table.long()
    current = table.gather(1, (past // page).unsqueeze(1)).squeeze(1)
    write_slots(cache_k, cache_v, current * page + past % page, key[:, :, 0], value[:, :, 0])
    # [B, W, page, KH, D] -> [B, KH, W * page, D]
    k = cache_k[table].reshape(batch, width * page, *cache_k.shape[-2:]).permute(0, 2, 1, 3)
    v = cache_v[table].reshape(batch, width * page, *cache_v.shape[-2:]).permute(0, 2, 1, 3)
    if n_rep > 1:
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)
    visible = torch.arange(width * page, device=query.device)[None, :] <= past[:, None]
    mask = torch.zeros(batch, 1, 1, width * page, device=query.device, dtype=query.dtype)
    mask = mask.masked_fill(~visible[:, None, None, :], torch.finfo(query.dtype).min)
    return F.scaled_dot_product_attention(query, k, v, attn_mask=mask)


def flash_decode(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cache: PagedKvCache,
    layer: int,
    block_table: torch.Tensor,
    seqlen: torch.Tensor,
) -> torch.Tensor:
    """q_len=1 for B rows. block_table and seqlen are caller-owned so a CUDA graph can replay them.

    seqlen is tokens already in the cache, shape [B] int32. The kernel appends key/value.
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


@dataclass(frozen=True)
class BatchSeq:
    """One sequence in a varlen step: its pages, tokens already stored, new tokens."""

    block_ids: tuple[int, ...]
    past_len: int
    q_len: int


@dataclass(frozen=True)
class BatchMeta:
    """Device tensors for one flattened step. T = total new tokens, B = sequences."""

    seqs: tuple[BatchSeq, ...]
    positions: torch.Tensor  # [T] long, absolute positions for RoPE
    slot_mapping: torch.Tensor  # [T] long, block * page_size + offset
    cu_seqlens_q: torch.Tensor  # [B + 1] int32
    cu_seqlens_k: torch.Tensor  # [B + 1] int32
    max_seqlen_q: int
    max_seqlen_k: int
    block_table: torch.Tensor  # [B, max_pages] int32, padded with 0
    last_index: torch.Tensor  # [B] long, row of each sequence's last new token


def build_batch(seqs: Sequence[BatchSeq], cache: PagedKvCache, device) -> BatchMeta:
    if not seqs:
        raise ValueError("batch needs at least one sequence")
    page = cache.page_size
    positions: list[int] = []
    slots: list[int] = []
    cu_q = [0]
    cu_k = [0]
    tables: list[list[int]] = []
    for seq in seqs:
        if seq.q_len < 1 or seq.past_len < 0:
            raise ValueError("each sequence needs q_len >= 1 and past_len >= 0")
        total = seq.past_len + seq.q_len
        cache.check_table(seq.block_ids, total)
        for pos in range(seq.past_len, total):
            positions.append(pos)
            slots.append(int(seq.block_ids[pos // page]) * page + pos % page)
        cu_q.append(cu_q[-1] + seq.q_len)
        cu_k.append(cu_k[-1] + total)
        tables.append(list(seq.block_ids[: pages_needed(total, page)]))
    width = max(len(t) for t in tables)
    padded = [t + [0] * (width - len(t)) for t in tables]
    return BatchMeta(
        seqs=tuple(seqs),
        positions=torch.tensor(positions, dtype=torch.long, device=device),
        slot_mapping=torch.tensor(slots, dtype=torch.long, device=device),
        cu_seqlens_q=torch.tensor(cu_q, dtype=torch.int32, device=device),
        cu_seqlens_k=torch.tensor(cu_k, dtype=torch.int32, device=device),
        max_seqlen_q=max(s.q_len for s in seqs),
        max_seqlen_k=max(s.past_len + s.q_len for s in seqs),
        block_table=torch.tensor(padded, dtype=torch.int32, device=device),
        last_index=torch.tensor([c - 1 for c in cu_q[1:]], dtype=torch.long, device=device),
    )


def write_slots(
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    slot_mapping: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> None:
    """key/value are [T, kv_heads, dim]; slot_mapping indexes the flattened pages."""
    flat_k = cache_k.view(-1, cache_k.shape[-2], cache_k.shape[-1])
    flat_v = cache_v.view(-1, cache_v.shape[-2], cache_v.shape[-1])
    flat_k.index_copy_(0, slot_mapping, key)
    flat_v.index_copy_(0, slot_mapping, value)


def attend_batch(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cache: PagedKvCache,
    layer: int,
    meta: BatchMeta,
    n_rep: int,
) -> torch.Tensor:
    """query [T, heads, dim], key/value [T, kv_heads, dim], RoPE'd. Returns [T, heads, dim]."""
    if query.ndim != 3 or key.shape[0] != query.shape[0] or value.shape != key.shape:
        raise ValueError("attend_batch expects query [T, H, D] and key/value [T, KH, D]")
    if int(meta.slot_mapping.shape[0]) != int(query.shape[0]):
        raise ValueError("slot_mapping length does not match the batch")
    cache_k, cache_v = cache.layer_kv(layer)
    write_slots(cache_k, cache_v, meta.slot_mapping, key, value)
    if _backend(query.dtype) == "flash":
        return _flash_varlen(query, cache_k, cache_v, cache.page_size, meta)
    pieces = []
    start = 0
    for seq in meta.seqs:
        end = start + seq.q_len
        q = query[start:end].transpose(0, 1).unsqueeze(0)
        k, v = read_kv(cache_k, cache_v, seq.block_ids, seq.past_len + seq.q_len, cache.page_size)
        k = _repeat_kv(k, n_rep).unsqueeze(0)
        v = _repeat_kv(v, n_rep).unsqueeze(0)
        pieces.append(_sdpa(q, k, v)[0].transpose(0, 1))
        start = end
    return torch.cat(pieces, dim=0)


def _flash_varlen(
    query: torch.Tensor,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    page_size: int,
    meta: BatchMeta,
) -> torch.Tensor:
    if page_size % FLASH_PAGE_SIZE != 0:
        raise ValueError(
            "flash paged attention requires page_size to be a multiple of "
            f"{FLASH_PAGE_SIZE}, got {page_size}"
        )
    from flash_attn import flash_attn_varlen_func

    return flash_attn_varlen_func(
        query.contiguous(),
        cache_k,
        cache_v,
        meta.cu_seqlens_q,
        meta.cu_seqlens_k,
        meta.max_seqlen_q,
        meta.max_seqlen_k,
        causal=True,
        block_table=meta.block_table,
    )
