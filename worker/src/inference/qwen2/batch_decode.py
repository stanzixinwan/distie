"""Pure-decode steps replayed from CUDA graphs, one graph per batch-size bucket.

Eager batched decode is CPU-bound: each step Python launches every layer's
kernels one by one, and the GPU sits idle between them. A CUDA graph records
the whole step once and replays it with a single launch. A graph has fixed
shapes and fixed buffer addresses, so three things are pinned down:

  batch size   rounded up to a bucket (1, 2, 4, 8, 16, 24, 32, ...);
               the extra rows are padding
  block table  always W columns wide; the kernel stops at each row's
               seqlen, so columns past the last page are never read
  padding rows decode token 0 at position 0 into a spare KV page that the
               block pool never hands out, so they cannot touch live K/V

Each step only copies the new token ids, positions, lengths and block table
into the bucket's device buffers (from pinned host staging), then replays.
"""

from __future__ import annotations

import bisect
import logging
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from inference.kv_cache import PagedKvCache, pages_needed

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DecodeRow:
    """One sequence decoding one token: past_len tokens are already in the slab."""

    token_id: int
    past_len: int
    block_ids: Sequence[int]


def decode_buckets(max_batch: int) -> tuple[int, ...]:
    """Batch sizes that get their own graph: powers of two up to 8, then steps of 8."""
    if max_batch < 1:
        raise ValueError("max_batch must be >= 1")
    sizes = {b for b in (1, 2, 4, 8) if b < max_batch}
    sizes.update(range(16, max_batch, 8))
    sizes.add(max_batch)
    return tuple(sorted(sizes))


class _Bucket:
    """Device buffers a graph reads from and the logits it writes to."""

    def __init__(self, size: int, width: int, device) -> None:
        self.size = size
        self.token = torch.zeros(size, 1, dtype=torch.long, device=device)
        self.position = torch.zeros(size, 1, dtype=torch.long, device=device)
        self.seqlen = torch.zeros(size, dtype=torch.int32, device=device)
        self.table = torch.zeros(size, width, dtype=torch.int32, device=device)
        self.graph: torch.cuda.CUDAGraph | None = None
        self.logits: torch.Tensor | None = None


class BatchDecodeGraphs:
    """Runs pure-decode batches through fixed-shape buffers, captured lazily per bucket.

    use_graphs=False runs the same buffers eagerly. Tests use it on CPU to
    check bucketing and padding without a GPU.
    """

    def __init__(
        self,
        model,
        cache: PagedKvCache,
        pad_block: int,
        max_batch: int,
        use_graphs: bool = True,
    ) -> None:
        if not hasattr(model, "forward_decode"):
            raise TypeError("BatchDecodeGraphs needs a model with forward_decode (Qwen2CausalLM)")
        if not 0 <= pad_block < cache.num_blocks:
            raise ValueError(f"pad_block {pad_block} outside the KV slab of {cache.num_blocks} blocks")
        device = cache.layer_kv(0)[0].device
        if use_graphs and device.type != "cuda":
            raise ValueError("CUDA graphs need a CUDA KV cache")
        self._model = model
        self._cache = cache
        self._pad = pad_block
        self._device = device
        self._use_graphs = use_graphs
        self._sizes = decode_buckets(max_batch)
        # Any sequence fits: it can never hold more pages than the slab has.
        self._width = cache.num_blocks
        self._buckets: dict[int, _Bucket] = {}
        self._pool = torch.cuda.graph_pool_handle() if use_graphs else None
        if hasattr(model, "rope_table"):
            # Fused-RoPE decode reads this table; it must exist before any capture.
            model.rope_table(device, cache.dtype)

        # Pinned staging so the per-step host-to-device copies are async.
        pin = device.type == "cuda"
        rows = self._sizes[-1]
        self._host_ids = torch.zeros(2, rows, dtype=torch.long, pin_memory=pin)  # token, position
        self._host_seqlen = torch.zeros(rows, dtype=torch.int32, pin_memory=pin)
        self._host_table = torch.zeros(rows, self._width, dtype=torch.int32, pin_memory=pin)
        # numpy views of the same memory: filling them is far cheaper than torch indexing.
        self._ids_np = self._host_ids.numpy()
        self._seqlen_np = self._host_seqlen.numpy()
        self._table_np = self._host_table.numpy()
        # Set after each step's copies; the next step waits on it before reusing the staging.
        self._copied: torch.cuda.Event | None = None
        _log.info(
            "batch decode buckets=%s width=%s pad_block=%s graphs=%s",
            self._sizes,
            self._width,
            pad_block,
            use_graphs,
        )

    @property
    def max_batch(self) -> int:
        return self._sizes[-1]

    @property
    def buckets(self) -> tuple[int, ...]:
        return self._sizes

    def bucket_for(self, rows: int) -> int:
        if rows < 1 or rows > self.max_batch:
            raise ValueError(f"decode batch of {rows} rows outside 1..{self.max_batch}")
        return self._sizes[bisect.bisect_left(self._sizes, rows)]

    def run(self, rows: Sequence[DecodeRow], timer=None) -> torch.Tensor:
        """Logits [len(rows), vocab] at each row's new token. K/V for that token is written.

        timer, if given, has start()/stop() called around the launch only,
        after host-side staging, so it measures GPU execution of the step.
        """
        size = self.bucket_for(len(rows))
        bucket = self._buckets.get(size)
        if bucket is None:
            bucket = _Bucket(size, self._width, self._device)
            self._buckets[size] = bucket
        self._stage(rows, bucket)
        if timer is not None:
            timer.start()
        try:
            return self._launch(bucket)[: len(rows), -1, :]
        finally:
            if timer is not None:
                timer.stop()

    def _launch(self, bucket: _Bucket) -> torch.Tensor:
        if not self._use_graphs:
            logits = self._forward(bucket)
        elif bucket.graph is None:
            logits = self._capture(bucket)
        else:
            bucket.graph.replay()
            logits = bucket.logits
        return logits

    def _stage(self, rows: Sequence[DecodeRow], bucket: _Bucket) -> None:
        if self._copied is not None:
            self._copied.synchronize()
        page = self._cache.page_size
        n = len(rows)
        for i, row in enumerate(rows):
            total = row.past_len + 1
            self._cache.check_table(row.block_ids, total)
            need = pages_needed(total, page)
            self._table_np[i, :need] = row.block_ids[:need]
        self._ids_np[0, :n] = [row.token_id for row in rows]
        self._ids_np[1, :n] = [row.past_len for row in rows]
        self._seqlen_np[:n] = [row.past_len for row in rows]
        # Padding rows: token 0 at position 0, written into the spare page.
        self._ids_np[:, n : bucket.size] = 0
        self._seqlen_np[n : bucket.size] = 0
        self._table_np[n : bucket.size, 0] = self._pad

        size = bucket.size
        bucket.token.copy_(self._host_ids[0, :size].view(size, 1), non_blocking=True)
        bucket.position.copy_(self._host_ids[1, :size].view(size, 1), non_blocking=True)
        bucket.seqlen.copy_(self._host_seqlen[:size], non_blocking=True)
        bucket.table.copy_(self._host_table[:size], non_blocking=True)
        if self._device.type == "cuda":
            self._copied = torch.cuda.Event()
            self._copied.record()

    def _forward(self, bucket: _Bucket) -> torch.Tensor:
        return self._model.forward_decode(bucket.token, self._cache, bucket.table, bucket.position, bucket.seqlen)

    def _capture(self, bucket: _Bucket) -> torch.Tensor:
        """Warm up on a side stream, record, then replay for this step.

        Kernels inside torch.cuda.graph() are recorded, not run. The warm-up
        and the replay both write this step's K/V, with identical values.
        """
        current = torch.cuda.current_stream()
        side = torch.cuda.Stream()
        side.wait_stream(current)
        with torch.cuda.stream(side):
            self._forward(bucket)
        current.wait_stream(side)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._pool):
            bucket.logits = self._forward(bucket)
        bucket.graph = graph
        graph.replay()
        _log.info("captured batch decode graph rows=%s", bucket.size)
        return bucket.logits
