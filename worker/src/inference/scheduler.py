"""Continuous-batching scheduler: which sequences run in the next forward step.

Pure bookkeeping, no torch, so it can be tested without a GPU. One owner
(the engine loop) calls it; nothing here is thread-safe.

Every step:
  1. reap aborted sequences and free their blocks;
  2. grow running sequences page by page; if the pool is empty, preempt the
     newest running sequence (recompute: drop its KV and requeue it at the
     front of the waiting queue as prompt + tokens generated so far);
  3. admit waiting sequences FIFO while seats, token budget, and blocks allow;
  4. return a StepPlan: for each running sequence, the tokens whose K/V are
     not yet in the slab (a whole prompt for prefill, one token for decode).

complete() appends one sampled token per sequence. Tokens are only ever
appended there, so a preempted sequence never emits a token twice.
"""

from __future__ import annotations

import enum
import logging
from collections import deque
from collections.abc import Sequence as SequenceT
from dataclasses import dataclass, field

from inference.engine import BlockAllocator, BlockPoolExhausted
from inference.kv_cache import pages_needed

_log = logging.getLogger(__name__)

DEFAULT_MAX_SEQS = 32
DEFAULT_TOKEN_BUDGET = 2048


class SeqState(enum.Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass(eq=False)
class Sequence:
    request_id: str
    prompt_ids: list[int]
    max_new_tokens: int
    temperature: float = 0.0
    eos_id: int | None = None
    output_ids: list[int] = field(default_factory=list)
    block_ids: list[int] = field(default_factory=list)
    # Tokens whose K/V are already in the slab.
    num_cached: int = 0
    state: SeqState = SeqState.WAITING
    preemptions: int = 0

    @property
    def all_ids(self) -> list[int]:
        return self.prompt_ids + self.output_ids

    @property
    def pending_ids(self) -> list[int]:
        return self.all_ids[self.num_cached :]


@dataclass(frozen=True)
class ScheduledSeq:
    seq: Sequence
    token_ids: list[int]
    past_len: int


@dataclass(frozen=True)
class StepPlan:
    items: tuple[ScheduledSeq, ...]

    @property
    def num_tokens(self) -> int:
        return sum(len(item.token_ids) for item in self.items)


@dataclass(frozen=True)
class Emission:
    seq: Sequence
    token_id: int
    finished: bool


class Scheduler:
    def __init__(
        self,
        pool: BlockAllocator,
        page_size: int,
        max_seqs: int = DEFAULT_MAX_SEQS,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
    ) -> None:
        if page_size < 1 or max_seqs < 1 or token_budget < 1:
            raise ValueError("page_size, max_seqs, and token_budget must be >= 1")
        self._pool = pool
        self._page = page_size
        self._max_seqs = max_seqs
        self._budget = token_budget
        self._waiting: deque[Sequence] = deque()
        self._running: list[Sequence] = []
        self._aborted: set[str] = set()
        self._ids: dict[str, Sequence] = {}

    @property
    def num_waiting(self) -> int:
        return len(self._waiting)

    @property
    def num_running(self) -> int:
        return len(self._running)

    def has_work(self) -> bool:
        return bool(self._waiting or self._running or self._aborted)

    def submit(self, seq: Sequence) -> None:
        if not seq.prompt_ids:
            raise ValueError("prompt_ids must be non-empty")
        if seq.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        if seq.request_id in self._ids:
            raise ValueError(f"duplicate request_id {seq.request_id}")
        # Worst case the sequence holds every page it will ever need at once.
        need = pages_needed(len(seq.prompt_ids) + seq.max_new_tokens, self._page)
        total = int(self._pool.num_blocks)
        if need > total:
            raise BlockPoolExhausted(
                f"out of KV blocks: request needs {need} pages, pool has {total}"
            )
        seq.state = SeqState.WAITING
        self._waiting.append(seq)
        self._ids[seq.request_id] = seq

    def abort(self, request_id: str) -> None:
        """Takes effect at the next schedule(); safe while a step is in flight."""
        if request_id in self._ids:
            self._aborted.add(request_id)

    def schedule(self) -> StepPlan:
        self._reap_aborted()
        self._grow_running()
        self._admit()
        items = tuple(
            ScheduledSeq(seq=s, token_ids=s.pending_ids, past_len=s.num_cached) for s in self._running
        )
        return StepPlan(items=items)

    def complete(self, plan: StepPlan, next_ids: SequenceT[int]) -> list[Emission]:
        if len(next_ids) != len(plan.items):
            raise ValueError("one sampled token per scheduled sequence is required")
        out: list[Emission] = []
        for item, token in zip(plan.items, next_ids):
            seq = item.seq
            if seq.request_id in self._aborted or seq.state is not SeqState.RUNNING:
                continue
            seq.num_cached = item.past_len + len(item.token_ids)
            seq.output_ids.append(int(token))
            finished = (seq.eos_id is not None and int(token) == seq.eos_id) or len(
                seq.output_ids
            ) >= seq.max_new_tokens
            if finished:
                self._finish(seq)
            out.append(Emission(seq=seq, token_id=int(token), finished=finished))
        return out

    def _reap_aborted(self) -> None:
        for request_id in list(self._aborted):
            seq = self._ids.pop(request_id, None)
            if seq is None:
                continue
            if seq.state is SeqState.WAITING:
                self._waiting.remove(seq)
            elif seq.state is SeqState.RUNNING:
                self._running.remove(seq)
                self._release(seq)
            seq.state = SeqState.FINISHED
            _log.info("sequence aborted request_id=%s", request_id)
        self._aborted.clear()

    def _grow_running(self) -> None:
        i = 0
        while i < len(self._running):
            seq = self._running[i]
            need = pages_needed(len(seq.all_ids), self._page) - len(seq.block_ids)
            while need > 0 and seq.state is SeqState.RUNNING:
                if self._pool.num_free >= need:
                    seq.block_ids.extend(self._pool.allocate(need))
                    need = 0
                else:
                    self._preempt(self._running[-1])
            if seq.state is SeqState.RUNNING:
                i += 1

    def _admit(self) -> None:
        tokens = sum(len(s.pending_ids) for s in self._running)
        while self._waiting and len(self._running) < self._max_seqs:
            seq = self._waiting[0]
            n = len(seq.all_ids)
            if tokens and tokens + n > self._budget:
                break
            need = pages_needed(n, self._page)
            if self._pool.num_free < need:
                break
            self._waiting.popleft()
            seq.block_ids = list(self._pool.allocate(need))
            seq.num_cached = 0
            seq.state = SeqState.RUNNING
            self._running.append(seq)
            tokens += n

    def _preempt(self, seq: Sequence) -> None:
        self._running.remove(seq)
        self._release(seq)
        seq.num_cached = 0
        seq.state = SeqState.WAITING
        seq.preemptions += 1
        self._waiting.appendleft(seq)
        _log.info(
            "sequence preempted request_id=%s tokens=%s preemptions=%s",
            seq.request_id,
            len(seq.all_ids),
            seq.preemptions,
        )

    def _finish(self, seq: Sequence) -> None:
        self._running.remove(seq)
        self._release(seq)
        seq.state = SeqState.FINISHED
        self._ids.pop(seq.request_id, None)

    def _release(self, seq: Sequence) -> None:
        if seq.block_ids:
            self._pool.free(seq.block_ids)
            seq.block_ids = []
