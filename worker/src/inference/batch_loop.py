"""The one loop that owns the GPU.

Requests enter through submit(), which returns that request's own
asyncio.Queue. A single asyncio task asks the Scheduler for a StepPlan,
runs it on a one-thread executor (so no two forwards ever touch the KV slab
together, and the event loop never blocks), then routes each sampled token
to its request's queue. Block allocation happens only in this task, so the
pool needs no lock.

With a StepStats attached, every step is timed in four parts (schedule,
GPU thread, event-loop resume, emit) and summarized every N steps.
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

from inference.scheduler import Emission, Scheduler, Sequence, StepPlan
from inference.step_stats import StepStats

_log = logging.getLogger(__name__)


class StepRunner(Protocol):
    def step(self, plan: StepPlan) -> list[int]: ...


class StepFailed(RuntimeError):
    """A forward step raised; every sequence in that step is dropped."""


class BatchLoop:
    def __init__(self, scheduler: Scheduler, runner: StepRunner, stats: StepStats | None = None) -> None:
        self._scheduler = scheduler
        self._runner = runner
        self._stats = stats
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="distie-gpu")
        self._queues: dict[str, asyncio.Queue] = {}
        self._wake = asyncio.Event()
        self._task: asyncio.Task | None = None
        self.steps = 0

    def submit(self, seq: Sequence) -> asyncio.Queue:
        """Raises BlockPoolExhausted or ValueError synchronously if the request can never run."""
        self._scheduler.submit(seq)
        queue: asyncio.Queue = asyncio.Queue()
        self._queues[seq.request_id] = queue
        self._ensure_task()
        self._wake.set()
        return queue

    def abort(self, request_id: str) -> None:
        """Called when a consumer stops reading; blocks are freed at the next step."""
        if self._queues.pop(request_id, None) is not None:
            self._scheduler.abort(request_id)
            self._wake.set()

    async def close(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        self._executor.shutdown(wait=True)
        if self._stats is not None:
            self._stats.flush()

    def _ensure_task(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(self._run(), name="distie-batch-loop")
            self._task.add_done_callback(_log_crash)

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        clock = time.perf_counter
        while True:
            t_sched = clock()
            plan = self._scheduler.schedule()
            if not plan.items:
                self._wake.clear()
                await self._wake.wait()
                continue
            t_run = clock()
            try:
                if self._stats is None:
                    next_ids = await loop.run_in_executor(self._executor, self._runner.step, plan)
                    gpu_s = 0.0
                else:
                    next_ids, gpu_s = await loop.run_in_executor(self._executor, self._timed_step, plan)
            except Exception as exc:
                _log.exception("forward step failed seqs=%s tokens=%s", len(plan.items), plan.num_tokens)
                self._fail(plan, exc)
                continue
            t_emit = clock()
            self.steps += 1
            for emission in self._scheduler.complete(plan, next_ids):
                self._deliver(emission)
            if self._stats is not None:
                decode = all(len(item.token_ids) == 1 for item in plan.items)
                self._stats.record(
                    "decode" if decode else "prefill",
                    rows=len(plan.items),
                    tokens=plan.num_tokens,
                    sched=(t_run - t_sched) * 1e3,
                    gpu=gpu_s * 1e3,
                    wait=max(0.0, (t_emit - t_run) - gpu_s) * 1e3,
                    emit=(clock() - t_emit) * 1e3,
                )

    def _timed_step(self, plan: StepPlan) -> tuple[list[int], float]:
        """Runs on the GPU thread, so the time excludes waiting for the event loop."""
        start = time.perf_counter()
        next_ids = self._runner.step(plan)
        return next_ids, time.perf_counter() - start

    def _deliver(self, emission: Emission) -> None:
        rid = emission.seq.request_id
        queue = self._queues.get(rid)
        if queue is None:
            return
        queue.put_nowait(emission)
        if emission.finished:
            self._queues.pop(rid, None)

    def _fail(self, plan: StepPlan, exc: Exception) -> None:
        # The traceback holds this loop's suspended frame. A consumer that clears
        # traceback frames (unittest does) would close the loop coroutine, so each
        # request gets its own exception and the cause carries no traceback.
        cause = exc.with_traceback(None)
        for item in plan.items:
            rid = item.seq.request_id
            queue = self._queues.pop(rid, None)
            self._scheduler.abort(rid)
            if queue is not None:
                err = StepFailed(f"forward step failed: {cause!r}")
                err.__cause__ = cause
                queue.put_nowait(err)


def _log_crash(task: asyncio.Task) -> None:
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        _log.error("batch loop stopped", exc_info=exc)
