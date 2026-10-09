"""Scheduler bookkeeping with an in-memory pool; no model involved."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from inference.engine import BlockPoolExhausted
from inference.scheduler import Scheduler, SeqState, Sequence
from recording_pool import RecordingPool

EOS = 99


def _seq(rid: str, prompt_len: int, max_new: int = 8) -> Sequence:
    return Sequence(request_id=rid, prompt_ids=list(range(1, prompt_len + 1)), max_new_tokens=max_new, eos_id=EOS)


def _step(sched: Scheduler, token: int = 7):
    plan = sched.schedule()
    return plan, sched.complete(plan, [token] * len(plan.items))


class AdmissionTest(unittest.TestCase):
    def test_fifo_and_prefill_then_decode(self) -> None:
        pool = RecordingPool(num_blocks=8)
        sched = Scheduler(pool, page_size=4)
        a, b = _seq("a", 5), _seq("b", 2)
        sched.submit(a)
        sched.submit(b)
        plan = sched.schedule()
        self.assertEqual([i.seq.request_id for i in plan.items], ["a", "b"])
        self.assertEqual([len(i.token_ids) for i in plan.items], [5, 2])
        self.assertEqual([i.past_len for i in plan.items], [0, 0])
        self.assertEqual(len(a.block_ids), 2)
        sched.complete(plan, [7, 8])
        plan = sched.schedule()
        self.assertEqual([(i.token_ids, i.past_len) for i in plan.items], [([7], 5), ([8], 2)])

    def test_max_seqs_and_token_budget(self) -> None:
        pool = RecordingPool(num_blocks=16)
        sched = Scheduler(pool, page_size=4, max_seqs=2, token_budget=6)
        for rid in ("a", "b", "c"):
            sched.submit(_seq(rid, 4))
        plan = sched.schedule()
        # a fits, b would exceed the budget of 6 tokens.
        self.assertEqual([i.seq.request_id for i in plan.items], ["a"])
        sched.complete(plan, [7])
        plan = sched.schedule()
        self.assertEqual([i.seq.request_id for i in plan.items], ["a", "b"])
        sched.complete(plan, [7, 7])
        plan = sched.schedule()
        self.assertEqual(len(plan.items), 2, "max_seqs keeps c waiting")
        self.assertEqual(sched.num_waiting, 1)

    def test_oversized_prompt_runs_alone(self) -> None:
        pool = RecordingPool(num_blocks=16)
        sched = Scheduler(pool, page_size=4, token_budget=4)
        sched.submit(_seq("a", 10))
        plan = sched.schedule()
        self.assertEqual(plan.num_tokens, 10)

    def test_waits_for_blocks(self) -> None:
        pool = RecordingPool(num_blocks=2)
        sched = Scheduler(pool, page_size=4)
        sched.submit(_seq("a", 7, max_new=1))
        sched.submit(_seq("b", 4, max_new=2))
        plan = sched.schedule()
        self.assertEqual([i.seq.request_id for i in plan.items], ["a"])
        self.assertEqual(pool.num_free, 0)
        emitted = sched.complete(plan, [7])
        self.assertTrue(emitted[0].finished)
        plan = sched.schedule()
        self.assertEqual([i.seq.request_id for i in plan.items], ["b"])

    def test_pool_too_small_for_request(self) -> None:
        sched = Scheduler(RecordingPool(num_blocks=2), page_size=4)
        with self.assertRaises(BlockPoolExhausted):
            sched.submit(_seq("a", 6, max_new=4))

    def test_rejects_duplicate_and_empty(self) -> None:
        sched = Scheduler(RecordingPool(num_blocks=4), page_size=4)
        sched.submit(_seq("a", 2))
        with self.assertRaises(ValueError):
            sched.submit(_seq("a", 2))
        with self.assertRaises(ValueError):
            sched.submit(Sequence(request_id="b", prompt_ids=[], max_new_tokens=1))


class FinishTest(unittest.TestCase):
    def test_eos_and_cap_free_blocks(self) -> None:
        pool = RecordingPool(num_blocks=4)
        sched = Scheduler(pool, page_size=4)
        sched.submit(_seq("a", 2, max_new=5))
        sched.submit(_seq("b", 2, max_new=1))
        plan = sched.schedule()
        emitted = sched.complete(plan, [EOS, 3])
        self.assertEqual([(e.seq.request_id, e.finished) for e in emitted], [("a", True), ("b", True)])
        self.assertEqual(pool.num_free, 4)
        self.assertFalse(sched.has_work())


class PreemptionTest(unittest.TestCase):
    def test_newest_is_recomputed_without_reemitting(self) -> None:
        pool = RecordingPool(num_blocks=2)
        sched = Scheduler(pool, page_size=4)
        a, b = _seq("a", 3, max_new=4), _seq("b", 3, max_new=4)
        sched.submit(a)
        sched.submit(b)
        sched.complete(sched.schedule(), [10, 20])
        sched.complete(sched.schedule(), [11, 21])  # 5 tokens each: page 1 is full
        plan = sched.schedule()
        # a needs a second page; the pool is empty, so b (newest) is preempted.
        self.assertEqual([i.seq.request_id for i in plan.items], ["a"])
        self.assertEqual(b.state, SeqState.WAITING)
        self.assertEqual(b.block_ids, [])
        self.assertEqual(b.output_ids, [20, 21])
        self.assertEqual(b.preemptions, 1)
        emitted = sched.complete(plan, [12])
        self.assertEqual([e.token_id for e in emitted], [12])

    def test_recompute_prefills_prompt_plus_output(self) -> None:
        pool = RecordingPool(num_blocks=3)
        sched = Scheduler(pool, page_size=4)
        a, b = _seq("a", 3, max_new=2), _seq("b", 3)
        sched.submit(a)
        sched.submit(b)
        sched.complete(sched.schedule(), [10, 20])
        plan = sched.schedule()  # a grows into page 2 of 3; b still fits in page 1
        self.assertEqual(len(plan.items), 2)
        sched.complete(plan, [11, 21])  # a finishes on the cap and frees 2 pages
        self.assertEqual(pool.num_free, 2)
        # Force a preemption of b by hand and check the recompute plan.
        sched._preempt(b)
        plan = sched.schedule()
        item = plan.items[0]
        self.assertEqual(item.seq.request_id, "b")
        self.assertEqual(item.token_ids, [1, 2, 3, 20, 21])
        self.assertEqual(item.past_len, 0)

    def test_queue_front_after_preemption(self) -> None:
        pool = RecordingPool(num_blocks=2)
        sched = Scheduler(pool, page_size=4)
        for rid in ("a", "b"):
            sched.submit(_seq(rid, 3, max_new=4))
        sched.complete(sched.schedule(), [1, 2])
        sched.complete(sched.schedule(), [1, 2])
        sched.submit(_seq("c", 1, max_new=4))
        sched.schedule()
        self.assertEqual(sched._waiting[0].request_id, "b")


class AbortTest(unittest.TestCase):
    def test_abort_running_frees_on_next_schedule(self) -> None:
        pool = RecordingPool(num_blocks=4)
        sched = Scheduler(pool, page_size=4)
        sched.submit(_seq("a", 2))
        plan = sched.schedule()
        sched.abort("a")
        self.assertEqual(sched.complete(plan, [5]), [], "aborted in flight: nothing emitted")
        self.assertEqual(sched.schedule().items, ())
        self.assertEqual(pool.num_free, 4)
        self.assertFalse(sched.has_work())

    def test_abort_waiting(self) -> None:
        sched = Scheduler(RecordingPool(num_blocks=1), page_size=4)
        sched.submit(_seq("a", 3, max_new=1))
        sched.submit(_seq("b", 3, max_new=1))
        sched.abort("b")
        plan = sched.schedule()
        self.assertEqual([i.seq.request_id for i in plan.items], ["a"])
        self.assertEqual(sched.num_waiting, 0)

    def test_abort_unknown_is_noop(self) -> None:
        sched = Scheduler(RecordingPool(num_blocks=1), page_size=4)
        sched.abort("missing")
        self.assertFalse(sched.has_work())


if __name__ == "__main__":
    unittest.main()
