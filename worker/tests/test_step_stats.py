"""Per-step time breakdown: aggregation, and BatchLoop recording one entry per step."""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from inference.batch_loop import BatchLoop
from inference.scheduler import Scheduler, Sequence, StepPlan
from inference.step_stats import StepStats
from recording_pool import RecordingPool


class StepStatsTest(unittest.TestCase):
    def test_groups_average_and_reset(self) -> None:
        stats = StepStats(every=100)
        stats.record("decode", rows=4, tokens=4, sched=1.0, gpu=8.0, wait=0.5, emit=0.5)
        stats.record("decode", rows=2, tokens=2, sched=1.0, gpu=6.0, wait=1.5, emit=1.5)
        stats.record("prefill", rows=3, tokens=30, sched=2.0, gpu=16.0, wait=1.0, emit=1.0)
        line = stats.format()
        self.assertIn("steps=3", line)
        self.assertIn("decode n=2 rows=3.0 tok=3 ms/step[sched=1.00 gpu=7.00 wait=1.00 emit=1.00] share=50%", line)
        self.assertIn("prefill n=1 rows=3.0 tok=30 ms/step[sched=2.00 gpu=16.00 wait=1.00 emit=1.00] share=50%", line)
        self.assertIn("wall sched=10% gpu=75% wait=8% emit=8%", line)
        with self.assertLogs("inference.step_stats", "INFO"):
            self.assertEqual(stats.flush(), line)
        self.assertIsNone(stats.flush())

    def test_logs_every_n_steps(self) -> None:
        stats = StepStats(every=2)
        with self.assertLogs("inference.step_stats", "INFO") as logs:
            for _ in range(5):
                stats.record("decode", rows=1, tokens=1, sched=0.1, gpu=1.0, wait=0.0, emit=0.1)
        self.assertEqual(len(logs.records), 2)

    def test_rejects_missing_parts_and_bad_interval(self) -> None:
        with self.assertRaises(ValueError):
            StepStats(every=0)
        with self.assertRaises(ValueError):
            StepStats(every=1).record("decode", rows=1, tokens=1, sched=0.1, gpu=1.0)


class EchoRunner:
    """Each row samples (last token + 1); records whether the step was pure decode."""

    def __init__(self) -> None:
        self.kinds: list[str] = []

    def step(self, plan: StepPlan) -> list[int]:
        self.kinds.append("decode" if all(len(i.token_ids) == 1 for i in plan.items) else "prefill")
        return [item.token_ids[-1] + 1 for item in plan.items]


class RecordingStats(StepStats):
    def __init__(self) -> None:
        super().__init__(every=1000)
        self.entries: list[tuple[str, int, int, dict]] = []

    def record(self, kind: str, rows: int, tokens: int, **parts: float) -> None:
        self.entries.append((kind, rows, tokens, parts))
        super().record(kind, rows, tokens, **parts)


class BatchLoopStatsTest(unittest.TestCase):
    def test_one_entry_per_step_and_same_tokens(self) -> None:
        async def run(stats: StepStats | None) -> tuple[list[list[int]], EchoRunner]:
            runner = EchoRunner()
            loop = BatchLoop(Scheduler(RecordingPool(32), page_size=4), runner, stats)
            seqs = [Sequence(request_id=f"r{i}", prompt_ids=[10 * i + 1, 10 * i + 2], max_new_tokens=3) for i in range(3)]
            queues = [loop.submit(s) for s in seqs]
            outputs = []
            for queue in queues:
                ids = []
                while True:
                    emission = await queue.get()
                    ids.append(emission.token_id)
                    if emission.finished:
                        break
                outputs.append(ids)
            await loop.close()
            return outputs, runner

        plain, _ = asyncio.run(run(None))
        stats = RecordingStats()
        with self.assertLogs("inference.step_stats", "INFO"):
            timed, runner = asyncio.run(run(stats))
        self.assertEqual(timed, plain)
        self.assertEqual([e[0] for e in stats.entries], runner.kinds)
        for kind, rows, tokens, parts in stats.entries:
            self.assertEqual(set(parts), {"sched", "gpu", "wait", "emit"})
            self.assertTrue(all(v >= 0 for v in parts.values()))
        prefill = [e for e in stats.entries if e[0] == "prefill"]
        self.assertEqual(prefill[0][1:3], (3, 6))


if __name__ == "__main__":
    unittest.main()
