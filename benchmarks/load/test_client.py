"""Client against an in-process Worker running FakeEngine (binds a local port)."""

from __future__ import annotations

import math
import unittest

import bench_serving  # noqa: F401  (puts worker/src on sys.path)
from client import run_benchmark
from inference.engine import FakeEngine
from worker.config import Config
from worker.server import build_server
from workload import BenchRequest


class _OnePagePool:
    num_blocks = 1

    def __init__(self) -> None:
        self._free = [0]

    def allocate(self, count: int) -> list[int]:
        if count > len(self._free):
            raise RuntimeError("out of blocks")
        return [self._free.pop() for _ in range(count)]

    def free(self, block_ids: list[int]) -> None:
        self._free.extend(block_ids)


def _requests(n: int, words: int = 6, max_tokens: int = 4) -> list[BenchRequest]:
    return [BenchRequest(" ".join(f"w{j}" for j in range(words)), max_tokens, -1) for _ in range(n)]


class RunBenchmarkTest(unittest.IsolatedAsyncioTestCase):
    async def _serve(self, engine) -> str:
        cfg = Config(listen_addr="127.0.0.1:0", worker_id="bench", shutdown_grace_s=1.0, token_delay_s=0.0)
        server, port = build_server(cfg, engine=engine)
        await server.start()
        self.addAsyncCleanup(server.stop, 0)
        return f"127.0.0.1:{port}"

    async def test_records_streaming_metrics(self) -> None:
        target = await self._serve(FakeEngine(token_delay_s=0.01))
        records, duration = await run_benchmark(target, _requests(8), rate=math.inf, warmup=1)
        self.assertEqual(len(records), 8)
        self.assertGreater(duration, 0)
        for r in records:
            self.assertTrue(r.ok, r)
            self.assertEqual(r.output_tokens, 4)
            self.assertLessEqual(r.ttft_s, r.latency_s)
            self.assertGreater(r.tpot_s, 0.005)
        self.assertEqual([r.request_id for r in records], [f"bench-{i}" for i in range(8)])

    async def test_server_errors_are_recorded_not_raised(self) -> None:
        target = await self._serve(FakeEngine(pool=_OnePagePool()))
        records, _ = await run_benchmark(target, _requests(2, words=12, max_tokens=12))
        self.assertEqual({r.error for r in records}, {"RESOURCE_EXHAUSTED"})

    async def test_poisson_arrivals_spread_sends(self) -> None:
        target = await self._serve(FakeEngine())
        records, _ = await run_benchmark(target, _requests(5), rate=50.0, seed=1)
        starts = [r.start_s for r in records]
        self.assertEqual(starts, sorted(starts))
        self.assertGreater(starts[-1] - starts[0], 0.01)

    async def test_max_concurrency_serializes(self) -> None:
        target = await self._serve(FakeEngine(token_delay_s=0.01))
        records, _ = await run_benchmark(target, _requests(3), max_concurrency=1)
        ordered = sorted(records, key=lambda r: r.start_s)
        for prev, nxt in zip(ordered, ordered[1:]):
            self.assertGreaterEqual(nxt.start_s, prev.start_s + prev.latency_s - 1e-3)

    async def test_unreachable_target(self) -> None:
        records, _ = await run_benchmark("127.0.0.1:1", _requests(1), timeout_s=2.0)
        self.assertEqual(records[0].error, "UNAVAILABLE")

    async def test_validates_arguments(self) -> None:
        with self.assertRaises(ValueError):
            await run_benchmark("127.0.0.1:1", [])
        with self.assertRaises(ValueError):
            await run_benchmark("127.0.0.1:1", _requests(1), rate=0)


if __name__ == "__main__":
    unittest.main()
