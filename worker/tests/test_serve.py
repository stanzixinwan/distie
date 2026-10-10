"""serve(): signal or stop event drains the server, then closes the engine."""

from __future__ import annotations

import asyncio
import os
import signal
import socket
import sys
import unittest
from pathlib import Path

import grpc

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from inference.engine import FakeEngine
from proto_gen import inference_pb2, inference_pb2_grpc
from recording_pool import RecordingPool
from worker.config import Config
from worker.server import serve


class _ClosingEngine(FakeEngine):
    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.closed = 0

    async def close(self) -> None:
        self.closed += 1


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ServeShutdownTest(unittest.IsolatedAsyncioTestCase):
    async def _start(self, engine, stop=None, grace_s: float = 0.2):
        port = _free_port()
        cfg = Config(listen_addr=f"127.0.0.1:{port}", worker_id="t", shutdown_grace_s=grace_s, token_delay_s=0.0)
        task = asyncio.create_task(serve(cfg, engine=engine, stop=stop))
        channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
        self.addAsyncCleanup(channel.close)
        await asyncio.wait_for(channel.channel_ready(), 5.0)
        return task, inference_pb2_grpc.InferenceServiceStub(channel)

    async def test_stop_cancels_streams_then_closes_engine(self) -> None:
        pool = RecordingPool(num_blocks=64)
        engine = _ClosingEngine(token_delay_s=0.05, pool=pool)
        stop = asyncio.Event()
        task, stub = await self._start(engine, stop)

        call = stub.InferStream(
            inference_pb2.InferenceRequest(request_id="r1", model_name="m", prompt=" ".join(["w"] * 200))
        )
        await call.read()
        stop.set()
        await asyncio.wait_for(task, 5.0)

        self.assertEqual(engine.closed, 1)
        self.assertEqual(pool.num_free, pool.num_blocks)
        with self.assertRaises(grpc.aio.AioRpcError):
            while (await call.read()) is not grpc.aio.EOF:
                pass

    @unittest.skipUnless(hasattr(signal, "SIGTERM") and os.name == "posix", "needs POSIX signals")
    async def test_sigterm_triggers_shutdown(self) -> None:
        engine = _ClosingEngine()
        task, _stub = await self._start(engine)
        os.kill(os.getpid(), signal.SIGTERM)
        await asyncio.wait_for(task, 5.0)
        self.assertEqual(engine.closed, 1)
        # Handlers are removed, so the test process keeps default signal behavior.
        self.assertIs(signal.getsignal(signal.SIGTERM), signal.SIG_DFL)

    async def test_engine_closed_when_cancelled(self) -> None:
        engine = _ClosingEngine()
        task, _stub = await self._start(engine)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(engine.closed, 1)


if __name__ == "__main__":
    unittest.main()
