from __future__ import annotations

import sys
import unittest
from pathlib import Path

import grpc

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from proto_gen import inference_pb2, inference_pb2_grpc
from worker.config import Config
from worker.server import build_server


class InferenceServicerTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        cfg = Config(
            listen_addr="127.0.0.1:0",
            worker_id="test-worker",
            shutdown_grace_s=1.0,
            token_delay_s=0.0,
        )
        self.server, port = build_server(cfg)
        await self.server.start()
        self.channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
        self.stub = inference_pb2_grpc.InferenceServiceStub(self.channel)

    async def asyncTearDown(self) -> None:
        await self.channel.close()
        await self.server.stop(0)

    async def test_infer_echoes_prompt(self) -> None:
        resp = await self.stub.Infer(
            inference_pb2.InferenceRequest(
                request_id="r1",
                model_name="fake-lm",
                prompt="hello world",
            )
        )
        self.assertEqual(resp.token, "hello world")
        self.assertTrue(resp.finished)
        self.assertEqual(resp.usage.completion_tokens, 2)

    async def test_infer_stream_marks_last_finished(self) -> None:
        chunks = [
            resp
            async for resp in self.stub.InferStream(
                inference_pb2.InferenceRequest(
                    request_id="r2",
                    model_name="fake-lm",
                    prompt="a b c",
                )
            )
        ]
        self.assertEqual([c.token for c in chunks], ["a", "b", "c"])
        self.assertFalse(chunks[0].finished)
        self.assertTrue(chunks[-1].finished)
        self.assertEqual(chunks[-1].usage.total_tokens, 6)

    async def test_infer_rejects_empty_prompt(self) -> None:
        with self.assertRaises(grpc.aio.AioRpcError) as ctx:
            await self.stub.Infer(
                inference_pb2.InferenceRequest(model_name="fake-lm", prompt="  ")
            )
        self.assertEqual(ctx.exception.code(), grpc.StatusCode.INVALID_ARGUMENT)


if __name__ == "__main__":
    unittest.main()
