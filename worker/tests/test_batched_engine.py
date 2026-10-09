"""TorchEngine with a paged cache: requests share one scheduler loop."""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import grpc
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from inference.engine import BlockPoolExhausted, GenerateRequest
from inference.kv_cache import KvShape, PagedKvCache
from inference.qwen2 import Qwen2CausalLM
from inference.torch_engine import TorchEngine
from proto_gen import inference_pb2, inference_pb2_grpc
from recording_pool import RecordingPool
from worker.config import Config
from worker.server import build_server


class WordTokenizer:
    """'3 5 7' -> [3, 5, 7]. EOS id 0 is unlikely to be picked by a random net."""

    eos_token_id = 0
    pad_token_id = 0
    chat_template = None

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [int(w) for w in text.split()]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return f"t{ids[0]}"


PROMPTS = ["3 5 7", "11 13", "2 4 6 8 10", "9", "21 22 23 24 25 26 27", "30 31"]


def _engine(num_blocks: int = 64, page_size: int = 4) -> tuple[TorchEngine, RecordingPool]:
    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=48,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        tie_word_embeddings=False,
    )
    model = Qwen2CausalLM.from_hf(Qwen2ForCausalLM(config).eval())
    pool = RecordingPool(num_blocks=num_blocks)
    kv = PagedKvCache(
        KvShape.from_hf_config(config),
        num_blocks=num_blocks,
        device="cpu",
        dtype=torch.float32,
        page_size=page_size,
    )
    return TorchEngine(model, WordTokenizer(), "cpu", pool=pool, kv_cache=kv), pool


def _req(i: int, prompt: str, max_tokens: int = 10) -> GenerateRequest:
    return GenerateRequest(request_id=f"r{i}", model_name="tiny", prompt=prompt, max_tokens=max_tokens)


async def _collect(engine: TorchEngine, req: GenerateRequest) -> list[str]:
    return [event.token async for event in engine.generate(req)]


async def _until_free(pool: RecordingPool, timeout_s: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while pool.num_free != pool.num_blocks:
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError(f"blocks not released: free {pool.num_free}/{pool.num_blocks}")
        await asyncio.sleep(0.01)


class BatchedEngineTest(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_matches_sequential(self) -> None:
        engine, pool = _engine()
        try:
            sequential = [await _collect(engine, _req(i, p)) for i, p in enumerate(PROMPTS)]
            steps_alone = engine._loop.steps
            concurrent = await asyncio.gather(*(_collect(engine, _req(i, p)) for i, p in enumerate(PROMPTS)))
            self.assertEqual(list(concurrent), sequential)
            batched_steps = engine._loop.steps - steps_alone
            self.assertLess(batched_steps, steps_alone, "concurrent requests should share steps")
            self.assertEqual(pool.num_free, pool.num_blocks)
        finally:
            await engine.close()

    async def test_preemption_keeps_outputs(self) -> None:
        roomy, _ = _engine()
        tight, pool = _engine(num_blocks=8)
        try:
            want = [await _collect(roomy, _req(i, p)) for i, p in enumerate(PROMPTS)]
            with self.assertLogs("inference.scheduler", level="INFO") as logs:
                got = await asyncio.gather(*(_collect(tight, _req(i, p)) for i, p in enumerate(PROMPTS)))
            self.assertTrue(any("preempted" in line for line in logs.output))
            self.assertEqual(list(got), want)
            self.assertEqual(pool.num_free, pool.num_blocks)
        finally:
            await roomy.close()
            await tight.close()

    async def test_cancel_frees_blocks(self) -> None:
        engine, pool = _engine()
        try:
            agen = engine.generate(_req(0, "3 5 7", max_tokens=50))
            await agen.__anext__()
            self.assertLess(pool.num_free, pool.num_blocks)
            await agen.aclose()
            await _until_free(pool)
        finally:
            await engine.close()

    async def test_request_larger_than_pool(self) -> None:
        engine, _ = _engine(num_blocks=2)
        try:
            with self.assertRaises(BlockPoolExhausted):
                await _collect(engine, _req(0, "1 2 3", max_tokens=20))
        finally:
            await engine.close()

    async def test_runner_error_reaches_caller(self) -> None:
        engine, pool = _engine()

        def boom(plan):
            raise RuntimeError("cuda error")

        try:
            engine._batch_loop()._runner.step = boom
            with self.assertRaises(RuntimeError):
                await _collect(engine, _req(0, "3 5"))
            await _until_free(pool)
        finally:
            await engine.close()


class ServicerCancelTest(unittest.IsolatedAsyncioTestCase):
    async def test_client_cancel_releases_blocks(self) -> None:
        engine, pool = _engine()
        cfg = Config(listen_addr="127.0.0.1:0", worker_id="t", shutdown_grace_s=1.0, token_delay_s=0.0)
        server, port = build_server(cfg, engine=engine)
        await server.start()
        channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
        stub = inference_pb2_grpc.InferenceServiceStub(channel)
        try:
            call = stub.InferStream(
                inference_pb2.InferenceRequest(
                    request_id="c1",
                    model_name="tiny",
                    prompt="3 5 7",
                    params=inference_pb2.GenerationParams(max_tokens=200),
                )
            )
            first = await call.read()
            self.assertTrue(first.token)
            call.cancel()
            await _until_free(pool)
        finally:
            await channel.close()
            await server.stop(0)
            await engine.close()


if __name__ == "__main__":
    unittest.main()
