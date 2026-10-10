"""OpenAI streaming client against httpx.MockTransport serving vLLM-shaped SSE."""

from __future__ import annotations

import asyncio
import json
import unittest

import httpx

import bench_serving  # noqa: F401  (puts worker/src on sys.path)
from client import run_benchmark
from openai_client import base_url
from workload import BenchRequest


def _chunk(content: str | None = None, finish: str | None = None, usage: dict | None = None) -> bytes:
    choices = [] if usage else [{"index": 0, "delta": {} if content is None else {"content": content}, "finish_reason": finish}]
    body = {"id": "chatcmpl-1", "object": "chat.completion.chunk", "choices": choices, "usage": usage}
    return f"data: {json.dumps(body)}\n\n".encode()


def _vllm_stream(tokens: list[str], usage: bool = True, done: bool = True) -> list[bytes]:
    parts = [b'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}\n\n']
    parts += [_chunk(t) for t in tokens]
    parts.append(_chunk(finish="stop"))
    if usage:
        parts.append(_chunk(usage={"prompt_tokens": 5, "completion_tokens": len(tokens), "total_tokens": 5 + len(tokens)}))
    if done:
        parts.append(b"data: [DONE]\n\n")
    return parts


async def _paced(parts: list[bytes], delay_s: float):
    for p in parts:
        await asyncio.sleep(delay_s)
        yield p


def _requests(n: int, max_tokens: int = 3) -> list[BenchRequest]:
    return [BenchRequest(f"prompt {i}", max_tokens, -1) for i in range(n)]


class OpenAIBackendTest(unittest.IsolatedAsyncioTestCase):
    async def _run(self, handler, n: int = 1, **kw):
        return await run_benchmark(
            "localhost:8000", _requests(n), backend="openai", transport=httpx.MockTransport(handler), **kw
        )

    async def test_records_streaming_metrics(self) -> None:
        bodies = []

        async def handler(request: httpx.Request) -> httpx.Response:
            bodies.append((request.url.path, json.loads(request.content)))
            return httpx.Response(200, content=_paced(_vllm_stream(["a", "b", "c"]), 0.005))

        records, duration = await self._run(handler, n=4, warmup=1)
        self.assertEqual(len(records), 4)
        self.assertGreater(duration, 0)
        for r in records:
            self.assertTrue(r.ok, r)
            self.assertEqual(r.output_tokens, 3)
            self.assertLess(r.ttft_s, r.latency_s)
            self.assertGreater(r.tpot_s, 0)
        path, body = bodies[0]
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(body["model"], "distie")
        self.assertTrue(body["stream"])
        self.assertEqual(body["stream_options"], {"include_usage": True})
        self.assertEqual(body["max_tokens"], 3)
        self.assertEqual(body["temperature"], 0.0)
        self.assertEqual(body["messages"][0]["role"], "user")
        self.assertEqual(len(bodies), 5)

    async def test_ttft_skips_role_only_chunk(self) -> None:
        parts = _vllm_stream(["x"])

        async def slow_first_token():
            yield parts[0]
            await asyncio.sleep(0.05)
            for p in parts[1:]:
                yield p

        records, _ = await self._run(lambda req: httpx.Response(200, content=slow_first_token()))
        self.assertGreaterEqual(records[0].ttft_s, 0.04)

    async def test_counts_chunks_without_usage(self) -> None:
        records, _ = await self._run(lambda req: httpx.Response(200, content=b"".join(_vllm_stream(["a", "b"], usage=False))))
        self.assertTrue(records[0].ok)
        self.assertEqual(records[0].output_tokens, 2)

    async def test_http_error_is_recorded(self) -> None:
        records, _ = await self._run(lambda req: httpx.Response(404, json={"message": "model not found"}))
        self.assertEqual(records[0].error, "HTTP_404")

    async def test_truncated_stream_is_incomplete(self) -> None:
        parts = [b'data: {"choices":[{"index":0,"delta":{"content":"a"}}]}\n\n']
        records, _ = await self._run(lambda req: httpx.Response(200, content=b"".join(parts)))
        self.assertEqual(records[0].error, "INCOMPLETE")

    async def test_deadline(self) -> None:
        records, _ = await self._run(
            lambda req: httpx.Response(200, content=_paced(_vllm_stream(["a"] * 20), 0.05)), timeout_s=0.1
        )
        self.assertEqual(records[0].error, "DEADLINE_EXCEEDED")

    async def test_connect_error_is_unavailable(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        records, _ = await self._run(handler)
        self.assertEqual(records[0].error, "UNAVAILABLE")

    async def test_rejects_unknown_backend(self) -> None:
        with self.assertRaises(ValueError):
            await run_benchmark("x", _requests(1), backend="tgi")


class BaseUrlTest(unittest.TestCase):
    def test_adds_scheme(self) -> None:
        self.assertEqual(base_url("localhost:8000"), "http://localhost:8000")
        self.assertEqual(base_url("https://host/"), "https://host")


if __name__ == "__main__":
    unittest.main()
