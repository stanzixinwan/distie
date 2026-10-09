"""Streaming client for an OpenAI-compatible server (vLLM `vllm serve`).

Sends POST /v1/chat/completions with stream=true and reads Server-Sent
Events: one `data: {json}` line per chunk, ending with `data: [DONE]`.
TTFT is the first chunk carrying content (vLLM's first chunk holds only the
role). Token counts come from the final usage chunk (stream_options).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import httpx

from stats import RequestRecord
from workload import BenchRequest

_log = logging.getLogger(__name__)

Sender = Callable[[BenchRequest, str, float], Awaitable[RequestRecord]]


def base_url(target: str) -> str:
    return target.rstrip("/") if target.startswith(("http://", "https://")) else f"http://{target}"


async def send_one(
    client: httpx.AsyncClient,
    req: BenchRequest,
    request_id: str,
    model_name: str,
    timeout_s: float,
    bench_start: float,
) -> RequestRecord:
    body = {
        "model": model_name,
        "messages": [{"role": "user", "content": req.prompt}],
        "max_tokens": req.max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = time.perf_counter()
    first: float | None = None
    chunks = 0
    completion_tokens: int | None = None
    finished = False
    error: str | None = None
    try:
        async with asyncio.timeout(timeout_s):
            async with client.stream("POST", "/v1/chat/completions", json=body) as resp:
                if resp.status_code != 200:
                    detail = (await resp.aread())[:200]
                    error = f"HTTP_{resp.status_code}"
                    _log.debug("request failed id=%s status=%s body=%r", request_id, resp.status_code, detail)
                else:
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            finished = True
                            break
                        chunk = json.loads(data)
                        usage = chunk.get("usage")
                        if usage:
                            completion_tokens = usage.get("completion_tokens")
                        for choice in chunk.get("choices") or ():
                            if choice.get("delta", {}).get("content"):
                                if first is None:
                                    first = time.perf_counter()
                                chunks += 1
                            if choice.get("finish_reason"):
                                finished = True
    except TimeoutError:
        error = "DEADLINE_EXCEEDED"
    except httpx.ConnectError as exc:
        error = "UNAVAILABLE"
        _log.debug("request failed id=%s connect error=%s", request_id, exc)
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        error = type(exc).__name__
        _log.debug("request failed id=%s error=%r", request_id, exc)
    ended = time.perf_counter()
    if error is None and not finished:
        error = "INCOMPLETE"
    return RequestRecord(
        request_id=request_id,
        prompt_len=req.prompt_len,
        max_tokens=req.max_tokens,
        start_s=started - bench_start,
        ttft_s=None if first is None else first - started,
        latency_s=ended - started,
        output_tokens=completion_tokens if completion_tokens is not None else chunks,
        error=error,
    )


@asynccontextmanager
async def sender(
    target: str,
    model_name: str,
    timeout_s: float,
    transport: httpx.AsyncBaseTransport | None = None,
) -> AsyncIterator[Sender]:
    # The deadline is enforced per request in send_one; httpx's own timeouts
    # are per read and would cut off long prefills, so they are disabled.
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
    async with httpx.AsyncClient(
        base_url=base_url(target), timeout=None, limits=limits, transport=transport
    ) as client:

        async def send(req: BenchRequest, request_id: str, bench_start: float) -> RequestRecord:
            return await send_one(client, req, request_id, model_name, timeout_s, bench_start)

        yield send
