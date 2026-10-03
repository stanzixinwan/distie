"""Open-loop async gRPC load generator for InferenceService.InferStream."""

from __future__ import annotations

import asyncio
import logging
import math
import random
import time
from collections.abc import Sequence

import grpc

from proto_gen import inference_pb2, inference_pb2_grpc
from stats import RequestRecord
from workload import BenchRequest

_log = logging.getLogger(__name__)


async def send_one(
    stub,
    req: BenchRequest,
    request_id: str,
    model_name: str,
    timeout_s: float,
    bench_start: float,
) -> RequestRecord:
    message = inference_pb2.InferenceRequest(
        request_id=request_id,
        model_name=model_name,
        prompt=req.prompt,
        params=inference_pb2.GenerationParams(max_tokens=req.max_tokens, temperature=0.0, stream=True),
    )
    started = time.perf_counter()
    first: float | None = None
    messages = 0
    completion_tokens: int | None = None
    error: str | None = None
    try:
        async for resp in stub.InferStream(message, timeout=timeout_s):
            if first is None:
                first = time.perf_counter()
            messages += 1
            if resp.finished:
                completion_tokens = resp.usage.completion_tokens
    except grpc.aio.AioRpcError as exc:
        error = exc.code().name
        _log.debug("request failed id=%s code=%s detail=%s", request_id, error, exc.details())
    ended = time.perf_counter()
    if error is None and completion_tokens is None:
        error = "INCOMPLETE"
    return RequestRecord(
        request_id=request_id,
        prompt_len=req.prompt_len,
        max_tokens=req.max_tokens,
        start_s=started - bench_start,
        ttft_s=None if first is None else first - started,
        latency_s=ended - started,
        output_tokens=completion_tokens if completion_tokens is not None else messages,
        error=error,
    )


async def run_benchmark(
    target: str,
    requests: Sequence[BenchRequest],
    rate: float = math.inf,
    max_concurrency: int | None = None,
    model_name: str = "distie",
    timeout_s: float = 300.0,
    warmup: int = 0,
    seed: int = 0,
) -> tuple[list[RequestRecord], float]:
    """Send requests with Poisson arrivals at `rate` req/s (inf = all at once).

    Returns (records in send order, wall-clock duration of the measured phase).
    Warmup requests run sequentially first and are not recorded.
    """
    if not requests:
        raise ValueError("requests must be non-empty")
    if rate <= 0:
        raise ValueError("rate must be > 0")
    if max_concurrency is not None and max_concurrency < 1:
        raise ValueError("max_concurrency must be >= 1")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")

    rng = random.Random(seed)
    limit = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    async with grpc.aio.insecure_channel(target) as channel:
        stub = inference_pb2_grpc.InferenceServiceStub(channel)

        for i in range(warmup):
            rec = await send_one(stub, requests[i % len(requests)], f"warmup-{i}", model_name, timeout_s, 0.0)
            if not rec.ok:
                raise RuntimeError(f"warmup request failed: {rec.error}")
        if warmup:
            _log.info("warmup done requests=%s", warmup)

        async def limited(i: int, req: BenchRequest, bench_start: float) -> RequestRecord:
            if limit is None:
                return await send_one(stub, req, f"bench-{i}", model_name, timeout_s, bench_start)
            async with limit:
                return await send_one(stub, req, f"bench-{i}", model_name, timeout_s, bench_start)

        bench_start = time.perf_counter()
        tasks = []
        for i, req in enumerate(requests):
            tasks.append(asyncio.create_task(limited(i, req, bench_start)))
            if not math.isinf(rate) and i < len(requests) - 1:
                await asyncio.sleep(rng.expovariate(rate))
        records = await asyncio.gather(*tasks)
        duration = time.perf_counter() - bench_start

    return list(records), duration
