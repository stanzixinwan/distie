"""Serving benchmark: TTFT / TPOT / throughput over a streaming API.

--backend grpc (default): point --target at a Worker (:50052) or the
Gateway (:50051). The Gateway's leaky bucket defaults to 20 req/s; start it
with DISTIE_RATE_LIMIT_RPS=0 unless rate limiting is what you are measuring.

--backend openai: point --target at an OpenAI-compatible server, e.g.
`vllm serve ... --served-model-name distie` on http://localhost:8000.

Usage:
  python benchmarks/load/bench_serving.py --workload sharegpt \
      --dataset benchmarks/data/ShareGPT_V3_unfiltered_cleaned_split.json \
      --num-requests 200 --rate 4 --out benchmarks/results/serving.json

Exit code: 0 run finished, 1 every request failed, 2 setup error.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import platform
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_WORKER_SRC = _HERE.parents[1] / "worker" / "src"
for _p in (_HERE, _WORKER_SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from client import BACKENDS, run_benchmark  # noqa: E402
from stats import summarize  # noqa: E402
from workload import BenchRequest, load_sharegpt, synthetic  # noqa: E402

_log = logging.getLogger("bench_serving")


def build_requests(args: argparse.Namespace) -> list[BenchRequest]:
    if args.workload == "synthetic":
        return synthetic(args.num_requests, seed=args.seed)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    return load_sharegpt(
        args.dataset,
        args.num_requests,
        tokenizer,
        seed=args.seed,
        max_output_len=args.max_output_len,
    )


def format_summary(summary: dict) -> str:
    lines = [
        f"requests      {summary['completed']}/{summary['requests']} ok"
        + (f"  errors={summary['errors']}" if summary["failed"] else ""),
        f"duration      {summary['duration_s']:.2f} s",
        f"throughput    {summary['request_throughput']:.2f} req/s   "
        f"{summary['output_throughput_tok_s']:.1f} output tok/s",
    ]
    for key, label in (("ttft_ms", "TTFT"), ("tpot_ms", "TPOT"), ("e2e_latency_ms", "E2E")):
        dist = summary[key]
        if dist is None:
            lines.append(f"{label:<13} n/a")
            continue
        lines.append(
            f"{label:<13} mean {dist['mean']:8.1f}  p50 {dist['p50']:8.1f}  "
            f"p90 {dist['p90']:8.1f}  p99 {dist['p99']:8.1f}  ms"
        )
    return "\n".join(lines)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=BACKENDS, default="grpc")
    parser.add_argument("--target", default="localhost:50052")
    parser.add_argument("--workload", choices=["sharegpt", "synthetic"], default="sharegpt")
    parser.add_argument("--dataset", type=Path, help="ShareGPT JSON (required for --workload sharegpt)")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--num-requests", type=int, default=200)
    parser.add_argument("--rate", type=float, default=math.inf, help="req/s, Poisson arrivals; inf = all at once")
    parser.add_argument("--max-concurrency", type=int)
    parser.add_argument("--max-output-len", type=int, default=256)
    parser.add_argument("--model-name", default="distie")
    parser.add_argument("--timeout", type=float, default=300.0, help="per-request deadline, seconds")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, help="write a JSON report here")
    args = parser.parse_args(argv)
    if args.workload == "sharegpt" and args.dataset is None:
        parser.error("--dataset is required for --workload sharegpt (see `make sharegpt`)")
    if args.num_requests < 1:
        parser.error("--num-requests must be >= 1")
    if args.rate <= 0:
        parser.error("--rate must be > 0")
    return args


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args(argv)

    try:
        requests = build_requests(args)
    except (ImportError, OSError, ValueError) as exc:
        _log.error("workload setup failed: %s", exc)
        return 2
    _log.info(
        "workload=%s requests=%s rate=%s backend=%s target=%s",
        args.workload, len(requests), args.rate, args.backend, args.target,
    )

    try:
        records, duration = asyncio.run(
            run_benchmark(
                args.target,
                requests,
                rate=args.rate,
                max_concurrency=args.max_concurrency,
                model_name=args.model_name,
                timeout_s=args.timeout,
                warmup=args.warmup,
                seed=args.seed,
                backend=args.backend,
            )
        )
    except ImportError as exc:
        _log.error("backend %s unavailable: %s", args.backend, exc)
        return 2
    except (RuntimeError, ValueError) as exc:
        _log.error("benchmark aborted: %s", exc)
        return 2

    summary = summarize(records, duration)
    print(format_summary(summary))

    if args.out:
        report = {
            "config": {
                "backend": args.backend,
                "target": args.target,
                "workload": args.workload,
                "dataset": str(args.dataset) if args.dataset else None,
                "tokenizer": args.tokenizer if args.workload == "sharegpt" else None,
                "num_requests": args.num_requests,
                "rate": "inf" if math.isinf(args.rate) else args.rate,
                "max_concurrency": args.max_concurrency,
                "max_output_len": args.max_output_len,
                "warmup": args.warmup,
                "seed": args.seed,
            },
            "client": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
            "summary": summary,
            "records": [r.to_dict() for r in records],
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        _log.info("report written to %s", args.out)

    return 1 if summary["completed"] == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
