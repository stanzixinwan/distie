"""Per-request records and aggregate serving metrics. Pure; no networking."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RequestRecord:
    request_id: str
    prompt_len: int
    max_tokens: int
    # Seconds since benchmark start when the request was sent.
    start_s: float
    # Seconds from send to the first stream message; None if none arrived.
    ttft_s: float | None
    latency_s: float
    output_tokens: int
    # gRPC status code name (or INCOMPLETE); None on success.
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def tpot_s(self) -> float | None:
        """Mean time per output token after the first."""
        if not self.ok or self.ttft_s is None or self.output_tokens < 2:
            return None
        return (self.latency_s - self.ttft_s) / (self.output_tokens - 1)

    def to_dict(self) -> dict:
        out = asdict(self)
        out["tpot_s"] = self.tpot_s
        return out


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 100] (numpy's default method)."""
    if not values:
        raise ValueError("percentile of empty sequence")
    if not 0 <= q <= 100:
        raise ValueError("q must be in [0, 100]")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def distribution_ms(values_s: Sequence[float]) -> dict | None:
    if not values_s:
        return None
    ms = [v * 1000 for v in values_s]
    return {
        "mean": sum(ms) / len(ms),
        "p50": percentile(ms, 50),
        "p90": percentile(ms, 90),
        "p99": percentile(ms, 99),
        "max": max(ms),
    }


def summarize(records: Sequence[RequestRecord], duration_s: float) -> dict:
    if not records:
        raise ValueError("no records to summarize")
    if duration_s <= 0:
        raise ValueError("duration_s must be > 0")
    ok = [r for r in records if r.ok]
    output_tokens = sum(r.output_tokens for r in ok)
    return {
        "requests": len(records),
        "completed": len(ok),
        "failed": len(records) - len(ok),
        "errors": dict(Counter(r.error for r in records if not r.ok)),
        "duration_s": duration_s,
        "request_throughput": len(ok) / duration_s,
        "output_throughput_tok_s": output_tokens / duration_s,
        "total_output_tokens": output_tokens,
        "ttft_ms": distribution_ms([r.ttft_s for r in ok if r.ttft_s is not None]),
        "tpot_ms": distribution_ms([r.tpot_s for r in ok if r.tpot_s is not None]),
        "e2e_latency_ms": distribution_ms([r.latency_s for r in ok]),
    }
