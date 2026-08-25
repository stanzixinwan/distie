from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    listen_addr: str
    worker_id: str
    shutdown_grace_s: float
    token_delay_s: float


def load() -> Config:
    grace = _env_float("OMNISERVE_WORKER_SHUTDOWN_GRACE_SEC", 5.0)
    if grace <= 0:
        raise ValueError("OMNISERVE_WORKER_SHUTDOWN_GRACE_SEC must be > 0")

    delay_ms = _env_float("OMNISERVE_FAKE_TOKEN_DELAY_MS", 0.0)
    if delay_ms < 0:
        raise ValueError("OMNISERVE_FAKE_TOKEN_DELAY_MS must be >= 0")

    return Config(
        listen_addr=os.getenv("OMNISERVE_WORKER_LISTEN_ADDR", ":50052"),
        worker_id=os.getenv("OMNISERVE_WORKER_ID", "worker-1"),
        shutdown_grace_s=grace,
        token_delay_s=delay_ms / 1000.0,
    )


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{key} must be a number") from exc
