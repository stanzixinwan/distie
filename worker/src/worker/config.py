from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    listen_addr: str
    worker_id: str
    shutdown_grace_s: float
    token_delay_s: float
    # Tests leave this False and inject a fake pool. load() turns it on.
    block_pool_enabled: bool = False
    num_blocks: int = 1024
    block_size_bytes: int = 256


def load() -> Config:
    grace = _env_float("WORKER_SHUTDOWN_GRACE_SEC", 5.0)
    if grace <= 0:
        raise ValueError("WORKER_SHUTDOWN_GRACE_SEC must be > 0")

    delay_ms = _env_float("FAKE_TOKEN_DELAY_MS", 0.0)
    if delay_ms < 0:
        raise ValueError("FAKE_TOKEN_DELAY_MS must be >= 0")

    num_blocks = _env_int("WORKER_NUM_BLOCKS", 1024)
    if num_blocks < 1:
        raise ValueError("WORKER_NUM_BLOCKS must be >= 1")

    block_size = _env_int("WORKER_BLOCK_SIZE_BYTES", 256)
    if block_size < 1:
        raise ValueError("WORKER_BLOCK_SIZE_BYTES must be >= 1")

    return Config(
        listen_addr=os.getenv("WORKER_LISTEN_ADDR", ":50052"),
        worker_id=os.getenv("WORKER_ID", "worker-1"),
        shutdown_grace_s=grace,
        token_delay_s=delay_ms / 1000.0,
        block_pool_enabled=_env_bool("WORKER_BLOCK_POOL", True),
        num_blocks=num_blocks,
        block_size_bytes=block_size,
    )


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{key} must be a number") from exc


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{key} must be an integer") from exc


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}
