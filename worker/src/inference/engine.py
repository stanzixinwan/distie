"""Fake generation engine.

Does not load a real LM. Echoes the prompt as a token stream so we can
prove gRPC streaming, cancellation, and C++ BlockPool checkout. Swap this
module for a Torch engine later; the servicer keeps the same
GenerateRequest / TokenEvent contract.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Protocol

DEFAULT_MAX_TOKENS = 16
HARD_MAX_TOKENS = 256
# Fake page size: how many echoed tokens share one C++ block.
TOKENS_PER_BLOCK = 4

_log = logging.getLogger(__name__)


class BlockPoolExhausted(RuntimeError):
    """The C++ pool had too few free blocks for this request."""


class BlockAllocator(Protocol):
    """Python sees integer IDs only; the arena lives in C++ (or a test fake)."""

    def allocate(self, count: int) -> Sequence[int]: ...

    def free(self, block_ids: Sequence[int]) -> None: ...

    def block_view(self, block_id: int) -> memoryview: ...


@dataclass(frozen=True)
class GenerateRequest:
    request_id: str
    model_name: str
    prompt: str
    max_tokens: int


@dataclass(frozen=True)
class TokenEvent:
    token: str
    finished: bool
    prompt_tokens: int = 0
    completion_tokens: int = 0
    time_to_first_token_ms: float = 0.0
    total_latency_ms: float = 0.0


def tokenize(text: str) -> list[str]:
    """Whitespace tokenizer used only by the fake engine."""
    return text.split()


class FakeEngine:
    def __init__(
        self,
        token_delay_s: float = 0.0,
        pool: BlockAllocator | None = None,
    ) -> None:
        if token_delay_s < 0:
            raise ValueError("token_delay_s must be >= 0")
        self._token_delay_s = token_delay_s
        self._pool = pool

    async def generate(self, req: GenerateRequest) -> AsyncIterator[TokenEvent]:
        started = time.perf_counter()
        pieces = tokenize(req.prompt)
        prompt_tokens = len(pieces)
        cap = _clamp_max_tokens(req.max_tokens)
        emitted = pieces[:cap]
        ttft_ms = 0.0
        block_ids: list[int] = []

        try:
            block_ids = self._reserve_blocks(req.request_id, len(emitted))

            if not emitted:
                now = time.perf_counter()
                yield TokenEvent(
                    token="",
                    finished=True,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=0,
                    time_to_first_token_ms=0.0,
                    total_latency_ms=_elapsed_ms(started, now),
                )
                return

            for i, piece in enumerate(emitted):
                if self._token_delay_s:
                    await asyncio.sleep(self._token_delay_s)
                now = time.perf_counter()
                if i == 0:
                    ttft_ms = _elapsed_ms(started, now)
                finished = i == len(emitted) - 1
                yield TokenEvent(
                    token=piece,
                    finished=finished,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=i + 1,
                    time_to_first_token_ms=ttft_ms if finished else 0.0,
                    total_latency_ms=_elapsed_ms(started, now) if finished else 0.0,
                )
        finally:
            self._release_blocks(req.request_id, block_ids)

    def _reserve_blocks(self, request_id: str, token_count: int) -> list[int]:
        if self._pool is None or token_count == 0:
            return []
        needed = (token_count + TOKENS_PER_BLOCK - 1) // TOKENS_PER_BLOCK
        try:
            ids = list(self._pool.allocate(needed))
        except RuntimeError as exc:
            raise BlockPoolExhausted(
                f"out of KV blocks: requested {needed} ({exc})"
            ) from exc
        self._touch_pages(ids, token_count)
        _log.info(
            "kv blocks reserved request_id=%s count=%s ids=%s",
            request_id,
            needed,
            ids,
        )
        return ids

    def _touch_pages(self, ids: list[int], token_count: int) -> None:
        """Write a tiny marker so we actually use the C++ arena, not just IDs."""
        if self._pool is None or not ids:
            return
        view = self._pool.block_view(ids[0])
        view[0] = min(token_count, 255)

    def _release_blocks(self, request_id: str, ids: list[int]) -> None:
        if self._pool is None or not ids:
            return
        try:
            self._pool.free(ids)
        except Exception:
            _log.exception(
                "failed to free kv blocks request_id=%s ids=%s",
                request_id,
                ids,
            )
            return
        _log.info("kv blocks released request_id=%s count=%s", request_id, len(ids))


def _clamp_max_tokens(max_tokens: int) -> int:
    if max_tokens <= 0:
        return DEFAULT_MAX_TOKENS
    return min(max_tokens, HARD_MAX_TOKENS)


def _elapsed_ms(started: float, now: float) -> float:
    return (now - started) * 1000.0
