"""Fake generation engine.

Phase 1 does not load a real LM. The engine echoes the prompt as a token
stream so we can prove gRPC streaming, usage stats, and cancellation
without a GPU. Swap this module for a Torch engine later; the servicer
keeps the same GenerateRequest / TokenEvent contract.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

DEFAULT_MAX_TOKENS = 16
HARD_MAX_TOKENS = 256


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
    def __init__(self, token_delay_s: float = 0.0) -> None:
        if token_delay_s < 0:
            raise ValueError("token_delay_s must be >= 0")
        self._token_delay_s = token_delay_s

    async def generate(self, req: GenerateRequest) -> AsyncIterator[TokenEvent]:
        started = time.perf_counter()
        pieces = tokenize(req.prompt)
        prompt_tokens = len(pieces)
        cap = _clamp_max_tokens(req.max_tokens)
        emitted = pieces[:cap]
        ttft_ms = 0.0

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


def _clamp_max_tokens(max_tokens: int) -> int:
    if max_tokens <= 0:
        return DEFAULT_MAX_TOKENS
    return min(max_tokens, HARD_MAX_TOKENS)


def _elapsed_ms(started: float, now: float) -> float:
    return (now - started) * 1000.0
