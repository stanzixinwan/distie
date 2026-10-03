"""Benchmark request sets. Pure data preparation; no networking."""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class Tokenizer(Protocol):
    def encode(self, text: str, add_special_tokens: bool = ...) -> Sequence[int]: ...


@dataclass(frozen=True)
class BenchRequest:
    prompt: str
    max_tokens: int
    # Token count under the benchmark tokenizer; -1 when unknown (synthetic).
    prompt_len: int


_HUMAN = {"human", "user"}
_ASSISTANT = {"gpt", "assistant", "chatgpt"}


def load_sharegpt(
    path: Path,
    num_requests: int,
    tokenizer: Tokenizer,
    seed: int = 0,
    min_len: int = 4,
    max_prompt_len: int = 1024,
    max_total_len: int = 2048,
    max_output_len: int = 256,
) -> list[BenchRequest]:
    """Workload A: first human turn as prompt, first reply length as max_tokens.

    Filters follow vLLM's benchmark_serving so numbers stay comparable.
    max_output_len should not exceed the worker's hard cap (HARD_MAX_TOKENS).
    """
    if num_requests < 1:
        raise ValueError("num_requests must be >= 1")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list of conversations")

    pairs = [_first_pair(item) for item in data]
    pairs = [p for p in pairs if p is not None]
    random.Random(seed).shuffle(pairs)

    out: list[BenchRequest] = []
    for prompt, reply in pairs:
        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
        output_len = len(tokenizer.encode(reply, add_special_tokens=False))
        if prompt_len < min_len or output_len < min_len:
            continue
        if prompt_len > max_prompt_len or prompt_len + output_len > max_total_len:
            continue
        out.append(BenchRequest(prompt, min(output_len, max_output_len), prompt_len))
        if len(out) == num_requests:
            return out
    raise ValueError(f"{path}: only {len(out)} conversations pass filters, need {num_requests}")


def synthetic(
    num_requests: int,
    seed: int = 0,
    prompt_words: tuple[int, int] = (16, 256),
    output_tokens: tuple[int, int] = (16, 128),
) -> list[BenchRequest]:
    """Random lengths, no dataset or tokenizer. For smoke tests against FakeEngine."""
    if num_requests < 1:
        raise ValueError("num_requests must be >= 1")
    _check_range("prompt_words", prompt_words)
    _check_range("output_tokens", output_tokens)
    rng = random.Random(seed)
    out = []
    for _ in range(num_requests):
        words = rng.randint(*prompt_words)
        prompt = " ".join(f"w{rng.randrange(10_000)}" for _ in range(words))
        out.append(BenchRequest(prompt, rng.randint(*output_tokens), -1))
    return out


def _first_pair(item) -> tuple[str, str] | None:
    turns = item.get("conversations") if isinstance(item, dict) else None
    if not isinstance(turns, list) or len(turns) < 2:
        return None
    first, second = turns[0], turns[1]
    if not (isinstance(first, dict) and isinstance(second, dict)):
        return None
    if first.get("from") not in _HUMAN or second.get("from") not in _ASSISTANT:
        return None
    prompt, reply = first.get("value"), second.get("value")
    if not isinstance(prompt, str) or not isinstance(reply, str) or not prompt.strip():
        return None
    return prompt, reply


def _check_range(name: str, bounds: tuple[int, int]) -> None:
    lo, hi = bounds
    if lo < 1 or hi < lo:
        raise ValueError(f"{name} must satisfy 1 <= low <= high, got {bounds}")
