"""Where each scheduler step spends its time, logged every N steps.

Enabled with WORKER_STEP_STATS=N (N steps per log line; unset or 0 = off).
Each step is split into four parts, measured by BatchLoop:

  sched  Scheduler.schedule() on the event loop
  gpu    the forward and sampling, timed inside the GPU thread; ends with
         the host sync in sampling, so it includes GPU execution
  wait   time the event loop took to resume after the GPU thread finished:
         other coroutines (detokenize, gRPC writes) holding the loop or the GIL
  emit   Scheduler.complete() and routing tokens to request queues

gpu is further split when the runner reports it: kernel is the forward's
GPU time between two CUDA events, and host = gpu - kernel is what the GPU
thread spends outside it (staging inputs, launching, sampling, syncing).

Steps are grouped by kind: "decode" (every row decodes one token, the
CUDA-graph path) and "prefill" (at least one row prefills, the eager path).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

_log = logging.getLogger(__name__)

PARTS = ("sched", "gpu", "wait", "emit")


@dataclass
class _Group:
    steps: int = 0
    rows: int = 0
    tokens: int = 0
    ms: dict[str, float] = field(default_factory=lambda: dict.fromkeys(PARTS, 0.0))
    kernel_ms: float = 0.0
    kernel_steps: int = 0
    kernel_gpu_ms: float = 0.0  # gpu time of the steps that reported a kernel time

    def add(self, rows: int, tokens: int, parts: dict[str, float], kernel: float | None) -> None:
        self.steps += 1
        self.rows += rows
        self.tokens += tokens
        for name in PARTS:
            self.ms[name] += parts[name]
        if kernel is not None:
            self.kernel_ms += kernel
            self.kernel_steps += 1
            self.kernel_gpu_ms += parts["gpu"]

    def total_ms(self) -> float:
        return sum(self.ms.values())


class StepStats:
    def __init__(self, every: int) -> None:
        if every < 1:
            raise ValueError("step stats interval must be >= 1")
        self.every = every
        self._groups: dict[str, _Group] = {}
        self._steps = 0

    def record(self, kind: str, rows: int, tokens: int, kernel: float | None = None, **parts_ms: float) -> None:
        missing = set(PARTS) - set(parts_ms)
        if missing:
            raise ValueError(f"missing step parts: {sorted(missing)}")
        self._groups.setdefault(kind, _Group()).add(rows, tokens, parts_ms, kernel)
        self._steps += 1
        if self._steps >= self.every:
            self.flush()

    def flush(self) -> str | None:
        """Log and reset the current window. Returns the line, or None if empty."""
        if self._steps == 0:
            return None
        line = self.format()
        _log.info("%s", line)
        self._groups.clear()
        self._steps = 0
        return line

    def format(self) -> str:
        wall = sum(g.total_ms() for g in self._groups.values()) or 1.0
        pieces = [f"step stats steps={self._steps}"]
        for kind in sorted(self._groups):
            g = self._groups[kind]
            n = g.steps
            avg = " ".join(f"{name}={g.ms[name] / n:.2f}" for name in PARTS)
            if g.kernel_steps:
                k = g.kernel_ms / g.kernel_steps
                host = (g.kernel_gpu_ms - g.kernel_ms) / g.kernel_steps
                avg += f" (kernel={k:.2f} host={host:.2f})"
            pieces.append(
                f"{kind} n={n} rows={g.rows / n:.1f} tok={g.tokens / n:.0f} "
                f"ms/step[{avg}] share={100 * g.total_ms() / wall:.0f}%"
            )
        totals = {name: sum(g.ms[name] for g in self._groups.values()) for name in PARTS}
        pieces.append("wall " + " ".join(f"{name}={100 * totals[name] / wall:.0f}%" for name in PARTS))
        return " | ".join(pieces)
