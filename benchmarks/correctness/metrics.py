"""Pure comparison logic for HF parity checks. No model loading, no I/O."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Thresholds:
    require_greedy_match: bool
    min_top1_agreement: float
    max_abs_logit_err: float | None


def default_thresholds(dtype: str) -> Thresholds:
    """fp32 must match HF token-for-token; half precision is judged statistically."""
    if dtype == "fp32":
        return Thresholds(require_greedy_match=True, min_top1_agreement=1.0, max_abs_logit_err=1e-3)
    if dtype in {"fp16", "bf16"}:
        return Thresholds(require_greedy_match=False, min_top1_agreement=0.98, max_abs_logit_err=None)
    raise ValueError(f"no default thresholds for dtype {dtype!r}")


@dataclass(frozen=True)
class PromptResult:
    index: int
    prompt_tokens: int
    reference_tokens: int
    candidate_tokens: int
    first_divergence: int | None
    top1_agreement: float
    max_abs_logit_err: float
    passed: bool

    @property
    def greedy_match(self) -> bool:
        return self.first_divergence is None

    def to_dict(self) -> dict:
        out = asdict(self)
        out["greedy_match"] = self.greedy_match
        return out


def first_divergence(reference: Sequence[int], candidate: Sequence[int]) -> int | None:
    """Index of the first differing token, or None if the sequences are identical."""
    for i, (r, c) in enumerate(zip(reference, candidate)):
        if r != c:
            return i
    if len(reference) != len(candidate):
        return min(len(reference), len(candidate))
    return None


def top1_agreement(reference_logits, candidate_logits) -> float:
    _check_logits(reference_logits, candidate_logits)
    if reference_logits.shape[0] == 0:
        return 1.0
    same = reference_logits.argmax(dim=-1) == candidate_logits.argmax(dim=-1)
    return float(same.float().mean().item())


def max_abs_error(reference_logits, candidate_logits) -> float:
    _check_logits(reference_logits, candidate_logits)
    if reference_logits.numel() == 0:
        return 0.0
    diff = (reference_logits.float() - candidate_logits.float()).abs()
    return float(diff.max().item())


def evaluate(
    index: int,
    prompt_tokens: int,
    reference_ids: Sequence[int],
    candidate_ids: Sequence[int],
    reference_logits,
    forced_logits,
    thresholds: Thresholds,
) -> PromptResult:
    """reference_* come from HF generate; forced_logits from the engine fed reference_ids."""
    divergence = first_divergence(reference_ids, candidate_ids)
    top1 = top1_agreement(reference_logits, forced_logits)
    err = max_abs_error(reference_logits, forced_logits)
    passed = top1 >= thresholds.min_top1_agreement
    if thresholds.require_greedy_match:
        passed = passed and divergence is None
    if thresholds.max_abs_logit_err is not None:
        passed = passed and err <= thresholds.max_abs_logit_err
    return PromptResult(
        index=index,
        prompt_tokens=prompt_tokens,
        reference_tokens=len(reference_ids),
        candidate_tokens=len(candidate_ids),
        first_divergence=divergence,
        top1_agreement=top1,
        max_abs_logit_err=err,
        passed=passed,
    )


def summarize(results: Sequence[PromptResult]) -> dict:
    if not results:
        raise ValueError("no results to summarize")
    return {
        "prompts": len(results),
        "passed": sum(r.passed for r in results),
        "greedy_match": sum(r.greedy_match for r in results),
        "min_top1_agreement": min(r.top1_agreement for r in results),
        "max_abs_logit_err": max(r.max_abs_logit_err for r in results),
        "all_passed": all(r.passed for r in results),
    }


def _check_logits(reference_logits, candidate_logits) -> None:
    if reference_logits.dim() != 2 or candidate_logits.dim() != 2:
        raise ValueError("logits must be rank-2 [steps, vocab]")
    if tuple(reference_logits.shape) != tuple(candidate_logits.shape):
        raise ValueError(
            f"logits shape mismatch: reference {tuple(reference_logits.shape)} "
            f"vs candidate {tuple(candidate_logits.shape)}"
        )
