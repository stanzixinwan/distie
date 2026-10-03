from __future__ import annotations

import unittest

import torch

from metrics import (
    default_thresholds,
    evaluate,
    first_divergence,
    max_abs_error,
    summarize,
    top1_agreement,
)


def _onehot(ids: list[int], vocab: int = 8) -> torch.Tensor:
    logits = torch.zeros(len(ids), vocab)
    for row, token in enumerate(ids):
        logits[row, token] = 10.0
    return logits


class FirstDivergenceTest(unittest.TestCase):
    def test_identical(self) -> None:
        self.assertIsNone(first_divergence([1, 2, 3], [1, 2, 3]))

    def test_mismatch_index(self) -> None:
        self.assertEqual(first_divergence([1, 2, 3], [1, 5, 3]), 1)

    def test_length_mismatch_is_divergence(self) -> None:
        self.assertEqual(first_divergence([1, 2, 3], [1, 2]), 2)
        self.assertEqual(first_divergence([1, 2], [1, 2, 3]), 2)


class LogitMetricsTest(unittest.TestCase):
    def test_top1_agreement(self) -> None:
        self.assertEqual(top1_agreement(_onehot([1, 2, 3, 4]), _onehot([1, 2, 0, 4])), 0.75)

    def test_max_abs_error(self) -> None:
        ref = torch.zeros(2, 4)
        cand = ref.clone()
        cand[1, 2] = -0.5
        self.assertAlmostEqual(max_abs_error(ref, cand), 0.5)

    def test_shape_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            top1_agreement(torch.zeros(2, 4), torch.zeros(3, 4))
        with self.assertRaises(ValueError):
            max_abs_error(torch.zeros(4), torch.zeros(4))


class EvaluateTest(unittest.TestCase):
    def test_fp32_requires_greedy_match(self) -> None:
        logits = _onehot([1, 2, 3])
        result = evaluate(0, 5, [1, 2, 3], [1, 2, 4], logits, logits, default_thresholds("fp32"))
        self.assertEqual(result.first_divergence, 2)
        self.assertFalse(result.passed)

    def test_fp32_rejects_logit_drift(self) -> None:
        ref = _onehot([1, 2])
        cand = ref.clone()
        cand[0, 5] = 0.01
        result = evaluate(0, 5, [1, 2], [1, 2], ref, cand, default_thresholds("fp32"))
        self.assertFalse(result.passed)

    def test_fp16_tolerates_late_divergence(self) -> None:
        logits = _onehot([1, 2, 3])
        result = evaluate(0, 5, [1, 2, 3], [1, 2, 4], logits, logits, default_thresholds("fp16"))
        self.assertTrue(result.passed)

    def test_fp16_fails_on_low_top1(self) -> None:
        result = evaluate(
            0, 5, [1, 2], [1, 2], _onehot([1, 2]), _onehot([3, 4]), default_thresholds("fp16")
        )
        self.assertFalse(result.passed)

    def test_unknown_dtype(self) -> None:
        with self.assertRaises(ValueError):
            default_thresholds("int8")


class SummarizeTest(unittest.TestCase):
    def test_counts(self) -> None:
        logits = _onehot([1, 2])
        thresholds = default_thresholds("fp32")
        ok = evaluate(0, 3, [1, 2], [1, 2], logits, logits, thresholds)
        bad = evaluate(1, 3, [1, 2], [1, 3], logits, logits, thresholds)
        summary = summarize([ok, bad])
        self.assertEqual(summary["passed"], 1)
        self.assertEqual(summary["greedy_match"], 1)
        self.assertFalse(summary["all_passed"])

    def test_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            summarize([])


if __name__ == "__main__":
    unittest.main()
