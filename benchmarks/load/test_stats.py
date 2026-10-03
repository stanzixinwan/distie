from __future__ import annotations

import unittest

from stats import RequestRecord, distribution_ms, percentile, summarize


def _rec(i: int, ttft: float | None = 0.1, latency: float = 1.1, tokens: int = 11, error: str | None = None):
    return RequestRecord(f"r{i}", 10, 32, float(i), ttft, latency, tokens, error)


class PercentileTest(unittest.TestCase):
    def test_matches_numpy_linear(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0]
        self.assertEqual(percentile(values, 0), 1.0)
        self.assertEqual(percentile(values, 100), 4.0)
        self.assertAlmostEqual(percentile(values, 50), 2.5)
        self.assertAlmostEqual(percentile(values, 90), 3.7)

    def test_single_value(self) -> None:
        self.assertEqual(percentile([7.0], 99), 7.0)

    def test_rejects_bad_input(self) -> None:
        with self.assertRaises(ValueError):
            percentile([], 50)
        with self.assertRaises(ValueError):
            percentile([1.0], 101)


class RecordTest(unittest.TestCase):
    def test_tpot_excludes_first_token(self) -> None:
        self.assertAlmostEqual(_rec(0).tpot_s, 0.1)

    def test_tpot_undefined_for_single_token_or_error(self) -> None:
        self.assertIsNone(_rec(0, tokens=1).tpot_s)
        self.assertIsNone(_rec(0, error="UNAVAILABLE").tpot_s)
        self.assertIsNone(_rec(0, ttft=None).tpot_s)


class SummarizeTest(unittest.TestCase):
    def test_aggregates_only_successful_requests(self) -> None:
        records = [_rec(0), _rec(1), _rec(2, ttft=None, latency=0.01, tokens=0, error="RESOURCE_EXHAUSTED")]
        summary = summarize(records, duration_s=2.0)
        self.assertEqual(summary["completed"], 2)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["errors"], {"RESOURCE_EXHAUSTED": 1})
        self.assertEqual(summary["total_output_tokens"], 22)
        self.assertAlmostEqual(summary["output_throughput_tok_s"], 11.0)
        self.assertAlmostEqual(summary["request_throughput"], 1.0)
        self.assertAlmostEqual(summary["ttft_ms"]["p50"], 100.0)
        self.assertAlmostEqual(summary["e2e_latency_ms"]["max"], 1100.0)

    def test_all_failed_has_no_distributions(self) -> None:
        summary = summarize([_rec(0, error="UNAVAILABLE")], duration_s=1.0)
        self.assertEqual(summary["completed"], 0)
        self.assertIsNone(summary["ttft_ms"])

    def test_rejects_bad_input(self) -> None:
        with self.assertRaises(ValueError):
            summarize([], 1.0)
        with self.assertRaises(ValueError):
            summarize([_rec(0)], 0.0)

    def test_distribution_empty(self) -> None:
        self.assertIsNone(distribution_ms([]))


if __name__ == "__main__":
    unittest.main()
