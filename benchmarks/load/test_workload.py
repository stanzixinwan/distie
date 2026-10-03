from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from workload import load_sharegpt, synthetic


class WordTokenizer:
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return list(range(len(text.split())))


def _conv(prompt: str, reply: str, first: str = "human", second: str = "gpt") -> dict:
    return {"conversations": [{"from": first, "value": prompt}, {"from": second, "value": reply}]}


class ShareGptTest(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.path = Path(self._dir.name) / "sharegpt.json"

    def tearDown(self) -> None:
        self._dir.cleanup()

    def _write(self, items: list) -> None:
        self.path.write_text(json.dumps(items), encoding="utf-8")

    def test_filters_and_caps_output(self) -> None:
        long_reply = " ".join(["x"] * 400)
        self._write(
            [
                _conv("a b c d e", long_reply),
                _conv("too short", "a b c d e"),
                _conv("a b c d e", "short"),
                _conv("a b c d e", "a b c d e", first="gpt", second="human"),
                {"conversations": [{"from": "human", "value": "only one turn"}]},
                "not a dict",
            ]
        )
        reqs = load_sharegpt(self.path, 1, WordTokenizer(), max_output_len=256)
        self.assertEqual(len(reqs), 1)
        self.assertEqual(reqs[0].prompt, "a b c d e")
        self.assertEqual(reqs[0].prompt_len, 5)
        self.assertEqual(reqs[0].max_tokens, 256)

    def test_rejects_overlong_prompt(self) -> None:
        self._write([_conv(" ".join(["p"] * 50), "a b c d e")])
        with self.assertRaises(ValueError):
            load_sharegpt(self.path, 1, WordTokenizer(), max_prompt_len=10)

    def test_not_enough_conversations(self) -> None:
        self._write([_conv("a b c d e", "f g h i j")])
        with self.assertRaises(ValueError):
            load_sharegpt(self.path, 2, WordTokenizer())

    def test_seed_is_deterministic(self) -> None:
        self._write([_conv(f"q{i} a b c d", "a b c d e") for i in range(20)])
        first = load_sharegpt(self.path, 5, WordTokenizer(), seed=3)
        again = load_sharegpt(self.path, 5, WordTokenizer(), seed=3)
        self.assertEqual(first, again)


class SyntheticTest(unittest.TestCase):
    def test_lengths_within_bounds(self) -> None:
        reqs = synthetic(50, seed=1, prompt_words=(4, 8), output_tokens=(2, 3))
        self.assertEqual(len(reqs), 50)
        for r in reqs:
            self.assertTrue(4 <= len(r.prompt.split()) <= 8)
            self.assertIn(r.max_tokens, (2, 3))
            self.assertEqual(r.prompt_len, -1)

    def test_seed_is_deterministic(self) -> None:
        self.assertEqual(synthetic(5, seed=7), synthetic(5, seed=7))

    def test_rejects_bad_ranges(self) -> None:
        with self.assertRaises(ValueError):
            synthetic(0)
        with self.assertRaises(ValueError):
            synthetic(1, prompt_words=(5, 4))


if __name__ == "__main__":
    unittest.main()
