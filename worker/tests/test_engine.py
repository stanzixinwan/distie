from __future__ import annotations

import sys
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from inference.engine import DEFAULT_MAX_TOKENS, FakeEngine, GenerateRequest, tokenize


class TokenizeTest(unittest.TestCase):
    def test_splits_on_whitespace(self) -> None:
        self.assertEqual(tokenize("hello world from stan"), ["hello", "world", "from", "stan"])

    def test_collapses_extra_space(self) -> None:
        self.assertEqual(tokenize("  a   b  "), ["a", "b"])


class FakeEngineTest(unittest.IsolatedAsyncioTestCase):
    async def test_echoes_prompt_tokens(self) -> None:
        engine = FakeEngine()
        events = [
            event
            async for event in engine.generate(
                GenerateRequest(
                    request_id="r1",
                    model_name="fake",
                    prompt="hello world",
                    max_tokens=8,
                )
            )
        ]
        self.assertEqual([e.token for e in events], ["hello", "world"])
        self.assertFalse(events[0].finished)
        self.assertTrue(events[-1].finished)
        self.assertEqual(events[-1].prompt_tokens, 2)
        self.assertEqual(events[-1].completion_tokens, 2)

    async def test_caps_at_max_tokens(self) -> None:
        engine = FakeEngine()
        events = [
            event
            async for event in engine.generate(
                GenerateRequest(
                    request_id="r1",
                    model_name="fake",
                    prompt="one two three four",
                    max_tokens=2,
                )
            )
        ]
        self.assertEqual([e.token for e in events], ["one", "two"])
        self.assertEqual(events[-1].prompt_tokens, 4)
        self.assertEqual(events[-1].completion_tokens, 2)

    async def test_default_max_tokens_when_zero(self) -> None:
        engine = FakeEngine()
        prompt = " ".join(f"t{i}" for i in range(DEFAULT_MAX_TOKENS + 5))
        events = [
            event
            async for event in engine.generate(
                GenerateRequest(
                    request_id="r1",
                    model_name="fake",
                    prompt=prompt,
                    max_tokens=0,
                )
            )
        ]
        self.assertEqual(len(events), DEFAULT_MAX_TOKENS)


if __name__ == "__main__":
    unittest.main()
