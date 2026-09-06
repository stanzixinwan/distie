from __future__ import annotations

import sys
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from inference.engine import (
    DEFAULT_MAX_TOKENS,
    TOKENS_PER_BLOCK,
    BlockPoolExhausted,
    FakeEngine,
    GenerateRequest,
    tokenize,
)
from recording_pool import RecordingPool


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

    async def test_reserves_and_releases_blocks(self) -> None:
        pool = RecordingPool()
        engine = FakeEngine(pool=pool)
        events = [
            event
            async for event in engine.generate(
                GenerateRequest(
                    request_id="r1",
                    model_name="fake",
                    prompt="a b c d e",
                    max_tokens=8,
                )
            )
        ]
        self.assertEqual(len(events), 5)
        self.assertEqual(len(pool.allocated), 1)
        self.assertEqual(len(pool.allocated[0]), 2)  # 5 tokens / 4 per block
        self.assertEqual(pool.freed, pool.allocated)
        self.assertEqual(pool.num_free, 16)
        self.assertEqual(pool.block_view(pool.allocated[0][0])[0], 5)

    async def test_releases_blocks_on_cancel(self) -> None:
        pool = RecordingPool()
        engine = FakeEngine(pool=pool)
        agen = engine.generate(
            GenerateRequest(
                request_id="r1",
                model_name="fake",
                prompt="one two three",
                max_tokens=8,
            )
        )
        first = await agen.__anext__()
        self.assertEqual(first.token, "one")
        self.assertEqual(len(pool.allocated), 1)
        self.assertEqual(len(pool.freed), 0)
        await agen.aclose()
        self.assertEqual(pool.freed, pool.allocated)
        self.assertEqual(pool.num_free, 16)

    async def test_exhausted_pool_raises(self) -> None:
        pool = RecordingPool(num_blocks=1)
        engine = FakeEngine(pool=pool)
        with self.assertRaises(BlockPoolExhausted):
            async for _ in engine.generate(
                GenerateRequest(
                    request_id="r1",
                    model_name="fake",
                    prompt="a b c d e",
                    max_tokens=8,
                )
            ):
                pass
        self.assertEqual(pool.freed, [])
        self.assertEqual(TOKENS_PER_BLOCK, 4)


if __name__ == "__main__":
    unittest.main()
