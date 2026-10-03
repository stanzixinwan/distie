from __future__ import annotations

import sys
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from inference.engine import FakeEngine, GenerateRequest
from inference.native import load_block_pool


class NativeBlockPoolTest(unittest.IsolatedAsyncioTestCase):
    async def test_generate_uses_cpp_pool(self) -> None:
        try:
            pool = load_block_pool(8)
        except ImportError:
            self.skipTest("distie_core not built")

        engine = FakeEngine(pool=pool)
        self.assertEqual(pool.num_used, 0)
        events = [
            event
            async for event in engine.generate(
                GenerateRequest(
                    request_id="r1",
                    model_name="fake",
                    prompt="hello world from distie",
                    max_tokens=8,
                )
            )
        ]
        self.assertEqual([e.token for e in events], ["hello", "world", "from", "distie"])
        self.assertEqual(pool.num_used, 0)
        self.assertEqual(pool.num_free, 8)


if __name__ == "__main__":
    unittest.main()
