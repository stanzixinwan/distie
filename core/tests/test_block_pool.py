from __future__ import annotations

import sys
import unittest
from pathlib import Path

_BUILD = Path(__file__).resolve().parents[1] / "build"
if str(_BUILD) not in sys.path:
    sys.path.insert(0, str(_BUILD))

import distie_core  # noqa: E402


class BlockPoolTest(unittest.TestCase):
    def test_allocate_and_stats(self) -> None:
        pool = distie_core.BlockPool(4, 8)
        ids = pool.allocate(3)
        self.assertEqual(list(ids), [0, 1, 2])
        self.assertEqual(pool.num_used, 3)
        self.assertEqual(pool.num_free, 1)
        self.assertAlmostEqual(pool.utilization, 0.75)

    def test_out_of_blocks_is_all_or_nothing(self) -> None:
        pool = distie_core.BlockPool(4, 8)
        pool.allocate(3)
        with self.assertRaises(RuntimeError):
            pool.allocate(2)
        self.assertEqual(pool.num_free, 1)

    def test_double_free(self) -> None:
        pool = distie_core.BlockPool(2, 8)
        ids = pool.allocate(1)
        pool.free(ids)
        with self.assertRaises(ValueError):
            pool.free(ids)

    def test_block_view_roundtrip(self) -> None:
        pool = distie_core.BlockPool(2, 4)
        ids = pool.allocate(1)
        view = pool.block_view(ids[0])
        view[0] = 42
        view[3] = 99
        again = pool.block_view(ids[0])
        self.assertEqual(again[0], 42)
        self.assertEqual(again[3], 99)
        pool.free(ids)
        with self.assertRaises(ValueError):
            pool.block_view(ids[0])


if __name__ == "__main__":
    unittest.main()
