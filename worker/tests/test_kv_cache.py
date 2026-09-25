from __future__ import annotations

import sys
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch

from inference.kv_cache import KvShape, PagedKvCache, pages_needed


def _shape() -> KvShape:
    return KvShape(num_layers=2, num_kv_heads=2, head_dim=4)


def _cache(page_size: int = 4, num_blocks: int = 8) -> PagedKvCache:
    return PagedKvCache(
        _shape(),
        num_blocks=num_blocks,
        device="cpu",
        dtype=torch.float32,
        page_size=page_size,
    )


def _kv(seq_len: int, fill: float) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    shape = _shape()
    keys = [
        torch.full((1, shape.num_kv_heads, seq_len, shape.head_dim), fill + layer)
        for layer in range(shape.num_layers)
    ]
    values = [
        torch.full((1, shape.num_kv_heads, seq_len, shape.head_dim), fill + 10 + layer)
        for layer in range(shape.num_layers)
    ]
    return keys, values


class PagesNeededTest(unittest.TestCase):
    def test_ceil_div(self) -> None:
        self.assertEqual(pages_needed(0, 16), 0)
        self.assertEqual(pages_needed(16, 16), 1)
        self.assertEqual(pages_needed(17, 16), 2)

    def test_rejects_bad_page_size(self) -> None:
        with self.assertRaises(ValueError):
            pages_needed(1, 0)


class KvShapeTest(unittest.TestCase):
    def test_from_hf_config_gqa(self) -> None:
        config = type(
            "Cfg",
            (),
            {
                "num_hidden_layers": 28,
                "num_attention_heads": 12,
                "num_key_value_heads": 2,
                "hidden_size": 1536,
            },
        )()
        shape = KvShape.from_hf_config(config)
        self.assertEqual(shape.num_layers, 28)
        self.assertEqual(shape.num_kv_heads, 2)
        self.assertEqual(shape.head_dim, 128)


class PagedKvCacheTest(unittest.TestCase):
    def test_scatter_gather_roundtrip(self) -> None:
        cache = _cache(page_size=4)
        keys, values = _kv(seq_len=6, fill=1.0)
        # Non-monotonic IDs: paging, not a contiguous arena slice.
        table = [3, 0]
        cache.scatter(table, 0, keys, values)
        got_k, got_v = cache.gather(table, 6)
        for layer in range(2):
            torch.testing.assert_close(got_k[layer], keys[layer])
            torch.testing.assert_close(got_v[layer], values[layer])

    def test_decode_appends_without_clobbering_prompt(self) -> None:
        cache = _cache(page_size=4)
        table = [5, 1]
        prompt_k, prompt_v = _kv(seq_len=4, fill=2.0)
        cache.scatter(table, 0, prompt_k, prompt_v)

        full_k, full_v = _kv(seq_len=5, fill=2.0)
        # Suffix at pos 4 is a different fill so we can see it was written.
        full_k[0][:, :, 4, :] = 99.0
        full_v[0][:, :, 4, :] = 77.0
        cache.scatter(table, 4, full_k, full_v)

        got_k, got_v = cache.gather(table, 5)
        torch.testing.assert_close(got_k[0][:, :, :4, :], prompt_k[0])
        self.assertEqual(got_k[0][0, 0, 4, 0].item(), 99.0)
        self.assertEqual(got_v[0][0, 0, 4, 0].item(), 77.0)

    def test_two_sequences_do_not_share_slots(self) -> None:
        cache = _cache(page_size=4)
        a_k, a_v = _kv(seq_len=3, fill=1.0)
        b_k, b_v = _kv(seq_len=5, fill=4.0)
        cache.scatter([2], 0, a_k, a_v)
        cache.scatter([7, 1], 0, b_k, b_v)
        got_a_k, _ = cache.gather([2], 3)
        got_b_k, _ = cache.gather([7, 1], 5)
        torch.testing.assert_close(got_a_k[0], a_k[0])
        torch.testing.assert_close(got_b_k[0], b_k[0])

    def test_short_block_table_raises(self) -> None:
        cache = _cache(page_size=4)
        keys, values = _kv(seq_len=5, fill=1.0)
        with self.assertRaises(ValueError):
            cache.scatter([0], 0, keys, values)

    def test_rejects_batched_kv(self) -> None:
        cache = _cache()
        keys = [torch.zeros(2, 2, 4, 4)]
        values = [torch.zeros(2, 2, 4, 4)]
        with self.assertRaises(ValueError):
            cache.scatter([0], 0, keys, values)


if __name__ == "__main__":
    unittest.main()
