"""Bucketed pure-decode batches against the varlen path they replace."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from inference.kv_cache import KvShape, PagedKvCache
from inference.qwen2 import Qwen2CausalLM
from inference.qwen2.batch_decode import BatchDecodeGraphs, DecodeRow, decode_buckets
from inference.qwen2.paged_attn import BatchSeq, build_batch, flash_enabled


def _config() -> Qwen2Config:
    return Qwen2Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=1024,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=False,
        attention_dropout=0.0,
    )


def _cache(config, page_size: int, num_blocks: int, device="cpu", dtype=torch.float32) -> PagedKvCache:
    return PagedKvCache(
        KvShape.from_hf_config(config),
        num_blocks=num_blocks,
        device=device,
        dtype=dtype,
        page_size=page_size,
    )


class BucketTest(unittest.TestCase):
    def test_bucket_sizes(self) -> None:
        self.assertEqual(decode_buckets(32), (1, 2, 4, 8, 16, 24, 32))
        self.assertEqual(decode_buckets(64), (1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64))
        self.assertEqual(decode_buckets(5), (1, 2, 4, 5))
        self.assertEqual(decode_buckets(1), (1,))
        with self.assertRaises(ValueError):
            decode_buckets(0)

    def test_bucket_for_rounds_up(self) -> None:
        config = _config()
        model = Qwen2CausalLM.from_hf(Qwen2ForCausalLM(config).eval())
        graphs = BatchDecodeGraphs(model, _cache(config, 4, 8), pad_block=7, max_batch=32, use_graphs=False)
        self.assertEqual([graphs.bucket_for(n) for n in (1, 3, 8, 9, 17, 32)], [1, 4, 8, 16, 24, 32])
        for bad in (0, 33):
            with self.assertRaises(ValueError):
                graphs.bucket_for(bad)

    def test_rejects_pad_block_outside_slab(self) -> None:
        config = _config()
        model = Qwen2CausalLM.from_hf(Qwen2ForCausalLM(config).eval())
        with self.assertRaises(ValueError):
            BatchDecodeGraphs(model, _cache(config, 4, 8), pad_block=8, max_batch=4, use_graphs=False)


class _Harness:
    """Two identical slabs: one advanced by the varlen path, one by BatchDecodeGraphs."""

    def __init__(self, device: str, dtype, page_size: int, use_graphs: bool) -> None:
        torch.manual_seed(0)
        self.config = _config()
        hf = Qwen2ForCausalLM(self.config).eval().to(device=device, dtype=dtype)
        self.model = Qwen2CausalLM.from_hf(hf)
        self.device = device
        # Block 11 is the spare page: no table below ever uses it.
        self.ref = _cache(self.config, page_size, 12, device, dtype)
        self.dut = _cache(self.config, page_size, 12, device, dtype)
        self.graphs = BatchDecodeGraphs(self.model, self.dut, pad_block=11, max_batch=8, use_graphs=use_graphs)

    def prefill(self, prompt: list[int], table: tuple[int, ...]) -> None:
        for cache in (self.ref, self.dut):
            meta = build_batch([BatchSeq(table, 0, len(prompt))], cache, self.device)
            self.model.forward_batch(torch.tensor(prompt, device=self.device), cache, meta)

    def step(self, rows: list[DecodeRow]) -> tuple[torch.Tensor, torch.Tensor]:
        meta = build_batch([BatchSeq(tuple(r.block_ids), r.past_len, 1) for r in rows], self.ref, self.device)
        ids = torch.tensor([r.token_id for r in rows], device=self.device)
        want = self.model.forward_batch(ids, self.ref, meta)
        got = self.graphs.run(rows)
        return want.float(), got.float()

    def live_kv_gap(self) -> float:
        """Largest K/V difference outside the spare page."""
        gap = 0.0
        for layer in range(self.config.num_hidden_layers):
            for a, b in zip(self.ref.layer_kv(layer), self.dut.layer_kv(layer)):
                gap = max(gap, (a[:11].float() - b[:11].float()).abs().max().item())
        return gap


def _decode_loop(test: unittest.TestCase, harness: _Harness, tol: float) -> None:
    """Varying batch sizes across steps, sequences joining and leaving, padding every step."""
    page = harness.ref.page_size
    seqs = {
        "a": [list(range(1, 6)), (0, 1, 2)],
        "b": [list(range(10, 13)), (3, 4)],
        "c": [list(range(20, 20 + page)), (5, 6, 7)],  # the first decode opens page 2
    }
    with torch.inference_mode():
        for prompt, table in seqs.values():
            harness.prefill(prompt, table)
        lengths = {name: len(p) for name, (p, _) in seqs.items()}
        schedule = [("a", "b", "c"), ("a", "c"), ("b",), ("a", "b", "c"), ("c", "a")]
        for step, names in enumerate(schedule):
            rows = [DecodeRow(30 + step, lengths[n], seqs[n][1]) for n in names]
            want, got = harness.step(rows)
            test.assertEqual(tuple(got.shape), (len(rows), harness.config.vocab_size))
            test.assertLess((want - got).abs().max().item(), tol, f"step {step} {names}")
            for n in names:
                lengths[n] += 1
    test.assertLess(harness.live_kv_gap(), tol)


class EagerBucketTest(unittest.TestCase):
    def test_padded_buckets_match_varlen(self) -> None:
        _decode_loop(self, _Harness("cpu", torch.float32, page_size=4, use_graphs=False), tol=1e-4)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA graphs need a GPU")
class CudaGraphTest(unittest.TestCase):
    def setUp(self) -> None:
        self._attn = os.environ.get("DISTIE_ATTN")
        os.environ["DISTIE_ATTN"] = "flash"
        if not flash_enabled(torch.float16):
            self.skipTest("flash_attn not installed")

    def tearDown(self) -> None:
        if self._attn is None:
            os.environ.pop("DISTIE_ATTN", None)
        else:
            os.environ["DISTIE_ATTN"] = self._attn

    def test_graph_replay_matches_varlen(self) -> None:
        harness = _Harness("cuda", torch.float16, page_size=256, use_graphs=True)
        _decode_loop(self, harness, tol=2e-2)
        # Each bucket is captured once and then replayed.
        self.assertEqual(sorted(harness.graphs._buckets), [1, 2, 4])


if __name__ == "__main__":
    unittest.main()
