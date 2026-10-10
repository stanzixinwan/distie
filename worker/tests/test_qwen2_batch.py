"""Flattened varlen batches against per-sequence paged forwards."""

from __future__ import annotations

import os
import sys
import types
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from inference.kv_cache import KvShape, PagedKvCache
from inference.qwen2 import Qwen2CausalLM
from inference.qwen2.paged_attn import BatchSeq, build_batch


def _config() -> Qwen2Config:
    return Qwen2Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=False,
        attention_dropout=0.0,
    )


def _cache(config: Qwen2Config, page_size: int, num_blocks: int = 8, dtype=torch.float32) -> PagedKvCache:
    return PagedKvCache(
        KvShape(config.num_hidden_layers, config.num_key_value_heads, config.hidden_size // config.num_attention_heads),
        num_blocks=num_blocks,
        device="cpu",
        dtype=dtype,
        page_size=page_size,
    )


class BuildBatchTest(unittest.TestCase):
    def test_slots_positions_and_boundaries(self) -> None:
        cache = _cache(_config(), page_size=4)
        meta = build_batch([BatchSeq((3,), 0, 2), BatchSeq((1, 5), 5, 1)], cache, "cpu")
        self.assertEqual(meta.positions.tolist(), [0, 1, 5])
        self.assertEqual(meta.slot_mapping.tolist(), [12, 13, 21])
        self.assertEqual(meta.cu_seqlens_q.tolist(), [0, 2, 3])
        self.assertEqual(meta.cu_seqlens_k.tolist(), [0, 2, 8])
        self.assertEqual(meta.block_table.tolist(), [[3, 0], [1, 5]])
        self.assertEqual(meta.last_index.tolist(), [1, 2])
        self.assertEqual((meta.max_seqlen_q, meta.max_seqlen_k), (2, 6))

    def test_rejects_short_table(self) -> None:
        cache = _cache(_config(), page_size=4)
        with self.assertRaises(ValueError):
            build_batch([BatchSeq((0,), 4, 1)], cache, "cpu")

    def test_rejects_empty(self) -> None:
        with self.assertRaises(ValueError):
            build_batch([], _cache(_config(), page_size=4), "cpu")


class ForwardBatchTest(unittest.TestCase):
    def test_mixed_batch_matches_single_sequences(self) -> None:
        torch.manual_seed(0)
        config = _config()
        model = Qwen2CausalLM.from_hf(Qwen2ForCausalLM(config).eval())
        prompts = [torch.randint(0, config.vocab_size, (n,)) for n in (5, 6, 3)]
        tables = [(0, 1), (2, 3), (4, 5)]
        nxt = [7, 9]

        single = _cache(config, page_size=4)
        batched = _cache(config, page_size=4)
        with torch.inference_mode():
            # Sequences 1 and 2 already have their prompt stored and decode one token.
            # Sequence 0 prefills in the same step.
            want = [model.forward_paged(prompts[0].unsqueeze(0), single, list(tables[0]), 0)[0, -1]]
            for i, tok in zip((1, 2), nxt):
                model.forward_paged(prompts[i].unsqueeze(0), single, list(tables[i]), 0)
                model.forward_paged(prompts[i].unsqueeze(0), batched, list(tables[i]), 0)
                step = torch.tensor([[tok]])
                want.append(model.forward_paged(step, single, list(tables[i]), len(prompts[i]))[0, -1])

            meta = build_batch(
                [
                    BatchSeq(tables[0], 0, len(prompts[0])),
                    BatchSeq(tables[1], len(prompts[1]), 1),
                    BatchSeq(tables[2], len(prompts[2]), 1),
                ],
                batched,
                "cpu",
            )
            ids = torch.cat([prompts[0], torch.tensor(nxt)])
            got = model.forward_batch(ids, batched, meta)
        self.assertEqual(tuple(got.shape), (3, config.vocab_size))
        for row, ref in enumerate(want):
            self.assertLess((got[row] - ref).abs().max().item(), 1e-4)
        # The batched write must leave the same K/V in the slab as the single path.
        k_single, _ = single.layer_kv(1)
        k_batched, _ = batched.layer_kv(1)
        self.assertLess((k_single - k_batched).abs().max().item(), 1e-5)

    def test_flash_varlen_call_contract(self) -> None:
        calls: dict = {}
        fake = types.ModuleType("flash_attn")

        def flash_attn_varlen_func(q, k, v, cu_q, cu_k, max_q, max_k, causal=False, block_table=None, **kwargs):
            calls["q"] = tuple(q.shape)
            calls["cache"] = tuple(k.shape)
            calls["cu_q"] = cu_q.tolist()
            calls["cu_k"] = cu_k.tolist()
            calls["max"] = (max_q, max_k)
            calls["table"] = block_table.tolist()
            calls["causal"] = causal
            return torch.zeros_like(q)

        def flash_attn_with_kvcache(*args, **kwargs):
            raise AssertionError("batched path must use the varlen kernel")

        fake.flash_attn_varlen_func = flash_attn_varlen_func
        fake.flash_attn_with_kvcache = flash_attn_with_kvcache
        sys.modules["flash_attn"] = fake
        os.environ["DISTIE_ATTN"] = "flash"
        try:
            config = _config()
            model = Qwen2CausalLM.from_hf(Qwen2ForCausalLM(config).eval()).half()
            cache = _cache(config, page_size=256, num_blocks=3, dtype=torch.float16)
            meta = build_batch([BatchSeq((2,), 0, 3), BatchSeq((0, 1), 300, 1)], cache, "cpu")
            with torch.inference_mode():
                logits = model.forward_batch(torch.tensor([1, 2, 3, 4]), cache, meta)
            self.assertEqual(tuple(logits.shape), (2, config.vocab_size))
            self.assertEqual(calls["q"], (4, 4, 8))
            self.assertEqual(calls["cache"], (3, 256, 2, 8))
            self.assertEqual(calls["cu_q"], [0, 3, 4])
            self.assertEqual(calls["cu_k"], [0, 3, 304])
            self.assertEqual(calls["max"], (3, 301))
            self.assertEqual(calls["table"], [[2, 0], [0, 1]])
            self.assertTrue(calls["causal"])
        finally:
            os.environ.pop("DISTIE_ATTN", None)
            sys.modules.pop("flash_attn", None)


if __name__ == "__main__":
    unittest.main()
