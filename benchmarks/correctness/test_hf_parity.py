"""End-to-end parity on a tiny random Qwen2: no download, CPU, fp32."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

import hf_parity  # noqa: F401  (puts worker/src on sys.path)
from hf_parity import run
from inference.kv_cache import KvShape, PagedKvCache
from inference.torch_engine import TorchEngine
from metrics import default_thresholds

_EOS = 2
_VOCAB = 64


class _Pool:
    def __init__(self, num_blocks: int) -> None:
        self.num_blocks = num_blocks
        self._free = list(range(num_blocks - 1, -1, -1))

    def allocate(self, count: int) -> list[int]:
        if count > len(self._free):
            raise RuntimeError("out of blocks")
        return [self._free.pop() for _ in range(count)]

    def free(self, block_ids: list[int]) -> None:
        self._free.extend(block_ids)


class _ByteTokenizer:
    eos_token_id = _EOS
    pad_token_id = _EOS
    chat_template = None

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [3 + (b % (_VOCAB - 3)) for b in text.encode("utf-8")]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return " ".join(str(i) for i in ids)


def _tiny_engine(num_blocks: int = 64) -> TorchEngine:
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=_VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        eos_token_id=_EOS,
        # Default 0.02 gives near-uniform logits that hide KV corruption.
        initializer_range=0.5,
    )
    model = Qwen2ForCausalLM(config).eval()
    pool = _Pool(num_blocks)
    kv = PagedKvCache(
        KvShape.from_hf_config(config),
        num_blocks=num_blocks,
        device="cpu",
        dtype=torch.float32,
        page_size=4,
    )
    return TorchEngine(model, _ByteTokenizer(), "cpu", pool=pool, kv_cache=kv)


class TinyQwenParityTest(unittest.TestCase):
    def test_fp32_matches_hf_generate(self) -> None:
        engine = _tiny_engine()
        prompts = ["hi", "a prompt that spans several kv pages"]
        results = run(engine, prompts, max_new_tokens=12, thresholds=default_thresholds("fp32"))
        for r in results:
            self.assertTrue(r.passed, r)
            self.assertIsNone(r.first_divergence)

    def test_ignores_checkpoint_sampling_defaults(self) -> None:
        engine = _tiny_engine()
        engine.model.generation_config.do_sample = True
        engine.model.generation_config.repetition_penalty = 3.0
        engine.model.generation_config.top_k = 2
        results = run(engine, ["hi there hi there"], max_new_tokens=12, thresholds=default_thresholds("fp32"))
        self.assertTrue(results[0].passed, results[0])

    def test_rejects_non_greedy_reference(self) -> None:
        engine = _tiny_engine()
        engine.model.generation_config.no_repeat_ngram_size = 1
        with self.assertRaises(RuntimeError):
            run(engine, ["hi"], max_new_tokens=12, thresholds=default_thresholds("fp32"))

    def test_detects_corrupted_kv(self) -> None:
        original = PagedKvCache.gather

        def corrupt_newest_key(self, block_ids, seq_len):
            out = original(self, block_ids, seq_len)
            if out is None:
                return out
            keys, values = out
            keys[0][:, :, -1, :] = 0
            return keys, values

        engine = _tiny_engine()
        with patch.object(PagedKvCache, "gather", corrupt_newest_key):
            results = run(engine, ["hi", "a prompt that spans several kv pages"], 12, default_thresholds("fp32"))
        self.assertTrue(all(not r.passed for r in results))

    def test_requires_paged_engine(self) -> None:
        paged = _tiny_engine()
        engine = TorchEngine(paged.model, paged.tokenizer, "cpu")
        with self.assertRaises(RuntimeError):
            run(engine, ["hi"], max_new_tokens=2, thresholds=default_thresholds("fp32"))


if __name__ == "__main__":
    unittest.main()
