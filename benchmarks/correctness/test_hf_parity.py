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


def _pair(num_blocks: int = 64):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from inference.qwen2 import Qwen2CausalLM

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
    config._attn_implementation = "sdpa"
    reference = Qwen2ForCausalLM(config).eval()
    model = Qwen2CausalLM.from_hf(reference)
    pool = _Pool(num_blocks)
    kv = PagedKvCache(
        KvShape.from_hf_config(config),
        num_blocks=num_blocks,
        device="cpu",
        dtype=torch.float32,
        page_size=4,
    )
    engine = TorchEngine(model, _ByteTokenizer(), "cpu", pool=pool, kv_cache=kv)
    return reference, engine


class TinyQwenParityTest(unittest.TestCase):
    def test_fp32_matches_hf_generate(self) -> None:
        reference, engine = _pair()
        prompts = ["hi", "a prompt that spans several kv pages"]
        results = run(engine, prompts, max_new_tokens=12, thresholds=default_thresholds("fp32"), reference=reference)
        for r in results:
            self.assertTrue(r.passed, r)
            self.assertIsNone(r.first_divergence)

    def test_ignores_checkpoint_sampling_defaults(self) -> None:
        reference, engine = _pair()
        reference.generation_config.do_sample = True
        reference.generation_config.repetition_penalty = 3.0
        reference.generation_config.top_k = 2
        results = run(
            engine,
            ["hi there hi there"],
            max_new_tokens=12,
            thresholds=default_thresholds("fp32"),
            reference=reference,
        )
        self.assertTrue(results[0].passed, results[0])

    def test_rejects_non_greedy_reference(self) -> None:
        reference, engine = _pair()
        reference.generation_config.no_repeat_ngram_size = 1
        with self.assertRaises(RuntimeError):
            run(engine, ["hi"], max_new_tokens=12, thresholds=default_thresholds("fp32"), reference=reference)

    def test_detects_corrupted_kv(self) -> None:
        import inference.qwen2.paged_attn as paged_attn

        original = paged_attn.read_kv

        def corrupt(cache_k, cache_v, block_ids, seq_len, page_size):
            keys, values = original(cache_k, cache_v, block_ids, seq_len, page_size)
            keys[:, -1, :] = 0
            return keys, values

        reference, engine = _pair()
        with patch.object(paged_attn, "read_kv", corrupt):
            results = run(
                engine,
                ["hi", "a prompt that spans several kv pages"],
                12,
                default_thresholds("fp32"),
                reference,
            )
        self.assertTrue(all(not r.passed for r in results))

    def test_requires_paged_engine(self) -> None:
        reference, paged = _pair()
        engine = TorchEngine(paged.model, paged.tokenizer, "cpu")
        with self.assertRaises(RuntimeError):
            run(engine, ["hi"], max_new_tokens=2, thresholds=default_thresholds("fp32"), reference=reference)


if __name__ == "__main__":
    unittest.main()
