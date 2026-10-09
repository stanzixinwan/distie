"""Dense Qwen2 forward against a tiny HuggingFace Qwen2ForCausalLM."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from inference.qwen2 import Qwen2CausalLM


def _tiny_config() -> Qwen2Config:
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


class Qwen2DenseParityTest(unittest.TestCase):
    def test_logits_match_hf_eager(self) -> None:
        torch.manual_seed(0)
        config = _tiny_config()
        hf = Qwen2ForCausalLM(config).eval()
        ours = Qwen2CausalLM.from_hf(hf)
        ids = torch.randint(0, config.vocab_size, (1, 7))
        with torch.inference_mode():
            ref = hf(ids, use_cache=False).logits
            got = ours(ids)
        err = (ref - got).abs().max().item()
        self.assertLess(err, 1e-4)
        self.assertEqual(int(ref[0, -1].argmax()), int(got[0, -1].argmax()))

    def test_tied_embeddings_share_storage(self) -> None:
        torch.manual_seed(1)
        config = _tiny_config()
        config.tie_word_embeddings = True
        hf = Qwen2ForCausalLM(config).eval()
        ours = Qwen2CausalLM.from_hf(hf)
        self.assertIs(ours.lm_head.weight, ours.embed_tokens.weight)
        ids = torch.randint(0, config.vocab_size, (1, 3))
        with torch.inference_mode():
            ref = hf(ids, use_cache=False).logits
            got = ours(ids)
        self.assertLess((ref - got).abs().max().item(), 1e-4)

    def test_rejects_sliding_window(self) -> None:
        config = _tiny_config()
        config.layer_types = ["sliding_attention", "full_attention"]
        hf = Qwen2ForCausalLM(_tiny_config()).eval()
        hf.config.layer_types = ["sliding_attention", "full_attention"]
        with self.assertRaises(ValueError):
            Qwen2CausalLM.from_hf(hf)

    def test_rejects_empty_sequence(self) -> None:
        torch.manual_seed(0)
        ours = Qwen2CausalLM.from_hf(Qwen2ForCausalLM(_tiny_config()).eval())
        with self.assertRaises(ValueError):
            ours(torch.zeros(1, 0, dtype=torch.long))


if __name__ == "__main__":
    unittest.main()
