"""Decode with RoPE applied inside the flash kernel (rotary_cos / rotary_sin)."""

from __future__ import annotations

import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from inference.kv_cache import KvShape, PagedKvCache
from inference.qwen2 import paged_attn
from inference.qwen2 import Qwen2CausalLM
from inference.qwen2.model import RotaryEmbedding, _apply_rope


def _config() -> Qwen2Config:
    return Qwen2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        rope_theta=10000.0,
        tie_word_embeddings=False,
        # Larger weights give peaked attention, so a wrong rotation shows up in the logits.
        initializer_range=0.2,
    )


def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate-half with half-width tables, as flash does with rotary_interleaved=False."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half].float(), x[..., half:].float()
    c, s = cos.float(), sin.float()
    return torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1).to(x.dtype)


class FakeFlash:
    """Reference flash_attn_with_kvcache: paged append, optional in-kernel rotary, GQA attention."""

    def __init__(self) -> None:
        self.rotary_calls = 0
        self.plain_calls = 0

    def __call__(self, q, k_cache, v_cache, k=None, v=None, cache_seqlens=None, block_table=None,
                 causal=False, rotary_cos=None, rotary_sin=None, rotary_interleaved=True, **kwargs):
        # q [B, 1, H, D], k/v [B, 1, KH, D], caches [blocks, page, KH, D]
        page = k_cache.shape[1]
        pos = cache_seqlens.long()
        if rotary_cos is not None:
            assert not rotary_interleaved, "Qwen uses rotate-half RoPE"
            self.rotary_calls += 1
            cos, sin = rotary_cos[pos][:, None, None, :], rotary_sin[pos][:, None, None, :]
            q, k = _rotate(q, cos, sin), _rotate(k, cos, sin)
        else:
            self.plain_calls += 1
        out = torch.empty_like(q)
        for b in range(q.shape[0]):
            p = int(pos[b])
            blk = int(block_table[b, p // page])
            k_cache[blk, p % page] = k[b, 0]
            v_cache[blk, p % page] = v[b, 0]
            n = p + 1
            pages = [int(block_table[b, i]) for i in range((n + page - 1) // page)]
            ks = torch.cat([k_cache[i] for i in pages])[:n].float()  # [n, KH, D]
            vs = torch.cat([v_cache[i] for i in pages])[:n].float()
            rep = q.shape[2] // ks.shape[1]
            ks, vs = ks.repeat_interleave(rep, dim=1), vs.repeat_interleave(rep, dim=1)
            scores = torch.einsum("hd,nhd->hn", q[b, 0].float(), ks) / q.shape[-1] ** 0.5
            out[b, 0] = torch.einsum("hn,nhd->hd", scores.softmax(-1), vs).to(q.dtype)
        return out


class RopeTableTest(unittest.TestCase):
    def test_half_table_reproduces_hf_rotation(self) -> None:
        torch.manual_seed(0)
        rope = RotaryEmbedding(head_dim=16, rope_theta=10000.0)
        q = torch.randn(3, 4, 1, 16)
        k = torch.randn(3, 2, 1, 16)
        positions = torch.tensor([[0], [7], [300]])
        cos, sin = rope(q, positions)
        want_q, want_k = _apply_rope(q, k, cos, sin)
        cos_t, sin_t = rope.half_table(512, "cpu", torch.float32)
        self.assertEqual(tuple(cos_t.shape), (512, 8))
        row_c, row_s = cos_t[positions[:, 0]][:, None, None, :], sin_t[positions[:, 0]][:, None, None, :]
        self.assertLess((_rotate(q, row_c, row_s) - want_q).abs().max().item(), 1e-5)
        self.assertLess((_rotate(k, row_c, row_s) - want_k).abs().max().item(), 1e-5)

    def test_table_cached_per_device_and_dtype(self) -> None:
        model = Qwen2CausalLM.from_hf(Qwen2ForCausalLM(_config()).eval())
        a = model.rope_table("cpu", torch.float16)
        self.assertIs(model.rope_table("cpu", torch.float16), a)
        self.assertEqual(tuple(a[0].shape), (512, 8))
        self.assertEqual(a[0].dtype, torch.float16)
        self.assertIsNot(model.rope_table("cpu", torch.float32), a)


class FusedRopeDecodeTest(unittest.TestCase):
    def setUp(self) -> None:
        self._env = {k: os.environ.get(k) for k in ("DISTIE_ATTN", "DISTIE_FUSED_ROPE")}
        self.fake = FakeFlash()
        module = types.ModuleType("flash_attn")
        module.flash_attn_with_kvcache = self.fake
        module.flash_attn_varlen_func = lambda *a, **k: (_ for _ in ()).throw(AssertionError("decode only"))
        sys.modules["flash_attn"] = module
        os.environ["DISTIE_ATTN"] = "flash"
        # The real kernel is fp16/bf16 only; the fake one is exact in fp32, so force
        # the flash route for fp32 and compare without rounding noise.
        self._backend = mock.patch.object(paged_attn, "_backend", lambda dtype: "flash")
        self._backend.start()

    def tearDown(self) -> None:
        self._backend.stop()
        sys.modules.pop("flash_attn", None)
        for key, value in self._env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _decode_steps(self, fused: bool) -> tuple[torch.Tensor, PagedKvCache]:
        os.environ["DISTIE_FUSED_ROPE"] = "1" if fused else "0"
        torch.manual_seed(0)
        config = _config()
        model = Qwen2CausalLM.from_hf(Qwen2ForCausalLM(config).eval())
        cache = PagedKvCache(KvShape.from_hf_config(config), num_blocks=6, device="cpu", dtype=torch.float32, page_size=256)
        # Two rows: positions start at 0 and 254, so row 1 crosses into its second page.
        table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
        seqlen = torch.tensor([0, 254], dtype=torch.int32)
        rows = []
        with torch.inference_mode():
            for step in range(4):
                ids = torch.tensor([[5 + step], [9 + step]])
                logits = model.forward_decode(ids, cache, table, seqlen.long().unsqueeze(1), seqlen)
                rows.append(logits[:, 0].float())
                seqlen = seqlen + 1
        return torch.stack(rows), cache

    def test_fused_matches_separate_rope(self) -> None:
        separate, sep_cache = self._decode_steps(fused=False)
        self.assertEqual((self.fake.rotary_calls, self.fake.plain_calls), (0, 8))
        fused, fused_cache = self._decode_steps(fused=True)
        self.assertEqual(self.fake.rotary_calls, 8)
        # The kernel stores the rotated K: both paths must leave the same slab.
        for layer in range(_config().num_hidden_layers):
            for a, b in zip(sep_cache.layer_kv(layer), fused_cache.layer_kv(layer)):
                self.assertLess((a - b).abs().max().item(), 1e-4)
        self.assertLess((fused - separate).abs().max().item(), 1e-4)

if __name__ == "__main__":
    unittest.main()
