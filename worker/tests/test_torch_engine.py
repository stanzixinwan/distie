from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from inference.engine import DEFAULT_MAX_TOKENS, GenerateRequest
from inference.kv_cache import KvShape, PagedKvCache
from inference.torch_engine import TorchEngine, resolve_device
from recording_pool import RecordingPool


class StubTokenizer:
    eos_token_id = 9
    pad_token_id = 9
    chat_template = None

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [1, 2]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return {3: "hello", 4: "world", 9: ""}.get(ids[0], f"t{ids[0]}")


class StubModel:
    def __init__(self, next_ids: list[int]) -> None:
        self._next = list(next_ids)
        self._i = 0

    def __call__(self, input_ids, past_key_values=None, use_cache=True):
        import torch

        vocab = 16
        seq = input_ids.shape[1]
        logits = torch.zeros(1, seq, vocab)
        token_id = self._next[min(self._i, len(self._next) - 1)]
        logits[0, -1, token_id] = 20.0
        self._i += 1
        return SimpleNamespace(logits=logits, past_key_values=past_key_values or ("cache",))


class ResolveDeviceTest(unittest.TestCase):
    def test_rejects_unknown(self) -> None:
        with self.assertRaises(ValueError):
            resolve_device("tpu")

    def test_cuda_requires_runtime(self) -> None:
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaises(RuntimeError):
                resolve_device("cuda")

    def test_cpu_always_ok(self) -> None:
        self.assertEqual(resolve_device("CPU"), "cpu")


class TorchEngineTest(unittest.IsolatedAsyncioTestCase):
    async def test_greedy_streams_until_eos(self) -> None:
        pool = RecordingPool()
        engine = TorchEngine(StubModel([3, 4, 9]), StubTokenizer(), "cpu", pool=pool)
        events = [
            event
            async for event in engine.generate(
                GenerateRequest(
                    request_id="r1",
                    model_name="stub",
                    prompt="hi",
                    max_tokens=8,
                )
            )
        ]
        self.assertEqual([e.token for e in events], ["hello", "world", ""])
        self.assertTrue(events[-1].finished)
        self.assertEqual(events[-1].prompt_tokens, 2)
        self.assertEqual(events[-1].completion_tokens, 3)
        self.assertEqual(pool.freed, pool.allocated)
        self.assertEqual(len(pool.allocated[0]), 2)  # cap 8 / 4 tokens per block

    async def test_caps_at_max_tokens(self) -> None:
        engine = TorchEngine(StubModel([3, 3, 3, 3]), StubTokenizer(), "cpu")
        events = [
            event
            async for event in engine.generate(
                GenerateRequest(
                    request_id="r1",
                    model_name="stub",
                    prompt="hi",
                    max_tokens=2,
                )
            )
        ]
        self.assertEqual(len(events), 2)
        self.assertTrue(events[-1].finished)

    async def test_releases_blocks_on_cancel(self) -> None:
        pool = RecordingPool()
        engine = TorchEngine(StubModel([3, 4, 3]), StubTokenizer(), "cpu", pool=pool)
        agen = engine.generate(
            GenerateRequest(
                request_id="r1",
                model_name="stub",
                prompt="hi",
                max_tokens=8,
            )
        )
        first = await agen.__anext__()
        self.assertEqual(first.token, "hello")
        self.assertEqual(len(pool.freed), 0)
        await agen.aclose()
        self.assertEqual(pool.freed, pool.allocated)

    async def test_default_max_tokens_when_zero(self) -> None:
        engine = TorchEngine(
            StubModel([3] * (DEFAULT_MAX_TOKENS + 2)),
            StubTokenizer(),
            "cpu",
        )
        events = [
            event
            async for event in engine.generate(
                GenerateRequest(
                    request_id="r1",
                    model_name="stub",
                    prompt="hi",
                    max_tokens=0,
                )
            )
        ]
        self.assertEqual(len(events), DEFAULT_MAX_TOKENS)

    async def test_paged_kv_passes_gathered_past(self) -> None:
        import torch

        pool = RecordingPool(num_blocks=8)
        shape = KvShape(num_layers=2, num_kv_heads=2, head_dim=4)
        kv = PagedKvCache(shape, num_blocks=8, device="cpu", dtype=torch.float32, page_size=4)
        model = KvStubModel()
        engine = TorchEngine(model, StubTokenizer(), "cpu", pool=pool, kv_cache=kv)
        events = [
            event
            async for event in engine.generate(
                GenerateRequest(
                    request_id="r1",
                    model_name="stub",
                    prompt="hi",
                    max_tokens=3,
                )
            )
        ]
        self.assertEqual(len(events), 3)
        # Prefill sees no past; each decode gathers prompt (+ earlier tokens).
        self.assertEqual(model.past_lens[0], 0)
        self.assertEqual(model.past_lens[1], 2)
        self.assertEqual(model.past_lens[2], 3)
        # 2 prompt + 3 new tokens, page_size 4 -> 2 pages
        self.assertEqual(len(pool.allocated[0]), 2)
        self.assertEqual(pool.freed, pool.allocated)

    def test_paged_cache_requires_pool(self) -> None:
        import torch

        kv = PagedKvCache(
            KvShape(num_layers=1, num_kv_heads=1, head_dim=4),
            num_blocks=4,
            device="cpu",
            dtype=torch.float32,
        )
        with self.assertRaises(ValueError):
            TorchEngine(StubModel([3]), StubTokenizer(), "cpu", kv_cache=kv)


class KvStubModel:
    """Emits real-shaped KV so TorchEngine can scatter/gather."""

    config = SimpleNamespace(
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_size=16,
    )

    def __init__(self) -> None:
        self.past_lens: list[int] = []

    def __call__(self, input_ids, past_key_values=None, use_cache=True):
        import torch

        seq = input_ids.shape[1]
        past = _legacy_cache(past_key_values)
        past_len = 0 if past is None else past[0][0].shape[2]
        self.past_lens.append(past_len)
        extra = tuple(
            (
                torch.full((1, 2, seq, 4), float(past_len + 1)),
                torch.full((1, 2, seq, 4), float(past_len + 2)),
            )
            for _ in range(2)
        )
        if past is None:
            cache = extra
        else:
            cache = tuple(
                (torch.cat([past[i][0], extra[i][0]], dim=2), torch.cat([past[i][1], extra[i][1]], dim=2))
                for i in range(2)
            )
        vocab = 16
        logits = torch.zeros(1, seq, vocab)
        logits[0, -1, 3] = 20.0
        return SimpleNamespace(logits=logits, past_key_values=cache)


def _legacy_cache(past):
    if past is None:
        return None
    if hasattr(past, "to_legacy_cache"):
        return past.to_legacy_cache()
    return past


if __name__ == "__main__":
    unittest.main()
