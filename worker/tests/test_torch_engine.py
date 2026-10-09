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
from inference.torch_engine import TorchEngine, resolve_device, resolve_dtype
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


class ResolveDtypeTest(unittest.TestCase):
    def test_auto_follows_device(self) -> None:
        import torch

        self.assertEqual(resolve_dtype("auto", "cuda"), torch.float16)
        self.assertEqual(resolve_dtype("auto", "cpu"), torch.float32)

    def test_explicit(self) -> None:
        import torch

        self.assertEqual(resolve_dtype("FP32", "cuda"), torch.float32)
        self.assertEqual(resolve_dtype("bf16", "cpu"), torch.bfloat16)

    def test_rejects_unknown(self) -> None:
        with self.assertRaises(ValueError):
            resolve_dtype("int8", "cpu")


class ChatTokenizer(StubTokenizer):
    """Mimics transformers 5: apply_chat_template returns a dict unless told not to."""

    chat_template = "{{ messages }}"

    def apply_chat_template(self, messages, add_generation_prompt, tokenize, return_dict=True):
        ids = [7, 8, 1, 2]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)} if return_dict else ids


class EncodeTest(unittest.TestCase):
    def test_chat_template_returns_token_ids(self) -> None:
        engine = TorchEngine(StubModel([3]), ChatTokenizer(), "cpu")
        self.assertEqual(engine.encode("hi"), [7, 8, 1, 2])

    def test_plain_tokenizer(self) -> None:
        engine = TorchEngine(StubModel([3]), StubTokenizer(), "cpu")
        self.assertEqual(engine.encode("hi"), [1, 2])


class TraceTest(unittest.TestCase):
    def test_free_run_stops_after_eos(self) -> None:
        pool = RecordingPool()
        engine = TorchEngine(StubModel([3, 4, 9, 5]), StubTokenizer(), "cpu", pool=pool)
        trace = engine.trace([1, 2], max_new_tokens=8)
        self.assertEqual(trace.token_ids, [3, 4, 9])
        self.assertEqual(tuple(trace.logits.shape), (3, 16))
        self.assertEqual(pool.freed, pool.allocated)

    def test_forced_feeds_given_tokens_past_eos(self) -> None:
        model = RecordingStubModel([9, 9, 9])
        engine = TorchEngine(model, StubTokenizer(), "cpu")
        trace = engine.trace([1, 2], max_new_tokens=3, forced_ids=[5, 6, 7])
        self.assertEqual(trace.token_ids, [9, 9, 9])
        self.assertEqual(model.inputs, [[1, 2], [5], [6]])

    def test_validates_arguments(self) -> None:
        engine = TorchEngine(StubModel([3]), StubTokenizer(), "cpu")
        with self.assertRaises(ValueError):
            engine.trace([], max_new_tokens=2)
        with self.assertRaises(ValueError):
            engine.trace([1], max_new_tokens=0)
        with self.assertRaises(ValueError):
            engine.trace([1], max_new_tokens=3, forced_ids=[5])

    def test_releases_blocks_on_error(self) -> None:
        pool = RecordingPool()
        engine = TorchEngine(ExplodingModel(), StubTokenizer(), "cpu", pool=pool)
        with self.assertRaises(RuntimeError):
            engine.trace([1, 2], max_new_tokens=4)
        self.assertEqual(pool.freed, pool.allocated)


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

    async def test_paged_qwen_frees_blocks(self) -> None:
        from transformers import Qwen2Config, Qwen2ForCausalLM

        from inference.qwen2 import Qwen2CausalLM

        import torch

        torch.manual_seed(0)
        config = Qwen2Config(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
        )
        pool = RecordingPool(num_blocks=8)
        kv = PagedKvCache(
            KvShape.from_hf_config(config),
            num_blocks=8,
            device="cpu",
            dtype=torch.float32,
            page_size=4,
        )
        engine = TorchEngine(
            Qwen2CausalLM.from_hf(Qwen2ForCausalLM(config).eval()),
            StubTokenizer(),
            "cpu",
            pool=pool,
            kv_cache=kv,
        )
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
        self.assertEqual(pool.num_free, pool.num_blocks)
        self.assertGreaterEqual(len(pool.allocated[0]), 1)
        await engine.close()

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


class RecordingStubModel(StubModel):
    def __init__(self, next_ids: list[int]) -> None:
        super().__init__(next_ids)
        self.inputs: list[list[int]] = []

    def __call__(self, input_ids, past_key_values=None, use_cache=True):
        self.inputs.append(input_ids[0].tolist())
        return super().__call__(input_ids, past_key_values, use_cache)


class ExplodingModel:
    def __call__(self, input_ids, past_key_values=None, use_cache=True):
        raise RuntimeError("cuda error")


if __name__ == "__main__":
    unittest.main()
