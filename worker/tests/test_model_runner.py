"""ModelRunner routing: pure-decode steps take the bucketed decode path, others stay varlen."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from inference.kv_cache import KvShape, PagedKvCache
from inference.model_runner import ModelRunner
from inference.qwen2 import Qwen2CausalLM
from inference.qwen2.batch_decode import BatchDecodeGraphs
from inference.scheduler import ScheduledSeq, Sequence, StepPlan


def _model() -> tuple[Qwen2CausalLM, Qwen2Config]:
    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        tie_word_embeddings=False,
    )
    return Qwen2CausalLM.from_hf(Qwen2ForCausalLM(config).eval()), config


def _cache(config: Qwen2Config) -> PagedKvCache:
    # 16 pool blocks plus the spare page 16.
    return PagedKvCache(KvShape.from_hf_config(config), num_blocks=17, device="cpu", dtype=torch.float32, page_size=4)


class CountingDecode(BatchDecodeGraphs):
    def __init__(self, *args, fail: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.calls = 0
        self.fail = fail

    def run(self, rows):
        self.calls += 1
        if self.fail:
            raise RuntimeError("capture failed")
        return super().run(rows)


def _generate(runner: ModelRunner, steps: int) -> list[list[int]]:
    """Drive the runner the way BatchLoop does: one prefill step, then decode steps."""
    prompts = [[3, 5, 7], [11, 13, 17, 19, 23], [29]]
    tables = [[0, 1, 2], [3, 4, 5], [6, 7]]
    seqs = [Sequence(request_id=f"r{i}", prompt_ids=p, max_new_tokens=steps, block_ids=t) for i, (p, t) in enumerate(zip(prompts, tables))]
    plan = StepPlan(tuple(ScheduledSeq(s, list(s.prompt_ids), 0) for s in seqs))
    for _ in range(steps):
        picks = runner.step(plan)
        for seq, item, tok in zip(seqs, plan.items, picks):
            seq.num_cached = item.past_len + len(item.token_ids)
            seq.output_ids.append(tok)
        plan = StepPlan(tuple(ScheduledSeq(s, [s.output_ids[-1]], s.num_cached) for s in seqs))
    return [s.output_ids for s in seqs]


class ModelRunnerRoutingTest(unittest.TestCase):
    def test_decode_path_matches_varlen(self) -> None:
        model, config = _model()
        eager = _generate(ModelRunner(model, _cache(config), "cpu"), steps=6)
        cache = _cache(config)
        decode = CountingDecode(model, cache, pad_block=16, max_batch=4, use_graphs=False)
        routed = _generate(ModelRunner(model, cache, "cpu", decode), steps=6)
        self.assertEqual(routed, eager)
        # Step 1 is a prefill and stays varlen; the other five are pure decode.
        self.assertEqual(decode.calls, 5)

    def test_failure_falls_back_to_varlen(self) -> None:
        model, config = _model()
        eager = _generate(ModelRunner(model, _cache(config), "cpu"), steps=4)
        cache = _cache(config)
        decode = CountingDecode(model, cache, pad_block=16, max_batch=4, use_graphs=False, fail=True)
        runner = ModelRunner(model, cache, "cpu", decode)
        self.assertEqual(_generate(runner, steps=4), eager)
        self.assertEqual(decode.calls, 1)
        self.assertFalse(runner.graphs_active)

    def test_batches_wider_than_the_graphs_stay_varlen(self) -> None:
        model, config = _model()
        cache = _cache(config)
        decode = CountingDecode(model, cache, pad_block=16, max_batch=2, use_graphs=False)
        _generate(ModelRunner(model, cache, "cpu", decode), steps=3)
        self.assertEqual(decode.calls, 0)


if __name__ == "__main__":
    unittest.main()
