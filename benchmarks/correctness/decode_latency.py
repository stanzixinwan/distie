"""Batch-1 decode latency: in-house flash paged attention vs HF generate.

Same prompt, same number of new tokens, fp16, greedy. TPOT is the mean gap
between tokens after the first. The flash path must not be slower.

  DISTIE_ATTN=flash python benchmarks/correctness/decode_latency.py \
      --out benchmarks/results/decode-latency-3080.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_WORKER_SRC = _HERE.parents[1] / "worker" / "src"
for _p in (_HERE, _WORKER_SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from hf_parity import _encode  # noqa: E402


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--num-blocks", type=int, default=8)
    parser.add_argument(
        "--prompt",
        default=(
            "Explain what a KV cache is in large language model inference, "
            "in three sentences."
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.device != "cuda":
        raise SystemExit("decode latency is a CUDA measurement")
    if args.max_new_tokens < 2:
        raise SystemExit("--max-new-tokens must be >= 2 so TPOT has a gap")
    if args.repeats < 1 or args.warmup < 1:
        raise SystemExit("--repeats and --warmup must be >= 1")
    return args


def _tpot(stamps: list[float]) -> tuple[float, float]:
    """stamps[0] is the start. Later stamps are the end of each new token."""
    gaps = [stamps[i] - stamps[i - 1] for i in range(1, len(stamps))]
    ttft_ms = gaps[0] * 1000.0
    tpot_ms = (sum(gaps[1:]) / len(gaps[1:])) * 1000.0
    return ttft_ms, tpot_ms


def _time_hf(model, tokenizer, prompt_ids: list[int], steps: int) -> tuple[list[int], float, float]:
    import torch
    from transformers import GenerationConfig, LogitsProcessor, LogitsProcessorList

    stamps: list[float] = []

    class Stamp(LogitsProcessor):
        def __call__(self, input_ids, scores):
            torch.cuda.synchronize()
            stamps.append(time.perf_counter())
            return scores

    device = next(model.parameters()).device
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    config = GenerationConfig(
        max_new_tokens=steps,
        min_new_tokens=steps,
        do_sample=False,
        repetition_penalty=1.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    torch.cuda.synchronize()
    stamps.append(time.perf_counter())
    with torch.inference_mode():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            generation_config=config,
            logits_processor=LogitsProcessorList([Stamp()]),
        )
    ids = out[0, len(prompt_ids) :].tolist()
    if len(stamps) != steps + 1:
        raise RuntimeError(f"HF produced {len(stamps) - 1} timed steps, expected {steps}")
    ttft_ms, tpot_ms = _tpot(stamps)
    return ids, ttft_ms, tpot_ms


def _time_flash(engine, prompt_ids: list[int], steps: int) -> tuple[list[int], float, float]:
    import torch

    block_ids = engine._reserve("latency", len(prompt_ids), steps)
    picks: list[int] = []
    stamps: list[float] = []
    try:
        current = list(prompt_ids)
        past = None
        seq_len = 0
        torch.cuda.synchronize()
        stamps.append(time.perf_counter())
        for _ in range(steps):
            logits, past, seq_len = engine._forward_step(current, past, block_ids, seq_len)
            stamps.append(time.perf_counter())
            pick = int(torch.argmax(logits).item())
            picks.append(pick)
            current = [pick]
    finally:
        from inference.engine import release_blocks

        release_blocks(engine._pool, "latency", block_ids)
    ttft_ms, tpot_ms = _tpot(stamps)
    return picks, ttft_ms, tpot_ms


def _median_run(fn, warmup: int, repeats: int):
    for _ in range(warmup):
        fn()
    rows = [fn() for _ in range(repeats)]
    tpots = [row[2] for row in rows]
    mid = sorted(range(repeats), key=lambda i: tpots[i])[len(tpots) // 2]
    return rows[mid], tpots


def main(argv: list[str] | None = None) -> int:
    os.environ["DISTIE_ATTN"] = "flash"
    args = _parse_args(argv)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from inference.kv_cache import KvShape, PagedKvCache
    from inference.native import load_block_pool
    from inference.qwen2 import Qwen2CausalLM
    from inference.qwen2.paged_attn import serving_page_size
    from inference.torch_engine import TorchEngine, resolve_dtype

    if not torch.cuda.is_available():
        print("CUDA is not available", file=sys.stderr)
        return 2
    dtype = resolve_dtype("fp16", args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompt_ids = _encode(tokenizer, args.prompt)

    hf = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, trust_remote_code=True)
    hf.to(args.device).eval()
    hf_row, hf_tpots = _median_run(
        lambda: _time_hf(hf, tokenizer, prompt_ids, args.max_new_tokens),
        args.warmup,
        args.repeats,
    )
    del hf
    torch.cuda.empty_cache()

    loaded = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, trust_remote_code=True)
    loaded.to(args.device).eval()
    model = Qwen2CausalLM.from_hf(loaded)
    del loaded
    torch.cuda.empty_cache()
    pool = load_block_pool(args.num_blocks)
    cache = PagedKvCache(
        KvShape(model.dims.num_hidden_layers, model.dims.num_key_value_heads, model.dims.head_dim),
        num_blocks=pool.num_blocks,
        device=args.device,
        dtype=dtype,
        page_size=serving_page_size(dtype),
    )
    engine = TorchEngine(model, tokenizer, args.device, pool=pool, kv_cache=cache)
    flash_row, flash_tpots = _median_run(
        lambda: _time_flash(engine, prompt_ids, args.max_new_tokens),
        args.warmup,
        args.repeats,
    )

    hf_ids, hf_ttft, hf_tpot = hf_row
    flash_ids, flash_ttft, flash_tpot = flash_row
    same = hf_ids == flash_ids
    report = {
        "environment": {
            "model": args.model,
            "device": args.device,
            "dtype": "fp16",
            "attn": "flash",
            "page_size": cache.page_size,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "prompt_tokens": len(prompt_ids),
        "max_new_tokens": args.max_new_tokens,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "tokens_match": same,
        "hf": {"ttft_ms": hf_ttft, "tpot_ms": hf_tpot, "tpot_ms_runs": hf_tpots},
        "flash": {"ttft_ms": flash_ttft, "tpot_ms": flash_tpot, "tpot_ms_runs": flash_tpots},
        "flash_tpot_le_hf": flash_tpot <= hf_tpot,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("tokens_match", "hf", "flash", "flash_tpot_le_hf")}, indent=2))
    if not same or flash_tpot > hf_tpot:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
