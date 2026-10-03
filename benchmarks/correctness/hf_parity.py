"""Check TorchEngine's paged-KV path against HuggingFace `generate`.

For each prompt:
  reference  = HF generate, pure greedy, with per-step logits
  free-run   = engine.trace(prompt)                      -> greedy token match
  forced     = engine.trace(prompt, forced=reference)    -> top-1 agreement, logit error

Usage:
  python benchmarks/correctness/hf_parity.py --model Qwen/Qwen2.5-1.5B-Instruct \
      --device cuda --dtype fp32 --out benchmarks/results/correctness-fp32.json

Exit code: 0 all prompts pass, 1 some prompt fails, 2 setup error.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_WORKER_SRC = _HERE.parents[1] / "worker" / "src"
for _p in (_HERE, _WORKER_SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from metrics import PromptResult, Thresholds, default_thresholds, evaluate, summarize  # noqa: E402

_log = logging.getLogger("hf_parity")

DEFAULT_PROMPTS = _HERE / "prompts.json"


def load_prompts(path: Path) -> list[str]:
    prompts = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(prompts, list) or not all(isinstance(p, str) and p.strip() for p in prompts):
        raise ValueError(f"{path} must be a JSON list of non-empty strings")
    return prompts


def reference_generate(model, tokenizer, prompt_ids: list[int], max_new_tokens: int):
    """Pure greedy HF generate. Returns (token_ids, logits[steps, vocab] float32 CPU).

    transformers fills every field left unset here from the checkpoint's
    generation_config (Qwen2.5 ships repetition_penalty=1.1), so neutral
    values must be explicit. The argmax check below catches any processor
    that still slips through.
    """
    import torch
    from transformers import GenerationConfig

    device = next(model.parameters()).device
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        output_logits=True,
        return_dict_in_generate=True,
    )
    with torch.inference_mode():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            generation_config=config,
        )
    ids = out.sequences[0, len(prompt_ids) :].tolist()
    logits = torch.stack([step[0] for step in out.logits]).float().cpu()
    greedy = logits.argmax(dim=-1).tolist()
    if greedy != ids:
        step = next(i for i, (g, r) in enumerate(zip(greedy, ids)) if g != r)
        raise RuntimeError(
            f"HF reference is not pure greedy at step {step} (picked {ids[step]}, argmax {greedy[step]}); "
            "a logits processor from the checkpoint's generation_config is active"
        )
    return ids, logits


def check_prompt(engine, index: int, prompt: str, max_new_tokens: int, thresholds: Thresholds) -> PromptResult:
    prompt_ids = engine.encode(prompt)
    ref_ids, ref_logits = reference_generate(engine.model, engine.tokenizer, prompt_ids, max_new_tokens)
    if not ref_ids:
        raise RuntimeError(f"prompt {index}: HF generate produced no tokens")
    free = engine.trace(prompt_ids, max_new_tokens)
    forced = engine.trace(prompt_ids, len(ref_ids), forced_ids=ref_ids)
    return evaluate(
        index=index,
        prompt_tokens=len(prompt_ids),
        reference_ids=ref_ids,
        candidate_ids=free.token_ids,
        reference_logits=ref_logits,
        forced_logits=forced.logits,
        thresholds=thresholds,
    )


def run(engine, prompts: list[str], max_new_tokens: int, thresholds: Thresholds) -> list[PromptResult]:
    if not engine.paged:
        raise RuntimeError("engine has no PagedKvCache; parity would only test HF against itself")
    results = []
    for i, prompt in enumerate(prompts):
        started = time.perf_counter()
        result = check_prompt(engine, i, prompt, max_new_tokens, thresholds)
        _log.info(
            "prompt=%s passed=%s divergence=%s top1=%.4f max_err=%.3e elapsed_s=%.1f",
            i,
            result.passed,
            result.first_divergence,
            result.top1_agreement,
            result.max_abs_logit_err,
            time.perf_counter() - started,
        )
        results.append(result)
    return results


def environment(model_id: str, device: str, dtype: str) -> dict:
    import torch
    import transformers

    env = {
        "model": model_id,
        "device": device,
        "dtype": dtype,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
    }
    if device == "cuda":
        env["gpu"] = torch.cuda.get_device_name(0)
    return env


def format_table(results: list[PromptResult]) -> str:
    lines = [f"{'#':>3} {'prompt':>6} {'ref':>4} {'cand':>4} {'diverge':>7} {'top1':>7} {'max_err':>10}  result"]
    for r in results:
        div = "-" if r.first_divergence is None else str(r.first_divergence)
        lines.append(
            f"{r.index:>3} {r.prompt_tokens:>6} {r.reference_tokens:>4} {r.candidate_tokens:>4} "
            f"{div:>7} {r.top1_agreement:>7.4f} {r.max_abs_logit_err:>10.3e}  "
            f"{'PASS' if r.passed else 'FAIL'}"
        )
    return "\n".join(lines)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--num-blocks", type=int, default=256)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--out", type=Path, help="write a JSON report here")
    args = parser.parse_args(argv)
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be >= 1")
    if args.num_blocks < 1:
        parser.error("--num-blocks must be >= 1")
    return args


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args(argv)

    try:
        from inference.native import load_block_pool
        from inference.torch_engine import TorchEngine

        prompts = load_prompts(args.prompts)
        pool = load_block_pool(args.num_blocks)
        engine = TorchEngine.load(args.model, device=args.device, pool=pool, dtype=args.dtype)
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        _log.error("setup failed: %s", exc)
        return 2

    thresholds = default_thresholds(args.dtype)
    results = run(engine, prompts, args.max_new_tokens, thresholds)
    summary = summarize(results)
    print(format_table(results))
    print(json.dumps(summary, indent=2))

    if args.out:
        report = {
            "environment": environment(args.model, args.device, args.dtype),
            "max_new_tokens": args.max_new_tokens,
            "thresholds": thresholds.__dict__,
            "summary": summary,
            "results": [r.to_dict() for r in results],
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        _log.info("report written to %s", args.out)

    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
