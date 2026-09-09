"""HuggingFace causal-LM engine. KV-Cache stays in PyTorch; BlockPool is occupancy only."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator

from inference.engine import (
    BlockAllocator,
    GenerateRequest,
    TokenEvent,
    clamp_max_tokens,
    release_blocks,
    reserve_blocks,
)

_log = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def resolve_device(requested: str) -> str:
    device = requested.strip().lower()
    if device not in {"cuda", "cpu"}:
        raise ValueError("WORKER_DEVICE must be 'cuda' or 'cpu'")
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "WORKER_DEVICE=cuda but torch.cuda.is_available() is False. "
            "Install a CUDA build of torch or set WORKER_DEVICE=cpu."
        )
    return device


class TorchEngine:
    def __init__(
        self,
        model,
        tokenizer,
        device: str,
        pool: BlockAllocator | None = None,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._pool = pool

    @classmethod
    def load(
        cls,
        model_id: str,
        device: str = "cuda",
        pool: BlockAllocator | None = None,
    ) -> TorchEngine:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        resolved = resolve_device(device)
        dtype = torch.float16 if resolved == "cuda" else torch.float32
        _log.info("loading torch model id=%s device=%s dtype=%s", model_id, resolved, dtype)

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        model.to(resolved)
        model.eval()
        _log.info("torch model ready id=%s device=%s", model_id, resolved)
        return cls(model, tokenizer, resolved, pool)

    async def generate(self, req: GenerateRequest) -> AsyncIterator[TokenEvent]:
        started = time.perf_counter()
        cap = clamp_max_tokens(req.max_tokens)
        prompt_ids = self._encode(req.prompt)
        prompt_tokens = len(prompt_ids)
        block_ids: list[int] = []
        ttft_ms = 0.0

        try:
            block_ids = reserve_blocks(self._pool, req.request_id, cap)
            if not prompt_ids:
                now = time.perf_counter()
                yield TokenEvent(
                    token="",
                    finished=True,
                    prompt_tokens=0,
                    completion_tokens=0,
                    time_to_first_token_ms=0.0,
                    total_latency_ms=_elapsed_ms(started, now),
                )
                return

            loop = asyncio.get_running_loop()
            current = prompt_ids
            past = None
            eos_id = self._tokenizer.eos_token_id

            for step in range(cap):
                logits, past = await loop.run_in_executor(
                    None, self._forward_step, current, past
                )
                token_id = _pick_token(logits, req.temperature)
                now = time.perf_counter()
                if step == 0:
                    ttft_ms = _elapsed_ms(started, now)
                text = self._tokenizer.decode(
                    [token_id], skip_special_tokens=True
                )
                finished = token_id == eos_id or step == cap - 1
                yield TokenEvent(
                    token=text,
                    finished=finished,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=step + 1,
                    time_to_first_token_ms=ttft_ms if finished else 0.0,
                    total_latency_ms=_elapsed_ms(started, now) if finished else 0.0,
                )
                if token_id == eos_id:
                    return
                current = [token_id]
        finally:
            release_blocks(self._pool, req.request_id, block_ids)

    def _encode(self, prompt: str) -> list[int]:
        if getattr(self._tokenizer, "chat_template", None):
            return self._tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=True,
            )
        return self._tokenizer.encode(prompt, add_special_tokens=True)

    def _forward_step(self, input_ids: list[int], past):
        import torch

        with torch.inference_mode():
            tensor = torch.tensor([input_ids], dtype=torch.long, device=self._device)
            out = self._model(
                input_ids=tensor,
                past_key_values=past,
                use_cache=True,
            )
            logits = out.logits[0, -1, :].float().cpu()
            return logits, out.past_key_values


def _pick_token(logits, temperature: float) -> int:
    import torch

    if temperature <= 0:
        return int(torch.argmax(logits).item())
    probs = torch.softmax(logits / temperature, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def _elapsed_ms(started: float, now: float) -> float:
    return (now - started) * 1000.0
