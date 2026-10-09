"""Qwen2 causal LM whose math matches HuggingFace eager attention.

HF attention only accepts a dense KV cache. This module is the same
network (RMSNorm, rotate-half RoPE, grouped-query attention, SwiGLU)
so a later step can read and write a paged slab instead of that cache.
Sliding-window layers are rejected: Qwen2.5-1.5B uses full attention.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import nn

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Qwen2Dims:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_hf_config(cls, config) -> Qwen2Dims:
        heads = int(config.num_attention_heads)
        kv_heads = int(config.num_key_value_heads)
        hidden = int(config.hidden_size)
        if heads < 1 or kv_heads < 1 or heads % kv_heads != 0:
            raise ValueError("num_attention_heads must be a positive multiple of num_key_value_heads")
        explicit_head_dim = getattr(config, "head_dim", None)
        head_dim = int(explicit_head_dim) if explicit_head_dim else hidden // heads
        if head_dim < 1 or hidden != heads * head_dim:
            raise ValueError("hidden_size must equal num_attention_heads * head_dim")
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotate-half RoPE")
        act = getattr(config, "hidden_act", "silu")
        if act != "silu":
            raise ValueError(f"only silu MLP is supported, got {act}")
        rope = _rope_theta(config)
        layers = int(config.num_hidden_layers)
        layer_types = list(getattr(config, "layer_types", []) or [])
        if layer_types and any(kind != "full_attention" for kind in layer_types):
            raise ValueError("sliding-window Qwen2 layers are not implemented")
        if layers < 1 or int(config.vocab_size) < 1 or int(config.intermediate_size) < 1:
            raise ValueError("invalid Qwen2 config")
        return cls(
            vocab_size=int(config.vocab_size),
            hidden_size=hidden,
            intermediate_size=int(config.intermediate_size),
            num_hidden_layers=layers,
            num_attention_heads=heads,
            num_key_value_heads=kv_heads,
            head_dim=head_dim,
            rms_norm_eps=float(config.rms_norm_eps),
            rope_theta=rope,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
        )


class RMSNorm(nn.Module):
    """T5-style RMSNorm. Variance is accumulated in fp32, matching HF."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype
        hidden = hidden_states.float()
        variance = hidden.pow(2).mean(-1, keepdim=True)
        hidden = hidden * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden.to(dtype)


class RotaryEmbedding(nn.Module):
    """Default RoPE: inv_freq = theta^{-2i/d}, then rotate-half."""

    def __init__(self, head_dim: int, rope_theta: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # position_ids: [batch, seq]. cos/sin match x's dtype after the trig.
        freqs = position_ids[..., None].float() * self.inv_freq.to(device=x.device, dtype=torch.float)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)


def _apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q, k: [batch, heads, seq, dim]. cos/sin: [batch, seq, dim].
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _repeat_kv(hidden: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden
    batch, kv_heads, seq, dim = hidden.shape
    hidden = hidden[:, :, None, :, :].expand(batch, kv_heads, n_rep, seq, dim)
    return hidden.reshape(batch, kv_heads * n_rep, seq, dim)


def _causal_mask(query_len: int, key_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Additive mask, 0 on allowed positions. Query i sees keys up to (past + i)."""
    past = key_len - query_len
    q_idx = torch.arange(query_len, device=device)[:, None]
    k_idx = torch.arange(key_len, device=device)[None, :]
    blocked = k_idx > (q_idx + past)
    mask = torch.zeros(query_len, key_len, device=device, dtype=dtype)
    return mask.masked_fill(blocked, torch.finfo(dtype).min).view(1, 1, query_len, key_len)


class Attention(nn.Module):
    """Grouped-query attention. Q/K/V have bias; the output projection does not."""

    def __init__(self, dims: Qwen2Dims) -> None:
        super().__init__()
        self.num_heads = dims.num_attention_heads
        self.num_kv_heads = dims.num_key_value_heads
        self.head_dim = dims.head_dim
        self.num_kv_groups = dims.num_key_value_groups
        self.scaling = self.head_dim**-0.5
        hidden = dims.hidden_size
        self.q_proj = nn.Linear(hidden, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden, bias=False)

    def forward(self, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = hidden.shape
        query = self.q_proj(hidden).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(hidden).view(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(hidden).view(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        query, key = _apply_rope(query, key, cos, sin)
        scores = torch.matmul(query, _repeat_kv(key, self.num_kv_groups).transpose(2, 3)) * self.scaling
        scores = scores + _causal_mask(seq, seq, scores.device, scores.dtype)
        weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        mixed = torch.matmul(weights, _repeat_kv(value, self.num_kv_groups))
        mixed = mixed.transpose(1, 2).contiguous().view(batch, seq, -1)
        return self.o_proj(mixed)


class MLP(nn.Module):
    def __init__(self, dims: Qwen2Dims) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dims.hidden_size, dims.intermediate_size, bias=False)
        self.up_proj = nn.Linear(dims.hidden_size, dims.intermediate_size, bias=False)
        self.down_proj = nn.Linear(dims.intermediate_size, dims.hidden_size, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class DecoderLayer(nn.Module):
    def __init__(self, dims: Qwen2Dims) -> None:
        super().__init__()
        self.self_attn = Attention(dims)
        self.mlp = MLP(dims)
        self.input_layernorm = RMSNorm(dims.hidden_size, dims.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(dims.hidden_size, dims.rms_norm_eps)

    def forward(self, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.self_attn(self.input_layernorm(hidden), cos, sin)
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class Qwen2CausalLM(nn.Module):
    def __init__(self, dims: Qwen2Dims) -> None:
        super().__init__()
        self.dims = dims
        self.embed_tokens = nn.Embedding(dims.vocab_size, dims.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(dims) for _ in range(dims.num_hidden_layers))
        self.norm = RMSNorm(dims.hidden_size, dims.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(dims.head_dim, dims.rope_theta)
        self.lm_head = nn.Linear(dims.hidden_size, dims.vocab_size, bias=False)
        if dims.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Dense prefill. input_ids [batch, seq] -> logits [batch, seq, vocab]."""
        if input_ids.ndim != 2 or input_ids.shape[1] < 1:
            raise ValueError("input_ids must be [batch, seq] with seq >= 1")
        hidden = self.embed_tokens(input_ids)
        position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        cos, sin = self.rotary_emb(hidden, position_ids)
        for layer in self.layers:
            hidden = layer(hidden, cos, sin)
        return self.lm_head(self.norm(hidden))

    @classmethod
    def from_hf(cls, hf_model) -> Qwen2CausalLM:
        """Copy weights out of a HuggingFace Qwen2ForCausalLM, then the caller can drop it."""
        dims = Qwen2Dims.from_hf_config(hf_model.config)
        weight = hf_model.model.embed_tokens.weight
        model = cls(dims).to(device=weight.device, dtype=weight.dtype)
        _copy_embed(model.embed_tokens, hf_model.model.embed_tokens)
        if len(hf_model.model.layers) != dims.num_hidden_layers:
            raise ValueError("HF layer count does not match config")
        for ours, theirs in zip(model.layers, hf_model.model.layers, strict=True):
            _copy_attn(ours.self_attn, theirs.self_attn)
            _copy_mlp(ours.mlp, theirs.mlp)
            _copy_norm(ours.input_layernorm, theirs.input_layernorm)
            _copy_norm(ours.post_attention_layernorm, theirs.post_attention_layernorm)
        _copy_norm(model.norm, hf_model.model.norm)
        if dims.tie_word_embeddings:
            model.lm_head.weight = model.embed_tokens.weight
        else:
            _copy_linear(model.lm_head, hf_model.lm_head)
        model.eval()
        _log.info(
            "qwen2 copied layers=%s heads=%s kv_heads=%s head_dim=%s tied=%s",
            dims.num_hidden_layers,
            dims.num_attention_heads,
            dims.num_key_value_heads,
            dims.head_dim,
            dims.tie_word_embeddings,
        )
        return model


def _rope_theta(config) -> float:
    params = getattr(config, "rope_parameters", None) or {}
    if isinstance(params, dict) and params.get("rope_type", "default") != "default":
        raise ValueError(f"unsupported RoPE type {params.get('rope_type')}")
    if isinstance(params, dict) and "rope_theta" in params:
        return float(params["rope_theta"])
    return float(config.rope_theta)


def _copy_linear(dst: nn.Linear, src: nn.Linear) -> None:
    if tuple(dst.weight.shape) != tuple(src.weight.shape):
        raise ValueError(f"weight shape {tuple(src.weight.shape)} != {tuple(dst.weight.shape)}")
    if (dst.bias is None) != (src.bias is None):
        raise ValueError("bias presence does not match the HuggingFace module")
    dst.weight.data.copy_(src.weight.detach())
    if dst.bias is not None and src.bias is not None:
        dst.bias.data.copy_(src.bias.detach())


def _copy_embed(dst: nn.Embedding, src: nn.Embedding) -> None:
    if tuple(dst.weight.shape) != tuple(src.weight.shape):
        raise ValueError("embedding shape does not match")
    dst.weight.data.copy_(src.weight.detach())


def _copy_norm(dst: RMSNorm, src: nn.Module) -> None:
    dst.weight.data.copy_(src.weight.detach())


def _copy_attn(dst: Attention, src: nn.Module) -> None:
    _copy_linear(dst.q_proj, src.q_proj)
    _copy_linear(dst.k_proj, src.k_proj)
    _copy_linear(dst.v_proj, src.v_proj)
    _copy_linear(dst.o_proj, src.o_proj)


def _copy_mlp(dst: MLP, src: nn.Module) -> None:
    _copy_linear(dst.gate_proj, src.gate_proj)
    _copy_linear(dst.up_proj, src.up_proj)
    _copy_linear(dst.down_proj, src.down_proj)
