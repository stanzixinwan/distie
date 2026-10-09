"""In-house Qwen2 forward. Weights come from a HuggingFace checkpoint."""

from inference.qwen2.model import Qwen2CausalLM

__all__ = ["Qwen2CausalLM"]
