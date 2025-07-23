"""
Medium configuration of model arguments.

This configuration contains:
    - 52 million parameters.

This configuration is too small for MoE, set: top_k, num_experts = 1, 1.
"""

from typing import Tuple
from dataclasses import dataclass

from configs.transformers.nlp.training_args import TrainingArgs
training_args = TrainingArgs()

@dataclass
class ModelArgs:
    """Extra small configuration of model arguments."""
    d_model: int = 256
    num_heads: int = 16
    query_groups: int = 2
    d_ffn: int = 1024
    num_layers: int = 8
    dropout: float = 0.1
    rope_base: float = 10000.0
    rms_norm_eps: float = 1e-7
    window_size: Tuple[int, int] = (128, 0)
    vocab_size: int = 50257
    max_seq_len: int = 512
    tie_weights: bool = True
    max_batch_size: int = 2048
    gradient_checkpointing: bool = True
    num_experts: int = 1
    top_k: int = 1

    def __post_init__(self):
        """Post initialization for assertions."""
        if self.d_model % self.num_heads != 0:
            raise ValueError(f"d_model must be divisble by 0, got {self.d_model} % {self.num_heads} != 0")
        if self.num_heads % self.query_groups != 0:
            raise ValueError(f"d_model must be divisble by 0, got {self.num_heads} % {self.query_groups} != 0")
        if self.d_model * 4 != self.d_ffn:
            raise ValueError(f"d_model * 4 must be equal to d_ffn, got {self.d_model} * 4 != {self.d_ffn}")
        if self.max_batch_size < training_args.batch_size:
            raise ValueError(f"max_batch_size must be >= batch_size, got {training_args.batch_size} < {self.max_batch_size}")
        if self.num_experts < self.top_k:
            raise ValueError(f"num_experts must be >= top_k, got {self.top_k} > {self.num_experts}")