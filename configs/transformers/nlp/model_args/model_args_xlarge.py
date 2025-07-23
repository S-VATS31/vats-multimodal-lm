"""
Extra large configuration of model arguments.

This configuration contains:
    - 13 billion parameters without MoE.
"""

from typing import Tuple
from dataclasses import dataclass

from configs.transformers.nlp.training_args import TrainingArgs
training_args = TrainingArgs()

@dataclass
class ModelArgs:
    """Extra large configuration of model arguments."""
    d_model: int = 5120
    num_heads: int = 40
    query_groups: int = 10
    d_ffn: int = 20480
    num_layers: int = 40
    dropout: float = 0.2
    rope_base: float = 10000.0
    rms_norm_eps: float = 1e-7
    window_size: Tuple[int, int] = (1024, 0)
    vocab_size: int = 65536
    max_seq_len: int = 32768
    tie_weights: bool = False
    gradient_checkpointing: bool = False
    num_experts: int = 64
    top_k: int = 2

    def __post_init__(self):
        """Post initialization for assertions."""
        if self.d_model % self.num_heads != 0:
            raise ValueError(f"d_model must be divisble by 0, got {self.d_model} / {self.num_heads} != 0")
        if self.num_heads % self.query_groups != 0:
            raise ValueError(f"d_model must be divisble by 0, got {self.num_heads} / {self.query_groups} != 0")
        if self.d_model * 4 != self.d_ffn:
            raise ValueError(f"d_model * 4 must be equal to d_ffn, got {self.d_model} * 4 != {self.d_ffn}")
        if self.max_batch_size < training_args.batch_size:
            raise ValueError(f"max_batch_size must be >= batch_size, got {training_args.batch_size} < {self.max_batch_size}")
        if self.num_experts < self.top_k:
            raise ValueError(f"num_experts must be >= top_k, got {self.top_k} > {self.num_experts}")