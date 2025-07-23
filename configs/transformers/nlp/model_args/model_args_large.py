"""
Medium configuration of model arguments.

This configuration contains:
    - 8 billion parameters without MoE.
"""

from typing import Tuple
from dataclasses import dataclass

from configs.transformers.nlp.training_args import TrainingArgs
training_args = TrainingArgs()

@dataclass
class ModelArgs:
    """Large configuration of model arguments."""
    d_model: int = 4096
    num_heads: int = 32
    query_groups: int = 8
    d_ffn: int = 14336
    num_layers: int = 32 
    dropout: float = 0.2
    rope_base: float = 10000.0
    rms_norm_eps: float = 1e-7
    window_size: Tuple[int, int] = (512, 0)
    vocab_size: int = 65536
    max_seq_len: int = 32768
    tie_weights: bool = True
    max_batch_size: int = 2048
    gradient_checkpointing: bool = True
    num_experts: int = 32
    top_k: int = 2

    def __post_init__(self):
        """Post initialization for assertions."""
        if self.d_model % self.num_heads != 0:
            raise ValueError(f"Expected d_model to be divisble by num_heads, got {self.d_model} % {self.num_heads} != 0")
        if self.num_heads % self.query_groups != 0:
            raise ValueError(f"Expected d_model to be divisble by num_heads, got {self.num_heads} % {self.query_groups} != 0")
        if self.d_model * 4 != self.d_ffn:
            raise ValueError(f"Expected d_model * 4 = d_ffn, got {self.d_model} * 4 != {self.d_ffn}")
        if self.max_batch_size < training_args.batch_size:
            raise ValueError(f"Expected max_batch_size >= batch_size, got {training_args.batch_size} < {self.max_batch_size}")
        if self.num_experts < self.top_k:
            raise ValueError(f"Expected num_experts >= top_k, got {self.top_k} > {self.num_experts}")
        if len(self.window_size) != 2:
            raise ValueError(f"Expected len(window_size) == 2, got {len(self.window_size)} != 2")
        if self.window_size[0] <= 0:
            raise ValueError(f"Expected window_size[0] > 0, got {self.window_size[0]} <= 0")
        if self.window_size[1] != 0:
            raise ValueError(f"Expected window_size[1] == 0, got {self.window_size[1]} != 0")
        