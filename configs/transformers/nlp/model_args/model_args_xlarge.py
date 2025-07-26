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
    """Extra large configuration of model arguments.
    
    Args:
        d_model (int): Dimensionality of the model's embeddings.
        num_heads (int): Number of attention heads.
        query_groups (int): Number of query groups for GQA.
        d_ffn (int): Dimensionality of the FFN. Typically, d_ffn = 4 * d_model.
        num_layers (int): Number of layers being stacked.
        dropout (float): Dropout probability.
        rope_base (float): Theta hyperparameter for RoPE.
        rms_norm_eps (float): Epsilon value to prevent numerical instability in RMSNorm.
        window_size (Tuple[int, int]): Window size for sliding window attention.
        vocab_size (int): Unique tokens that the model can recognize.
        max_seq_len (int): Largest length (in tokens) that can be inputted.
        tie_weights (bool): Whether to tie weights or not.
        max_batch_size (int): Max batch size hyperparameter for KV cache.
        gradient_checkpointing (bool): Whether to use gradient checkpointing or not.
        num_experts (int): Number of experts for MoE.
        top_k (int): Top-k routing for MoE.
    """
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
        