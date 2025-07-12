"""
Medium configuration of model arguments.

This configuration contains:
    - 52 million parameters.

This configuration is too small for MoE, set: top_k, num_experts = 1, 1.
"""

from dataclasses import dataclass

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
    vocab_size: int = 50257
    max_seq_len: int = 512
    tie_weights: bool = True
    pad_token_id: int = 0
    eos_token_id: int = 8191
    max_batch_size: int = 2048
    gradient_checkpointing: bool = True
    num_experts: int = 4
    top_k: int = 2
    
    # TODO: Add post_init for assertions