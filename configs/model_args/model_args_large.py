"""
Medium configuration of model arguments.

This configuration contains:
    - 8 billion parameters without MoE.
"""

from dataclasses import dataclass

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
    vocab_size: int = 65536
    max_seq_len: int = 32768
    tie_weights: bool = True
    pad_token_id: int = 0
    eos_token_id: int = 65535
    max_batch_size: int = 2048
    gradient_checkpointing: bool = False
    num_experts: int = 32
    top_k: int = 2