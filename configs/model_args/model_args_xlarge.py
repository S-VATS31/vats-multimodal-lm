"""
Extra large configuration of model arguments.

This configuration contains:
    - X billion parameters with MoE
    - Y billion parameters without MoE

To "turn off" MoE, set num_experts=1 and top_k=1.
"""

from dataclasses import dataclass

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
    vocab_size: int = 65536
    max_seq_len: int = 32768
    tie_weights: bool = False
    pad_token_id: int = 0
    eos_token_id: int = 65535
    gradient_checkpointing: bool = False
    num_experts: int = 64
    top_k: int = 2