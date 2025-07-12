"""
Small configuration of model arguments.

This configuration contains:
    - 379 million parameters.

This configuration is too small for MoE, set: top_k, num_experts = 1, 1.
"""


from dataclasses import dataclass

@dataclass
class ModelArgs:
    """Small configuration of model arguments."""
    d_model: int = 512
    num_heads: int = 8
    query_groups: int = 4
    d_ffn: int = 2048  # 4 * d_model
    num_layers: int = 12
    dropout: float = 0.2
    rope_base: float = 10000.0
    rms_norm_eps: float = 1e-7
    vocab_size: int = 65536
    max_seq_len: int = 2048
    tie_weights: bool = False
    pad_token_id: int = 0
    eos_token_id: int = 65535
    max_batch_size: int = 2048
    gradient_checkpointing: bool = True
    num_experts: int = 8
    top_k: int = 2

    # TODO: add post_init for assertions