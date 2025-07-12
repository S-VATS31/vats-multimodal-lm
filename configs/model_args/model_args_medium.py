"""
Medium configuration of model arguments.

This configuration contains ~8 billion parameters.

Note that the training arguments will need to updated 
based on different ModelArgs hyperparameter values for better
training and convergence.

NOTE: To essentially "turn off" MoE, set num_experts=1.
"""

from dataclasses import dataclass

@dataclass
class ModelArgs:
    """
    Dataclass containing model hyperparameters.
    """
    d_model: int = 1440
    num_heads: int = 24
    query_groups: int = 12
    d_ffn: int = 5760
    num_layers: int = 20
    dropout: float = 0.2
    rope_base: float = 10000.0
    rms_norm_eps: float = 1e-7
    vocab_size: int = 65536
    max_seq_len: int = 2048
    tie_weights: bool = False
    pad_token_id: int = 0
    eos_token_id: int = 65535
    gradient_checkpointing: bool = True
    num_experts: int = 1
    top_k: int = 2

    def __post_init__(self):
        """Post initialization for assertions."""
        assert self.d_model % self.num_heads == 0
        assert self.num_heads % self.query_groups == 0
        assert self.d_ffn == self.d_model * 4
