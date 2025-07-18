"""
Small configuration of model arguments.

This configuration contains:
    - 379 million parameters.

This configuration is too small for MoE, set: top_k, num_experts = 1, 1.
"""


from dataclasses import dataclass

from configs.transformers.nlp.training_args import TrainingArgs
training_args = TrainingArgs

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