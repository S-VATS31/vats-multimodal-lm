from typing import Optional
from dataclasses import dataclass

# TODO: update and calculate parameters

@dataclass
class ModelArgs:
    """Small configuration of model arguments."""
    d_model: int = 512
    num_heads: int = 8
    query_groups: int = 4
    softmax_scale: Optional[float] = None
    d_ffn: int = 2048
    num_layers: int = 12
    dropout: float = 0.2
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-7
    left_window: int = 256
    right_window: int = 0
    vocab_size: int = 65536
    max_seq_len: int = 2048
    use_weight_tying: bool = False
    use_checkpointing: bool = True
    use_proj_bias: bool = False
    use_qkv_proj: bool = True
    use_causal: bool = True
    enable_mqa: bool = True
    use_cache: bool = False
    use_diffusion: bool = True
