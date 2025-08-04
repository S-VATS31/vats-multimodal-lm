import math
from typing import Optional
from dataclasses import dataclass

@dataclass
class ModelArgs:
    """Extra small configuration of model arguments."""
    d_model: int = 256
    num_heads: int = 16
    query_groups: int = 2
    softmax_scale: Optional[float] = None
    d_ffn: int = 1024
    num_layers: int = 8
    dropout: float = 0.1
    rope_base: float = 10000.0
    rms_norm_eps: float = 1e-7
    left_window: int = 128
    right_window: int = 0
    vocab_size: int = 50257
    max_seq_len: int = 512
    tie_weights: bool = True
    max_batch_size: int = 2048
    gradient_checkpointing: bool = True
    use_proj_bias: bool = False
    use_qkv_proj: bool = True
    use_causal: bool = True
    use_mqa: bool = True
    use_cache: bool = False
    num_experts: int = 1
    top_k: int = 1

    def __post_init__(self):
        """Post initialization to set softmax scale dynamically."""
        if self.softmax_scale is None:
            self.softmax_scale = 1 / math.sqrt(self.d_model // self.num_heads)
