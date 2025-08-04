import math
from typing import Optional
from dataclasses import dataclass

@dataclass
class ModelArgs:
    """Extra large configuration of model arguments."""
    d_model: int = 5120
    num_heads: int = 40
    query_groups: int = 10
    softmax_scale: Optional[float] = None
    d_ffn: int = 20480
    num_layers: int = 40
    dropout: float = 0.2
    rope_base: float = 10000.0
    rms_norm_eps: float = 1e-7
    left_window: int = 1024
    right_window: int = 0
    vocab_size: int = 65536
    max_seq_len: int = 32768
    tie_weights: bool = False
    gradient_checkpointing: bool = False
    use_proj_bias: bool = False
    use_qkv_proj: bool = True
    use_causal: bool = True
    use_mqa: bool = True
    use_cache: bool = False
    max_batch_size: int = 2048
    num_experts: int = 64
    top_k: int = 2

    def __post_init__(self):
        """Post initialization to set softmax scale dynamically."""
        if self.softmax_scale is None:
            self.softmax_scale = 1 / math.sqrt(self.d_model // self.num_heads)
