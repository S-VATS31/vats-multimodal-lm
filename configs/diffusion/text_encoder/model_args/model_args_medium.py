import math
from typing import Optional
from dataclasses import dataclass

@dataclass
class ModelArgs:
    """Medium configuration of model arguments."""
    d_model: int = 1440
    num_heads: int = 24
    query_groups: int = 8
    softmax_scale: Optional[float] = None
    d_ffn: int = 5760
    num_layers: int = 20
    dropout: float = 0.2
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-7
    left_window: int = 384
    right_window: int = 0
    vocab_size: int = 65536
    max_seq_len: int = 2048
    use_checkpointing: bool = True
    use_proj_bias: bool = False
    use_qkv_proj: bool = True
    use_diffusion: bool = True
    enable_mqa: bool = False
    max_batch_size: int = 1024

    # def __post_init__(self):
    #     """Post initialization to set softmax scale dynamically."""
    #     if self.softmax_scale is None:
    #         self.softmax_scale = 1 / math.sqrt(self.d_model // self.num_heads)

    #     # Call assertions
    #     super().__post_init__()
