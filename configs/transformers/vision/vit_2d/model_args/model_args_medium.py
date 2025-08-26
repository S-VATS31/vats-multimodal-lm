import math
from dataclasses import dataclass
from typing import Optional

from configs.transformers.vision.vit_2d.model_args.post_init import PostInitMixin

@dataclass
class ModelArgs(PostInitMixin):
    """Medium configuration of model arguments, containing 320 million parameters."""
    target_size: int = 384
    patch_size: int = 16
    C_in: int = 3
    d_model: int = 1152
    num_heads: int = 16
    query_groups: int = 8
    softmax_scale: Optional[float] = None
    d_ffn: int = 4608
    num_layers: int = 16
    left_window: int = -1
    right_window: int = -1
    dropout: float = 0.2
    rope_theta: float = 30000.0
    rms_norm_eps: float = 1e-7
    use_checkpointing: bool = True
    use_windowed_attn: bool = True
    use_proj_bias: bool = False
    use_fused_proj: bool = True
    use_mqa: bool = False
    use_qk_norm: bool = True

    def __post_init__(self):
        """Post initialization for dynamic softmax scale."""
        if self.softmax_scale is None:
            self.softmax_scale = 1 / math.sqrt(self.d_model // self.num_heads)
            
        # Call assertions
        super().__init__()