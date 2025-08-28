from typing import Optional
from dataclasses import dataclass
import math

from configs.autoregressive_image_gen.autoregressive_transformer.model_args.post_init import PostInitMixin

# TODO: update and calculate parameters

@dataclass
class ModelArgs(PostInitMixin):
    """Small configuration of model arguments."""
    d_model: int = 512
    num_heads: int = 8
    query_groups: int = 4
    max_batch_size: int = 32
    softmax_scale: Optional[float] = None
    d_ffn: int = 2048
    num_layers: int = 12
    dropout: float = 0.2
    rope_theta: float = 10000.0
    use_ntk_rope: bool = True
    ntk_scale_factor: float = 0.7
    left_window: int = 512
    right_window: int = 0
    rms_norm_eps: float = 1e-12
    vocab_size: int = 65536
    max_position_embeddings: int = 2048
    use_checkpointing: bool = True
    use_proj_bias: bool = False
    use_qkv_proj: bool = True
    enable_mqa: bool = True
    use_qk_norm: bool = True
    use_causal: bool = True
    use_windowed_attn: bool = True

    def __post_init__(self):
        """Post-init to calculate softmax scale dynamically."""
        if self.softmax_scale is None:
            self.softmax_scale = 1 / math.sqrt(self.d_model // self.num_heads)

        # Call assertions
        super().__post_init__()
        