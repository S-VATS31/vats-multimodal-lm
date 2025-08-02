from typing import Tuple
from dataclasses import dataclass

from configs.transformers.vision.vit_3d.model_args.post_init import PostInitMixin


@dataclass
class ModelArgs(PostInitMixin):
    """Small configuration of model arguments, containing 321 million parameters."""
    patch_size: Tuple[int, int, int] = (2, 16, 16)
    target_size: Tuple[int, int] = (256, 256)
    max_frames: int = 200
    C_in: int = 3
    d_model: int = 1152
    num_heads: int = 16
    query_groups: int = 8
    d_ffn: int = 4608
    num_layers: int = 16
    window_size: Tuple[int, int] = (384, 384)
    dropout: float = 0.2
    rope_theta: float = 30000.0
    rms_norm_eps: float = 1e-7
    num_classes: int = 1000
    use_checkpointing: bool = True
