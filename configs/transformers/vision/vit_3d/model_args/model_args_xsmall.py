from typing import Tuple
from dataclasses import dataclass

from configs.transformers.vision.vit_3d.model_args.post_init import PostInitMixin

@dataclass
class ModelArgs(PostInitMixin):
    """Extra small configuration of model arguments, containing 241 million parameters."""
    patch_size: Tuple[int, int, int] = (2, 8, 8)
    target_size: Tuple[int, int] = (128, 128)
    max_frames: int = 32
    C_in: int = 3
    d_model: int = 240
    num_heads: int = 4
    query_groups: int = 2
    d_ffn: int = 960
    num_layers: int = 4
    window_size: Tuple[int, int] = (128, 128)
    dropout: float = 0.1
    rope_theta: float = 30000.0
    rms_norm_eps: float = 1e-7
    num_classes: int = 1000
    use_checkpointing: bool = False
    use_mqa: bool = False
    use_qk_norm: bool = True
