from dataclasses import dataclass
from typing import Tuple

@dataclass
class ModelArgs:
    """Medium configuration of model arguments, containing 860 million parameters."""
    patch_size: Tuple[int, int, int] = (2, 16, 16)
    C_in: int = 3
    d_model: int = 1920
    num_heads: int = 32
    query_groups: int = 8
    d_ffn: int = 1920*4
    num_layers: int = 16
    window_size: Tuple[int, int] = (384, 384)
    dropout: float = 0.2
    rope_theta: float = 25000.0
    rms_norm_eps: float = 1e-7
    num_classes: int = 1000
    use_checkpointing: bool = True

    def __post_init__(self):
        """Post initialization for assertions."""
        if self.d_model % self.num_heads != 0:
            raise ValueError(f"Expected d_model to be divible by num_heads, got {self.d_model} % {self.num_heads} != 0.")
        if self.num_heads % self.query_groups != 0:
            raise ValueError(f"Expected num_heads to be divible by query_groups, got {self.num_heads} % {self.query_groups} != 0.")
        if self.d_model * 4 != self.d_ffn:
            raise ValueError(f"Expected d_ffn = d_model * 4, got {self.d_model} * 4 != {self.d_ffn}")
        if len(self.window_size) != 2:
            raise ValueError(f"Expected len(window_size) to be equal to 2, got {len(self.window_size)}")
        if self.window_size[0] != self.window_size[1]:
            raise ValueError(f"Expected left and right windows to be equal, got {self.window_size[0]} != {self.window_size[1]}")
        if len(self.patch_size) != 3:
            raise ValueError(f"Expected len(patch_size) == 3 for T, H, W dimensions, got {len(self.patch_size)} != 3.")
