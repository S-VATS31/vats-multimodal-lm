from configs.setup_env import device, dtype

from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch.amp import autocast

class NTKRoPE2D(nn.Module):
    """2D NTK RoPE layer for robust positional embeddings.
    
    Args:
        head_dim (int): Dimensionality of each attention head.
        rope_theta (float): Exponential base of inv freq.
        ntk_scale_factor (float): NTK scaling factor multipled by inv freq.
        max_position_embeddings (int): Max sequence length if not using NTK RoPE.
    """
    def __init__(
        self,
        head_dim: int,
        rope_theta: float,
        ntk_scale_factor: Optional[float] = None,
        max_position_embeddings: Optional[int] = None
    ):
        super().__init__()

        pass
