from configs.setup_env import (
    device,
    dtype,
    gpu_dtypes,
    use_flash_attn,
    flash_attn_varlen_qkvpacked_func
)

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

from src.autoregressive_image_gen.autoregressive_transformer.attention.cross_attention import CrossAttention

class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()

        pass

    def _optimized_attention(self) -> torch.Tensor:
        pass

    def _torch_attention(self) -> torch.Tensor:
        pass

    def _causal_self_attention(self) -> torch.Tensor:
        pass

    