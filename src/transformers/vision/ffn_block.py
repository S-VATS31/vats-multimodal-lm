from configs.setup_env import device, dtype

import torch
import torch.nn as nn
from torch.amp import autocast

from src.rms_norm import RMSNorm
from src.swiglu_activation import SwiGLUActivation

class FFNBlock(nn.Module):
    """FFN block with a pass through the FFN, dropout, normalization, and residuals applied.
    
    Args:
        d_model (int): Dimensionality of the model's embeddings.
        d_ffn (int): Dimensionality of the FFN.
        dropout (float): Dropout probability.
        eps (float): Small epsilon value to prevent numerical instability.
    """
    def __init__(self, d_model: int, d_ffn: int, dropout: float, eps: float):
        super().__init__()

        self.gated_ffn = SwiGLUActivation(d_model, d_ffn, dropout)
        self.rms_norm = RMSNorm(d_model, eps)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform forward pass of the FFN layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, N, d_model].

        Returns:
            torch.Tensor: Output tensor with the same shape.
        """
        with autocast(device_type=device.type, dtype=dtype):
            return self.dropout(self.gated_ffn(self.rms_norm(x)))
