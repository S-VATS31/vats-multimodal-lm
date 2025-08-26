from configs.setup_env import device, dtype

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

class RMSNorm(nn.Module):
    """Apply RMSNorm to the features dimension.

    Formula:
        x_norm = (x / sqrt(mean(x**2))) * self.weight
    
    Args:
        d_model (int): Dimensionality of the model's embeddings.
        eps (float): Small epsilon value to prevent numerical stability.
    """
    def __init__(self, d_model: int, eps: float):
        super().__init__()

        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform forward pass of RMSNorm layer.

        Args:
            x (torch.Tensor): Input tensor of shape [B, T, d_model].    
        """
        with autocast(device_type=device.type, dtype=dtype):
            return self.weight * (
                F.normalize(x, p=2, dim=-1)
            )
        