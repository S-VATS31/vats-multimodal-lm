from configs.setup_env import device

import torch
import torch.nn as nn
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
        with autocast(device_type=device.type, enabled=False):
            assert (
                x.size(-1) == self.d_model
            ), f"expected {self.d_model}, got {x.size(-1)}"
            return self.weight * (
                x / torch.sqrt(torch.mean(x**2, keepdim=True, dim=-1) + self.eps)
            )
        