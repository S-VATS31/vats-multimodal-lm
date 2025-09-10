from configs.setup_env import device, dtype

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

class SwiGLUActivation(nn.Module):
    """SwiGLU expert layer.

    Args:
        d_model (int): Dimensionality of the model's embeddings.
        d_ffn (int): Dimensionality of the feed forward network.
        dropout (float): Probability of model components being dropped out.
    """
    def __init__(self, d_model: int, d_ffn: int, dropout: float):
        super().__init__()

        self.weight1 = nn.Linear(d_model, d_ffn, bias=False)
        self.weight2 = nn.Linear(d_model, d_ffn, bias=False)
        self.weight3 = nn.Linear(d_ffn, d_model, bias=False)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of SwiGLU layer.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor of same shape.
        """
        with autocast(device_type=device.type, dtype=dtype):
            return self.dropout(self.weight3(F.silu(self.weight1(x)) * self.weight2(x)))

def test_four_dim_input():
    d_model, dropout = 744, 0.15
    d_ffn = 4 * d_model
    swiglu_func = SwiGLUActivation(d_model, d_ffn, dropout).to(device)
    B, T, H, W = 1, 2, 144, 144
    x = torch.randn(B, T, H*W, d_model).to(device)
    x_out = swiglu_func(x)
    return x_out

if __name__ == "__main__":
    x = test_four_dim_input()
    print(x.shape) # [1, 2, 20736, 744]
