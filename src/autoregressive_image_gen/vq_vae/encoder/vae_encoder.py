from configs.setup_env import device, dtype

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

class VQVAEEncoder(nn.Module):
    """Encoder layer using a convolutional neural network (CNN).
    
    Args:
        C_in (int): Number of input channels to the CNN.
        d_model (int): Dimensionality of model embeddings.
        activation_func (Literal["relu", "leaky_relu", "sigmoid"]): Non-linear activation.
    """
    def __init__(
        self, 
        C_in: int,
        d_model: int,
        activation_func: Literal["relu", "leaky_relu", "sigmoid"]
    ):
        super().__init__()

        # Block 1 -> [4, 2, 1]
        self.conv1 = nn.Conv2d(
            in_channels=C_in,
            out_channels=d_model,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self.batch_norm1 = nn.BatchNorm2d(d_model)

        # Block 2 -> [4, 2, 1]
        self.conv2 = nn.Conv2d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self.batch_norm2 = nn.BatchNorm2d(d_model)

        # Block 3 -> [3, 1, 1]
        self.conv3 = nn.Conv2d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.batch_norm3 = nn.BatchNorm2d(d_model)

        # Non-linear activation
        if activation_func == "relu":
            self.activation = nn.ReLU()
        elif activation_func == "leaky_relu":
            self.activation = nn.LeakyReLU()
        elif activation_func == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(
                f"expected ['relu', 'leaky_relu', 'sigmoid'], got {activation_func}"
            )
        
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of encoder layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C_in, H, W].

        Returns:
            torch.Tensor: Output tensor of shape [B, H, W, d_model].
        """
        with autocast(device_type=device.type, dtype=dtype):
            assert x.dim() == 4, f"expected 4 dims, got {x.dim()} dims"

            # Pass through blocks; shape: [B, d_model, H, W]
            x = self.activation(self.batch_norm1(self.conv1(x)))
            x = self.activation(self.batch_norm2(self.conv2(x)))
            x = self.activation(self.batch_norm3(self.conv3(x)))

            assert x.dim() == 4, f"expected 4 dims, got {x.dim()} dims"
            assert x.size(1) == self.d_model, f"expected {self.d_model}, got {x.size(1)}"

            # Reshape to [B, H, W, d_model]
            x = x.permute(0, -2, -1, 1).contiguous()

            assert x.dim() == 4, f"expected 4 dims, got {x.dim()} dims"
            assert x.size(-1) == self.d_model, f"expected {self.d_model}, got {x.size(-1)}"

            return x
        
def test_encoder():
    C_in, d_model = 3, 512
    encoder = VQVAEEncoder(
        C_in=C_in, d_model=d_model, activation_func="sigmoid"
    ).to(device)
    B, H, W = 1, 144, 144
    x = torch.randn(B, C_in, H, W).to(device)
    x_out = encoder(x)
    return x_out

if __name__ == "__main__":
    x = test_encoder()
    # to get H' and W' use
    # H' = floor_div((H+2p-k) / s) + 1
    # W' = floor_div((W+2p-k) / s) + 1
    # for all blocks
    # H' and W' should be 36 after calculations
    print(x.shape) # [1, 36, 36, 512]
