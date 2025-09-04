from configs.setup_env import device, dtype

from typing import Tuple

import torch
import torch.nn as nn
from torch.amp import autocast

class Decoder3D(nn.Module):
    """Decoder for 3D video generation, reverse mirror Encoder3D.
    
    Args:
        d_model (int): Dimensionality of model embeddings.
        C_out (int): Number of output channels.
        patch_size (Tuple[int, int, int]): T, H, W patches.
    """
    def __init__(
        self, 
        d_model: int, 
        C_out: int, 
        patch_size: Tuple[int, int, int]
    ):
        super().__init__()

        self.d_model = d_model
        self.C_out = C_out
        self.patch_size = patch_size

        self.deconv1 = nn.ConvTranspose3d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm3d(d_model)

        self.deconv2 = nn.ConvTranspose3d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm3d(d_model)

        self.deconv3 = nn.ConvTranspose3d(
            in_channels=d_model,
            out_channels=C_out,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False
        )

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of 3D Encoder layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, T_frames, H, W, d_model].
        
        Returns:
            torch.Tensor: Output tensor of shape [B, C_out, T_frames, H, W].
        """
        with autocast(device_type=device.type, dtype=dtype):
            assert x.dim() == 5, f"expected 5 dims, got {x.dim()} dims"
            assert (
                x.size(-1) == self.d_model
            ), f"expected {self.d_model}, got {x.size(-1)}"

            B, T, H, W, _ = x.shape

            # Reshape to [B, d_model, T_frames, H, W]
            x = x.permute(0, -1, 1, -3, -2).contiguous()

            assert (
                x.shape == (B, self.d_model, T, H, W)
            ), f"expected {(B, self.d_model, T, H, W)}, got {x.shape}"

            # [B, C_out, new_T, new_H, new_W]
            x = self.relu(self.bn1(self.deconv1(x)))
            x = self.relu(self.bn2(self.deconv2(x)))
            x = self.sigmoid(self.deconv3(x))

            assert (
                x.size(1) == self.C_out
            ), f"expected {self.C_out}, got {x.size(1)}"

            return x
        
def test_decoder():
    d_model, C_out = 512, 3
    patch_size = (2, 8, 8)
    decoder = Decoder3D(d_model, C_out, patch_size).to(device)
    B, T_frames, H, W = 1, 8, 16, 16
    x = torch.randn(B, T_frames, H, W, d_model).to(device)
    x_out = decoder(x)
    return x_out

if __name__ == "__main__":
    x = test_decoder()
    print(x.shape)
