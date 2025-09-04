from configs.setup_env import device, dtype

from typing import Tuple

import torch
import torch.nn as nn
from torch.amp import autocast


class Encoder3D(nn.Module):
    def __init__(
        self,
        d_model: int,
        C_in: int,
        patch_size: Tuple[int, int, int]
    ):
        super().__init__()

        self.d_model = d_model
        self.C_in = C_in
        self.patch_size = patch_size

        self.conv1 = nn.Conv3d(
            in_channels=C_in,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False
        )
        self.bn1 = nn.BatchNorm3d(d_model)
        
        self.conv2 = nn.Conv3d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm3d(d_model)

        self.conv3 = nn.Conv3d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn3 = nn.BatchNorm3d(d_model)

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of VQ VAE encoder layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, T_frames, H, W].
        
        Returns:
            torch.Tensor: Output tensor of shape [B, T_frames, H*W, d_model].
        """
        with autocast(device_type=device.type, dtype=dtype):
            assert x.dim() == 5, f"expected 5 dims, got {x.dim()} dims"
            assert (
                x.size(1) == self.C_in
            ), f"expected {self.C_in}, got {x.size(1)}"
            assert (
                len(self.patch_size) == 3
            ), f"expected 3, got {len(self.patch_size)}"

            pt, ph, pw = self.patch_size

            assert (
                x.size(2) % pt == 0
            ), f"{x.size(2)} % {pt} must be 0"
            assert (
                x.size(-2) % ph == 0
            ), f"{x.size(-2)} % {ph} must be 0"
            assert (
                x.size(-1) % pw == 0
            ), f"{x.size(-1)} % {pw} must be 0"

            # [B, d_model, T_frames, H, W]
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.relu(self.bn3(self.conv3(x)))

            B, _, new_T, new_H, new_W = x.shape

            assert (
                x.size(1) == self.d_model
            ), f"expected {self.d_model}, got {x.size(1)}"

            # Reshape to [B, T_frames, H, W, d_model]
            x = x.permute(0, 2, -2, -1, 1).contiguous().view(B, new_T, -1, self.d_model)

            assert (
                x.shape == (B, new_T, new_H*new_W, self.d_model)
            ), f"expeced {(B, new_T, new_H*new_W, self.d_model)}, got {x.shape}"

            return x
        
def test_encoder():
    d_model, C_in = 512, 3
    patch_size = (2, 8, 8)
    encoder = Encoder3D(d_model, C_in, patch_size).to(device)
    B, T_frames, H, W = 1, 16, 32, 32
    x = torch.randn(B, C_in, T_frames, H, W).to(device)
    x_out = encoder(x)
    return x_out

if __name__ == "__main__":
    x_out = test_encoder()
    print(x_out.shape)
