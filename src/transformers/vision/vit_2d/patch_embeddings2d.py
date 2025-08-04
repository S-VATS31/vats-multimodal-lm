from configs.setup_env import device, dtype

import torch
import torch.nn as nn
from torch.amp import autocast

class PatchEmbeddings2D(nn.Module):
    """Patch Embeddings using Conv2d for Vision Transformers.

    Args:
        img_size (int): Height and width of the image (assumed square).
        patch_size (int): Height and width of each patch (assumed square).
        C_in (int): Number of input channels.
        d_model (int): Dimension of output embeddings.
    """
    def __init__(self, img_size: int, patch_size: int, C_in: int, d_model: int):
        super().__init__()

        if img_size % patch_size != 0:
            raise ValueError("img_size must be divisible by patch_size")
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.d_model = d_model

        # Patch projection
        self.proj = nn.Conv2d(C_in, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with patch projection.

        Args:
            x (torch.Tensor): [B, C, H, W] input image tensor.

        Returns:
            torch.Tensor: [B, num_patches, d_model].
        """
        with autocast(device_type=device.type, dtype=dtype):
            assert (
                x.dim() == 4
            ), f"x must have 4 dimensions, got {x.dim()} dimensions"

            # Get height/width of image
            B, _, H, W = x.shape
            if H != self.img_size or W != self.img_size:
                raise ValueError(f"Expected input of size {self.img_size}x{self.img_size}, got {H}x{W}")
            
            # Project patches
            x = self.proj(x) # [B, d_model, H/P, W/P]
            assert (
                x.shape == (B, self.d_model, H / self.patch_size, W / self.patch_size)
            ), f"x must have shape of {(B, self.d_model, H / self.patch_size, W / self.patch_size)}, got {x.shape}"

            x = x.flatten(2) # [B, d_model, num_patches]
            assert (
                x.shape == (B, self.d_model, self.num_patches)
            ), f"x must have shape of {(B, self.d_model, self.num_patches)}, got {x.shape}"

            x = x.transpose(1, 2) # [B, num_patches, d_model]
            assert (
                x.shape == (B, self.num_patches, self.d_model)
            ), f"x must have shape of {(B, self.num_patches, self.d_model)}, got {x.shape}"

        return x
