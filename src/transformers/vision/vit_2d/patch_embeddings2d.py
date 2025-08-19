from configs.setup_env import device, dtype

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.amp import autocast

class PatchEmbeddings2D(nn.Module):
    """Patch Embeddings using Conv2d for Vision Transformers.

    Args:
        patch_size (int): Height and width of each patch (assumed square).
        target_size (int): Final square size after resize + center crop.
        C_in (int): Number of input channels.
        d_model (int): Dimension of output embeddings.
    """
    def __init__(
        self, 
        patch_size: int,
        target_size: int,
        C_in: int, 
        d_model: int
    ):
        super().__init__()

        self.patch_size = patch_size
        self.target_size = target_size
        self.num_patches = (target_size // patch_size) ** 2
        self.d_model = d_model

        # Patch projection
        self.proj = nn.Conv2d(
            in_channels=C_in, 
            out_channels=d_model, 
            kernel_size=patch_size, 
            stride=patch_size
        )

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
            B, C, H, W = x.shape

            # Assert input channels
            assert (
                C == self.proj.in_channels
            ), f"Expected {self.proj.in_channels} channels, got {C}"

            # Resize proportionally
            short_side = min(H, W)
            scale = self.target_size / short_side
            new_H = int(round(H * scale))
            new_W = int(round(W * scale))

            # Resize
            x = F.interpolate(
                x,
                size=(new_H, new_W), 
                mode="bilinear", 
                align_corners=False
            )

            # Center crop to final square size
            x = TF.center_crop(x, output_size=(self.target_size, self.target_size))

            # Project patches
            x = self.proj(x) # [B, d_model, H_patch, W_patch]

            # Flatten and transpose
            x = x.view(B, self.d_model, -1) # [B, d_model, num_patches]
            assert (
                x.shape == (B, self.d_model, self.num_patches)
            ), f"x must have shape {(B, self.d_model, self.num_patches)}, got {x.shape}"

            x = x.transpose(1, 2) # [B, num_patches, d_model]
            assert (
                x.shape == (B, self.num_patches, self.d_model)
            ), f"x must have shape {(B, self.num_patches, self.d_model)}, got {x.shape}"

        return x

def test_2d_patch_embed():
    patch_size, target_size = 16, 144
    C, d_model = 3, 512
    patch_embeddings = PatchEmbeddings2D(
        patch_size, target_size, C, d_model
    ).to(device)
    B, H, W = 1, 224, 224
    x = torch.randn(B, C, H, W).to(device)
    x_out = patch_embeddings(x)
    return x_out

if __name__ == "__main__":
    x = test_2d_patch_embed()
    # new_H, new_W = 144, 144
    # x.size(2) == 144//16 * 144//16
    # x.size(2) == 81
    print(x.shape) # [1, 81, 512]