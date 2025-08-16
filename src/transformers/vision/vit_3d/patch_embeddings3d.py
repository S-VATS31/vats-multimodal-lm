from configs.setup_env import device, dtype

import warnings
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

class PatchEmbeddings3D(nn.Module):
    """Patch embeddings to split the video into 3D patches for spatiotemporal transformers.
    
    Args:
        C_in (int): Number of input channels.
        patch_size (Tuple[int, int, int]): Patch size in (T, H, W).
        target_size (Tuple[int, int]): Optimal video height and width: (H, W).
        max_frames (int): Maximum frames to train the ViT with. Used for padding/truncation.
        d_model (int): Dimensionality of the model's embeddings.
    """
    def __init__(
        self,
        C_in: int,
        patch_size: Tuple[int, int, int],
        target_size: Tuple[int, int],
        max_frames: int,
        d_model: int,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.d_model = d_model
        self.target_size = target_size
        self.max_frames = max_frames

        # Conv3D projection
        self.projection = nn.Conv3d(
            in_channels=C_in, # typically 3 for RGB
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False
        )

    def forward(
        self, 
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[int, int, int], torch.Tensor, Tuple[int, int, int]]:
        """Perform forward pass of Patch Embeddings 3D layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C_in, T, H, W].

        Returns:
            Tuple:
                - torch.Tensor: Patch embeddings of shape [B, N, d_model].
                - Tuple[int, int, int]: Processed video shape (T, H, W).
                - torch.Tensor: Padding mask of shape [B, N].
                - Tuple[int, int, int]: Grid size as (grid_t, grid_h, grid_w)
        """
        with autocast(device_type=device.type, dtype=dtype):
            assert(
                x.dim() == 5
            ), f"x must be a 5 dimenional tensor, got {x.dim()} dimensions"
            B, C, T, H, W = x.shape
            assert (
                len(self.patch_size) == 3
            ), f"len(patch_size) must be equal to 3, got {len(self.patch_size)} elements."
            pt, ph, pw = self.patch_size

            # Reshape input video over the height and width pixels, H and W
            x = F.interpolate(
                x.transpose(1, 2).contiguous().view(B * T, C, H, W), 
                size=self.target_size, 
                mode='bilinear',
                align_corners=False
            ).view(B, C, T, *self.target_size) # [B, C, T, target_size[0], target_size[1]]

            assert (
                x.size(-2) == self.target_size[0] and x.size(-1) == self.target_size[1]
            ), (
                f"x.size(-2) must be {self.target_size[0]}, got {x.size(-2)} "
                f"x.size(-1) must be {self.target_size[1]}, got {x.size(-1)}."
            )

            # Construct padding mask
            if padding_mask is None:
                padding_mask = torch.ones(
                    (B, T), dtype=torch.bool
                ).to(device) # Assume all frames are valid to start

                assert (
                    padding_mask.shape == (B, T)
                ), f"padding_mask must have shape of {(B, T)}, got {padding_mask.shape}"
                assert (
                    padding_mask.dtype == torch.bool
                ), f"padding_mask must be a boolean tensor, got {padding_mask.dtype}"
                assert (
                    torch.all(padding_mask == True)
                ), "All positions must start True."
            
            assert (
                padding_mask is not None
            ), "padding mask should not be None at this point."

            # Apply padding or truncation based on input frames, T
            if T < self.max_frames:
                frames_to_pad = self.max_frames - T
                x = F.pad(
                    x, (0, 0, 0, 0, 0, frames_to_pad), 
                    mode="constant", value=0
                ) # Pad over end of time dimension to fill video frames
                # Concatenate frames to pad with padding mask as padded positions
                pad_frames = torch.zeros((B, frames_to_pad), dtype=padding_mask.dtype).to(padding_mask.device)
                assert (
                    torch.all(pad_frames == False)
                ), "All padding frames must be False."
                assert (
                    pad_frames.dtype == padding_mask.dtype
                ), "pad_frames and padding_mask must have the same dtype"
                assert (
                    pad_frames.device.type == padding_mask.device.type
                ), "pad_frames and padding_mask must have the same device"

                padding_mask = torch.cat([padding_mask, pad_frames], dim=1) # [B, max_frames]
                assert (
                    padding_mask.size(1) == self.max_frames
                ), f"padding_mask.size(1) must be {self.max_frames}, got {padding_mask.size(1)}"

            elif T > self.max_frames:
                warnings.warn(
                    f"Maximum input frames allowed: {self.max_frames}, received: {T} frames "
                    f"Trucating {T - self.max_frames} frames."
                )
                # [B, C, max_frames, new_H, new_W]
                x = x[:, :, :self.max_frames] # Truncate over time in frames dimension
                assert (
                    x.shape == (B, C, self.max_frames, *self.target_size)
                ), (
                    f"x must have shape of {(B, C, self.max_frames, *self.target_size)}, got {x.shape}"
                )

                padding_mask = padding_mask[:, :self.max_frames] # [B, max_frames]
                assert(
                    padding_mask.shape == (B, self.max_frames)
                ), f"padding_mask must have shape of {(B, self.max_frames)}, got {padding_mask.shape}"

            # Store processed dimensions for dynamic grid computation later
            processed_T = x.size(2)
            processed_H, processed_W = x.size(3), x.size(4)
            processed_shape = (processed_T, processed_H, processed_W)

            # Compute total number of patches
            grid_t = processed_T // pt
            grid_h = processed_H // ph
            grid_w = processed_W // pw
            grid_size = (grid_t, grid_h, grid_w)
            N = grid_t * grid_h * grid_w # USE FOR PATCH MASK, NOT FOR OUT TENSOR

            # Project patch embeddings
            x = self.projection(x) # [B, d_model, T, H, W]; C_in (in) -> d_model (out)
            assert (
                x.size(1) == self.d_model
            ), f"x.size(1) must {self.d_model}, got {x.size(1)}"

            # Convert frame level mask (T) to patch level (N)
            # Instead of checking frame by frame validity, we check patch by patch
            # max_pool1d expects float tensor
            frame_mask = padding_mask[:, :processed_T].unsqueeze(1).float() # [B, 1, T]
            assert (
                frame_mask.size(1) == 1
            ), f"frame_mask.size(1) must be 1, got {frame_mask.size(1)}"

            pooled = (
                F.max_pool1d(frame_mask, kernel_size=pt, stride=pt, ceil_mode=True) # gracefully rounds, no truncation
                .squeeze(1)
                .bool()
            ) # [B, grid_t]
            assert (
                pooled.shape == (B, grid_t)
            ), f"pooled must have shape of {(B, grid_t)}, got {pooled.shape}"
            assert (
                pooled.dtype == torch.bool
            ), f"pooled must be a boolean tensor, got {pooled.dtype}"

            # Create patch mask of shape [B, N]
            patch_mask = (
                pooled[:, :, None, None] # [B, grid_t, 1, 1]; need singleton dimensions to expand
                .expand(B, grid_t, grid_h, grid_w) # expand returns non-contiguous tensor
                .contiguous()
                .view(B, N)
            )
            assert (
                patch_mask.shape == (B, N)
            ), f"patch_mask must have shape of {(B, N)}, got {patch_mask.shape}"
            
            assert(
                x.size(1) == self.d_model
            ), f"x.size(1) must be {self.d_model}, got {x.size(1)}"
            assert(
                x.dim() == 5
            ), f"x must be a 5 dimenional tensor, got {x.dim()} dimensions"

            x = x.view(B, grid_t, -1, self.d_model) # [B, grid_T, grid_H*grid_W, d_model]
            assert (
                x.shape == (B, grid_t, grid_h*grid_w, self.d_model) # [B, grid_T, grid_H * grid_W, d_model]
            ), f"x must have shape of {(B, grid_t, grid_h*grid_w, self.d_model)}, got {x.shape}"

            return x, processed_shape, patch_mask, grid_size

def main() -> torch.Tensor:
    C_in, d_model = 3, 512
    patch_size = (2, 32, 32)
    target_size = (384, 384)
    max_frames = 10
    patch_embeddings = PatchEmbeddings3D(
        C_in, patch_size, target_size, 
        max_frames, d_model
    ).to(device)
    B, T, H, W = 4, 24, 512, 512
    x = torch.randn(B, C_in, T, H, W).to(device)
    x_out, processed_shape, patch_mask, grid_size = patch_embeddings(x)
    return x_out, processed_shape, patch_mask, grid_size

if __name__ == "__main__":
    x, processed_shape, patch_mask, grid_size = main()
    # [B, grid_T, grid_H * grid_W, d_model]
    # B = 4
    # x_interpolate shape: [4, 3, 24, 384, 384]
    # x_trunc = [4, 3, 10, 384, 384]
    # processed_T = x.size(2) = 10
    # processed_H = x.size(3) = 384
    # processed_W = x.size(4) = 384
    # grid_T = 10 // 2 = 5
    # grid_H = 384 // 32 = 12
    # grid_W = 384 // 32 = 12
    # out.shape: [B, grid_T, grid_H*grid_W, d_model]
    # out.shape = [4, 5, 144, 512]
    # patch_mask.shape: [B, grid_T*grid_H*grid_W]
    # patch_mask.shape = [4, 720]
    # processed_shape = (max_frames, target_size[0], target_size[1])
    print(x.shape)
    print(processed_shape)
    print(patch_mask.shape)
