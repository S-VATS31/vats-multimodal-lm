from configs.setup_env import device, dtype

from typing import Tuple, Literal

import torch
import torch.nn as nn
from torch.amp import autocast

class RoPE3D(nn.Module):
    """3D Rotary Position Embedding for spatiotemporal transformers.
    
    Args:
        head_dim (int): Dimension of each attention head.
        theta (float): Base frequency for RoPE.
        patch_size (Tuple[int, int, int]): Patch size in (T, H, W).
    """
    def __init__(
        self,
        head_dim: int,
        theta: float,
        patch_size: Tuple[int, int, int],
    ):
        super().__init__()
        
        # Ensure head_dim is divisble by 6 for time, height, and width dimensions.
        if head_dim % 6 != 0:
            raise ValueError(
                f"head_dim must be divisible by 6 for 3D RoPE (2 dims per spatial dimension), got {head_dim}"
            )

        self.head_dim = head_dim
        self.theta = theta
        self.patch_size = patch_size
        
        # Compute dim_per_axis
        self.dim_per_axis = head_dim // 3
        if self.dim_per_axis % 2 != 0:
            raise ValueError(
                f"head_dim // 3 must be even for proper rotation pairs, got head_dim={head_dim}, dim_per_axis={self.dim_per_axis}"
            )
        
        self._precompute_freqs()
        
    def _precompute_freqs(self) -> None:
        """Precompute inverse frequencies for time, height, and width axis' and store in non-learnable buffers."""
        # Initialize inv_freq list to store T, H, W inverse freqs
        freqs_per_dim = []

        # Compute inverse frequencies for T, H, W
        # inv_freq = 1 / (theta ^ (2i/d)) where d = dim_per_axis
        for i in range(3): # T, H, W
            num_pairs = self.dim_per_axis // 2
            freqs = 1.0 / (
                self.theta ** (
                    torch.arange(0, num_pairs, dtype=torch.float32) * 2.0 / self.dim_per_axis)
                )
            freqs_per_dim.append(freqs)

            assert (
                freqs_per_dim[i].dtype == torch.float32
            ), f"All inverse frequencies must have dtype of float32, got {freqs_per_dim[i].dtype}"
            assert (
                freqs_per_dim[i].shape == (self.head_dim // 6,)
            ), f"inv_freq must have shape of {(self.head_dim // 6,)}, got {freqs_per_dim[i].shape}"
        
        # Store inverse frequencies non-learnable buffers
        # No return, we can directly access buffers using self.freqs_t, ...
        self.register_buffer('freqs_t', freqs_per_dim[0])
        self.register_buffer('freqs_h', freqs_per_dim[1])
        self.register_buffer('freqs_w', freqs_per_dim[2])
        
    def _get_3d_grid_positions(
        self, 
        grid_t: int, 
        grid_h: int, 
        grid_w: int
    ) -> torch.Tensor:
        """Create a lookup table to compute rotation angles.
        
        Args:
            grid_t (int): Number of patches for time dimension.
            grid_h (int): Number of patches for height dimension.
            grid_w (int): Number of patches for width dimension.

        Returns:
            torch.Tensor: Lookup grid of shape [N, 3], where N = grid_t * grid_h * grid_w.
        """
        t_coords = torch.arange(grid_t).to(device)
        h_coords = torch.arange(grid_h).to(device)
        w_coords = torch.arange(grid_w).to(device)
        
        # Create lookup table and flatten
        t_grid, h_grid, w_grid = torch.meshgrid(t_coords, h_coords, w_coords, indexing='ij')
        # [N, 3] for number of patches (N = grid_t * grid_h * grid_w) and 3 for T, H, W dimensions
        return torch.stack([t_grid.flatten(), h_grid.flatten(), w_grid.flatten()], dim=-1) 
        
    def _apply_rotary_embedding_1d(
        self, 
        x: torch.Tensor, 
        positions: torch.Tensor, 
        freqs: torch.Tensor,
        start_dim: int
    ) -> torch.Tensor:
        """Rotate input vectors via complex multiplication.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, N, num_heads, head_dim].
            positions (torch.Tensor): Index tensor to compute rotation angles.
            freqs (torch.Tensor): Inverse frequencies for T, H, W dimensions.
            start_dim (int): Start dimension based on splitting of `head_dim`.

        Returns:
            torch.Tensor: Rotated input tensor.
        """
        assert (
            x.dim() == 4
        ), f"x must be a 4 dimensional tensor, got {x.dim()}"
        B, N, num_heads, _ = x.shape

        # Get number of pairs and compute end dimension
        num_pairs = len(freqs)
        end_dim = start_dim + num_pairs * 2
        
        # Apply RoPE over head_dim dimension
        x_rope = x[..., start_dim:end_dim]
        assert (
            x_rope.shape == (B, N, num_heads, num_pairs * 2)
        ), f"x_rope must have shape of {((B, N, num_heads, num_pairs * 2))}, got {x_rope.shape}"

        # Concatenate unrotated and rotated parts of input tensor to use later
        x_pass = torch.cat([
            x[..., :start_dim], # before RoPE application
            x[..., end_dim:]    # after RoPE appliaction
        ], dim=-1) if start_dim > 0 or end_dim < x.size(-1) else torch.empty_like(x[..., :0])
        
        x_rope = x_rope.view(B, N, num_heads, num_pairs, 2)
        assert (
            x_rope.shape == (B, N, num_heads, num_pairs, 2)
        ), f"x_rope must have shape of {(B, N, num_heads, num_pairs, 2)}, got {x_rope.shape}"
        
        # Compute angles via p * w where p = positions, w = freqs
        # positions shape: [N, 3] -> [N, 1, 3]
        # freqs shape: [head_dim // 6] -> [1, head_dim // 6]
        angles = positions[:, None] * freqs[None]

        # Compute cosine and sine matrices for rotation matrix
        # Add singleton dimensions for broadcastability
        cos_vals = torch.cos(angles)[None, :, None, :, None]
        sin_vals = torch.sin(angles)[None, :, None, :, None]
        
        # rotation matrix = [[cos(x), -sin(x)], [sin(x), cos(x)]]
        # x_rot = x * rotation_matrix (element wise)
        # We use stack to concatenate the rotated x dim and rotated y dim
        x_rope_rotated = torch.stack([
            x_rope[..., 0] * cos_vals.squeeze(-1) - x_rope[..., 1] * sin_vals.squeeze(-1),
            x_rope[..., 0] * sin_vals.squeeze(-1) + x_rope[..., 1] * cos_vals.squeeze(-1)
        ], dim=-1)
        
        x_rope_rotated = x_rope_rotated.view(B, N, num_heads, num_pairs * 2)
        assert (
            x_rope_rotated.shape == (B, N, num_heads, num_pairs * 2)
        ), f"x_rope_rotated must have shape {(B, N, num_heads, num_pairs * 2)}, got {x_rope_rotated.shape}"
        
        # Greater than 0 means rotation did not occur
        if x_pass.size(-1) > 0:
            if start_dim == 0:
                # Concatenate rotated + unrotated
                return torch.cat([x_rope_rotated, x_pass], dim=-1)
            elif end_dim == x.size(-1):
                # Concatenate unrotated + rotated
                return torch.cat([x_pass, x_rope_rotated], dim=-1)
            else:
                # concat(start dim, rotated dim, end dim)
                return torch.cat([
                    x[..., :start_dim],
                    x_rope_rotated,
                    x[..., end_dim:]
                ], dim=-1)
        # x_pass.size(-1) == 0, rotation occured, return as is
        else:
            return x_rope_rotated
        
    def _compute_3d_rope_embeddings(
        self, 
        x: torch.Tensor, 
        grid_shape: Tuple[int, int, int],
        attn_mode: Literal["spatial", "temporal"],
    ) -> torch.Tensor:
        """Compute positional embeddings over time, height, and width dimensions.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, N, num_heads, head_dim].
            grid_shape (Tuple[int, int, int]): Grid shape containing number of patches for T, H, W dimensions.
            attn_mode (Literal["spatial", "temporal"]): Whether attention is being applied spatially or temporally.

        Returns:
            torch.Tensor: Query or key tensor with rotation applied.
        """
        # Get number of patches for each dimension
        grid_t, grid_h, grid_w = grid_shape
        # Apply RoPE based on whether we apply spatial or temporal attention
        if attn_mode == "spatial":
            spatial_positions = self._get_3d_grid_positions(1, grid_h, grid_w) # 1, H, W for spatial
            x = self._apply_rotary_embedding_1d(
                x, spatial_positions[:, 1], self.freqs_h, start_dim=self.dim_per_axis
            )
            x = self._apply_rotary_embedding_1d(
                x, spatial_positions[:, 2], self.freqs_w, start_dim=2 * self.dim_per_axis
            )

        elif attn_mode == "temporal":
            temporal_positions = self._get_3d_grid_positions(grid_t, 1, 1) # T, 1, 1 for temporal
            x = self._apply_rotary_embedding_1d(
                x, temporal_positions[:, 0], self.freqs_t, start_dim=0
            )
        else:
            raise ValueError(f"attn_mode must be 'spatial' or 'temporal' got {attn_mode}")
        
        return x
        
    def forward(
        self, 
        x: torch.Tensor, 
        grid_shape: Tuple[int, int, int],
        attn_mode: Literal["spatial", "temporal"]
    ) -> torch.Tensor:
        """Apply 3D RoPE to input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, N, num_heads, head_dim].
            grid_shape (Tuple[int, int, int]): Patch grid shape (T, H, W).

        Returns:
            torch.Tensor: Rotated query or key tensor with embedded positional awareness.
        """
        with autocast(device_type=device.type, dtype=dtype):
            return self._compute_3d_rope_embeddings(x, grid_shape, attn_mode)
        