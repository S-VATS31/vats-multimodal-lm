from configs.setup_env import device, dtype

import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch.amp import autocast

# TODO: ADD TARGET SIZE FOR IMAGE GEN MODEL ARGS
#
# SQUARE IMAGE GEN:
# target_size: int = H/W size
# USING INT INSTEAD OF TUPLE TO GENERATE SQUARE IMAGES

class NTKRoPE2D(nn.Module):
    """2D NTK RoPE layer for robust positional embeddings.
    
    Args:
        head_dim (int): Dimensionality of each attention head.
        rope_theta (float): Exponential base of inv freq.
        ntk_scale_factor (float): NTK scaling factor multipled by inv freq.
    """
    def __init__(
        self,
        head_dim: int,
        rope_theta: float,
        use_ntk_rope: bool,
        ntk_scale_factor: Optional[float] = None,
    ):
        super().__init__()

        # Ensure head_dim/4 for 2D RoPE
        if head_dim % 4 != 0:
            raise ValueError(
                f"expected head_dim % 4 == 0, got {head_dim} % 4 != 0."
            )
        
        if use_ntk_rope:
            assert (
                ntk_scale_factor is not None
            ), "Must be given ntk_scale_factor for NTK RoPE."
        else:
            ntk_scale_factor = None

        # Each dim uses half of head_dim
        half_dim = head_dim // 2

        # Initialize freqs
        inv_freqs = []
        for _ in range(2): # H, W dims
            # Compute inverse frequencies as:
            # inv_freq = 1 / (theta^{2i/d})
            inv_freq = 1.0 / (
                rope_theta ** (
                    torch.arange(0, half_dim, 2).to(torch.float32) / half_dim # [half_dim//2]
                )
            )
            inv_freqs.append(inv_freq)

        assert (
            inv_freqs[0].shape == (half_dim // 2,)
        ), f"expected {(half_dim // 2,),}, got {inv_freqs[0].shape}"
        assert (
            inv_freqs[1].shape == (half_dim // 2,)
        ), f"expected {(half_dim // 2,),}, got {inv_freqs[1].shape}"

        # Register frequencies as buffers
        self.register_buffer("h_freqs", inv_freqs[0])
        self.register_buffer("w_freqs", inv_freqs[1])

        self.head_dim = head_dim
        self.use_ntk_rope = use_ntk_rope
        self.ntk_scale_factor = ntk_scale_factor

    def _compute_grid_size(self, grid_size: int) -> torch.Tensor:
        """Compute H x W grid for 2D RoPE.
        
        Args:
            grid_size (int): Flattened height and width dimensions (H*W).

        Returns:
            torch.Tensor: Grid size of shape [grid_size, 2].
        """
        H = W = math.isqrt(grid_size) # Generating square images
        assert H*H == grid_size, f"grid_size not a perfect square: {grid_size}"

        # Compute 1D grids
        grid_H = torch.arange(H, dtype=dtype).to(device) # [H]
        grid_W = torch.arange(W, dtype=dtype).to(device) # [W]
        assert grid_H.shape == (H,), f"expected {(H,)}, got {grid_H.shape}"
        assert grid_W.shape == (W,), f"expected {(W,)}, got {grid_W.shape}"

        # [H, W, 2]
        spatial_grid = (
            torch.stack(
                torch.meshgrid(grid_H, grid_W, indexing="ij"), dim=-1
            )
        )
        assert (
            spatial_grid.shape == (H, W, 2)
        ), f"expected {(H, W, 2)}, got {spatial_grid.shape}"

        # 2D tensor
        spatial_grid = spatial_grid.view(-1, 2) # [H*W, 2]
        assert (
            spatial_grid.shape == (H*W, 2)
        ), f"expected {(grid_size, 2)}, got {spatial_grid.shape}"

        return spatial_grid

    def _apply_ntk_scaling(self,  spatial_grid: torch.Tensor) -> torch.Tensor:
        """Apply NTK scaling if enabled.
        
        Args:
            spatial_grid (torch.Tensor): Grid containing 1D height and with grids.

        Returns:
            torch.Tensor: Scaled spatial_grid
        """
        return spatial_grid * self.ntk_scale_factor

    def _compute_freqs_cis(self, grid_size: int) -> torch.Tensor:
        """Compute sine and cosine frequencies for rotation.
        
        Args:
            grid_size (int): Spatial grid size.

        Returns:
            torch.Tensor: Sine and cosine freqs in complex space.

        NOTE:
            cis(theta) = cos(theta) + isin(theta)
        """
        with autocast(device_type=device.type, enabled=False): # fp32 for sin/cos matrices
            spatial_grid = self._compute_grid_size(grid_size)
            assert (
                spatial_grid.shape == (grid_size, 2)
            ), f"expected {(grid_size, 2)}, got {spatial_grid.shape}"

            # Apply NTK scaling to total grid
            if self.use_ntk_rope:
                spatial_grid = self._apply_ntk_scaling(spatial_grid)

            # Outer products
            outer_h_freqs = torch.outer(spatial_grid[:, 0], self.h_freqs) # [grid_size, head_dim//4]
            outer_w_freqs = torch.outer(spatial_grid[:, 1], self.w_freqs) # [grid_size, head_dim//4]

            assert (
                outer_h_freqs.shape == (grid_size, self.head_dim//4)
            ), f"expected {(grid_size, self.head_dim//4)}, got {outer_h_freqs.shape}"
            assert (
                outer_w_freqs.shape == (grid_size, self.head_dim//4)
            ), f"expected {grid_size, self.head_dim//4}, got {outer_w_freqs.shape}"

            # Get frequencies and convert to complex exponentials
            freqs = torch.cat([outer_h_freqs, outer_w_freqs], dim=-1) # [grid_size, head_dim//2]
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

            assert (
                freqs_cis.shape == (grid_size, self.head_dim//2)
            ), f"expected {(grid_size, self.head_dim//2)}, got {freqs_cis.shape}"

            return freqs_cis

    def _apply_rotary_emb(
        self, 
        x: torch.Tensor, 
        grid_size: int
    ) -> torch.Tensor:
        """Rotate QK vectors using complex multiplication.
        
        Args:
            x (torch.Tensor): Input Q or K vector of shape [B, H*W, num_heads, head_dim].
            grid_size (int): Grid size computed as H*W.

        Returns:
            torch.Tensor: Rotated Q or K tensor of same shape.
        """
        freqs_cis = self._compute_freqs_cis(grid_size)

        # Expand freqs_cis to Q or K vectors
        freqs_cis = freqs_cis[None, :, None, :] # [1, grid_size, 1, head_dim//2]
        assert (
            freqs_cis.shape == (1, grid_size, 1, self.head_dim//2)
        ), f"expected: {(1, grid_size, 1, self.head_dim//2)}, got {freqs_cis.shape}"

        # Complex view
        x_complex = torch.view_as_complex(
            x.view(*x.shape[:-1], -1, 2)
        ) # [B, T, num_heads, head_dim//2]
        assert (
            x_complex.shape == (x.size(0), grid_size, x.size(2), self.head_dim//2)
        ), f"expected: {(x.size(0), grid_size, x.size(2), self.head_dim//2)}, got {x_complex.shape}"

        # Complex multiplication
        x_rot = torch.view_as_real(
            x_complex * freqs_cis
        ).flatten(-2) # [B, H*W, num_heads, head_dim]
        assert (
            x_rot.shape == x.shape
        ), (
            "rotated tensor must  have same shape as input tensor. "
            f"in: {x.shape}, out: {x_rot.shape}"
        )

        return x_rot

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 2D NTK RoPE to QK vectors.

        Args:
            x (torch.Tensor): Input query or key tensor of shape [B, H*W, num_heads, head_dim].

        Returns:
            torch.Tensor: Output tensor with same shape.
        """
        with autocast(device_type=device.type, enabled=False):
            return self._apply_rotary_emb(x, grid_size=x.size(1))

def test_rope():
    d_model, num_heads = 512, 32
    head_dim = d_model//num_heads
    rope_theta, use_ntk_rope, ntk_scale_factor = 10000.0, True, 0.7
    query_groups = 8
    ntk_rope = NTKRoPE2D(
        head_dim=head_dim,
        rope_theta=rope_theta,
        use_ntk_rope=use_ntk_rope,
        ntk_scale_factor=ntk_scale_factor
    ).to(device)
    B, H, W = 1, 32, 32
    q = torch.randn(B, H*W, num_heads, head_dim).to(device)
    k = torch.randn(B, H*W, query_groups, head_dim).to(device)
    q_rot, k_rot = ntk_rope(q), ntk_rope(k)
    return q_rot, k_rot

if __name__ == "__main__":
    q_rot, k_rot = test_rope()
    print(q_rot.shape)
    print(k_rot.shape)
    