from configs.setup_env import device, dtype

from typing import Optional

import torch
import torch.nn as nn

from torch.amp import autocast

class NTKRoPE3D(nn.Module):
    """NTK RoPE 3D for robust positional embeddings for video generation systems.
    
    Args:
        head_dim (int): Dimensionality of each attention head.
        rope_theta (float): Exponential base of inverse frequency.
        use_ntk_rope (bool): Whether to use NTK RoPE or classic.
        ntk_scale_factor (Optional[float]): NTK alpha hyperparameter.
    """
    def __init__(
        self,
        head_dim: int,
        rope_theta: float,
        use_ntk_rope: bool,
        ntk_scale_factor: Optional[float] = None
    ):
        super().__init__()

        self.head_dim = head_dim
        self.ntk_scale_factor = ntk_scale_factor

        if head_dim % 6 != 0:
            raise ValueError(
                f"expected {head_dim} % 6 == 0, got {head_dim % 6}"
            )

        if use_ntk_rope:
            assert ntk_scale_factor is not None, "must be given scale factor for NTK"
        else:
            ntk_scale_factor = None

        dim_per_ax = head_dim // 3

        inv_freqs = []
        for _ in range(3): # T, H, W
            inv_freq = 1.0 / (
                rope_theta ** (torch.arange(0, dim_per_ax, 2).float() / dim_per_ax
                ).to(device)
            )
            inv_freqs.append(inv_freq)

        assert (
            inv_freqs[0].shape == (dim_per_ax//2,) 
        ), f"expected {(dim_per_ax//2,)}, got {inv_freqs[0].shape}"
        assert (
            inv_freqs[1].shape == (dim_per_ax//2,)
        ), f"expected {(dim_per_ax//2,)}, got {inv_freqs[1].shape}"
        assert (
            inv_freqs[2].shape == (dim_per_ax//2,)
        ), f"expected {(dim_per_ax//2,)}, got {inv_freqs[2].shape}"
        
        self.register_buffer("t_freqs", inv_freqs[0])
        self.register_buffer("h_freqs", inv_freqs[1])
        self.register_buffer("w_freqs", inv_freqs[2])

    def _compute_grid_size(self, grid_size: int) -> torch.Tensor:
        pass

    def _compute_freqs_cis(self, ) -> torch.Tensor:
        pass

    def _apply_ntk_rope(self) -> torch.Tensor:
        pass

    def _apply_rotary_emb(self) -> torch.Tensor:
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with autocast(device_type=device.type, dtype=dtype):
            return x
    

def test():
    head_dim, rope_theta = 42, 10000.0
    rope = NTKRoPE3D(head_dim, rope_theta, True, 1/4).to(device)
    return rope

if __name__ == "__main__":
    rope = test()
    print(rope.t_freqs)
