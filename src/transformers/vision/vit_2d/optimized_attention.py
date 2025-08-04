from configs.setup_env import (
    device,
    dtype,
    use_flash_attn,
    flash_attn_varlen_qkvpacked_func
)

import math
import warnings
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

from src.rms_norm import RMSNorm

class RoPE(nn.Module):
    """Apply 2D rotary positional embeddings to query, key vectors.

    Args:
        head_dim (int): Dimensionality of each attention head.
        img_size (int): Size of the input image (assumes square images).
        patch_size (int): Size of each patch (assumes square patches).
        base (float): Denominator raised to the power of 2i/d.

    Raises:
        ValueError if `head_dim % 4 != 0`
    """
    def __init__(
        self, 
        head_dim: int, 
        img_size: int, 
        patch_size: int, 
        base: float
    ):
        super().__init__()

        # Ensure head_dim is divisible by 4 for 2D RoPE
        if head_dim % 4 != 0:
            raise ValueError(
                f"head_dim must be divisible by 4 for 2D RoPE, head_dim: {head_dim}"
                )
        
        self.head_dim = head_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Calculate inverse frequency for both x and y dimensions
        freq_dim = head_dim // 4
        inv_freq = 1.0 / (base ** (torch.arange(0, freq_dim, dtype=torch.float32) / freq_dim))
        assert (
            inv_freq.dtype == torch.float32
        ), f"inv_freq must have dtype of torch.float32, got {inv_freq.dtype}"

        self.register_buffer("inv_freq", inv_freq)

    def _compute_sine_cosine(
        self,
        grid_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute 2D Sine and Cosine Rotation Matrices for spatial positions.

        Args:
            grid_size (Optional[int]): Grid size (height and width) of the patch grid.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - torch.Tensor: Sine values for x-axis of shape [1, 1, num_patches, head_dim//4].
                - torch.Tensor: Cosine values for x-axis of shape [1, 1, num_patches, head_dim//4].
                - torch.Tensor: Sine values for y-axis of shape [1, 1, num_patches, head_dim//4].
                - torch.Tensor: Cosine values for y-axis of shape [1, 1, num_patches, head_dim//4].
        """
        if grid_size is None:
            grid_size = self.grid_size

        assert (
            grid_size is not None
        ), f"grid_size cannot be None at this point."

        # Create 2D position grid
        pos_x = torch.arange(grid_size, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        pos_y = torch.arange(grid_size, dtype=self.inv_freq.dtype, device=self.inv_freq.device)

        # Create meshgrid and flatten
        grid_x, grid_y = torch.meshgrid(pos_x, pos_y, indexing="ij")

        # Add singleton dimension for broadcasting
        pos_x_flat = grid_x.flatten()[:, None] # [num_patches, 1]
        pos_y_flat = grid_y.flatten()[:, None] # [num_patches, 1]

        assert (
            pos_x_flat.shape == (self.num_patches, 1)
        ), f"pos_x_flat must have shape of {(self.num_patches, 1)}, got {pos_x_flat.shape}"
        assert (
            pos_y_flat.shape == (self.num_patches, 1)
        ), f"pos_y_flat must have shape of {(self.num_patches, 1)}, got {pos_y_flat.shape}"

        # Compute rotation angles for x and y
        # rotation angles = positions * inverse frequency
        theta_x = pos_x_flat * self.inv_freq # [num_patches, head_dim//4]
        theta_y = pos_y_flat * self.inv_freq # [num_patches, head_dim//4]

        assert (
            theta_x.shape == (self.num_patches, self.head_dim // 4)
        ), f"theta_x must have shape of {(self.num_patches, self.head_dim // 4)}, got {theta_x.shape}"
        assert (
            theta_y.shape == (self.num_patches, self.head_dim // 4)
        ), f"theta_y must must shape of {(self.num_patches, self.head_dim // 4)}, got {theta_y.shape}"

        # Add singleton dimension to match q, k vectors number of dimensions
        sin_x = torch.sin(theta_x)[None, None] # [1, 1, num_patches, head_dim//4]
        cos_x = torch.cos(theta_x)[None, None] # [1, 1, num_patches, head_dim//4]
        sin_y = torch.sin(theta_y)[None, None] # [1, 1, num_patches, head_dim//4]
        cos_y = torch.cos(theta_y)[None, None] # [1, 1, num_patches, head_dim//4]
        assert (
            sin_x.dim() == 4 and
            cos_x.dim() == 4 and
            sin_y.dim() == 4 and
            cos_y.dim() == 4
        ), f"sin_x, cos_x, sin_y, cos_y all must have 4 dimensions."

        return sin_x, cos_x, sin_y, cos_y

    def _create_rotary(
        self,
        x: torch.Tensor,
        sin_x: torch.Tensor,
        cos_x: torch.Tensor,
        sin_y: torch.Tensor,
        cos_y: torch.Tensor
    ) -> torch.Tensor:
        """Create 2D rotary positional embeddings.

        Args:
            x (torch.Tensor): Input tensor of shape [B, T, num_heads, head_dim].
            sin_x (torch.Tensor): Sine values for x-axis of shape [1, 1, T, head_dim//4].
            cos_x (torch.Tensor): Cosine values for x-axis of shape [1, 1, T, head_dim//4].
            sin_y (torch.Tensor): Sine values for y-axis of shape [1, 1, T, head_dim//4].
            cos_y (torch.Tensor): Cosine values for y-axis of shape [1, 1, T, head_dim//4].

        Returns:
            torch.Tensor: Rotated tensor with shape: [B, T, num_heads, head_dim].
        """
        # Split head_dim into 4 parts for 2D rotation (x1, x2, y1, y2)
        freq_dim = self.head_dim // 4
        x_reshaped = x.reshape(*x.shape[:-1], 4, freq_dim) # [B, T, num_heads, 4, head_dim//4]
        x1, x2, y1, y2 = x_reshaped.unbind(dim=-2) # Each have shape: [B, T, num_heads, head_dim//4]

        # Expand sin/cos to match tensor dimensions
        sin_x = sin_x.permute(0, 2, 1, 3).expand(x1.shape[0], x1.shape[1], x1.shape[2], x1.shape[3])
        cos_x = cos_x.permute(0, 2, 1, 3).expand(x1.shape[0], x1.shape[1], x1.shape[2], x1.shape[3])
        sin_y = sin_y.permute(0, 2, 1, 3).expand(y1.shape[0], y1.shape[1], y1.shape[2], y1.shape[3])
        cos_y = cos_y.permute(0, 2, 1, 3).expand(y1.shape[0], y1.shape[1], y1.shape[2], y1.shape[3])

        # Apply 2D rotary embeddings
        # Complex multiplication via rotation matrix
        # rotation matrix = [[cos(x), -sin(x)], [sin(x), cos(x)]]
        # x_rot = x * rotation_matrix
        # y_rot = y * rotation_matrix
        x1_rot = x1 * cos_x - x2 * sin_x
        x2_rot = x1 * sin_x + x2 * cos_x
        y1_rot = y1 * cos_y - y2 * sin_y
        y2_rot = y1 * sin_y + y2 * cos_y

        # Stack back together
        x_rotated = torch.stack((x1_rot, x2_rot, y1_rot, y2_rot), dim=-2) # [B, T, num_heads, 4, head_dim//4]
        x_rotated = x_rotated.reshape(*x.shape) # [B, T, num_heads, head_dim]
        return x_rotated

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 2D rotary positional embeddings to input tensors (qk tensors).

        Args:
            x (torch.Tensor): Input tensor of shape: [B, num_heads, T, head_dim]

        Returns:
            torch.Tensor: Tensor with applied 2D rotary positional embeddings of shape: [B, num_heads, T, head_dim].
        """
        assert (
            x.dim() == 4
        ), f"x must have 4 dimensions, got {x.dim()} dimensions."
        T = x.size(2)
        
        # Calculate grid size from number of patches
        grid_size = int(math.sqrt(T))
        sin_x, cos_x, sin_y, cos_y = self._compute_sine_cosine(grid_size)

        # We want q, k shapes of [B, T, num_heads, head_dim] so transpose(1, 2)
        x = self._create_rotary(x.transpose(1, 2), sin_x, cos_x, sin_y, cos_y)

        # Transpose back to [B, num_heads, T, head_dim]
        return x.transpose(1, 2)


class GroupedQueryAttention(nn.Module):
    """Grouped query attention layer.

    Args:
        d_model (int): Dimensionality of the model's input/output representations.
        num_heads (int): Number of attention heads for queries.
        query_groups (int): Number of key/value groups (heads).
        rope_module (RoPE): An instance of the RoPE module for applying rotary embeddings.

    Raises:
        ValueError: If `d_model` is not divisible by `num_heads`.
        ValueError: If `num_heads` is not divisible by `query_groups`.
    """
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        query_groups: int, 
        rope_module: RoPE,
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divible by num_heads ({num_heads})"
                )
        if num_heads % query_groups != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by query_groups ({query_groups})"
                )

        self.d_model = d_model
        self.num_heads = num_heads
        self.query_groups = query_groups
        self.head_dim = d_model // num_heads
        self.heads_per_group = num_heads // query_groups

        # QKV projection
        self.w_qkv = nn.Linear(
            d_model,
            num_heads * self.head_dim + 2 * query_groups * self.head_dim,
            bias=False,
            dtype=dtype
        )

        # O projection
        self.w_o = nn.Linear(
            d_model,
            num_heads * self.head_dim,
            bias=False,
            dtype=dtype
        )

        # Initialize 2D RoPE
        self.rope_module = rope_module

    def _expand_query_groups(
        self, 
        input_tensor: torch.Tensor, 
        heads_per_group: int, 
        dim_to_repeat: int,
        use_mqa: bool = False,
    ) -> torch.Tensor:
        """Expand kv heads to query heads for GQA
        
        Args:
            input_tensor (torch.Tensor): Input key or value tensor to get expanded.
            heads_per_group (int): Heads per group computed as num_heads // query_groups.
            dim (int): Dimension of tensor to be repeated over.
            use_mqa (bool): Whether to use Multi-query attention or not.
                Constraints: input_tensor.size(dim_to_repeat) == 1.
                MQA not recommended for Vision Transformers due to loss of expressiveness.

        Returns:
            torch.Tensor: Output tensor with kv heads expanded.
        """
        if use_mqa and input_tensor.size(dim_to_repeat) == 1:
            warnings.warn("Using MQA, consider switching to GQA for better results.")
            return input_tensor
        return torch.repeat_interleave(input_tensor, heads_per_group, dim=dim_to_repeat)

    def forward(
        self, 
        x: torch.Tensor, 
        window_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Perform forward pass of Attention layer.

        Args:
            x (torch.Tensor): Input tensor of shape [B, T, d_model].
            window_size (Tuple[int, int]): Window size for sliding window attention.

        Returns:
            torch.Tensor: Output tensor transformed with same shape.

        Raises:
            ValueError: If `x` (input tensor) is not 3 dimensional.
            ValueError: If `D` is not equal to `d_model`.
            ValueError: If `q.shape[-1]` is not equal to `k.shape[-1]`.
            ValueError: If `softmax_attn.shape[-1]` is not equal to `v.shape[-2]`.

        Requirements for Flash Attention V2:
            Flash Attention import must be succesful.
            `device` must be cuda.
            q, k, v must have dtype of float16 or bfloat16.
        """
        with autocast(device_type=device.type, dtype=dtype):
            if x.dim() != 3:
                raise ValueError(f"Input tensor, x, must have 3 dimensions, got: {x.dim()} dimensions")
            B, T, _ = x.shape
            
            # Chunked projection matrix
            qkv = self.w_qkv(x) # [B, T, num_heads * head_dim + 2 * query_groups * head_dim]
            assert (
                qkv.shape == (B, T, self.num_heads * self.head_dim + 2 * self.query_groups * self.head_dim)
            ), (
                f"qkv shape must be {(B, T, self.num_heads * self.head_dim + 2 * self.query_groups * self.head_dim)} "
                f"got {qkv.shape}"
            )

            # q: [B, T, num_heads * head_dim]
            # kv: [B, T, 2 * query_groups * head_dim]
            q, kv = torch.split(qkv, [self.num_heads * self.head_dim, 2 * self.query_groups * self.head_dim], dim=-1)
            assert (
                q.shape == (B, T, self.num_heads * self.head_dim)
            ), f"q must have shape of {(B, T, self.num_heads * self.head_dim)}, got {q.shape}"
            assert(
                kv.shape == (B, T, 2 * self.query_groups * self.head_dim)
            ), f"kv must have shape of {(B, T, 2 * self.query_groups, self.head_dim)}, got {kv.shape}"

            # k, v shape: [B, T, query_groups * head_dim]
            k, v = torch.chunk(kv, chunks=2, dim=-1)
            assert (
                k.shape == (B, T, self.query_groups * self.head_dim)
                
            ), f"k must have shape of {(B, T, self.query_groups * self.head_dim)}, got {k.shape}"
            assert (
                v.shape == (B, T, self.query_groups * self.head_dim)
            ), f"v must have shape of {(B, T, self.query_groups * self.head_dim)}, got {v.shape}"
            
            # Reshape into 4D tensors for RoPE
            # q shape: [B, num_heads, T, head_dim]
            # kv shape: [B, query_groups, T, head_dim]
            q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(B, T, self.query_groups, self.head_dim).transpose(1, 2)
            v = v.view(B, T, self.query_groups, self.head_dim).transpose(1, 2)

            assert (
                q.shape == (B, self.num_heads, T, self.head_dim)
            ), f"q must have shape of {(B, self.num_heads, T, self.head_dim)}, got {q.shape}"
            assert (
                k.shape == (B, self.query_groups, T, self.head_dim)
            ), f"k must have shape of {(B, self.query_groups, T, self.head_dim)}, got {k.shape}"
            assert (
                v.shape == (B, self.query_groups, T, self.head_dim)
            ), f"v must have shape of {(B, self.query_groups, T, self.head_dim)}, got {v.shape}"

            # Apply RoPE to q and k using the new method that handles the correct tensor shapes
            q = self.rope_module(q) # [B, num_heads, T, head_dim]
            k = self.rope_module(k) # [B, query_groups, T, head_dim]

            assert (
                q.shape == (B, self.num_heads, T, self.head_dim)
            ), f"q must have shape of {(B, self.num_heads, T, self.head_dim)}, got {q.shape}"
            assert (
                k.shape == (B, self.query_groups, T, self.head_dim)
            ), f"k must have shape of {(B, self.query_groups, T, self.head_dim)}, got {k.shape}"

            # Expand kv heads to num heads for GQA
            k_expanded = self._expand_query_groups(k, self.heads_per_group, dim_to_repeat=1)
            v_expanded = self._expand_query_groups(v, self.heads_per_group, dim_to_repeat=1)

            assert (
                k_expanded.size(1) == self.num_heads or k_expanded.size(1) == 1
            ), f"k_expanded.size(1) must be {self.num_heads} or 1, got {k_expanded.size(1)}"
            assert (
                v_expanded.size(1) == self.num_heads or v_expanded.size(1) == 1
            ), f"v_expanded.size(1) must be {self.num_heads} or 1, got {v_expanded.size(1)}"

            # Flash Attention 2 + GQA + SWA
            if (
                use_flash_attn 
                and device.type == "cuda"
                and q.dtype in [torch.float16, torch.bfloat16]
                and k_expanded.dtype in [torch.float16, torch.bfloat16]
                and v_expanded.dtype in [torch.float16, torch.bfloat16]
            ):
                # Stack tensors along the 3rd dimension
                qkv_packed = (
                    torch.stack([q, k_expanded, v_expanded], dim=3)
                    .transpose(1, 2)
                    .contiguous()
                    ) # [B, T, num_heads, 3, head_dim]
                
                assert (
                    qkv_packed.shape == (B, T, self.num_heads, 3, self.head_dim)
                ), f"qkv_packed must have shape {(B, T, self.num_heads, 3, self.head_dim)}, got {qkv_packed.shape}"
                assert (
                    qkv_packed.is_contiguous()
                ), "qkv_packed must be contiguous."
                
                # Cumulative sequence lengths for all T
                cu_seqlens = torch.arange(0, (B + 1) * T, step=T, dtype=torch.int32).to(device) # [B + 1]
                assert (
                    cu_seqlens.shape == (B + 1)
                ), f"cu_seqlens must have shape of {(B + 1)}, got {cu_seqlens.shape}"
                assert (
                    cu_seqlens.dtype == torch.int32
                ), f"cu_seqlens must have dtype of int32, got {cu_seqlens.dtype}"

                # Get maximum sequence length
                seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
                max_seqlen = seqlens.max().item()

                assert (
                    len(seqlens) == len(cu_seqlens) - 1
                ), f"len(seqlens) must be equal to len(cu_seqlens) - 1, got {len(seqlens)} != {len(cu_seqlens) - 1}"

                # Flatten QKV
                qkv_flattened = (
                    qkv_packed.view(-1, self.num_heads, 3, self.head_dim)
                    .transpose(1, 2)
                    .contiguous()
                ) # [B * T, 3, num_heads, head_dim]
                assert (
                    qkv_flattened.shape == (B * T, 3, self.num_heads, self.head_dim)
                ), f"qkv_flattened must have shape of {(B * T, 3, self.num_heads, self.head_dim)}, got {qkv_flattened.shape}"
                assert (
                    qkv_flattened.is_contiguous()
                ), "qkv_flattened must be contiguous"

                # Compute attention output
                attn_out = flash_attn_varlen_qkvpacked_func(
                    qkv_flattened,
                    cu_seqlens,
                    max_seqlen,
                    causal=False,
                    softmax_scale=1.0 / (math.sqrt(self.head_dim)),
                    window_size=window_size,
                ) # [B * T, num_heads, head_dim]

                assert (
                    attn_out.shape == (B * T, self.num_heads, self.head_dim)
                ), f"attn_out must have shape of {(B * T, self.num_heads, self.head_dim)}, got {attn_out.shape}"

                # Reshape output to [B, T, d_model]
                attn_out = attn_out.contiguous().view(B, T, self.d_model)
                assert (
                    attn_out.shape == (B, T, self.d_model)
                ), f"attn_out must have shape of {(B, T, self.d_model)}, got {attn_out.shape}"

            # PyTorch SDPA (leverages Flash Attention if available)
            else:
                warnings.warn("Flash Attention V2/SWA not available, falling back PyTorch SDPA.")

                # PyTorch SDPA
                attn_out = F.scaled_dot_product_attention(
                    q, k_expanded, v_expanded,
                    is_causal=False,
                ) # [B, num_heads, T, head_dim]

                assert (
                    attn_out.shape == (B, self.num_heads, T, self.head_dim)
                ), f"attn_out must have shape of {(B, self.num_heads, T, self.head_dim)}, got {attn_out.shape}"
                
                # Reshape output to [B, T, d_model]
                attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.d_model)
                assert (
                    attn_out.shape == (B, T, self.d_model)
                ), f"attn_out must have shape of {(B, T, self.d_model)}, got {attn_out.shape}"

            return self.w_o(attn_out)


class GQABlock(nn.Module):
    """GQA layer with dropout, RMSNorm and residuals applied.

    Args:
        d_model (int): Dimensionality of the model's input/output representations.
        num_heads (int): Number of attention heads for queries.
        query_groups (int): Number of key/value groups (heads).
        eps (float): Epsilon value to maintain numerical stability in RMSNorm.
        rope_module (RoPE): An instance of the RoPE module for applying rotary embeddings.
        dropout (float): 
    """
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        query_groups: int, 
        rope_module: RoPE,
        eps: float,
        dropout: float,
    ):
        super().__init__()

        self.rms_norm = RMSNorm(d_model, eps)
        self.attn = GroupedQueryAttention(d_model, num_heads, query_groups, rope_module)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, window_size: Tuple[int, int]) -> torch.Tensor:
        """Perform forward pass of GQA Block.

        Args:
            x (torch.Tensor): Input tensor of shape [B, T, d_model].

        Returns:
            torch.Tensor: Output tensor with RMSNorm, GQA, Dropout, and residuals applied.
        """
        with autocast(device_type=device.type, dtype=dtype):
            return x + self.dropout(self.attn(self.rms_norm(x), window_size))
