from configs.setup_env import (
    device,
    dtype,
    gpu_dtypes,
    use_flash_attn,
    flash_attn_varlen_qkvpacked_func
)

import math
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
        target_size (int): Target height and width all images will be reshaped to.
        patch_size (int): Size of each patch (assumes square patches).
        base (float): Denominator raised to the power of 2i/d.

    Raises:
        ValueError if `head_dim % 4 != 0`
    """
    def __init__(
        self, 
        head_dim: int,
        target_size: int,
        patch_size: int,
        rope_theta: float
    ):
        super().__init__()

        # Ensure head_dim is divisible by 4 for 2D RoPE
        if head_dim % 4 != 0:
            raise ValueError(
                f"head_dim must be divisible by 4 for 2D RoPE, head_dim: {head_dim}"
            )
        
        self.head_dim = head_dim
        self.patch_size = patch_size
        self.grid_size = target_size // patch_size
        self.num_patches = (target_size // patch_size) ** 2

        # Calculate inverse frequency for both x and y dimensions
        freq_dim = head_dim // 4
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, freq_dim, dtype=torch.float32) / freq_dim))
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
            x (torch.Tensor): Input tensor of shape: [B, T, num_heads, head_dim]

        Returns:
            torch.Tensor: Tensor with applied 2D rotary positional embeddings of shape: [B, H*W, num_heads, head_dim].
        """
        with autocast(device_type=device.type, dtype=dtype):
            assert (
                x.dim() == 4
            ), f"x must have 4 dimensions, got {x.dim()} dimensions."
            num_spatial_patches = x.size(1)
            
            # Calculate grid size from number of patches
            grid_size = int(math.sqrt(num_spatial_patches))
            sin_x, cos_x, sin_y, cos_y = self._compute_sine_cosine(grid_size)

            # [B, H*W, num_heads, head_dim]
            x = self._create_rotary(x, sin_x, cos_x, sin_y, cos_y)

            # [B, H*W, num_heads, head_dim]
            return x

class SpatialAttention(nn.Module):
    """Spatial attention layer.
    
    Args:
        d_model (int): Dimensionality of model embeddings.
        num_heads (int): Number of attention heads for GQA.
        query_groups (int): Number of query groupsf for GQA.
        rope_theta (float): Exponential base of inv freq for RoPE.
        target_size (int): Target height and width images will be reshaped to.
        patch_size (int): Height and width square patches.
        softmax_scale (float): Scaling factor for attention scores.
        use_windowed_attn (bool): Whether to use sliding window attention or not.
        use_proj_bias (bool): Whether to use bias for projection matrices.
        use_fused_proj (bool): Whether to use single qkv projection or seperate projections.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        query_groups: int,
        rope_theta: float,
        target_size: int,
        patch_size: int,
        softmax_scale: float,
        use_windowed_attn: bool,
        use_proj_bias: bool,
        use_fused_proj: bool,
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model must be divisble by num_heads, got {d_model} % {num_heads} != 0."
            )
        if num_heads % query_groups != 0:
            raise ValueError(
                f"num_heads must be divisble by query_groups, got {num_heads} % {query_groups} != 0."
            )
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.query_groups = query_groups
        self.head_dim = d_model // num_heads
        self.heads_per_group = num_heads // query_groups
        self.softmax_scale = softmax_scale
        self.use_windowed_attn = use_windowed_attn
        self.use_fused_proj = use_fused_proj

        # Fused projection
        if use_fused_proj:
            self.qkv_proj = nn.Linear(
                d_model,
                num_heads * self.head_dim + 2 * query_groups * self.head_dim,
                bias=use_proj_bias
            )
        else:
            self.q_proj = nn.Linear(
                d_model,
                num_heads * self.head_dim,
                bias=use_proj_bias
            )
            self.k_proj = nn.Linear(
                d_model,
                query_groups * self.head_dim,
                bias=use_proj_bias
            )
            self.v_proj = nn.Linear(
                d_model,
                query_groups * self.head_dim,
                bias=use_proj_bias
            )
        # Output projection
        self.o_proj = nn.Linear(
            d_model,
            d_model,
            bias=use_proj_bias
        )

        self.rope = RoPE(
            head_dim=self.head_dim,
            target_size=target_size,
            patch_size=patch_size,
            rope_theta=rope_theta
        )

    def _extend_kv_heads(
        self,
        input: torch.Tensor,
        repeats: int,
        dim: int,
        use_mqa: bool
    ) -> torch.Tensor:
        """Repeat KV heads for specific number of times.
        
        Args:
            input (torch.Tensor): Input key or value tensor.
            repeats (int): Number of repeats for specific dimension.
            dim (int): Dimension to be repeated (query_groups dim).
            use_mqa (bool): Whether to use MQA or not.

        Returns:
            torch.Tensor: Output tensor with specific dimension repeated.
        """
        if use_mqa and input.size(dim) == 1:
            return input
        return input.repeat_interleave(repeats=repeats, dim=dim)

    def _optimized_attention(
        self,
        x: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        left_window: int,
        right_window: int,
    ) -> torch.Tensor:
        """Optimized attention using Flash Attention V2, SWA, and GQA or MQA.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, H*W, d_model] used only for shape validation.
            query (torch.Tensor): Query tensor of shape [B, H*W, num_heads, head_dim].
            key (torch.Tensor): Key tensor of shape [B, H*W, num_heads, head_dim].
            value (torch.Tensor): Value tensor of shape [B, H*W, num_heads, head_dim].
            left_window (int): Left window for SWA.
            right_window (int): Right window for SWA.

        Returns:
            torch.Tensor: Output tensor of shape [B, H*W, d_model].
        """
        if (
            flash_attn_varlen_qkvpacked_func is not None 
            and use_flash_attn
            and device.type == "cuda"
            and query.dtype in gpu_dtypes
            and key.dtype in gpu_dtypes
            and value.dtype in gpu_dtypes
            and query.is_cuda
            and key.is_cuda
            and value.is_cuda
        ):
            # Stack qkv for flash attention v2
            # qkv_stacked shape: [B, H*W, num_heads, 3, head_dim]
            qkv_stacked = torch.stack([query, key, value], dim=3).contiguous()

            # Create cumulative sequence lengths of shape [B+1]
            cu_seqlens = torch.arange(
                0, (query.size(0) + 1) * query.size(1), dtype=torch.int32
            ).to(device)

            # Get max sequence length
            seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
            max_seqlen = seqlens.max().item()

            # Flatten qkv and reshape
            qkv_flattened = (
                qkv_stacked
                .view(-1, self.num_heads, 3, self.head_dim) # [B*H*W, num_heads, 3, head_dim]
                .transpose(1, 2) # [B*H*W, 3, num_heads, head_dim]
                .contiguous()
            )

            # Get attn out
            attn_out = flash_attn_varlen_qkvpacked_func(
                qkv_flattened,
                cu_seqlens,
                max_seqlen,
                causal=False,
                softmax_scale=self.softmax_scale,
                window_size=(left_window, right_window),
            ) # [B*H*W, num_heads, head_dim]
            
            # Reshape to 3D output
            attn_out = attn_out.view(query.size(0), query.size(1), -1)

            return attn_out

        else:
            return self._torch_attention(x, query, key, value)

    def _torch_attention(
        self,
        x: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Apply PyTorch scaled dot product attention.
        
        Args:
            x (torch.Tensor): Input tensor to be used for shape validation; shape: [B, H*W, d_model].
            query (torch.Tensor): Query tensor of shape [B, H*W, num_heads, head_dim].
            key (torch.Tensor): Key tensor of shape [B, H*W, num_heads, head_dim].
            value (torch.Tensor): Value tensor of shape [B, H*W, num_heads, head_dim].

        Returns:
            torch.Tensor: Output tensor of shape [B, H*W, d_model].
        """
        # We only pass input tensor for shape validation across steps
        B, num_spatial_patches, _ = x.shape
        
        # Transpose to [B, num_heads, H*W, head_dim]
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        
        assert (
            query.shape == (B, self.num_heads, num_spatial_patches, self.head_dim)
        ), (
            f"query must have shape of {(B, self.num_heads, num_spatial_patches, self.head_dim)}, "
            f"got {query.shape}"
        )
        assert (
            key.shape == (B, self.num_heads, num_spatial_patches, self.head_dim) or
            key.shape == (B, 1, num_spatial_patches, self.head_dim)
        ), (
            f"key must have shape of {(B, self.num_heads, num_spatial_patches, self.head_dim)} or "
            f"{(B, 1, num_spatial_patches, self.head_dim)}, got {key.shape}"
        )
        assert (
            value.shape == (B, self.num_heads, num_spatial_patches, self.head_dim) or
            value.shape == (B, 1, num_spatial_patches, self.head_dim)
        ), (
            f"value must have shape of {(B, self.num_heads, num_spatial_patches, self.head_dim)} or "
            f"{(B, 1, num_spatial_patches, self.head_dim)}, got {value.shape}"
        )

        # SDPA
        attn_out = F.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            is_causal=False,
            enable_gqa=False # we do this manually
        ) # [B, num_heads, H*W, head_dim]

        assert (
            attn_out.shape == (B, self.num_heads, num_spatial_patches, self.head_dim)
        ), (
            f"attn_out must have shape of {(B, self.num_heads, num_spatial_patches, self.head_dim)}, "
            f"got {attn_out.shape}"
        )

        # Reshape back to 3D tensor
        attn_out = (
            attn_out
            .transpose(1, 2)
            .contiguous()
            .view(query.size(0), query.size(2), -1)
        ) # [B, H*W, d_model]

        assert (
            attn_out.shape == (B, num_spatial_patches, self.d_model)
        ), f"attn_out must have shape of {(B, num_spatial_patches, self.d_model)}, got {attn_out.shape}"

        return attn_out

    def _spatial_attention(
        self,
        x: torch.Tensor,
        use_mqa: bool,
        left_window: int,
        right_window: int,
    ) -> torch.Tensor:
        """Apply spatial attention to input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, H*W, d_model].
            use_mqa (bool): Whether to use MQA or not.
            left_window (int): Left window for SWA.
            right_window (int): Right window for SWA.

        Returns:
            torch.Tensor: Output tensor of shape as input.
        """
        q, k, v = self._setup_qkv(x, use_mqa=use_mqa)
        # Get attention output
        spatial_out = self._optimized_attention(
            x=x,
            query=q,
            key=k, 
            value=v,
            left_window=left_window,
            right_window=right_window
        )

        return spatial_out

    def _setup_qkv(
        self,
        x: torch.Tensor,
        use_mqa: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Setup query, key, and value tensors.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, H*W, d_model].
            use_mqa (bool): Whether to use MQA or not.

        Returns:
            Tuple:
                - torch.Tensor: Query tensor of shape [B, H*W, num_heads, head_dim].
                - torch.Tensor: Key tensor of shape [B, H*W, num_heads or 1, head_dim].
                - torch.Tensor: Value tensor of shape [B, H*W, num_heads or 1, head_dim].
        """
        assert (
            x.dim() == 3
        ), f"x must have 3 dims, got {x.dim()}"
        B, num_spatial_patches, _ = x.shape

        # Get q, k, v
        if self.use_fused_proj:
            # qkv shape: [B, H*W, num_heads*head_dim + 2*query_groups*head_dim]
            qkv = self.qkv_proj(x)
            assert (
                qkv.shape == (
                    B, num_spatial_patches, self.num_heads*self.head_dim + 2*self.query_groups*self.head_dim
                )
            ), f"qkv must have shape of {(
                B, num_spatial_patches, self.num_heads*self.head_dim + 2*self.query_groups*self.head_dim
            )}, got {qkv.shape}"

            # q shape: [B, H*W, num_heads*head_dim]
            # kv shape: [B, H*W, 2*query_groups*head_dim]
            q, kv = torch.split(
                qkv, [self.num_heads*self.head_dim, 2*self.query_groups*self.head_dim], dim=-1
            )
            assert (
                kv.shape == (B, num_spatial_patches, 2*self.query_groups*self.head_dim)
            ), f"kv must have shape of {(B, num_spatial_patches, 2*self.query_groups*self.head_dim)}."

            # k, v shape: [B, H*W, query_groups*head_dim]
            k, v = kv.chunk(chunks=2, dim=-1)
        else:
            q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Assert together
        assert (
                q.shape == (B, num_spatial_patches, self.num_heads*self.head_dim)
            ), f"q must have shape of {(B, num_spatial_patches, self.num_heads*self.head_dim)}, got {q.shape}"
        assert (
                k.shape == (B, num_spatial_patches, self.query_groups*self.head_dim)
            ), f"k must have shape of {(B, num_spatial_patches, self.query_groups*self.head_dim)}, got {k.shape}"
        assert (
                v.shape == (B, num_spatial_patches, self.query_groups*self.head_dim)
        ), f"v must have shape of {(B, num_spatial_patches, self.query_groups*self.head_dim)}, got {v.shape}"

        # Reshape into 4D tensors
        q = q.view(B, num_spatial_patches, self.num_heads, self.head_dim)
        k = k.view(B, num_spatial_patches, self.query_groups, self.head_dim)
        v = v.view(B, num_spatial_patches, self.query_groups, self.head_dim)

        assert (
            q.shape == (B, num_spatial_patches, self.num_heads, self.head_dim)
        ), f"q must have shape of {(B, num_spatial_patches, self.num_heads, self.head_dim)}, got {q.shape}"
        assert (
            k.shape == (B, num_spatial_patches, self.query_groups, self.head_dim)
        ), f"k must have shape of {(B, num_spatial_patches, self.query_groups, self.head_dim)}, got {k.shape}"
        assert (
            v.shape == (B, num_spatial_patches, self.query_groups, self.head_dim)
        ), f"v must have shape of {(B, num_spatial_patches, self.query_groups, self.head_dim)}, got {v.shape}"

        # Apply RoPE; forward expects [B, H*W, num_heads, head_dim]
        # q shape: [B, H*W, num_heads, head_dim]
        # k shape: [B, H*W, query_groups, head_dim]
        q = self.rope(q)
        k = self.rope(k)

        assert (
            q.shape == (B, num_spatial_patches, self.num_heads, self.head_dim)
        ), f"q must have shape of {(B, num_spatial_patches, self.num_heads, self.head_dim)}, got {q.shape}"
        assert (
            k.shape == (B, num_spatial_patches, self.query_groups, self.head_dim)
        ), f"q must have shape of {(B, num_spatial_patches, self.query_groups, self.head_dim)}, got {q.shape}"

        # Extend kv heads
        k = self._extend_kv_heads(
            input=k,
            repeats=self.heads_per_group,
            dim=2,
            use_mqa=use_mqa
        )
        v = self._extend_kv_heads(
            input=v,
            repeats=self.heads_per_group,
            dim=2,
            use_mqa=use_mqa
        )

        assert (
            k.shape == (B, num_spatial_patches, self.num_heads, self.head_dim) or
            k.shape == (B, num_spatial_patches, 1, self.head_dim)
        ), (
            f"k must have shape of {(B, num_spatial_patches, self.num_heads, self.head_dim)} or "
            f"{(B, num_spatial_patches, 1, self.head_dim)}, got {k.shape}"
        )
        assert (
            v.shape == (B, num_spatial_patches, self.num_heads, self.head_dim) or
            v.shape == (B, num_spatial_patches, 1, self.head_dim)
        ), (
            f"v must have shape of {(B, num_spatial_patches, self.num_heads, self.head_dim)} or "
            f"{(B, num_spatial_patches, 1, self.head_dim)}, got {v.shape}"
        )

        return q, k, v

    def forward(
        self,
        x: torch.Tensor,
        use_mqa: bool,
        left_window: int,
        right_window: int
    ) -> torch.Tensor:
        """Forward pass of spatial attention layer.

        Args:
            x (torch.Tensor): Input tensor of shape [B, H*W, d_model].
            use_mqa (bool): Whether to use MQA or not.
            left_window (int): Left window for SWA.
            right_window (int): Right window for SWA.

        Returns:
            torch.Tensor: Output tensor of shape as input.
        """
        with autocast(device_type=device.type, dtype=dtype):
            # Set windows to -1 if no SWA
            if not self.use_windowed_attn:
                left_window, right_window = -1, -1
            # [B, H*W, d_model]
            spatial_out = self._spatial_attention(
                x=x,
                use_mqa=use_mqa,
                left_window=left_window,
                right_window=right_window
            )

            return self.o_proj(spatial_out)


class SpatialAttentionBlock(nn.Module):
    """Spatial attention block with residuals, normalization, and dropout.
    
    Args:
        d_model (int): Dimensionality of model embeddings.
        num_heads (int): Number of attention heads for GQA.
        query_groups (int): Number of query groupsf for GQA.
        rope_theta (float): Exponential base of inv freq for RoPE.
        target_size (int): Target height and width images will be reshaped to.
        patch_size (int): Height and width square patches.
        softmax_scale (float): Scaling factor for attention scores.
        use_windowed_attn (bool): Whether to use sliding window attention or not.
        use_proj_bias (bool): Whether to use bias for projection matrices.
        use_fused_proj (bool): Whether to use single qkv projection or seperate projections.
        eps (float): Small epsilon value to prevent numerical instability.
        dropout (float): Dropout probability.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        query_groups: int,
        rope_theta: float,
        target_size: int,
        patch_size: int,
        softmax_scale: int,
        use_windowed_attn: bool,
        use_proj_bias: bool,
        use_fused_proj: bool,
        eps: float,
        dropout: float,
    ):
        super().__init__()

        self.attention = SpatialAttention(
            d_model=d_model,
            num_heads=num_heads,
            query_groups=query_groups,
            rope_theta=rope_theta,
            target_size=target_size,
            patch_size=patch_size,
            softmax_scale=softmax_scale,
            use_windowed_attn=use_windowed_attn,
            use_proj_bias=use_proj_bias,
            use_fused_proj=use_fused_proj
        )
        self.rms_norm = RMSNorm(
            d_model=d_model, eps=eps
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        use_mqa: bool,
        left_window: int,
        right_window: int,
    ) -> torch.Tensor:
        """Forward pass of attention block.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, H*W, d_model].
            use_mqa (bool): Whether to use MQA or not.
            left_window (int): Left window for SWA.
            right_window (int): Right window for SWA.

        Returns:
            torch.Tensor: Output tensor of same shape as input.
        """
        with autocast(device_type=device.type, dtype=dtype):
            return x + self.dropout(self.rms_norm(
                self.attention(
                    x=x,
                    use_mqa=use_mqa,
                    left_window=left_window,
                    right_window=right_window
                )
            ))

def test_attention_block():
    d_model, num_heads, query_groups = 512, 32, 8
    rope_theta, target_size, patch_size = 10000.0, 144, 16
    softmax_scale = 1 / (d_model // num_heads) ** 0.5
    eps, dropout = 1e-7, 0.15
    attention = SpatialAttentionBlock(
        d_model, num_heads, query_groups, rope_theta, target_size, 
        patch_size, softmax_scale, False, False, True, eps, dropout
    ).to(device)
    B = 1
    grid_size = target_size // patch_size
    num_patches = grid_size ** 2
    x = torch.randn(B, num_patches, d_model).to(device)
    x_out = attention(x, False, -1, -1)
    return x_out

if __name__ == "__main__":
    x = test_attention_block()
    # [1, num_patches, 512]
    # num_patches = (target_size // patch_size) ** 2
    print(x.shape)
    