from configs.setup_env import (
    device,
    dtype,
    use_flash_attn,
    flash_attn_varlen_qkvpacked_func,
    use_xformers_swiglu,
    swiglu
)

import warnings
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.checkpoint import checkpoint

from configs.transformers.vision.vit_3d.model_args.model_args_large import ModelArgs

class PatchEmbeddings3D(nn.Module):
    """Patch embeddings to split the video into 3D patches for spatiotemporal transformers.
    
    Args:
        C_in (int): Number of input channels.
        patch_size (Tuple[int, int, int]): Patch size in (T, H, W).
        d_model (int): Dimensionality of the model's embeddings.
        dropout (float): Dropout probability applied after normalization.
        eps (float): Epsilon for RMSNorm stability.
    """
    def __init__(
        self,
        C_in: int,
        patch_size: Tuple[int, int, int],
        d_model: int,
        dropout: float,
        eps: float,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.d_model = d_model

        # Conv3D projection
        self.projection = nn.Conv3d(
            in_channels=C_in,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False
        )

        # Set up RMS Norm/Dropout
        self.rms_norm = RMSNorm(d_model, eps)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x (torch.Tensor): Input tensor of shape [B, C_in, T, H, W].

        Returns:
            torch.Tensor: Patch embeddings of shape [B, N, d_model].
        """
        with autocast(device_type=x.device.type, dtype=x.dtype):
            assert(
                x.dim() == 5
            ), f"x must be a 5 dimenional tensor, got {x.dim()} dimensions"
            B, _, T, H, W = x.shape
            pt, ph, pw = self.patch_size
            
            # Calculate padding needed to make T, H, W divisible by patch size
            pad_t = (pt - T % pt) % pt
            pad_h = (ph - H % ph) % ph
            pad_w = (pw - W % pw) % pw

            # Pad in reverse order: (W_left, W_right, H_left, H_right, T_left, T_right)
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_t), mode='constant', value=0)
            
            # Project patch embeddings and flatten
            x = self.projection(x) # [B, d_model, T, H, W] C_in (in) -> d_model (out)
            
            assert(
                x.size(1) == self.d_model
            ), f"x.size(1) must be {self.d_model}, got {x.size(1)}"
            assert(
                x.dim() == 5
            ), f"x must be a 5 dimenional tensor, got {x.dim()} dimensions"

            x = x.view(B, self.d_model, -1).transpose(1, 2) # [B, N, d_model]

            # Apply normalization and dropout
            return self.rms_norm(self.dropout(x))


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
        for _ in range(3): # T, H, W
            num_pairs = self.dim_per_axis // 2
            freqs = 1.0 / (self.theta ** (torch.arange(0, num_pairs) * 2.0 / self.dim_per_axis))
            freqs_per_dim.append(freqs)
        
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
        B, N, num_heads, _ = x.shape

        # Get number of pairs and compute end dimension
        num_pairs = len(freqs)
        end_dim = start_dim + num_pairs * 2
        
        # Apply RoPE over sequence length dimension
        x_rope = x[..., start_dim:end_dim]

        # Concatenate unrotated and rotated parts of input tensor to use later
        x_pass = torch.cat([
            x[..., :start_dim], 
            x[..., end_dim:]
        ], dim=-1) if start_dim > 0 or end_dim < x.size(-1) else torch.empty_like(x[..., :0])
        
        x_rope = x_rope.view(B, N, num_heads, num_pairs, 2)
        
        # Compute angles via p * w where p = positions, w = freqs
        angles = positions.unsqueeze(1) * freqs.unsqueeze(0)

        # Compute cosine and sine matrices for rotation matrix:
        cos_vals = torch.cos(angles).unsqueeze(0).unsqueeze(2).unsqueeze(4)
        sin_vals = torch.sin(angles).unsqueeze(0).unsqueeze(2).unsqueeze(4)
        
        # rotation matrix = [[cos(x), -sin(x)], [sin(x), cos(x)]]
        # x_rot = x * rotation_matrix (element wise)
        # We use stack to concatenate the rotated x dim and rotated y dim
        x_rope_rotated = torch.stack([
            x_rope[..., 0] * cos_vals.squeeze(-1) - x_rope[..., 1] * sin_vals.squeeze(-1),
            x_rope[..., 0] * sin_vals.squeeze(-1) + x_rope[..., 1] * cos_vals.squeeze(-1)
        ], dim=-1)
        
        x_rope_rotated = x_rope_rotated.view(B, N, num_heads, num_pairs * 2)
        
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
        grid_shape: Tuple[int, int, int]
    ) -> torch.Tensor:
        """Compute positional embeddings over time, height, and width dimensions.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, N, num_heads, head_dim].
            grid_shape (Tuple[int, int, int]): Grid shape containing number of patches for T, H, W dimensions.

        Returns:
            torch.Tensor: Query or key tensor with rotation applied.
        """
        # Get number of patches for each dimension
        grid_t, grid_h, grid_w = grid_shape
        positions_3d = self._get_3d_grid_positions(grid_t, grid_h, grid_w) # [N, 3]
        
        # Get rotations via complex rotation
        # positions_3d[:, 0] = T, positions_3d[:, 1] = H, positions_3d[:, 2] = W
        # freqs_... = non-learnable buffers containing inverse frequences for T, H, W
        x = self._apply_rotary_embedding_1d(x, positions_3d[:, 0], self.freqs_t, start_dim=0)
        x = self._apply_rotary_embedding_1d(x, positions_3d[:, 1], self.freqs_h, start_dim=self.dim_per_axis)
        x = self._apply_rotary_embedding_1d(x, positions_3d[:, 2], self.freqs_w, start_dim=2 * self.dim_per_axis)
        
        return x
        
    def forward(
        self, 
        x: torch.Tensor, 
        grid_shape: Tuple[int, int, int]
    ) -> torch.Tensor:
        """Apply 3D RoPE to input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, N, num_heads, head_dim].
            grid_shape (Tuple[int, int, int]): Patch grid shape (T, H, W).

        Returns:
            torch.Tensor: Rotated query or key tensor with embedded positional awareness.
        """
        with autocast(device_type=device.type, dtype=dtype):
            _, N, _, _ = x.shape
            
            # Ensure N = grid_t * grid_h * grid_w
            grid_t, grid_h, grid_w = grid_shape
            expected_patches = grid_t * grid_h * grid_w
            
            if expected_patches != N:
                raise ValueError(
                    f"Grid shape {grid_shape} implies {expected_patches} patches, but got {N} patches"
                )
            
            return self._compute_3d_rope_embeddings(x, grid_shape)


class RMSNorm(nn.Module):
    """Apply RMSNorm to the features dimension.

    Formula:
        x_norm = (x / sqrt(mean(x**2))) * self.weight
    
    Args:
        d_model (int): Dimensionality of the model's embeddings.
        eps (float): Small epsilon value to prevent numerical stability.
    """
    def __init__(self, d_model: int, eps: float):
        super().__init__()

        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform forward pass of RMSNorm layer.

        Args:
            x (torch.Tensor): Input tensor of shape [B, N, d_model].    
        """
        with autocast(device_type=device.type, dtype=dtype):
            return self.weight * (
                x / (torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True)) + self.eps)
            )
        

class Attention(nn.Module):
    """Attention layer with Flash Attention 2, SWA, and GQA support.
    
    Args:
        d_model (int): Dimensionality of the model's embeddings.
        num_heads (int): Number of attention heads for the queries.
        query_groups (int): Number of query groups for GQA.
        rope_theta (float): Theta hyperparameter for RoPE.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        query_groups: int,
        rope_theta: float,
        patch_size: Tuple[int, int, int],
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                f"Expected d_model to be divisble by num_heads, got {d_model} % {num_heads} != 0"
                )
        if num_heads % query_groups != 0:
            raise ValueError(
                f"Expected num_heads to be divisble by query_groups, got {num_heads} % {query_groups} != 0"
                )
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.query_groups = query_groups
        self.head_dim = d_model // num_heads
        self.heads_per_group = num_heads // query_groups

        # QKV projection matrix
        self.w_qkv = nn.Linear(
            d_model,
            num_heads * self.head_dim + 2 * query_groups * self.head_dim,
            bias=False, # decrease parameter count
            dtype=dtype
        )

        # O projection matrix
        self.w_o = nn.Linear(
            d_model,
            d_model,
            bias=False,
            dtype=dtype
        )

        # Initialize RoPE
        self.rope = RoPE3D(self.head_dim, rope_theta, patch_size)

    def _extend_kv_heads(
        self,
        kv_tensor: torch.Tensor,
        heads_per_group: int,
        kv_heads_dim: int,
        use_mqa: bool = False, 
    ) -> torch.Tensor:
        """Extend kv heads to num_heads.
        
        Args:
            kv_tensor (torch.Tensor): Input key or value tensor.
            heads_per_group (int): Heads per group computed as num_heads // query_groups.
            kv_heads_dim (int): Dimension to be repeated.
            use_mqa (bool): Whether to use Multi-Query attention or not. Constraints: query_groups == 1.
                It is strongly recommended to set use_mqa=False for video transformers unless you are
                prioritizing speed/efficiency.

        Returns:
            torch.Tensor: K or V tensor with kv heads dimension repeated, now equal to num_heads.
        """
        if use_mqa and kv_tensor.size(kv_heads_dim) == 1:
            warnings.warn(
                "Although MQA is memory efficient and fast, it is not recommended on video transformers. "
                "Consider using GQA for reduced memory usage while still maintaining expressiveness."
            )
            return kv_tensor
        return torch.repeat_interleave(kv_tensor, repeats=heads_per_group, dim=kv_heads_dim)

    def _optimized_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        B: int,
        N: int,
        window_size: Tuple[int, int],
    ) -> torch.Tensor:
        """Optimized attention method leveraging flash attention 2, sliding window attention, and GQA.
        
        Args:
            query (torch.Tensor): Query tensor of shape [B, N, num_heads, head_dim].
            key (torch.Tensor): Key tensor of shape [B, N, num_heads, head_dim].
            value (torch.Tensor): Value tensor of shape [B, N, num_heads, head_dim].
            B (int): Batch size.
            N (int): Number of patches.
            window_size (Tuple[int, int]): Window size for sliding window attention.

        Returns:
            torch.Tensor: Output tensor of shape [B, N, d_model].

        NOTE: OPTIMIZED ATTENTION HAS NOT BEEN TESTED DUE TO HARDWARE REQUIREMENTS.
        """
        # Optimized Flash Attention 2 + SWA + GQA or MQA route
        # Requirements:
        # - Flash attention import must be succesful
        # - `device` must be cuda
        # - qkv tensors must have a dtype of bf16/fp16.
        if (
            use_flash_attn and device.type == "cuda"
            and query.dtype in [torch.float16, torch.bfloat16]
            and key.dtype in [torch.float16, torch.bfloat16] 
            and value.dtype in [torch.float16, torch.bfloat16]
        ):
            # Concatenate tensors along 3rd dimension
            qkv_packed = torch.stack([query, key, value], dim=3).contiguous() # [B, N, num_heads, 3, head_dim]

            assert(
                qkv_packed.shape == (B, N, self.num_heads, 3, self.head_dim)
            ), f"qkv_packed must have shape {(B, N, self.num_heads, 3, self.head_dim)}, got{qkv_packed.shape}"
            assert(
                qkv_packed.is_contiguous()
            ), "qkv_packed must be contiguous."

            # Get cumulative sequence lengths
            cu_seqlens = torch.arange(0, (B + 1) * N, N, dtype=torch.int32).to(device) # [B + 1]

            assert(
                cu_seqlens.shape == (B + 1)
            ), f"cu_seqlens must have shape {(B + 1)}, got {cu_seqlens.shape}"
            assert(
                cu_seqlens.dtype == torch.int32
            ), f"cu_seqlens dtype must be int32, got {cu_seqlens.dtype}"

            # We can compute the maximum seqeunce length by taking the difference between cumulative sequences
            # This gives us our true sequence length where we can compute the max
            seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
            max_seqlen = seq_lens.max().item()

            assert(
                len(seq_lens) == len(cu_seqlens) - 1
            ), f"len(seq_lens) must be len(cu_seqlens) - 1, got {len(seq_lens)} != {len(cu_seqlens)} - 1"

            # Compute total tokens
            total_tokens = B * max_seqlen

            # Flatten packed tensor
            qkv_flattened = qkv_packed.view(
                total_tokens, 
                self.num_heads, 3, 
                self.head_dim).transpose(1, 2).contiguous() # [total_tokens, 3, num_heads, head_dim]
            
            assert(
                qkv_flattened.shape == (total_tokens, 3, self.num_heads, self.head_dim)
            ), f"qkv_flattened must have shape {(total_tokens, 3, self.num_heads, self.head_dim)}, got {qkv_flattened.shape}"
            assert(
                qkv_flattened.is_contiguous()
            ), "qkv_flattened must be contiguous"
            
            # Call FlashAttention 2
            attn_out = flash_attn_varlen_qkvpacked_func(
                qkv_flattened,
                cu_seqlens,
                max_seqlen,
                causal=False,
                softmax_scale=1.0 / (self.head_dim ** 0.5),
                window_size=window_size,
            ) # [total_tokens, num_heads, head_dim]

            assert(
                attn_out.shape == (total_tokens, self.num_heads, self.head_dim)
            ), f"attn_out must have shape of {(total_tokens, self.num_heads, self.head_dim)}, got {attn_out.shape}"

            attn_out = attn_out.contiguous().view(B, N, -1) # [B, N, d_model]
            assert(
                attn_out.shape == (B, N, self.d_model)
            ), f"attn_out must have shape of {(B, N, self.d_model)}, got {attn_out.shape}"

            return attn_out
        
        # Either import didn't work, or no cuda; fallback to gqa/flash attn, w/o swa
        else:
            warnings.warn("Optimized attention not available, using PyTorch SDPA.")
            return self._grouped_query_attention(query, key, value, B, N)

    def _grouped_query_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        B: int,
        N: int,
    ) -> torch.Tensor:
        """PyTorch's scaled dot production attention with GQA, no SWA available.

        Args:
            query (torch.Tensor): Query tensor of shape [B, N, num_heads, head_dim].
            key (torch.Tensor): Key tensor of shape [B, N, num_heads, head_dim].
            value (torch.Tensor): Value tensor of shape [B, N, num_heads, head_dim].
            B (int): Batch size.
            N (int): Number of patches.

        Returns:
            torch.Tensor: Output tensor of shape [B, N, d_model].
        """
        # q, k, v shape after transpose: [B, num_heads, N, head_dim]
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Apply PyTorch SDPA
        attn_out = F.scaled_dot_product_attention(
            query, key, value,
            is_causal=False
        ) # [B, num_heads, N, head_dim]

        assert(
            attn_out.shape == (B, self.num_heads, N, self.head_dim)
        ), f"attn_out must have shape of {(B, self.num_heads, N, self.head_dim)}, got {attn_out.shape}"

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, self.d_model) # [B, N, d_model]
        assert(
            attn_out.shape == (B, N, self.d_model)
        ), f"attn_out must have shape of {(B, N, self.d_model)}, got {attn_out.shape}"

        return attn_out

    def forward(
        self, 
        x: torch.Tensor,
        grid_size: Tuple[int, int, int],
        window_size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """Perform forward pass of the attention layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, N, d_model].
            grid_size (Tuple[int, int, int]): Grid size parameter for RoPE.
            window_size (Tuple[int, int]): Window size for SWA.

        Returns:
            torch.Tensor: Output tensor of shape [B, N, d_model].
        """
        with autocast(device_type=device.type, dtype=dtype):
            B, N, _ = x.shape
            if x.shape != (B, N, self.d_model):
                raise ValueError(f"Expected x shape to be [B, N, d_model], got {x.shape}")

            # Project QKV
            qkv = self.w_qkv(x) # [B, N, num_heads * head_dim + 2 * query_groups * head_dim]
            assert(
                qkv.shape == (B, N, self.num_heads * self.head_dim + 2 * self.query_groups * self.head_dim)
            ), (
                f"qkv must have shape of {(B, N, self.num_heads * self.head_dim + 2 * self.query_groups * self.head_dim)} "
                f"got {qkv.shape}"
            )
            

            # q shape: [B, N, num_heads * head_dim], kv shape: [B, N, 2 * query_groups * head_dim]
            q, kv = torch.split(qkv, [self.num_heads * self.head_dim, 2 * self.query_groups * self.head_dim], dim=-1)
            assert(
                q.shape == (B, N, self.num_heads * self.head_dim)
            ), f"q must have shape of {(B, N, self.num_heads * self.head_dim)}, got {q.shape}"
            assert(
                kv.shape == (B, N, 2 * self.query_groups * self.head_dim)
            ), f"kv must have shape of {(B, N, 2 * self.query_groups * self.head_dim)}, got {kv.shape}"

            k, v = torch.chunk(kv, 2, dim=-1) # [B, N, head_dim * query_groups]
            assert(
                k.shape == (B, N, self.head_dim * self.query_groups)
            ), f"k must have shape of {(B, N, self.head_dim * self.query_groups)}, got {k.shape}"
            assert(
                v.shape == (B, N, self.head_dim * self.query_groups)
            ), f"v must have shape of {(B, N, self.head_dim * self.query_groups)}, got {v.shape}"

            # q shape: [B, N, num_heads, head_dim], k, v shape: [B, N, query_groups, head_dim]
            q = q.view(B, N, self.num_heads, self.head_dim)
            k = k.view(B, N, self.query_groups, self.head_dim)
            v = v.view(B, N, self.query_groups, self.head_dim)
            
            assert(
                q.shape == (B, N, self.num_heads, self.head_dim)
            ), f"q must have shape of {(B, N, self.num_heads, self.head_dim)}, got {q.shape}"
            assert(
                k.shape == (B, N, self.query_groups, self.head_dim) or
                k.shape == (B, N, 1, self.head_dim)
            ),  f"k must have shape of {(B, N, self.query_groups, self.head_dim)}, got {k.shape}"
            assert(
                v.shape == (B, N, self.query_groups, self.head_dim)
            ), f"v must have shape of {(B, N, self.query_groups, self.head_dim)}, got {v.shape}"

            # Check if kv heads need to be extended 
            if q.size(2) != k.size(2) or q.size(2) != v.size(2):
                # Extended kv heads for GQA
                k = self._extend_kv_heads(
                    kv_tensor=k, 
                    heads_per_group=self.heads_per_group,
                    kv_heads_dim=2,
                )
                v = self._extend_kv_heads(
                    kv_tensor=v, 
                    heads_per_group=self.heads_per_group,
                    kv_heads_dim=2
                )
            
            assert(
                k.size(2) == self.num_heads or k.size(2) == 1
            ), f"k.size(2) must be equal to {self.num_heads} or 1, got {k.size(2)}"
            assert(
                v.size(2) == self.num_heads or v.size(2) == 1
            ), f"v.size(2) must be equal to {self.num_heads} or 1, got {v.size(2)}"

            # Apply RoPE3D to qk tensors
            q = self.rope(q, grid_size)
            k = self.rope(k, grid_size)

            assert(
                q.shape == (B, N, self.num_heads, self.head_dim)
            ), f"q must have shape of {(B, N, self.num_heads, self.head_dim)}, got {q.shape}"
            assert(
                k.shape == (B, N, self.num_heads, self.head_dim) or
                k.shape == (B, N, 1, self.head_dim)
            ), (
                f"k must have shape of {(B, N, self.num_heads, self.head_dim)} "
                f"or {(B, N, 1, self.head_dim)} got {k.shape}"
            )

            # Apply optimized attention if available
            if window_size is not None:
                attn_out = self._optimized_attention(q, k, v, B, N, window_size)
            else:
                attn_out = self._grouped_query_attention(q, k, v, B, N)

            assert(
                attn_out.shape == (B, N, self.d_model)
            ), f"attn_out must have shape of {(B, N, self.d_model)}, got {attn_out.shape}"

            return self.w_o(attn_out) # [B, N, d_model]


class GatedFFN(nn.Module):
    """Gated FFN layer with SwiGLU activation.

    Formula:
        Dropout(SwiGLU(x)) = Dropout(w3 @ (Swish(w1 @ x) * (w2 @ x)))
        Set bias=False, better empirical results.
        
    Args:
        d_model (int): Dimensionality of the model's embeddings.
        d_ffn (int): Dimensionality of the FFN.
        dropout (float): Dropout probability.
    """
    def __init__(self, d_model: int, d_ffn: int, dropout: float):
        super().__init__()

        self.w1 = nn.Linear(d_model, d_ffn, bias=False)
        self.w2 = nn.Linear(d_model, d_ffn, bias=False)
        self.w3 = nn.Linear(d_ffn, d_model, bias=False) # output projection matrix
        self.dropout = nn.Dropout(p=dropout)

    def _optimized_swiglu(self, x: torch.Tensor) -> torch.Tensor:
        """Optimized SwiGLU via xformers.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, N, d_model].

        Returns:
            torch.Tensor: Output tensor of same shape.

        Requirements:
            xformers import must be succesful
            `device` must be cuda
            x.dtype must be float16/bfloat16
        """
        with autocast(device_type=device.type, dtype=dtype):
            # Optimized SwiGLU
            if (
                use_xformers_swiglu and device.type == "cuda"
                and x.dtype in [torch.float16, torch.bfloat16]
            ):
                return self.dropout(swiglu(
                    x.contiguous(), 
                    self.w1.weight.T, None, 
                    self.w2.weight.T, None, 
                    self.w3.weight.T, None
                ))
            # Fallback to PyTorch implementation of SwiGLU
            else:
                warnings.warn("xformers SwiGLU not available, falling back to PyTorch SwiGLU.")
                return self._pytorch_swiglu(x)
    
    def _pytorch_swiglu(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch implementation of SwiGLU, fallback for xformers SwiGLU.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, N, d_model].
        
        Returns:
            torch.Tensor: Output tensor of same shape.
        """
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform forward pass of the Gated FFN layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, N, d_model].

        Returns:
            torch.Tensor: Output tensor of same shape.
        """
        with autocast(device_type=device.type, dtype=dtype):
            assert(
                x.dim() == 3
            ), f"x must be a 3 dimensional tensor, got {x.dim()} dimensions"

            return self._optimized_swiglu(x)

class AttentionBlock(nn.Module):
    """Attention block with attention, normalization, dropout, and residuals applied.
    
    Args:
        d_model (int): Dimensionality of the model's embeddings.
        num_heads (int): Number of attention heads.
        query_groups (int): Number of query groups for GQA.
        rope_theta (float): Theta hyperparameter for RoPE.
        eps (float): Small value to prevent numerical instability.
        dropout (float): Dropout probability.
        patch_size (Tuple[int, int, int]): T, H, W sizes for each 3D patch.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        query_groups: int,
        rope_theta: float,
        eps: float,
        dropout: float,
        patch_size: Tuple[int, int, int],
    ):
        super().__init__()

        self.attention = Attention(d_model, num_heads, query_groups, rope_theta, patch_size)
        self.rms_norm = RMSNorm(d_model, eps)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self, 
        x: torch.Tensor,
        grid_size: Tuple[int, int, int],
        window_size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """Perform forward pass of the Attention layer.

        Args:
            x (torch.Tensor): Input tensor of shape [B, N, d_model].
            grid_size (Tuple[int, int, int]): Grid size parameter for RoPE.
            window_size (Optional[Tuple[int, int]]): Window size for SWA.

        Returns:
            torch.Tensor: Output tensor of shape [B, N d_model].
        """
        with autocast(device_type=device.type, dtype=dtype):
            return x + self.dropout(self.attention(self.rms_norm(x), grid_size, window_size))
    

class GatedFFNBlock(nn.Module):
    """Gated FFN block with a pass through the FFN, dropout, normalization, and residuals applied.
    
    Args:
        d_model (int): Dimensionality of the model's embeddings.
        d_ffn (int): Dimensionality of the FFN.
        dropout (float): Dropout probability.
        eps (float): Small epsilon value to prevent numerical instability.
    """
    def __init__(self, d_model: int, d_ffn: int, dropout: float, eps: float):
        super().__init__()

        self.gated_ffn = GatedFFN(d_model, d_ffn, dropout)
        self.rms_norm = RMSNorm(d_model, eps)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform forward pass of the Gated FFN layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, N, d_model].

        Returns:
            torch.Tensor: Output tensor with the same shape.
        """
        with autocast(device_type=device.type, dtype=dtype):
            return self.dropout(self.gated_ffn(self.rms_norm(x)))
        

class TransformerBlock(nn.Module):
    """Transformer block where Attention/FFN layers are stacked.
    
    Args:
        d_model (int): Dimensionality of the model's embeddings.
        num_heads (int): Number of attention heads.
        query_groups (int): Number of query groups for GQA.
        rope_theta (float): Theta hyperparameter for RoPE.
        d_ffn (int): Dimensionality of the FFN.
        dropout (float): Dropout probability.
        eps (float): Small epsilon value to prevent numerical instability.
        patch_size (Tuple[int, int, int]): T, H, W sizes for 3D patch of video.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        query_groups: int,
        rope_theta: float,
        d_ffn: int,
        dropout: float,
        eps: float,
        patch_size: Tuple[int, int, int],
    ):
        super().__init__()

        self.attention_block = AttentionBlock(d_model, num_heads, query_groups, rope_theta, eps, dropout, patch_size)
        self.gated_ffn_block = GatedFFNBlock(d_model, d_ffn, dropout, eps)

    def forward(
        self, 
        x: torch.Tensor,
        grid_size: Tuple[int, int, int],
        window_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """Perform forward pass of the Transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape [B, N, d_model].
            grid_size (Tuple[int, int, int]): Grid size parameter for RoPE.
            window_size (Optional[Tuple[int, int]]): Window size for SWA.

        Returns:
            torch.Tensor: Output tensor of shape [B, N, d_model].
        """
        with autocast(device_type=device.type, dtype=dtype):
            return self.gated_ffn_block(self.attention_block(x, grid_size, window_size))
        

class VideoTransformer(nn.Module):
    """Complete video transformer module.
    
    Args:
        model_args (ModelArgs): Dataclass containing all model hyperparameters.
    """
    def __init__(self, model_args: ModelArgs):
        super().__init__()

        self.model_args = model_args

        # Set up patch embeddings
        self.patch_embeddings = PatchEmbeddings3D(
            C_in=model_args.C_in,
            patch_size=model_args.patch_size,
            d_model=model_args.d_model,
            dropout=model_args.dropout,
            eps=model_args.rms_norm_eps,
        )

        # Stack transformer encoder layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=model_args.d_model,
                num_heads=model_args.num_heads,
                query_groups=model_args.query_groups,
                rope_theta=model_args.rope_theta,
                d_ffn=model_args.d_ffn,
                dropout=model_args.dropout,
                eps=model_args.rms_norm_eps,
                patch_size=model_args.patch_size,
            ) for _ in range(model_args.num_layers)
        ])

        # Set up RMSNorm and Dropout
        self.rms_norm = RMSNorm(model_args.d_model, model_args.rms_norm_eps)
        self.dropout = nn.Dropout(p=model_args.dropout)

        # Set up pool for adaptive average pooling
        self.pool = nn.AdaptiveAvgPool1d(1) # Pool over number of patches, N

        # Set up classifier
        self.classifier = nn.Linear(model_args.d_model, model_args.num_classes)

        # Initialize weights
        self.apply(self._init_weights)
        self._apply_depth_scaled_init()

    def _init_weights(self, module) -> None:
        """Weight initialization for VideoTransformer modules.
        
        Args:
            module: PyTorch module to initialize.
        """
        if isinstance(module, nn.Linear):
            # For linear layers, use Xavier/Glorot uniform with proper scaling
            if hasattr(module, 'weight') and module.weight is not None:
                # Special handling for different linear layer types
                if any(name in str(module) for name in ['w_qkv', 'w_o']):
                    std = (2.0 / (module.weight.size(-1) + module.weight.size(-2))) ** 0.5
                    nn.init.normal_(module.weight, mean=0.0, std=std)
                elif 'classifier' in str(module):
                    # Final classifier layer - use smaller initialization for stability
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                elif any(name in str(module) for name in ['w1', 'w3']):
                    # FFN input projections - Xavier uniform
                    nn.init.xavier_uniform_(module.weight, gain=1.0)
                elif 'w2' in str(module):
                    # FFN output projection - smaller initialization for residual stability
                    std = 0.02 / (2 * self.model_args.num_layers) ** 0.5
                    nn.init.normal_(module.weight, mean=0.0, std=std)
                else:
                    # Default for other linear layers
                    nn.init.xavier_uniform_(module.weight, gain=1.0)
            
            # Initialize biases to zero if they exist
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
        
        elif isinstance(module, nn.Conv3d):
            # 3D convolution for patch embedding - use Kaiming initialization
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.kaiming_normal_(
                    module.weight, 
                    mode='fan_out', 
                    nonlinearity='linear'
                )
            
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
        
        elif isinstance(module, RMSNorm):
            # RMSNorm weight initialization
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.ones_(module.weight)
        
        elif isinstance(module, nn.Embedding):
            # Embedding layers
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
        elif isinstance(module, nn.Parameter):
            # Handle any standalone parameters
            if module.dim() == 1:
                # 1D parameters (like normalization weights)
                nn.init.ones_(module)
            else:
                # Multi-dimensional parameters
                nn.init.xavier_uniform_(module)

    def _apply_depth_scaled_init(self) -> None:
        """Apply depth-scaled initialization as a post-processing step."""
        for layer in self.layers:
            # Scale residual connections by layer depth
            scale_factor = (2 * self.model_args.num_layers) ** -0.5
            
            # Scale attention output projection
            if hasattr(layer.attention_block.attention, 'w_o'):
                layer.attention_block.attention.w_o.weight.data.mul_(scale_factor)
            
            # Scale FFN output projection  
            if hasattr(layer.gated_ffn_block.gated_ffn, 'w2'):
                layer.gated_ffn_block.gated_ffn.w2.weight.data.mul_(scale_factor)
    
    def _compute_grid_size(
        self,
        video_shape: Tuple[int, int, int],
        patch_size: Tuple[int, int, int],
    ) -> Tuple[int, int, int]:
        """Compute grid size dynamically based on input T, H, W and patch sizes.
        
        Args:
            video_shape (Tuple[int, int, int]): Input video shape as (T, H, W).
            patch_size (Tuple[int, int, int]): Patch size as (pt, ph, pw).

        Returns:
            Tuple[int, int, int]: Returns grid size as (grid_t, grid_h, grid_w).
        """
        # Initialize grid dimensions
        grid_dims = []
        for dim_length, patch_length in zip(video_shape, patch_size):
            # Compute the number of patches needed (rounding up to cover all input)
            num_patches = (dim_length + patch_length - 1) // patch_length
            grid_dims.append(num_patches)

        return tuple(grid_dims)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform forward pass of the entire video transformer.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, T, H, W].

        Returns:
            torch.Tensor: Returns logits of shape [B, num_classses].
        """
        # Get T, H, W and store as a tuple
        _, _, T, H, W = x.size()
        assert(
            x.dim() == 5
        ), f"x must be a 5 dimensional tensor, got {x.dim()} dimensions"

        video_shape = (T, H, W)
        grid_size = self._compute_grid_size(video_shape=video_shape, patch_size=self.model_args.patch_size)

        # Apply patch embeddings and dropout
        x = self.dropout(self.patch_embeddings(x)) # [B, N, d_model]
        assert(
            x.dim() == 3
        ), f"x must be a 3 dimenional tensor after patch embeddings, got {x.dim()} dimensions."

        # Pass through transformer encoder layers
        for layer in self.layers:
            if self.model_args.use_checkpointing:
                x = checkpoint(
                    layer, 
                    x, 
                    grid_size,
                    self.model_args.window_size, 
                    use_reentrant=False
                )
            else:
                x = layer(x, grid_size, self.model_args.window_size)

        # Apply final RMSNorm
        x = self.rms_norm(x) # [B, N, d_model]

        # Apply adaptive average pooling
        x = x.transpose(1, 2) # [B, d_model, N]
        x = self.pool(x) # [B, d_model, 1]
        assert(
            x.size(-1) == 1
        ), f"x.size(-1) must be equal to 1, got {x.size(-1)}"

        x = x.squeeze(-1) # [B, d_model]
        assert(
            x.dim() == 2
        ), f"x must be a 2 dimensional tensor, got {x.dim()} dimensions"

        # Get logits through classifier
        logits = self.classifier(x)
        assert(
            logits.dim() == 2
        ), f"logits must be a 2 dimensional tensor, got {logits.dim()} dimensions"

        return logits # [B, num_classes]
