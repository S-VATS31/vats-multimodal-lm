from configs.setup_env import (
    device,
    dtype,
    use_flash_attn,
    flash_attn_varlen_qkvpacked_func,
)

import math
import warnings
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.checkpoint import checkpoint

from src.rms_norm import RMSNorm
from src.swiglu_activation import SwiGLUActivation
from configs.transformers.vision.vit_3d.model_args.model_args_large import ModelArgs

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
            in_channels=C_in,
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
            ).view(B, C, T, *self.target_size) # [B, C, T, new_H, new_W]

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
                ), "All padded frames must be False."
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
                    f"Truncation failed: "
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
            N = grid_t * grid_h * grid_w

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
                pooled[:, :, None, None] # [B, grid_t, 1, 1], need singleton dimensions to expand
                .expand(B, grid_t, grid_h, grid_w) # returns non-contiguous tensor, call .contiguous()
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

            x = x.view(B, self.d_model, -1).transpose(1, 2) # [B, N, d_model]
            assert (
                x.shape == (B, N, self.d_model)
            ), f"x must have shape of {(B, N, self.d_model)}, got {x.shape}"

            return x, processed_shape, patch_mask, grid_size


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
        grid_shape: Tuple[int, int, int],
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
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Optimized attention method leveraging flash attention 2, sliding window attention, and GQA.
        
        Args:
            query (torch.Tensor): Query tensor of shape [B, N, num_heads, head_dim].
            key (torch.Tensor): Key tensor of shape [B, N, num_heads, head_dim].
            value (torch.Tensor): Value tensor of shape [B, N, num_heads, head_dim].
            B (int): Batch size.
            N (int): Number of patches.
            window_size (Tuple[int, int]): Window size for sliding window attention.
            padding_mask (Optional[torch.Tensor]): Padding mask of shape [B, N].

        Returns:
            torch.Tensor: Output tensor of shape [B, N, d_model].

        Notes:
            B * N gives us total patches.

        Requirements:
            Flash attention import must be succesful.
            `device` must be cuda.
            q, k, v tensors must be float16 or bfloat16.

        NOTE: OPTIMIZED ATTENTION HAS NOT BEEN TESTED DUE TO HARDWARE REQUIREMENTS.
        """
        # Optimized Flash Attention 2 + SWA + GQA or MQA route
        if (
            use_flash_attn and device.type == "cuda"
            and query.dtype in [torch.float16, torch.bfloat16]
            and key.dtype in [torch.float16, torch.bfloat16] 
            and value.dtype in [torch.float16, torch.bfloat16]
        ):
            if padding_mask is not None:
                # Create padding mask
                assert (
                    padding_mask.shape == (B, N)
                ), f"padding_mask must have shape {(B, N)}, got {padding_mask.shape}."
                
                # padding_mask: True = valid patches, False = padded patches
                valid_mask = padding_mask.bool()
                seq_lens = valid_mask.sum(dim=1).to(torch.int32) # [B]
                # Get cumulative sequence lengths
                cu_seqlens = F.pad(torch.cumsum(seq_lens, dim=0), pad=(1, 0)) # [B + 1]
                max_seqlen = seq_lens.max().item()

                # Stack tensors along 3rd dimension
                qkv_packed = (
                    torch.stack([query, key, value], dim=3).contiguous()
                )  # [B, N, num_heads, 3, head_dim]

                assert(
                    qkv_packed.shape == (B, N, self.num_heads, 3, self.head_dim)
                ), f"qkv_packed must have shape {(B, N, self.num_heads, 3, self.head_dim)}, got {qkv_packed.shape}"
                assert(
                    qkv_packed.is_contiguous()
                ), "qkv_packed must be contiguous."

                # Get valid patches (to not be padded)
                valid_patches = valid_mask.view(-1) # [B * N]

                # Flatten packed tensor
                qkv_flattened = (
                    qkv_packed.view(-1, self.num_heads, 3, self.head_dim)
                    .transpose(1, 2)
                    .contiguous()
                ) # [B * N, 3, num_heads, head_dim]

                # Index by valid patches - only keep valid patches for flash attention
                qkv_valid = qkv_flattened[valid_patches]
                
                assert(
                    qkv_valid.shape == (valid_patches.sum().item(), 3, self.num_heads, self.head_dim)
                ), f"qkv_valid must have correct shape, got {qkv_valid.shape}"
                assert(
                    qkv_valid.is_contiguous()
                ), "qkv_valid must be contiguous"
                
                # Call FlashAttention 2
                attn_out = flash_attn_varlen_qkvpacked_func(
                    qkv_valid,
                    cu_seqlens,
                    max_seqlen,
                    causal=False,
                    softmax_scale=1.0 / (math.sqrt(self.head_dim)),
                    window_size=window_size,
                ) # [num_valid_patches, num_heads, head_dim]

                # Reconstruct padded positions
                attn_out_full = (
                    torch.zeros(B * N, self.num_heads, self.head_dim, dtype=attn_out.dtype)
                    .to(attn_out.device)
                )
                # Fill valid positions
                attn_out_full[valid_patches] = attn_out
                attn_out_full = attn_out_full.view(B, N, -1) # [B, N, d_model]

                return attn_out_full # return output with padded positions
            else:
                # TODO: maybe implement this, but typically padding_mask != None
                warnings.warn("no fallback to padding_mask == None.")

        # Either import didn't work, or no cuda; fallback to gqa/flash attn, w/o swa
        else:
            warnings.warn("Optimized attention not available, using PyTorch SDPA.")
            return self._grouped_query_attention(query, key, value, B, N, padding_mask)


    def _grouped_query_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        B: int,
        N: int,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """PyTorch's scaled dot production attention with GQA, no SWA available.

        Args:
            query (torch.Tensor): Query tensor of shape [B, N, num_heads, head_dim].
            key (torch.Tensor): Key tensor of shape [B, N, num_heads, head_dim].
            value (torch.Tensor): Value tensor of shape [B, N, num_heads, head_dim].
            B (int): Batch size.
            N (int): Number of patches.
            padding_mask (Optional[torch.Tensor]): Padding mask of shape [B, N].

        Returns:
            torch.Tensor: Output tensor of shape [B, N, d_model].
        """
        # q, k, v shape after transpose: [B, num_heads, N, head_dim]
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Set up padding mask to be broadcastable
        if padding_mask is not None:
            # True (valid) -> True (attend), False (padded) -> False (don't attend)
            attention_mask = padding_mask.bool() # Keep valid patches as True
            attention_mask = attention_mask[:, None, None, :] # [B, 1, 1, N]
        else:
            attention_mask = None

        # Apply PyTorch SDPA
        attn_out = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attention_mask
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
        window_size: Optional[Tuple[int, int]] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Perform forward pass of the attention layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, N, d_model].
            grid_size (Tuple[int, int, int]): Grid size parameter for RoPE.
            window_size (Tuple[int, int]): Window size for SWA.
            padding_mask (Optional[torch.Tensor]): Padding mask of shape [B, N].

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
                attn_out = self._optimized_attention(q, k, v, B, N, window_size, padding_mask)
            else:
                attn_out = self._grouped_query_attention(q, k, v, B, N, padding_mask)

            assert(
                attn_out.shape == (B, N, self.d_model)
            ), f"attn_out must have shape of {(B, N, self.d_model)}, got {attn_out.shape}"

            return self.w_o(attn_out) # [B, N, d_model]


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
        window_size: Optional[Tuple[int, int]] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Perform forward pass of the Attention layer.

        Args:
            x (torch.Tensor): Input tensor of shape [B, N, d_model].
            grid_size (Tuple[int, int, int]): Grid size parameter for RoPE.
            window_size (Optional[Tuple[int, int]]): Window size for SWA.
            padding_mask: (Optional[torch.Tensor]): Padding mask of shape [B, N].

        Returns:
            torch.Tensor: Output tensor of shape [B, N d_model].
        """
        with autocast(device_type=device.type, dtype=dtype):
            return x + self.dropout(
                self.attention(
                    self.rms_norm(x),
                    grid_size=grid_size,
                    window_size=window_size,
                    padding_mask=padding_mask,
                )
            )


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

        self.gated_ffn = SwiGLUActivation(d_model, d_ffn, dropout)
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

        self.attention_block = AttentionBlock(
            d_model, num_heads, query_groups, 
            rope_theta, eps, dropout, patch_size
        )
        self.gated_ffn_block = GatedFFNBlock(
            d_model, d_ffn, dropout, eps
        )

    def forward(
        self, 
        x: torch.Tensor,
        grid_size: Tuple[int, int, int],
        window_size: Optional[Tuple[int, int]] = None,
        padding_mask: Optional[torch.Tensor] = None,
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
            return self.gated_ffn_block(
                self.attention_block(x, grid_size, window_size, padding_mask)
            )
        

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
            target_size=model_args.target_size,
            max_frames=model_args.max_frames,
            d_model=model_args.d_model,
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
        
    def forward(
        self, 
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Perform forward pass of the entire video transformer.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, T, H, W].

        Returns:
            torch.Tensor: Returns logits of shape [B, num_classses].
        """
        assert(
            x.dim() == 5
        ), f"x must be a 5 dimensional tensor, got {x.dim()} dimensions"

        # Apply patch embeddings and dropout, get processed dimensions 
        x, _, padding_mask, grid_size = self.patch_embeddings(x)
        x = self.dropout(x) # [B, N, d_model]
        
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
                    padding_mask,
                    use_reentrant=False
                )
            else:
                x = layer(x, grid_size, self.model_args.window_size, padding_mask)

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

def main():
    model_args = ModelArgs()
    model = VideoTransformer(model_args).to(device)
    x = torch.randn(1, 3, 4, 16, 16).to(device)
    logits = model(x)
    return logits

if __name__ == "__main__":
    logits = main()
    print(logits.shape)
