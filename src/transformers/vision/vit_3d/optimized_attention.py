from configs.setup_env import (
    device,
    dtype,
    gpu_dtypes,
    use_flash_attn,
    flash_attn_varlen_qkvpacked_func
)

import math
import warnings
from typing import Tuple, Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

from src.rms_norm import RMSNorm
from src.transformers.vision.vit_3d.rope_3d import RoPE3D

class SpatioTemporalAttention(nn.Module):
    """Factorized attention layer.
    
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
        )

        # O projection matrix
        self.w_o = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )

        # Initialize RoPE
        self.rope = RoPE3D(self.head_dim, rope_theta, patch_size)

    def _extend_kv_heads(
        self,
        kv_tensor: torch.Tensor,
        heads_per_group: int,
        kv_heads_dim: int,
        use_mqa: bool,
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
            and query.dtype in gpu_dtypes
            and key.dtype in gpu_dtypes
            and value.dtype in gpu_dtypes
        ):
            if padding_mask is not None:
                # Handle padding mask
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
                raise ValueError("no fallback to padding_mask = None.")

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
    
    def _spatial_attention() -> torch.Tensor:
        pass

    def _temporal_attention() -> torch.Tensor:
        pass

    def _setup_qkv(
        self,
        x: torch.Tensor,
        use_mqa: bool,
        grid_shape: Tuple[int, int, int],
        attn_mode: Literal["spatial", "temporal"]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Set up query, key, and value tensors based on spatial or temporal attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, grid_T, grid_H * grid_W, d_model]
            attn_mode (Literal["spatial", "temporal"]): Attention mode to set qkv up properly.

        Returns:
            Tuple:
                - torch.Tensor: Query tensor.
                - torch.Tensor: Key tensor.
                - torch.Tensor: Value tensor.

        NOTE:
            A better method of assertions could assert the first 2 dims for x after first reshape.
            Example:
            >>> assert x.shape[:2] == (correct_shape)
            Then for later assertions, we can assume the first two dimensions are correct and
            only assert the final dim(s).
            Example:
            >>> assert q.shape[-1] == (correct_shape)
            This would remove the need for doing if attn_method == "method" before all assertions, however
            if the starting dimensions are wrong, we would not get the assertion call.
        """
        assert (
            x.dim() == 4
        ), f"x must be a 4-dim tensor, got {x.dim()}"
        B, T, num_spatial_patches, _ = x.shape
        
        if attn_mode == "spatial":
            x = x.view(B*T, num_spatial_patches, -1) # reshape to [B*T, H*W, d_model]
            assert (
                x.shape == (B*T, num_spatial_patches, self.d_model)
            ), f"x must have shape of {(B*T, num_spatial_patches, self.d_model)}, got {x.shape}"
        elif attn_mode == "temporal":
            x = x.transpose(1, 2).contiguous().view(B*num_spatial_patches, T, -1) # reshape to [B*H*W, T, d_model]
            assert (
                x.shape == (B*num_spatial_patches, T, self.d_model)
            ), f"x must have shape of {(B*num_spatial_patches, T, self.d_model)}, got {x.shape}"
        else:
            raise ValueError(f"attn_mode must be 'spatial' or 'temporal', got {attn_mode}")
        
        # [:, :, num_heads * head_dim + 2 * query_groups * head_dim]
        qkv = self.w_qkv(x)
        if attn_mode == "spatial":
            assert (
                qkv.shape == (
                    B*T, num_spatial_patches, self.num_heads * self.head_dim + 2 * self.query_groups * self.head_dim
                    )
            ), (
                f"qkv must have shape of {(
                    B*T, num_spatial_patches, self.num_heads * self.head_dim + 2 * self.query_groups * self.head_dim
                    )} "
                f"got {qkv.shape}"
            )
        elif attn_mode == "temporal":
            assert (
                qkv.shape == (
                    B*num_spatial_patches, T, self.num_heads * self.head_dim + 2 * self.query_groups * self.head_dim
                    )
            ), (
                f"qkv must have shape of {(
                    B*num_spatial_patches, T, self.num_heads * self.head_dim + 2 * self.query_groups * self.head_dim
                    )} "
                f"got {qkv.shape}"
            )
        
        # q shape: [:, :, num_heads * head_dim]
        # kv shape: [:, :, 2 * query_groups * head_dim]
        q, kv = torch.split(
            qkv, [self.num_heads * self.head_dim, 2 * self.query_groups * self.head_dim], dim=-1
        )
        if attn_mode == "spatial":
            assert (
                q.shape == (B*T, num_spatial_patches, self.d_model)
            ), f"q must have shape of {(B*T, num_spatial_patches, self.d_model)}, got {q.shape}"
            assert (
                kv.shape == (B*T, num_spatial_patches, 2 * self.query_groups * self.head_dim)
            ), f"kv must have shape of {(B*T, num_spatial_patches, 2 * self.query_groups * self.head_dim)}, got {kv.shape}"
        elif attn_mode == "temporal":
            assert (
                q.shape == (B*num_spatial_patches, T, self.d_model)
            ), f"q mut have shape of {(B*num_spatial_patches, T, self.d_model)}, got {q.shape}"
            assert (
                kv.shape == (B*num_spatial_patches, T, 2 * self.query_groups * self.head_dim)
            )
        
        # k, v shape: [:, :, query_groups * head_dim]
        k, v = kv.chunk(chunks=2, dim=-1)
        if attn_mode == "spatial":
            assert (
                k.shape == (B*T, num_spatial_patches, self.query_groups * self.head_dim)
            ), f"k must have shape of {(B*T, num_spatial_patches, self.query_groups * self.head_dim)}, got {k.shape}"
            assert (
                v.shape == (B*T, num_spatial_patches, self.query_groups * self.head_dim)
            ), f"v must have shape of {(B*T, num_spatial_patches, self.query_groups * self.head_dim)}, got {v.shape}"
        elif attn_mode == "temporal":
            assert (
                k.shape == (B*num_spatial_patches, T, self.query_groups * self.head_dim)
            ), f"k must have shape of {(B*num_spatial_patches, T, self.query_groups * self.head_dim)}, got {k.shape}"
            assert (
                k.shape == (B*num_spatial_patches, T, self.query_groups * self.head_dim)
            ), f"k must have shape of {(B*num_spatial_patches, T, self.query_groups * self.head_dim)}, got {k.shape}"

        # Reshape into 4D tensors
        # q shape: [:, :, num_heads, head_dim]
        # k, v shape: [:, :, query_groups, head_dim]
        if attn_mode == "spatial":
            q = q.view(B*T, num_spatial_patches, self.num_heads, self.head_dim)
            k = k.view(B*T, num_spatial_patches, self.query_groups, self.head_dim)
            v = v.view(B*T, num_spatial_patches, self.query_groups, self.head_dim)

            assert (
                q.shape == (B*T, num_spatial_patches, self.num_heads, self.head_dim)
            ), f"q must have shape of {(B*T, num_spatial_patches, self.num_heads, self.head_dim)}, got {q.shape}"
            assert (
                k.shape == (B*T, num_spatial_patches, self.query_groups, self.head_dim)
            ), f"k must have shape of {(B*T, num_spatial_patches, self.query_groups, self.head_dim)}, got {k.shape}"
            assert (
                v.shape == (B*T, num_spatial_patches, self.query_groups, self.head_dim)
            ), f"v must have shape of {(B*T, num_spatial_patches, self.query_groups, self.head_dim)}, got {v.shape}"
        elif attn_mode == "temporal":
            q = q.view(B*num_spatial_patches, T, self.num_heads, self.head_dim)
            k = k.view(B*num_spatial_patches, T, self.query_groups, self.head_dim)
            v = v.view(B*num_spatial_patches, T, self.query_groups, self.head_dim)

            assert (
                q.shape == (B*num_spatial_patches, T, self.num_heads, self.head_dim)
            ), f"q must have shape of {(B*num_spatial_patches, T, self.num_heads, self.head_dim)}, got {q.shape}"
            assert (
                k.shape == (B*num_spatial_patches, T, self.query_groups, self.head_dim)
            ), f"k must have shape of {(B*num_spatial_patches, T, self.query_groups, self.head_dim)}, got {k.shape}"
            assert (
                v.shape == (B*num_spatial_patches, T, self.query_groups, self.head_dim)
            ), f"v must have shape of {(B*num_spatial_patches, T, self.query_groups, self.head_dim)}, got {v.shape}"

        # Apply RoPE to qk tensors, same shapes from earlier
        q = self.rope(q, grid_shape, attn_mode)
        k = self.rope(k, grid_shape, attn_mode)

        # Extend heads
        k = self._extend_kv_heads(
            kv_tensor=k,
            heads_per_group=self.heads_per_group,
            kv_heads_dim=2,
            use_mqa=use_mqa
        )
        v = self._extend_kv_heads(
            kv_tensor=v,
            heads_per_group=self.heads_per_group,
            kv_heads_dim=2,
            use_mqa=use_mqa
        )

        return q, k, v

    def forward(
        self, 
        x: torch.Tensor,
        grid_size: Tuple[int, int, int],
        use_mqa: bool,
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

            # Extend kv heads
            k = self._extend_kv_heads(
                kv_tensor=k, 
                heads_per_group=self.heads_per_group,
                kv_heads_dim=2,
                use_mqa=use_mqa
            )
            v = self._extend_kv_heads(
                kv_tensor=v, 
                heads_per_group=self.heads_per_group,
                kv_heads_dim=2,
                use_mqa=use_mqa,
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

        self.attention = SpatioTemporalAttention(d_model, num_heads, query_groups, rope_theta, patch_size)
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

def test_setup_qkv(attn_mode):
    patch_size = (2, 32, 32)
    B, T, H, W, d_model, theta, num_heads, query_groups = (
        4, 10, 384, 384, 744, 10000.0, 124, 1
    )
    attention = SpatioTemporalAttention(
        d_model, num_heads, query_groups, theta, patch_size
    ).to(device)
    pt, ph, pw = patch_size
    processed_T, processed_H, processed_W = T // pt, H // ph, W // pw
    x = torch.randn(B, processed_T, processed_H * processed_W, d_model)
    grid_shape = (processed_T, processed_H, processed_W)
    q, k, v = attention._setup_qkv(x, use_mqa=True, grid_shape=grid_shape, attn_mode=attn_mode)
    return q, k, v

# SPATIAL ATTENTION
# q, k, v shape: [B * (T // pt),  (H // ph) * (W // ph),  num_heads, head_dim]
# q, k, v shape: [4 * (10 // 2), (384 // 32) * (384 // 32), 124, 744 // 124]
# q, k, v shape: [20, 144, 124, 6]
#
# with use_mqa=True (query_groups must be == 1)
# q has same shape
# k, v shape: [4 * (10 // 2), (384 // 32) * (384 // 32), 1, 724 // 124]
# k, v shape: [20, 144, 1, 6]

# TEMPORAL ATTENTION
# q, k, v shape: [B * (H // ph) * (W // pw), T // pt, num_heads, head_dim]
# q, k, v shape: [4 * (384 // 32) * (384 // 32), 10 // 2, 124, 6]
# q, k, v shape: [576, 5, 124, 6]
#
# with use_mqa=True (query_groups must b == 1)
# q has same shape
# k, v shape: [576, 5, 1, 6]

q, k, v = test_setup_qkv("temporal")
print(q.shape) 
print(k.shape)
print(v.shape)
