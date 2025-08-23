from configs.setup_env import (
    device,
    dtype,
    gpu_dtypes,
    use_flash_attn,
    flash_attn_varlen_qkvpacked_func
)

import math
from typing import Tuple, Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

from src.rms_norm import RMSNorm
from utils.attention_utils import extend_kv_heads, setup_projections
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

        # Get qkv and output projections
        self.w_qkv, self.w_o = setup_projections(
            d_model=d_model,
            num_heads=num_heads,
            head_dim=self.head_dim,
            use_fused_proj=True,
            use_gqa=True,
            use_proj_bias=False,
            query_groups=query_groups
        )

        # Initialize RoPE
        self.rope = RoPE3D(self.head_dim, rope_theta, patch_size)

    def _optimized_attention(
        self,
        x: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        window_size: Tuple[int, int],
        attn_mode: Literal["spatial", "temporal"],
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Optimized attention method leveraging flash attention 2, sliding window attention, and GQA.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, T, H*W, d_model].
            query (torch.Tensor): Query tensor of shape [B, N, num_heads, head_dim].
            key (torch.Tensor): Key tensor of shape [B, N, num_heads, head_dim].
            value (torch.Tensor): Value tensor of shape [B, N, num_heads, head_dim].
            window_size (Tuple[int, int]): Window size for sliding window attention.
            padding_mask (Optional[torch.Tensor]): Padding mask of shape [B, N].
            attn_mode (Literal["spatial", "temporal"]): Whether we are applying spatial or temporal attn.

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
                    padding_mask.shape == (query.size(0), query.size(1))
                ), f"padding_mask must have shape {(query.size(0), query.size(1))}, got {padding_mask.shape}."
                
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
                    qkv_packed.shape == (query.size(0), query.size(1), self.num_heads, 3, self.head_dim)
                ), f"qkv_packed must have shape {(query.size(0), query.size(1), self.num_heads, 3, self.head_dim)}, got {qkv_packed.shape}"
                assert(
                    qkv_packed.is_contiguous()
                ), "qkv_packed must be contiguous."

                # Get valid patches (to not be padded)
                valid_patches = valid_mask.view(-1) # [B * N]

                # Flatten packed tensor
                qkv_flattened = (
                    qkv_packed.view(-1, self.num_heads, 3, self.head_dim)
                    .contiguous()
                    .transpose(1, 2)
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
                    causal=True,
                    softmax_scale=1.0 / (math.sqrt(self.head_dim)),
                    window_size=window_size,
                ) # [B*N, num_heads, head_dim]

                # Reconstruct padded positions
                attn_out_full = (
                    torch.zeros(
                        query.size(0) * query.size(1), self.num_heads, self.head_dim, dtype=attn_out.dtype
                    ).to(attn_out.device)
                )
                # Fill valid positions
                attn_out_full[valid_patches] = attn_out
                attn_out_full = (
                    attn_out_full.view(query.size(0), query.size(1), self.d_model)
                ) # [B, N, d_model]

                return attn_out_full # [B, N, d_model]
            else:
                raise ValueError("no fallback to padding_mask = None.")

        # Either import didn't work, or no cuda; fallback to gqa/flash attn, w/o swa
        else:
            return self._grouped_query_attention(x, query, key, value, attn_mode, padding_mask)

    def _grouped_query_attention(
        self,
        x: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mode: Literal["spatial", "temporal"],
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """PyTorch's scaled dot production attention with GQA, no SWA available.

        Args:
            x (torch.Tensor): Input tensor of shape [B, T, H*W, d_model].
                We only pass x to get extract exact shapes.
            query (torch.Tensor): Query tensor of shape [B, N, num_heads, head_dim].
            key (torch.Tensor): Key tensor of shape [B, N, num_heads, head_dim].
            value (torch.Tensor): Value tensor of shape [B, N, num_heads, head_dim].
            B (int): Batch size.
            N (int): Number of patches.
            padding_mask (Optional[torch.Tensor]): Padding mask of shape [B, N].

        Returns:
            torch.Tensor: Output tensor of shape [B, N, d_model].
        """
        assert (
            x.dim() == 4
        ), f"x must have 4 dimensions, got {x.dim()} dimensions."
        # Get B, T, H*W dims to use later
        B, T, num_spatial_patches, _ = x.shape
        # q, k, v shape after transpose: [:, num_heads, :, head_dim]
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        if attn_mode == "spatial":
            assert (
                query.shape == (B*T, self.num_heads, num_spatial_patches, self.head_dim)
            ), (
                f"query must have shape of {(B*T, self.num_heads, num_spatial_patches, self.head_dim)}, " 
                f" for spatial attention, got {query.shape}."
            )
            assert (
                key.shape == (B*T, self.num_heads, num_spatial_patches, self.head_dim) or
                key.shape == (B*T, 1, num_spatial_patches, self.head_dim)
            ), (
                f"key must have shape of {(B*T, self.num_heads, num_spatial_patches, self.head_dim)} or "
                f"{(B*T, 1, num_spatial_patches, self.head_dim)} for spatial attention, got {key.shape}"
            )
            assert (
                value.shape == (B*T, self.num_heads, num_spatial_patches, self.head_dim) or
                value.shape == (B*T, 1, num_spatial_patches, self.head_dim)
            ), (
                f"value must have shape of {(B*T, self.num_heads, num_spatial_patches, self.head_dim)} or "
                f"{(B*T, 1, num_spatial_patches, self.head_dim)} for spatial attention, got {value.shape}"
            )

        elif attn_mode == "temporal":
            assert (
                query.shape == (B*num_spatial_patches, self.num_heads, T, self.head_dim)
            ), (
                f"query must have shape of {(B*num_spatial_patches, self.num_heads, T, self.head_dim)} "
                f"for spatial attention, got {query.shape}."
            )
            assert (
                key.shape == (B*num_spatial_patches, self.num_heads, T, self.head_dim) or
                key.shape == (B*num_spatial_patches, 1, T, self.head_dim)
            ), (
                f"key must have shape of {(B*num_spatial_patches, self.num_heads, T, self.head_dim)} or "
                f"{(B*num_spatial_patches, 1, T, self.head_dim)}, got {key.shape}."
            )
            assert (
                value.shape == (B*num_spatial_patches, self.num_heads, T, self.head_dim) or
                value.shape == (B*num_spatial_patches, 1, T, self.head_dim)
            ), (
                f"key must have shape of {(B*num_spatial_patches, self.num_heads, T, self.head_dim)} or "
                f"{(B*num_spatial_patches, 1, T, self.head_dim)}, got {value.shape}."
            )

        # Set up padding mask to be broadcastable
        if padding_mask is not None:
            if attn_mode == "spatial":
                padding_mask = padding_mask.view(B*T, num_spatial_patches) # [B*T, H*W]
                assert (
                    padding_mask.shape == (B*T, num_spatial_patches)
                ), f"padding_mask must have shape of {(B*T, num_spatial_patches)}, got {padding_mask.shape}"
            elif attn_mode == "temporal":
                padding_mask = padding_mask.view(-1, T) # [B*H*W, T]
                assert (
                    padding_mask.shape == (B*num_spatial_patches, T)
                ), f"padding_mask must have shape of {(B*num_spatial_patches, T)}, got {padding_mask.shape}"
            # True = valid positions, False = padded positions
            attention_mask = padding_mask.bool()
            attention_mask = attention_mask[:, None, None, :] # [:, 1, 1, :]
        else:
            attention_mask = None

        # check mask shape and dtype
        if attention_mask is not None:
            assert (
                attention_mask.dtype == torch.bool
            ), f"attention mask must be a bool tensor, got {attention_mask.dtype}"
            if attn_mode == "spatial":
                assert (
                    attention_mask.shape == (B*T, 1, 1, num_spatial_patches)
                ), (
                    f"attention mask must have shape {(B*T, 1, 1, num_spatial_patches)} "
                    f"for spatial attention, got {attention_mask.shape}"
                )
            elif attn_mode == "temporal":
                assert (
                    attention_mask.shape == (B*num_spatial_patches, 1, 1, T)
                ),(
                    f"attention mask must have shape {(B*T, 1, 1, num_spatial_patches)} "
                    f"for temporal attention, got {attention_mask.shape}"
                )

        # Apply PyTorch SDPA
        attn_out = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attention_mask,
            is_causal=False,
            enable_gqa=False
        ) # [:, num_heads, :, head_dim]

        if attn_mode == "spatial":
            assert (
                attn_out.shape == (B*T, self.num_heads, num_spatial_patches, self.head_dim)
            ), (
                f"attn_out must have shape of {(B*T, self.num_heads, num_spatial_patches, self.head_dim)} "
                f"for spatial attention, got {attn_out.shape}"
            )
        elif attn_mode == "temporal":
            assert (
                attn_out.shape == (B*num_spatial_patches, self.num_heads, T, self.head_dim)
            ), (
                f"attn_out must have shape of {(B*num_spatial_patches, self.num_heads, T, self.head_dim)} "
                f"for spatial attention, got {attn_out.shape}"
            )

        # Reshape output back to 3D tensor
        # In the forward pass we will reshape back to 4 dimensional tensor as [B, T, H*W, d_model]
        attn_out = (
            attn_out
            .transpose(1, 2) # [:, :, num_heads, head_dim]
            .contiguous()
            .view(query.size(0), query.size(2), self.d_model)
        ) # [:, :, d_model]
        
        if attn_mode == "spatial":
            assert (
                attn_out.shape == (B*T, num_spatial_patches, self.d_model)
            ), (
                f"attn_out must have shape of {(B*T, num_spatial_patches, self.d_model)}, "
                f"for spatial attention, got {attn_out.shape}"
            )
        elif attn_mode == "temporal":
            assert (
                attn_out.shape == (B*num_spatial_patches, T, self.d_model)
            ), (
                f"attn_out must have shape of {(B*num_spatial_patches, T, self.d_model)}, "
                f"for spatial attention, got {attn_out.shape}"
            )

        return attn_out
    
    def _spatial_attention(
        self,
        x: torch.Tensor,
        use_mqa: bool,
        grid_shape: Tuple[int, int, int],
        window_size: Tuple[int, int],
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply spatial attention as 1 x H x W.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, T, H*W, d_model].
                In _setup_qkv() we internally reshape to [B*T, H*W, d_model].
            use_mqa (bool): Whether to use multi-query attention or not.
            grid_shape (Tuple[int, int, int]): T, H, W dims after padding, reshaping, and creating patches.
            window_size (Tuple[int, int]): Left and right windows for SWA.
            padding_mask (Optional[torch.Tensor]): Padding tensor of shape [B, N].

        Returns:
            torch.Tensor: Output for the spatial attention layer of shape [B*T, H*W, d_model].
        """
        q, k, v = self._setup_qkv(x, use_mqa, grid_shape, attn_mode="spatial")
        # Get attention output, spatial
        spatial_out = self._optimized_attention(
            x=x,
            query=q, 
            key=k, 
            value=v,
            window_size=window_size,
            padding_mask=padding_mask,
            attn_mode="spatial"
        ) # [B*T, H*W, d_model]

        return spatial_out

    def _temporal_attention(
        self,
        x: torch.Tensor,
        use_mqa: bool,
        grid_shape: Tuple[int, int, int],
        window_size: Tuple[int, int],
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply temporal attention as T x 1 x 1.

        Args:
            x (torch.Tensor): Input tensor of shape [B, T, H*W, d_model].
               In _setup_qkv() we internally reshape to [B*H*W, T, d_model].
            use_mqa (bool): Whether to use multi-query attention or not.
            grid_shape (Tuple[int, int, int]): T, H, W dims after padding, reshaping, and creating patches.
            window_size (Tuple[int, int]): Left and right windows for SWA.
            padding_mask (Optional[torch.Tensor]): Padding tensor of shape [B, N].
        """
        q, k, v = self._setup_qkv(x, use_mqa, grid_shape, attn_mode="temporal")
        # Get attention output, temporal
        temporal_out = self._optimized_attention(
            x=x,
            query=q,
            key=k,
            value=v,
            window_size=window_size,
            padding_mask=padding_mask,
            attn_mode="temporal"
        ) # [B*H*W, T, d_model]

        return temporal_out

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
        if x.dim() == 4:
            B, T, num_spatial_patches, _ = x.shape

        if attn_mode == "spatial":
            x = x.view(B*T, num_spatial_patches, -1) # reshape to [B*T, H*W, d_model]

            assert (
                x.shape == (B*T, num_spatial_patches, self.d_model)
            ), f"x must have shape of {(B*T, num_spatial_patches, self.d_model)}, got {x.shape}"
        elif attn_mode == "temporal":
            if x.shape == (B, T, num_spatial_patches, self.d_model):
                x = (
                    x
                    .view(B, T, num_spatial_patches, self.d_model)
                    .transpose(1, 2).contiguous()
                    .view(B*num_spatial_patches, T, self.d_model)
                ) # reshape to [B*H*W, T, d_model]
            else:
                raise ValueError(
                    f"x must be in shape {(B*T, num_spatial_patches, self.d_model)}, got {x.shape}"
                )

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

        # Extend kv heads
        k = extend_kv_heads(
            input=k,
            repeats=self.heads_per_group,
            dim=2,
            use_mqa=use_mqa
        )
        v = extend_kv_heads(
            input=v,
            repeats=self.heads_per_group,
            dim=2,
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
            x (torch.Tensor): Input tensor of shape [B, T, H*W, d_model].
            grid_size (Tuple[int, int, int]): Grid size parameter for RoPE.
            use_mqa (bool): Whether to use multi-query attention or not.
            window_size (Tuple[int, int]): Window size for SWA.
            padding_mask (Optional[torch.Tensor]): Padding mask of shape [B, N].

        Returns:
            torch.Tensor: Output tensor of shape [B, T, H*W, d_model].
        """
        with autocast(device_type=device.type, dtype=dtype):
            spatial_out = self._spatial_attention(
                x=x,
                use_mqa=use_mqa,
                grid_shape=grid_size,
                window_size=window_size,
                padding_mask=padding_mask
            ) # [B*grid_T, H*W, d_model]
            spatial_out = spatial_out.view(
                x.size(0), grid_size[0], -1, self.d_model
            ) # [B, grid_T, H*W, d_model]
            temporal_out = self._temporal_attention(
                x=spatial_out,
                use_mqa=use_mqa,
                grid_shape=grid_size,
                window_size=window_size,
                padding_mask=padding_mask
            ) # [B*H*W, grid_T, d_model]
            spatio_temporal_out = temporal_out.view(
                x.size(0), grid_size[0], -1, self.d_model
            ) # [B, T, H*W, d_model]

            return self.w_o(spatio_temporal_out)
        

class SpatioTemporalAttentionBlock(nn.Module):
    """Attention block to apply attn, normalization, dropout, and residuals.
    
    Args:
        d_model (int): Dimensionality of model embeddings.
        num_heads (int): Number of attention heads for GQA.
        query_groups (int): Number of query groups for GQA.
        rope_theta (float): Exponential base of inv_freq for RoPE.
        patch_size (Tuple[int, int, int]): Patches for T, H, W dims.
        eps (float): Small epsilon value to maintain numerical stability in RMSNorm.
        dropout (float): Dropout probability.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        query_groups: int,
        rope_theta: float,
        patch_size: Tuple[int, int, int],
        eps: float,
        dropout: float
    ):
        super().__init__()

        self.attention = SpatioTemporalAttention(
            d_model=d_model,
            num_heads=num_heads,
            query_groups=query_groups,
            rope_theta=rope_theta,
            patch_size=patch_size
        )
        self.rms_norm = RMSNorm(
            d_model, eps
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        grid_size: Tuple[int, int, int],
        use_mqa: bool,
        window_size: Tuple[int, int],
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass of attention block.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, T, H*W, d_model].
            grid_size (Tuple[int, int, int]): Tuple containing T, H, W grids.
            use_mqa (bool): Whether to use multi-query attention or not.
            window_size (Tuple[int, int]): Left and right windows for SWA.
            padding_mask (Optional[torch.Tensor]): Padding tensor with shape [B, T*H*W]

        Returns:
            torch.Tensor: Output tensor with same shape as input.
        """
        with autocast(device_type=device.type, dtype=dtype):
            return x + self.dropout(
                self.attention(
                    self.rms_norm(x),
                    grid_size=grid_size,
                    use_mqa=use_mqa,
                    window_size=window_size,
                    padding_mask=padding_mask
                )
            )

def test_attention(use_pad: bool):
    d_model, num_heads, query_groups, rope_theta = 744, 124, 2, 10000.0
    patch_size = (2, 32, 32)
    attention = SpatioTemporalAttention(
        d_model, num_heads, query_groups, rope_theta, patch_size
    ).to(device)
    B, T, H, W = 4, 10, 144, 144
    pt, ph, pw = patch_size
    new_T, new_H, new_W = T // pt, H // ph, W // pw
    grid_size = (new_T, new_H, new_W)
    x = torch.randn(B, new_T, new_H * new_W, d_model).to(device)
    if use_pad:
        print("using padding")
        padding_mask = torch.randint(0, 2, (B, new_T*new_H*new_W), dtype=torch.bool).to(device)
    else:
        print("not using padding")
        padding_mask = None
    x_out = attention(
        x=x,
        grid_size=grid_size,
        use_mqa=False,
        window_size=(-1, -1),
        padding_mask=padding_mask
    )
    return x_out

def test_attention_block(use_pad: bool):
    d_model, num_heads, query_groups, rope_theta, eps, dropout = (
        744, 124, 2, 10000.0, 1e-7, 0.15
    )
    patch_size = (2, 32, 32)
    attn_block = SpatioTemporalAttentionBlock(
        d_model, num_heads, query_groups,
        rope_theta, patch_size, eps, dropout
    ).to(device)
    B, T, H, W = 4, 10, 144, 144
    pt, ph, pw = patch_size
    new_T, new_H, new_W = T//pt, H//ph, W//pw
    grid_size = (new_T, new_H, new_W)
    x = torch.randn(B, new_T, new_H * new_W, d_model).to(device)
    if use_pad:
        print("using padding")
        padding_mask = torch.randint(0, 2, (B, new_T*new_H*new_W), dtype=torch.bool).to(device)
    else:
        print("not using padding")
        padding_mask = None
    x_out = attn_block(
        x=x,
        grid_size=grid_size,
        use_mqa=False,
        window_size=(-1, -1),
        padding_mask=padding_mask
    )
    return x_out

if __name__ == "__main__":
    x = test_attention_block(use_pad=True) # [B, 10//2, 144//32 * 144//32, d_model]
    print(x.shape)
