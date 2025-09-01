from configs.setup_env import (
    device,
    dtype,
    gpu_dtypes,
    use_flash_attn,
    flash_attn_varlen_qkvpacked_func
)

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

from src.rms_norm import RMSNorm

class CrossAttention(nn.Module):
    """Cross attention layer acting as a bridge between input embeddings and video to be generated.
    
    Args:
        d_model (int): Dimensionality of model embeddings.
        num_heads (int): Number of attention heads.
        softmax_scale (float): Attention scores scaling factor.
        use_proj_bias (bool): Whether to use projection bias or not.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        softmax_scale: float,
        use_proj_bias: bool,
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model must be divisble by num_heads, got {d_model} % {num_heads} != 0."
            )
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.softmax_scale = softmax_scale
        self.head_dim = d_model // num_heads

        # Projections
        self.q_proj = nn.Linear(d_model, d_model, bias=use_proj_bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=use_proj_bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=use_proj_bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=use_proj_bias)

    def _optimized_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Optimized attention mechanism leveraging flash attention.
        
        Args:
            query (torch.Tensor): Query tensor of shape [B, T, num_heads, head_dim].
            key (torch.Tensor): Key tensor of shape [B, T, num_heads, head_dim].
            value (torch.Tesnor): Value tensor of shape [B, T, num_heads, head_dim].
            padding_mask (torch.Tensor): Padding tensor of shape [B, T_k].

        Returns:
            torch.Tensor: Output tensor of shape [B, T, d_model].
        """
        if (
            flash_attn_varlen_qkvpacked_func is not None
            and use_flash_attn and device.type == "cuda"
            and query.dtype in gpu_dtypes
            and key.dtype in gpu_dtypes
            and value.dtype in gpu_dtypes
            and query.is_cuda and key.is_cuda and value.is_cuda
        ):
            if padding_mask is not None:
                pass
            else:
                raise ValueError("must using padding mask")
        else:
            return self._torch_attention(query, key, value, padding_mask)

    def _torch_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Fallback cross attention leveraging PyTorch scaled dot product attention.
        
        Args:
            query (torch.Tensor): Query tensor of shape [B, T_q, num_heads, head_dim].
                Q tensor represents image to be generated -> getting info from text embeddings (K, V).
            key (torch.Tensor): Key tensor of shape [B, T_k, num_heads, head_dim].
            value (torch.Tensor): Value tensor of shape [B, T_k, num_heads, head_dim].
                K, V tensors represent text embeddings -> giving info to image to be generated (Q).
            padding_mask: Optional[torch.Tensor]: Padding mask of shape [B, T_k].

        Returns:
            torch.Tensor: Attention output of shape [B, H*W, d_model].
        """
        # Shapes after transpose:
        # q: [B, num_heads, T_q, head_dim]
        # k, v: [B, num_heads, T_k, head_dim]
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Set up padding mask
        if padding_mask is not None:
            padding_mask = padding_mask.bool()
            assert (
                padding_mask.shape == (query.size(0), key.size(2))
            ), f"expected {query.size(0), key.size(2)}, got {padding_mask.shape}"
            attn_mask = padding_mask[:, None, None, :] # [B, 1, 1, T]
        else:
            attn_mask = None

        if attn_mask is not None:
            assert (
                attn_mask.shape == (query.size(0), 1, 1, key.size(2))
            ), f"expected: {(query.size(0), 1, 1, key.size(2))}, got {attn_mask.shape}"
            assert (
                attn_mask.dtype == torch.bool
            ), f"expected torch.bool, got {attn_mask.dtype}"

        # Scaled dot product attention
        attn_out = F.scaled_dot_product_attention(
            query=query,
            key=key, 
            value=value,
            attn_mask=attn_mask,
            is_causal=False,
            scale=self.softmax_scale,
            enable_gqa=False,
        ) # [B, num_heads, T_q, head_dim]

        assert (
            attn_out.shape == (query.size(0), self.num_heads, query.size(2), self.head_dim)
        ), f"expected {(query.size(0), self.num_heads, query.size(2), self.head_dim)}, got {attn_out.shape}"

        # Reshape to 3D output
        attn_out = (
            attn_out
            .transpose(1, 2)
            .contiguous()
            .view(query.size(0), query.size(2), self.d_model)
        )

        assert (
            attn_out.shape == (query.size(0), query.size(2), self.d_model)
        ), f"expected {(query.size(0), query.size(2), self.d_model)}, got {attn_out.shape}"

        return attn_out

    def _setup_qkv(
        self,
        x: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Set up query, key, and value tensors for attention.
        
        Args:
            x (torch.Tensor): Image tokens of shape [B, T, H*W, d_model].
            text_embeddings (torch.Tensor): Text tokens of shape [B, T, d_model].

        Returns:
            Tuple:
                - torch.Tensor: Query tensor of shape [B, T_q, num_heads, head_dim].
                - torch.Tensor: Key tensor of shape [B, T_k, num_heads, head_dim].
                - torch.Tensor: Value tensor of shape [B, T_k, num_heads, head_dim].
        """
        # T_q = H*W = num_spatial_patches
        B, T_q, _ = x.shape
        _, T_k, _ = text_embeddings.shape

        assert (
            x.size(0) == text_embeddings.size(0)
        ), "First dims of inputs should be equal."
        assert (
            x.size(-1) == text_embeddings.size(-1)
        ), "Last dims of inputs should be equal."

        # Project to q, k, v
        q = self.q_proj(x) # [B, T_q, d_model]
        k = self.k_proj(text_embeddings) # [B, T_k, d_model]
        v = self.v_proj(text_embeddings) # [B, T_k, d_model]

        assert (
            q.shape == (B, T_q, self.d_model)
        ), f"q must have shape of {(B, T_q, self.d_model)}"
        assert (
            k.shape == (B, T_k, self.d_model)
        ), f"k must have shape of {(B, T_k, self.d_model)}"
        assert (
            v.shape == (B, T_k, self.d_model)
        ), f"v must have shape of {(B, T_k, self.d_model)}"

        # Reshape to 4D tensors for attention computation
        q = q.view(B, T_q, self.num_heads, self.head_dim)
        k = k.view(B, T_k, self.num_heads, self.head_dim)
        v = v.view(B, T_k, self.num_heads, self.head_dim)

        assert (
            q.shape == (B, T_q, self.num_heads, self.head_dim)
        ), f"q must have shape of {(B, T_q, self.num_heads, self.head_dim)}, got {q.shape}"
        assert (
            k.shape == (B, T_k, self.num_heads, self.head_dim)
        ), f"k must have shape of {(B, T_k, self.num_heads, self.head_dim)}, got {k.shape}"
        assert (
            v.shape == (B, T_k, self.num_heads, self.head_dim)
        ), f"v must have shape of {(B, T_k, self.num_heads, self.head_dim)}, got {v.shape}"

        return q, k, v

    def _cross_attention(
        self,
        x: torch.Tensor,
        text_embeddings: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply cross attention.
        
        Args:
            x (torch.Tensor): Image tokens of shape [B, H*W, d_model].
            text_embeddings (torch.Tensor): Text tokens of shape [B, T, d_model].
            padding_mask (Optional[torch.Tensor]): Padding tensor of shape [B, T].

        Returns:
            torch.Tensor: Output of cross attention layer, shape: [B, H*W, d_model].
        """
        q, k, v = self._setup_qkv(x, text_embeddings)
        # Get cross attention output
        cross_attn_out = self._optimized_attention(q, k, v, padding_mask)
        
        return cross_attn_out

    def forward(
        self,
        x: torch.Tensor,
        text_embeddings: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply cross attention.
        
        Args:
            x (torch.Tensor): Image tokens of shape [B, H*W, d_model].
            text_embeddings (torch.Tensor): Text tokens of shape [B, T, d_model].
            padding_mask (Optional[torch.Tensor]): Padding tensor of shape [B, T].

        Returns:
            torch.Tensor: Output of cross attention layer, shape: [B, H*W, d_model].
        """
        with autocast(device_type=device.type, dtype=dtype):
            cross_attn_out = self._cross_attention(
                x, 
                text_embeddings=text_embeddings,
                padding_mask=padding_mask
            ) # [B, H*W, d_model]
            
            return self.o_proj(cross_attn_out) # [B, H*W, d_model]
