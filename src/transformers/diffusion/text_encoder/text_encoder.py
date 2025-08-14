from configs.setup_env import (
    device,
    dtype,
    use_flash_attn,
    flash_attn_varlen_qkvpacked_func
)

import warnings
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

from src.optimized_attention import RoPE
from src.rms_norm import RMSNorm
from src.swiglu_activation import SwiGLUActivation
from src.ffn_block import FFNBlock
from configs.transformers.diffusion.model_args.model_args_medium import ModelArgs


class Attention(nn.Module):
    """Attention module for encoder.

    Args:
        d_model (int): Dimensionality of model embeddings.
        num_heads (int): Number of attention heads for queries.
        query_groups (int): Number of query groups for GQA.
        theta (float): Exponential base for RoPE inverse frequency.
        softmax_scale (float): Softmax scale for attention computation.
        use_proj_bias (bool): Whether to use bias for projection matrices.
        use_qkv_proj (bool): Whether to use fused QKV projection or single projections.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        query_groups: int,
        theta: float,
        softmax_scale: float,
        use_proj_bias: bool = False,
        use_qkv_proj: bool = True,
    ):
        super().__init__()

        if self.d_model % num_heads != 0:
            raise ValueError(
                f"d_model must be divisble by num_heads, got {d_model} % {num_heads} != 0."
            )
        if num_heads % query_groups != 0:
            raise ValueError(
                f"num_heads must be divisble by query_groups, got {num_heads} % {query_groups} != 0"
            )
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.query_groups = query_groups
        self.softmax_scale = softmax_scale
        self.use_qkv_proj = use_qkv_proj
        self.head_dim = d_model // num_heads
        self.heads_per_group = num_heads // query_groups

        # qkv projection
        if use_qkv_proj:
            self.qkv_proj = nn.Linear(
                d_model,
                num_heads * self.head_dim + 2 * query_groups * self.head_dim,
                bias=use_proj_bias,
                dtype=dtype
            )
        else:
            # query projection
            self.q_proj = nn.Linear(
                d_model,
                num_heads * self.head_dim,
                bias=use_proj_bias,
                dtype=dtype,
            )
            # key projection
            self.k_proj = nn.Linear(
                d_model,
                query_groups * self.head_dim,
                bias=use_proj_bias,
                dtype=dtype
            )
            # value projection
            self.v_proj = nn.Linear(
                d_model,
                query_groups * self.head_dim,
                bias=use_proj_bias,
                dtype=dtype
            )

        # output projection
        self.o_proj = nn.Linear(
            d_model,
            d_model,
            bias=use_proj_bias,
            dtype=dtype,
        )

        self.rope = RoPE(self.head_dim, theta)

    def _extend_kv_heads(
        self,
        input: torch.Tensor,
        heads_per_group: int,
        dim: int,
        enable_mqa: bool
    ) -> torch.Tensor:
        """Extend key value heads to num_heads (query heads) for GQA.
        
        Args:
            input (torch.Tensor): Input key or value tensor.
            heads_per_groups (int): Heads per group computed as num_heads // query_groups.
            dim (int): Dimension to be repeated.
            enable_mqa (bool): Whether to use MQA or not. 
                Contrainsts for MQA: input.size(dim) must be 1 for MQA and enable_mqa must be True.

        Returns:
            torch.Tensor: Tensor with key or value head repeated to num_heads
        """
        if input.size(dim) == 1 and enable_mqa:
            return input
        return input.repeat_interleave(repeats=heads_per_group, dim=dim)

    def _optimized_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        B: int,
        T: int,
        left_window: int,
        right_window: int,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Optimized attention leveraging global flash attention.
        
        Args:
            query (torch.Tensor): Query tensor of shape [B, T, num_heads, head_dim].
            key (torch.Tensor): Key tensor of shape [B, T, num_heads, head_dim] or [B, T, 1, head_dim].
            value (torch.Tensor): Value tensor of shape [B, T, num_heads, head_dim] or [B, T, 1, head_dim].
            B (int): Batch size.
            T (int): Sequence length.
            left_window (int): Left window for sliding window attention. Must be -1 for diffusion text encoder.
            right_window (int): Right window for sliding window attention. Must be -1 for diffusion text encoder.
            padding_mask (Optional[torch.Tensor]): Padding tensor of shape [B, T].

        Returns:
            torch.Tensor: Output tensor of shape [B, T, d_model].
        """
        if (
            
        ):
            pass
        else:
            return self._torch_attention(query, key, value, B, T, padding_mask)
        

    def _torch_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        B: int,
        T: int,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Fallback attention method utilizing PyTorch SDPA + GQA.

        Args:
            query (torch.Tensor): Query tensor of shape [B, T, num_heads, head_dim].
            key (torch.Tensor): Key tensor of shape [B, T, num_heads, head_dim] or [B, T, 1, head_dim].
            value (torch.Tensor): Value tensor of shape [B, T, num_heads, head_dim] or [B, T, 1, head_dim].
            B (int): Batch size.
            T (int): Sequence length.
            padding_mask (torch.Tensor): Padding tensor of shape [B, T]

        Returns:
            torch.Tensor: Output tensor of shape [B, T, d_model].

        Notes:
            We expect same input tensor shape as optimized attention, but we transpose dimensions 1 and 2 of the qkv tensors
            giving us a shape of [B, num_heads, T, head_dim] for all qkv tensors.
        """
        # Transpose to [B, num_heads, T, head_dim]
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        assert (
            query.shape == (B, self.num_heads, T, self.head_dim)
        ), f"query must have shape of {(B, self.num_heads, T, self.head_dim)}, got {query.shape}"
        assert (
            key.shape == (B, self.num_heads, T, self.head_dim) or
            key.shape == (B, 1, T, self.head_dim)
        ), (
            f"key must have shape of {(B, self.num_heads, T, self.head_dim)} or "
            f"shape of {(B, 1, T, self.head_dim)}, got {key.shape}"
        )
        assert (
            value.shape == (B, self.num_heads, T, self.head_dim) or
            value.shape == (B, 1, T, self.head_dim)
        ), (
            f"value must have shape of {(B, self.num_heads, T, self.head_dim)} or "
            f"shape of {(B, 1, T, self.head_dim)}, got {value.shape}"
        )

        # Create attention mask for PyTorch SDPA
        if padding_mask is not None:
            padding_mask = padding_mask.bool() # True = attend, False = don't attend
            attn_mask = padding_mask[:, None, None, :] # [B, 1, 1, T]
        else:
            attn_mask = None

        # Scaled Dot Product Attention
        attn_out = F.scaled_dot_product_attention(
            query=query, 
            key=key, 
            value=value,
            attn_mask=attn_mask, 
            is_causal=False, 
            scale=self.softmax_scale
        ) # [B, num_heads, T, head_dim]

        assert (
            attn_out.shape == (B, self.num_heads, T, self.head_dim)
        ), f"attn_out must have shape of {(B, self.num_heads, T, self.head_dim)}, got {attn_out.shape}"

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        assert (
            attn_out.shape == (B, T, self.d_model)
        ), f"attn_out must have shape of {(B, T, self.d_model)}, got {attn_out.shape}"

        return attn_out
    
    def _setup_qkv(
        self,
        x: torch.Tensor,
        enable_mqa: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        """Set up query, key, and value vectors for attention.

        Args:
            x (torch.Tensor): Input tensor of shape [B, T, d_model].

        Returns:
            Tuple:
                - torch.Tensor: Query tensor of shape [B, T, num_heads, head_dim]
                - torch.Tensor: Key tensor of shape [B, T, num_heads, head_dim] or [B, T, 1, head_dim].
                - torch.Tensor: Value tensor of shape [B, T, num_heads, head_dim] or [B, T, 1, head_dim].
                - int: Batch size.
                - int: Sequence length.
        """
        B, T, _ = x.shape

        # Fused projection matrix
        if self.use_qkv_proj:
            qkv = self.qkv_proj(x) # [B, T, num_heads * head_dim + 2 * query_groups * head_dim]

            # q shape: [B, T, num_heads * head_dim]
            # kv shape: [B, T, 2 * query_groups * head_dim]
            q, kv = torch.split(
                qkv, [self.num_heads * self.head_dim + 2 * self.query_groups * self.head_dim], dim=-1
            )

            # Split once again
            # k, v shapes: [B, T, query_groups * head_dim]
            k, v = kv.chunk(chunks=2, dim=-1)
        
        # Seperate projection matrices
        else:
            # q shape: [B, T, num_heads * head_dim]
            # k, v shapes: [B, T, query_groups * head_dim]
            q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Reshape into 4D tensors
        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.query_groups, self.head_dim)
        v = v.view(B, T, self.query_groups, self.head_dim)

        # Apply RoPE to qk tensors
        q = self.rope(q)
        k = self.rope(k)

        # Extend KV heads for GQA
        k = self._extend_kv_heads(
            input=k,
            heads_per_group=self.heads_per_group,
            dim=2,
            enable_mqa=enable_mqa
        )
        v = self._extend_kv_heads(
            input=v,
            heads_per_group=self.heads_per_group,
            dim=2,
            enable_mqa=enable_mqa
        )

        return q, k, v, B, T

    def forward(
        self,
        x: torch.Tensor,
        left_window: int,
        right_window: int,
        enable_mqa: bool,
        padding_mask: Optional[torch.Tensor] = None,
        use_diffusion: bool = True
    ) -> torch.Tensor:
        """Forward pass of the attention layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, T, d_model].
            left_window (int): Left window for SWA.
            right_window (int): Right window for SWA.
            enable_mqa (bool): Whether to use MQA or not.
            padding_mask (Optional[torch.Tensor]): Padding tensor of shape [B, T].
            use_diffusion (bool): Whether this encoder will be used for diffusion or not.
                Will set left_window, right_window = -1, -1 if True.

        Returns:
            torch.Tensor: Output tensor of shape: [B, T, d_model].
        """
        with autocast(device_type=device.type, dtype=dtype):
            q, k, v, B, T = self._setup_qkv(x, enable_mqa)
            # Set window sizes to -1, -1 for SWA for diffusion, this is done because in the 
            # flash attn 2 func a window size of (-1, -1) equates to global attention.
            if use_diffusion:
                warnings.warn(
                    "Diffusion being used, setting attention to global attention."
                )
                left_window, right_window = -1, -1 # global attn window sizes

            # Get attention output
            attn_out = self._optimized_attention(
                query=q, key=k, value=v, B=B, T=T,
                left_window=left_window,
                right_window=right_window,
                padding_mask=padding_mask
            )

            return self.o_proj(attn_out)


class AttentionBlock(nn.Module):
    pass

class TransformerBlock(nn.Module):
    pass

class TransformerTextEncoder(nn.Module):
    pass
