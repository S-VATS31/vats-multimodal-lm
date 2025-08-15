from configs.setup_env import (
    device,
    dtype,
    use_flash_attn,
    flash_attn_varlen_qkvpacked_func,
    gpu_dtypes
)

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.checkpoint import checkpoint

from src.optimized_attention import RoPE
from src.rms_norm import RMSNorm
from src.ffn_block import FFNBlock
from configs.diffusion.text_encoder.model_args.model_args_large import ModelArgs

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
        use_proj_bias: bool,
        use_qkv_proj: bool
    ):
        super().__init__()

        if d_model % num_heads != 0:
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
            
        Requirements:
            - use_flash_must be True.
            - flash_attn_varlen_qkvpacked_func cannot be None.
            - device.type must be cuda.
            - query.dtype must be float16 or bfloat16.
            - key.dtype must be float16 or bfloat16.
            - value.dtype must be float16 or bfloat16.
            - query.device must be cuda.
            - key.device must be cuda.
            - value.device must be cuda.
        """
        if (
            use_flash_attn
            and flash_attn_varlen_qkvpacked_func is not None
            and device.type == "cuda"
            and query.dtype in gpu_dtypes
            and key.dtype in gpu_dtypes
            and value.dtype in gpu_dtypes
            and query.is_cuda
            and key.is_cuda
            and value.is_cuda
        ):
            if padding_mask is not None:
                # Stack qkv along 3rd dim
                qkv_stacked = torch.stack(
                    [query, key, value], dim=3
                ).contiguous() # [B, T, 3, num_heads, head_dim]

                assert (
                    qkv_stacked.shape == (B, T, 3, self.num_heads, self.head_dim)
                ), f"qkv_stacked must have shape of {(B, T, 3, self.num_heads, self.head_dim)}, got {qkv_stacked.shape}"
                assert (
                    qkv_stacked.is_contiguous()
                ), "qkv_stacked must be contiguous."
                
                # Ensure padding mask is a boolean tensor
                padding_mask = padding_mask.bool()
                assert (
                    padding_mask.shape == (B, T)
                ), f"padding_mask have shape of {(B, T)}, got {padding_mask.shape}"
                
                # Get seqlens from padding mask
                seqlens = padding_mask.sum(dim=1).int() # [B]
                assert (
                    seqlens.shape == (B,)
                ), f"seqlens must have shape of {(B,)}, got {seqlens.shape}"

                # Get max sequence length
                max_seqlen = seqlens.max().item()

                # Compute cumulative sequence lengths
                cu_seqlens = torch.cat([
                    torch.tensor([0], dtype=torch.int32).to(device), # shape: [1]
                    seqlens.cumsum(0) # [B]
                ], dim=0) # [B + 1]

                assert (
                    cu_seqlens.shape == (B + 1,)
                ), f"cu_seqlens must have shape of {(B + 1,)}, got {cu_seqlens.shape}"
                assert (
                    cu_seqlens.dtype == torch.int32
                ), f"cu_seqlens.dtype must be int32, got {cu_seqlens.dtype}"

                # Flatten padding mask
                valid_tokens = padding_mask.flatten() # [B*T]
                assert (
                    valid_tokens.shape == (B*T,)
                ), f"valid_tokens must have shape of {(B*T,)}, got {valid_tokens.shape}"

                # Index by valid tokens
                qkv_out = (
                        qkv_stacked.view(-1, self.num_heads, 3, self.head_dim)
                        .transpose(1, 2)
                        .contiguous()
                        )[valid_tokens] # [B * T, 3, num_heads, head_dim]
                
                assert (
                    qkv_out.shape == (B*T, 3, self.num_heads, self.head_dim)
                ), f"qkv_out must have shape of {(B*T, 3, self.num_heads, self.head_dim)}, got {qkv_out.shape}"
                assert (
                    qkv_out.is_contiguous()
                ), "qkv_out must be contiguous"
                
                # Get attention output
                attn_out = flash_attn_varlen_qkvpacked_func(
                    qkv_out,
                    cu_seqlens,
                    max_seqlen,
                    causal=False,
                    softmax_scale=self.softmax_scale,
                    window_size=(left_window, right_window),
                ) # [B * T, num_heads, head_dim]

                assert (
                    attn_out.shape == (B*T, self.num_heads, self.head_dim)
                ), f"attn_out must have shape of {(B*T, self.num_heads, self.head_dim)}, got {attn_out.shape}"

                # Reconstruct padding mask
                full_out = torch.zeros(
                    B * T,
                    self.num_heads, 
                    self.head_dim,  
                    dtype=attn_out.dtype, 
                ).to(attn_out.device)
                # Update valid and padded token positions where valid_tokens is a boolean tensor.
                full_out[valid_tokens] = attn_out
                attn_out = full_out.view(B, T, self.d_model) # Reshape output to [B, T, d_model]
                assert (
                    attn_out.shape == (B, T, self.d_model)
                )

                return attn_out
            
            # No padding mask, raise error
            else:
                raise ValueError(
                    "padding_mask must be given for text encoder."
                )
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
            # Ensure padding mask (B, T)
            assert (
                padding_mask.shape == (B, T)
            ), f"padding_mask must have shape of {(B, T)}, got {padding_mask.shape}"

            padding_mask = padding_mask.bool() # True = attend, False = don't attend

            # Ensure boolean mask
            assert (
                padding_mask.dtype == torch.bool
            ), f"padding_mask must have dtype of torch.bool, got {padding_mask.dtype}"

            attn_mask = padding_mask[:, None, None, :] # [B, 1, 1, T]
        else:
            attn_mask = None

        # PyTorch SDPA
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
        assert (
            x.dim() == 3
        ), f"x must have 3 dimensions, got {x.dim()}"
        B, T, _ = x.shape
        if T == 0:
            return torch.empty(B, 0, self.d_model, dtype=x.dtype).to(x.device)

        # Fused projection matrix
        if self.use_qkv_proj: 
            qkv = self.qkv_proj(x) # [B, T, num_heads * head_dim + 2 * query_groups * head_dim]
            assert (
                qkv.shape == (
                    B, T, self.num_heads * self.head_dim + 2 * self.query_groups * self.head_dim
                )
            ), (
                f"qkv must have shape of {(
                    B, T, self.num_heads * self.head_dim + 2 * self.query_groups * self.head_dim
                    )}, got {qkv.shape}"
            )

            # q shape: [B, T, num_heads * head_dim]
            # kv shape: [B, T, 2 * query_groups * head_dim]
            q, kv = torch.split(
                qkv, [self.num_heads * self.head_dim, 2 * self.query_groups * self.head_dim], dim=-1
            )
            assert (
                q.shape == (B, T, self.num_heads * self.head_dim)
            ), f"q must have shape of {(B, T, self.num_heads * self.head_dim)}, got {q.shape}"
            assert (
                kv.shape == (B, T, 2 * self.query_groups * self.head_dim)
            ), f"kv must have shape of {(B, T, 2 * self.query_groups * self.head_dim)}, got {kv.shape}"

            # Split once again
            # k, v shapes: [B, T, query_groups * head_dim]
            k, v = kv.chunk(chunks=2, dim=-1)
            assert (
                k.shape == (B, T, self.query_groups * self.head_dim)
            ), f"k must have shape of {(B, T, self.query_groups * self.head_dim)}, got {k.shape}"
            assert (
                v.shape == (B, T, self.query_groups * self.head_dim)
            ), f"v must have shape of {(B, T, self.query_groups * self.head_dim)}, got {v.shape}"
        
        # Seperate projection matrices
        else:
            # q shape: [B, T, num_heads * head_dim]
            # k, v shapes: [B, T, query_groups * head_dim]
            q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)

            assert (
                q.shape == (B, T, self.num_heads * self.head_dim)
            ), f"q must have shape of {(B, T, self.num_heads * self.head_dim)}"
            assert (
                k.shape == (B, T, self.query_groups * self.head_dim)
            ), f"k must have shape of {(B, T, self.query_groups * self.head_dim)}, got {k.shape}"
            assert (
                v.shape == (B, T, self.query_groups * self.head_dim)
            ), f"v must have shape of {(B, T, self.query_groups * self.head_dim)}, got {v.shape}"

        # Reshape into 4D tensors
        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.query_groups, self.head_dim)
        v = v.view(B, T, self.query_groups, self.head_dim)

        assert (
            q.shape == (B, T, self.num_heads, self.head_dim)
        ), f"q must have shape of {(B, T, self.num_heads, self.head_dim)}, got {q.shape}"
        assert (
            k.shape == (B, T, self.query_groups, self.head_dim)
        ), f"k must have shape of {(B, T, self.query_groups, self.head_dim)}, got {k.shape}"
        assert (
            v.shape == (B, T, self.query_groups, self.head_dim)
        ), f"v must have shape of {(B, T, self.query_groups, self.head_dim)}, got {v.shape}"

        # Apply RoPE to qk tensors
        q = self.rope(q)
        k = self.rope(k)

        assert (
            q.shape == (B, T, self.num_heads, self.head_dim)
        ), f"q must have shape of {(B, T, self.num_heads, self.head_dim)}, got {q.shape}"
        assert (
            v.shape == (B, T, self.query_groups, self.head_dim)
        ), f"v must have shape of {(B, T, self.query_groups, self.head_dim)}, got {v.shape}"

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

        assert (
            k.shape == (B, T, self.num_heads, self.head_dim) or
            k.shape == (B, T, 1, self.head_dim)
        ), (
            f"k must have shape of {(B, T, self.num_heads, self.head_dim)} or "
            f" {(B, T, 1, self.head_dim)}, got {k.shape}"
        )
        assert (
            v.shape == (B, T, self.num_heads, self.head_dim) or
            v.shape == (B, T, 1, self.head_dim)
        ), (
            f"v must have shape of {(B, T, self.num_heads, self.head_dim)} or "
            f" {(B, T, 1, self.head_dim)}, got {v.shape}"
        )

        return q, k, v, B, T

    def forward(
        self,
        x: torch.Tensor,
        left_window: int,
        right_window: int,
        enable_mqa: bool,
        padding_mask: Optional[torch.Tensor] = None,
        use_diffusion: bool = True,
        *,
        _return_qkv: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Forward pass of the attention layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, T, d_model].
            left_window (int): Left window for SWA.
            right_window (int): Right window for SWA.
            enable_mqa (bool): Whether to use MQA or not.
            padding_mask (Optional[torch.Tensor]): Padding tensor of shape [B, T].
            use_diffusion (bool): Whether this encoder will be used for diffusion or not.
                Will set left_window, right_window = -1, -1 if use_diffusion=True.
            _return_qkv (bool): Debugging feature; whether to return q, k, v tensors or not.

        Returns:
            Union:
                - torch.Tensor: Attention output.
                - Tuple:
                    - torch.Tensor: Attention output.
                    - torch.Tensor: Query tensor.
                    - torch.Tensor: Key tensor.
                    - torch.Tensor: Value tensor.
        """
        with autocast(device_type=device.type, dtype=dtype):
            q, k, v, B, T = self._setup_qkv(x, enable_mqa)
            # Global attention for diffusion (image-gen)
            if use_diffusion:
                left_window, right_window = -1, -1

            # Get attention output
            attn_out = self._optimized_attention(
                query=q, key=k, value=v, B=B, T=T,
                left_window=left_window,
                right_window=right_window,
                padding_mask=padding_mask
            )

            # Use only for debugging
            if _return_qkv:
                return self.o_proj(attn_out), q, k, v
            return self.o_proj(attn_out)


class AttentionBlock(nn.Module):
    """Attention block applying normalization, residuals, and dropout.
    
    Args:
        d_model (int): Dimensionality of model embeddings.
        num_heads (int): Number of attention heads for queries.
        query_groups (int): Number of query groups for GQA.
        theta (float): Exponential base of the inverse frequency for RoPE.
        softmax_scale (float): Factor to scale the attention computation before softmax by.
        use_proj_bias (bool): Whether to use bias for q, k, v projection matrices.
        use_qkv_proj (bool): Whether to use fused qkv proj or seperate projections.
        eps (float): Epsilon value to prevent numerical instability in RMSNorm.
        dropout (float): Dropout probability.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        query_groups: int,
        theta: float,
        softmax_scale: float,
        use_proj_bias: bool,
        use_qkv_proj: bool,
        eps: float,
        dropout: float,
    ):
        super().__init__()

        self.attention = Attention(
            d_model=d_model,
            num_heads=num_heads,
            query_groups=query_groups,
            theta=theta,
            softmax_scale=softmax_scale,
            use_proj_bias=use_proj_bias,
            use_qkv_proj=use_qkv_proj,
        )
        self.rms_norm = RMSNorm(
            d_model=d_model, eps=eps
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        left_window: int,
        right_window: int,
        enable_mqa: bool,
        padding_mask: Optional[torch.Tensor] = None,
        use_diffusion: bool = True,
    ) -> torch.Tensor:
        """Forward pass of attention block.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, T, d_model].
            left_window (int): Left window for SWA.
            right_window (int): Right window for SWA.
            enable_mqa (bool): Whether to use MQA or not.
            padding_mask (torch.Tensor): Padding tensor of shape [B, T].
            use_diffusion (bool): Whether this encoder is being used for diffusion or not.
                If True, both window sizes will be set to -1 for global attention.

        Returns:
            torch.Tensor: Output tensor of same shape as input.
        """
        with autocast(device_type=device.type, dtype=dtype):
            return x + self.dropout(self.rms_norm(self.attention(
                x=x,
                left_window=left_window,
                right_window=right_window,
                enable_mqa=enable_mqa,
                padding_mask=padding_mask,
                use_diffusion=use_diffusion,
            )))
        

class TransformerBlock(nn.Module):
    """  
    Args:
        d_model (int): Dimensionality of model embeddings.
        num_heads (int): Number of attention heads for queries.
        query_groups (int): Number of query groups for GQA.
        d_ffn (int): Dimensionality of the FFN.
        theta (float): Exponential base of the inverse frequency for RoPE.
        softmax_scale (float): Factor to scale the attention computation before softmax by.
        use_proj_bias (bool): Whether to use bias for q, k, v projection matrices.
        use_qkv_proj (bool): Whether to use fused qkv proj or seperate projections.
        eps (float): Epsilon value to prevent numerical instability in RMSNorm.
        dropout (float): Dropout probability.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        query_groups: int,
        d_ffn: int,
        theta: float,
        softmax_scale: float,
        use_proj_bias: bool,
        use_qkv_proj: bool,
        eps: float,
        dropout: float
    ):
        super().__init__()

        self.attention_block = AttentionBlock(
            d_model=d_model,
            num_heads=num_heads,
            query_groups=query_groups,
            theta=theta,
            softmax_scale=softmax_scale,
            use_proj_bias=use_proj_bias,
            use_qkv_proj=use_qkv_proj,
            eps=eps,
            dropout=dropout,
        )
        self.ffn_block = FFNBlock(
            d_model=d_model,
            d_ffn=d_ffn,
            dropout=dropout,
            eps=eps
        )
    
    def forward(
        self,
        x: torch.Tensor,
        left_window: int,
        right_window: int,
        enable_mqa: bool,
        padding_mask: Optional[torch.Tensor] = None,
        use_diffusion: bool = True
    ) -> torch.Tensor:
        """Forward pass of Transformer block layer.

        Args:
            x (torch.Tensor): Input tensor of shape [B, T, d_model].
            left_window (int): Left window for SWA.
            right_window (int): Right window for SWA.
            enable_mqa (bool): Whether to use MQA or not.
            padding_mask (torch.Tensor): Padding tensor of shape [B, T].
            use_diffusion (bool): Whether this encoder is being used for diffusion or not.
                If True, both window sizes will be set to -1 for global attention.

        Returns:
            torch.Tensor: Output tensor of same shape as input.
        """
        return self.ffn_block(self.attention_block(
            x=x,
            left_window=left_window,
            right_window=right_window,
            enable_mqa=enable_mqa,
            padding_mask=padding_mask,
            use_diffusion=use_diffusion
        ))
        
class TransformerTextEncoder(nn.Module):
    """Text encoder complete module.
    
    Args:
        model_args (ModelArgs): Model hyperparameters.
    """
    def __init__(self, model_args): # TODO: type hint to ModelArgs once complete
        super().__init__()

        self.model_args = model_args

        self.token_embedding = nn.Embedding(model_args.d_model, model_args.vocab_size)
        self.dropout = nn.Dropout(p=model_args.dropout)

        # Stack encoder blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=model_args.d_model,
                num_heads=model_args.num_heads,
                query_groups=model_args.query_groups,
                d_ffn=model_args.d_ffn,
                theta=model_args.rope_theta,
                softmax_scale=model_args.softmax_scale,
                use_proj_bias=model_args.use_proj_bias,
                use_qkv_proj=model_args.use_qkv_proj,
                eps=model_args.rms_norm_eps,
                dropout=model_args.dropout
            ).to(device) for _ in range(model_args.num_layers)
        ])

        # Initialize final RMSNorm
        self.rms_norm = RMSNorm(model_args.d_model, model_args.rms_norm_eps).to(device)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(module) -> None:
        pass

    def forward(
        self,
        input_ids: torch.LongTensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass encoder.
        
        Args:
            input_ids (torch.LongTensor): Input tensor of shape [B, T].
            padding_mask (Optional[torch.Tensor]): Padding tensor of shape [B, T].

        Returns:
            torch.Tensor
        """
        input_ids = input_ids.to(torch.int64, copy=False)
        assert (
            input_ids.dim() == 2
        ), f"input_ids must be of shape [B, T], got {input_ids.dim()} dimensions."
        assert (
            input_ids.dtype == torch.int64
        ), f"input_ids must have dtype of int64, got {input_ids.dtype}"

        # Apply embeddings
        x = self.dropout(self.token_embedding(input_ids)) # [B, T, d_model]
        assert (
            x.dim() == 3
        ), f"x must be of shape [B, T, d_model], got {x.dim()} dimensions."
        assert (
            input_ids.shape == x.shape[:2]
        ), "input_ids and x must have first two dims equal."

        # Loop through layers
        for layer in self.layers:
            if self.model_args.use_checkpointing:
                x = checkpoint(
                    layer,
                    x,
                    self.model_args.left_window,
                    self.model_args.right_window,
                    self.model_args.enable_mqa,
                    padding_mask,
                    self.model_args.use_diffusion,
                    use_reentrant=False
                )
            else:
                x = layer(
                    x=x,
                    left_window=self.model_args.left_window,
                    right_window=self.model_args.right_window,
                    enable_mqa=self.model_args.enable_mqa,
                    padding_mask=padding_mask,
                    use_diffusion=self.model_args.use_diffusion
                )

        assert (
            x.dim() == 3
        ), f"x must be a 3 dimensional tensor, got {x.dim()}"

        # Apply final RMSNorm
        x = self.rms_norm(x)

        assert (
            x.dim() == 3
        ), f"x must be a 3 dimensional tensor, got {x.dim()}"

        return x
    