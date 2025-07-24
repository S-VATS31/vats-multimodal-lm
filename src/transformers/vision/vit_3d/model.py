from configs.transformers.vision.vit_3d.setup_env import (
    device,
    dtype,
    use_flash_attn,
    flash_attn_varlen_qkvpacked_func
)

from typing import Tuple, Optional
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

from configs.transformers.vision.vit_3d.model_args.model_args_large import ModelArgs
from utils.setup_logger import setup_logger

# Set up logger
logger = setup_logger(name="train_logger", log_file="training.log", level=logging.INFO)

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

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

        # Projection using Conv3D.
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
        """
        Args:
            x (torch.Tensor): Input tensor of shape [B, C_in, T, H, W].
                Where:
                    T is time in frames.
                    H is height.
                    W is width.

        Returns:
            torch.Tensor: Patch embeddings of shape [B, N, d_model].
        """
        with autocast(device_type=x.device.type, dtype=x.dtype):
            _, _, T, H, W = x.shape
            pt, ph, pw = self.patch_size
            
            # Calculate padding needed to make T, H, W divisible by patch size
            pad_t = (pt - T % pt) % pt
            pad_h = (ph - H % ph) % ph
            pad_w = (pw - W % pw) % pw

            # Pad in reverse order: (W_left, W_right, H_left, H_right, T_left, T_right)
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_t), mode='constant', value=0)
            
            # Project patch embeddings and flatten
            x = self.projection(x) # [B, d_model, T, H, W]
            x = x.flatten(2).transpose(1, 2) # [B, N, d_model]

            # Apply normalization
            return self.rms_norm(self.dropout(x))


# TODO: implement 3D RoPE

class RMSNorm(nn.Module):
    """Apply RMSNorm to the features dimension.

    Formula:
        x_norm = x / sqrt(mean(x**2))
    
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
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(f"Expected d_model to be divisble by num_heads, got {d_model} % {num_heads} != 0")
        if num_heads % query_groups != 0:
            raise ValueError(f"Expected num_heads to be divisble by query_groups, got {num_heads} % {query_groups} != 0")
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.query_groups = query_groups
        self.head_dim = d_model // num_heads
        self.heads_per_group = num_heads // query_groups

        # QKV projection matrix
        self.w_qkv = nn.Linear(
            d_model,
            num_heads * self.head_dim + 2 * query_groups * self.head_dim,
            bias=False,
            dtype=dtype
        )

        # O projection matrix
        self.w_o = nn.Linear(
            d_model,
            d_model,
            bias=False,
            dtype=dtype
        )

        # TODO:
        # self.rope = RoPE3D(num_heads, rope_theta)

    def _extend_kv_heads(
        self,
        kv_tensor: torch.Tensor,
        heads_per_group: int,
        kv_heads_dim: int,
    ) -> torch.Tensor:
        """Extend kv heads to num_heads.
        
        Args:
            kv_tensor (torch.Tensor): Input key or value tensor.
            heads_per_group (int): Heads per group computed as num_heads // query_groups.
            kv_heads_dim (int): Dimension to be repeated.

        Returns:
            torch.Tensor: K or V tensor with kv heads dimension repeated, now equal to num_heads.
        """
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
        """
        # We can only use flash attn if the import works and device is cuda
        if use_flash_attn and device.type == "cuda":
            logger.info("Using Flash Attention 2 + GQA + SWA")
            # Check if kv heads need to be extended 
            if query.size(2) != key.size(2) or query.size(2) != value.size(2):
                # Extended kv heads for GQA
                key = self._extend_kv_heads(
                    kv_tensor=key, 
                    heads_per_group=self.heads_per_group,
                    kv_heads_dim=2
                )
                value = self._extend_kv_heads(
                    kv_tensor=value, 
                    heads_per_group=self.heads_per_group,
                    kv_heads_dim=2
                )

            # Concatenate tensors along 3rd dimension
            qkv_packed = torch.stack([query, key, value], dim=3).contiguous() # [B, N, num_heads, 3, head_dim]

            # Get cumulative sequence lengths
            cu_seqlens = torch.arange(0, (B + 1) * key.size(1), key.size(1), dtype=torch.int32).to(device) # [B + 1]

            # Get max sequence length and compute total tokens
            max_seqlen = key.size(1)
            total_tokens = B * max_seqlen

            # Flatten packed tensor
            qkv_flattened = qkv_packed.view(total_tokens, self.num_heads, 3, self.head_dim)

            # Call FlashAttention 2
            attn_out = flash_attn_varlen_qkvpacked_func(
                qkv_flattened,
                cu_seqlens,
                max_seqlen,
                causal=False,
                softmax_scale=1.0 / (self.head_dim ** 0.5),
                window_size=window_size,
            ) # [B * N, num_heads, head_dim]

            attn_out = attn_out.contiguous().view(B, N, -1) # [B, N, d_model]
            return attn_out
        
        # Either import didn't work, or no cuda; fallback to gqa/flash attn, w/o swa
        else:
            logger.info("Using PyTorch's SDPA + flash attention if available.")
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
        # Check if kv heads need to be extended
        if query.size(2) != key.size(2) or query.size(2) != value.size(2):
            # Extended kv heads for GQA
            key = self._extend_kv_heads(
                kv_tensor=key, 
                heads_per_group=self.heads_per_group,
                kv_heads_dim=2
            )
            value = self._extend_kv_heads(
                kv_tensor=value, 
                heads_per_group=self.heads_per_group,
                kv_heads_dim=2
            )
        
        attn_out = F.scaled_dot_product_attention(
            query, key, value,
            is_causal=False
        ) # [B, num_heads, N, head_dim]

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, self.d_model)
        return attn_out

    def forward(self, x: torch.Tensor, window_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """Perform forward pass of the attention layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, N, d_model].
            window_size (Tuple[int, int]): Window size for SWA.

        Returns:
            torch.Tensor: Output tensor of shape [B, N, d_model].
        """
        with autocast(device_type=device.type, dtype=dtype):
            B, N, _ = x.shape
            if x.shape != (B, N, self.d_model):
                raise ValueError(f"Expected x shape to be [B, N, d_model], got {x.shape}")
            if N == 0:
                # Unlikely for video transformers (only will occur if something goes totally wrong)
                return torch.empty(B, 0, self.d_model, device=x.device, dtype=x.dtype)

            # Project QKV
            qkv = self.w_qkv(x) # [B, N, num_heads * head_dim + 2 * query_groups * head_dim]

            # q shape: [B, N, num_heads * head_dim]
            # kv shape: [B, N, 2 * query_groups * head_dim]
            q, kv = torch.split(qkv, [self.num_heads * self.head_dim, 2 * self.query_groups * self.head_dim], dim=-1)

            # k shape: [B, N, query_groups * head_dim]
            # v shape: [B, N, query_groups * head_dim]
            k, v = torch.chunk(kv, 2, dim=-1)

            # Reshape into 4D tensors for GQA
            q = q.view(B, N, self.num_heads, self.head_dim) # [B, N, num_heads, head_dim]
            k = k.view(B, N, self.query_groups, self.head_dim) # [B, N, query_groups, head_dim]
            v = v.view(B, N, self.query_groups, self.head_dim) # [B, N, query_groups, head_dim]

            # TODO: Apply RoPE3D
            # q = self.rope(q)
            # k = self.rope(k)

            if window_size is not None:
                attn_out = self._optimized_attention(q, k, v, B, N, window_size)
            else:
                attn_out = self._grouped_query_attention(q, k, v, B, N)

            return self.w_o(attn_out) # Project output, [B, N, d_model]

class GatedFFN(nn.Module):
    """Gated FFN layer with SwiGLU activation.

    Formula:
        Dropout(SwiGLU(x)) = Dropout(w2 @ (swish(w1 @ x + b1) * (w3 @ x + b3)) + b2)
        
    Args:
        d_model (int): Dimensionality of the model's embeddings.
        d_ffn (int): Dimensionality of the FFN.
        dropout (float): Dropout probability.
    """
    def __init__(self, d_model: int, d_ffn: int, dropout: float):
        super().__init__()

        self.w1 = nn.Linear(d_model, d_ffn)
        self.w2 = nn.Linear(d_ffn, d_model) # Output projection matrix
        self.w3 = nn.Linear(d_model, d_ffn)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform forward pass of the Gated FFN layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, N, d_model].

        Returns:
            torch.Tensor: Output tensor passed through the FFN with same shape.
        """
        with autocast(device_type=device.type, dtype=dtype):
            return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
        
class AttentionBlock(nn.Module):
    """Attention block with attention, normalization, dropout, and residuals applied.
    
    Args:
        d_model (int): Dimensionality of the model's embeddings.
        num_heads (int): Number of attention heads.
        query_groups (int): Number of query groups for GQA.
        rope_theta (float): Theta hyperparameter for RoPE.
        eps (float): Small value to prevent numerical instability.
        dropout (float): Dropout probability
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        query_groups: int,
        rope_theta: float,
        eps: float,
        dropout: float,
    ):
        super().__init__()

        self.attention = Attention(d_model, num_heads, query_groups, rope_theta)
        self.rms_norm = RMSNorm(d_model, eps)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, window_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """Perform forward pass of the Attention layer.

        Args:
            x (torch.Tensor): Input tensor of shape [B, N, d_model].
            window_size (Optional[Tuple[int, int]]): Window size for SWA.

        Returns:
            torch.Tensor: Output tensor of shape [B, N d_model].
        """
        with autocast(device_type=device.type, dtype=dtype):
            return x + self.dropout(self.attention(self.rms_norm(x), window_size))
    
class GatedFFNBlock(nn.Module):
    """Gated FFN block with a pass through the FFN, dropout, normalization, and residuals applied.
    
    Args:
        d_model (int): Dimensionality of the model's embeddings.
        d_ffn (int): Dimensionality of the FFN.
        dropout (float): Dropout probability.
        eps (float): Small epsilon value to prevent numerical instability.
    """
    def __init__(self, d_model: int, d_ffn: int, dropout: float, eps):
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
    ):
        super().__init__()

        self.attention_block = AttentionBlock(d_model, num_heads, query_groups, rope_theta, eps, dropout)
        self.gated_ffn_block = GatedFFNBlock(d_model, d_ffn, dropout, eps)

    def forward(self, x: torch.Tensor, window_size: Optional[Tuple[int, int]] = None):
        """Perform forward pass of the Transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape [B, N, d_model].
            window_size (Optional[Tuple[int, int]]): Window size for SWA.

        Returns:
            torch.Tensor: Output tensor of shape [B, N, d_model].
        """
        with autocast(device_type=device.type, dtype=dtype):
            return self.gated_ffn_block(self.attention_block(x, window_size))
        
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
            )
        ])

        # Set up RMSNorm
        self.rms_norm = RMSNorm(model_args.d_model, model_args.rms_norm_eps)