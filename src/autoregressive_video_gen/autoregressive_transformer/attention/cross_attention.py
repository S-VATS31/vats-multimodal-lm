from configs.setup_env import (
    device,
    dtype,
    gpu_dtypes,
    use_flash_attn,
    flash_attn_varlen_qkvpacked_func
)

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

from src.rms_norm import RMSNorm
from utils.attention_utils import (
    setup_projections,
    extend_kv_heads,
    apply_qk_norm
)

# TODO: implement or remove optimized attention

class FactorizedCrossAttention(nn.Module):
    """Factorized cross attention layer for video generation.
    
    Args:
        d_model (int): Dimensionality of model embeddings.
        num_heads (int): Number of attention heads for queries.
        query_groups (int): Number of query groups for GQA.
        softmax_scale (float): Scale for attention scores.
        use_proj_bias (bool): Whether to use projection bias or not.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        query_groups: int,
        softmax_scale: float,
        use_proj_bias: bool
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                f"expected d_model % num_heads == 0, got {d_model} % {num_heads} != 0."
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.query_groups = query_groups
        self.softmax_scale = softmax_scale
        self.head_dim = d_model // num_heads
        self.heads_per_group = num_heads // query_groups

        # Get q, k, v projections
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = setup_projections(
            d_model=d_model, 
            num_heads=num_heads, 
            head_dim=self.head_dim,
            use_fused_proj=False, 
            use_gqa=True, 
            use_proj_bias=use_proj_bias, 
            query_groups=query_groups
        )

    def _optimized_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Optimized cross attention leveraging flash attention.
        
        Args:
            query (torch.Tensor): Query tensor for video to be generated.
            key (torch.Tensor): Key tensor for input text tokens.
            value (torch.Tensor): Value tensor for input text tokens.
            padding_mask (torch.Tensor): Padding mask of shape [B, T_tokens].

        Args:
            torch.Tensor: Attention output of shape [B, T_frames, H*W, d_model].
        """
        if (
            flash_attn_varlen_qkvpacked_func is not None
            and use_flash_attn and device.type == "cuda"
            and query.dtype in gpu_dtypes
            and key.dtype in gpu_dtypes
            and value.dtype in gpu_dtypes
            and query.is_cuda and key.is_cuda and value.is_cuda
        ):
            pass
        else:
            return self._torch_attention(
                query=query.transpose(1, 2),
                key=key.transpose(1, 2),
                value=value.transpose(1, 2),
                padding_mask=padding_mask
            )

    def _torch_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Fallback to optimized attention using PyTorch dot product attention.
        
        Args:
            query (torch.Tensor): Query tensor for video to be generated.
            key (torch.Tensor): Key tensor for input text tokens.
            value (torch.Tensor): Value tensor for input text tokens.
            padding_mask (torch.Tensor): Padding mask of shape [B, T_tokens].

        Args:
            torch.Tensor: Attention output of shape [B, T_frames, H*W, d_model].
        """
        
        # Handle padding mask
        if padding_mask is not None:
            assert (
                padding_mask.shape == (key.size(0), key.size(2))
            ), f"expected {key.size(0), key.size(2)}, got {padding_mask.shape}"
            # [B, T_tokens], torch.BoolTensor
            padding_mask = padding_mask.bool()
            padding_mask = padding_mask[:, None, None, :] # [B, 1, 1, T_tokens]
            assert (
                padding_mask.shape == (key.size(0), 1, 1, key.size(2))
            ), f"expected {(key.size(0), 1, 1, key.size(2))}, got {padding_mask.shape}"
            # if spatial: [B*T_frames, 1, H*W, T_tokens]
            # if temporal: [B*H*W, 1, T_frames, T_tokens]
            attn_mask = padding_mask.expand(
                query.size(0), 1, query.size(2), key.size(2)
            )
            assert (
                attn_mask.shape == (query.size(0), 1, query.size(2), key.size(2))
            ), f"expected {(query.size(0), 1, query.size(2), key.size(2))}, got {attn_mask.shape}"
        else:
            attn_mask = None

        # Manually expand to num_heads
        if attn_mask is not None:
            attn_mask = attn_mask.expand(
                query.size(0), self.num_heads, query.size(2), key.size(2)
            )
            assert (
                attn_mask.shape == (
                    query.size(0), self.num_heads, query.size(2), key.size(2)
                )
            ), f"expected {(
                query.size(0), self.num_heads, query.size(2), key.size(2)
            )}, got {attn_mask.shape}"

        # Get attn out
        attn_out = F.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            is_causal=False,
            scale=self.softmax_scale,
            enable_gqa=False
        ) # [:, num_heads, :, head_dim]

        assert (
            attn_out.shape == (
                query.size(0), self.num_heads, query.size(2), self.head_dim
            )
        ), f"expected {(
            query.size(0), self.num_heads, query.size(2), self.head_dim
        )}, got {attn_out.shape}"

        attn_out = (
            attn_out
            .transpose(1, 2)
            .contiguous()
            .view(query.size(0), query.size(2), -1)
        )

        return attn_out

    def _spatial_cross_attention(
        self,
        x: torch.Tensor,
        text_embeddings: torch.Tensor,
        use_mqa: bool,
        use_qk_norm: bool,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Spatial cross attention applied as 1 x H x W.

        Args:
            x (torch.Tensor): Input image tensor of shape [B, T_frames, H*W, d_model].
            text_embeddings (torch.Tensor): Input text tensor of shape [B, T_tokens, d_model].
            use_mqa (bool): Whether to use MQA or not.
            use_qk_norm (bool): Whether to use QK normalization or not.
            padding_mask (Optional[torch.Tensor]): Padding tensor of shape [B, T_tokens].

        Returns:
            torch.Tensor: Spatial cross attention output of shape [B*T_frames, H*W, d_model].
        """
        q, k, v = self._setup_spatial_qkv(
            x,
            text_embeddings=text_embeddings,
            use_mqa=use_mqa,
            use_qk_norm=use_qk_norm
        )
        spatial_out = self._optimized_attention(
            query=q,
            key=k,
            value=v,
            padding_mask=padding_mask
        )

        return spatial_out

    def _temporal_cross_attention(
        self,
        x: torch.Tensor,
        text_embeddings: torch.Tensor,
        use_mqa: bool,
        use_qk_norm: bool,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Temporal cross attention applied as T x 1 x 1.

        Args:
            x (torch.Tensor): Input image tensor of shape [B, T_frames, H*W, d_model].
            text_embeddings (torch.Tensor): Input text tensor of shape [B, T_tokens, d_model].
            use_mqa (bool): Whether to use MQA or not.
            use_qk_norm (bool): Whether to use QK normalization or not.
            padding_mask (Optional[torch.Tensor]): Padding tensor of shape [B, T_tokens].

        Returns:
            torch.Tensor: Spatial cross attention output of shape [B*H*W, T_frames, d_model].
        """
        q, k, v = self._setup_temporal_qkv(
            x,
            text_embeddings=text_embeddings,
            use_mqa=use_mqa,
            use_qk_norm=use_qk_norm
        )
        temporal_out = self._optimized_attention(
            query=q,
            key=k,
            value=v,
            padding_mask=padding_mask
        )

        return temporal_out

    def _setup_spatial_qkv(
        self,
        x: torch.Tensor,
        text_embeddings: torch.Tensor,
        use_mqa: bool,
        use_qk_norm: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Set up spatial QKV tensors.
        
        Args:
            x (torch.Tensor): Input image tensor of shape [B, T_frames, H*W, d_model].
            text_embeddings (torch.Tensor): Input text tensors of shape [B, T_tokens, d_model].
            use_mqa (bool): Whether to use MQA or not.
            use_qk_norm (bool): Whther to use QK normalization or not.

        Returns:
            Tuple:
                - torch.Tensor: Query tensor of shape [B*T, H*W, num_heads, head_dim].
                - torch.Tensor: Key tensor of shape [B*T, H*W, num_heads or 1, head_dim].
                - torch.Tensor: Value tensor of shape [B*T, H*W, num_heads or 1, head_dim].
        """
        B, T_frames, num_spatial_patches, _ = x.shape
        _, T_tokens, _ = text_embeddings.shape

        assert (
            x.size(0) == text_embeddings.size(0)
        ), "expected x.size(0) == text_embeddings.size(0)"
        assert (
            x.size(-1) == text_embeddings.size(-1)
        ), "expected x.size(-1) == text_embeddings.size(-1)"

        # Reshape image tensor to [B*T_frames, H*W, d_model] for spatial attn
        x = x.view(-1, num_spatial_patches, self.d_model)

        # Project input tensors
        # q: [B*T_frames, H*W, d_model]; k, v shape: [B, T_tokens, query_groups*head_dim]
        q = self.q_proj(x)
        k = self.k_proj(text_embeddings)
        v = self.v_proj(text_embeddings)

        assert (
            q.shape == (B*T_frames, num_spatial_patches, self.d_model)
        ), f"expected {(B*T_frames, num_spatial_patches, self.d_model)}, got {q.shape}"
        assert (
            k.shape == (B, T_tokens, self.query_groups*self.head_dim)
        ), f"expected {(B, T_tokens, self.query_groups*self.head_dim)}, got {k.shape}"
        assert (
            v.shape == (B, T_tokens, self.query_groups*self.head_dim)
        ), f"expected {(B, T_tokens, self.query_groups*self.head_dim)}, got {v.shape}"

        # Reshape into 4D tensors
        q = q.view(B*T_frames, num_spatial_patches, self.num_heads, self.head_dim)
        k = k.view(B, T_tokens, self.query_groups, self.head_dim)
        v = v.view(B, T_tokens, self.query_groups, self.head_dim)

        assert (
            q.shape == (B*T_frames, num_spatial_patches, self.num_heads, self.head_dim)
        ), f"expected {(B*T_frames, num_spatial_patches, self.num_heads, self.head_dim)}, got {q.shape}"
        assert (
            k.shape == v.shape == (B, T_tokens, self.query_groups, self.head_dim)
        ), f"expected {(B, T_tokens, self.query_groups, self.head_dim)}, got {k.shape}, {v.shape}"

        # Apply QK normalization
        if use_qk_norm:
            q, k = apply_qk_norm(q, k)

        # Extend KV heads if using GQA
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

        assert (
            k.size(2) == v.size(2) == self.num_heads or
            k.size(2) == v.size(2) == 1
        ), f"got {k.size(2)}, {v.size(2)}"
        
        return q, k, v

    def _setup_temporal_qkv(
        self,
        x: torch.Tensor,
        text_embeddings: torch.Tensor,
        use_mqa: bool,
        use_qk_norm: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Set up spatial QKV tensors.
        
        Args:
            x (torch.Tensor): Input image tensor of shape [B, T_frames, H*W, d_model].
            text_embeddings (torch.Tensor): Input text tensors of shape [B, T_tokens, d_model].
            use_mqa (bool): Whether to use MQA or not.
            use_qk_norm (bool): Whther to use QK normalization or not.

        Returns:
            Tuple:
                - torch.Tensor: Query tensor of shape [B*H*W, T, num_heads, head_dim].
                - torch.Tensor: Key tensor of shape [B*H*W, T, num_heads or 1, head_dim].
                - torch.Tensor: Value tensor of shape [B*H*W, T, num_heads or 1, head_dim].
        """
        B, T_frames, num_spatial_patches, _ = x.shape
        _, T_tokens, _ = text_embeddings.shape

        assert (
            x.size(0) == text_embeddings.size(0)
        ), "x.size(0) == text_embeddings.size(0) must be True."
        assert (
            x.size(-1) == text_embeddings.size(-1)
        ), "x.size(-1) == text_embeddings.size(-1) must be True."

        # Reshape x to [B*H*W, T_frames, d_model] for temporal attn
        x = x.view(-1, T_frames, self.d_model)

        assert (
            x.shape == (B*num_spatial_patches, T_frames, self.d_model)
        ), f"expected {(B*num_spatial_patches, T_frames, self.d_model)}, got {x.shape}"

        # Get QKV
        # q shape: [B*H*W, T_frames, d_model]; k, v shape: [B, T_tokens, d_model]
        q = self.q_proj(x)
        k = self.k_proj(text_embeddings)
        v = self.v_proj(text_embeddings)

        assert (
            q.shape == (B*num_spatial_patches, T_frames, self.d_model)
        ), f"expected {(B*num_spatial_patches, T_frames, self.d_model)}, got {q.shape}"
        assert (
            k.shape == v.shape == (B, T_tokens, self.query_groups*self.head_dim)
        ), f"expected {(B, T_tokens, self.query_groups*self.head_dim)}, got {k.shape}, {v.shape}"

        # Reshape to 4D tensors
        q = q.view(B*num_spatial_patches, T_frames, self.num_heads, self.head_dim)
        k = k.view(B, T_tokens, self.query_groups, self.head_dim)
        v = v.view(B, T_tokens, self.query_groups, self.head_dim)

        assert (
            q.shape == (B*num_spatial_patches, T_frames, self.num_heads, self.head_dim)
        ), f"expected {(B*num_spatial_patches, T_frames, self.num_heads, self.head_dim)}, got {q.shape}"
        assert (
            k.shape == v.shape == (B, T_tokens, self.query_groups, self.head_dim)
        ), f"expected {(B, T_tokens, self.query_groups, self.head_dim)}, got {k.shape}, {v.shape}"

        # Apply QK normalization
        if use_qk_norm:
            q, k = apply_qk_norm(q, k)
        
        # Extend KV heads
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

        assert (
            k.size(2) == v.size(2) == self.num_heads or 
            k.size(2) == v.size(2) == 1
        ), f"expected {self.num_heads} or 1."

        return q, k, v

    def forward(
        self,
        x: torch.Tensor,
        text_embeddings: torch.Tensor,
        use_mqa: bool,
        use_qk_norm: bool,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass of factorized cross attention layer.
        
        Args:
            x (torch.Tensor): Input image tensor of shape [B, T_frames, H*W, d_model].
            text_embeddings (torch.Tensor): Input text tensor of shape [B, T_tokens, d_model].
            use_mqa (bool): Whether to use MQA or not.
            use_qk_norm (bool): Whether to use QK normalization or not.
            padding_mask (Optional[torch.Tensor]): Padding tensor of shape [B, T_tokens].

        Returns:
            torch.Tensor: Output tensor of shape [B, T, H*W, d_model].
        """
        with autocast(device_type=device.type, dtype=dtype):
            # [B*T_frames, H*W d_model]
            spatial_out = self._spatial_cross_attention(
                x,
                text_embeddings=text_embeddings,
                use_mqa=use_mqa,
                use_qk_norm=use_qk_norm,
                padding_mask=padding_mask
            )

            # [B, T_frames, H*W, d_model]
            spatial_out = spatial_out.view(
                x.size(0), x.size(1), -1, self.d_model
            )

            # [B*H*W, T_frames, d_model]
            temporal_out = self._temporal_cross_attention(
                spatial_out,
                text_embeddings=text_embeddings,
                use_mqa=use_mqa,
                use_qk_norm=use_qk_norm,
                padding_mask=padding_mask
            )

            # [B, T_frames, H*W, d_model]
            spatio_temporal_out = temporal_out.view(
                x.size(0), x.size(1), -1, self.d_model
            )

            return self.o_proj(spatio_temporal_out)

class FactorizedCrossAttentionBlock(nn.Module):
    """Factorized cross attention layer applying attn, normalizagion, dropout, and residuals.
    
    Args:
        d_model (int): Dimensionality of model embeddings.
        num_heads (int): Number of attention heads for queries.
        query_groups (int): Number of query groups for GQA.
        softmax_scale (float): Scale for attention scores.
        use_proj_bias (bool): Whether to use projection bias or not.
        eps (float): Small epsilon value to maintain numerical stability in RMSNorm.
        dropout (float): Dropout probability for regularization.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        query_groups: int,
        softmax_scale: float,
        use_proj_bias: bool,
        eps: float,
        dropout: float
    ):
        super().__init__()

        self.cross_attention = FactorizedCrossAttention(
            d_model=d_model,
            num_heads=num_heads,
            query_groups=query_groups,
            softmax_scale=softmax_scale,
            use_proj_bias=use_proj_bias
        )
        self.rms_norm = RMSNorm(
            d_model=d_model, eps=eps
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        text_embeddings: torch.Tensor,
        use_mqa: bool,
        use_qk_norm: bool,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass of factorized cross attention block.
        
        Args:
            x (torch.Tensor): Input image tensor of shape [B, T_frames, H*W, d_model].
            text_embeddings (torch.Tensor): Input text tensor of shape [B, T_tokens, d_model].
            use_mqa (bool): Whether to use MQA or not.
            use_qk_norm (bool): Whether to use QK normalization or not.
            padding_mask (Optional[torch.Tensor]): Padding tensor of shape [B, T_tokens].

        Returns:
            torch.Tensor: Output tensor of shape [B, T, H*W, d_model].
        """
        with autocast(device_type=device.type, dtype=dtype):
            return x + self.dropout(
                self.cross_attention(
                    self.rms_norm(x),
                    text_embeddings=text_embeddings,
                    use_mqa=use_mqa,
                    use_qk_norm=use_qk_norm,
                    padding_mask=padding_mask
                )
            )


def test_spatial_qkv():
    d_model, num_heads, query_groups = 512, 32, 8
    softmax_scale = (d_model // num_heads) ** 0.5
    cross_attn = FactorizedCrossAttention(
        d_model, num_heads, query_groups,
        softmax_scale, use_proj_bias=False
    ).to(device)
    B, T_frames, H, W = 1, 8, 32, 32
    T_tokens = 16
    x = torch.randn(B, T_frames, H*W, d_model).to(device)
    text_embeddings = torch.randn(B, T_tokens, d_model).to(device)
    q, k, v = cross_attn._setup_spatial_qkv(
        x, text_embeddings, use_mqa=False, use_qk_norm=True
    )
    return q, k, v

def test_temporal_qkv():
    d_model, num_heads, query_groups = 512, 32, 8
    softmax_scale = (d_model // num_heads) ** 0.5
    cross_attn = FactorizedCrossAttention(
        d_model, num_heads, query_groups,
        softmax_scale, use_proj_bias=False
    ).to(device)
    B, T_frames, H, W = 1, 8, 32, 32
    T_tokens = 16
    x = torch.randn(B, T_frames, H*W, d_model).to(device)
    text_embeddings = torch.randn(B, T_tokens, d_model).to(device)
    q, k, v = cross_attn._setup_temporal_qkv(
        x, text_embeddings, use_mqa=False, use_qk_norm=True
    )
    return q, k, v

def test_attention():
    d_model, num_heads, query_groups = 512, 32, 8
    softmax_scale = (d_model // num_heads) ** 0.5
    cross_attn = FactorizedCrossAttention(
        d_model, num_heads, query_groups,
        softmax_scale, use_proj_bias=False
    ).to(device)
    B, T_frames, H, W = 1, 8, 32, 32
    T_tokens = 16
    x = torch.randn(B, T_frames, H*W, d_model).to(device)
    text_embeddings = torch.randn(B, T_tokens, d_model).to(device)
    padding_mask = torch.randint(
        0, 2, (B, T_tokens), dtype=torch.bool
    )
    x_out = cross_attn(
        x, text_embeddings, False, True, padding_mask
    )
    return x_out

def test_attention_block():
    d_model, num_heads, query_groups = 512, 32, 8
    softmax_scale = (d_model // num_heads) ** 0.5
    eps, dropout = 1e-12, 0.15
    cross_attn = FactorizedCrossAttentionBlock(
        d_model, num_heads, query_groups,
        softmax_scale, use_proj_bias=False, 
        eps=eps, dropout=dropout
    ).to(device)
    B, T_frames, H, W = 1, 8, 32, 32
    T_tokens = 16
    x = torch.randn(B, T_frames, H*W, d_model).to(device)
    text_embeddings = torch.randn(B, T_tokens, d_model).to(device)
    padding_mask = torch.randint(
        0, 2, (B, T_tokens), dtype=torch.bool
    )
    x_out = cross_attn(
        x, text_embeddings, False, True, padding_mask
    )
    return x_out

if __name__ == "__main__":
    # [B, T_frames, H*W, d_model]
    # [1, 8, 1024, 512]
    x = test_attention_block()
    print(x.shape)

# if __name__ == "__main__":
#     q, k, v = test_temporal_qkv()
#     print(q.shape)
#     print(k.shape)
#     print(v.shape)
