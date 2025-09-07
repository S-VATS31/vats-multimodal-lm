from configs.setup_env import (
    device,
    dtype,
    gpu_dtypes,
    use_flash_attn,
    flash_attn_varlen_qkvpacked_func
)

from typing import Optional, Tuple, Literal, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

from src.rms_norm import RMSNorm
from src.optimized_attention import KVCache
from utils.attention_utils import extend_kv_heads, setup_projections, apply_qk_norm
from src.autoregressive_video_gen.autoregressive_transformer.attention.rope3d import NTKRoPE3D


class CausalFactorizedAttention(nn.Module):
    """Causal attention layer utilizing factorized attention.
    
    Args:
        d_model (int): Dimensionality of model embeddings.
        num_heads (int): Number of attention heads for queries.
        query_groups (int): Number of query groups for GQA.
        rope_theta (float): Exponential base of inv freq for RoPE.
        softmax_scale (float): Softmax scale for attention scores.
        use_proj_bias (bool): Whether to use projection bias or not.
        use_fused_proj (bool): Whether to use QKV projection or not.
        use_windowed_attn (bool): Whether to use windowed attention or not.
        use_ntk_rope (bool): Whether to use NTK RoPE or classic.
        ntk_scale_factor (Optional[float]): Alpha hyperparameter for NTK RoPE.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        query_groups: int,
        rope_theta: float,
        softmax_scale: float,
        use_proj_bias: bool,
        use_fused_proj: bool,
        use_windowed_attn: bool,
        use_ntk_rope: bool,
        ntk_scale_factor: Optional[float] = None
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                f"expcted d_model % num_heads == 0, got {d_model} % {num_heads} != 0"
            )
        if num_heads % query_groups != 0:
            raise ValueError(
                f"expected num_heads % query_groups == 0, got {num_heads} % {query_groups} != 0."
            )
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.query_groups = query_groups
        self.softmax_scale = softmax_scale
        self.use_fused_proj = use_fused_proj
        self.use_windowed_attention = use_windowed_attn
        self.head_dim = d_model // num_heads
        self.heads_per_group = num_heads // query_groups

        # QKV proj
        if use_fused_proj:
            self.qkv_proj, self.o_proj = setup_projections(
                d_model=d_model,
                num_heads=num_heads,
                head_dim=self.head_dim,
                use_fused_proj=use_fused_proj,
                use_gqa=True,
                use_proj_bias=use_proj_bias,
                query_groups=query_groups
            )
        else:
            self.q_proj, self.k_proj, self.v_proj, self.o_proj = setup_projections(
                d_model=d_model,
                num_heads=num_heads,
                head_dim=self.head_dim,
                use_fused_proj=use_fused_proj,
                use_gqa=True,
                use_proj_bias=use_proj_bias,
                query_groups=query_groups
            )

        # Set up 3D RoPE
        self.ntk_rope3d = NTKRoPE3D(
            head_dim=self.head_dim,
            rope_theta=rope_theta,
            use_ntk_rope=use_ntk_rope,
            ntk_scale_factor=ntk_scale_factor
        )
    
    def _update_temporal_kv(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        layer_idx: Optional[int] = None
    ) -> None:
        """Update KV cache for temporal layers.
        
        Args:
            key (torch.Tensor): Key tensor of shape [B*H*W, T, num_heads, head_dim].
            value (torch.Tensor): Value tensor of shape [B*H*W, T, num_heads, head_dim].
            kv_cache (Optional[KVCache]): KV caching module.
            layer_idx (Optional[int]): Layer to be updated.
        """
        if kv_cache is None or layer_idx is None:
            return
        
        # Only concatenate if there are already tokens in cache
        if kv_cache.current_seq_len is not None and kv_cache.current_seq_len > 0:
            cached_k, cached_v = kv_cache.get(layer_idx, kv_cache.current_seq_len)
            key = torch.cat([cached_k, key], dim=1)
            value = torch.cat([cached_v, value], dim=1)

        # Update KVCache 
        kv_cache.update(layer_idx, key, value)

    def _optimized_attention(
        self,
        x: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        use_causal: bool,
        left_window: int,
        right_window: int,
        attn_mode: Literal["spatial", "temporal"],
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
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
            if use_causal and padding_mask is not None:
                pass
            else:
                raise ValueError("needs causal + padding for optimized attn")
        else:
            return self._torch_attention(
                x,
                query=query.transpose(1, 2),
                key=key.transpose(1, 2),
                value=value.transpose(1, 2),
                use_causal=use_causal,
                padding_mask=padding_mask,
                attn_mode=attn_mode
            )

    def _torch_attention(
        self,
        x: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        use_causal: bool,
        attn_mode: Literal["spatial", "temporal"],
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """PyTorch fallback to optimized attention.
        
        Args:
            query (torch.Tensor): Query tensor of shape [:, num_heads, :, head_dim].
            key (torch.Tensor): Key tensor of shape [:, num_heads or 1, :, head_dim].
            value (torch.Tensor): Value tensor of shape [:, num_heads or 1, :, head_dim].
            use_causal (bool): Whether to use causal masking or not.
            padding_mask (Optional[torch.Tensor]): Padding tensor of shape [B, T*H*W].

        Returns:
            torch.Tensor: Attention output of shape [:, :, d_model].
        """
        assert x.dim() == 4, f"expected 4 dims, got {x.dim()} dims"
        assert query.dim() == 4, f"expected 4 dims, got {query.dim()} dims"
        assert key.dim() == 4, f"expected 4 dims, got {key.dim()} dims"
        assert value.dim() == 4, f"expected 4 dims, got {value.dim()} dims"
        
        B, T, num_spatial_patches, _ = x.shape

        if padding_mask is not None:
            padding_mask = padding_mask.bool()
            if attn_mode == "spatial":
                # Reshape to [B*T, H*W] for spatial padding
                padding_mask = padding_mask.view(-1, num_spatial_patches)
                assert (
                    padding_mask.shape == (B*T, num_spatial_patches)
                ), f"expected {(B*T, num_spatial_patches)}, got {padding_mask.shape}"

                padding_mask = padding_mask[:, None, :, None] # [B*T, 1, H*W, 1]
                attn_mask = padding_mask.expand(
                    B*T, 1, num_spatial_patches, key.size(2) # [B*T, 1, T_q, T_k]
                )
                assert (
                    attn_mask.shape == (B*T, 1, num_spatial_patches, key.size(2))
                ), f"expected {(B*T, 1, num_spatial_patches, key.size(2))}, got {attn_mask.shape}"

                if use_causal:
                    # [T_q, T_k]
                    causal_mask = torch.tril(
                        torch.ones(
                            query.size(2), key.size(2), dtype=torch.bool
                        )
                    )
                    assert (
                        causal_mask.shape == (query.size(2), key.size(2))
                    ), f"expected {(query.size(2), key.size(2))}, got {causal_mask.shape}"

                    # [B*T, 1, T_q, T_k]
                    causal_mask = (
                        causal_mask[None, None, :, :]
                        .expand(B*T, 1, query.size(2), key.size(2))
                    )
                    assert (
                        causal_mask.shape == (B*T, 1, query.size(2), key.size(2))
                    ), f"expected {(B*T, 1, query.size(2), key.size(2))}, got {causal_mask.shape}"
            else:
                # Reshape to [B*H*W, T] for temporal padding
                padding_mask = padding_mask.view(-1, T)
                assert (
                    padding_mask.shape == (B*num_spatial_patches, T)
                ), f"expected {(B*num_spatial_patches, T)}, got {padding_mask.shape}"

                padding_mask = padding_mask[:, None, :, None] # [B*H*W, 1, T, 1]
                attn_mask = padding_mask.expand(
                    B*num_spatial_patches, 1, T, key.size(2) # [B*H*W, 1, T_q, T_k]
                )
                assert (
                    attn_mask.shape == (B*num_spatial_patches, 1, T, key.size(2))
                ), f"expected {(B*num_spatial_patches, 1, T, key.size(2))}, got {attn_mask.shape}"

                if use_causal:
                    # [T_q, T_k]
                    causal_mask = torch.tril(
                        torch.ones(
                            query.size(2), key.size(2), dtype=torch.bool
                        )
                    )
                    assert (
                        causal_mask.shape == (query.size(2), key.size(2))
                    ), f"expected {(query.size(2), key.size(2))}, got {causal_mask.shape}"

                    # [1, 1, T_q, T_k]
                    causal_mask = (
                        causal_mask[None, None, :, :]
                        .expand(B*num_spatial_patches, 1, query.size(2), key.size(2))
                    )
                    assert (
                        causal_mask.shape == (B*num_spatial_patches, 1, query.size(2), key.size(2))
                    ), f"expected {B*num_spatial_patches, 1, query.size(2), key.size(2)}, got {causal_mask.shape}"

            # Aggregate padding and causal masks
            attn_mask = attn_mask & causal_mask
        else:
            attn_mask = None

        # Expand to num_heads
        if attn_mask is not None:
            if attn_mode == "spatial":
                attn_mask = attn_mask.expand(
                    B*T, self.num_heads, num_spatial_patches, key.size(2)
                )
                assert (
                    attn_mask.shape == (B*T, self.num_heads, num_spatial_patches, key.size(2))
                ), f"expected {(B*T, self.num_heads, num_spatial_patches, key.size(2))}, got {attn_mask.shape}"
            else:
                attn_mask = attn_mask.expand(
                    B*num_spatial_patches, self.num_heads, T, key.size(2)
                )
                assert (
                    attn_mask.shape == (B*num_spatial_patches, self.num_heads, T, key.size(2))
                ), f"expected {(B*num_spatial_patches, self.num_heads, T, key.size(2))}, got {attn_mask.shape}"

        # Get attn out
        attn_out = F.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            is_causal=use_causal if attn_mask is None else False,
            scale=self.softmax_scale,
            enable_gqa=False
        ) # [:, num_heads, :, head_dim]

        if attn_mode == "spatial":
            assert (
                attn_out.shape == (B*T, self.num_heads, num_spatial_patches, self.head_dim)
            ), f"expected {(B*T, self.num_heads, num_spatial_patches, self.head_dim)}, got {attn_out.shape}"
        else:
            assert (
                attn_out.shape == (B*num_spatial_patches, self.num_heads, T, self.head_dim)
            ), f"expected {(B*num_spatial_patches, self.num_heads, T, self.head_dim)}, got {attn_out.shape}"
        
        attn_out = (
            attn_out.transpose(1, 2)
            .contiguous()
            .view(query.size(0), query.size(2), self.d_model)
        )

        if attn_mode == "spatial":
            assert (
                attn_out.shape == (B*T, num_spatial_patches, self.d_model)
            ), f"expected {(B*T, num_spatial_patches, self.d_model)}, got {attn_out.shape}"
        else:
            assert (
                attn_out.shape == (B*num_spatial_patches, T, self.d_model)
            ), f"expected {(B*num_spatial_patches, T, self.d_model)}, got {attn_out.shape}"

        return attn_out

    def _spatial_attention(
        self,
        x: torch.Tensor,
        use_mqa: bool,
        use_qk_norm: bool,
        use_causal: bool,
        left_window: int,
        right_window: int,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Spatial attention to input tensor of shape [B*T, H*W, d_model].
        
        Args:
            x (torch.Tensor): Input tensor.
            use_mqa (bool): Whether to use MQA or not.
            use_qk_norm (bool): Whether to use QK normalization or not.
            use_causal (bool): Whether to use causal masking or not.
            left_window (int): Left window for sliding window attention.
            right_window (int): Right window for sliding window attention.
            padding_mask (Optional[torch.Tensor]): Padding tensor of shape [B, T*H*W].

        Returns:
            torch.Tensor: Spatial attention output.
        """
        q, k, v = self._setup_qkv(
            x,
            use_mqa=use_mqa,
            use_qk_norm=use_qk_norm,
            attn_mode="spatial"
        )
        spatial_attn_out = self._optimized_attention(
            x,
            query=q,
            key=k,
            value=v,
            use_causal=use_causal,
            left_window=left_window,
            right_window=right_window,
            padding_mask=padding_mask,
            attn_mode="spatial"
        )

        return spatial_attn_out

    def _temporal_attention(
        self,
        x: torch.Tensor,
        use_mqa: bool,
        use_qk_norm: bool,
        use_causal: bool,
        left_window: int,
        right_window: int,
        use_cache: bool,
        padding_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        layer_idx: Optional[int] = None
    ) -> torch.Tensor:
        """Temporal attention to input tensor of shape [B*H*W, T, d_model].
        
        Args:
            x (torch.Tensor): Input tensor.
            use_mqa (bool): Whether to use MQA or not.
            use_qk_norm (bool): Whether to use QK normalization or not.
            use_causal (bool): Whether to use causal masking or not.
            left_window (int): Left window for sliding window attention.
            right_window (int): Right window for sliding window attention.
            use_cache (bool): Whether to use KV caching or not.
            padding_mask (Optional[torch.Tensor]): Padding tensor of shape [B, T*H*W].
            kv_cache (Optional[KVCache]): KV caching module.
            layer_idx (Optional[int]): Layer to be updated.

        Returns:
            torch.Tensor: Temporal attention output.
        """
        # Get QKV
        q, k, v = self._setup_qkv(
            x,
            use_mqa=use_mqa,
            use_qk_norm=use_qk_norm,
            attn_mode="temporal"
        )

        # Update temporal KVs
        if use_cache and kv_cache is not None and layer_idx is not None:
            self._update_temporal_kv(
                key=k, 
                value=v, 
                kv_cache=kv_cache, 
                layer_idx=layer_idx
            )

        # Get temporal output
        temporal_attn_out = self._optimized_attention(
            x,
            query=q,
            key=k,
            value=v,
            use_causal=use_causal,
            left_window=left_window,
            right_window=right_window,
            padding_mask=padding_mask,
            attn_mode="temporal"
        )

        return temporal_attn_out

    def _setup_qkv(
        self,
        x: torch.Tensor,
        use_mqa: bool,
        use_qk_norm: bool,
        attn_mode: Literal["spatial", "temporal"]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Set up QKV tensors with respect to spatial or temporal attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, T, H*W, d_model].
            use_mqa (bool): Whether to use MQA or not.
            use_qk_norm (bool): Whether to use QK normalization or not.
            attn_mode (Literal["spatial", "temporal"]): Set up QKV tensors.

        Returns:
            Tuple:
                - torch.Tensor: Query tensor.
                - torch.Tensor: Key tensor.
                - torch.Tensor: Value tensor.
        """
        if x.dim() == 4:
            B, T, num_spatial_patches, _ = x.shape
        else:
            raise ValueError(f"expected 4 dims, got {x.dim()} dims.")
        
        # Reshape x to [B*T, H*W, d_model] for spatial
        # Reshape x to [B*H*W, T, d_model] for temporal
        if attn_mode == "spatial":
            x = x.view(-1, num_spatial_patches, self.d_model)
            assert (
                x.shape == (B*T, num_spatial_patches, self.d_model)
            ), f"expected {(B*T, num_spatial_patches, self.d_model)}, got {x.shape}"
        elif attn_mode == "temporal":
            x = (
                x.transpose(1, 2).contiguous().view(-1, T, self.d_model)
            )
            assert (
                x.shape == (B*num_spatial_patches, T, self.d_model)
            ), f"expected {(B*num_spatial_patches, T, self.d_model)}, got {x.shape}"
        else:
            raise ValueError(f"expected ['spatial', 'temporal'], got {attn_mode}")

        # QKV proj
        if self.use_fused_proj:
            qkv = self.qkv_proj(x)

            if attn_mode == "spatial":
                assert (
                    qkv.shape == (
                        B*T, 
                        num_spatial_patches, 
                        self.num_heads*self.head_dim + 2*self.query_groups*self.head_dim
                    )
                ), f"expected {(
                        B*T, 
                        num_spatial_patches, 
                        self.num_heads * self.head_dim + 2*self.query_groups*self.head_dim
                    )}, got {qkv.shape}"
            else:
                assert (
                    qkv.shape == (
                        B*num_spatial_patches,
                        T,
                        self.num_heads * self.head_dim + 2 *self.query_groups*self.head_dim
                    )
                ), f"expected {(
                        B*num_spatial_patches,
                        T,
                        self.num_heads * self.head_dim + 2 *self.query_groups*self.head_dim
                    )}, got {qkv.shape}"
            # Split qkv into q, kv
            q, kv = torch.split(
                qkv, [self.num_heads * self.head_dim, 2 *self.query_groups*self.head_dim], dim=-1
            )

            if attn_mode == "spatial":
                assert (
                    q.shape == (B*T, num_spatial_patches, self.num_heads*self.head_dim)
                ), f"expected {(B*T, num_spatial_patches, self.num_heads*self.head_dim)} got {qkv.shape}"
                assert (
                    kv.shape == (B*T, num_spatial_patches, 2*self.query_groups*self.head_dim)
                ), f"expected {(B*T, num_spatial_patches, 2*self.query_groups*self.head_dim)}, got {kv.shape}"
            else:
                assert (
                    q.shape == (B*num_spatial_patches, T, self.num_heads*self.head_dim)
                ), f"expected {(B*num_spatial_patches, T, self.num_heads*self.head_dim)}, got {q.shape}"
                assert (
                    kv.shape == (B*num_spatial_patches, T, 2*self.query_groups*self.head_dim)
                ), f"expected {(B*num_spatial_patches, T, 2*self.query_groups*self.head_dim)}"
            
            # Split kv in to k, v
            k, v = kv.chunk(chunks=2, dim=-1)

            if attn_mode == "spatial":
                assert (
                    k.shape == v.shape == (B*T, num_spatial_patches, self.query_groups*self.head_dim)
                ), (
                    f"expected {(B*T, num_spatial_patches, self.d_model)}, got "
                    f"k.shape = {k.shape}, v.shape = {v.shape}"
                )
            else:
                assert (
                    k.shape == v.shape == (B*num_spatial_patches, T, self.query_groups*self.head_dim)
                ), (
                    f"expected {(B*num_spatial_patches, T, self.d_model)}, got "
                    f"k.shape = {k.shape}, v.shape = {v.shape}"
                )

        # Q, K, V proj's
        else:
            q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
            if attn_mode == "spatial":
                assert (
                    q.shape == (B*T, num_spatial_patches, self.num_heads*self.head_dim)
                ), f"expected {(B*T, num_spatial_patches, self.num_heads*self.head_dim)}, got {q.shape}"
                assert (
                    k.shape == v.shape == (B*num_spatial_patches, T, self.num_heads*self.head_dim)
                ), (
                    f"expected {(B*num_spatial_patches, T, self.query_groups*self.head_dim)}"
                )
            else:
                assert (
                    q.shape == (B*num_spatial_patches, T, self.num_heads*self.head_dim)
                ), f"expected {(B*num_spatial_patches, T, self.num_heads*self.head_dim)}, got {q.shape}"
                assert (
                    k.shape == v.shape == (B*num_spatial_patches, T, self.query_groups*self.head_dim)
                ), (
                    f"expected {(B*num_spatial_patches, T, self.query_groups*self.head_dim)}, got "
                    f"k.shape = {k.shape}, v.shape = {v.shape}"
                )
        
        # Reshape to 4D tensors for RoPE
        if attn_mode == "spatial":
            q = q.view(B*T, num_spatial_patches, self.num_heads, self.head_dim)
            k = k.view(B*T, num_spatial_patches, self.query_groups, self.head_dim)
            v = v.view(B*T, num_spatial_patches, self.query_groups, self.head_dim)
            assert (
                q.shape == (B*T, num_spatial_patches, self.num_heads, self.head_dim)
            ), f"expected {(B*T, num_spatial_patches, self.num_heads, self.head_dim)}, got {q.shape}"
            assert (
                k.shape == v.shape == (B*T, num_spatial_patches, self.query_groups, self.head_dim)
            ), (
                f"expected {(B*T, num_spatial_patches, self.query_groups, self.head_dim)}, got "
                f"k.shape = {k.shape}, v.shape = {v.shape}"
            )
        else:
            q = q.view(B*num_spatial_patches, T, self.num_heads, self.head_dim)
            k = k.view(B*num_spatial_patches, T, self.query_groups, self.head_dim)
            v = v.view(B*num_spatial_patches, T, self.query_groups, self.head_dim)
            assert (
                q.shape == (B*num_spatial_patches, T, self.num_heads, self.head_dim)
            ), f"expected {(B*num_spatial_patches, T, self.num_heads, self.head_dim)}, got {q.shape}"
            assert(
                k.shape == v.shape == (B*num_spatial_patches, T, self.query_groups, self.head_dim)
            ), (
                f"expected {(B*num_spatial_patches, T, self.query_groups, self.head_dim)}, got "
                f"k.shape = {k.shape}, v.shape = {v.shape}"
            )

        # Apply QK normalization
        if use_qk_norm:
            q, k = apply_qk_norm(query=q, key=k)

        # Apply NTKRoPE3D
        # q = self.ntk_rope3d(q)
        # k = self.ntk_rope3d(k)

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
        ), (
            f"expected {self.num_heads} or 1, got k.size(2) = {k.size(2)}, v.size(2) = {v.size(2)}"
        )

        return q, k, v

    def forward(
        self,
        x: torch.Tensor,
        use_mqa: bool,
        use_qk_norm: bool,
        use_causal: bool,
        left_window: int,
        right_window: int,
        use_cache: bool,
        padding_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        layer_idx: Optional[int] = None,
        *,
        return_qkv: bool = False
    ) -> Union[torch.Tensor, Tuple]:
        """Forward pass spatio-temporal attention layer.
        
        Args:   
            x (torch.Tensor): Input tensor of shape [B, T, H*W, d_model].
            use_mqa (bool): Whether to use MQA or not.
            use_qk_norm (bool): Whether to use QK normalization or not.
            use_causal (bool): Whether to use causal masking or not.
            left_window (int): Left window for sliding window attention.
            right_window (int): Right window for sliding window attention.
            use_cache (bool): Whether to use KV caching or not.
            padding_mask (Optional[torch.Tensor]): Padding tensor of shape [B, T*H*W].
            kv_cache (Optional[KVCache]): KVCache module.
            layer_idx (Optional[int]): Current layer to update key and value tensors wrt.
        
        Keyword args:
            return_qkv (bool): Whether to return QKV tensors for debugging.

        Returns:
            Union:
                - torch.Tensor: Spatio-temporal output of same shape as input.
                - Tuple:
                    - torch.Tensor: Spatio-temporal output of same shape as input.
                    - torch.Tensor: Spatial query tensor.
                    - torch.Tensor: Spatial key tensor.
                    - torch.Tensor: Spatial value tensor.
                    - torch.Tensor: Temporal query tensor.
                    - torch.Tensor: Temporal key tensor.
                    - torch.Tensor: Temporal value tensor.
        """
        with autocast(device_type=device.type, dtype=dtype):
            # Set right window to 0 for causal gen
            if use_causal:
                right_window = 0
            
            # Global attention setup
            if not self.use_windowed_attention:
                left_window, right_window = -1, -1

            # [B*T, H*W, d_model]
            spatial_out = self._spatial_attention(
                x,
                use_mqa=use_mqa,
                use_qk_norm=use_qk_norm,
                use_causal=use_causal,
                left_window=left_window,
                right_window=right_window,
                padding_mask=padding_mask
            )

            # [B, T, H*W, d_model]
            spatial_out = spatial_out.view(
                x.size(0), x.size(1), -1, self.d_model
            )

            # [B*H*W, T, d_model]
            temporal_out = self._temporal_attention(
                spatial_out,
                use_mqa=use_mqa,
                use_qk_norm=use_qk_norm,
                use_causal=use_causal,
                left_window=left_window,
                right_window=right_window,
                use_cache=use_cache,
                padding_mask=padding_mask,
                kv_cache=kv_cache,
                layer_idx=layer_idx
            )

            # [B, T, H*W, d_model]
            spatio_temporal_out = (
                temporal_out.view(x.size(0), x.size(1), -1, self.d_model)
            )

            # Return QKV for spatial and temporal if applied
            if return_qkv:
                q_spatial, k_spatial, v_spatial = self._setup_qkv(
                    x,
                    use_mqa=use_mqa,
                    use_qk_norm=use_qk_norm,
                    attn_mode="spatial"
                )
                q_temporal, k_temporal, v_temporal = self._setup_qkv(
                    x,
                    use_mqa=use_mqa,
                    use_qk_norm=use_qk_norm,
                    attn_mode="temporal"
                )
                return (
                    self.o_proj(spatio_temporal_out),
                    q_spatial, k_spatial, v_spatial,
                    q_temporal, k_temporal, v_temporal
                )

            return self.o_proj(spatio_temporal_out)


class CausalFactorizedAttentionBlock(nn.Module):
    """Attention block applying attention, normalization, residuals and dropout.
    
    Args:
        d_model (int): Dimensionality of model embeddings.
        num_heads (int): Number of attention heads for queries.
        query_groups (int): Number of query groups for GQA.
        rope_theta (float): Exponential base of inv freq for RoPE.
        softmax_scale (float): Softmax scale for attention scores.
        use_proj_bias (bool): Whether to use projection bias or not.
        use_fused_proj (bool): Whether to use QKV projection or not.
        use_windowed_attn (bool): Whether to use windowed attention or not.
        use_ntk_rope (bool): Whether to use NTK RoPE or classic.
        eps (float): RMSNorm epsilon value to maintain numerical stability.
        dropout (float): Dropout probability for regularization.
        ntk_scale_factor (Optional[float]): Alpha hyperparameter for NTK RoPE.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        query_groups: int,
        rope_theta: float,
        softmax_scale: float,
        use_proj_bias: bool,
        use_fused_proj: bool,
        use_windowed_attn: bool,
        use_ntk_rope: bool,
        eps: float,
        dropout: float,
        ntk_scale_factor: Optional[float] = None,
    ):
        super().__init__()

        self.attention = CausalFactorizedAttention(
            d_model=d_model,
            num_heads=num_heads,
            query_groups=query_groups,
            rope_theta=rope_theta,
            softmax_scale=softmax_scale,
            use_proj_bias=use_proj_bias,
            use_fused_proj=use_fused_proj,
            use_windowed_attn=use_windowed_attn,
            use_ntk_rope=use_ntk_rope,
            ntk_scale_factor=ntk_scale_factor
        )
        self.rms_norm = RMSNorm(
            d_model=d_model, eps=eps
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        use_mqa: bool,
        use_qk_norm: bool,
        use_causal: bool,
        left_window: int,
        right_window: int,
        use_cache: bool,
        padding_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        layer_idx: Optional[int] = None
    ) -> torch.Tensor:
        """Forward pass spatio-temporal attention layer.
        
        Args:   
            x (torch.Tensor): Input tensor of shape [B, T, H*W, d_model].
            use_mqa (bool): Whether to use MQA or not.
            use_qk_norm (bool): Whether to use QK normalization or not.
            use_causal (bool): Whether to use causal masking or not.
            left_window (int): Left window for sliding window attention.
            right_window (int): Right window for sliding window attention.
            use_cache (bool): Whether to use KV caching or not.
            padding_mask (Optional[torch.Tensor]): Padding tensor of shape [B, T*H*W].
            kv_cache (Optional[KVCache]): KVCache module.
            layer_idx (Optional[int]): Current layer to update key and value tensors wrt.

        Returns:
            torch.Tensor: Spatio-temporal output of same shape as input.
        """
        with autocast(device_type=device.type, dtype=dtype):
            return x + self.dropout(
                self.attention(
                    self.rms_norm(x),
                    use_mqa=use_mqa,
                    use_qk_norm=use_qk_norm,
                    use_causal=use_causal,
                    left_window=left_window,
                    right_window=right_window,
                    use_cache=use_cache,
                    padding_mask=padding_mask,
                    kv_cache=kv_cache,
                    layer_idx=layer_idx
                )
            )

def test_attention():
    d_model, num_heads, query_groups, rope_theta = 512, 32, 8, 10000.0
    softmax_scale = 1 / (d_model // num_heads) ** 0.5
    use_proj_bias, use_fused_proj, use_windowed_attn = False, True, True
    use_ntk_rope, ntk_scale_factor = True, 0.7
    attention = CausalFactorizedAttention(
        d_model, num_heads, query_groups,rope_theta, 
        softmax_scale, use_proj_bias,use_fused_proj, 
        use_windowed_attn, use_ntk_rope, ntk_scale_factor
    ).to(device)
    B, T, H, W = 1, 8, 32, 32
    x = torch.randn(B, T, H*W, d_model).to(device)
    padding_mask = torch.randint(
        0, 2, (B, T*H*W), dtype=torch.bool
    ).to(device)
    (
        x_out, 
        spatial_q, spatial_k, spatial_v, 
        temporal_q, temporal_k, temporal_v
    ) = attention(
        x,
        use_mqa=False,
        use_qk_norm=True,
        use_causal=True,
        left_window=-1,
        right_window=-1,
        use_cache=False,
        padding_mask=padding_mask,
        kv_cache=None,
        layer_idx=None,
        return_qkv=True
    )    

    return (
        x_out,
        spatial_q, spatial_k, spatial_v,
        temporal_q, temporal_k, temporal_v
    )

def test_attention_block():
    d_model, num_heads, query_groups, rope_theta = 512, 32, 8, 10000.0
    softmax_scale = 1 / (d_model // num_heads) ** 0.5
    use_proj_bias, use_fused_proj, use_windowed_attn = False, True, True
    use_ntk_rope, ntk_scale_factor = True, 0.7
    eps, dropout = 1e-12, 0.15
    attention = CausalFactorizedAttentionBlock(
        d_model, num_heads, query_groups,rope_theta, 
        softmax_scale, use_proj_bias,use_fused_proj, 
        use_windowed_attn, use_ntk_rope, eps, dropout, ntk_scale_factor
    ).to(device)
    B, T, H, W = 1, 8, 32, 32
    x = torch.randn(B, T, H*W, d_model).to(device)
    padding_mask = torch.randint(
        0, 2, (B, T*H*W), dtype=torch.bool
    ).to(device)
    x_out = attention(
        x,
        use_mqa=False,
        use_qk_norm=True,
        use_causal=True,
        left_window=-1,
        right_window=-1,
        use_cache=False,
        padding_mask=padding_mask,
        kv_cache=None,
        layer_idx=None,
    )    

    return x_out

if __name__ == "__main__":
    x_attn_block = test_attention_block()
    print(x_attn_block.shape)

# if __name__ == "__main__":
#     x_out, spatial_q, spatial_k, spatial_v, temporal_q, temporal_k, temporal_v = test_attention()
#     print(x_out.shape)
#     print(spatial_q.shape)
#     print(spatial_k.shape)
#     print(spatial_v.shape)
#     print(temporal_q.shape)
#     print(temporal_k.shape)
#     print(temporal_v.shape)
