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
from src.optimized_attention import KVCache
from utils.attention_utils import extend_kv_heads, setup_projections, apply_qk_norm
from src.autoregressive_image_gen.autoregressive_transformer.attention.rope_2d import NTKRoPE2D

class CausalSelfAttention(nn.Module):
    """Causal self attention layer for autoregressive image generation.
    
    Args:
        d_model (int): Dimensionality of model embeddings.
        num_heads (int): Number of attention heads.
        query_groups (int): Number of query_groups for GQA.
        rope_theta (float): Exponential base for RoPE inv freq.
        softmax_scale (float): Value to scale attention scores by.
        use_proj_bia (bool): Whether to use projection bias or not.
        use_fused_proj (bool): Whether to use qkv or seperate projections.
        use_window_attn (bool): Whether to use windowed attention or not.
        use_ntk_rope (bool): Whether to use NTK RoPE or classic RoPE.
        ntk_scale_factor (Optional[float]): Scaling factor for NTK RoPE.
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
                f"d_model % num_heads must be 0, got {d_model} % {num_heads} != 0."
            )
        if num_heads % query_groups != 0:
            raise ValueError(
                f"num_heads % query_groups must be 0, got {num_heads} % {query_groups} != 0."
            )
        
        if use_ntk_rope:
            assert ntk_scale_factor is not None, "Must be given scale factor for NTK RoPE"
        else:
            ntk_scale_factor = None

        self.d_model = d_model
        self.num_heads = num_heads
        self.query_groups = query_groups
        self.softmax_scale = softmax_scale
        self.use_fused_proj = use_fused_proj
        self.use_windowed_attn = use_windowed_attn
        self.head_dim = d_model // num_heads
        self.heads_per_group = num_heads // query_groups

        # Setup projections
        if use_fused_proj:
            self.qkv_proj, self.o_proj = setup_projections(
                d_model=d_model,
                num_heads=num_heads,
                head_dim=self.head_dim,
                use_fused_proj=use_fused_proj,
                use_gqa=True,
                use_proj_bias=use_proj_bias,
                query_groups=self.query_groups
            )
        else:
            self.q_proj, self.k_proj, self.v_proj, self.o_proj = setup_projections(
                d_model=d_model,
                num_heads=num_heads,
                head_dim=self.head_dim,
                use_fused_proj=use_fused_proj,
                use_gqa=True,
                use_proj_bias=use_proj_bias,
                query_groups=self.query_groups
            )

        # Set up RoPE
        self.ntk_rope = NTKRoPE2D(
            head_dim=self.head_dim,
            rope_theta=rope_theta,
            use_ntk_rope=use_ntk_rope,
            ntk_scale_factor=ntk_scale_factor,
        )

    def _optimized_attention(
        self,
        x: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal: bool,
        left_window: int,
        right_window: int,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Optimized attention utilizing flash attn V2 and sliding window attn (SWA).
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, T, d_model]. Only using for shape validation.
            query (torch.Tensor): Query tensor of shape [B, T, num_heads, head_dim].
            key (torch.Tensor): Key tensor of shape [B, T, num_heads, head_dim].
            value (torch.Tensor): Value tensor of shape [B, T, num_heads, head_dim].
            causal (bool): Whether to use causal masking or not.
            left_window (int): Left window for SWA.
            right_window (int): Right window for SWA.
            padding_mask (Optional[torch.Tensor]): Padding tensor of shape [B, T].

        Returns:
            torch.Tensor: Output tensor of shape [B, T, d_model].
        """
        B, T, _ = x.shape

        # Flash Attention
        if (
            flash_attn_varlen_qkvpacked_func is not None
            and use_flash_attn and device.type == "cuda"
            and query.dtype in gpu_dtypes
            and key.dtype in gpu_dtypes
            and value.dtype in gpu_dtypes
            and query.is_cuda and key.is_cuda and value.is_cuda
        ):
            if causal and padding_mask is not None:
                # Stack qkv over dim3
                qkv_stacked = torch.stack(
                    [query, key, value], dim=3
                ).contiguous() # [B, T, num_heads, 3, head_dim]

                assert (
                    qkv_stacked.shape == (B, T, self.num_heads, 3, self.head_dim)
                ), f"expected: {(B, T, self.num_heads, 3, self.head_dim)}, got {qkv_stacked.shape}"

                # Get cumulative sequence lengths
                # get number of valid tokens per seq
                # do cumsum to get end idx of each seq
                seqlens = padding_mask.sum(dim=1).to(torch.int32) # [B]
                max_seqlen = seqlens.max().item() # we will pass this to flash attn func
                cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0), (1, 0)) # [B+1], concat 0th idx

                assert (
                    seqlens.shape == (B,)
                ), f"expected: {(B,)}, got {seqlens.shape}"
                assert (
                    cu_seqlens.shape == (B+1,)
                ), f"expected: {B+1,}, got {cu_seqlens.shape}"
                assert (
                    cu_seqlens.dtype == torch.int32
                ), f"expected int32, got {cu_seqlens.dtype}"

                # Flatten padding mask to get valid tokens
                valid_tokens = padding_mask.flatten() # [B*T]

                assert (
                    valid_tokens.shape == (B*T,)
                ), f"expected: {(B*T,)}, got {valid_tokens.shape}"

                # Flatten qkv and reshape for attention output
                # we have: qkv_stacked = [B, T, num_heads, 3, head_dim]
                # we want: qkv_flattened = [B*T, 3, num_heads, head_dim]
                qkv_flattened = (
                    qkv_stacked
                    .view(-1, self.num_heads, 3, self.head_dim) # [B*T, num_heads, 3, head_dim]
                    .transpose(1, 2) # [B*T, 3, num_heads, head_dim]
                    .contiguous()
                )
                
                assert (
                    qkv_flattened.shape == (B*T, 3, self.num_heads, self.head_dim)
                ), f"expected: {(B*T, 3, self.num_heads, self.head_dim)}, got {qkv_flattened.shape}"

                # flash attn
                attn_out = flash_attn_varlen_qkvpacked_func(
                    qkv_flattened,
                    cu_seqlens,
                    max_seqlen,
                    causal=causal,
                    softmax_scale=self.softmax_scale,
                    window_size=(left_window, right_window),
                ) # [B * T, num_heads, head_dim]

                assert (
                    attn_out.shape == (B*T, self.num_heads, self.head_dim)
                ), f"expected: {(B*T, self.num_heads, self.head_dim)}, got {attn_out.shape}"

                # [B, T, d_model]
                attn_out = attn_out.view(B, T, -1)

                return attn_out

            else:
                raise ValueError("only causal + padding supported")
            
        else:
            return self._torch_attention(query, key, value, causal, padding_mask)

    def _torch_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal: bool,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """PyTorch scaled dot product attention.
        
        Args:
            query (torch.Tensor): Query tensor of shape [B, T, num_heads, head_dim].
            key (torch.Tensor): Key tensor of shape [B, T, num_heads or 1, head_dim].
            value (torch.Tenspor): Value tensor of shape [B, T, num_heads or 1, head_dim].
            causal (bool): Whether to use causal masking or not.
            padding_mask (Optional[torch.Tensor]): Padding tensor of shape [B, T].

        Returns:
            torch.Tensor: Output tensor of same shape.
        """
        # Reshape to [B, :, T, head_dim], where : can be num_heads or 1 for KV else num_heads
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Handle padding
        if padding_mask is not None:
            assert (
                padding_mask.shape == (query.size(0), query.size(2)) # [B, T]
            ), f"expected: {(query.size(0), query.size(2))}, got {padding_mask.shape}"
            # True -> attend, False -> pad (don't attend)
            padding_mask = padding_mask.bool()
            padding_mask = padding_mask[:, None, None, :] # [B, 1, 1, T]
            padding_mask = padding_mask.expand(query.size(0), 1, query.size(2), key.size(2)) # [B, 1, T_q, T_k]

            assert (
                padding_mask.dtype == torch.bool
            ), f"expected: bool, got {padding_mask.dtype}"
            assert (
                padding_mask.shape == (query.size(0), 1, query.size(2), key.size(2))
            ), f"expected: {(query.size(0), 1, query.size(2), key.size(2))}, got {padding_mask.shape}"

            # Handle causal masking
            if causal:
                causal_mask = (
                    torch.tril(
                        torch.ones(query.size(2), key.size(2), dtype=torch.bool).to(device)
                    )
                ) # [T_q, T_k]

                assert torch.all(
                    causal_mask.float().triu(1) == 0
                ), f"Causal masking failed, all should have been zeros: \n{causal_mask.float().triu(1)}"

                assert (
                    causal_mask.shape == (query.size(2), key.size(2))
                ), f"expected: {(query.size(2), key.size(2))}, got {causal_mask.shape}"
                assert (
                    causal_mask.dtype == torch.bool
                ), f"expected: bool, got {causal_mask.dtype}"

                # Add singleton dimensions to broadcast with padding mask
                causal_mask = causal_mask[None, None, ...] # [1, 1, T_q, T_k]

                assert (
                    causal_mask.dim() == padding_mask.dim()
                ), f"expected 4, 4 got {causal_mask.dim()}, {padding_mask.dim()}"

                # Aggregate into global mask
                attn_mask = padding_mask & causal_mask

            attn_mask = attn_mask.expand(
                query.size(0), self.num_heads, query.size(2), key.size(2)
            ) # [B, num_heads, T_q, T_k]
        else:
            attn_mask = None

        if attn_mask is not None:
            assert (
                attn_mask.shape == (query.size(0), self.num_heads, query.size(2), key.size(2))
            ), f"expected {(query.size(0), self.num_heads, query.size(2), key.size(2))}, got {attn_mask.shape}"

        # [B, num_heads, T, head_dim]
        attn_out = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attn_mask,
            is_causal=causal if padding_mask is None else False,
            scale=self.softmax_scale,
            enable_gqa=False # Manually done
        )

        assert (
            attn_out.shape == (query.size(0), self.num_heads, query.size(2), self.head_dim)
        ), f"expected {(query.size(0), self.num_heads, query.size(2), self.head_dim)}, got {attn_out.shape}"

        # Reshape to 3D output
        attn_out = (
            attn_out
            .transpose(1, 2)
            .contiguous()
            .view(query.size(0), query.size(2), self.d_model)
        ) # [B, T, d_model]

        assert (
            attn_out.shape == (query.size(0), query.size(2), self.d_model)
        ), f"expected {(query.size(0), query.size(2), self.d_model)}, got {attn_out.shape}"

        return attn_out

    def _causal_self_attention(
        self,
        x: torch.Tensor,
        use_mqa: bool,
        use_qk_norm: bool,
        causal: bool,
        left_window: int,
        right_window: int,
        use_cache: bool,
        padding_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        layer_idx: Optional[int] = None
    ) -> torch.Tensor:
        """Apply causal self attention to input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, T, d_model].
            use_mqa (bool): Whether to use MQA or not.
            use_qk_norm (bool): Whether to use QK normalization or not.
            causal (bool): Whether to use causal masking or not.
            left_window (int): Left window for SWA.
            right_window (int): Right window for SWA.
            use_cache (bool): Whether to use KV caching or notl.
            padding_mask (Optional[torch.Tensor]): Padding tensor of shape [B, T].
            kv_cache (Optional[KVCache]): KVCache module.
            layer_idx (Optional[int]): Current layer to update KV's with respect to.

        Returns:
            torch.Tensor: Output tensor of shape [B, T, d_model].
        """
        # Get q, k, v
        q, k, v = self._setup_qkv(
            x, 
            use_mqa=use_mqa, 
            use_qk_norm=use_qk_norm
        )

        # Update cache if given
        if use_cache and kv_cache is not None and layer_idx is not None:
            self._update_cache(k, v, kv_cache=kv_cache, layer_idx=layer_idx)

        # Get attention output
        attn_out = self._optimized_attention(
            x,
            query=q, 
            key=k, 
            value=v,
            causal=causal,
            left_window=left_window,
            right_window=right_window,
            padding_mask=padding_mask,
        )

        return attn_out # [B, T, d_model]

    def _update_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        layer_idx: Optional[int] = None
    ) -> None:
        """Update KVCache in-place using new KV tokens.
        
        Args:
            key (torch.Tensor): Key tensor of shape [B, T, num_heads, head_dim].
            value (torch.Tensor): Value tensor of shape [B, T, num_heads, head_dim].
            kv_cache (Optional[KVCache]): KVCache module.
            layer_idx (Optional[int]): Current layer to be updated with respect to.
        """
        if kv_cache is None or layer_idx is None:
            return

        # Only concatenate if there are already tokens in cache
        if kv_cache.current_seq_len is not None and kv_cache.current_seq_len > 0:
            cached_k, cached_v = kv_cache.get(layer_idx, kv_cache.current_seq_len)
            key = torch.cat([cached_k, key], dim=1)
            value = torch.cat([cached_v, value], dim=1)

        # Update KVCache (internally updates current seqlen)
        kv_cache.update(layer_idx, key, value)

    def _setup_qkv(
        self,
        x: torch.Tensor,
        use_mqa: bool,
        use_qk_norm: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Set up qkv tensors using input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, T, d_model].
            use_mqa (bool): Whether to use MQA or not.
            use_qk_norm (bool): Whether to use QK normalization or not.

        Returns:
            Tuple:
                - torch.Tensor: Query tensor of shape [B, T, num_heads, head_dim].
                - torch.Tensor: Key tensor of shape [B, T, num_heads or 1, head_dim].
                - torch.Tensor: Value tensor of shape [B, T, num_heads or 1, head_dim].
        """
        assert x.dim() == 3, f"x must be 3 dim tensor, got {x.dim()} dims."
        B, T, _ = x.shape
        
        # Handle empty input (tokens=T=0)
        if T == 0:
            if use_mqa and self.query_groups == 1:
                return (
                    torch.empty(B, 0, self.num_heads, self.head_dim, dtype=dtype, device=device),
                    torch.empty(B, 0, 1, self.head_dim, dtype=dtype, device=device),
                    torch.empty(B, 0, 1, self.head_dim, dtype=dtype, device=device)
                )
            else:
                return (
                    torch.empty(B, 0, self.num_heads, self.head_dim, dtype=dtype, device=device),
                    torch.empty(B, 0, self.num_heads, self.head_dim, dtype=dtype, device=device),
                    torch.empty(B, 0, self.num_heads, self.head_dim, dtype=dtype, device=device)
                )

        # Get q, k, v tensors through fused projection
        if self.use_fused_proj:
            # Project to qkv
            qkv = self.qkv_proj(x) # [B, T, num_heads*head_dim + 2*query_groups*head_dim]
            assert (
                qkv.shape == (B, T, self.num_heads*self.head_dim + 2*self.query_groups*self.head_dim)
            ), f"qkv must have shape of {(
                B, T, self.num_heads*self.head_dim + 2*self.query_groups*self.head_dim
            )}, got {qkv.shape}"

            # Split into q and kv
            # q shape: [B, T num_heads*head_dim]
            # kv shape: [B, T, 2*query_groups*head_dim]
            q, kv = torch.split(
                qkv, [self.num_heads*self.head_dim, 2*self.query_groups*self.head_dim], dim=-1
            )
            assert (
                kv.shape == (B, T, 2*self.query_groups*self.head_dim)
            ), f"kv must have shape of {(B, T, 2*self.query_groups*self.head_dim)}, got {kv.shape}"

            # Split again to k and v
            # k, v shape: [B, T, query_groups, head_dim]
            k, v = kv.chunk(chunks=2, dim=-1)

        # Get q, k, v through individual projections
        else:
            q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)

            assert (
                q.shape == (B, T, self.num_heads*self.head_dim)
            ), f"q must have shape of {(B, T, self.num_heads*self.head_dim)}, got {q.shape}"
            assert (
                k.shape == v.shape == (B, T, self.query_groups*self.head_dim)
            ), (
                f"k and v must have shape of {(B, T, self.query_groups*self.head_dim)}, "
                f"got k.shape = {k.shape}, got v.shape = {v.shape}"
            )
        
        # Reshape into 4D tensors for RoPE/head expansion
        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.query_groups, self.head_dim)
        v = v.view(B, T, self.query_groups, self.head_dim)

        assert (
            q.shape == (B, T, self.num_heads, self.head_dim)
        ), f"q must have shape of {(B, T, self.num_heads, self.head_dim)}, got {q.shape}"
        assert (
            k.shape == v.shape == (B, T, self.query_groups, self.head_dim)
        ), (
            f"k and v must have shape of {(B, T, self.query_groups, self.head_dim)}, "
            f"got k.shape = {k.shape}, got v.shape = {v.shape}"
        )

        # Apply QK normalization
        if use_qk_norm:
            q, k = apply_qk_norm(query=q, key=k)

        # Apply NTKRoPE2D
        q = self.ntk_rope(q)
        k = self.ntk_rope(k)

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

        assert (
            k.shape == v.shape == (B, T, self.num_heads, self.head_dim) or 
            k.shape == v.shape == (B, T, 1, self.head_dim) # mqa
        ), (
            f"k and v must have shape of {(B, T, self.num_heads, self.head_dim)} or "
            f"{(B, T, 1, self.head_dim)}, got k.shape = {k.shape}, got v.shape = {v.shape}"
        )

        return q, k, v

    def forward(
        self,
        x: torch.Tensor,
        use_mqa: bool,
        use_qk_norm: bool,
        causal: bool,
        left_window: int,
        right_window: int,
        use_cache: bool,
        padding_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        layer_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """Forward pass of causal attention layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, T, d_model].
            use_mqa (bool): Whether to use MQA or not.
            use_qk_norm (bool): Whether to use QK normalization or not.
            causal (bool): Whether to use causal or not.
            left_window (int): Left window for causal masking.
            right_window (int): Right window for causal masking.
            use_cache (bool): Whether to use KV caching or not.
            padding_mask (Optional[torch.Tensor]): Padding tensor of shape [B, T].
            kv_cache (Optional[KVCache]): KVCache module.
            layer_idx (Optional[int]): Current layer to update keys/values with respect to.

        Returns:
            torch.Tensor: Output tensor of same shape.
        """
        with autocast(device_type=device.type, dtype=dtype):
            # Set right window to 0 for causal LM
            if causal:
                right_window = 0
            
            # Global attention
            if not self.use_windowed_attn:
                left_window, right_window = -1, -1
            
            # Attention output with caching
            attn_out = self._causal_self_attention(
                x,
                use_mqa=use_mqa,
                use_qk_norm=use_qk_norm,
                causal=causal,
                left_window=left_window,
                right_window=right_window,
                use_cache=use_cache,
                padding_mask=padding_mask,
                kv_cache=kv_cache,
                layer_idx=layer_idx
            )

            return self.o_proj(attn_out)

class CausalSelfAttentionBlock(nn.Module):
    """Causal attention block applying residuals, RMSNorm, dropout, and attention.

    Args:
        d_model (int): Dimensionality of model embeddings.
        num_heads (int): Number of attention heads.
        query_groups (int): Number of query_groups for GQA.
        rope_theta (float): Exponential base for RoPE inv freq.
        softmax_scale (float): Value to scale attention scores by.
        use_proj_bia (bool): Whether to use projection bias or not.
        use_fused_proj (bool): Whether to use qkv or seperate projections.
        use_window_attn (bool): Whether to use windowed attention or not.
        use_ntk_rope (bool): Whether to use NTK RoPE or classic RoPE.
        dropout (float): Dropout probability used for regularization.
        eps (float): Small value to maintain numerical stability in RMSNorm.
        ntk_scale_factor (Optional[float]): Scaling factor for NTK RoPE.
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
        dropout: float,
        eps: float,
        ntk_scale_factor: Optional[float] = None
    ):
        super().__init__()

        self.attention = CausalSelfAttention(
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
        self.rms_norm = RMSNorm(d_model=d_model, eps=eps)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        use_mqa: bool,
        use_qk_norm: bool,
        causal: bool,
        left_window: int,
        right_window: int,
        use_cache: bool,
        padding_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        layer_idx: Optional[int] = None
    ) -> torch.Tensor:
        """Forward pass of causal attention block layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, T, d_model].
            use_mqa (bool): Whether to use MQA or not.
            use_qk_norm (bool): Whether to use QK normalization or not.
            causal (bool): Whether to use causal or not.
            left_window (int): Left window for causal masking.
            right_window (int): Right window for causal masking.
            use_cache (bool): Whether to use KV caching or not.
            padding_mask (Optional[torch.Tensor]): Padding tensor of shape [B, T].
            kv_cache (Optional[KVCache]): KVCache module.
            layer_idx (Optional[int]): Current layer to update keys/values with respect to.

        Returns:
            torch.Tensor: Output tensor of same shape.
        """
        with autocast(device_type=device.type, dtype=dtype):
            return x + self.dropout(self.attention(
                self.rms_norm(x),
                use_mqa=use_mqa,
                use_qk_norm=use_qk_norm,
                causal=causal,
                left_window=left_window,
                right_window=right_window,
                use_cache=use_cache,
                padding_mask=padding_mask,
                kv_cache=kv_cache,
                layer_idx=layer_idx
            ))

def test_attention():
    d_model, num_heads, query_groups, rope_theta = 512, 32, 8, 10000.0
    eps, dropout = 1e-7, 0.15
    softmax_scale = 1 / (d_model // num_heads) ** 0.5
    attention = CausalSelfAttentionBlock(
        d_model, num_heads, query_groups, rope_theta, softmax_scale, 
        use_proj_bias=False, use_fused_proj=True, use_windowed_attn=False, 
        use_ntk_rope=True, eps=eps, dropout=dropout,
        ntk_scale_factor=0.5,
    ).to(device)
    kv_cache = KVCache(
        max_batch_size=32,
        max_seq_len=2048,
        num_heads=num_heads,
        head_dim=d_model//num_heads,
        num_layers=4
    )
    B, T = 4, 16
    x = torch.randn(B, T, d_model).to(device)
    padding_mask = torch.randint(0, 2, (B, T), dtype=torch.bool).to(device)
    x_out = attention(
        x,
        use_mqa=False,
        use_qk_norm=False,
        causal=True,
        left_window=-1,
        right_window=-1,
        use_cache=True,
        padding_mask=padding_mask,
        kv_cache=kv_cache,
        layer_idx=2
    )
    return x_out

if __name__ == "__main__":
    x = test_attention()
    print(x.shape)
