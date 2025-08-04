from configs.setup_env import (
    device, 
    dtype,
    use_flash_attn,
    flash_attn_varlen_qkvpacked_func
)

import math
import warnings
from typing import Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.transformers.nlp.model import RoPE
from src.transformers.nlp.model import KVCache

# TODO: split attention module into 4 main functions:
# - extend_kv_heads() -> complete
# - _optimized_attention() -> uncomplete
# - _fallback_attention() -> uncomplete
# - forward() -> update forward to support new attn funcs

class Attention(nn.Module):
    """Apply Grouped Query Attention (GQA) to QKV vectors as well as causal masking.

    Args:
        d_model (int): Dimensionality of the model's embeddings.
        num_heads (int): Number of attention heads for KV vectors, shared across grouped query heads in GQA.
        query_groups (int): Number of query groups that partition the queries into subsets.
        theta (float): Exponential base of the inverse frequency.
    """
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        query_groups: int, 
        theta: float
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.query_groups = query_groups
        self.head_dim = d_model // num_heads
        self.heads_per_group = num_heads // query_groups

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
                )
        if num_heads % query_groups != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by query_groups ({query_groups})"
                )

        # QKV projection matrix
        self.w_qkv = nn.Linear(
            d_model,
            num_heads * self.head_dim + 2 * query_groups * self.head_dim,
            bias=False,
            dtype=dtype
        ).to(device)

        # O projection matrix
        self.w_o = nn.Linear(
            d_model,
            d_model,
            bias=False,
            dtype=dtype
        ).to(device)

        self.rope = RoPE(self.head_dim, theta)

    def _extend_kv_heads(
        self,
        kv_tensor: torch.Tensor, 
        heads_per_group: int, 
        kv_heads_dim: int,
        use_mqa: bool = False,
    ):
        """Extend kv heads to query heads (num_heads).
        
        Args:
            input_tensor (torch.Tensor): Key or value tensor to be repeated.
            heads_per_group (int): Heads per group computed as num_heads // query_groups.
            dim_to_repeat (int): Dimension to of the input tensor to be repeated.
            use_mqa (bool): Apply Multi-Query attention instead of GQA.
                Constraints: query_groups == 1.

        Returns:
            torch.Tensor: Key or value tensor with specific dimension extended.
        """
        if use_mqa and kv_tensor.size(kv_heads_dim) == 1:
            warnings.warn(
                "Using multi-query attention, consider switching to GQA for more expressiveness"
                )
            # For MQA, we return tensor as is
            return kv_tensor
        # For GQA, we expand kv heads and then return tensor
        return torch.repeat_interleave(kv_tensor, heads_per_group, dim=kv_heads_dim)

    def forward(
        self,
        x: torch.Tensor,
        window_size: Tuple[int, int],
        causal: bool = True,
        padding_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        layer_idx: Optional[int] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Forward pass for GQA layer.

        Args:
            x (torch.Tensor): Input tensor of shape [B, T, D].
            window_size (Tuple[int, int]): Window size for sliding window attention
            causal (bool): To apply causal masking or not.
            padding_mask (torch.Tensor, optional): Padding mask of shape [B, T] where
                True indicates valid tokens and False indicates padding tokens.
            kv_cache (Optional[KVCache]): Key-value cache for efficient generation.
            layer_idx (Optional[int]): Index of the current layer for cache access.
            use_cache (bool): Whether to use KV caching during forward pass.

        Returns:
            Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
                - torch.Tensor: Attention output tensor of shape [B, T, d_model].
                - Optional[Dict[str, torch.Tensor]]: Cache dictionary with 'k' and 'v' tensors.

        Raises:
            ValueError if `x` does not have 3 shape [B, T, d_model].
            ValueError if `padding_mask` does not have shape [B, T].

        Requirements for Flash Attention V2:
            Flash Attention import must be successful.
            `device` must be cuda.
            q, k, v tensors must have dtype of float16 or bfloat16.

        NOTE: FLASH ATTENTION 2 + SWA HAS NOT BEEN TESTED DUE TO HARDWARE LIMITATIONS.
        """
        with torch.amp.autocast(device_type=device.type, dtype=dtype):
            B, T, D = x.shape
            if x.shape != (B, T, D):
                raise ValueError(f"Expected x to have shape [B, T, d_model], got {x.shape}")
            if T == 0:
                # Return empty tensor if sequence length is 0
                return torch.empty(B, 0, D, device=x.device, dtype=x.dtype), None

            # Project QKV
            qkv = self.w_qkv(x) # [B, T, num_heads * head_dim + 2 * query_groups * head_dim]
            assert (
                qkv.shape == (B, T, self.num_heads * self.head_dim + 2 * self.query_groups * self.head_dim)
            ), (
                f"qkv must have shape of {(B, T, self.num_heads * self.head_dim + 2 * self.query_groups * self.head_dim)} "
                f"got, {qkv.shape}"
            )

            # q shape: [B, T, num_heads * head_dim]
            # kv shape: [B, T, 2 * query_groups * head_dim]
            q, kv = torch.split(qkv, [self.num_heads * self.head_dim, 2 * self.query_groups * self.head_dim], dim=-1)
            assert (
                q.shape == (B, T, self.num_heads * self.head_dim)
            ), f"q must have shape of {(B, T, self.num_heads * self.head_dim)}, got {q.shape}"
            assert (
                kv.shape == (B, T, 2 * self.query_groups * self.head_dim)
            ), f"kv must have shape of {(B, T, 2 * self.query_groups * self.head_dim)}, got {kv.shape}"

            # k shape: [B, T, query_groups * head_dim]
            # v shape: [B, T, query_groups * head_dim]
            k, v = torch.chunk(kv, 2, dim=-1)
            assert (
                k.shape == (B, T, self.query_groups * self.head_dim)
            ), f"k must have shape of {(B, T, self.query_groups * self.head_dim)}, got {k.shape}"
            assert (
                v.shape == (B, T, self.query_groups * self.head_dim)
            ), f"v must have shape of {(B, T, self.query_groups * self.head_dim)}, got {v.shape}"

            # Assertions before reshaping
            assert (
                q.shape == (B, T, self.num_heads * self.head_dim)
            ), f"q must have shape {(B, T, self.num_heads * self.head_dim)}, got {q.shape}"
            assert (
                k.shape == (B, T, self.query_groups * self.head_dim)
            ), f"k must have shape {(B, T, self.query_groups * self.head_dim)}, got {k.shape}"
            assert (
                v.shape == (B, T, self.query_groups * self.head_dim)
            ), f"v must have shape {(B, T, self.query_groups * self.head_dim)}, got {v.shape}"

            # Reshape into 4D tensors for GQA
            q = q.contiguous().view(B, T, self.num_heads, self.head_dim) # [B, T, num_heads, head_dim]
            k = k.contiguous().view(B, T, self.query_groups, self.head_dim) # [B, T, query_groups, head_dim]
            v = v.contiguous().view(B, T, self.query_groups, self.head_dim) # [B, T, query_groups, head_dim]

            # Apply RoPE
            q = self.rope(q) # [B, T, num_heads, head_dim]
            k = self.rope(k) # [B, T, query_groups, head_dim]

            assert (
                q.shape == (B, T, self.num_heads, self.head_dim)
            ), f"q must have shape {(B, T, self.num_heads, self.head_dim)}, got {q.shape}"
            assert (
                k.shape == (B, T, self.query_groups, self.head_dim)
            ), f"k must have shape {(B, T, self.query_groups, self.head_dim)}, got {k.shape}"
            assert (
                v.shape == (B, T, self.query_groups, self.head_dim)
            ), f"v must have shape {(B, T, self.query_groups, self.head_dim)}, got {v.shape}"

            # Handle KV cache
            if use_cache and kv_cache is not None and layer_idx is not None:
                # Get KV cache with the current sequence length
                cached_k, cached_v = kv_cache.get(layer_idx, kv_cache.current_seq_len)
                if cached_k is not None and cached_v is not None:
                    # Concatenate cached and new KV
                    k = torch.cat([cached_k, k], dim=1)
                    v = torch.cat([cached_v, v], dim=1)
                # Update cache using only the T most recent tokens
                kv_cache.update(layer_idx, k[:, -T:], v[:, -T:])
            
            # Extend KV tensors (if query_groups==1, no expansion is done)
            k = self._extend_kv_heads(kv_tensor=k, heads_per_group=self.heads_per_group, kv_heads_dim=2)
            v = self._extend_kv_heads(kv_tensor=v, heads_per_group=self.heads_per_group, kv_heads_dim=2)

            # Assert extension of kv heads worked or MQA was succesful
            assert (
                k.size(2) == self.num_heads or k.size(2) == 1
            ), f"k.size(2) must be {self.num_heads} or 1, got {k.size(2)}"
            assert (
                v.size(2) == self.num_heads or k.size(2) == 1
            ), f"k.size(2) must be {self.num_heads} or 1, got {k.size(2)}"

            # Assert right window is 0 for causal LM
            assert (
                window_size[1] == 0
            ), (
                f"right window must be equal to 0, got {window_size[1]}. "
                f"set window_size to (left, 0)"
            )

            # FlashAttention 2 - requires CUDA/flash attn available/bf16 or fp16
            if (
                use_flash_attn  and device.type == "cuda"
                and q.dtype in [torch.float16, torch.bfloat16]
                and k.dtype in [torch.float16, torch.bfloat16]
                and v.dtype in [torch.float16, torch.bfloat16]
            ):
                qkv_packed = torch.stack([q, k, v], dim=3).contiguous() # [B, T, num_heads, 3, head_dim]
                assert (
                    qkv_packed.is_contiguous()
                ), "qkv_packed must be contiguous."

                # Handle padding mask for FlashAttention
                if padding_mask is not None:
                    if padding_mask.shape != (B, T):
                        raise ValueError(
                            f"Expected padding mask shape of {(B, T)}, got {padding_mask.shape}"
                        )
                    # Ensure padding mask is a boolean tensor
                    padding_mask = padding_mask.bool()
                    assert (
                        padding_mask.dtype == torch.bool
                    ), f"padding_mask must be boolean tensor, got {padding_mask.dtype}"

                    # Get sequence lengths
                    seqlens = padding_mask.sum(dim=1).int() # [B]
                    assert (
                        len(seqlens) == B
                    ), f"len(seqlens) must be equal to B, got {len(seqlens)} != {B}"

                    # Get maximum sequence length
                    max_seqlen = seqlens.max().item()

                    # Compute cumulative sequence lengths
                    cu_seqlens = torch.cat([
                        torch.tensor([0], dtype=torch.int32).to(device), # [0]
                        seqlens.cumsum(0) # [B]
                    ], dim=0) # [B + 1]

                    # Assert cu_seqlens shape/dtype
                    assert (
                        cu_seqlens.shape == (B + 1)
                    ), f"cu_seqlens must have shape {(B + 1)}, got {cu_seqlens.shape}"
                    assert (
                        cu_seqlens.dtype == torch.int32
                    ), f"cu_seqlens must have dtype int32, got {cu_seqlens.dtype}"

                    # Flatten padding mask
                    valid_tokens = padding_mask.flatten() # [B * T]
                    assert (
                        valid_tokens.shape == (B * T)
                    ), f"valid_tokens must have shape {(B * T)} got {valid_tokens.shape}"

                    # Index by valid tokens to only get non-padding token positions
                    qkv_packed = (
                        qkv_packed.view(-1, self.num_heads, 3, self.head_dim)
                        .transpose(1, 2)
                        .contiguous()
                        )[valid_tokens] # [B * T, 3, num_heads, head_dim]
                    
                    assert (
                        qkv_packed.shape == (B * T, 3, self.num_heads, self.head_dim)
                    ), f"qkv_packed must have shape {(B*T, 3, self.num_heads, self.head_dim)}, got {qkv_packed.shape}"
                    assert (
                        qkv_packed.is_contiguous()
                    ), "qkv_packed must be contiguous"
                else:
                    # Get cumulative sequence lengths
                    cu_seqlens = torch.arange(
                        0,
                        (B + 1) * T,
                        T,
                        dtype=torch.int32,
                    ).to(device) # [B + 1]

                    # Get maximum sequence length
                    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
                    max_seqlen = seqlens.max().item()

                    # cu_seqlens dtype/shape check
                    assert (
                        cu_seqlens.shape == (B + 1)
                    ), f"cu_seqlens must have shape {(B + 1)}, got {cu_seqlens.shape}"
                    assert (
                        cu_seqlens.dtype == torch.int32
                    ), f"cu_seqlens must have dtype int32, got {cu_seqlens.dtype}"

                    qkv_packed = (
                        qkv_packed.view(-1, self.num_heads, 3, self.head_dim)
                        .contiguous()
                        .transpose(1, 2)
                    ) # [B * T, 3, num_heads, head_dim]

                    # qkv_packed shape check
                    assert (
                        qkv_packed.shape == (B * T, 3, self.num_heads, self.head_dim)
                    ), f"qkv_packed must have shape {(B * T, 3, self.num_heads, self.head_dim)}, got {qkv_packed.shape}"
                    assert (
                        qkv_packed.is_contiguous()
                    ), "qkv_packed must be contiguous"

                # Call FlashAttention 2
                out = flash_attn_varlen_qkvpacked_func(
                    qkv_packed,
                    cu_seqlens,
                    max_seqlen,
                    causal=causal,
                    softmax_scale=1.0 / (math.sqrt(self.head_dim)),
                    window_size=window_size,
                ) # [B * T, num_heads, head_dim]

                assert (
                    out.shape == (B * T, self.num_heads, self.head_dim)
                ), f"out shape must be {(B * T, self.num_heads, self.head_dim)} got {out.shape}"

                # This is done because the FlashAttention 2 function returns only unpadded tokens
                if padding_mask is not None:
                    full_out = torch.zeros(
                        B * T,
                        self.num_heads, 
                        self.head_dim,  
                        dtype=out.dtype, 
                    ).to(out.device)
                    # Update valid and padded token positions where valid_tokens is a boolean tensor.
                    full_out[valid_tokens] = out
                    out = full_out.view(B, T, D) # Reshape output to [B, T, d_model]
                else:
                    # If no padding, reshape right away
                    out = out.view(B, T, D)

            # Fallback to PyTorch SPDA if no cuda
            else:
                warnings.warn(
                    "Flash Attention 2/SWA not available, falling back to PyTorch SDPA."
                    )
                # Reshape q, k, v to [B, num_heads, T, head_dim] (assuming kv_heads get extended)
                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)

                assert (
                    q.shape == (B, self.num_heads, T, self.head_dim)
                ), f"q must have shape {(B, self.num_heads, T, self.head_dim)}"
                assert (
                    k.shape == (B, self.num_heads, T, self.head_dim) or
                    k.shape == (B, 1, T, self.head_dim)
                ), (
                    f"k must have shape {(B, self.num_heads, T, self.head_dim)} "
                    f"or {(B, 1, T, self.head_dim)} got {k.shape}"
                )
                assert (
                    v.shape == (B, self.num_heads, T, self.head_dim) or
                    v.shape == (B, 1, T, self.head_dim)
                ), (
                    f"v must have shape {(B, self.num_heads, T, self.head_dim)}, "
                    f"or {(B, 1, T, self.head_dim)} got {k.shape}"
                )

                # Initialize padding mask
                attn_mask = None

                # Apply padding mask for PyTorch SDPA
                if padding_mask is not None:
                    if padding_mask.shape != (B, T):
                        raise ValueError(
                            f"Expected padding mask of shape ({B, T}), got {padding_mask.shape}"
                            )
                    padding_mask = padding_mask.bool() # Ensure padding mask is a boolean tensor
                    attn_mask = padding_mask.unsqueeze(1).unsqueeze(2) # [B, 1, 1, T]
                    attn_mask = attn_mask.expand(B, 1, T, k.size(2)) # [B, 1, T_q, T_k]
                    assert (
                        attn_mask.shape == (B, 1, q.size(2), k.size(2))
                    ), f"attn_mask must have shape {(B, 1, q.size(2), k.size(2))}, got {attn_mask.shape}"

                    # Apply causal masking
                    if causal:
                        # Causal mask shape: [T_q, T_k] where the upper right diagonal portion is False.
                        causal_mask = torch.tril(torch.ones(T, k.size(2), dtype=torch.bool).to(device))
                        assert (
                            T == q.size(2)
                        ), f"T must be equal to {q.size(2)}, got {T} != {q.size(2)}"
                        assert (
                            causal_mask.shape == (q.size(2), k.size(2)) # [T_q, T_k]
                        ), f"causal_mask must have shape {(q.size(2), k.size(2))}, got {causal_mask.shape}"
                        # Since the attention scores tensor has shape [B, num_heads, T_q, T_k]
                        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0) # [1, 1, T_q, T_k]
                        assert (
                            attn_mask.dim() == causal_mask.dim()
                        ), "attn_mask and causal_mask must be of same length for broadcasting."
                        attn_mask = attn_mask & causal_mask # [B, 1, T_q, T_k]
                        assert (
                            attn_mask.shape == (B, 1, q.size(2), k.size(2))
                        ), f"attn_mask must have shape {(B, 1, q.size(2), q.size(2))}, got {attn_mask.shape}"

                    # Expand to [B, num_heads, T_q, T_k] only now to avoid unnecessary memory usage.
                    attn_mask = attn_mask.expand(B, self.num_heads, T, k.size(2))
                    assert (
                        attn_mask.shape == (B, self.num_heads, q.size(2), k.size(2))
                    ), f"attn_mask must have shape {(B, self.num_heads, q.size(2), k.size(2))}, got {attn_mask.shape}"

                # Call PyTorch's scaled dot product attention
                out = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    is_causal=causal if padding_mask is None else False,
                ) # [B, num_heads, T, head_dim]
                assert (
                    out.shape == (B, self.num_heads, T, self.head_dim)
                ), f"out must have shape {(B, self.num_heads, T, self.head_dim)}, got {out.shape}"

                # [B, num_heads, T, head_dim] -> [B, T, d_model]
                out = out.transpose(1, 2).contiguous().view(B, T, D)
                assert (
                    out.shape == (B, T, self.d_model)
                ), f"out must have shape {(B, T, self.d_model)}, got {out.shape}"

            # Get cache output for the Attention layer
            cache_out = {'k': k[:, -T:], 'v': v[:, -T:]} if use_cache else None

            # Final projection
            return self.w_o(out), cache_out
