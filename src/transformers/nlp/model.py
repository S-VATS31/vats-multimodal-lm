from configs.setup_env import (
    device,
    dtype, 
    use_flash_attn, 
    flash_attn_varlen_qkvpacked_func,
    use_xformers_swiglu,
    swiglu
)

import math
import warnings
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from configs.transformers.nlp.model_args.model_args_medium import ModelArgs

class RoPE(nn.Module):
    """Rotary positional embeddings (RoPE) to be applied to the query and key vectors.

    Args:
        head_dim (int): Dimension of each attention head.
        theta (float): Exponential base of the inverse frequency.

    Raises:
        ValueError if `head_dim` is not divisble by 2.
    """
    def __init__(self, head_dim: int, theta: float):
        super().__init__()

        if head_dim % 2 != 0:
            raise ValueError(f"head_dim ({head_dim}) must be divisible by 2 for even splitting.")
        self.head_dim = head_dim

        # Compute inverse frequency
        # inv_freq = 1 / (theta ^ (2i/d)) where d = head_dim
        # We set dtype=float32 to maintain numerical stability when exponentiating
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)).to(device)
        assert (
            inv_freq.dtype == torch.float32
        ), f"inv_freq must have dtype of float32, got {inv_freq.dtype}"
        self.register_buffer("inv_freq", inv_freq)

        # Lazy cache - will be populated as needed to avoid recomputing sin/cos
        self.register_buffer("cos_cache", torch.empty(0))
        self.register_buffer("sin_cache", torch.empty(0))
        self.cached_seq_len = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the RoPE layer.

        Args:
            x (torch.Tensor): Input tensor of shape [B, T, num_heads, head_dim].

        Returns:
            torch.Tensor: Tensor with RoPE applied, same shape as input.
        """
        with torch.amp.autocast(device_type=device.type, dtype=dtype):
            seq_len = x.size(1)

            # Update cache if needed
            if seq_len > self.cached_seq_len:
                self._update_cache(seq_len)

            # Use cached values instead of recomputing
            # Update all values up till sequence length
            cos_freqs = self.cos_cache[:seq_len] # [T, head_dim // 2]
            sin_freqs = self.sin_cache[:seq_len] # [T, head_dim // 2]

            assert (
                cos_freqs.shape == (seq_len, self.head_dim // 2)
            ), f"cos_freqs must have shape {(seq_len, self.head_dim //2)} got {cos_freqs.shape}"
            assert (
                sin_freqs.shape == (seq_len, self.head_dim // 2)
            ), f"sin_freqs must have shape {(seq_len, self.head_dim //2)} got {sin_freqs.shape}"

            # Apply rotary embeddings
            return self._apply_rope(x, cos_freqs, sin_freqs)

    def _update_cache(self, seq_len: int) -> None:
        """Cache sine and cosine to prevent re-computation during forward pass.

        Args:
            seq_len (int): Sequence length to cache for.
        """
        # Create position indices
        pos = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)

        # Compute frequencies for each position
        freqs = torch.outer(pos, self.inv_freq) # [T, head_dim//2]

        # Create rotation matrix components and cache them
        self.cos_cache = torch.cos(freqs) # [T, head_dim//2]
        self.sin_cache = torch.sin(freqs) # [T, head_dim//2]
        self.cached_seq_len = seq_len

    def _apply_rope(
        self,
        x: torch.Tensor,
        cos_freqs: torch.Tensor,
        sin_freqs: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the rotation using the RoPE formula.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, num_heads, head_dim].
            cos_freqs (torch.Tensor): Cosine frequencies of shape [seq_len, head_dim//2].
            sin_freqs (torch.Tensor): Sine frequencies of shape [seq_len, head_dim//2].

        Returns:
            torch.Tensor: Rotated output tensor with positional awareness.
        """
        with torch.amp.autocast(device_type=device.type, dtype=dtype):
            if x.dim() != 4:
                raise ValueError(f"x must have 4 dimensions, got {x.dim()}")
            # Split x into even and odd dimensions
            x1 = x[..., ::2]  # even (2i)
            x2 = x[..., 1::2] # odd (2i+1)

            # Expand frequency tensors to match input dimensions
            # We need these to be 4 dimensional tensors for correct broadcasting
            cos_freqs = cos_freqs[None, :, None, :] # [1, T, 1, head_dim//2]
            sin_freqs = sin_freqs[None, :, None, :] # [1, T, 1, head_dim//2]

            assert (
                cos_freqs.shape == (1, x.size(1), 1, self.head_dim // 2)
            ), f"cos_freqs must have shape {(1, x.size(1), 1, self.head_dim // 2)}, got {cos_freqs.shape}"
            assert (
                sin_freqs.shape == (1, x.size(1), 1, self.head_dim // 2)
            ), f"sin_freqs must have shape {(1, x.size(1), 1, self.head_dim // 2)}, got {sin_freqs.shape}"

            # Complex rotation via rotation matrix
            # rotation_matrix = [[cos(x), -sin(x)], [sin(x), cos(x)]]
            # x_even_rot = x_even * rotation_matrix[0]
            # x_odd_rot = x_odd * rotation_matrix[1]
            rotated_x1 = x1 * cos_freqs - x2 * sin_freqs
            rotated_x2 = x1 * sin_freqs + x2 * cos_freqs

            # Interleave the rotated components back
            rotated_x = torch.stack([rotated_x1, rotated_x2], dim=-1)
            rotated_x = rotated_x.flatten(-2)

            return rotated_x

    def get_cos_sin_cache(
        self,
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cache sine and cosine to prevent re-computation during forward pass.

        Args:
            seq_len (int): Sequence length

        Returns:
            tuple: Cached values.
                - torch.Tensor: Cached cosine frequencies.
                - torch.Tensor: Cached sine frequencies.
        """
        # Ensure cache is up to date for this sequence length
        if seq_len > self.cached_seq_len:
            self._update_cache(seq_len)

        # Return the cached values for the requested length
        cos_freqs = self.cos_cache[:seq_len]
        sin_freqs = self.sin_cache[:seq_len]
        return cos_freqs, sin_freqs


class RMSNorm(nn.Module):
    """Apply RMSNorm to input tensors to normalize their root mean square norm.

    Formula:
        x_normalized = x / RMS
        Where RMS = sqrt(mean(x**2))

    Args:
        d_model (int): Dimensionality of the model's embeddings.
        eps (float): Small epsilon value to ensure numerical stability.
    """
    def __init__(self, d_model: int, eps: float):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model)).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform forward pass of the RMSNorm layer.

        Args:
            x (torch.Tensor): Input tensor of shape [B, T, d_model].

        Returns:
            torch.Tensor: Normalized output tensor of same shape.
        """
        with torch.amp.autocast(device_type=device.type, dtype=dtype):
            return self.weight * (
                x / torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
            )


class KVCache:
    """Key-Value cache for efficient autoregressive generation.

    Args:
        max_batch_size (int): Maximum batch size supported by the cache.
        max_seq_len (int): Maximum sequence length supported by the cache.
        num_heads (int): Number of attention heads (or query groups for GQA).
        head_dim (int): Dimension of each attention head.
        num_layers (int): Number of transformer layers.
    """
    def __init__(
        self,
        max_batch_size: int,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        num_layers: int,
    ):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_layers = num_layers

        # Initialize cache state to None
        self.cache = None
        self.current_seq_len = None
        self.batch_size = None

    def initialize(self, batch_size: int) -> None:
        """Initialize or reset the cache for a given batch size.

        Args:
            batch_size (int): Number of sequences to process.

        Raises:
            ValueError: If batch_size exceeds max_batch_size.
        """
        if batch_size > self.max_batch_size:
            raise ValueError(f"Batch size {batch_size} exceeds maximum {self.max_batch_size}")

        self.batch_size = batch_size
        self.current_seq_len = 0

        # Initialize KV cache with zeros
        self.cache = [
            {
                'k': torch.zeros((
                    batch_size, self.max_seq_len, self.num_heads, self.head_dim
                    ), dtype=dtype).to(device),
                'v': torch.zeros((
                    batch_size, self.max_seq_len, self.num_heads, self.head_dim
                    ), dtype=dtype).to(device)
            }
            for _ in range(self.num_layers)
        ]

    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Update the cache with new key and value tensors for a specific layer. Updates KV cache
        in-place.

        Args:
            layer_idx (int): Index of the transformer layer to query.
            k (torch.Tensor): Key tensor of shape [batch_size, seq_len, num_heads, head_dim].
            v (torch.Tensor): Value tensor of shape [batch_size, seq_len, num_heads, head_dim].

        Raises:
            ValueError: If sequence length exceeds max_seq_len.
        """
        if self.cache is None or self.batch_size != k.size(0):
            self.initialize(k.size(0))

        new_seq_len = k.size(1)
        # Ensure current T + new T <= max T
        if self.current_seq_len + new_seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {self.current_seq_len + new_seq_len} exceeds maximum {self.max_seq_len}")

        # Update cache with new key and value tensors
        self.cache[layer_idx]['k'][:, self.current_seq_len:self.current_seq_len + new_seq_len] = k
        self.cache[layer_idx]['v'][:, self.current_seq_len:self.current_seq_len + new_seq_len] = v

    def get(
        self, 
        layer_idx: int, 
        seq_len: int
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Retrieve key and value tensors up to the specified sequence length.

        Args:
            layer_idx (int): Index of the transformer layer to query.
            seq_len (int): Sequence length to retrieve.

        Returns:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]: Key and value tensors up to seq_len,
                or (None, None) if cache is uninitialized or seq_len exceeds current_seq_len.
        """
        if self.cache is None or seq_len > self.current_seq_len:
            return None, None

        # Return KV cache up to the requested sequence length
        return (
            self.cache[layer_idx]['k'][:, :seq_len],
            self.cache[layer_idx]['v'][:, :seq_len]
        )

    def increment_seq_len(self, increment: int) -> None:
        """Increment the current sequence length after updating the cache.

        Args:
            increment (int): Amount to increment the current sequence length.
        """
        self.current_seq_len += increment

    def reset(self) -> None:
        """Reset the cache to its initial state."""
        self.cache = None
        self.current_seq_len = None
        self.batch_size = None


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


# TODO: add assertions to all MoE components

class SwiGLUExpert(nn.Module):  
    """SwiGLU expert layer.

    Args:
        d_model (int): Dimensionality of the model's embeddings.
        d_ffn (int): Dimensionality of the feed forward network.
        dropout (float): Probability of model components being dropped out.
    """
    def __init__(self, d_model: int, d_ffn: int, dropout: float):
        super().__init__()

        self.weight1 = nn.Linear(d_model, d_ffn, bias=False)
        self.weight2 = nn.Linear(d_model, d_ffn, bias=False)
        self.weight3 = nn.Linear(d_ffn, d_model, bias=False)
        self.dropout = nn.Dropout(p=dropout)
    
    def _optimized_swiglu(self, x: torch.Tensor) -> torch.Tensor:
        """Optimized SwiGLU activation using xformers.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, T, d_model].
            
        Returns:
            torch.Tensor: Output tensor of same shape.

        Notes:
            - We do self.matrix.weight.T as xformers func needs the matrix to be broadcastable
            with x raw (x @ matrix) (basically, without transpose we cannot apply matrix multiplication
            due to a shape error).
            - When we call SwiGLU, all the `None`s being passed, are `Optional[torch.Tensor]`s
            representing bias.

        Requirements:
            xformers swiglu() import must be succesful.
            `device` must be cuda.
            x must live on `device` of cuda.
            x must have dtype of float16 or bfloat16.
        """
        # xformers swiglu
        if (
            use_xformers_swiglu
            and device.type == "cuda"
            and x.is_cuda
            and x.dtype in [torch.float16, torch.bfloat16]
        ):
            return self.dropout(swiglu(
                x.contiguous(),
                self.weight1.weight.T, None,
                self.weight2.weight.T, None,
                self.weight3.weight.T, None
            ))
        # pytorch swiglu fallback
        else:
            warnings.warn("xformers SwiGLU not available, falling back to PyTorch SwiGLU.")
            return self._pytorch_swiglu(x)
    
    def _pytorch_swiglu(self, x: torch.Tensor) -> torch.Tensor:
        """Optimized SwiGLU activation using xformers.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, T, d_model].
            
        Returns:
            torch.Tensor: Output tensor of same shape.
        """
        return self.dropout(self.weight3(F.silu(self.weight1(x)) * self.weight2(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SwiGLU Expert layer.

        Formula:
            SwiGLU(x) = Dropout(Swish((W1 @ x) * (W3 @ x)) @ W2)

        Args:
            x (torch.Tensor): Input tensor of shape [B, T, d_model].

        Returns:
            torch.Tensor: Output tensor with the same shape.
        """
        with torch.amp.autocast(device_type=device.type, dtype=dtype):
            return self._optimized_swiglu(x)

class TopKRouter(nn.Module):
    """TopK Routing for MoE layer.

    Args:
        d_model (int): Dimensionality of the model's embeddings.
        num_experts (int): Number of FFNs (SwiGLU in this case).
        top_k (int): Number of experts each token is routed to.
        use_aux_loss (bool): Flag to use auxiliary loss or not.
    """
    def __init__(
            self,
            d_model: int,
            num_experts: int,
            top_k: int,
            use_aux_loss: bool = True,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_aux_loss = use_aux_loss

        # Set up router (projects from d_model to num_experts)
        self.router = nn.Linear(d_model, num_experts)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  
        """Forward pass of the routing layer.

        Args:
            x (torch.Tensor): Input tensor of shape [B, T, d_model].

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]. A tuple containing:
                - torch.Tensor: Tensor containing gating scores.
                - Torch.Tensor: Indices of gating scores.
                - torch.Tensor: Tensor containing auxiliary loss.
        """
        with torch.amp.autocast(device_type=device.type, dtype=dtype):
            B, T, d_model = x.shape

            # Flatten for efficiency
            x_flattened = x.view(-1, d_model) # [B * T, d_model]

            # Compute logits
            logits = self.router(x_flattened) # [B * T, num_experts]

            # Get probabilities
            prob_scores = F.softmax(logits, dim=-1) # [B * T, num_experts]

            # Get top-k experts
            top_k_values, top_k_indices = torch.topk(prob_scores, self.top_k, dim=-1) # Both: [B * T, top_k]

            # Get weights
            top_k_weights = top_k_values / torch.sum(top_k_values, dim=-1, keepdim=True)

            # Reshape
            expert_weights = top_k_weights.view(B, T, self.top_k)
            expert_indices = top_k_indices.view(B, T, self.top_k)

            # Compute auxiliary loss
            aux_loss = torch.tensor(0.0).to(x.device)
            if self.use_aux_loss and self.training:
                aux_loss = self._compute_aux_loss(prob_scores)

            return expert_weights, expert_indices, aux_loss

    def _compute_aux_loss(self, prob_scores: torch.Tensor) -> torch.Tensor:
        """Compute the auxiliary loss if applicable.

        Args:
            prob_scores (torch.Tensor): Probability of each expert being chosen over all tokens.

        Returns:
            cv (torch.Tensor): Coefficient of variation.
        """
        experts = prob_scores.sum(dim=0) # Sum over total tokens
        experts_fractions = experts / experts.sum()

        # Compute coefficient of variation
        cv = experts_fractions.std(unbiased=False) / experts_fractions.mean()

        return cv


class MoELayer(nn.Module):
    """Mixture of Experts layer.

    Args:
        d_model (int): Dimensionality of the model's embeddings.
        d_ffn (int): Dimensionality of the feed forward network.
        dropout (float): Probability of model components being dropped out.
        num_experts (float): Number of feed forward networks.
        top_k (float): Number of experts each token is routed to.
        eps (float): Small value to maintain numerical stability in RMSNorm.
    """
    def __init__(
            self,
            d_model: int,
            d_ffn: int,
            dropout: float,
            num_experts: int,
            top_k: int,
            eps: float,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_ffn = d_ffn
        self.num_experts = num_experts
        self.top_k = top_k

        # Router
        self.router = TopKRouter(
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k,
            use_aux_loss=True
        )

        # Experts
        self.experts = nn.ModuleList([
            SwiGLUExpert(d_model, d_ffn, dropout).to(device) for _ in range(num_experts)
        ])

        # Set up RMSNorm
        self.rms_norm = RMSNorm(d_model, eps)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the Mixture of Experts layer.

        Args
            x (torch.Tensor): Input tensor of shape [B, T, d_model].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]. A tuple containing:
                - out: Ouput tensor of shape [B, T, d_model].
                - aux_loss: Auxiliary loss.
        """
        B, T, d_model = x.shape

        # Apply RMSNorm
        x = self.rms_norm(x)

        # Get routers
        expert_weights, expert_indices, aux_loss = self.router(x)

        # Flatten for efficiency
        x_flattened = x.view(-1, d_model) # [B * T, d_model]

        # Initialize output
        out = torch.zeros_like(x_flattened)

        # Process all experts
        for expert_id in range(self.num_experts):
            expert_mask = (expert_indices == expert_id) # [B, T, top_k]

            if expert_mask.any():
                # Get positions where this expert is used
                expert_positions = expert_mask.nonzero(as_tuple=False) # [num_matches, 3] [B, T, expert_id]

                if expert_positions.numel() > 0:
                    # Get the corresponding tokens
                    batch_indices = expert_positions[:, 0]
                    seq_indices = expert_positions[:, 1]
                    topk_indices = expert_positions[:, 2]

                    # Convert to flattened indices
                    flat_indices = batch_indices * T + seq_indices

                    # Get input tokens for this expert
                    expert_input = x_flattened[flat_indices] # [num_matches, d_model]

                    # Get weights for this expert
                    expert_weight_vals = expert_weights[batch_indices, seq_indices, topk_indices] # [num_matches]

                    # Forward through expert
                    expert_output = self.experts[expert_id](expert_input) # [num_matches, d_model]

                    # Apply weights and accumulate
                    weighted_output = expert_output * expert_weight_vals.unsqueeze(-1) # [num_matches, d_model]

                    # Add to output
                    out[flat_indices] += weighted_output

        # Reshape back
        out = out.view(B, T, d_model)

        return out, aux_loss


class AttentionBlock(nn.Module):
    """Attention block where RMSNorm, Dropout, and residuals are applied.

    Args:
        d_model (int): Dimensionality of the model's embeddings.
        num_heads (int): Number of attention heads for KV vectors, shared across grouped query heads in GQA.
        query_groups (int): Number of query groups that partition the queries into subsets.
        dropout (float): Probability that model components will be randomly dropped out.
        theta (float): Exponential base of the inverse frequency.
        eps (float): Small epsilon value to ensure numerical stability.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        query_groups: int,
        dropout: float,
        theta: float,
        eps: float,
    ):
        super().__init__()

        self.rms_norm = RMSNorm(d_model, eps)
        self.attn = Attention(d_model, num_heads, query_groups, theta)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        window_size: Tuple[int, int],
        padding_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        layer_idx: Optional[int] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Forward pass of the Attention Block.

        Args:
            x (torch.Tensor): Input tensor of shape [B, T, d_model].
            window_size (Tuple[int, int]): Window size for SWA.
            padding_mask (Optional[torch.Tensor]): Padding tensor of shape [B, T].
            kv_cache (Optional[KVCache]): Key-value cache for efficient generation.
            layer_idx (Optional[int]): Index of the current layer for cache access.
            use_cache (bool): Whether to use KV caching during forward pass.

        Returns:
            Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
                - torch.Tensor: Output tensor of shape [B, T, d_model].
                - Dict[str, torch.Tensor]: Cache dictionary with 'k' and 'v' tensors.
        """
        with torch.amp.autocast(device_type=device.type, dtype=dtype):
            attn_out, cache_out = self.attn(
                self.rms_norm(x),
                window_size,
                padding_mask=padding_mask,
                kv_cache=kv_cache,
                layer_idx=layer_idx,
                use_cache=use_cache
            )
            return x + self.dropout(attn_out), cache_out


class MoEBlock(nn.Module):
    """Mixture of Experts block where RMSNorm, Dropout, and residuals are applied.

    Args:
        d_model (int): Dimensionality of the model's embeddings.
        d_ffn (int): Dimensionality of the feed forward network.
        dropout (float): Probability that model components will be randomly dropped out.
        num_experts (int): Number of feed forward networks in the MoE layer.
        top_k (int): Number of experts each token is routed to.
        eps (float): Small epsilon value to ensure numerical stability.
    """
    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        dropout: float,
        num_experts: int,
        top_k: int,
        eps: float,
    ):
        super().__init__()

        self.rms_norm = RMSNorm(d_model, eps)
        self.moe = MoELayer(d_model, d_ffn, dropout, num_experts, top_k, eps)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the MoE Block.

        Args:
            x (torch.Tensor): Input tensor of shape [B, T, d_model].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Output tensor passed through the MoE Block with the same shape.
                - Auxiliary loss from the MoE layer.
        """
        with torch.amp.autocast(device_type=device.type, dtype=dtype):
            moe_out, aux_loss = self.moe(self.rms_norm(x))
            return x + self.dropout(moe_out), aux_loss


class TransformerBlock(nn.Module):
    """Transformer block that will be stacked in the final Transformer class.

    Args:
        d_model (int): Dimensionality of the model's embeddings.
        num_heads (int): Number of attention heads for KV vectors, shared across grouped query heads in GQA.
        query_groups (int): Number of query groups that partition the queries into subsets.
        d_ffn (int): Dimensionality of the feed forward network.
        dropout (float): Probability that model components will be randomly dropped out.
        num_experts (int): Number of feed forward networks in the MoE layer.
        top_k (int): Number of experts each token is routed to.
        theta (float): Exponential base of the inverse frequency.
        eps (float): Small epsilon value to ensure numerical stability.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        query_groups: int,
        d_ffn: int,
        dropout: float,
        num_experts: int,
        top_k: int,
        theta: float,
        eps: float,
    ):
        super().__init__()

        self.attn_block = AttentionBlock(d_model, num_heads, query_groups, dropout, theta, eps)
        self.moe_block = MoEBlock(d_model, d_ffn, dropout, num_experts, top_k, eps)

    def forward(
        self,
        x: torch.Tensor,
        window_size: Tuple[int, int],
        padding_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        layer_idx: Optional[int] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        """Forward pass of the Transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape [B, T, d_model].
            window_size (Tuple[int, int]): Window size for SWA.
            padding_mask (Optional[torch.Tensor]): Padding tensor of shape [B, T].
            kv_cache (Optional[KVCache]): Key-value cache for efficient generation.
            layer_idx (Optional[int]): Index of the current layer for cache access.
            use_cache (bool): Whether to use KV caching during forward pass.

        Returns:
            Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]], torch.Tensor]:
                - Output tensor of shape [B, T, d_model].
                - Cache dictionary with 'k' and 'v' tensors if use_cache is True, else None.
                - Auxiliary loss from the MoE layer.
        """
        with torch.amp.autocast(device_type=device.type, dtype=dtype):
            x, cache_out = self.attn_block(
                x,
                window_size,
                padding_mask=padding_mask,
                kv_cache=kv_cache,
                layer_idx=layer_idx,
                use_cache=use_cache
            )
            x, aux_loss = self.moe_block(x)
            return x, cache_out, aux_loss


class Transformer(nn.Module):
    """Complete Transformer class stacking all decoder blocks.

    Args:
        model_args (ModelArgs): Dataclass containing all model arguments.
    """
    def __init__(self, model_args: ModelArgs):
        super().__init__()

        self.model_args = model_args

        self.token_embed = nn.Embedding(model_args.vocab_size, model_args.d_model).to(device)
        self.dropout = nn.Dropout(p=model_args.dropout).to(device)

        # Stack transformer blocks with MoE
        self.layers = nn.ModuleList([
            TransformerBlock(
                model_args.d_model,
                model_args.num_heads,
                model_args.query_groups,
                model_args.d_ffn,
                model_args.dropout,
                model_args.num_experts,
                model_args.top_k,
                model_args.rope_base,
                model_args.rms_norm_eps,
            ).to(device) for _ in range(model_args.num_layers)
        ])

        self.rms_norm = RMSNorm(model_args.d_model, model_args.rms_norm_eps).to(device)

        # Initialize KV Cache
        self.kv_cache = KVCache(
            max_batch_size=model_args.max_batch_size,
            max_seq_len=model_args.max_seq_len,
            num_heads=model_args.query_groups,
            head_dim=model_args.d_model // model_args.num_heads,
            num_layers=model_args.num_layers,
        )

        # Language modeling head, bias=False for weight tying
        self.lm_head = nn.Linear(model_args.d_model, model_args.vocab_size, bias=False).to(device)

        self.apply(self._init_weights)

        # Tie weights if flag is True after initialization
        if model_args.tie_weights:
            self.lm_head.weight = self.token_embed.weight

    def _init_weights(self, module) -> None:
        """Initializes model weights according to layer type.
        
        Args:
            module: PyTorch module to be initialized.
        """
        num_layers = self.model_args.num_layers
        init_std = 0.02

        # Initialize Embeddings
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=init_std)

        # Initialize linear layers
        elif isinstance(module, nn.Linear):
            # Identify linear layers
            is_qkv = hasattr(module, '_is_qkv') or any(hasattr(p, 'w_qkv') and p.w_qkv is module for p in self.modules())
            is_attn_out = hasattr(module, '_is_attn_out') or any(hasattr(p, 'w_o') and p.w_o is module for p in self.modules())
            is_ffn_gate = hasattr(module, '_is_ffn_gate') or any(hasattr(p, 'weight1') and p.weight1 is module for p in self.modules())
            is_ffn_up = hasattr(module, '_is_ffn_up') or any(hasattr(p, 'weight3') and p.weight3 is module for p in self.modules())
            is_ffn_down = hasattr(module, '_is_ffn_down') or any(hasattr(p, 'weight2') and p.weight2 is module for p in self.modules())
            is_lm_head = hasattr(self, 'lm_head') and self.lm_head is module
            is_router = any(hasattr(p, 'router') and hasattr(p.router, 'weight') and p.router.weight is module for p in self.modules())

            # Initialize input projections
            if is_qkv or is_ffn_gate or is_ffn_up or is_router:
                nn.init.xavier_uniform_(module.weight)
                if num_layers > 12:
                    module.weight.data *= (1.0 / math.sqrt(num_layers / 6.0))

            # Initialize output projections
            elif is_attn_out or is_ffn_down:
                scaled_std = init_std / math.sqrt(2 * num_layers)
                nn.init.normal_(module.weight, mean=0.0, std=scaled_std)

            # Initialize language modeling head
            elif is_lm_head:
                nn.init.normal_(module.weight, mean=0.0, std=init_std)

            # Default to Xavier initialization
            else:
                nn.init.xavier_uniform_(module.weight)

            # Initialize bias
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

        # Initialize RMSNorm weight (scaling factor)
        elif hasattr(module, 'weight') and module.__class__.__name__ == 'RMSNorm':
            nn.init.constant_(module.weight, 1.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[Dict[str, torch.Tensor]]], torch.Tensor]:
        """Forward pass of the entire Transformer.

        Args:
            input_ids (torch.Tensor): Input tensor of shape [B, T].
            window_size (Tuple[int, int]): Window size for SWA.
            padding_mask (Optional[torch.Tensor]): Padding tensor of shape [B, T].
            use_cache (bool): Whether to use KV caching during forward pass.

        Returns:
            Tuple[torch.Tensor, Optional[List[Dict[str, torch.Tensor]]], torch.Tensor]:
                - torch.Tensor: Output logits of shape [B, T, vocab_size].
                - Optional[List[Dict[str, torch.Tensor]]]: List of cache dictionaries for each layer.
                - Sum of auxiliary losses from all MoE layers.
        """
        with torch.amp.autocast(device_type=device.type, dtype=dtype):
            # Ensure input_ids.dtype == torch.int64 for embeddings
            input_ids = input_ids.to(torch.int64)

            # Ensure padding mask/input_ids are the same shape
            if padding_mask is not None:
                assert (
                    input_ids.shape == padding_mask.shape
                ), "input_ids and padding_mask must have the same shape for correct masking."

            # input_ids dimensions/dtype assertions
            assert (
                input_ids.dim() == 2
            ), f"input_ids must be a 2 dimensional tensor, got {input_ids.dim()}"
            assert (
                input_ids.dtype == torch.int64
            ), f"input_ids dtype must be int64, got {input_ids.dtype}"

            # Apply embeddings
            x = self.token_embed(input_ids) # [B, T, d_model]

            assert (
                x.dim() == 3
            ), f"x must be a 3 dimensional tensor, got {x.dim()}"

            # Final dropout
            x = self.dropout(x)

            # Initialize KV cache outputs as list
            cache_outs = [] if use_cache else None

            # Initialize aux loss as float32 tensor
            total_aux_loss = torch.tensor(0.0, dtype=torch.float32).to(device)

            assert (
                total_aux_loss.dtype == torch.float32
            ), f"total_aux_loss dtype must be float32, got {total_aux_loss.dtype}"

            # TODO: look into chunked checkpointing for larger memory savings.
            # Stack transformer layers
            for i, layer in enumerate(self.layers):
                if self.model_args.gradient_checkpointing:
                    x, cache_out, aux_loss = checkpoint(
                        layer,
                        x,
                        self.model_args.window_size,
                        padding_mask,
                        self.kv_cache,
                        i,
                        use_cache,
                        use_reentrant=False
                    )
                else:
                    x, cache_out, aux_loss = layer(
                        x,
                        window_size=self.model_args.window_size,
                        padding_mask=padding_mask,
                        kv_cache=self.kv_cache,
                        layer_idx=i,
                        use_cache=use_cache
                    )
                # Accumulate auxiliary loss
                if aux_loss is not None:
                    total_aux_loss += aux_loss
                
                # Apped to cache
                if use_cache:
                    cache_outs.append(cache_out)

            # Final RMSNorm
            x = self.rms_norm(x)

            # Final projection to logits
            logits = self.lm_head(x) # [B, T, V]
            assert (
                logits.dim() == 3
            ), f"logits must have 3 dimenions, got {logits.dim()}"

            return logits, cache_outs, total_aux_loss
