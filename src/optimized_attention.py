from configs.setup_env import (
    device, 
    dtype,
    gpu_dtypes,
    use_flash_attn,
    flash_attn_varlen_qkvpacked_func
)

import warnings
from typing import Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rms_norm import RMSNorm

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
        softmax_scale (float): Float to scale the attention scores by.
        use_proj_bias (bool): Whether to use bias in q, k, v, o projections.
        use_qkv_proj (bool): Whether to use fused qkv proj or individual q, k, v projections.
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

        self.d_model = d_model
        self.num_heads = num_heads
        self.query_groups = query_groups
        self.head_dim = d_model // num_heads
        self.heads_per_group = num_heads // query_groups
        self.softmax_scale = softmax_scale
        self.use_qkv_proj = use_qkv_proj

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
                )
        if num_heads % query_groups != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by query_groups ({query_groups})"
                )

        # QKV projection matrix
        if use_qkv_proj:
            self.w_qkv = nn.Linear(
                d_model,
                num_heads * self.head_dim + 2 * query_groups * self.head_dim,
                bias=use_proj_bias,
                dtype=dtype
            ).to(device)
        else:
            self.w_q = nn.Linear(d_model, num_heads * self.head_dim, dtype=dtype, bias=use_proj_bias)
            self.w_k = nn.Linear(d_model, self.query_groups * self.head_dim, dtype=dtype, bias=use_proj_bias)
            self.w_v = nn.Linear(d_model, self.query_groups * self.head_dim, dtype=dtype, bias=use_proj_bias)

        # O projection matrix
        self.w_o = nn.Linear(
            d_model,
            d_model,
            bias=use_proj_bias,
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
            use_mqa (bool): Whether to Multi-Query attention instead of GQA.
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
        left_window: int,
        right_window: int,
        causal: bool = True,
        padding_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        layer_idx: Optional[int] = None,
        use_cache: bool = False,
        use_mqa: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Forward pass for GQA layer.

        Args:
            x (torch.Tensor): Input tensor of shape [B, T, D].
            left_window (int): Left window for SWA.
            right_window (int): Right window for SWA.
            causal (bool): To apply causal masking or not.
            padding_mask (torch.Tensor, optional): Padding mask of shape [B, T] where
                True indicates valid tokens and False indicates padding tokens.
            kv_cache (Optional[KVCache]): Key-value cache for efficient generation.
            layer_idx (Optional[int]): Index of the current layer for cache access.
            use_cache (bool): Whether to use KV caching during forward pass.
            use_mqa (bool): Whether to use multi-query attention or not.
                Constraints: query_groups == 1.

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
            q, k, v tensors must be on device of cuda.

        NOTE: FLASH ATTENTION 2 + SWA HAS NOT BEEN TESTED DUE TO HARDWARE LIMITATIONS.
        """
        with torch.amp.autocast(device_type=device.type, dtype=dtype):
            B, T, _ = x.shape
            if x.shape != (B, T, self.d_model):
                raise ValueError(f"Expected x to have shape [B, T, d_model], got {x.shape}")
            if T == 0:
                # Return empty tensor if sequence length is 0
                return torch.empty(B, 0, self.d_model, device=x.device, dtype=x.dtype), None

            if self.use_qkv_proj:
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
            else:
                warnings.warn(
                    "Using seperate projections, set use_qkv_proj=True to use single qkv projection."
                )
                q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)

                assert (
                    q.shape == (B, T, self.d_model)
                ), f"q must have shape of {(B, T, self.d_model)}, got {q.shape}"
                assert (
                    k.shape == (B, T, self.query_groups * self.head_dim)
                ), f"k must have shape of {((B, T, self.query_groups * self.head_dim))}, got {k.shape}"
                assert (
                    v.shape == (B, T, self.query_groups * self.head_dim)
                ), f"v must have shape of {(B, T, self.query_groups * self.head_dim)}, got {v.shape}"

            # Reshape into 4D tensors for GQA
            q = q.view(B, T, self.num_heads, self.head_dim) # [B, T, num_heads, head_dim]
            k = k.view(B, T, self.query_groups, self.head_dim) # [B, T, query_groups, head_dim]
            v = v.view(B, T, self.query_groups, self.head_dim) # [B, T, query_groups, head_dim]

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
            
            # Extend KV tensors (if query_groups==1 and use_mqa=True, no expansion is done)
            k = self._extend_kv_heads(
                kv_tensor=k,
                heads_per_group=self.heads_per_group, 
                kv_heads_dim=2, 
                use_mqa=use_mqa
            )
            v = self._extend_kv_heads(
                kv_tensor=v, 
                heads_per_group=self.heads_per_group, 
                kv_heads_dim=2, 
                use_mqa=use_mqa
            )

            # Assert extension of kv heads worked or MQA was succesful
            assert (
                k.size(2) == self.num_heads or k.size(2) == 1
            ), f"k.size(2) must be {self.num_heads} or 1, got {k.size(2)}"
            assert (
                v.size(2) == self.num_heads or k.size(2) == 1
            ), f"k.size(2) must be {self.num_heads} or 1, got {k.size(2)}"

            # for causal LM, we want right window to be 0
            if causal:
                right_window = 0

            # FlashAttention 2 - requires CUDA/flash attn available/bf16 or fp16
            if (
                use_flash_attn  and device.type == "cuda"
                and q.dtype in gpu_dtypes
                and k.dtype in gpu_dtypes
                and v.dtype in gpu_dtypes
                and q.is_cuda and k.is_cuda and v.is_cuda
            ):
                qkv_packed = torch.stack([q, k, v], dim=3).contiguous() # [B, T, num_heads, 3, head_dim]
                assert (
                    qkv_packed.shape == (B, T, self.num_heads, 3, self.head_dim)
                ), f"qkv_packed must have shape of {(B, T, self.num_heads, 3, self.head_dim)}, got {qkv_packed.shape}"
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
                    ), f"qkv_packed must have shape {(B * T, 3, self.num_heads, self.head_dim)}, got {qkv_packed.shape}"
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
                    softmax_scale=self.softmax_scale,
                    window_size=(left_window, right_window),
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
                    out = full_out.view(B, T, self.d_model) # Reshape output to [B, T, d_model]
                else:
                    # If no padding, reshape right away
                    out = out.view(B, T, self.d_model)

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
                else:
                    attn_mask = None

                # Call PyTorch's scaled dot product attention
                out = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    is_causal=causal if padding_mask is None else False,
                    scale=self.softmax_scale
                ) # [B, num_heads, T, head_dim]
                assert (
                    out.shape == (B, self.num_heads, T, self.head_dim)
                ), f"out must have shape {(B, self.num_heads, T, self.head_dim)}, got {out.shape}"

                # [B, num_heads, T, head_dim] -> [B, T, d_model]
                out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
                assert (
                    out.shape == (B, T, self.d_model)
                ), f"out must have shape {(B, T, self.d_model)}, got {out.shape}"

            # Get cache output for the Attention layer
            cache_out = {'k': k[:, -T:], 'v': v[:, -T:]} if use_cache else None

            # Final projection
            return self.w_o(out), cache_out


class AttentionBlock(nn.Module):
    """Attention block where RMSNorm, Dropout, and residuals are applied.

    Args:
        d_model (int): Dimensionality of the model's embeddings.
        num_heads (int): Number of attention heads for KV vectors, shared across grouped query heads in GQA.
        query_groups (int): Number of query groups that partition the queries into subsets.
        softmax_scale (float): Float to scale the attention scores by.
        use_proj_bias (bool): Whether to use bias in q, k, v, o projections.
        use_qkv_proj (bool): Whether to use fused qkv proj or individual q, k, v projections.
        dropout (float): Probability that model components will be randomly dropped out.
        theta (float): Exponential base of the inverse frequency.
        eps (float): Small epsilon value to ensure numerical stability.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        query_groups: int,
        softmax_scale: float,
        use_proj_bias: bool,
        use_qkv_proj: bool,
        dropout: float,
        theta: float,
        eps: float,
    ):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.rms_norm = RMSNorm(d_model, eps)
        self.attn = Attention(
            d_model=d_model, 
            num_heads=num_heads, 
            query_groups=query_groups, 
            theta=theta, 
            softmax_scale=softmax_scale, 
            use_proj_bias=use_proj_bias, 
            use_qkv_proj=use_qkv_proj
        )

    def forward(
        self,
        x: torch.Tensor,
        left_window: int,
        right_window: int,
        causal: bool = True,
        padding_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        layer_idx: Optional[int] = None,
        use_cache: bool = False,
        use_mqa: bool = False
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
                x=self.rms_norm(x),
                left_window=left_window,
                right_window=right_window,
                causal=causal,
                padding_mask=padding_mask,
                kv_cache=kv_cache,
                layer_idx=layer_idx,
                use_cache=use_cache,
                use_mqa=use_mqa
            )
            return x + self.dropout(attn_out), cache_out
        