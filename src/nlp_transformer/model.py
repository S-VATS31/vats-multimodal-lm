from configs.setup_env import (
    device,
    dtype, 
    use_flash_attn, 
    flash_attn_varlen_qkvpacked_func,
    logger,
    )

from configs.model_args.model_args_medium import ModelArgs

import math
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

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
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim)).to(device)
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
            cos_freqs = self.cos_cache[:seq_len] # [T, head_dim//2]
            sin_freqs = self.sin_cache[:seq_len] # [T, head_dim//2]

            # Apply rotary embeddings
            return self._apply_rope(x, cos_freqs, sin_freqs)

    def _update_cache(self, seq_len: int):
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
            # Split x into even and odd dimensions
            x1 = x[..., ::2]
            x2 = x[..., 1::2]

            # Expand frequency tensors to match input dimensions
            cos_freqs = cos_freqs.unsqueeze(0).unsqueeze(2) # [1, T, 1, head_dim//2]
            sin_freqs = sin_freqs.unsqueeze(0).unsqueeze(2) # [1, T, 1, head_dim//2]

            # Complex rotation via rotation matrix
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
    def __init__(self, d_model: int, eps: float = 1e-7):
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
            rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
            return self.weight * (x / rms)

class KVCache:
    """Key-Value cache for efficient autoregressive generation.

    Args:
        max_batch_size (int): Maximum batch size supported by the cache.
        max_seq_len (int): Maximum sequence length supported by the cache.
        num_heads (int): Number of attention heads (or query groups for GQA).
        head_dim (int): Dimension of each attention head.
        num_layers (int): Number of transformer layers.
        dtype (torch.dtype): Data type for cache tensors. Defaults to global dtype.
        device (torch.device): Device for cache tensors. Defaults to global device.
    """
    def __init__(
        self,
        max_batch_size: int,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        num_layers: int,
        dtype: torch.dtype = dtype,
        device: torch.device = device
    ):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.dtype = dtype
        self.device = device

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
                    ), dtype=self.dtype, device=self.device),
                'v': torch.zeros((
                    batch_size, self.max_seq_len, self.num_heads, self.head_dim
                    ), dtype=self.dtype, device=self.device)
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
    def __init__(self, d_model: int, num_heads: int, query_groups: int, theta: float):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.query_groups = query_groups
        self.head_dim = d_model // num_heads

        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        if num_heads % query_groups != 0:
            raise ValueError(f"num_heads ({num_heads}) must be divisible by query_groups ({query_groups})")

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

    def forward(
        self,
        x: torch.Tensor,
        causal: bool = True,
        padding_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        layer_idx: Optional[int] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Forward pass for GQA layer.

        Args:
            x (torch.Tensor): Input tensor of shape [B, T, D].
            causal (bool): To apply causal masking or not.
            padding_mask (torch.Tensor, optional): Padding mask of shape [B, T] where
                True indicates valid tokens and False indicates padding tokens.
            kv_cache (KVCache, optional): Key-value cache for efficient generation.
            layer_idx (int, optional): Index of the current layer for cache access.
            use_cache (bool): Whether to use KV caching during forward pass.

        Returns:
            Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
                - torch.Tensor: Attention output tensor of shape [B, T, d_model].
                - Optional[Dict[str, torch.Tensor]]: Cache dictionary with 'k' and 'v' tensors.

        Raises:
            ValueError if `x` does not have 3 shape [B, T, d_model].
            ValueError if `padding_mask` does not have shape [B, T].
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

            # q shape: [B, T, num_heads * head_dim]
            # kv shape: [B, T, 2 * query_groups * head_dim]
            q, kv = torch.split(qkv, [self.num_heads * self.head_dim, 2 * self.query_groups * self.head_dim], dim=-1)

            # k shape: [B, T, query_groups * head_dim]
            # v shape: [B, T, query_groups * head_dim]
            k, v = torch.chunk(kv, 2, dim=-1)

            # Reshape into 4D tensors for GQA
            q = q.view(B, T, self.num_heads, self.head_dim) # [B, T, num_heads, head_dim]
            k = k.view(B, T, self.query_groups, self.head_dim) # [B, T, query_groups, head_dim]
            v = v.view(B, T, self.query_groups, self.head_dim) # [B, T, query_groups, head_dim]

            # Apply RoPE
            q = self.rope(q) # [B, T, num_heads, head_dim]
            k = self.rope(k) # [B, T, query_groups, head_dim]

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

            # Expand KV for GQA
            if self.query_groups != self.num_heads:
                heads_per_group = self.num_heads // self.query_groups
                k = k.repeat_interleave(heads_per_group, dim=2) # [B, T, num_heads, head_dim]
                v = v.repeat_interleave(heads_per_group, dim=2) # [B, T, num_heads, head_dim]

            # FlashAttention 2 - requires CUDA/flash attn 2 available
            if use_flash_attn and device.type == "cuda":
                qkv_packed = torch.stack([q, k, v], dim=3) # [B, T, num_heads, 3, head_dim]
                qkv_packed = qkv_packed.contiguous()

                # Handle padding mask for FlashAttention
                if padding_mask is not None:
                    if padding_mask.shape != (B, T):
                        raise ValueError(
                            f"Expected padding mask shape [B, T], got {padding_mask.shape}"
                            )
                    # Ensure padding mask is a boolean tensor
                    padding_mask = padding_mask.bool()
                    seq_lens = padding_mask.sum(dim=1).int() # [B]

                    # Compute cumulative sequence lengths
                    cu_seqlens = torch.cat([
                        torch.tensor([0],dtype=torch.int32).to(device), # [0]
                        seq_lens.cumsum(0) # [B]
                    ], dim=0) # Shape: [B + 1]

                    # This is an expected parameter for the FlashAttention function
                    max_seqlen = seq_lens.max().item()

                    # Flatten padding mask
                    valid_tokens = padding_mask.flatten() # [B * T]

                    # Index by valid tokens to only get non-padding token positions
                    qkv_packed = qkv_packed.view(-1, self.num_heads, 3, self.head_dim)[valid_tokens]
                else:
                    # Get cumulative sequence lengths
                    cu_seqlens = torch.arange(
                        0,
                        (B + 1) * k.size(1),
                        k.size(1),
                        dtype=torch.int32,
                    ).to(device) # [B + 1]
                    max_seqlen = k.size(1)

                    # Total tokens if calculated as B * T
                    total_tokens = B * max_seqlen

                    qkv_packed = qkv_packed.view(total_tokens, self.num_heads, 3, self.head_dim)

                # Call FlashAttention 2
                out = flash_attn_varlen_qkvpacked_func(
                    qkv_packed,
                    cu_seqlens,
                    max_seqlen,
                    causal=causal,
                    softmax_scale=1.0 / (self.head_dim ** 0.5)
                ) # [B * T, num_heads, head_dim]

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
                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)

                # Initialize padding mask
                attn_mask = None

                # Apply padding mask for PyTorch SDPA
                if padding_mask is not None:
                    if padding_mask.shape != (B, T):
                        raise ValueError(
                            f"Expected padding mask shape [B, T], got {padding_mask.shape}"
                            )
                    padding_mask = padding_mask.bool() # Ensure padding mask is a boolean tensor
                    attn_mask = padding_mask.unsqueeze(1).unsqueeze(2) # [B, 1, 1, T]
                    attn_mask = attn_mask.expand(B, 1, T, k.size(2)) # [B, 1, T_q, T_k]

                    # Apply causal masking
                    if causal:
                        # Causal mask shape: [T_q, T_k] where the upper right diagonal portion is False.
                        causal_mask = torch.tril(torch.ones(T, k.size(2), dtype=torch.bool).to(device))
                        # Since the attention scores tensor has shape [B, num_heads, T_q, T_k]
                        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0) # [1, 1, T_q, T_k]
                        attn_mask = attn_mask & causal_mask # [B, 1, T_q, T_k]

                    # Expand to [B, num_heads, T_q, T_k] only now to avoid unnecessary memory usage.
                    attn_mask = attn_mask.expand(B, self.num_heads, T, k.size(2))

                # Call PyTorch's scaled dot product attention
                out = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    is_causal=causal if padding_mask is None else False
                ) # [B, num_heads, T, head_dim]

                # [B, num_heads, T, head_dim] -> [B, T, num_heads, head_dim]
                out = out.transpose(1, 2).contiguous().view(B, T, D)

            # Get cache output for the Attention layer
            cache_out = {'k': k[:, -T:], 'v': v[:, -T:]} if use_cache else None

            # Final projection
            return self.w_o(out), cache_out

class SwiGLUExpert(nn.Module):
    """SwiGLU expert layer.

    Args:
        d_model (int): Dimensionality of the model's embeddings.
        d_ffn (int): Dimensionality of the feed forward network.
        dropout (float): Probability of model components being dropped out.
    """
    def __init__(self, d_model: int, d_ffn: int, dropout: float):
        super().__init__()

        self.weight1 = nn.Linear(d_model, d_ffn)
        self.weight2 = nn.Linear(d_ffn, d_model)
        self.weight3 = nn.Linear(d_model, d_ffn)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SwiGLU Expert layer.

        Formula:
            SwiGLU(x) = Dropout(((W1 @ x) * Swish(W3 @ x)) @ W2)

        Args:
            x (torch.Tensor): Input tensor of shape [B, T, d_model].

        Returns:
            torch.Tensor: Output tensor passed through the expert layer with same shape.
        """
        with torch.amp.autocast(device_type=device.type, dtype=dtype):
            return self.dropout(self.weight2(F.silu(self.weight1(x)) * self.weight3(x)))

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
                aux_loss = self.compute_aux_loss(prob_scores)

            return expert_weights, expert_indices, aux_loss

    def compute_aux_loss(self, prob_scores: torch.Tensor) -> torch.Tensor:
        """Compute the auxiliary loss if applicable.

        Args:
            prob_scores (torch.Tensor): Probability of each expert being chosen over all tokens.

        Returns:
            cv (torch.Tensor): Coefficient of variation.
        """
        experts = prob_scores.sum(dim=0) # Sum over total tokens
        experts_fractions = experts / experts.sum()

        # Compute coefficient of variation
        cv = torch.std(experts_fractions) / torch.mean(experts_fractions)

        return cv

class MoELayer(nn.Module):
    """Mixture of Experts layer.

    Args:
        d_model (int): Dimensionality of the model's embeddings.
        d_ffn (int): Dimensionality of the feed forward network.
        dropout (float): Probability of model components being dropped out.
        num_experts (float): Number of feed forward networks.
        top_k (float): Number of experts each token is routed to.
    """
    def __init__(
            self,
            d_model: int,
            d_ffn: int,
            dropout: float,
            num_experts: int,
            top_k: int,
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
        self.rms_norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the Mixture of Experts layer.

        Args
            x (torch.Tensor): Input tensor of shape [B, T, d_model].
            padding_mask (torch.Tensor): Padding mask tensor of shape [B, T]. Not used here.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]. A tensor containing
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
        padding_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        layer_idx: Optional[int] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Forward pass of the Attention Block.

        Args:
            x (torch.Tensor): Input tensor of shape [B, T, d_model].
            padding_mask (torch.Tensor): Padding tensor of shape [B, T].
            kv_cache (KVCache, optional): Key-value cache for efficient generation.
            layer_idx (int, optional): Index of the current layer for cache access.
            use_cache (bool): Whether to use KV caching during forward pass.

        Returns:
            Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
                - torch.Tensor: Output tensor of shape [B, T, d_model].
                - Dict[str, torch.Tensor]: Cache dictionary with 'k' and 'v' tensors.
        """
        with torch.amp.autocast(device_type=device.type, dtype=dtype):
            attn_out, cache_out = self.attn(
                self.rms_norm(x),
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
        self.moe = MoELayer(d_model, d_ffn, dropout, num_experts, top_k)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the MoE Block.

        Args:
            x (torch.Tensor): Input tensor of shape [B, T, d_model].
            padding_mask (torch.Tensor): Padding tensor of shape [B, T].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Output tensor passed through the MoE Block with the same shape.
                - Auxiliary loss from the MoE layer.
        """
        with torch.amp.autocast(device_type=device.type, dtype=dtype):
            moe_out, aux_loss = self.moe(self.rms_norm(x), padding_mask)
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
        padding_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        layer_idx: Optional[int] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        """Forward pass of the Transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape [B, T, d_model].
            padding_mask (torch.Tensor): Padding tensor of shape [B, T].
            kv_cache (KVCache, optional): Key-value cache for efficient generation.
            layer_idx (int, optional): Index of the current layer for cache access.
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
                padding_mask=padding_mask,
                kv_cache=kv_cache,
                layer_idx=layer_idx,
                use_cache=use_cache
            )
            x, aux_loss = self.moe_block(x, padding_mask=padding_mask)
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

        # Language modeling head, bias=False for weight tying
        self.lm_head = nn.Linear(model_args.d_model, model_args.vocab_size, bias=False).to(device)

        # Initialize KV cache
        self.kv_cache = KVCache(
            max_batch_size=model_args.max_batch_size,
            max_seq_len=model_args.max_seq_len,
            num_heads=model_args.query_groups,
            head_dim=model_args.d_model // model_args.num_heads,
            num_layers=model_args.num_layers,
            dtype=dtype,
            device=device
        )

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
            padding_mask (torch.Tensor): Padding tensor of shape [B, T].
            use_cache (bool): Whether to use KV caching during forward pass.

        Returns:
            Tuple[torch.Tensor, Optional[List[Dict[str, torch.Tensor]]], torch.Tensor]:
                - torch.Tensor: Output logits of shape [B, T, vocab_size].
                - Optional[List[Dict[str, torch.Tensor]]]: List of cache dictionaries for each layer.
                - Sum of auxiliary losses from all MoE layers.
        """
        with torch.amp.autocast(device_type=device.type, dtype=dtype):
            # Ensure input_ids is a LongTensor (int64)
            if input_ids.dtype != torch.int64:
                input_ids = input_ids.long()

            # Apply embeddings
            x = self.token_embed(input_ids) # [B, T, d_model]

            # Final dropout
            x = self.dropout(x)

            # Initialize KV cache outputs as list
            cache_outs = [] if use_cache else None

            # Initialize aux loss as float32 tensor
            total_aux_loss = torch.tensor(0.0, dtype=torch.float32).to(device)

            # Stack transformer layers
            for i, layer in enumerate(self.layers):
                if self.model_args.gradient_checkpointing:
                    x, cache_out, aux_loss = checkpoint(
                        layer,
                        x,
                        padding_mask,
                        self.kv_cache,
                        i,
                        use_cache,
                        use_reentrant=False
                    )
                else:
                    x, cache_out, aux_loss = layer(
                        x,
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
            logits = self.lm_head(x)

            return logits, cache_outs, total_aux_loss

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Generate tokens autoregressively using decoding methods.

        Args:
            input_ids (torch.Tensor): int64 tensor containing tokens.
            max_new_tokens (int): Maximum number of tokens the model can generate at a time.
            temperature (float): Decoding method to encourage more randomness/determinism based on value.
            top_k (int): Top-k logits to be sampled.
            top_p (float): Top-p hyperparameter used as a threshold for masking out certain logits.
            do_sample (bool): Whether to apply sampling or greedy decoding.
            pad_token_id (Optional[int]): Special value of the padding token to be masked out.
            eos_token_id (Optional[int]): End of sequence token appended to the end of each token.
            attention_mask (Optional[torch.Tensor]): Padding mask of shape [B, T].
            use_cache (bool): Boolean to whether use the KV cache or not.

        Returns:
            torch.Tensor: Returns a tensor of generated tokens of shape [B, T].
        """
        if pad_token_id is None:
            pad_token_id = self.model_args.pad_token_id
        if eos_token_id is None:
            eos_token_id = self.model_args.eos_token_id

        B, T = input_ids.shape
        device = input_ids.device

        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = (input_ids != pad_token_id)

        generated_ids = input_ids.clone()
        unfinished_sequences = torch.ones(B, dtype=torch.bool).to(device) # All sequences start unfinished

        self.eval()
        with torch.no_grad():
            # Reset and initialize cache
            if use_cache:
                self.kv_cache.reset()
                self.kv_cache.initialize(B)

            # Process initial sequence
            logits, _, _ = self.forward(
                input_ids=generated_ids, padding_mask=attention_mask, use_cache=use_cache
                )

            if use_cache:
                self.kv_cache.increment_seq_len(T)

            # Generation loop
            for step in range(max_new_tokens):
                current_seq_len = generated_ids.shape[1]

                # Check sequence length limit
                if current_seq_len >= self.model_args.max_seq_len:
                    break

                # Skip if all sequences are finished
                if not unfinished_sequences.any():
                    break

                # Get logits for next token prediction
                if use_cache and step > 0:
                    # For cached generation, only process the last token
                    last_token = generated_ids[:, -1:].contiguous()
                    last_attention = torch.ones(B, 1, dtype=torch.bool).to(device)
                    # Only process unfinished sequences
                    last_attention = last_attention & unfinished_sequences.unsqueeze(1)

                    logits, _, _ = self.forward(
                        input_ids=last_token, padding_mask=last_attention, use_cache=True
                        )
                    self.kv_cache.increment_seq_len(1)
                else:
                    # For non-cached or first step, process full sequence
                    if attention_mask.shape[1] < current_seq_len:
                        # Extend attention mask for new tokens
                        new_attention = torch.cat([
                            attention_mask,
                            unfinished_sequences.unsqueeze(1).expand(-1, current_seq_len - attention_mask.shape[1])
                        ], dim=1)
                    else:
                        new_attention = attention_mask[:, :current_seq_len]

                    logits, _, _ = self.forward(
                        input_ids=generated_ids, padding_mask=new_attention, use_cache=False
                        )

                # Get logits for the last position
                next_token_logits = logits[:, -1, :].clone()

                # Apply temperature
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature
                else:
                    # Temperature 0 means greedy sampling
                    do_sample = False
                    logger.info("Temperature of 0 received, applying greedy decoding.")

                # Apply top-k filtering
                if top_k > 0 and top_k < self.model_args.vocab_size:
                    # Get the top-k values and set others to -inf
                    topk_values, _ = torch.topk(next_token_logits, top_k, dim=-1)
                    min_topk = topk_values[:, -1:].expand_as(next_token_logits)
                    next_token_logits = torch.where(
                        next_token_logits < min_topk,
                        torch.full_like(next_token_logits, float('-inf')),
                        next_token_logits
                    )
                elif top_k == 1:
                    do_sample = False
                    logger.info("Top-k value of 1 received, applying greedy decoding.")

                # Apply top-p (nucleus) filtering
                if 0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Keep at least one token
                    sorted_indices_to_remove[:, 0] = False
                    # Shift right to keep the first token above threshold
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()

                    # Convert back to original indices
                    indices_to_remove = torch.zeros_like(next_token_logits, dtype=torch.bool)
                    indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')

                # Apply sampling
                if do_sample and temperature > 0:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                # Greedy decoding
                else:
                    next_tokens = torch.argmax(next_token_logits, dim=-1)

                # Only update unfinished sequences
                next_tokens = torch.where(unfinished_sequences, next_tokens, pad_token_id)

                # Append new tokens
                generated_ids = torch.cat([generated_ids, next_tokens.unsqueeze(1)], dim=1)

                # Update attention mask
                attention_mask = torch.cat([
                    attention_mask,
                    unfinished_sequences.unsqueeze(1)
                ], dim=1)

                # Check for EOS tokens
                if eos_token_id is not None:
                    unfinished_sequences = unfinished_sequences & (next_tokens != eos_token_id)

        # Reset KV cache
        if use_cache:
            self.kv_cache.reset()

        return generated_ids
