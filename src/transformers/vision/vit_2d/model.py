from configs.setup_env import (
    device,
    dtype,
    use_flash_attn,
    flash_attn_varlen_qkvpacked_func
)

import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.checkpoint import checkpoint

from configs.transformers.vision.vit_2d.model_args.model_args_large import ModelArgs

class PatchEmbeddings(nn.Module):
    """Efficient Patch Embeddings using Conv2d for Vision Transformers.

    Args:
        img_size (int): Height and width of the image (assumed square).
        patch_size (int): Height and width of each patch (assumed square).
        C_in (int): Number of input channels.
        d_model (int): Dimension of output embeddings.
    """
    def __init__(self, img_size: int, patch_size: int, C_in: int, d_model: int):
        super().__init__()

        if img_size % patch_size != 0:
            raise ValueError("img_size must be divisible by patch_size")
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Patch projection
        self.proj = nn.Conv2d(C_in, d_model, kernel_size=patch_size, stride=patch_size)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with patch projection and CLS token.

        Args:
            x (torch.Tensor): [B, C, H, W] input image tensor.

        Returns:
            torch.Tensor: [B, num_patches + 1, d_model]
        """
        with autocast(device_type=device.type, dtype=dtype):
            # Get batch size, height, and width
            B, _, H, W = x.shape
            if H != self.img_size or W != self.img_size:
                raise ValueError(f"Expected input of size {self.img_size}x{self.img_size}, got {H}x{W}")
            
            # Project patches
            x = self.proj(x) # [B, d_model, H/P, W/P]
            x = x.flatten(2) # [B, d_model, num_patches]
            x = x.transpose(1, 2) # [B, num_patches, d_model]

            # Add CLS token
            cls_tokens = self.cls_token.expand(B, -1, -1) # [B, 1, d_model]
            x = torch.cat((cls_tokens, x), dim=1) # [B, num_patches + 1, d_model]

        return x


class RMSNorm(nn.Module):
    """RMSNorm layer applied during GQA/FFN block.

    Args:
        d_model (int): Dimensionality of the model's input/output representations.
        eps (float): Small floating point value for preventing division by zero.
    """
    def __init__(self, d_model: int, eps: float = 1e-7):
        super().__init__()

        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model)) # Scaling factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform forward pass of the RMSNorm layer.

        Args:
            x (torch.Tensor): Input tensor of shape [B, T, d_model].

        Returns:
            torch.Tensor: Normalized output tensor with same shape.
        """
        with autocast(device_type=device.type, dtype=dtype):
            return self.gamma * (
                x / torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
            )


class RoPE(nn.Module):
    """Apply 2D rotary positional embeddings to query, key vectors.

    Args:
        head_dim (int): Dimensionality of each attention head.
        img_size (int): Size of the input image (assumes square images).
        patch_size (int): Size of each patch (assumes square patches).
        base (float): Denominator raised to the power of 2i/d.

    Raises:
        ValueError if `head_dim % 4 != 0`
    """
    def __init__(self, head_dim: int, img_size: int, patch_size: int, base: float = 10000.0):
        super().__init__()

        # Ensure head_dim is divisible by 4 for 2D RoPE
        if head_dim % 4 != 0:
            raise ValueError(f"head_dim must be divisible by 4 for 2D RoPE, head_dim: {head_dim}")
        self.head_dim = head_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size

        # Calculate inverse frequency for both x and y dimensions
        freq_dim = head_dim // 4
        inv_freq = 1.0 / (base ** (torch.arange(0, freq_dim, dtype=torch.float32) / freq_dim))
        self.register_buffer("inv_freq", inv_freq)

    def compute_sine_cosine(
        self,
        grid_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute 2D Sine and Cosine Rotation Matrices for spatial positions.

        Args:
            grid_size (Optional[int]): Grid size (height and width) of the patch grid.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - torch.Tensor: Sine values for x-axis of shape [1, 1, num_patches, head_dim//4].
                - torch.Tensor: Cosine values for x-axis of shape [1, 1, num_patches, head_dim//4].
                - torch.Tensor: Sine values for y-axis of shape [1, 1, num_patches, head_dim//4].
                - torch.Tensor: Cosine values for y-axis of shape [1, 1, num_patches, head_dim//4].
        """
        if grid_size is None:
            grid_size = self.grid_size

        # Create 2D position grid
        pos_x = torch.arange(grid_size, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        pos_y = torch.arange(grid_size, dtype=self.inv_freq.dtype, device=self.inv_freq.device)

        # Create meshgrid and flatten
        grid_x, grid_y = torch.meshgrid(pos_x, pos_y, indexing="ij")
        pos_x_flat = grid_x.flatten().unsqueeze(1) # [num_patches, 1]
        pos_y_flat = grid_y.flatten().unsqueeze(1) # [num_patches, 1]

        # Compute rotation angles for x and y
        theta_x = pos_x_flat * self.inv_freq # [num_patches, head_dim//4]
        theta_y = pos_y_flat * self.inv_freq # [num_patches, head_dim//4]

        # Unsqueeze to match q, k vectors number of dimensions
        sin_x = torch.sin(theta_x).unsqueeze(0).unsqueeze(0) # [1, 1, num_patches, head_dim//4]
        cos_x = torch.cos(theta_x).unsqueeze(0).unsqueeze(0) # [1, 1, num_patches, head_dim//4]
        sin_y = torch.sin(theta_y).unsqueeze(0).unsqueeze(0) # [1, 1, num_patches, head_dim//4]
        cos_y = torch.cos(theta_y).unsqueeze(0).unsqueeze(0) # [1, 1, num_patches, head_dim//4]
        return sin_x, cos_x, sin_y, cos_y

    def create_rotary(
        self,
        x: torch.Tensor,
        sin_x: torch.Tensor,
        cos_x: torch.Tensor,
        sin_y: torch.Tensor,
        cos_y: torch.Tensor
    ) -> torch.Tensor:
        """Create 2D rotary positional embeddings.

        Args:
            x (torch.Tensor): Input tensor of shape [B, T, num_heads, head_dim].
            sin_x (torch.Tensor): Sine values for x-axis of shape [1, 1, T, head_dim//4].
            cos_x (torch.Tensor): Cosine values for x-axis of shape [1, 1, T, head_dim//4].
            sin_y (torch.Tensor): Sine values for y-axis of shape [1, 1, T, head_dim//4].
            cos_y (torch.Tensor): Cosine values for y-axis of shape [1, 1, T, head_dim//4].

        Returns:
            torch.Tensor: Rotated tensor with shape: [B, T, num_heads, head_dim].
        """
        # Split head_dim into 4 parts for 2D rotation (x1, x2, y1, y2)
        freq_dim = self.head_dim // 4
        x_reshaped = x.reshape(*x.shape[:-1], 4, freq_dim) # [B, T, num_heads, 4, head_dim//4]
        x1, x2, y1, y2 = x_reshaped.unbind(dim=-2) # Each have shape: [B, T, num_heads, head_dim//4]

        # Expand sin/cos to match tensor dimensions
        sin_x = sin_x.permute(0, 2, 1, 3).expand(x1.shape[0], x1.shape[1], x1.shape[2], x1.shape[3])
        cos_x = cos_x.permute(0, 2, 1, 3).expand(x1.shape[0], x1.shape[1], x1.shape[2], x1.shape[3])
        sin_y = sin_y.permute(0, 2, 1, 3).expand(y1.shape[0], y1.shape[1], y1.shape[2], y1.shape[3])
        cos_y = cos_y.permute(0, 2, 1, 3).expand(y1.shape[0], y1.shape[1], y1.shape[2], y1.shape[3])

        # Apply 2D rotary embeddings
        # Complex multiplication via rotation matrix
        # rotation matrix = [[cos(x), -sin(x)], [sin(x), cos(x)]]
        # x_rot = x * rotation_matrix
        # y_rot = y * rotation_matrix
        x1_rot = x1 * cos_x - x2 * sin_x
        x2_rot = x1 * sin_x + x2 * cos_x
        y1_rot = y1 * cos_y - y2 * sin_y
        y2_rot = y1 * sin_y + y2 * cos_y

        # Stack back together
        x_rotated = torch.stack((x1_rot, x2_rot, y1_rot, y2_rot), dim=-2) # [B, T, num_heads, 4, head_dim//4]
        x_rotated = x_rotated.reshape(*x.shape) # [B, T, num_heads, head_dim]
        return x_rotated

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 2D rotary positional embeddings to input tensors (qk tensors).

        Args:
            x (torch.Tensor): Input tensor of shape: [B, num_heads, T, head_dim]

        Returns:
            torch.Tensor: Tensor with applied 2D rotary positional embeddings of shape: [B, num_heads, T, head_dim].
        """
        T = x.size(2)
        
        # Transpose to [B, T, num_heads, head_dim] for processing
        x_transposed = x.transpose(1, 2) # [B, T, num_heads, head_dim]

        # Calculate grid size from number of patches
        grid_size = int(math.sqrt(T - 1)) # Exclude CLS token for grid size calculation
        sin_x, cos_x, sin_y, cos_y = self.compute_sine_cosine(grid_size)

        # Apply RoPE only to patch tokens (skip CLS token at position 0)
        cls_token_x = x_transposed[:, :1, :, :] # [B, 1, num_heads, head_dim]
        patch_tokens_x = x_transposed[:, 1:, :, :] # [B, T-1, num_heads, head_dim]
        rotated_patch_tokens = self.create_rotary(patch_tokens_x, sin_x, cos_x, sin_y, cos_y)
        x_final = torch.cat([cls_token_x, rotated_patch_tokens], dim=1) # [B, T, num_heads, head_dim]

        # Transpose back to [B, num_heads, T, head_dim]
        return x_final.transpose(1, 2)


class GroupedQueryAttention(nn.Module):
    """Grouped query attention layer.

    Args:
        d_model (int): Dimensionality of the model's input/output representations.
        num_heads (int): Number of attention heads for queries.
        query_groups (int): Number of key/value groups (heads).
        rope_module (RoPE): An instance of the RoPE module for applying rotary embeddings.
        attn_method (str): Attention method.
            `flash-attn` for GQA + Flash Attention 2 + SWA + GQA.
            `torch-spda` for PyTorch's scaled dot product attention + GQA.
            `manual`     for manual integration of GQA.

    Raises:
        ValueError: If `d_model` is not divisible by `num_heads`.
        ValueError: If `num_heads` is not divisible by `query_groups`.
    """
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        query_groups: int, 
        rope_module: RoPE,
        attn_method: str = "torch-sdpa",
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divible by num_heads ({num_heads})"
                )
        if num_heads % query_groups != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by query_groups ({query_groups})"
                )

        self.d_model = d_model
        self.num_heads = num_heads
        self.query_groups = query_groups
        self.head_dim = d_model // num_heads
        self.attn_method = attn_method

        # QKV projection
        self.w_qkv = nn.Linear(
            d_model,
            num_heads * self.head_dim + 2 * query_groups * self.head_dim,
            bias=False,
            dtype=dtype
        )

        # O projection
        self.w_o = nn.Linear(
            d_model,
            num_heads * self.head_dim,
            bias=False,
            dtype=dtype
        )

        # Initialize 2D RoPE
        self.rope_module = rope_module

    def _expand(
        self, 
        input_tensor: torch.Tensor, 
        heads_per_group: int, 
        dim_to_repeat: int
    ) -> torch.Tensor:
        """Expand kv heads to query heads for GQA
        
        Args:
            input_tensor (torch.Tensor): Input key or value tensor to get expanded.
            heads_per_group (int): Heads per group computed as num_heads // query_groups.
            dim (int): Dimension of tensor to be repeated over.

        Returns:
            torch.Tensor: Output tensor with kv heads expanded.
        """
        return torch.repeat_interleave(input_tensor, heads_per_group, dim=dim_to_repeat)

    def forward(
        self, x: torch.Tensor, 
        window_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Perform forward pass of Attention layer.

        Args:
            x (torch.Tensor): Input tensor of shape [B, T, d_model] where T includes CLS token.
            window_size (Tuple[int, int]): Window size for sliding window attention.

        Returns:
            torch.Tensor: Output tensor transformed with same shape.

        Raises:
            ValueError: If `x` (input tensor) is not 3 dimensional.
            ValueError: If `D` is not equal to `d_model`.
            ValueError: If `q.shape[-1]` is not equal to `k.shape[-1]`.
            ValueError: If `softmax_attn.shape[-1]` is not equal to `v.shape[-2]`.
        """
        with autocast(device_type=device.type, dtype=dtype):
            if x.dim() != 3:
                raise ValueError(f"Input tensor, x, must have 3 dimensions, got: {x.dim()} dimensions")
            B, T, D = x.shape
            if D != self.d_model:
                raise ValueError(f"D ({D}) must be equal to d_model ({self.d_model}).")
            
            # Chunked projection matrix
            qkv = self.w_qkv(x) # [B, T, num_heads * head_dim + 2 * query_groups * head_dim]

            # q: [B, T, num_head * head_dim]
            # kv: [B, T, 2 * query_groups * head_dim]
            q, kv = torch.split(qkv, [self.num_heads * self.head_dim, 2 * self.query_groups * self.head_dim], dim=-1)

            # k, v shape: [B, T, query_groups * head_dim]
            k, v = torch.chunk(kv, chunks=2, dim=-1)
            
            # Reshape into 4D tensors for RoPE
            # q shape: [B, num_heads, T, head_dim]
            # kv shape: [B, query_groups, T, head_dim]
            q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(B, T, self.query_groups, self.head_dim).transpose(1, 2)
            v = v.view(B, T, self.query_groups, self.head_dim).transpose(1, 2)

            # Apply RoPE to q and k using the new method that handles the correct tensor shapes
            q = self.rope_module(q) # [B, num_heads, T, head_dim]
            k = self.rope_module(k) # [B, query_groups, T, head_dim]

            # Compute heads_per_group for GQA
            heads_per_group = self.num_heads // self.query_groups

            # Expand kv heads to num heads for GQA
            k_expanded = self._expand(k, heads_per_group, dim_to_repeat=1)
            v_expanded = self._expand(v, heads_per_group, dim_to_repeat=1)

            # TODO: add checks to make sure dtype in [fp16, bf16]
            # Flash Attention 2 + GQA + SWA
            if self.attn_method == "flash-attn" and use_flash_attn and device.type == "cuda":
                # Stack tensors along the 3rd dimension
                qkv_packed = torch.stack([q, k_expanded, v_expanded], dim=3).contiguous() # [B, T, num_heads, 3, head_dim]
                # Cumulative sequence lengths for all T
                cu_seqlens = torch.arange(0, (B + 1) * T, step=T, dtype=torch.int32).to(device)
                max_seqlen = k_expanded.size(2)
                total_tokens = B * max_seqlen
                qkv_flattened = qkv_packed.view(total_tokens, self.num_heads, 3, self.head_dim)

                # Compute attention output
                attn_out = flash_attn_varlen_qkvpacked_func(
                    qkv_flattened,
                    cu_seqlens,
                    max_seqlen,
                    causal=False,
                    softmax_scale=1.0 / (self.head_dim ** 0.5),
                    window_size=window_size,
                ) # [B * T, num_heads, head_dim]

                # Reshape to [B, T, num_heads, head_dim]
                attn_out = attn_out.contiguous().view(B, T, self.num_heads, self.head_dim)

            # Manual integration of GQA
            elif self.attn_method == "manual":
                if q.shape[-1] != k_expanded.shape[-1]:
                    raise ValueError(
                        f"q.shape[-1] ({q.shape[-1]}) must be equal to k_expanded.shape[-1] ({k_expanded.shape[-1]}) for matrix multiplication."
                    )
                # Compute attention scores
                attn = torch.matmul(q, k_expanded.transpose(-2, -1)) # [B, num_heads, T, T]
                scaled_attn = attn / math.sqrt(self.head_dim)
                softmax_attn = F.softmax(scaled_attn, dim=-1)
                # Compute attention output
                if softmax_attn.shape[-1] != v_expanded.shape[-2]:
                    raise ValueError(
                        f"softmax_attn.shape[-1] ({softmax_attn.shape[-1]}) must be equal to v_expanded.shape[-2] ({v_expanded.shape[-2]}) for matrix multiplication"
                    )
                attn_out = torch.matmul(softmax_attn, v_expanded)

            # PyTorch SDPA (leverages Flash Attention if available)
            elif self.attn_method == "torch-sdpa":
                attn_out = F.scaled_dot_product_attention(
                    q, k_expanded, v_expanded,
                    is_causal=False,
                ) # [B, num_heads, T, head_dim]

            # Invalid attention method
            else:
                raise ValueError(f"Expected 'flash-attn', 'torch-sdpa', or 'manual', got {self.attn_method}")

            # Concatenate heads
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D) # [B, T, d_model]
            return self.w_o(attn_out)


class GQABlock(nn.Module):
    """GQA layer with dropout, RMSNorm and residuals applied.

    Args:
        d_model (int): Dimensionality of the model's input/output representations.
        num_heads (int): Number of attention heads for queries.
        query_groups (int): Number of key/value groups (heads).
        rope_module (RoPE): An instance of the RoPE module for applying rotary embeddings.
    """
    def __init__(self, d_model: int, num_heads: int, query_groups: int, rope_module: RoPE, dropout: float = 0.15):
        super().__init__()

        self.rms_norm = RMSNorm(d_model)
        self.attn = GroupedQueryAttention(d_model, num_heads, query_groups, rope_module)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, window_size: Tuple[int, int]) -> torch.Tensor:
        """Perform forward pass of GQA Block.

        Args:
            x (torch.Tensor): Input tensor of shape [B, T, d_model] where T includes CLS token.

        Returns:
            torch.Tensor: Output tensor with RMSNorm, GQA, Dropout, and residuals applied.
        """
        with autocast(device_type=device.type, dtype=dtype):
            return x + self.dropout(self.attn(self.rms_norm(x), window_size))


class FFN(nn.Module):
    """Feed forward network with SwiGLU activation.

    Args:
        d_model (int): Input and output dimension.
        d_ffn (int): Hidden dimension (usually 4 * d_model).
        dropout (float): Dropout rate.
    """
    def __init__(self, d_model: int, d_ffn: int, dropout: float = 0.15):
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.linear3 = nn.Linear(d_model, d_ffn)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform forward pass through FFN.

        Args:
            x (torch.Tensor): Input tensor of shape [B, T, d_model].

        Returns:
            torch.Tensor: Output tensor of shape [B, T, d_model].
        """
        with autocast(device_type=device.type, dtype=dtype):
            return self.dropout(self.linear2(F.silu(self.linear1(x)) * self.linear3(x)))


class FFNBlock(nn.Module):
    """FFN block which applies RMSNorm, Dropout, and a pass through the FFN.

    Args:
        d_model (int): Dimensionality of the model's input/output representations.
        d_ffn (int): Dimensionality of the feed-forward network.
        dropout (float): Regularizes the model and helps prevent dropout.
    """
    def __init__(self, d_model: int, d_ffn: int, dropout: float = 0.15):
        super().__init__()

        self.rms_norm = RMSNorm(d_model)
        self.ffn = FFN(d_model, d_ffn)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through FFN Block.

        Args:
            x (torch.Tensor): Input tensor of shape [B, T, d_model].

        Returns:
            torch.Tensor: Output tensor with RMSNorm, FFN, Dropout, and residuals applied.
        """
        with autocast(device_type=device.type, dtype=dtype):
            return x + self.dropout(self.ffn(self.rms_norm(x)))


class TransformerEncoder(nn.Module):
    """Encoder block where attention block and FFN blocks are stacked.

    Args:
        d_model (int): Dimensionality of the model's input/output representations.
        num_heads (int): Number of attention heads for queries.
        query_groups (int): Number of key/value groups (heads).
        rope_module (nn.Module): An instance of the RoPE module for applying rotary embeddings.
        d_ffn (int): Dimensionality of the feed-forward network. Typically, d_ffn = 4 * d_model.
        dropout (float): Regularizes the model and helps prevent overfitting.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        query_groups: int,
        rope_module: RoPE,
        d_ffn: int,
        dropout: float = 0.15
    ):
        super().__init__()

        self.attn_block = GQABlock(d_model, num_heads, query_groups, rope_module, dropout)
        self.ffn_block = FFNBlock(d_model, d_ffn, dropout)

    def forward(self, x: torch.Tensor, window_size: Tuple[int, int]) -> torch.Tensor:
        """Perform forward pass of the Transformer Encoder.

        Args:
            x (torch.Tensor): Input tensor of shape [B, T, d_model].

        Returns:
            torch.Tensor: Transformed output tensor of same shape.
        """
        with autocast(device_type=device.type, dtype=dtype):
            return self.ffn_block(self.attn_block(x, window_size))


class VisionTransformer(nn.Module):
    """Complete Vision Transformer class where the encoder blocks will be stacked.

    Args:
        model_args (ModelArgs): Dataclass containing all model hyperparameters.
    """
    def __init__(self, model_args: ModelArgs):
        super().__init__()

        self.model_args = model_args

        # Patch embeddings
        self.patch_embeddings = PatchEmbeddings(
            img_size=model_args.img_size,
            patch_size=model_args.patch_size,
            C_in=model_args.C_in,
            d_model=model_args.d_model
        )

        # RoPE
        head_dim = model_args.d_model // model_args.num_heads
        self.rope = RoPE(
            head_dim=head_dim,
            img_size=model_args.img_size,
            patch_size=model_args.patch_size,
            base=model_args.rope_base
        )

        # Stack Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerEncoder(
                d_model=model_args.d_model,
                num_heads=model_args.num_heads,
                query_groups=model_args.query_groups,
                rope_module=self.rope,
                d_ffn=model_args.d_ffn,
                dropout=model_args.dropout
            ) for _ in range(model_args.num_layers)
        ])

        # RMSNorm
        self.rms_norm = RMSNorm(model_args.d_model, model_args.rms_norm_eps)

        # Classification head
        self.classifier = nn.Linear(model_args.d_model, model_args.num_classes)

        # Dropout
        self.dropout = nn.Dropout(p=model_args.dropout)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Linear) -> None:
        """Initialize weights using Xavier initialization.

        Args:
            module (nn.Linear): Module to initialize.
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform forward pass through the Vision Transformer.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W].

        Returns:
            torch.Tensor: Class logits of shape [B, num_classes].
        """
        with autocast(device_type=device.type, dtype=dtype):
            x = self.dropout(self.patch_embeddings(x)) # [B, num_patches + 1, d_model]

            # Pass through transformer layers
            for layer in self.transformer_layers:
                if self.model_args.use_checkpointing:
                    x = checkpoint(
                        layer, 
                        x, 
                        self.model_args.window_size, 
                        use_reentrant=False
                    ) # [B, num_patches + 1, d_model]
                else:
                    x = layer(x, self.model_args.window_size)

            # Apply final RMSNorm
            x = self.rms_norm(x) # [B, num_patches + 1, d_model]

            # Extract CLS token for classification
            cls_token = x[:, 0] # [B, d_model]

            # Classification head
            logits = self.classifier(cls_token) # [B, num_classes]
            return logits
