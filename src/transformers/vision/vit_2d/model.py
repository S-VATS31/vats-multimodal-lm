from configs.setup_env import device, dtype

from typing import Tuple

import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.checkpoint import checkpoint

from src.rms_norm import RMSNorm
from src.swiglu_activation import SwiGLUActivation
from src.transformers.vision.vit_2d.optimized_attention import GQABlock, RoPE
from configs.transformers.vision.vit_2d.model_args.model_args_large import ModelArgs

class PatchEmbeddings(nn.Module):
    """Patch Embeddings using Conv2d for Vision Transformers.

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
        self.d_model = d_model

        # Patch projection
        self.proj = nn.Conv2d(C_in, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with patch projection.

        Args:
            x (torch.Tensor): [B, C, H, W] input image tensor.

        Returns:
            torch.Tensor: [B, num_patches, d_model].
        """
        with autocast(device_type=device.type, dtype=dtype):
            assert (
                x.dim() == 4
            ), f"x must have 4 dimensions, got {x.dim()} dimensions"

            # Get height/width of image
            B, _, H, W = x.shape
            if H != self.img_size or W != self.img_size:
                raise ValueError(f"Expected input of size {self.img_size}x{self.img_size}, got {H}x{W}")
            
            # Project patches
            x = self.proj(x) # [B, d_model, H/P, W/P]
            assert (
                x.shape == (B, self.d_model, H / self.patch_size, W / self.patch_size)
            ), f"x must have shape of {(B, self.d_model, H / self.patch_size, W / self.patch_size)}, got {x.shape}"

            x = x.flatten(2) # [B, d_model, num_patches]
            assert (
                x.shape == (B, self.d_model, self.num_patches)
            ), f"x must have shape of {(B, self.d_model, self.num_patches)}, got {x.shape}"

            x = x.transpose(1, 2) # [B, num_patches, d_model]
            assert (
                x.shape == (B, self.num_patches, self.d_model)
            ), f"x must have shape of {(B, self.num_patches, self.d_model)}, got {x.shape}"

        return x

class FFNBlock(nn.Module):
    """FFN block which applies RMSNorm, Dropout, and a pass through the FFN.

    Args:
        d_model (int): Dimensionality of the model's input/output representations.
        d_ffn (int): Dimensionality of the feed-forward network.
        eps (float): Small value to maintain numerical stability in RMSNorm.
        dropout (float): Regularizes the model and helps prevent dropout.
    """
    def __init__(self, d_model: int, d_ffn: int, eps: float, dropout: float):
        super().__init__()

        self.rms_norm = RMSNorm(d_model, eps)
        self.ffn = SwiGLUActivation(d_model, d_ffn, dropout)
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
        eps: (float): Small value to maintain numerical stability in RMSNorm.
        dropout (float): Regularizes the model and helps prevent overfitting.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        query_groups: int,
        rope_module: RoPE,
        d_ffn: int,
        eps: float,
        dropout: float,
    ):
        super().__init__()

        self.attn_block = GQABlock(d_model, num_heads, query_groups, rope_module, eps, dropout)
        self.ffn_block = FFNBlock(d_model, d_ffn, eps, dropout)

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
                eps=model_args.rms_norm_eps,
                dropout=model_args.dropout
            ) for _ in range(model_args.num_layers)
        ])

        # RMSNorm
        self.rms_norm = RMSNorm(model_args.d_model, model_args.rms_norm_eps)

        # Adaptive average pooling
        # output size of 1 as we aggregate all predictions into a single vector
        self.pool = nn.AdaptiveAvgPool1d(1)

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
            assert (
                x.dim() == 4
            ), f"x must be a 4 dimensional tensor, got {x.dim()} dimensions"
            
            # Patch embeddings + dropout
            x = self.dropout(self.patch_embeddings(x)) # [B, num_patches, d_model]
            assert (
                x.dim() == 3
            ), f"x must be a 3 dimensional tensor, got {x.dim()} dimensions."

            # Pass through transformer layers
            for layer in self.transformer_layers:
                if self.model_args.use_checkpointing:
                    x = checkpoint(
                        layer, 
                        x, 
                        self.model_args.window_size, 
                        use_reentrant=False
                    ) # [B, num_patches, d_model]
                else:
                    x = layer(x, self.model_args.window_size)

            # Apply final RMSNorm
            x = self.rms_norm(x) # [B, num_patches, d_model]
            x = x.transpose(1, 2) # [B, d_model, num_patches]
            x = self.pool(x) # [B, d_model, 1]
            assert (
                x.size(-1) == 1
            ), f"x.size(-1) must be equal to 1, got {x.size(-1)}"

            x = x.squeeze(-1) # [B, d_model]
            assert (
                x.dim() == 2
            ), f"x must be a 2 dimensional tensor, got {x.dim()}"
            
            # Get output logits
            logits = self.classifier(x)
            assert (
                logits.dim() == 2
            ), f"logits must be a 2 dimensional tensor, got {logits.dim()}"

            return logits # [B, num_classes]
