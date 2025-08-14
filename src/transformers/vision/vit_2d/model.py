from configs.setup_env import device, dtype

from typing import Tuple

import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.checkpoint import checkpoint

from src.rms_norm import RMSNorm
from src.ffn_block import FFNBlock
from src.transformers.vision.vit_2d.patch_embeddings2d import PatchEmbeddings2D
from src.transformers.vision.vit_2d.optimized_attention import GQABlock, RoPE
from configs.transformers.vision.vit_2d.model_args.model_args_large import ModelArgs

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
        self.patch_embeddings = PatchEmbeddings2D(
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
