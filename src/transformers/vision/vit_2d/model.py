from configs.setup_env import device, dtype

import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.checkpoint import checkpoint

from src.rms_norm import RMSNorm
from src.ffn_block import FFNBlock
from src.transformers.vision.vit_2d.optimized_attention import SpatialAttentionBlock
from src.transformers.vision.vit_2d.patch_embeddings2d import PatchEmbeddings2D
from configs.transformers.vision.vit_2d.model_args.model_args_medium import ModelArgs

class SpatialTransformerBlock(nn.Module):
    """Transformer block stacking attn/ffn blocks.

    Args:
        d_model (int): Dimensionality of model embeddings.
        num_heads (int): Number of attention heads for GQA.
        query_groups (int): Number of query groupsf for GQA.
        rope_theta (float): Exponential base of inv freq for RoPE.
        target_size (int): Target height and width images will be reshaped to.
        patch_size (int): Height and width square patches.
        softmax_scale (float): Scaling factor for attention scores.
        use_windowed_attn (bool): Whether to use sliding window attention or not.
        use_proj_bias (bool): Whether to use bias for projection matrices.
        use_fused_proj (bool): Whether to use single qkv projection or seperate projections.
        eps (float): Small epsilon value to prevent numerical instability.
        dropout (float): Dropout probability.
        d_ffn (int): Dimensionality of the feed forward network.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        query_groups: int,
        rope_theta: float,
        target_size: int,
        patch_size: int,
        softmax_scale: float,
        use_windowed_attn: bool,
        use_proj_bias: bool,
        use_fused_proj: bool,
        eps: float,
        dropout: float,
        d_ffn: int
    ):
        super().__init__()

        self.attention_block = SpatialAttentionBlock(
            d_model=d_model,
            num_heads=num_heads,
            query_groups=query_groups,
            rope_theta=rope_theta,
            target_size=target_size,
            patch_size=patch_size,
            softmax_scale=softmax_scale,
            use_windowed_attn=use_windowed_attn,
            use_proj_bias=use_proj_bias,
            use_fused_proj=use_fused_proj,
            eps=eps,
            dropout=dropout
        )
        self.ffn_block = FFNBlock(
            d_model=d_model,
            d_ffn=d_ffn,
            dropout=dropout,
            eps=eps
        )

    def forward(
        self,
        x: torch.Tensor,
        use_mqa: bool,
        left_window: int,
        right_window: int
    ) -> torch.Tensor:
        """Forward pass of the transformer block.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, H*W, d_model].
            use_mqa (bool): Whether to use MQA or not.
            left_window (int): Left window for SWA.
            right_window (int): Right window for SWA.

        Returns:
            torch.Tensor: Output tensor of same shape as input.
        """
        with autocast(device_type=device.type, dtype=dtype):
            return self.ffn_block(
                self.attention_block(
                    x=x,
                    use_mqa=use_mqa,
                    left_window=left_window,
                    right_window=right_window
                )
            )

class ImageEncoderTransformer(nn.Module):
    def __init__(self, model_args: ModelArgs):
        super().__init__()

        self.model_args = model_args

        # Set up patch embeddings
        self.patch_embeddings = PatchEmbeddings2D(
            patch_size=model_args.patch_size,
            target_size=model_args.target_size,
            C_in=model_args.C_in,
            d_model=model_args.d_model
        ).to(device)

        # Set up transformer blocks
        self.layers = nn.ModuleList([
            SpatialTransformerBlock(
                d_model=model_args.d_model,
                num_heads=model_args.num_heads,
                query_groups=model_args.query_groups,
                rope_theta=model_args.rope_theta,  # ADD TO MODEL ARGS
                target_size=model_args.target_size,  # ADD TO MODEL ARGS
                patch_size=model_args.patch_size,
                softmax_scale=model_args.softmax_scale,  # ADD TO MODEL ARGS
                use_windowed_attn=model_args.use_windowed_attn,  # ADD TO MODEL ARGS
                use_proj_bias=model_args.use_proj_bias,  # ADD TO MODEL ARGS
                use_fused_proj=model_args.use_fused_proj,  # ADD TO MODEL ARGS
                eps=model_args.rms_norm_eps,
                dropout=model_args.dropout,
                d_ffn=model_args.d_ffn
            ).to(device) for _ in range(model_args.num_layers)
        ])

        # Set up final RMSNorm and dropout
        self.rms_norm = RMSNorm(model_args.d_model, model_args.rms_norm_eps).to(device)
        self.dropout = nn.Dropout(p=model_args.dropout)

        # Initialize weights
        self.apply(self._init_weights)


    def _init_weights(self, module) -> None:
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of image encoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W].

        Returns:
            torch.Tensor: Output tensor with same shape.
        """
        assert (
            x.dim() == 4
        ), f"x must have 4 dimensions, got {x.dim()} dimensions."
        x = self.dropout(self.patch_embeddings(x)) # [B, H*W, d_model]

        # Stack transformer blocks
        for layer in self.layers:
            if self.model_args.use_checkpointing:
                x = checkpoint(
                    layer,
                    x,
                    self.model_args.use_mqa,
                    self.model_args.left_window,
                    self.model_args.right_window,
                    use_reentrant=False
                )
            else:
                x = layer(
                    x=x,
                    use_mqa=self.model_args.use_mqa,
                    left_window=self.model_args.left_window,
                    right_window=self.model_args.right_window,
                )
        
        # Apply final RMSNorm
        x = self.rms_norm(x)

        return x