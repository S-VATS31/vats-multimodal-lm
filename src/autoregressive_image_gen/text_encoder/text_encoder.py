from configs.setup_env import device, dtype

import math
from typing import Optional

import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.checkpoint import checkpoint

from src.rms_norm import RMSNorm
from src.ffn_block import FFNBlock
from src.autoregressive_image_gen.text_encoder.encoder_attention import AttentionBlock
from configs.autoregressive_image_gen.text_encoder.model_args.model_args_xsmall import ModelArgs

class TransformerBlock(nn.Module):
    """  
    Args:
        d_model (int): Dimensionality of model embeddings.
        num_heads (int): Number of attention heads for queries.
        query_groups (int): Number of query groups for GQA.
        d_ffn (int): Dimensionality of the FFN.
        theta (float): Exponential base of the inverse frequency for RoPE.
        softmax_scale (float): Factor to scale the attention computation before softmax by.
        use_proj_bias (bool): Whether to use bias for q, k, v projection matrices.
        use_qkv_proj (bool): Whether to use fused qkv proj or seperate projections.
        eps (float): Epsilon value to prevent numerical instability in RMSNorm.
        dropout (float): Dropout probability.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        query_groups: int,
        d_ffn: int,
        theta: float,
        softmax_scale: float,
        use_proj_bias: bool,
        use_qkv_proj: bool,
        eps: float,
        dropout: float
    ):
        super().__init__()

        self.attention_block = AttentionBlock(
            d_model=d_model,
            num_heads=num_heads,
            query_groups=query_groups,
            theta=theta,
            softmax_scale=softmax_scale,
            use_proj_bias=use_proj_bias,
            use_qkv_proj=use_qkv_proj,
            eps=eps,
            dropout=dropout,
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
        left_window: int,
        right_window: int,
        enable_mqa: bool,
        padding_mask: Optional[torch.Tensor] = None,
        use_diffusion: bool = True
    ) -> torch.Tensor:
        """Forward pass of Transformer block layer.

        Args:
            x (torch.Tensor): Input tensor of shape [B, T, d_model].
            left_window (int): Left window for SWA.
            right_window (int): Right window for SWA.
            enable_mqa (bool): Whether to use MQA or not.
            padding_mask (torch.Tensor): Padding tensor of shape [B, T].
            use_diffusion (bool): Whether this encoder is being used for diffusion or not.
                If True, both window sizes will be set to -1 for global attention.

        Returns:
            torch.Tensor: Output tensor of same shape as input.
        """
        with autocast(device_type=device.type, dtype=dtype):
            return self.ffn_block(self.attention_block(
                x,
                left_window=left_window,
                right_window=right_window,
                enable_mqa=enable_mqa,
                padding_mask=padding_mask,
                use_diffusion=use_diffusion
            ))
        
class TransformerTextEncoder(nn.Module):
    """Text encoder complete module.
    
    Args:
        model_args (ModelArgs): Model hyperparameters.
    """
    def __init__(self, model_args: ModelArgs):
        super().__init__()

        self.model_args = model_args

        self.token_embedding = nn.Embedding(model_args.vocab_size, model_args.d_model).to(device)
        self.dropout = nn.Dropout(p=model_args.dropout).to(device)

        # Stack encoder blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=model_args.d_model,
                num_heads=model_args.num_heads,
                query_groups=model_args.query_groups,
                d_ffn=model_args.d_ffn,
                theta=model_args.rope_theta,
                softmax_scale=model_args.softmax_scale,
                use_proj_bias=model_args.use_proj_bias,
                use_qkv_proj=model_args.use_qkv_proj,
                eps=model_args.rms_norm_eps,
                dropout=model_args.dropout
            ).to(device) for _ in range(model_args.num_layers)
        ])

        # Initialize final RMSNorm
        self.rms_norm = RMSNorm(model_args.d_model, model_args.rms_norm_eps).to(device)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module) -> None:
        """Weight initialization for a text encoder used in image generation."""
        if isinstance(module, nn.Linear):
            # Xavier uniform initialization
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Embedding weights: normal or Xavier uniform
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            # Sometimes used for positional encoding projections
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.LongTensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass encoder.
        
        Args:
            input_ids (torch.LongTensor): Input tensor of shape [B, T].
            padding_mask (Optional[torch.Tensor]): Padding tensor of shape [B, T].

        Returns:
            torch.Tensor: Output tensor of shape [B, T, d_model].
        """
        with autocast(device_type=device.type, dtype=dtype):
            input_ids = input_ids.to(torch.int64, copy=False)
            assert (
                input_ids.dim() == 2
            ), f"input_ids must be of shape [B, T], got {input_ids.dim()} dimensions."
            assert (
                input_ids.dtype == torch.int64
            ), f"input_ids must have dtype of int64, got {input_ids.dtype}"

            # Apply embeddings
            x = self.dropout(self.token_embedding(input_ids)) # [B, T, d_model]

            assert (
                x.dim() == 3
            ), f"x must be of shape [B, T, d_model], got {x.dim()} dimensions."
            assert (
                input_ids.shape == x.shape[:2]
            ), "input_ids and x must have first two dims equal."

            # Loop through layers
            for layer in self.layers:
                if self.model_args.use_checkpointing:
                    x = checkpoint(
                        layer,
                        x,
                        self.model_args.left_window,
                        self.model_args.right_window,
                        self.model_args.enable_mqa,
                        padding_mask,
                        self.model_args.use_diffusion,
                        use_reentrant=False
                    )
                else:
                    x = layer(
                        x=x,
                        left_window=self.model_args.left_window,
                        right_window=self.model_args.right_window,
                        enable_mqa=self.model_args.enable_mqa,
                        padding_mask=padding_mask,
                        use_diffusion=self.model_args.use_diffusion
                    )

            assert (
                x.dim() == 3
            ), f"x must be a 3 dimensional tensor, got {x.dim()}"

            # Apply final RMSNorm
            x = self.rms_norm(x)

            assert (
                x.dim() == 3
            ), f"x must be a 3 dimensional tensor, got {x.dim()}"

            return x
    
def main(use_pad: bool):
    model_args = ModelArgs()
    model = TransformerTextEncoder(model_args).to(device)
    B, T = 4, 16
    input_ids = torch.randint(
        0, model_args.vocab_size, (B, T), dtype=torch.int64
    ).to(device)
    if use_pad:
        print("using pad")
        padding_mask = torch.randint(0, 2, (B, T), dtype=torch.bool).to(device)
        return model(input_ids, padding_mask)
    print("not using pad")
    return model(input_ids)

if __name__ == "__main__":
    x = main(use_pad=True)
    print(x.shape)
