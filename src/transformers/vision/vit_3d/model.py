from configs.setup_env import device, dtype

from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.checkpoint import checkpoint

from src.rms_norm import RMSNorm
from src.transformers.vision.ffn_block import FFNBlock
from src.transformers.vision.vit_3d.optimized_attention import AttentionBlock
from src.transformers.vision.vit_3d.patch_embeddings3d import PatchEmbeddings3D
from configs.transformers.vision.vit_3d.model_args.model_args_large import ModelArgs
        
class TransformerBlock(nn.Module):
    """Transformer block where Attention/FFN layers are stacked.
    
    Args:
        d_model (int): Dimensionality of the model's embeddings.
        num_heads (int): Number of attention heads.
        query_groups (int): Number of query groups for GQA.
        rope_theta (float): Theta hyperparameter for RoPE.
        d_ffn (int): Dimensionality of the FFN.
        dropout (float): Dropout probability.
        eps (float): Small epsilon value to prevent numerical instability.
        patch_size (Tuple[int, int, int]): T, H, W sizes for 3D patch of video.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        query_groups: int,
        rope_theta: float,
        d_ffn: int,
        dropout: float,
        eps: float,
        patch_size: Tuple[int, int, int],
    ):
        super().__init__()

        self.attention_block = AttentionBlock(
            d_model, num_heads, query_groups, 
            rope_theta, eps, dropout, patch_size
        )
        self.gated_ffn_block = FFNBlock(
            d_model, d_ffn, dropout, eps
        )

    def forward(
        self, 
        x: torch.Tensor,
        grid_size: Tuple[int, int, int],
        window_size: Optional[Tuple[int, int]] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Perform forward pass of the Transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape [B, N, d_model].
            grid_size (Tuple[int, int, int]): Grid size parameter for RoPE.
            window_size (Optional[Tuple[int, int]]): Window size for SWA.

        Returns:
            torch.Tensor: Output tensor of shape [B, N, d_model].
        """
        with autocast(device_type=device.type, dtype=dtype):
            return self.gated_ffn_block(
                self.attention_block(x, grid_size, window_size, padding_mask)
            )
        

class VideoTransformer(nn.Module):
    """Complete video transformer module.
    
    Args:
        model_args (ModelArgs): Dataclass containing all model hyperparameters.
    """
    def __init__(self, model_args: ModelArgs):
        super().__init__()

        self.model_args = model_args

        # Set up patch embeddings
        self.patch_embeddings = PatchEmbeddings3D(
            C_in=model_args.C_in,
            patch_size=model_args.patch_size,
            target_size=model_args.target_size,
            max_frames=model_args.max_frames,
            d_model=model_args.d_model,
        )

        # Stack transformer encoder layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=model_args.d_model,
                num_heads=model_args.num_heads,
                query_groups=model_args.query_groups,
                rope_theta=model_args.rope_theta,
                d_ffn=model_args.d_ffn,
                dropout=model_args.dropout,
                eps=model_args.rms_norm_eps,
                patch_size=model_args.patch_size,
            ) for _ in range(model_args.num_layers)
        ])

        # Set up RMSNorm and Dropout
        self.rms_norm = RMSNorm(model_args.d_model, model_args.rms_norm_eps)
        self.dropout = nn.Dropout(p=model_args.dropout)

        # Set up pool for adaptive average pooling
        self.pool = nn.AdaptiveAvgPool1d(1) # Pool over number of patches, N

        # Set up classifier
        self.classifier = nn.Linear(model_args.d_model, model_args.num_classes)

        # Initialize weights
        self.apply(self._init_weights)
        self._apply_depth_scaled_init()

    def _init_weights(self, module) -> None:
        """Weight initialization for VideoTransformer modules.
        
        Args:
            module: PyTorch module to initialize.
        """
        if isinstance(module, nn.Linear):
            # For linear layers, use Xavier/Glorot uniform with proper scaling
            if hasattr(module, 'weight') and module.weight is not None:
                # Special handling for different linear layer types
                if any(name in str(module) for name in ['w_qkv', 'w_o']):
                    std = (2.0 / (module.weight.size(-1) + module.weight.size(-2))) ** 0.5
                    nn.init.normal_(module.weight, mean=0.0, std=std)
                elif 'classifier' in str(module):
                    # Final classifier layer - use smaller initialization for stability
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                elif any(name in str(module) for name in ['w1', 'w3']):
                    # FFN input projections - Xavier uniform
                    nn.init.xavier_uniform_(module.weight, gain=1.0)
                elif 'w2' in str(module):
                    # FFN output projection - smaller initialization for residual stability
                    std = 0.02 / (2 * self.model_args.num_layers) ** 0.5
                    nn.init.normal_(module.weight, mean=0.0, std=std)
                else:
                    # Default for other linear layers
                    nn.init.xavier_uniform_(module.weight, gain=1.0)
            
            # Initialize biases to zero if they exist
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
        
        elif isinstance(module, nn.Conv3d):
            # 3D convolution for patch embedding - use Kaiming initialization
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.kaiming_normal_(
                    module.weight, 
                    mode='fan_out', 
                    nonlinearity='linear'
                )
            
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
        
        elif isinstance(module, RMSNorm):
            # RMSNorm weight initialization
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.ones_(module.weight)
        
        elif isinstance(module, nn.Embedding):
            # Embedding layers
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
        elif isinstance(module, nn.Parameter):
            # Handle any standalone parameters
            if module.dim() == 1:
                # 1D parameters (like normalization weights)
                nn.init.ones_(module)
            else:
                # Multi-dimensional parameters
                nn.init.xavier_uniform_(module)

    def _apply_depth_scaled_init(self) -> None:
        """Apply depth-scaled initialization as a post-processing step."""
        for layer in self.layers:
            # Scale residual connections by layer depth
            scale_factor = (2 * self.model_args.num_layers) ** -0.5
            
            # Scale attention output projection
            if hasattr(layer.attention_block.attention, 'w_o'):
                layer.attention_block.attention.w_o.weight.data.mul_(scale_factor)
            
            # Scale FFN output projection  
            if hasattr(layer.gated_ffn_block.gated_ffn, 'w2'):
                layer.gated_ffn_block.gated_ffn.w2.weight.data.mul_(scale_factor)
        
    def forward(
        self, 
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Perform forward pass of the entire video transformer.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, T, H, W].

        Returns:
            torch.Tensor: Returns logits of shape [B, num_classses].
        """
        assert(
            x.dim() == 5
        ), f"x must be a 5 dimensional tensor, got {x.dim()} dimensions"

        # Apply patch embeddings and dropout, get processed dimensions 
        x, _, padding_mask, grid_size = self.patch_embeddings(x)
        x = self.dropout(x) # [B, N, d_model]
        
        assert(
            x.dim() == 3
        ), f"x must be a 3 dimenional tensor after patch embeddings, got {x.dim()} dimensions."

        # Pass through transformer encoder layers
        for layer in self.layers:
            if self.model_args.use_checkpointing:
                x = checkpoint(
                    layer, 
                    x, 
                    grid_size,
                    self.model_args.window_size,
                    padding_mask,
                    use_reentrant=False
                )
            else:
                x = layer(x, grid_size, self.model_args.window_size, padding_mask)

        # Apply final RMSNorm
        x = self.rms_norm(x) # [B, N, d_model]

        # Apply adaptive average pooling
        x = x.transpose(1, 2) # [B, d_model, N]
        x = self.pool(x) # [B, d_model, 1]
        assert(
            x.size(-1) == 1
        ), f"x.size(-1) must be equal to 1, got {x.size(-1)}"

        x = x.squeeze(-1) # [B, d_model]
        assert(
            x.dim() == 2
        ), f"x must be a 2 dimensional tensor, got {x.dim()} dimensions"

        # Get logits through classifier
        logits = self.classifier(x)
        assert(
            logits.dim() == 2
        ), f"logits must be a 2 dimensional tensor, got {logits.dim()} dimensions"

        return logits # [B, num_classes]

def main():
    model_args = ModelArgs()
    model = VideoTransformer(model_args).to(device)
    x = torch.randn(1, 3, 4, 16, 16).to(device)
    logits = model(x)
    return logits

if __name__ == "__main__":
    logits = main()
    print(logits.shape)
