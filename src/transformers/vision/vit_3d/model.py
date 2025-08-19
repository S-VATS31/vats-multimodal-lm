from configs.setup_env import device, dtype

from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.checkpoint import checkpoint

from src.rms_norm import RMSNorm
from src.ffn_block import FFNBlock
from src.transformers.vision.vit_3d.optimized_attention import SpatioTemporalAttentionBlock
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

        self.attention_block = SpatioTemporalAttentionBlock(
            d_model, num_heads, query_groups, 
            rope_theta, patch_size, eps, dropout
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
            x (torch.Tensor): Input tensor of shape [B, T, H*W, d_model].
            grid_size (Tuple[int, int, int]): Grid size parameter for RoPE.
            window_size (Optional[Tuple[int, int]]): Window size for SWA.
            padding_mask (Optional[torch.Tensor]): Padding tensor.

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
                patch_size=model_args.patch_size
            ).to(device) for _ in range(model_args.num_layers)
        ])

        # Set up RMSNorm and Dropout
        self.rms_norm = RMSNorm(model_args.d_model, model_args.rms_norm_eps)
        self.dropout = nn.Dropout(p=model_args.dropout)

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
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform forward pass of the entire video transformer.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, T, H, W].

        Returns:
            torch.Tensor: Returns logits of shape [B, num_classses].
        """
        # We only need batch size for assertions as C gets projected to d_model
        # and T, H, W get dynamically calculated through grid size
        B = x.size(0)
        assert(
            x.dim() == 5
        ), f"x must be a 5 dimensional tensor, got {x.dim()} dimensions"

        # Apply patch embeddings and dropout, get processed dimensions 
        x, padding_mask, grid_size = self.patch_embeddings(x)
        x = self.dropout(x) # [B, T, H*W, d_model]

        # Grid size contains 'real' T, H, W values after padding/truncating
        new_T, new_H, new_W = grid_size

        assert (
            x.shape == (B, new_T, new_H*new_W, self.model_args.d_model)
        ), (
            f"x must have shape of {(B, new_T, new_H*new_W, self.model_args.d_model)}, "
            f"got {x.shape}"
        )
        assert (
            padding_mask.shape == (B, new_T*new_H*new_W)
        ), (
            f"padding_mask must have shape of {(B, new_T*new_H*new_W)}, "
            f"got {padding_mask.shape}"
        )

        # Stack transformer blocks
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
                x = layer(
                    x=x,
                    grid_size=grid_size,
                    window_size=self.model_args.window_size,
                    padding_mask=padding_mask
                )

        assert (
            x.shape == (B, new_T, new_H*new_W, self.model_args.d_model)
        ), (
            f"x must have shape of {(B, new_T, new_H*new_W, self.model_args.d_model)}, "
            f"got {x.shape}"
        )
                
        # Apply final RMSNorm
        x = self.rms_norm(x)

        assert (
            x.shape == (B, new_T, new_H*new_W, self.model_args.d_model)
        ), (
            f"x must have shape of {(B, new_T, new_H*new_W, self.model_args.d_model)}, "
            f"got {x.shape}"
        )

        return x # [B, T, H*W, d_model]
        


def test_transformer_block(use_pad: bool):
    d_model, num_heads, query_groups, rope_theta = 744, 124, 2, 10000.0
    d_ffn, dropout, eps, = 4*d_model, 0.15, 1e-7
    patch_size = (2, 32, 32)
    transformer_block = TransformerBlock(
        d_model, num_heads, query_groups, rope_theta,
        d_ffn, dropout, eps, patch_size
    )
    B, T, H, W = 1, 2, 144, 144
    pt, ph, pw = patch_size
    new_T, new_H, new_W = T // pt, H // ph, W // pw
    grid_size = (new_T, new_H, new_W)
    x = torch.randn(B, new_T, new_H*new_W, d_model).to(device)
    if use_pad:
        print("USING PADDING")
        padding_mask = torch.randint(0, 2, (B, new_T*new_H*new_W), dtype=torch.bool).to(device)
    else:
        print("NOT USING PADDING")
        padding_mask = None
    x_out = transformer_block(
        x=x,
        grid_size=grid_size,
        window_size=(-1, -1),
        padding_mask=padding_mask,
    )
    return x_out

def test_entire_forward():
    model_args = ModelArgs()
    model = VideoTransformer(model_args).to(device)
    B, C, T, H, W = 1, 3, 11, 144, 144
    x = torch.randn(B, C, T, H, W).to(device)
    x_out = model(x)
    return x_out

if __name__ == "__main__":
    x = test_entire_forward()
    # [B, T, H*W, d_model]
    # [1, T//pt, (H//ph) * (W//pw), 2112]
    # [1, 8//2, (224//16) * (224//16), 2112]
    # we do 8//2 because max frames is 8
    # we do 224 for H and W since our target size is 224
    # [1, 4, 196, 2112]
    print(x.shape)