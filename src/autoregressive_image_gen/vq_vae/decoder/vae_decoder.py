from configs.setup_env import device, dtype

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

class VQVAEDecoder(nn.Module):
    """VQ VAE Decoder layer using deconvolution.
    
    Args:
        C_out (int): Number of output channels. Should be equal to C_in in encoder.
        d_model (int): Dimensionality of model embeddings.
        activation_func (Literal["relu", "leaky_relu", "sigmoid"]): Non-linear activation.
    """
    def __init__(
        self,
        C_out: int, 
        d_model: int,
        activation_func: Literal["relu", "leaky_relu", "sigmoid"]
    ):
        super().__init__()

        # Deconvolution block 1; [3, 1, 1]
        self.deconv1 = nn.ConvTranspose2d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.batch_norm1 = nn.BatchNorm2d(
            num_features=d_model
        )

        # Deconvolution block 2; [4, 2, 1]
        self.deconv2 = nn.ConvTranspose2d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self.batch_norm2 = nn.BatchNorm2d(
            num_features=d_model
        )

        # Deconvolution block 3; [4, 2, 1]
        self.deconv3 = nn.ConvTranspose2d(
            in_channels=d_model,
            out_channels=C_out,
            kernel_size=4,
            stride=2,
            padding=1
        )

        # Set up activation for intermediate layers
        if activation_func == "relu":
            self.activation = nn.ReLU()
        elif activation_func == "leaky_relu":
            self.activation = nn.LeakyReLU()
        elif activation_func == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(
                f"expected ['relu', 'leaky_relu', 'sigmoid'], got {activation_func}"
            )
        
        # Set up sigmoid activation output layer
        self.sigmoid = nn.Sigmoid()
        
        self.C_out = C_out
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of Decoder layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, H*W, d_model].

        Returns:
            torch.Tensor: Output tensor of shape [B, C_out, H, W].
        """
        with autocast(device_type=device.type, dtype=dtype):
            assert x.dim() == 3, f"expected 3 dims, got {x.dim()} dims."
            assert x.size(-1) == self.d_model, f"expected {self.d_model}, got {x.size(-1)}"
            
            # Reshape x to [B, d_model, H, W]
            B, num_spatial_patches, _ = x.shape
            H = W = math.isqrt(num_spatial_patches)
            assert (
                H * H == num_spatial_patches and
                W * W == num_spatial_patches
            ), f"num_spatial_patches must be a perfect square, got {num_spatial_patches}"

            x = (
                x.transpose(1, 2) # [B, d_model, H*W]
                .contiguous()
                .view(B, self.d_model, H, W)
            ) # [B, d_model, H, W]
            assert (
                x.shape == (B, self.d_model, H, W)
            ), f"expected: {(B, self.d_model, H, W)}, got {x.shape}"

            # Pass through deconvolution blocks
            x = self.activation(self.batch_norm1(self.deconv1(x)))
            x = self.activation(self.batch_norm2(self.deconv2(x)))

            # Final deconvolution block
            x = self.sigmoid(self.deconv3(x)) # [B, C_out, H, W]

            assert x.dim() == 4, f"expected 3 dims, got {x.dim()} dims"
            assert x.size(1) == self.C_out, f"expected {self.C_out}, got {x.size(1)}"

            return x
        
def test_decoder():
    C_out, d_model = 3, 512
    decoder = VQVAEDecoder(C_out, d_model, "relu").to(device)
    B, H, W = 1, 144, 144
    x = torch.randn(B, H*W, d_model).to(device)
    x_out = decoder(x)
    return x_out

if __name__ == "__main__":
    x = test_decoder()
    print(x.shape)
    