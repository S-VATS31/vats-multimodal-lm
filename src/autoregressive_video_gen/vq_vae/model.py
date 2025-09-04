from configs.setup_env import device, dtype

from typing import Tuple

import torch
import torch.nn as nn
from torch.amp import autocast

from src.autoregressive_video_gen.vq_vae.encoder import Encoder3D
from src.autoregressive_video_gen.vq_vae.decoder import Decoder3D
from src.autoregressive_video_gen.vq_vae.quantizer import VectorQuantizer
from configs.autoregressive_video_gen.autoregressive_transformer.model_args.model_args_large import ModelArgs

class VQVAE3D(nn.Module):
    def __init__(self, model_args: ModelArgs):
        super().__init__()

        self.encoder = Encoder3D(
            d_model=model_args.d_model, 
            C_in= model_args.C_in_out, 
            patch_size=model_args.patch_size
        )
        self.decoder = Decoder3D(
            d_model=model_args.d_model,
            C_out=model_args.C_in_out,
            patch_size=model_args.patch_size
        )
        self.quantizer = VectorQuantizer(
            d_model=model_args.d_model,
            num_embeddings=model_args.num_embeddings,
            commitment_beta=model_args.commitment_beta
        )

    def forward(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of 3D VQ VAE.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C_in, T_frames, H, W].

        Returns:
            Tuple:
                - torch.Tensor: Reconstructed input tensor.
                - torch.Tensor: Total loss.
                - torch.Tensor: Encoding indices.
        """
        with autocast(device_type=device.type, dtype=dtype):
            z = self.encoder(x)
            z_q, loss, encoding_indices = self.quantizer(z)
            x_reconstructed = self.decoder(z_q)
            return x_reconstructed, loss, encoding_indices
        
def test_model():
    model_args = ModelArgs()
    model = VQVAE3D(model_args).to(device)
    B, C, T, H, W = 1, model_args.C_in_out, 8, 16, 16
    x = torch.randn(B, C, T, H, W).to(device)
    x_recon, loss, indices = model(x)
    return x_recon, loss, indices

if __name__ == "__main__":
    x_recon, loss, indices = test_model()
    print(x_recon.shape)
    print(loss)
    print(indices.shape)
    