from configs.setup_env import device, dtype

from typing import Tuple

import torch
import torch.nn as nn
from torch.amp import autocast

from src.autoregressive_image_gen.vq_vae.encoder.vae_encoder import VQVAEEncoder
from src.autoregressive_image_gen.vq_vae.decoder.vae_decoder import VQVAEDecoder
from src.autoregressive_image_gen.vq_vae.quantizer.vector_quantization import VectorQuantizer
from configs.autoregressive_image_gen.autoregressive_transformer.model_args.model_args_xsmall import ModelArgs

class VQVAE(nn.Module):
    def __init__(self, model_args: ModelArgs):
        super().__init__()

        self.encoder = VQVAEEncoder(
            C_in=model_args.C_in_out,
            d_model=model_args.d_model,
            activation_func=model_args.vae_encoder_activation
        ).to(device)

        self.quantizer = VectorQuantizer(
            d_model=model_args.d_model,
            num_embeddings=model_args.num_embeddings,
            commitment_beta=model_args.commitment_beta
        ).to(device)

        self.decoder = VQVAEDecoder(
            C_out=model_args.C_in_out,
            d_model=model_args.d_model,
            activation_func=model_args.vae_encoder_activation
        ).to(device)

    def forward(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of VQ-VAE layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W].
        
        Returns:
            Tuple:
                - torch.Tensor: Output tensor (reconstruction of input tensor).
                - torch.Tensor: Total loss.
                - torch.Tensor: Codebook indices.
        """
        with autocast(device_type=device.type, dtype=dtype):
            z = self.encoder(x)
            z_q, loss, encoding_indices = self.quantizer(z)
            x_reconstructed = self.decoder(z_q)
            return x_reconstructed, loss, encoding_indices

def main():
    model_args = ModelArgs()
    model = VQVAE(model_args).to(device)
    B, C, H, W = 1, model_args.C_in_out, 144, 144
    x = torch.randn(B, C, H, W).to(device)
    x_reconstructed, loss, encoding_indices = model(x)
    return x, x_reconstructed, loss, encoding_indices

if __name__ == "__main__":
    x, x_reconstructed, loss, encoding_indices = main()
    print(x[0, 0, 0, 0])
    print(x_reconstructed[0, 0, 0, 0])
    print(x.shape)
    print(x_reconstructed.shape)
    print(loss)
    print(encoding_indices.shape)
    