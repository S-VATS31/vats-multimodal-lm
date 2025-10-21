from configs.setup_env import device, dtype

from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch.amp import autocast

from src.autoregressive_image_gen.autoregressive_transformer.model import AutoregressiveImageTransformer
from src.autoregressive_image_gen.vq_vae.encoder.vae_encoder import VQVAEEncoder
from src.autoregressive_image_gen.vq_vae.decoder.vae_decoder import VQVAEDecoder
from src.autoregressive_image_gen.vq_vae.quantizer.vector_quantization import VectorQuantizer
from configs.autoregressive_image_gen.autoregressive_transformer.model_args.model_args_xsmall import ModelArgs

class VQVAE(nn.Module):
    def __init__(self, model_args: ModelArgs):
        super().__init__()

        self.model = AutoregressiveImageTransformer(model_args).to(device)

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
        input_ids: torch.Tensor,
        text_embeddings: torch.Tensor,
        image_attention_mask: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of VQ-VAE layer.
        
        Args:
            input_ids (torch.Tensor): Input tensor of shape [B, C, H, W].
        
        Returns:
            Tuple:
                - torch.Tensor: Output tensor (reconstruction of input tensor).
                - torch.Tensor: Total loss.
                - torch.Tensor: Codebook indices.
                - torch.Tensor: Output of the transformer model.
        """
        import torch.nn.functional as F

        with autocast(device_type=device.type, dtype=dtype):
            z = self.encoder(input_ids)
            z_q, loss, encoding_indices = self.quantizer(z)
            # Downsample image_attention_mask to match latent H', W'
            if image_attention_mask is not None:
                B, H, W = input_ids.shape[0], input_ids.shape[2], input_ids.shape[3]
                H_down, W_down = z.shape[1], z.shape[2]
                mask = image_attention_mask.view(B, 1, H, W).float()
                mask_down = F.interpolate(mask, size=(H_down, W_down), mode='nearest')
                image_attention_mask = mask_down.view(B, H_down * W_down).bool()
            # Model forward
            transformer_out = self.model(
                encoding_indices=encoding_indices,
                text_embeddings=text_embeddings,
                use_cache=use_cache,
                causal_padding_mask=image_attention_mask,
                cross_padding_mask=text_attention_mask
            )
            x_reconstructed = self.decoder(z_q)
            return x_reconstructed, loss, encoding_indices, transformer_out

def main():
    model_args = ModelArgs()
    model = VQVAE(model_args).to(device)
    T_tokens = 9
    B, C, H, W = 1, model_args.C_in_out, 144, 144
    x = torch.randn(B, C, H, W).to(device)
    text_embeddings = torch.randn(B, T_tokens, model_args.d_model).to(device)
    image_mask = torch.randint(
        0, 2, (B, H*W), dtype=torch.bool
    ).to(device)
    text_mask = torch.randint(
        0, 2, (B, T_tokens), dtype=torch.bool
    ).to(device)
    x_reconstructed, loss, encoding_indices, transformer_out = model(
        x, text_embeddings, image_mask, text_mask, use_cache=False
    )
    return x, x_reconstructed, loss, encoding_indices, transformer_out

if __name__ == "__main__":
    x, x_reconstructed, loss, encoding_indices, transformer_out = main()
    print(x[0, 0, 0, 0])
    print(x_reconstructed[0, 0, 0, 0])
    print(x.shape)
    print(x_reconstructed.shape)
    print(loss)
    print(encoding_indices.shape)
