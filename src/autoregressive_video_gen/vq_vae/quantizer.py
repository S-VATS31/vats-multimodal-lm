from configs.setup_env import device, dtype

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast


class VectorQuantizer(nn.Module):
    """Vector quantization for autoencoder with stop-gradient handling.
    
    Args:
        d_model (int): Dimensionality of model embeddings.
        num_embeddings (int): Number of embeddings for embedding vector.
        commitment_beta (float): Commitment weight to compute loss.
    """
    def __init__(
        self,
        d_model: int,
        num_embeddings: int,
        commitment_beta: float
    ):
        super().__init__()

        self.d_model = d_model
        self.num_embeddings = num_embeddings
        self.commitment_beta = commitment_beta

        # Create embeddings of size d_model
        self.embedding = nn.Embedding(num_embeddings, d_model)
        nn.init.uniform_(
            self.embedding.weight, -1.0 / num_embeddings, 1.0 / num_embeddings
        )
            
    def forward(
        self, 
        z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with autocast(device_type=device.type, dtype=dtype):
            """Forward pass of vector quantizer.
            
            Args:
                z (torch.Tensor): Encoder output of shape [B, T_frames, H, W, d_model].

            Returns:
                Tuple:
                    - torch.Tensor: Straight-through estimator for backpropagation.
                    - torch.Tensor: Reconstruction loss.
                    - torch.Tensor: Encoding indices.
            """
            assert z.dim() == 5, f"expected 5 dims, got {z.dim()} dims"
            assert (
                z.size(-1) == self.d_model
            ), f"expected {self.d_model}, got {z.size(-1)}"

            B, T_frames, H, W, _ = z.shape

            # Flatten z to [B*T*H*W, d_model]
            z = z.view(-1, self.d_model)

            assert (
                z.shape == (B*T_frames*H*W, self.d_model)
            ), f"expected {(B*T_frames*H*W, self.d_model)}, got {z.shape}"

            # matmul
            assert (
                z.size(-1) == self.embedding.weight.t().size(0)
            ), f"{z.size(-1)} must be equal to {self.embedding.weight.t().size(0)} for matmul"

            # z: [B*T*H*W, d_model]
            # embedding.t(): [d_model, num_embeddings]
            # distances: [B*T*H*W, num_embeddings]
            distances = (
                torch.sum(z**2, dim=-1, keepdim=True) + 
                torch.sum(self.embedding.weight**2, dim=-1) - 
                2 * torch.matmul(z, self.embedding.weight.t())
            )

            assert (
                distances.shape == (B*T_frames*H*W, self.num_embeddings)
            ), f"expected {(B*T_frames*H*W, self.num_embeddings)}, got {distances.shape}"


            encoding_indices = torch.argmin(distances, dim=-1) # non-differentiable step
            z_q = self.embedding(encoding_indices) # [B*T*H*W, d_model]

            assert (
                z_q.shape == (B*T_frames*H*W, self.d_model)
            ), f"expected {(B*T_frames*H*W, self.d_model)}, got {z_q.shape}"

            # Compute loss
            codebook_loss = F.mse_loss(z_q.detach(), z)
            commitment_loss = F.mse_loss(z_q, z.detach())
            total_loss = codebook_loss + commitment_loss * self.commitment_beta

            z_q = z + (z_q - z).detach() # straight through estimator

            return (
                z_q.view(B, T_frames, H, W, self.d_model), 
                total_loss, 
                encoding_indices.view(B, T_frames, H, W)
            )
        
def test_vector_quantization():
    d_model, num_embeddings, commitment_beta = 512, 256, 0.25
    quantizer = VectorQuantizer(d_model, num_embeddings, commitment_beta).to(device)
    B, T_frames, H, W = 1, 16, 32, 32
    z = torch.randn(B, T_frames, H, W, d_model).to(device)
    ste, loss, indices = quantizer(z)
    return ste, loss, indices

if __name__ == "__main__":
    ste, loss, indices = test_vector_quantization()
    print(ste.shape)
    print(loss)
    print(indices.shape)
