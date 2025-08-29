from configs.setup_env import device, dtype

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

class VectorQuantizer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_embeddings: int,
        commitment_beta: float,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_embeddings = num_embeddings
        self.commitment_beta = commitment_beta

        # [num_embeddings, d_model]
        self.embedding = nn.Embedding(num_embeddings, d_model)
        nn.init.uniform_(self.embedding.weight, -1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(
        self, 
        z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of Vector quantizer.
        
        Args:
            z (torch.Tensor): Encoder output of shape [B, H*W, d_model].

        Returns:
            Tuple:
                - torch.Tensor: Staight-through estimator.
                - torch.Tensor: Total loss.
                - torch.Tensor: Encoding indices.
        """
        with autocast(device_type=device.type, dtype=dtype):
            assert z.dim() == 3, f"z must have 3 dims, got {z.dim()} dims"
            assert z.size(-1) == self.d_model, f"expected {self.d_model}, got {z.size(-1)}"

            # Flatten to [B*H*W, d_model]
            B, num_spatial_patches, _ = z.shape
            z = z.view(-1, self.d_model)

            assert (
                z.shape == (B*num_spatial_patches, self.d_model)
            ), f"expected: {(B*num_spatial_patches, self.d_model)}, got {z.shape}"

            assert (
                self.embedding.weight.t().size(0) == z.size(-1)
            ), "must be equal for matrix multiplication."

            # Compute distances as following:
            # distance_ij = ||z_i-e_j||_2^2
            # distance_ij = ||z_i||^2 - ||e_j||^2 -2 * z_i @ e_j
            # Shape: [B*H*W, num_embeddings]
            distances = (
                torch.sum(z**2, dim=1, keepdim=True)
                + torch.sum(self.embedding.weight**2, dim=1)
                - 2 * torch.matmul(z, self.embedding.weight.t())
            )

            assert (
                distances.shape == (B*num_spatial_patches, self.num_embeddings)
            ), f"expected {(B*num_spatial_patches, self.num_embeddings)}, got {distances.shape}"

            # Get encoding indices
            encoding_indices = torch.argmin(distances, dim=-1) # [B*H*W]
            z_q = self.embedding(encoding_indices) # [B*H*W, d_model]

            assert (
                encoding_indices.shape == (B*num_spatial_patches,)
            ), f"expected {(B*num_spatial_patches,)}, got {encoding_indices.shape}"
            assert (
                z_q.shape == (B*num_spatial_patches, self.d_model)
            ), f"expected {(B*num_spatial_patches, self.d_model)}, got {z_q.shape}"
            
            # Reshape to input shape
            z_q = z_q.view_as(z) # [B*H*W, d_model]
            assert (
                z_q.shape == (B*num_spatial_patches, self.d_model)
            ), f"expected {(B*num_spatial_patches, self.d_model)}, got {z_q.shape}"

            # Compute total loss
            codebook_loss = F.mse_loss(z_q.detach(), z)
            commit_loss = F.mse_loss(z_q, z.detach())
            total_loss = codebook_loss + self.commitment_beta * commit_loss

            # Get straight through estimator for backprop
            z_q = z + (z_q - z).detach()

            return z_q, total_loss, encoding_indices.view(z.size(0), -1)

def test_vq():
    num_embeddings, d_model, commitment_beta = 256, 512, 0.7
    vq = VectorQuantizer(d_model, num_embeddings, commitment_beta).to(device)
    B, H, W = 1, 144, 144
    z = torch.randn(B, H*W, d_model).to(device)
    z_q, loss, encoding_indices = vq(z)
    return z_q, loss, encoding_indices 

if __name__ == "__main__":
    z_q, loss, encoding_indices = test_vq()
    print(z_q.shape)
    print(loss)
    print(encoding_indices.shape)
    