from configs.setup_env import device, dtype

from typing import Tuple

import torch
import torch.nn as nn
from torch.amp import autocast

from src.autoregressive_video_gen.vq_vae.encoder import Encoder3D
from src.autoregressive_video_gen.vq_vae.decoder import Decoder3D
from src.autoregressive_video_gen.vq_vae.quantizer import VectorQuantizer

class VQVAE3D(nn.Module):
    def __init__(self):
        super().__init__()

        pass

    def forward(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of 3D VQ VAE.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, T, H, W].

        Returns:
            Tuple.
        """
        pass
