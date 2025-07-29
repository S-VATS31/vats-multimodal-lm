from configs.setup_env import device

import math
import logging
from typing import Tuple

import torch

# Set up logger
from utils.setup_logger import setup_logger
cutmix_logger = setup_logger(name="cutmix_logger", log_file="training.log", level=logging.INFO)

def cutmix_data(
    images: torch.Tensor,
    targets: torch.Tensor,
    alpha: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply cutmix augmentation to training data as a form of regularization.

    Args:
        images (torch.Tensor): Input tensor of shape [B, C, H, W].
        targets (torch.Tensor): Target labels of shape [B].
        alpha (float): CutMix hyperparameter controlling the strength of mixing.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
            - torch.Tensor: Mixed images.
            - torch.Tensor: Original targets.
            - torch.Tensor: Permuted targets.
            - float: Mixing coefficient.
    """
    # No CutMix applied, return original images
    if alpha == 0:
        return images, targets, targets, 1.0

    # Get batch size, height, and width from images tensor
    B, _, H, W = images.shape
    if H != W:
        cutmix_logger.error(f"H, W must be equal, got {H} != {W}")

    # Beta distribution (single sample) to get lambda
    lam = torch.distributions.beta.Beta(alpha, alpha).sample().item()

    # Permuted numbers from 0 to B-1
    index = torch.randperm(B).to(device) # [B,]

    # Cutting ratios
    cut_ratio = math.sqrt(1.0 - lam)
    W_cut = int(cut_ratio * W)
    H_cut = int(cut_ratio * H)

    # Uniform center
    cx = torch.randint(W, (1,)).item() # x-coordinate (0, W-1)
    cy = torch.randint(H, (1,)).item() # y-coordinate (0, H-1)

    # Calculate box boundaries
    x1 = max(cx - W_cut // 2, 0)
    y1 = max(cy - H_cut // 2, 0)
    x2 = min(cx + W_cut // 2, W)
    y2 = min(cy + H_cut // 2, H)

    # Create mixed images
    mixed_images = images.clone()
    mixed_images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]

    # Update lambda
    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))

    # Get targets
    targets_a, targets_b = targets, targets[index]
    return mixed_images, targets_a, targets_b, lam
