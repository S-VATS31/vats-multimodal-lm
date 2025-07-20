from typing import Tuple

import torch

from src.transformers.vision.vit_2d.data_augmentation.cutmix_augmentation import cutmix_data
from src.transformers.vision.vit_2d.data_augmentation.mixup_augmentation import mixup_data

def random_augmentation(
    images: torch.Tensor,
    targets: torch.Tensor, alpha: float,
    device: torch.device = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply mixup/cutmix or no augmentation to training data as a form of regularization.

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
    choices = ["mixup", "cutmix", "none"]

    # Randomly choose data augmentation method
    method = choices[torch.randint(0, len(choices), (1,)).item()]

    # Return randomly selected method
    if method == "mixup":
        return mixup_data(images, targets, alpha, device)
    elif method == "cutmix":
        return cutmix_data(images, targets, alpha, device)
    else:
        return images, targets, targets, 1.0
    