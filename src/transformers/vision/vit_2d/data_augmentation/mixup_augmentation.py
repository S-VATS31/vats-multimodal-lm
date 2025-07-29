from configs.setup_env import device

import torch

def mixup_data(
    images: torch.Tensor,
    targets: torch.Tensor,
    alpha: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply mixup augmentation to training data as a form of regularization.

    Args:
        images (torch.Tensor): Input images of shape [B, C, H, W].
        targets (torch.Tensor): Target labels of shape [B].
        alpha (float): Mixup hyperparameter controlling the strength of mixing.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
            - torch.Tensor: Mixed images.
            - torch.Tensor: Original targets.
            - torch.Tensor: Permuted targets.
            - float: Mixing coefficient.
    """
    # No mixup applied, return original images
    if alpha == 0:
        return images, targets, targets, 1.0

    # Sample lambda from Beta distribution
    lam = torch.distributions.beta.Beta(alpha, alpha).sample().item()
    batch_size = images.size(0)

    # Permuted numbers from 0 to B-1
    index = torch.randperm(batch_size).to(device)

    # Compute weighted sum
    mixed_images = lam * images + (1 - lam) * images[index]

    # Original labels, mixed labels
    targets_a, targets_b = targets, targets[index]
    return mixed_images, targets_a, targets_b, lam
