from dataclasses import dataclass
from typing import Tuple

# TODO: check all .../training_args.py to check if docstring matches args

@dataclass
class TrainingArgs:
    """Dataclass containing model training arguments
    
    Args:
        learning_rate (float): Hyperparameter controlling learning rate.
        epochs (int): Number of passes through the total dataset.
        batch_size (int): Number of examples being processed during each step.
        epsilon (float): Epsilon value for AdamW optimizer.
        max_norm (float): Maximum norm to clip gradients to.
        weight_decay (float): Weight decay for AdamW optimizer.
        betas (Tuple[float, float]): Tuple of beta value for AdamW optimizer.
        fused (bool): Fused bool for AdamW optimizer.
        warmup_epochs (int): Number of epochs that are warmup.
        eta_min (float): Minimum learning rate for cosine scheduler.
        save_checkpoint_freq (int): Regular checkpoint to save every N epochs.
        mixup_alpha (float): Alpha hyperparameter for mixup augmentation.
        cutmix_alpha (float): Alpha hyperparameter for cutmix augmentation.
        label_smoothing (float): Label smoothing hyperparameter for CE loss.
        random_erasing_prob (float): Probability a portion of the image gets erased.
        color_jitter (float): Regularization to prevent overfitting.
        auto_augment (bool): Automatically sets augmentations fit to dataset, rather than manually setting.
        num_workers (int): Number of workers to load the data.
        pin_memory (bool): Whether to pin memory or not.
        persistent_workers (bool): Whether workers are kept alive over epochs.
        grad_accum_steps (int): Gradient accumulation steps for a larger effective batch size.
    """
    learning_rate: float = 2e-4
    epochs: int = 300
    batch_size: int = 256
    epsilon: float = 1e-6
    max_norm: float = 1.0
    weight_decay: float = 5e-4
    betas: Tuple[float, float] = (0.9, 0.95)
    fused: bool = True
    warmup_epochs: int = 50
    eta_min: float = 6e-7
    save_checkpoint_freq: int = 1
    mixup_alpha: float = 0.8
    cutmix_alpha: float = 0.8
    label_smoothing: float = 0.1
    random_erasing_prob: float = 0.4
    color_jitter: float = 0.4
    auto_augment: bool = True
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    grad_accum_steps: int = 4
