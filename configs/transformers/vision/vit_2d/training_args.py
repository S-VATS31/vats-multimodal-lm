from dataclasses import dataclass
from typing import Tuple

@dataclass
class TrainingArgs:
    """Dataclass containing model hyperparameters."""
    learning_rate: float = 2e-4
    epochs: int = 300
    batch_size: int = 256
    num_classes: int = 1000
    epsilon: float = 1e-6
    max_norm: float = 1.0
    weight_decay: float = 5e-4
    betas: Tuple[float, float] = (0.9, 0.95)
    fused: bool = True
    warmup_epochs: int = 50
    eta_min: float = 6e-7
    save_checkpoint_freq: int = 1
    mixup_alpha: float = 0.8
    cutmix_alpha: float = 1.0
    label_smoothing: float = 0.1
    random_erasing_prob: float = 0.4
    color_jitter: float = 0.4
    auto_augment: bool = True
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    grad_accum_steps: int = 4
    