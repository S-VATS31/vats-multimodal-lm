from typing import Tuple
from dataclasses import dataclass

# TODO: Update training args to specialize in 3D video transformers.

@dataclass
class TrainingArgs:
    """Dataclass containing model training arguments.
    
    Args:
        learning_rate (float): Hyperparameter controlling learning rate.
        epochs (int): Number of passes through the total dataset.
        batch_size (int): Number of examples being processed during each step.
        epsilon (float): Epsilon value for AdamW optimizer.
        clip_grad_norm (float): Maximum norm to clip gradients to.
        weight_decay (float): Weight decay for AdamW optimizer.
        betas (Tuple[float, float]): Tuple of beta value for AdamW optimizer.
        fused (bool): Fused bool for AdamW optimizer.
        warmup_ratio (float): Ratio of steps that are considered as warmup steps.
        aux_loss_weight (float): Auxiliary loss weight to compute total loss.
        eta_min (float): Minimum learning rate for cosine scheduler.
        label_smoothing (float): Label smoothing parameter for computing loss.
        num_cycles (float): Number of cosine cycles completed during training.
        num_workers (int): Number of workers to load data.
        pin_memory (bool): Whether to pin memory or not.
        persistent_workers (bool): Whether to keep workers alive through epochs.
        drop_last (bool): Whether the final batch (smaller than the actual batch size) is used or dropped.
        train_ratio (float): Percentage of training examples.
        grad_accum_steps (int): Gradient accumulation steps to get a larger effective batch size.
        TODO: add the rest of the hyperparameters.
    """
    learning_rate: float = 2e-4
    epochs: int = 3
    batch_size: int = 256
    epsilon: float = 1e-6
    clip_grad_norm: float = 1.0
    weight_decay: float = 5e-4
    betas: Tuple[float, float] = (0.9, 0.95)
    fused: bool = True
    warmup_ratio: float = 0.05
    aux_loss_weight: float = 0.01
    eta_min: float = 6e-7
    label_smoothing: float = 0.1
    num_cycles: float = 0.5
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    drop_last: bool = True
    train_ratio: float = 0.95  # TODO: deprecate this or actually use it 
    grad_accum_steps: int = 4

    # TODO: actually use this for logging or remove
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 500
    max_eval_batches: int = 250
