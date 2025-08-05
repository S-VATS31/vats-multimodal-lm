from typing import Tuple
from dataclasses import dataclass

@dataclass
class TrainingArgs:
    """Dataclass containing model training arguments."""
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
