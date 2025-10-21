from typing import Tuple
from dataclasses import dataclass

@dataclass
class TrainingArgs:
    """Dataclass containing model training arguments."""
    learning_rate: float = 6e-4
    batch_size: int = 32
    epsilon: float = 1e-6
    clip_grad_norm: float = 1.0
    weight_decay: float = 5e-4
    betas: Tuple[float, float] = (0.9, 0.95)
    fused: bool = True
    warmup_ratio: float = 0.05
    aux_loss_weight: float = 0.01
    eta_min: float = 6e-7
    num_cycles: float = 0.5
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    drop_last: bool = True
    grad_accum_steps: int = 4
    logging_tokens_freq: int = 1_000_000_000
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 500
    max_eval_batches: int = 250
    max_skipped_steps: int = 1000
    max_train_tokens: int = 20_000_000_000
    clear_cache_freq: int = 500_000_000
