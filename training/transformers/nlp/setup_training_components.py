from configs.setup_env import device

from typing import Tuple, Optional

import torch.nn as nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR

from torch.amp import GradScaler

from configs.transformers.nlp.training_args import TrainingArgs
from training.transformers.nlp.cosine_scheduler import cosine_with_warmup_scheduler

def setup_training_components(
    model: nn.Module,
    training_args: TrainingArgs,
    num_training_steps: int,
) -> Tuple[Optimizer, LambdaLR, Optional[GradScaler]]:
    """Setup optimizer, scheduler, and scaler for training.
    
    Args:
        model (nn.Module): Transformer architecture.
        training_args (TrainingArgs): Training hyperparameters.
        num_training_steps (int): Number of training steps.

    Returns:
        Tuple[Optimizer, LambdaLR, Optional[GradScaler]]:
            - Optimizer: AdamW optimizer.
            - LambdaLR: Custom cosine decay lr scheduler.
            - Optional[GradScaler]: Gradient scaling for bf16/fp16 gradients.
    """
    # Setup optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=training_args.learning_rate,
        betas=training_args.betas,
        eps=training_args.epsilon,
        weight_decay=training_args.weight_decay,
        fused=training_args.fused
    )

    # Setup scheduler
    num_warmup_steps = int(training_args.warmup_ratio * num_training_steps)
    scheduler = cosine_with_warmup_scheduler(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=training_args.num_cycles
    )
    
    # Setup gradient scaler for AMP
    scaler = GradScaler() if device.type == "cuda" else None
    
    return optimizer, scheduler, scaler
