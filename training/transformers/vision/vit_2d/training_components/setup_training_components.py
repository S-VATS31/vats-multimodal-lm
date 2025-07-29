from configs.setup_env import device

from typing import Tuple

import torch.nn as nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import GradScaler

from configs.transformers.vision.vit_2d.training_args import TrainingArgs
from training.transformers.vision.vit_2d.training_components.cosine_scheduler import WarmupCosineScheduler

def get_training_components(model: nn.Module, training_args: TrainingArgs) -> Tuple[
    Optimizer, 
    LambdaLR, 
    GradScaler
]:
    """Set up optimizer, learning rate scheduler, and gradient scaler.
    
    Args:
        model (nn.Module): Vision Transformer architecture.
        training_args (TrainingArgs): Training hyperparameters.

    Returns:
        Tuple[
            Optimizer, 
            optim.lr_scheduler.LambdaLR, 
            GradScaler
        ]:
            - Optimizer: AdamW optimizer.
            - LambdaLR: Custom cosine warmup learning rate scheduler.
            - GradScaler: Gradient scaling for bf16/fp16 gradients.
    """
    # Set up AdamW optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=training_args.learning_rate,
        betas=training_args.betas,
        eps=training_args.epsilon,
        weight_decay=training_args.weight_decay,
        fused=training_args.fused,
    )

    # Set up custom CosineScheduler
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_epochs=training_args.warmup_epochs,
        total_epochs=training_args.epochs,
        eta_min=training_args.eta_min,
    )

    # Set up GradScaler
    scaler = GradScaler(device=device.type) if device.type == "cuda" else None

    return optimizer, scheduler, scaler
