from typing import Optional
import logging
import os

import torch
import torch.nn as nn
from torch.amp import GradScaler

from configs.transformers.vision.vit_2d.training_args import TrainingArgs
from configs.transformers.vision.vit_2d.model_args.model_args_large import ModelArgs

# Set up logger
from utils.setup_logger import setup_logger
checkpoint_logger = setup_logger(name="checkpoint_logger", log_file="checkpoints.log", level=logging.INFO)

def save_checkpoint(
    model: nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    loss: float,
    training_args: TrainingArgs,
    model_args: ModelArgs,
    scaler: Optional[GradScaler] = None,
    is_best: bool = False,
    checkpoint_dir: str = "model_checkpoints"
) -> str:
    """Save checkpoint to .pth file.
    
    Args:
        model (nn.Module): Transformer architecture.
        optimizer: PyTorch optimizer.
        scheduler: PyTorch scheduler.
        epoch (int): Current epoch to save checkpoint to.
        loss (float): Current loss to save checkpoint to.
        training_args (TrainingArgs): Training hyperparameters.
        model_args (ModelArgs): Model hyperparameters.
        scaler (Optional[GradScaler]): Save if GradScaler is not None.
        is_best (bool): Whether the current checkpoint contains the lowest validation loss or not.

    Returns:
        str: Returns path to save checkpoint so it can be loaded later.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    try:
        # Create checkpoint data
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
            'training_args': training_args.__dict__,
            'model_args': model_args.__dict__,
        }

        # Add scaler state if using AMP
        if scaler is not None:
            checkpoint_data['scaler_state_dict'] = scaler.state_dict()
        
        # Create filename
        filename = "best_model.pth" if is_best else f"checkpoint_epoch_{epoch}.pth"
        
        # Load checkpoint data to filename
        save_path = os.path.join(checkpoint_dir, filename)

        torch.save(checkpoint_data, save_path)
        checkpoint_logger.info(f"Succesfully saved checkpoint to {filename}")
        
        return save_path

    except Exception as e:
        checkpoint_logger.error(f"Failed to save checkpoint as {filename}: {e}")
        raise