import logging
from typing import Optional
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import GradScaler

from configs.transformers.nlp.training_args import TrainingArgs
from configs.transformers.nlp.model_args.model_args_medium import ModelArgs
from utils.setup_logger import setup_logger

# Set up logger
checkpoint_logger = setup_logger(name="checkpoint_logger", log_file="checkpoints.log", level=logging.INFO)

def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    epoch: int,
    step: int,
    loss: float,
    training_args: TrainingArgs,
    model_args: ModelArgs,
    scaler: Optional[GradScaler] = None,
    is_best: bool = False,
    checkpoints_dir: Path = Path("checkpoints")
) -> str:
    """Save checkpoint to .pt file.

    Args:
        model (nn.Module): Transformer architecture.
        optimizer (Optimizer): PyTorch optimizer.
        scheduler: PyTorch scheduler.
        epoch (int): Current epoch to save checkpoint to.
        step (int): Current step to save checkpoint to.
        loss (float): Current loss to save checkpoint to.
        training_args (TrainingArgs): Training hyperparameters.
        model_args (ModelArgs): Model hyperparameters.
        scaler (Optional[GradScaler]): Save if GradScaler is not None.
        is_best (bool): Whether the current checkpoint contains the lowest validation loss or not.
        checkpoints_dir (str): Directory to where checkpoints will be saved.

    Returns:
        str: Returns path to save checkpoint so it can be loaded later.
    """
    checkpoints_dir.mkdir(exist_ok=True)
    try:
        # Create checkpoint data
        checkpoint_data = {
            'epoch': epoch,
            'step': step,
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
        filename = "best_model.pt" if is_best else f"checkpoint_step_{step}_epoch{epoch}.pt"
        
        # Load checkpoint data to filename
        save_path = checkpoints_dir / filename
        torch.save(checkpoint_data, save_path)
        checkpoint_logger.info(f"Succesfully saved checkpoint to {filename}")
        
        return str(save_path)

    except Exception as e:
        checkpoint_logger.error(f"Failed to save checkpoint as {filename}: {e}")
        raise # We don't want to load faulty checkpoints, so no return
