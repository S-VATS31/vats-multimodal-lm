import logging
from pathlib import Path
from typing import Optional, Dict, Union

import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from configs.transformers.vision.vit_3d.training_args import TrainingArgs
from configs.transformers.vision.vit_3d.model_args.model_args_large import ModelArgs

# Set up logger
from utils.setup_logger import setup_logger
checkpoint_logger = setup_logger(
    name="checkpoint_logger", 
    log_file="checkpoints.log", 
    level=logging.INFO
)

def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    epoch: int,
    loss: float,
    training_args: TrainingArgs,
    model_args: ModelArgs,
    scaler: Optional[GradScaler] = None,
    is_best: bool = False,
    checkpoint_dir: Path = Path("vit_3d_checkpoints")
) -> str:
    """Save checkpoint to .pth file.
    
    Args:
        model (nn.Module): Transformer architecture.
        optimizer (Optimizer): PyTorch optimizer.
        scheduler (LambdaLR): PyTorch scheduler.
        epoch (int): Current epoch to save checkpoint to.
        loss (float): Current loss to save checkpoint to.
        training_args (TrainingArgs): Training hyperparameters.
        model_args (ModelArgs): Model hyperparameters.
        scaler (Optional[GradScaler]): Save if GradScaler is not None.
        is_best (bool): Whether the current checkpoint contains the lowest validation loss or not.

    Returns:
        str: Returns path to save checkpoint so it can be loaded later.
    """
    checkpoint_dir.mkdir(exist_ok=True)
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
        save_path = "best_model.pth" if is_best else f"checkpoint_epoch_{epoch}.pth"

        torch.save(checkpoint_data, checkpoint_dir / save_path)
        checkpoint_logger.info(f"Succesfully saved checkpoint to {save_path}")
        
        return str(save_path)

    except Exception as e:
        checkpoint_logger.error(f"Failed to save checkpoint as {save_path}: {e}")
        raise

def load_checkpoint(
    save_path: str,
    model: nn.Module,
    optimizer,
    scheduler,
    scaler: Optional[GradScaler] = None,
    device: torch.device = None,
) -> Dict[str, Union[int, float]]:
    """Load checkpoint from saved .pt file.
    `
    Args:
        save_path (str): File path where checkpoint is saved.
        model (nn.Module): Transformer architecture.
        optimizer: PyTorch optimizer.
        scheduler: PyTorch scheduler.
        scaler (Optional[GradScaler]): Gradient scaling for bf16/fp16 gradients.
        device (torch.device): Accelerator at use.

    Returns:
        Dict[str, Union[int, float]]: State dict returning current epoch, and loss.
            - int: Current epoch.
            - float: Current loss.
    """
    try:
        # Load checkpoint
        checkpoint = torch.load(save_path, map_location=device)
        
        # Load states
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load scaler state dict if using AMP
        if scaler is not None and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        checkpoint_logger.info(f"Succesfully loaded checkpoint from {save_path}")
        
        return {
            'epoch': checkpoint['epoch'], # int 
            'loss': checkpoint['loss'], # float
        }
            
    except Exception as e:
        checkpoint_logger.error(f"Failed to load checkpoint from {save_path}: {e}")
        raise
    