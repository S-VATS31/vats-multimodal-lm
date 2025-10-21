from configs.setup_env import device

import logging
from pathlib import Path
from typing import Optional, Dict, Union

import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from utils.setup_logger import setup_logger
from configs.transformers.nlp.training_args import TrainingArgs
from configs.transformers.nlp.model_args.model_args_medium import ModelArgs

# Set up logger
checkpoint_logger = setup_logger(
    name="checkpoint_logger", 
    log_file="checkpoints.log", 
    level=logging.INFO
)

def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    tokens_seen: int,
    loss: float,
    training_args: TrainingArgs,
    model_args: ModelArgs,
    scaler: Optional[GradScaler] = None,
    is_best: bool = False,
    checkpoints_dir: Path = Path("nlp_checkpoints")
) -> str:
    """Save checkpoint to .pt file.

    Args:
        model (nn.Module): Transformer architecture.
        optimizer (Optimizer): PyTorch optimizer.
        scheduler: PyTorch scheduler.
        tokens_seen (int): Number of tokens seen so far.
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
            'tokens_seen': tokens_seen,
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
        filename = "best_model.pt" if is_best else f"checkpoint_tokens_seen_{tokens_seen}.pt"
        
        # Load checkpoint data to filename
        save_path = checkpoints_dir / filename
        torch.save(checkpoint_data, save_path)
        checkpoint_logger.info(f"Succesfully saved checkpoint to {filename}")
        
        return str(save_path)

    except Exception as e:
        checkpoint_logger.error(f"Failed to save checkpoint as {filename}: {e}")
        raise # We don't want to load faulty checkpoints, so no return

def load_checkpoint(
    filename: str,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    scaler: Optional[GradScaler] = None
) -> Dict[str, Union[int, float, dict]]:
    """Load checkpoint from saved .pt file.
    
    Args:
        filename (str): Filename where checkpoint is saved.
        model (nn.Module): Transformer architecture.
        optimizer (Optimizer): PyTorch optimizer.
        scheduler (LambdaLR): PyTorch scheduler.
        scaler (Optional[GradScaler]): Gradient scaling for bf16/fp16 gradients.

    Returns:
        Dict[str, Union[int, float, dict]]:
            - Dict[str, int]: Number of tokens seen so far.
            - Dict[str, float]: Loss based on training.
            - Dict[str, dict]: Training arguments and model arguments.
    """
    try:
        # Load checkpoint
        checkpoint = torch.load(filename, map_location=device)
        
        # Load states
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load scaler state dict if using AMP
        if scaler is not None and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        checkpoint_logger.info(f"Succesfully loaded checkpoint from {filename}")
        
        return {
            'tokens_seen': checkpoint['tokens_seen'],
            'loss': checkpoint['loss'],
            'training_args': checkpoint['training_args'],
            'model_args': checkpoint['model_args'],
        }
        
    except Exception as e:
        checkpoint_logger.error(f"Failed to load checkpoint from {filename}: {e}")
        raise
