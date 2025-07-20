from typing import Optional, Dict, Union
import logging

import torch
import torch.nn as nn
from torch.amp import GradScaler

# Set up logger
from utils.setup_logger import setup_logger
checkpoint_logger = setup_logger(name="checkpoint_logger", log_file="checkpoints.log", level=logging.INFO)

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
    