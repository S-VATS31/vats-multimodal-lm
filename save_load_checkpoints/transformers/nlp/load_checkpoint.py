from configs.transformers.nlp.setup_env import device

from typing import Optional, Dict, Union

import torch # TODO: remove this when device (global var) is imported
import torch.nn as nn
from torch.amp import GradScaler

# TODO: set up and import logger

def load_checkpoint(
    filename: str,
    model: nn.Module,
    optimizer,
    scheduler,
    scaler: Optional[GradScaler] = None,
    device: torch.device = None,
) -> Dict[str, Union[int, float]]:
    """Load checkpoint from saved .pt file.
    
    Args:
        filename (str): Filename where checkpoint is saved.
        model (nn.Module): Transformer architecture.
        optimizer: PyTorch optimizer.
        scheduler: PyTorch scheduler.
        scaler (Optional[GradScaler]): Gradient scaling for bf16/fp16 gradients.
        device (torch.device): Accelerator at use.

    Returns:
        Dict[str, Union[int, float]]: State dict returning current step, epoch, and loss.
            - int: Current epoch.
            - int: Current step.
            - float: Current loss.
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
        
        logger.info(f"Succesfully loaded checkpoint from {filename}")
        
        return {
            'epoch': checkpoint['epoch'],
            'step': checkpoint['step'],
            'loss': checkpoint['loss'],
        }
        
    except Exception as e:
        logger.error(f"Failed to load checkpoint from {filename}: {e}")
        raise
