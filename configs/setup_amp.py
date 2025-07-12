import contextlib
from typing import ContextManager

import torch

def device_specific_amp(device: torch.device, dtype: torch.dtype) -> ContextManager[None]:
    """Setup function to ensure bfloat16 if device is CUDA or XLA. This ensures
    code is not duplicated for different devices.
    
    Args:
        device (torch.device): PyTorch accelerator.
        dtype (torch.dtype): Data type tensors will be casted to.
    
    returns:
        ContextManager[None]: Returns Autocast mode based on the device being used.
    """
    # Apply normal AMP for CUDA/CPU
    if device.type == "cuda" or device.type == "cpu":
        return torch.amp.autocast(device_type=device.type, dtype=dtype)
    # Apply bfloat16 for XLA
    else:
        return contextlib.nullcontext()
    