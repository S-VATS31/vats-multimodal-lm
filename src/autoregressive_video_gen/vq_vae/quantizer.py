from configs.setup_env import device, dtype

import torch
import torch.nn as nn
from torch.amp import autocast

class VectorQuantizer(nn.Module):
    def __init__(self):
        super().__init__()

        pass

    def forward(z: torch.Tensor) -> torch.Tensor:
        with autocast(device_type=device.type, dtype=dtype):
            pass
