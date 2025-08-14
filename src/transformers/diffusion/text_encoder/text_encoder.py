from configs.setup_env import device, dtype

from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

from src.optimized_attention import RoPE
from src.rms_norm import RMSNorm
from src.swiglu_activation import SwiGLUActivation
from src.ffn_block import FFNBlock

