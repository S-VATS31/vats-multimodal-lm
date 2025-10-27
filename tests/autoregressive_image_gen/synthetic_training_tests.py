from configs.setup_env import device, dtype

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from src.autoregressive_image_gen.autoregressive_transformer.model import AutoregressiveImageTransformer
from training.autoregressive_image_gen.autoregressive_transformer.loops.training_loop import ImageGenTrainer
from training.autoregressive_image_gen.autoregressive_transformer.cosine_scheduler import cosine_with_warmup_scheduler
from training.autoregressive_image_gen.autoregressive_transformer.setup_training_components import setup_training_components

class SyntheticDataset(Dataset):
    """Synthetic dataset creation for testing."""
    def __init__():
        pass
