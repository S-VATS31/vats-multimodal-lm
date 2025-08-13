from configs.setup_env import device, dtype

from typing import Tuple

from torch.utils.data import DataLoader

from configs.transformers.vision.vit_3d.model_args.model_args_large import ModelArgs
from configs.transformers.vision.vit_3d.training_args import TrainingArgs

def setup_loaders(
    model_args: ModelArgs,
    training_args: TrainingArgs
) -> Tuple[DataLoader, DataLoader]:
    """Set up training and validation data loaders.
    
    Args:
        model_args (ModelArgs): Model hyperparameters.
        training_args (TrainingArgs): Training hyperparameters.
    
    Returns:
        Tuple:
            - DataLoader: Training data loader.
            - DataLoader: Validation data loader.
    """
    pass