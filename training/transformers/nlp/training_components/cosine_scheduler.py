import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

def cosine_with_warmup_scheduler(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
) -> LambdaLR:
    """Custom cosine decay learning rate scheduler with linear warmup.

    Args:
        optimizer (Optimizer): PyTorch optimizer to udpate weights.
        num_warmup_steps (int): Number of warmup steps.
        num_training_steps (int): Total number of training steps.
        num_cycles (int): Number of cosine waves are completed.
        last_epoch (int): Parameter for custom LambdaLR.

    Returns:
        LambdaLR: Custom cosine decay scheduler.
    """
    def lr_lambda(current_step: int):
        # Linear warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(num_warmup_steps)
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(num_training_steps - num_warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * 2.0 * num_cycles * progress))

    return LambdaLR(optimizer, lr_lambda, last_epoch)
