import math

from torch.optim.lr_scheduler import LambdaLR

class WarmupCosineScheduler(LambdaLR):
    """Custom scheduler combining linear warmup with cosine annealing.

    Args:
        optimizer (Optimizer): The optimizer for which to schedule the learning rate.
        warmup_epochs (int): Number of epochs to linearly increase the learning rate.
        total_epochs (int): Total number of training epochs.
        eta_min (float): Minimum learning rate multiplier (used as a fraction of the base LR).
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs, eta_min):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                # Linear warmup from 0 to 1
                return (epoch + 1) / warmup_epochs
            else:
                # Cosine decay from 1 to eta_min
                progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                return eta_min + (1 - eta_min) * cosine_decay

        super().__init__(optimizer, lr_lambda)
