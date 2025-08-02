from configs.setup_env import device, dtype

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.transformers.vision.vit_3d.training_args import TrainingArgs

def validate(
    model: nn.Module,
    val_loader: DataLoader,
    training_args: TrainingArgs,
) -> Tuple[float, float]:
    """Test Video Transformer on validation data.

    Args:
        model (nn.Module): Video Transformer.
        val_loader (DataLoader): DataLoader containing validation examples.
        training_args (TrainingArgs): Training hyperparameters.

    Returns:
        Tuple[float, float]: (avg_val_loss, val_acc)
    """
    model.eval() # Set modal to evaluation

    # Initialize
    correct = 0
    total = 0
    val_loss = 0.0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating") # Set up dataloader

        # Validatoin loop
        for images, targets in pbar:
            images, targets = images.to(device), targets.to(device)

            with autocast(device_type=device.type, dtype=dtype):
                logits = model(images) # Get logits
                loss = F.cross_entropy(logits, targets, label_smoothing=training_args.label_smoothing)

            # Get loss and accuracy
            val_loss += loss.item() * targets.size(0)
            predicted = logits.argmax(dim=1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

            acc = 100.0 * correct / total
            avg_loss = val_loss / total
            pbar.set_postfix({"Loss": f"{avg_loss:.4f}", "Acc": f"{acc:.2f}%"})

    # Average loss/accuracy
    avg_val_loss = val_loss / total
    val_acc = 100.0 * correct / total
    return avg_val_loss, val_acc
