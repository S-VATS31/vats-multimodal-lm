from configs.transformers.vision.vit_2d.setup_env import device, dtype

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm

from configs.transformers.vision.vit_2d.training_args import TrainingArgs

def validate(
    model: nn.Module,
    val_loader: DataLoader,
    training_args: TrainingArgs,
) -> Tuple[float, float]:
    """Test Vision Transformer on validation data.

    Args:
        model (nn.Module): Vision Transformer architecture.
        val_loader (DataLoader): DataLoader containing validation examples
        training_args (TrainingArgs): Training hyperparameters.

    Returns:
        Tuple[float, float]: Tuple containing validation accuracy and loss.
            - float: Validation accuracy.
            - float: Validation loss.
    """
    model.eval() # Set model to evaluation

    # Initialize accuracy
    correct = 0
    total = 0
    val_loss = 0.0

    # Turn off gradient calculation for validation set
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating")

        # Validation loop
        for images, targets in pbar:
            images, targets = images.to(device), targets.to(device) # Ensure images, labels on same device

            with torch.amp.autocast(device_type=device.type, dtype=dtype):
                outputs = model(images) # Forward pass
                # Calculate loss, no augmentation for validation
                loss = F.cross_entropy(outputs, targets, label_smoothing=training_args.label_smoothing)

            # Statistics
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Update progress bar
            acc = 100. * correct / total
            avg_loss = val_loss / (total / targets.size(0))
            pbar.set_postfix({"Loss": f"{avg_loss:.4f}", "Acc": f"{acc:.2f}%"})

    # Calculate avg validation loss/accuracy
    val_acc = 100. * correct / total
    avg_val_loss = val_loss / len(val_loader)
    return avg_val_loss, val_acc
