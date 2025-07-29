from configs.setup_env import device, dtype

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from tqdm import tqdm

from configs.transformers.vision.vit_2d.training_args import TrainingArgs
from src.transformers.vision.vit_2d.data_augmentation.random_augmentation import random_augmentation

def train(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    training_args: TrainingArgs,
    scaler: Optional[GradScaler],
    epoch: int,
) -> Tuple[float, float]:
    """Train Vision Transformer for a single epoch.

    Args:
        model (nn.Module): Vision Transformer.
        train_loader (DataLoader): DataLoader containing training_examples.
        optimizer (optim.Optimizer): Optimizer to update weights.
        training_args (TrainingArgs): Training hyperparameters.
        scaler (Optional[GradScaler]): Gradient scaler for bf16/fp16 gradients.
        epoch (int): Current epoch during training.

    Returns:
        Tuple[float, float]:
            - float: Average loss over epoch.
            - float: Accuracy over epoch.
    """
    model.train() # Turn on training mode

    # Initialize
    total_loss = 0.0
    correct = 0
    total = 0

    # Set up pbar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{training_args.epochs}")

    # Training loop
    for batch_idx, (images, targets) in enumerate(pbar):
        images, targets = images.to(device), targets.to(device)

        # Apply randomized augmentation
        images, targets_a, targets_b, lam = random_augmentation(images, targets, training_args.mixup_alpha)

        if batch_idx % training_args.grad_accum_steps == 0:
            optimizer.zero_grad()

        # GPU accelerated path - CUDA available
        if scaler is not None:
            with torch.amp.autocast(device_type=device.type, dtype=dtype):
                outputs = model(images) # Forward pass of ViT

                # Compute weighted loss and scale
                loss = (
                    lam * F.cross_entropy(outputs, targets_a, label_smoothing=training_args.label_smoothing) +
                    (1 - lam) * F.cross_entropy(outputs, targets_b, label_smoothing=training_args.label_smoothing)
                )
                loss = loss / training_args.grad_accum_steps

            # Accumulate and backpropagate loss
            total_loss += loss.item() * training_args.grad_accum_steps
            scaler.scale(loss).backward()

            if (batch_idx + 1) % training_args.grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                # Unscale scaled up gradients
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), training_args.max_norm) # Clip L2 Norm
                scaler.step(optimizer)
                scaler.update() # Update weights
        # CPU path - No CUDA available
        else:
            outputs = model(images)
            loss = (
                lam * F.cross_entropy(outputs, targets_a, label_smoothing=training_args.label_smoothing) +
                (1 - lam) * F.cross_entropy(outputs, targets_b, label_smoothing=training_args.label_smoothing)
            )
            loss = loss / training_args.grad_accum_steps
            total_loss += loss.item() * training_args.grad_accum_steps
            loss.backward()

            if (batch_idx + 1) % training_args.grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                nn.utils.clip_grad_norm_(model.parameters(), training_args.max_norm)
                optimizer.step()

        # Calculate accuracy for non-augmented batches
        if lam == 1.0:
            predicted = torch.argmax(outputs, dim=1)
            total += targets.size(0)
            correct += predicted.eq(targets_a).sum().item()

        # Logging
        if batch_idx % 100 == 0:
            acc = 100. * correct / total
            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({
                "Loss": f"{avg_loss:.4f}",
                "Acc": f"{acc:.2f}%",
                "LR": f"{optimizer.param_groups[0]['lr']:.6f}"
            })

    # Get average loss
    epoch_loss = total_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc
