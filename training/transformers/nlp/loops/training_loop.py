from configs.transformers.nlp.setup_env import dtype

from typing import Optional, Dict, Union, Tuple

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from tqdm import tqdm

from configs.transformers.nlp.training_args import TrainingArgs
from utils.transformers.nlp.compute_metrics import compute_loss

# Set up logger
# TODO: set up logger in utils folder and import here (take one from ViT project)

def train_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    optimizer,
    training_args: TrainingArgs,
    device: torch.device,
    step: int,
    scaler: Optional[GradScaler] = None,
) -> Dict[str, Union[float, bool]]:
    """Single training step with AMP support.
    
    Args:
        model (nn.Module): Transformer architecture.
        batch (Dict[str, torch.Tensor]): Dictionary containing input_ids, labels, and attention mask.
        optimizer: PyTorch optimizer.
        training_args (TrainingArgs): Training hyperparameters.
        device (torch.device): Accelerator at use.
        step (int): Current step during training.
        scaler (Optional[GradScaler]): Gradient scaling for bf16/fp16 gradients.

    Returns:
        Dict[str, Union[float, bool]]: Dictionary containing loss and whether the step was a success.
    """
    try:
        # Get input_ids, labels and attention mask
        input_ids = batch['input_ids'].to(device, non_blocking=training_args.pin_memory)
        labels = batch['labels'].to(device, non_blocking=training_args.pin_memory)
        attention_mask = batch['attention_mask'].to(device, non_blocking=training_args.pin_memory)

        # Forward pass with AMP
        if scaler is not None:
            with autocast(device_type=device.type, dtype=dtype):
                logits, _, aux_loss = model(input_ids, padding_mask=attention_mask, use_cache=False)
                loss, lm_loss, aux_loss_val = compute_loss(logits, labels, training_args, aux_loss)
                loss = loss / training_args.grad_accum_steps # Scale loss by gradient accumulation steps
        else:
            # Standard FP32 forward pass
            logits, _, aux_loss = model(input_ids, padding_mask=attention_mask, use_cache=False)
            loss, lm_loss, aux_loss_val = compute_loss(logits, labels, training_args, aux_loss)
            loss = loss / training_args.grad_accum_steps

        # Backward pass with gradient scaling
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        return {
            'loss': loss.item() * training_args.grad_accum_steps,
            'lm_loss': lm_loss.item(),
            'aux_loss': aux_loss_val.item(),
            'success': True
        }

    # Catch OOM error
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.warning(f"OOM at step {step}, emptying CUDA cache.")
            if device.type == "cuda":
                torch.cuda.empty_cache()
            optimizer.zero_grad()
            return {'success': False, 'error': 'oom'}
        else:
            logger.error(f"Training step failed: {e}")
            return {'success': False, 'error': str(e)}

def train(
    model: nn.Module,
    dataloader: DataLoader, 
    optimizer, 
    scheduler, 
    training_args: TrainingArgs, 
    device: torch.device, 
    epoch: int, 
    scaler: Optional[GradScaler] = None,
) -> Tuple[float, float, float]:
    """Train for one epoch with AMP support.
    
    Args:
        model (nn.Module): Transformer architecture.
        dataloader (DataLoader): PyTorch DataLoader.
        optimizer: PyTorch optimizer.
        scheduler: PyTorch scheduler.
        training_args (TrainingArgs): Training hyperparameters.
        device (torch.device): Accelerator at use.
        epoch (int): Current epoch during training.
        scaler (Optional[GradScaler]): Gradient scaler to scale bf16/fp16 gradients.

    Returns:
        Tuple[float, float, float]: Tuple containing total loss, lm loss, and aux loss.
            - float: Total loss scaled by succesful steps.
            - float: LM loss scaled by succesful steps.
            - float: Aux loss scaled by succesful steps.
    """
    model.train() # Set model to training mode

    # Initialization
    total_loss = 0
    total_lm_loss = 0
    total_aux_loss = 0
    successful_steps = 0
    skipped_steps = 0

    # Set up pbar
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
    optimizer.zero_grad()

    # Loop through all steps
    for step, batch in enumerate(progress_bar):
        result = train_step(model, batch, optimizer, training_args, device, step, scaler)

        # If step was successful, accumulate loss
        if result['success']:
            total_loss += result['loss']
            total_lm_loss += result['lm_loss']
            total_aux_loss += result['aux_loss']
            successful_steps += 1
        else:
            skipped_steps += 1
            # If 10% of steps fail, stop epoch
            if skipped_steps > len(dataloader) * 0.1:
                logger.error("Too many failed steps, stopping epoch")
                break

        # Gradient accumulation and optimizer step
        if (step + 1) % training_args.grad_accum_steps == 0:
            if successful_steps > 0:
                # AMP available
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    # Clip L2 Norm
                    nn.utils.clip_grad_norm_(model.parameters(), training_args.clip_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                # No AMP available
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), training_args.clip_grad_norm)
                    optimizer.step()
                scheduler.step()
            optimizer.zero_grad()

        # Logging
        if (step + 1) % training_args.logging_steps == 0 and successful_steps > 0:
            avg_loss = total_loss / successful_steps
            avg_lm_loss = total_lm_loss / successful_steps
            avg_aux_loss = total_aux_loss / successful_steps
            lr = scheduler.get_last_lr()[0]

            progress_bar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "lm_loss": f"{avg_lm_loss:.4f}",
                "aux_loss": f"{avg_aux_loss:.4f}",
                "lr": f"{lr:.2e}",
                "skipped_steps": skipped_steps
            })

    # Handle remaining gradients
    if len(dataloader) % training_args.grad_accum_steps != 0:
        if successful_steps > 0:
            # AMP available
            if scaler is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), training_args.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            # AMP not available
            else:
                nn.utils.clip_grad_norm_(model.parameters(), training_args.clip_grad_norm)
                optimizer.step()
        optimizer.zero_grad()

    # No succesful steps, all loss = inf
    if successful_steps == 0:
        logger.error("No successful training steps in epoch")
        return float('inf'), float('inf'), float('inf')

    return (total_loss / successful_steps,
            total_lm_loss / successful_steps,
            total_aux_loss / successful_steps)
