from configs.transformers.nlp.setup_env import device, dtype

import logging
from typing import Optional, Dict, Union, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from tqdm import tqdm

from configs.transformers.nlp.training_args import TrainingArgs
from configs.transformers.nlp.generation_args import GenerationArgs
from utils.transformers.nlp.compute_metrics import compute_loss
from utils.setup_logger import setup_logger

# Set up logger
train_logger = setup_logger(name="train_logger", log_file="training.log", level=logging.INFO)

def train_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    optimizer: Optimizer,
    training_args: TrainingArgs,
    generation_args: GenerationArgs,
    step: int,
    scaler: Optional[GradScaler] = None,
) -> Dict[str, Union[float, bool]]:
    """Single training step with AMP support.
    
    Args:
        model (nn.Module): Transformer architecture.
        batch (Dict[str, torch.Tensor]): Dictionary containing input_ids, labels, and attention mask.
        optimizer (Optimizer): PyTorch optimizer.
        training_args (TrainingArgs): Training hyperparameters.
        generation_args (GenerationArgs): Generation hyperparameters.
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

        # Count tokens
        tokens_this_step = (input_ids != generation_args.pad_token_id).sum().item()

        # Forward pass with AMP
        if scaler is not None:
            with autocast(device_type=device.type, dtype=dtype):
                logits, _, aux_loss = model(input_ids, padding_mask=attention_mask, use_cache=False)
                loss, lm_loss, aux_loss_val = compute_loss(logits, labels, training_args, aux_loss)
                loss = loss / training_args.grad_accum_steps # Scale loss by gradient accumulation steps
        # Standard FP32 forward pass
        else:
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
            'tokens': tokens_this_step,
            'success': True
        }

    # Catch OOM error
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            train_logger.warning(f"OOM at step {step}, emptying CUDA cache.")
            if device.type == "cuda":
                torch.cuda.empty_cache()
            optimizer.zero_grad()
            return {'success': False, 'error': 'oom'}
        else:
            train_logger.error(f"Training step failed: {e}")
            return {'success': False, 'error': str(e)}

def train(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    training_args: TrainingArgs,
    generation_args: GenerationArgs,
    epoch: int,
    scaler: Optional[GradScaler] = None,
    tokens_seen: int = 0,
) -> Tuple[float, float, float, int, bool]:
    """
    Train for one epoch (or until token budget is exceeded) with AMP support.

    Returns:

    """
    model.train() # Set model to training

    # Initialization
    total_loss = 0
    total_lm_loss = 0
    total_aux_loss = 0
    successful_steps = 0
    skipped_steps = 0
    total_tokens = 0
    stop_early = False

    # Set up pbar
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
    optimizer.zero_grad()

    # Loop through dataloader
    for step, batch in enumerate(progress_bar):
        result = train_step(model, batch, optimizer, training_args, generation_args, step, scaler)

        if result['success']:
            tokens_this_step = result['tokens']

            # Early exit if token budget will be exceeded
            if (
                training_args.max_train_tokens is not None and
                tokens_seen + total_tokens + tokens_this_step > training_args.max_train_tokens
            ):
                stop_early = True
                break
            
            # Accumulate loss/tokens/steps
            total_loss += result['loss']
            total_lm_loss += result['lm_loss']
            total_aux_loss += result['aux_loss']
            total_tokens += tokens_this_step
            successful_steps += 1
        else:
            skipped_steps += 1
            # Too many failed steps, break out of loop
            if skipped_steps > training_args.max_skipped_steps:
                train_logger.error("Too many failed steps, stopping epoch")
                break

        # Gradient accumulation and optimizer step
        if (step + 1) % training_args.grad_accum_steps == 0:
            if successful_steps > 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), training_args.clip_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
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

    # Final optimizer flush if partial gradient accumulation left
    if (successful_steps % training_args.grad_accum_steps) != 0 and successful_steps > 0:
        if scaler is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), training_args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            nn.utils.clip_grad_norm_(model.parameters(), training_args.clip_grad_norm)
            optimizer.step()
        optimizer.zero_grad()

    if successful_steps == 0:
        train_logger.error("No successful training steps in epoch")
        return float('inf'), float('inf'), float('inf'), 0, stop_early

    return (
        total_loss / successful_steps,
        total_lm_loss / successful_steps,
        total_aux_loss / successful_steps,
        total_tokens,
        stop_early
    )
