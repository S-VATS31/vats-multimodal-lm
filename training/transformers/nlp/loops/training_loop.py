from configs.setup_env import device

import logging
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from tqdm import tqdm

from src.transformers.nlp.model import AutoregressiveTextTransformer
from configs.transformers.nlp.training_args import TrainingArgs
from configs.transformers.nlp.generation_args import GenerationArgs
from utils.transformers.nlp.compute_metrics import compute_loss, compute_perplexity
from utils.setup_logger import setup_logger

# Set up logger
train_logger = setup_logger(name="train_logger", log_file="training.log", level=logging.INFO)

def train_step(
    model: AutoregressiveTextTransformer,
    batch: Dict[str, torch.Tensor],
    training_args: TrainingArgs,
    generation_args: GenerationArgs,
    scaler: Optional[GradScaler] = None,
) -> Tuple[float, float, float, bool, int]:
    """Single training step.
    
    Args:
        model (AutoregressiveTextTransformer): Transformer architecture.
        batch (Dict[str, torch.Tensor]): Dictionary containing input_ids, labels, and attention mask.
        optimizer (Optimizer): PyTorch optimizer.
        training_args (TrainingArgs): Training hyperparameters.
        generation_args (GenerationArgs): Generation hyperparameters.
        scaler (Optional[GradScaler]): Gradient scaling for bf16/fp16 gradients.

    Returns:
        Tuple:
            - float: Total loss * gradient accumulation steps.
            - float: Language modeling loss.
            - float: Auxiliary loss.
            - bool: Whether the step was a success or not.
    """
    try:
        # Get input_ids, labels and attention mask
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        logits, _, aux_loss = model(input_ids, padding_mask=attention_mask, use_cache=False)
        loss, lm_loss, aux_loss_val = compute_loss(logits, labels, training_args, aux_loss)
        loss = loss / training_args.grad_accum_steps

        # Count non-padded number of tokens
        tokens_in_step = (input_ids != generation_args.pad_token_id).sum().item()

        # Backprop
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        return (
            loss.item() * training_args.grad_accum_steps,
            lm_loss.item(),
            aux_loss_val.item(),
            True,
            tokens_in_step
        )

    # Step failed
    except Exception as e:
        if "out of memory" in str(e).lower():
            if device.type == "cuda":
                torch.cuda.empty_cache()
        return (float("inf"), float("inf"), float("inf"), False, 0)

def train(
    model: AutoregressiveTextTransformer,
    dataloader: DataLoader,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    training_args: TrainingArgs,
    generation_args: GenerationArgs,
    scaler: Optional[GradScaler] = None,
) -> Tuple[float, float, float, float, int, bool]:
    """
    Train for set amount of tokens.

    Args:
        model (AutoregressiveTextTransformer): Transformer.
        dataloader (DataLoader): Dataloader containing training examples.
        optimizer (Optimizer): PyTorch optimizer for weight updates.
        scheduler (LambdaLR): PyTorch scheduler to update learning rate.
        training_args (TrainingArgs): Training hyperparameters.
        scaler (Optional[GradScaler]): Gradient scaling for FP16/BF16 gradients.
    
    Returns:
        Tuple:
            - float: Average total loss.
            - float: Average LM loss.
            - float: Average aux loss.
            - float: Average perplexity.
            - int: Total tokens seen.
            - bool: Whether to stop early or not.
    """
    model.train()

    # Initialization
    total_loss = 0
    total_lm_loss = 0
    total_aux_loss = 0
    total_perplexity = 0
    successful_steps = 0
    skipped_steps = 0
    total_tokens_seen = 0
    stop_early = False

    # Set up pbar
    progress_bar = tqdm(dataloader, desc="Training")
    optimizer.zero_grad(set_to_none=True)

    # Loop through dataloader
    for step, batch in enumerate(progress_bar):
        loss, lm_loss, aux_loss, success, tokens_in_step = train_step(
            model, batch, training_args, generation_args, scaler
        )

        if success:
            # Accumulate loss/perplexity/tokens
            perplexity = compute_perplexity(lm_loss)
            total_loss += loss
            total_lm_loss += lm_loss
            total_aux_loss += aux_loss
            total_perplexity += perplexity
            total_tokens_seen += tokens_in_step
            successful_steps += 1
            # Update progress bar
            progress_bar.set_postfix({"tokens_seen": total_tokens_seen})
            if total_tokens_seen >= training_args.max_train_tokens:
                train_logger.info("Max train tokens reached, stopping early")
                stop_early = True
                break
        else:
            skipped_steps += 1
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
            optimizer.zero_grad(set_to_none=True)

        # Logging
        if (step + 1) % training_args.logging_steps == 0 and successful_steps > 0:
            avg_loss = total_loss / successful_steps
            avg_lm_loss = total_lm_loss / successful_steps
            avg_aux_loss = total_aux_loss / successful_steps
            avg_perplexity = total_perplexity / successful_steps
            lr = scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "lm_loss": f"{avg_lm_loss:.4f}",
                "aux_loss": f"{avg_aux_loss:.4f}",
                "perplexity": f"{avg_perplexity:.4f}",
                "lr": f"{lr:.2e}",
                "skipped_steps": skipped_steps,
                "tokens_seen": total_tokens_seen
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
        optimizer.zero_grad(set_to_none=True)

    if successful_steps == 0:
        train_logger.error("No successful training steps in epoch")
        return float("inf"), float("inf"), float("inf"), float("inf"), 0, True

    return (
        total_loss / successful_steps,
        total_lm_loss / successful_steps,
        total_aux_loss / successful_steps,
        total_perplexity / successful_steps,
        total_tokens_seen,
        stop_early
    )
