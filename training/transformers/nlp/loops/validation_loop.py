from configs.transformers.nlp.setup_env import device, dtype

import logging
from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm

from configs.transformers.nlp.training_args import TrainingArgs
from utils.transformers.nlp.compute_metrics import compute_loss
from utils.setup_logger import setup_logger

# Set up logger
validation_logger = setup_logger(name="validation_logger", log_file="training.log", level=logging.INFO)

def validate(
    model: nn.Module,
    dataloader: DataLoader,
    training_args: TrainingArgs,
    max_batches: Optional[int] = None,
) -> Tuple[float, float, float]:
    """Evaluate model with AMP support.
    
    Args:
        model (nn.Module): Transformer architecture
        dataloader (DataLoader): PyTorch DataLoader.
        training_args (TrainingArgs): Training hyperparameters.
        device (torch.device): Accelerator at use.
        max_batches (Optional[int]): Max batches to evaluate on.

    Returns:
    Tuple[float, float, float]: Tuple containing total loss, lm loss, and aux loss.
        - float: Total loss scaled by succesful steps.
        - float: LM loss scaled by succesful steps.
        - float: Aux loss scaled by succesful steps.
    """
    model.eval() # No dropout

    # Initialize loss and batches
    total_loss = 0
    total_lm_loss = 0
    total_aux_loss = 0
    successful_batches = 0

    # Turn off gradient tracking for evaluation
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            # Break if max batches reached
            if max_batches is not None and i >= max_batches:
                break

            try:
                # Get input_ids, labels, and attention_mask
                input_ids = batch['input_ids'].to(device, non_blocking=training_args.pin_memory)
                labels = batch['labels'].to(device, non_blocking=training_args.pin_memory)
                attention_mask = batch['attention_mask'].to(device, non_blocking=training_args.pin_memory)

                with torch.amp.autocast(device_type=device.type, dtype=dtype):
                    logits, _, aux_loss = model(input_ids, padding_mask=attention_mask, use_cache=False)
                    loss, lm_loss, aux_loss_val = compute_loss(logits, labels, training_args, aux_loss)

                # Accumulate loss
                total_loss += loss.item()
                total_lm_loss += lm_loss.item()
                total_aux_loss += aux_loss_val.item()
                successful_batches += 1
            
            # Catch OOM error
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    validation_logger.warning("OOM during evaluation, skipping batch")
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    continue
                else:
                    validation_logger.warning(f"Evaluation error: {e}")
                    continue

    # No succesful batches, all loss = inf
    if successful_batches == 0:
        validation_logger.error("No successful evaluation batches")
        return float('inf'), float('inf'), float('inf')

    return (total_loss / successful_batches,
            total_lm_loss / successful_batches,
            total_aux_loss / successful_batches)