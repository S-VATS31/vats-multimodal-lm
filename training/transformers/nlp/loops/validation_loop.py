from configs.setup_env import device

import logging
from typing import Tuple, Optional

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

from src.transformers.nlp.model import AutoregressiveTextTransformer
from configs.transformers.nlp.training_args import TrainingArgs
from utils.transformers.nlp.compute_metrics import compute_loss, compute_perplexity
from utils.setup_logger import setup_logger

# Set up logger
validation_logger = setup_logger(name="validation_logger", log_file="training.log", level=logging.INFO)

def validate(
    model: AutoregressiveTextTransformer,
    dataloader: DataLoader,
    training_args: TrainingArgs,
    max_batches: Optional[int] = None,
) -> Tuple[float, float, float, float]:
    """Evaluate model for a set number of batches.
    
    Args:
        model (AutoregressiveTextTransformer): Transformer architecture
        dataloader (DataLoader): PyTorch DataLoader.
        training_args (TrainingArgs): Training hyperparameters.
        max_batches (Optional[int]): Max batches to evaluate on.

    Returns:
        Tuple.
            - float: Average loss.
            - float: Average LM loss.
            - float: Average auxiliary loss.
            - float: Average perplexity.
    """
    model.eval() # No dropout

    # Initialize loss and batches
    total_loss = 0
    total_lm_loss = 0
    total_aux_loss = 0
    total_perplexity = 0
    successful_batches = 0

    # Turn off gradient tracking for evaluation
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            if max_batches is not None and step >= max_batches:
                break

            try:
                # Get input_ids, labels, and attention_mask
                input_ids = batch['input_ids'].to(device, non_blocking=training_args.pin_memory)
                labels = batch['labels'].to(device, non_blocking=training_args.pin_memory)
                attention_mask = batch['attention_mask'].to(device, non_blocking=training_args.pin_memory)
                logits, _, aux_loss = model(input_ids, padding_mask=attention_mask, use_cache=False)
                loss, lm_loss, aux_loss_val = compute_loss(logits, labels, training_args, aux_loss)

                # Accumulate loss
                perplexity = compute_perplexity(lm_loss)
                total_loss += loss.item()
                total_lm_loss += lm_loss.item()
                total_aux_loss += aux_loss_val.item()
                total_perplexity += perplexity
                successful_batches += 1
            
            # Skip failed batch
            except Exception as e:
                if "out of memory" in str(e).lower():
                    validation_logger.error(f"OOM error at batch {step}")
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    continue
                else:
                    validation_logger.error(f"Error at batch {step}, {str(e)}")
                    continue

    # No succesful batches
    if successful_batches == 0:
        validation_logger.error("No successful evaluation batches")
        return float("inf"), float("inf"), float("inf"), float("inf")

    return (
            total_loss / successful_batches,
            total_lm_loss / successful_batches,
            total_aux_loss / successful_batches,
            total_perplexity / successful_batches,
    )
