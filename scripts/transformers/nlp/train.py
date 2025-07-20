from configs.transformers.nlp.setup_env import device

from typing import List, Tuple, Optional

import torch
from transformers import AutoTokenizer

from src.transformers.nlp.model import Transformer

from configs.transformers.nlp.model_args.model_args_medium import ModelArgs
from configs.transformers.nlp.training_args import TrainingArgs

from src.transformers.nlp.text_cleaning.text_quality_filter import TextQualityFilter
from src.transformers.nlp.text_cleaning.deduplication_filter import DeduplicationFilter

from data.transformers.nlp.create_dataloader import create_dataloader

from training.transformers.nlp.training_components.setup_training_components import setup_training_components

from training.transformers.nlp.loops.training_loop import train
from training.transformers.nlp.loops.validation_loop import validate

from save_load_checkpoints.transformers.nlp.save_checkpoint import save_checkpoint
from save_load_checkpoints.transformers.nlp.load_checkpoint import load_checkpoint

from utils.transformers.nlp.compute_metrics import compute_perplexity

# TODO: change training logic to use total tokens as seen rather than epochs as tracking

# TODO: set up and import logger

# TODO: set up utils/transformers/nlp/visualization.py using func: len(plot_metrics(List[float]...)) = 4 -> None

def main(
    dataset_name: str,
    resume_from_checkpoint: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Main training loop.
    
    Args:
        dataset_name (str): Name of the dataset to be downloaded.
        resume_from_checkpoint (Optional[str]): File path to resume from checkpoint.
        max_samples (Optional[int]): Number of samples to download (None for all samples).

    Returns:
        Tuple[List[float], List[float], List[float], List[float]]:
            - List[float]: Training language modeling loss.
            - List[float]: Training perplexity.
            - List[float]: Validation language modeling loss loss.
            - List[float]: Validation perplexity.
    """
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    logger.info(f"Initialized {tokenizer.name_or_path} tokenizer.")
    # Initialize pad/eos token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Initialize model and training arguments
    model_args = ModelArgs()
    model_args.vocab_size = tokenizer.vocab_size # 32000 for mistral tokenizer
    model_args.pad_token_id = tokenizer.pad_token_id
    model_args.eos_token_id = tokenizer.eos_token_id

    training_args = TrainingArgs()
    logger.info("Initialized model arguments and training arguments.")

    # Initialize model
    model = Transformer(model_args).to(device)
    logger.info("Initialized model")
    
    # Setup filters
    quality_filter = TextQualityFilter()
    dedup_filter = DeduplicationFilter()
    logger.info("Initialized text quality and deduplication filters.")
    
    # Create train loader using streaming dataset
    logger.info("Creating streaming train dataloader...")
    train_loader = create_dataloader(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        max_length=model_args.max_seq_len,
        training_args=training_args,
        quality_filter=quality_filter,
        dedup_filter=dedup_filter,
        max_samples=max_samples,
        split="train",
        streaming=True
    )

    # Create validation loader using streaming dataset
    logger.info("Creating streaming validation dataloader...")
    val_loader = create_dataloader(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        max_length=model_args.max_seq_len,
        training_args=training_args,
        quality_filter=quality_filter,
        dedup_filter=dedup_filter,
        max_samples=max_samples // 10 if max_samples else None,
        split="validation" if "validation" in dataset_name else "train",
        streaming=True
    )
    
    if max_samples is not None:
        estimated_samples_per_epoch = max_samples
    else:
        # Use full Falcon RefinedWeb dataset size
        estimated_samples_per_epoch = 968_000_015

    estimated_steps_per_epoch = estimated_samples_per_epoch // training_args.batch_size
    num_training_steps = (estimated_steps_per_epoch * training_args.epochs) // training_args.grad_accum_steps
    
    # Setup training components
    optimizer, scheduler, scaler = setup_training_components(
        model, training_args, num_training_steps
    )

    # Initialize start epoch, loss and perplexity
    start_epoch = 0
    best_train_lm_loss, best_train_perplexity = float("inf"), float("inf")
    best_val_lm_loss, best_val_perplexity = float("inf"), float("inf")

    # Initialize lists for visualization
    train_lm_losses, train_perplexities = [], []
    val_lm_losses, val_perplexities = [], []
    
    # Resume from checkpoint if provided
    if resume_from_checkpoint is not None:
        logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
        checkpoint_info = load_checkpoint(
            resume_from_checkpoint, model, optimizer, scheduler, scaler, device
        )
        start_epoch = checkpoint_info['epoch'] + 1
        best_val_lm_loss = checkpoint_info['loss']

    # Dataset info
    logger.info("DATASET INFORMATION")
    logger.info("-" * 50)
    logger.info(f"Training dataset: {dataset_name}")
    logger.info(f"Streaming mode: True")
    logger.info(f"Max samples per epoch: {max_samples if max_samples else 'All available'}")
    logger.info(f"Estimated steps per epoch: {estimated_steps_per_epoch}\n")

    # Tokenization info
    logger.info("TOKENIZATION INFORMATION")
    logger.info("-" * 50)
    logger.info(f"Pad token: {tokenizer.pad_token} | EOS token: {tokenizer.eos_token}")
    logger.info(f"Pad token id: {model_args.pad_token_id} | EOS token id: {model_args.eos_token_id}")
    logger.info(f"Vocab size: {model_args.vocab_size}")
    logger.info(f"Max sequence length: {model_args.max_seq_len}\n")

    # Model info
    logger.info("MODEL INFORMATION")
    logger.info("-" * 50)
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")

    # Training components
    logger.info("TRAINING COMPONENTS")
    logger.info("-" * 50)
    logger.info(f"Optimizer: {type(optimizer).__name__}")
    logger.info(f"Scheduler: {type(scheduler).__name__}")
    logger.info(f"Scaler available: {bool(scaler)}\n") # If device.type == cuda: True, else scaler=None: False

    # Training steps/epochs
    logger.info("TRAINING LENGTHS")
    logger.info("-" * 50)
    logger.info(f"Number of Epochs: {training_args.epochs}")
    logger.info(f"Estimated Number of Steps: {num_training_steps}")
    logger.info(f"Number of warmup steps: {int(training_args.warmup_ratio * num_training_steps)}\n")

    # Training hyperparameters
    logger.info("TRAINING HYPERPARAMETERS")
    logger.info("-" * 50)
    logger.info(f"Learning rate: {training_args.learning_rate}")
    logger.info(f"Batch size: {training_args.batch_size}")
    logger.info(f"Max evaluation batches: {training_args.max_eval_batches}")

    # Training loop
    logger.info("Training starting...")

    for epoch in range(start_epoch, training_args.epochs):
        logger.info(f"Starting epoch {epoch + 1}/{training_args.epochs}")

        # Train for one epoch and compute perplexity
        train_loss, train_lm_loss, train_aux_loss = train(
            model, train_loader, optimizer, scheduler, training_args, device, epoch, scaler
        )

        train_perplexity = compute_perplexity(train_lm_loss)

        # Validate for one epoch and compute perplexity
        val_loss, val_lm_loss, val_aux_loss = validate(
            model, val_loader, training_args, device, training_args.max_eval_batches
        )

        val_perplexity = compute_perplexity(val_lm_loss)

        # Update lists to store losses and perplexities
        train_lm_losses.append(train_lm_loss)
        val_lm_losses.append(val_lm_loss)
        train_perplexities.append(train_perplexity)
        val_perplexities.append(val_perplexity)

        # Get best training loss (best validation loss used for saving best checkpoint)
        if train_lm_loss < best_train_lm_loss:
            best_train_lm_loss = train_lm_loss

        # Get best perplexities
        if train_perplexity < best_train_perplexity:
            best_train_perplexity = train_perplexity

        if val_perplexity < best_val_perplexity:
            best_val_perplexity = val_perplexity

        # Log training loss and perplexity
        logger.info(f"Epoch {epoch + 1} Training Loss & Perplexity:")
        logger.info(f"Train Loss: {train_loss:.4f} | Train LM Loss: {train_lm_loss:.4f} | Train Aux Loss: {train_aux_loss:.4f}")
        logger.info(f"Train Perplexity: {train_perplexity:.4f}")

        # Log validation loss and perplexity
        logger.info(f"Epoch {epoch + 1} Validation Loss & Perplexity:")
        logger.info(f"Val Loss: {val_loss:.4f} | Val LM Loss: {val_lm_loss:.4f} | Val Aux Loss: {val_aux_loss:.4f}")
        logger.info(f"Val Perplexity: {val_perplexity:.4f}")

        # Save regular checkpoint
        if (epoch + 1) % training_args.save_freq == 0:
            checkpoint_path = save_checkpoint(
                model, optimizer, scheduler, epoch, 0, train_loss,
                training_args, model_args, scaler, is_best=False
            )
            logger.info(f"Saved checkpoint to {checkpoint_path}")

        # Save best model (lowest validation loss)
        if val_lm_loss < best_val_lm_loss:
            best_val_lm_loss = val_lm_loss
            best_checkpoint_path = save_checkpoint(
                model, optimizer, scheduler, epoch, 0, val_loss,
                training_args, model_args, scaler, is_best=True
            )
            logger.info(f"New best model saved to {best_checkpoint_path}")

        # Clean up GPU memory at the end of each epoch
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Log best loss and perplexity
    logger.info("Training Complete! Best Loss & Perplexity:")
    logger.info(f"Best Training Loss: {best_train_lm_loss:.4f} | Best Training Perplexity: {best_train_perplexity:.4f}")
    logger.info(f"Best Validation Loss: {best_val_lm_loss:.4f} | Best Validation Perplexity: {best_val_perplexity:.4f}")

    return train_lm_losses, train_perplexities, val_lm_losses, val_perplexities

if __name__ == "__main__":
    main(
        dataset_name="tiiuae/falcon-refinedweb",
        resume_from_checkpoint=None,
        max_samples=None
    )
