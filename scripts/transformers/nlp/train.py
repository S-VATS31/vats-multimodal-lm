from configs.setup_env import device

import logging
from typing import List, Tuple, Optional

from torch.cuda import empty_cache
from transformers import AutoTokenizer

from src.transformers.nlp.model import Transformer
from configs.transformers.nlp.model_args.model_args_medium import ModelArgs
from configs.transformers.nlp.training_args import TrainingArgs
from configs.transformers.nlp.generation_args import GenerationArgs
from src.transformers.nlp.text_cleaning.text_quality_filter import TextQualityFilter
from src.transformers.nlp.text_cleaning.deduplication_filter import DeduplicationFilter
from data.transformers.nlp.create_dataloader import create_dataloader
from training.transformers.nlp.setup_training_components import setup_training_components
from training.transformers.nlp.loops.training_loop import train
from training.transformers.nlp.loops.validation_loop import validate
from utils.transformers.nlp.checkpointing import save_checkpoint, load_checkpoint
from utils.transformers.nlp.compute_metrics import compute_perplexity
from utils.transformers.nlp.visualization import plot_metrics
from utils.setup_logger import setup_logger
from src.transformers.nlp.generate import AutoregressiveTokenGenerator

# Set up logger
training_logger = setup_logger(
    name="training_logger", 
    log_file="training.log", 
    level=logging.INFO
)

def main(
    dataset_name: str,
    resume_from_checkpoint: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Train LLM for a set number of samples and reume from checkpoint if available.

    Args:
        dataset_name (str): Name of dataset.
        resume_from_checkpoint (Optional[str]): Resume from checkpoint if available.
        max_samples (Optional[int]): Max number of samples to download (None for full dataset).

    Returns:
        Tuple[List[float], List[float], List[float], List[float]]:
            - List[float]: List containing train losses.
            - List[float]: List containing train perplexities.
            - List[float]: List containing validation losses.
            - List[float]: List containing validation perplexities.
    """
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    training_logger.info(f"Initialized {tokenizer.name_or_path} tokenizer.")
    # Initialize pad/eos token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Initialize all args
    model_args = ModelArgs()
    model_args.vocab_size = tokenizer.vocab_size

    generation_args = GenerationArgs()
    generation_args.pad_token_id = tokenizer.pad_token_id
    generation_args.eos_token_id = tokenizer.eos_token_id

    training_args = TrainingArgs()
    training_logger.info("Initialized model, training, and generation arguments.")

    # Initialize model
    model = Transformer(model_args).to(device)
    training_logger.info("Initialized model.")

    # Initialize text quality filters
    quality_filter = TextQualityFilter()
    dedup_filter = DeduplicationFilter()
    training_logger.info("Initialized text quality and deduplication filters.")

    # Initialize token generator
    token_generator = AutoregressiveTokenGenerator(model_args)
    training_logger.info("Initialized token generator")

    # Create training dataloader
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
    training_logger.info("Created train_loader for training examples.")

    # Create validation dataloader
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
    training_logger.info("Created val_loader for validation examples.")

    # 80% of max sequence length estimate
    estimated_tokens_per_sample = model_args.max_seq_len * 0.8
    estimated_samples = int(training_args.max_train_tokens // estimated_tokens_per_sample)
    estimated_steps = estimated_samples // training_args.batch_size
    num_training_steps = estimated_steps // training_args.grad_accum_steps

    # Set up optimizer, scheduler, and scaler
    optimizer, scheduler, scaler = setup_training_components(
        model, training_args, num_training_steps
    )

    # Initialize loss and perplexity
    start_epoch = 0
    best_train_lm_loss, best_train_perplexity = float("inf"), float("inf")
    best_val_lm_loss, best_val_perplexity = float("inf"), float("inf")

    # Initialize lists for visualization
    train_lm_losses, train_perplexities = [], []
    val_lm_losses, val_perplexities = [], []

    # Resume from checkpoint if available
    if resume_from_checkpoint is not None:
        training_logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
        checkpoint_info = load_checkpoint(
            resume_from_checkpoint, model, optimizer, scheduler, scaler
        )
        start_epoch = checkpoint_info.get("epoch", 0) + 1
        best_val_lm_loss = checkpoint_info.get("loss", float("inf"))

    # Dataset info
    training_logger.info("DATASET INFORMATION")
    training_logger.info("-" * 50)
    training_logger.info(f"Training dataset: {dataset_name}")
    training_logger.info(f"Streaming mode: True") # always do streaming for large datasets
    training_logger.info(f"Max samples per epoch: {max_samples if max_samples else 'All available'}")
    training_logger.info(f"Estimated steps per epoch: {num_training_steps}\n")

    # Tokenization info
    training_logger.info("TOKENIZATION INFORMATION")
    training_logger.info("-" * 50)
    training_logger.info(f"Pad token: {tokenizer.pad_token} | EOS token: {tokenizer.eos_token}")
    training_logger.info(f"Pad token id: {tokenizer.pad_token_id} | EOS token id: {tokenizer.eos_token_id}")
    training_logger.info(f"Vocab size: {model_args.vocab_size}")
    training_logger.info(f"Max sequence length: {model_args.max_seq_len}\n")

    # Model information
    training_logger.info("MODEL INFORMATION")
    training_logger.info("-" * 50)
    training_logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    training_logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")

    # Training info
    training_logger.info("TRAINING COMPONENTS")
    training_logger.info("-" * 50)
    training_logger.info(f"Optimizer: {type(optimizer).__name__}")
    training_logger.info(f"Scheduler: {type(scheduler).__name__}")
    training_logger.info(f"Scaler available: {bool(scaler)}\n")

    # Training lengths info
    training_logger.info("TRAINING LENGTHS")
    training_logger.info("-" * 50)
    training_logger.info(f"Number of Epochs: {training_args.epochs}")
    training_logger.info(f"Estimated Number of Steps: {num_training_steps}")
    training_logger.info(f"Number of warmup steps: {int(training_args.warmup_ratio * num_training_steps)}\n")

    # Training hyperparameters info
    training_logger.info("TRAINING HYPERPARAMETERS")
    training_logger.info("-" * 50)
    training_logger.info(f"Learning rate: {training_args.learning_rate}")
    training_logger.info(f"Batch size: {training_args.batch_size}")
    training_logger.info(f"Max evaluation batches: {training_args.max_eval_batches}")

    training_logger.info("Training starting...")

    total_tokens_seen = 0

    # Start training
    for epoch in range(start_epoch, training_args.epochs):
        training_logger.info(f"Starting epoch {epoch + 1}/{training_args.epochs}")

        # Train model
        train_loss, train_lm_loss, train_aux_loss, tokens_this_epoch, stop_early = train(
            model, train_loader, optimizer, scheduler, training_args, 
            generation_args, epoch, scaler, total_tokens_seen
        )

        # Accumulate tokens
        total_tokens_seen += tokens_this_epoch
        training_logger.info(f"Epoch {epoch + 1} — Tokens this epoch: {tokens_this_epoch:,}, Total tokens seen: {total_tokens_seen:,}")

        # Compute training perplexity
        train_perplexity = compute_perplexity(train_lm_loss)

        # Validate model
        val_loss, val_lm_loss, val_aux_loss = validate(
            model, val_loader, training_args, training_args.max_eval_batches
        )

        # Compute validation perplexity
        val_perplexity = compute_perplexity(val_lm_loss)

        # Store losses/perplexities
        train_lm_losses.append(train_lm_loss)
        val_lm_losses.append(val_lm_loss)
        train_perplexities.append(train_perplexity)
        val_perplexities.append(val_perplexity)

        # Update best loss (val loss will be used for checkpointing)
        if train_lm_loss < best_train_lm_loss:
            best_train_lm_loss = train_lm_loss

        # Update best perplexities
        if train_perplexity < best_train_perplexity:
            best_train_perplexity = train_perplexity

        if val_perplexity < best_val_perplexity:
            best_val_perplexity = val_perplexity

        # Log training loss/perplexity
        training_logger.info(f"Epoch {epoch + 1} Training Loss & Perplexity:")
        training_logger.info(f"Train Loss: {train_loss:.4f} | Train LM Loss: {train_lm_loss:.4f} | Train Aux Loss: {train_aux_loss:.4f}")
        training_logger.info(f"Train Perplexity: {train_perplexity:.4f}")

        # Log validation loss/perplexity
        training_logger.info(f"Epoch {epoch + 1} Validation Loss & Perplexity:")
        training_logger.info(f"Val Loss: {val_loss:.4f} | Val LM Loss: {val_lm_loss:.4f} | Val Aux Loss: {val_aux_loss:.4f}")
        training_logger.info(f"Val Perplexity: {val_perplexity:.4f}")

        # Save regular checkpoint
        if total_tokens_seen % training_args.save_freq == 0:
            checkpoint_path = save_checkpoint(
                model, optimizer, scheduler, epoch, 0, train_loss,
                training_args, model_args, scaler, is_best=False
            )
            training_logger.info(f"Saved checkpoint to {checkpoint_path}")

        # Save best checkpoint (lowest validation loss)
        if val_lm_loss < best_val_lm_loss:
            best_val_lm_loss = val_lm_loss
            best_checkpoint_path = save_checkpoint(
                model, optimizer, scheduler, epoch, 0, val_loss,
                training_args, model_args, scaler, is_best=True
            )
            training_logger.info(f"New best model saved to {best_checkpoint_path}")

        # Test generation every N tokens
        if total_tokens_seen % generation_args.generation_frequency == 0:
            prompt = "Once upon a time,"
            generated_text = token_generator.generate_tokens(
                prompt=prompt,
                generation_args=generation_args,
                tokenizer=tokenizer
            )
            training_logger.info(f"Prompt: {prompt}")
            training_logger.info(f"Generated text: {generated_text}")

        # Total training tokens has been reached
        if stop_early or (training_args.max_train_tokens is not None and total_tokens_seen >= training_args.max_train_tokens):
            training_logger.info(f"\nStopping training: token budget of {training_args.max_train_tokens:,} reached.")
            break

        # Empty GPU cache
        if device.type == "cuda":
            empty_cache()

    # Log best loss/perplexity for train/val
    training_logger.info("Training Complete! Best Loss & Perplexity:")
    training_logger.info(f"Best Training Loss: {best_train_lm_loss:.4f} | Best Training Perplexity: {best_train_perplexity:.4f}")
    training_logger.info(f"Best Validation Loss: {best_val_lm_loss:.4f} | Best Validation Perplexity: {best_val_perplexity:.4f}")

    return train_lm_losses, train_perplexities, val_lm_losses, val_perplexities

if __name__ == "__main__":
    train_lm_losses, train_perplexities, val_lm_losses, val_perplexities = main(
        dataset_name="tiiuae/falcon-refinedweb",
        resume_from_checkpoint=None,
        max_samples=None
    )
    plot_metrics(
        train_lm_losses,
        val_lm_losses,
        train_perplexities,
        val_perplexities
    )
