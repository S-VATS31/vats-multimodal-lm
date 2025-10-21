from configs.setup_env import device

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
from typing import Optional

import torch
from torch.utils.data import random_split
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from data.transformers.nlp.data_streaming import TextDataset
from src.transformers.nlp.model import AutoregressiveTextTransformer
from configs.transformers.nlp.model_args.model_args_small import ModelArgs
from configs.transformers.nlp.training_args import TrainingArgs
from configs.transformers.nlp.generation_args import GenerationArgs
from training.transformers.nlp.setup_training_components import setup_training_components
from training.transformers.nlp.loops.training_loop import train
from training.transformers.nlp.loops.validation_loop import validate
from utils.transformers.nlp.checkpointing import save_checkpoint, load_checkpoint
from utils.setup_logger import setup_logger
from src.transformers.nlp.inference.generate import AutoregressiveTokenGenerator

# Set up logger
training_logger = setup_logger(
    name="training_logger", 
    log_file="training.log", 
    level=logging.INFO
)

def main(
    dataset_name: str,
    text_field: str = "content",
    early_stopping_threshold: int = 3,
    resume_from_checkpoint: Optional[str] = None,
    max_samples: Optional[int] = None, 
) -> None:
    """Train SLM for a set number of samples and resume from checkpoint if available.

    Args:
        dataset_name (str): Name of dataset.
        text_field (str): Column name containing the text data.
        early_stopping_threshold (int): Number of tokens to wait for decrease in loss before stopping training.
        resume_from_checkpoint (Optional[str]): Resume from checkpoint if available.
        max_samples (Optional[int]): Max number of samples to download (None for full dataset).
    """
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    training_logger.info(f"Initialized {tokenizer.name_or_path} tokenizer.")
    # Initialize pad/eos token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Initialize all args with defaults
    model_args = ModelArgs()
    training_args = TrainingArgs()

    # If resuming from checkpoint, load model_args and training_args first
    if resume_from_checkpoint is not None:
        training_logger.info(f"Loading checkpoint args from: {resume_from_checkpoint}")
        checkpoint_info = load_checkpoint(
            filename=resume_from_checkpoint,
            model=None,
            optimizer=None,
            scheduler=None,
            scaler=None,
            load_only_args=True
        )
        model_args = checkpoint_info.get("model_args", model_args)
        training_args = checkpoint_info.get("training_args", training_args)

    # Set vocab size for model args
    model_args.vocab_size = tokenizer.vocab_size

    generation_args = GenerationArgs()
    generation_args.pad_token_id = tokenizer.pad_token_id
    generation_args.eos_token_id = tokenizer.eos_token_id

    training_logger.info("Initialized model, training, and generation arguments.")

    # Initialize model with possibly checkpointed model_args
    model = AutoregressiveTextTransformer(model_args).to(device)
    training_logger.info("Initialized model.")

    # Initialize token generator
    token_generator = AutoregressiveTokenGenerator(model_args)
    training_logger.info("Initialized token generator")

    # Original train dataset
    full_train_dataset = TextDataset(
        tokenizer=tokenizer,
        model_args=model_args,
        dataset_name=dataset_name,
        split="train"
    )

    # Calculate lengths for 90% train, 10% validation
    train_len = int(0.9 * len(full_train_dataset))
    val_len = len(full_train_dataset) - train_len

    # Split dataset
    train_dataset, val_dataset = random_split(full_train_dataset, [train_len, val_len])

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False    
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=training_args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    training_logger.info("Created train_loader and val_loader for training and validation examples.")

    # Use actual dataset length for training steps calculation
    num_training_steps = (len(train_loader) // training_args.grad_accum_steps)

    # Set up optimizer, scheduler, and scaler with possibly checkpointed training_args
    optimizer, scheduler, scaler = setup_training_components(
        model, training_args, num_training_steps
    )

    # Initialize loss and perplexity
    best_train_lm_loss, best_train_ppl = float("inf"), float("inf")
    best_val_lm_loss, best_val_ppl = float("inf"), float("inf")

    # Dataset info
    training_logger.info("DATASET INFORMATION")
    training_logger.info("-" * 50)
    training_logger.info(f"Training dataset: {dataset_name}")
    training_logger.info(f"Text field: {text_field}")
    training_logger.info(f"Streaming mode: False")

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
    training_logger.info(f"Scaler available? {bool(scaler)}\n")

    # Training lengths info
    training_logger.info("TRAINING LENGTHS")
    training_logger.info("-" * 50)

    # Training hyperparameters info
    training_logger.info("TRAINING HYPERPARAMETERS")
    training_logger.info("-" * 50)
    training_logger.info(f"Learning rate: {training_args.learning_rate}")
    training_logger.info(f"Batch size: {training_args.batch_size}")
    training_logger.info(f"Max evaluation batches: {training_args.max_eval_batches}")

    training_logger.info("Training starting...")

    total_tokens_seen = 0
    last_logged_tokens = 0
    early_stopping_counter = 0
    stop_early = False
    last_save_tokens = 0
    last_generation_tokens = 0
    last_clear_cache_tokens = 0

    # Resume from checkpoint if available
    if resume_from_checkpoint is not None:
        training_logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
        checkpoint_info = load_checkpoint(
            filename=resume_from_checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler
        )
        total_tokens_seen = checkpoint_info.get("tokens_seen", 0)
        best_val_lm_loss = checkpoint_info.get("loss", float("inf"))

    # Training loop
    while total_tokens_seen < training_args.max_train_tokens and not stop_early:
        train_loss, train_lm_loss, train_aux_loss, train_ppl, tokens_seen, stop_early = train(
            model, train_loader, optimizer, scheduler, training_args, generation_args, scaler
        )
        val_loss, val_lm_loss, val_aux_loss, val_ppl, = validate(
            model, val_loader, training_args, max_batches=training_args.max_eval_batches
        )
        total_tokens_seen += tokens_seen
        training_logger.info(f"Total tokens seen: {total_tokens_seen}")
        print(f"Total tokens seen: {total_tokens_seen}")

        # Log every >= logging_tokens_freq tokens
        if total_tokens_seen - last_logged_tokens >= training_args.logging_tokens_freq:
            last_logged_tokens = total_tokens_seen
            # Update best stats
            if train_lm_loss < best_train_lm_loss:
                best_train_lm_loss = train_lm_loss
            if train_ppl < best_train_ppl:
                best_train_ppl = train_ppl
            if val_ppl < best_val_ppl:
                best_val_ppl = val_ppl

            # Save best checkpoint based on validation lm loss
            if val_lm_loss < best_val_lm_loss:
                best_val_lm_loss = val_lm_loss
                early_stopping_counter = 0
                training_logger.info(f"New best validation LM loss: {val_lm_loss}")
                # Save best checkpoint
                best_path = save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    tokens_seen=total_tokens_seen,
                    loss=val_lm_loss,
                    training_args=training_args,
                    model_args=model_args,
                    scaler=scaler,
                    is_best=True
                )
                training_logger.info(f"Saved checkpoint to {best_path}")
            else:
                training_logger.info("+1 added to early stopping counter")
                early_stopping_counter += 1

            if early_stopping_counter >= early_stopping_threshold:
                training_logger.info(f"Early stopping activated at {total_tokens_seen} tokens seen")
                break
            
            # Logging
            training_logger.info(f"Total tokens seen: {total_tokens_seen}")
            training_logger.info(f" - Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")
            training_logger.info(f" - Train LM Loss: {train_lm_loss:.4f} | Validation LM Loss: {val_lm_loss:.4f}")
            training_logger.info(f" - Train Aux Loss: {train_aux_loss:.4f} | Validation Aux Loss: {val_aux_loss:.4f}")
            training_logger.info(f" - Train PPL: {train_ppl:.4f} | Validation PPL: {val_ppl:.4f}")
        
        # Save regular checkpoint
        if total_tokens_seen - last_save_tokens >= training_args.logging_tokens_freq:
            last_save_tokens = total_tokens_seen
            save_path = save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                tokens_seen=total_tokens_seen,
                loss=val_lm_loss,
                training_args=training_args,
                model_args=model_args,
                scaler=scaler,
                is_best=False
            )
            training_logger.info(f"Saved regular checkpoint to {save_path}")
        # Test generation every n tokens for coherent generation
        if total_tokens_seen - last_generation_tokens >= generation_args.generation_frequency:
            last_generation_tokens = total_tokens_seen
            prompt = "Once upon a time, "
            generated_text = token_generator.generate_tokens(
                prompt=prompt, 
                generation_args=generation_args, 
                tokenizer=tokenizer
            )
            training_logger.info(f"{prompt} -> {generated_text}")

        # Clear CUDA cache every n tokens
        if total_tokens_seen - last_clear_cache_tokens >= training_args.clear_cache_freq:
            last_clear_cache_tokens = total_tokens_seen
            if device == "cuda":
                torch.cuda.empty_cache()
        
    # Training loop complete
    training_logger.info("Training complete!")
    training_logger.info(f"Best Train LM Loss: {best_train_lm_loss:.4f} | Best Validation LM Loss: {best_val_lm_loss:.4f}")
    training_logger.info(f"Best Train PPL: {best_train_ppl:.4f} | Best Validation PPL: {best_val_ppl:.4f}")

if __name__ == "__main__":
    main(
        dataset_name="tiiuae/falcon-refinedweb",
        text_field="content",
        early_stopping_threshold=3,
        resume_from_checkpoint=None,
        max_samples=None
    )
