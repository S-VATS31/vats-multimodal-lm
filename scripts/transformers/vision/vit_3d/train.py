from configs.setup_env import device

import warnings
import logging
from typing import Optional, Tuple, List

from configs.transformers.vision.vit_3d.model_args.model_args_large import ModelArgs
from configs.transformers.vision.vit_3d.training_args import TrainingArgs
from src.transformers.vision.vit_3d.model import VideoTransformer
from training.transformers.vision.setup_training_components import get_training_components
from training.transformers.vision.vit_3d.train_val.training_loop import train
from training.transformers.vision.vit_3d.train_val.validation_loop import validate
from data.transformers.vision.vit_3d.setup_data import setup_loaders
from utils.transformers.vision.vit_3d.checkpointing import save_checkpoint, load_checkpoint
from utils.transformers.vision.visualization import plot_metrics
from utils.setup_logger import setup_logger

# Set up logger
logger = setup_logger(
    name="train_logger",
    log_file="training.log",
    level=logging.INFO
)

def main(
    patience: Optional[None],
    resume_from_checkpoint: Optional[str] = None
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Main training loop for 3D ViT.
    
    Args:
        patience (int): Number of epochs to wait before stopping training early.

        resume_from_checkpoint (Optional[str]): Checkpoint path to resume from if given.
    Returns:
        Tuple:
            - List[float]: List containing training losses.
            - List[float]: List containing training accuracies
            - List[float]: List containing validation losses.
            - List[float]: List containing validation accuracies.
    """
    if patience is None:
        warnings.warn(
            "Got patience=None, not implementing early stopping."
        )
    # Initialize loss, start epoch, and early stop counter
    # we will update train/validation loss, start epoch if checkpoint given
    best_loss = float("inf")
    start_epoch = 0
    early_stop_counter = 0

    # Initialize lists to hold losses/accuracies
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    # Initialize model and training hyperparameters
    model_args = ModelArgs()
    training_args = TrainingArgs()
    logger.info("Initialized model and training arguments")

    # Initialize model
    model = VideoTransformer(model_args).to(device)
    logger.info("Initialized model and training arguments.")

    # Initialize training components
    optimizer, scheduler, scaler = get_training_components(model, training_args)
    if scaler is None:
        warnings.warn(
            "got non-GPU device, autocast not available, training in float32."
        )
    logger.info("Initialized optimizer, scheduler, and scaler")

    # Get training and validation loaders
    train_loader, val_loader = setup_loaders(model_args, training_args)

    # Resume from checkpoint if given
    if resume_from_checkpoint is not None:
        try:
            checkpoint_info = load_checkpoint(
                save_path=resume_from_checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=device
            )
            # use .get() for fallback if updating fails silently
            start_epoch = checkpoint_info.get("epoch", 0) + 1
            best_loss = checkpoint_info.get("loss", float("inf"))
        except FileNotFoundError as e:
            logger.info(f"Could not find {resume_from_checkpoint}, {e}")
        except Exception as e:
            logger.info(f"Failed to load {resume_from_checkpoint}, {e}")

    # Log stats before training
    # TODO: log hyperparams, training lengths, scheduler/optim/scaler types, etc.

    logger.info(f"Starting training from epoch {start_epoch}, best loss of {best_loss}")
    # Start training loop
    for epoch in range(start_epoch, training_args.epochs):
        logger.info(f"\nEpoch {epoch + 1}/{training_args.epochs}")
        logger.info("-" * 50)
        
        # Train
        train_loss, train_acc = train(
            model, train_loader, optimizer, 
            training_args, scaler, epoch
        )

        # Validate
        val_loss, val_acc = validate(
            model, val_loader, training_args
        )

        # Update lr
        scheduler.step()

        # Update lists for graphing later
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        

        # Log epoch results
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        logger.info(f"Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}")

        # Save regular checkpoint
        if training_args.save_checkpoint_freq == 0:
            try:
                regular_checkpoint = save_checkpoint(
                    model, optimizer, scheduler, epoch,
                    val_loss, training_args, model_args, scaler, False
                )
                logger.info(f"Succesfully saved checkpoint to {regular_checkpoint}")
            except Exception as e:
                logger.info(f"Failed to save regular checkpoint, {e}")

        # Save best checkpoint
        if val_loss < best_loss:
            best_loss = val_loss
            early_stop_counter += 1
            try:
                best_checkpoint = save_checkpoint(
                    model, optimizer, scheduler, epoch,
                    val_loss, training_args, model_args, scaler, True
                )
                logger.info(f"Succesfully saved best checkpoint to {best_checkpoint}")
            except Exception as e:
                logger.info(f"Failed to save best checkpoint: {e}")
        else:
            # failed epoch, add to early stopping counter
            early_stop_counter += 1 

        # early stopping triggered
        if patience is not None and early_stop_counter >= patience:
            logger.info(f"Early stopping reached with best loss of {best_loss} on epoch {epoch}")
            break

    return train_losses, train_accuracies, val_losses, val_accuracies

if __name__ == "__main__":
    train_losses, train_accuracies, val_losses, val_accuracies = main(
        patience=3, resume_from_checkpoint=None
    )
    plot_metrics(train_losses, train_accuracies, val_losses, val_accuracies)
