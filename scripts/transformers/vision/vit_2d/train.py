from configs.transformers.vision.vit_2d.setup_env import device

from typing import Optional, List, Tuple
import logging

from src.transformers.vision.vit_2d.model import VisionTransformer
from configs.transformers.vision.vit_2d.training_args import TrainingArgs
from configs.transformers.vision.vit_2d.model_args.model_args_large import ModelArgs
from utils.transformers.vision.vit_2d.visualization import plot_metrics
from training.transformers.vision.vit_2d.loops.training_loop import train
from training.transformers.nlp.loops.validation_loop import validate
from data.transformers.vision.vit_2d.setup_data import setup_data_loaders
from save_load_checkpoints.transformers.vision.vit_2d.save_checkpoint import save_checkpoint
from save_load_checkpoints.transformers.vision.vit_2d.load_checkpoint import load_checkpoint
from training.transformers.vision.vit_2d.training_components.setup_training_components import get_training_components

# Set up logger
from utils.setup_logger import setup_logger
training_logger = setup_logger(name="training_logger", log_file="training.log", level=logging.INFO)

def main(
    resume_from_checkpoint: Optional[str] = None, 
    patience: int = 3
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Main training loop.

    Args:
        resume_from (Optional[str]): Path to checkpoint to resume training from.
        patience (int): Number of epochs to wait for improvement before early stopping.

    Returns:
        Tuple[List[float], List[float], List[float], List[float]]:
            - List[float]: List containing training losses.
            - List[float]: List containing training accuracies.
            - List[float]: List containing validation losses.
            - List[float]: List containing validation accuracies.
    """
    start_epoch = 0
    best_loss = float("inf")
    early_stop_counter = 0

    # Initialize lists for visualization
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    # Initialize model arguments and training arguments
    model_args = ModelArgs()
    training_args = TrainingArgs()

    # Initialize model
    model = VisionTransformer(model_args).to(device)

    # Get training components
    optimizer, scheduler, scaler = get_training_components(model, training_args)

    # Get dataloaders
    train_loader, val_loader = setup_data_loaders(model_args, training_args)

    # Resume from checkpoint if provided
    try:
        if resume_from_checkpoint is not None:
            training_logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
            checkpoint_info = load_checkpoint(
                resume_from_checkpoint, model, optimizer, scheduler, scaler, device
            )
            start_epoch = checkpoint_info['epoch'] + 1
            best_loss = checkpoint_info['loss']
    except FileNotFoundError as e:
        training_logger.error(f"Checkpoint: {resume_from_checkpoint} was not found.")
    except RuntimeError as e:
        training_logger.error(f"Failed to resume from {resume_from_checkpoint}: {e}")
    except Exception as e:
        training_logger.error(f"Failed to resume from {resume_from_checkpoint}: {e}")

    # Dataset info
    training_logger.info("DATASET INFORMATION:")
    training_logger.info("=" * 50)
    training_logger.info(f"Number of Training Examples: {len(train_loader)}")
    training_logger.info(f"Number of Validation Examples: {len(val_loader)}")
    training_logger.info(f"Number of Classes: {training_args.num_classes}\n")

    # Training info
    training_logger.info("TRAINING INFORMATION:")
    training_logger.info("=" * 50)
    training_logger.info(f"Optimizer: {type(optimizer).__name__}")
    training_logger.info(f"Scheduler: {type(scheduler).__name__}")
    training_logger.info(f"Scaler is available: {bool(scaler)}\n") # no cuda -> scaler=None -> bool(None) -> False

    # Training lengths info
    training_logger.info("TRAINING LENGTH INFORMATION:")
    training_logger.info("=" * 50)
    training_logger.info(f"Total epochs: {training_args.epochs}")
    training_logger.info(f"Warmup epochs: {training_args.warmup_epochs}")
    training_logger.info(f"Saving regular checkpoint every: {training_args.save_checkpoint_freq} epoch(s)\n")

    # Hyperparameter info
    training_logger.info("HYPERPARAMETER INFORMATION:")
    training_logger.info("=" * 50)
    training_logger.info(f"Learning rate: {training_args.learning_rate}")
    training_logger.info(f"Batch size: {training_args.batch_size}")
    training_logger.info(f"Gradient checkpointing being used: {model_args.use_checkpointing}")

    # Start training
    training_logger.info(f"Starting training from epoch {start_epoch}")

    # Main training loop
    for epoch in range(start_epoch, training_args.epochs):
        training_logger.info(f"\nEpoch {epoch + 1}/{training_args.epochs}")
        training_logger.info("-" * 50)

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

        # Track metrics for visualization
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Log epoch results
        training_logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        training_logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        training_logger.info(f"Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}")

        # Save regular checkpoint
        if (epoch + 1) % training_args.save_checkpoint_freq == 0:
            checkpoint_path = save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss,
                training_args, model_args, scaler, is_best=False
            )
            training_logger.info(f"saved checkpoint to {checkpoint_path}")

        # Save best checkpoint (based on validation loss)
        if val_loss < best_loss:
            best_loss = val_loss
            early_stop_counter = 0
            best_checkpoint_path = save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss,
                training_args, model_args, scaler, is_best=True
            )
            training_logger.info(f"saved best checkpoint to {best_checkpoint_path}")
        else:
            early_stop_counter += 1

        # Early stopping
        if early_stop_counter >= patience:
            training_logger.info(f"Early stopping triggered after {patience} epochs without improvement.")
            break

    return train_losses, train_accuracies, val_losses, val_accuracies

if __name__ == "__main__":
    train_losses, train_accuracies, val_losses, val_accuracies = main()
    plot_metrics(train_losses, train_accuracies, val_losses, val_accuracies)