from typing import List
import matplotlib.pyplot as plt

def plot_metrics(
    train_lm_losses: List[float], 
    val_lm_losses: List[float], 
    train_perplexities: List[float], 
    val_perplexities: List[float],
) -> None:
    """Plot training and validation loss and perplexity over epochs.
    
    Args:
        train_lm_losses (List[float]): List of training language modeling losses.
        val_lm_losses (List[float]): List of validation language modeling losses.
        train_perplexities (List[float]): List of training perplexities.
        val_perplexities (List[float]): List of validation perplexities.
    """
    plt.figure(figsize=(12, 8))

    # Training Loss
    plt.subplot(2, 2, 1)
    plt.plot(range(1, len(train_lm_losses) + 1), train_lm_losses, label="Train Loss", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()

    # Validation Loss
    plt.subplot(2, 2, 2)
    plt.plot(range(1, len(val_lm_losses) + 1), val_lm_losses, label="Validation Loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss")
    plt.legend()

    # Training Perplexity
    plt.subplot(2, 2, 3)
    plt.plot(range(1, len(train_perplexities) + 1), train_perplexities, label="Train Perplexity", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.title("Training Perplexity")
    plt.legend()

    # Validation Perplexity
    plt.subplot(2, 2, 4)
    plt.plot(range(1, len(val_perplexities) + 1), val_perplexities, label="Validation Perplexity", color="purple")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.title("Validation Perplexity")
    plt.legend()

    plt.tight_layout()
    plt.show()
