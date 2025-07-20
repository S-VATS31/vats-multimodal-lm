from typing import List
import matplotlib.pyplot as plt

def plot_metrics(
    train_losses: List[float], 
    train_accuracies: List[float], 
    val_losses: List[float], 
    val_accuracies: List[float],
) -> None:
    """Plot training and validation loss and accuracy over epochs.
    
    Args:
        train_losses (List[float]): List of training losses.
        train_accuracies (List[float]): List of training accuracies.
        val_losses (List[float]): List of validation losses.
        val_accuracies (List[float]): List of validation accuracies.
    """
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 8))

    # Training Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()

    # Validation Loss
    plt.subplot(2, 2, 2)
    plt.plot(epochs, val_losses, label="Validation Loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss")
    plt.legend()

    # Training Accuracy
    plt.subplot(2, 2, 3)
    plt.plot(epochs, train_accuracies, label="Train Accuracy", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training Accuracy")
    plt.legend()

    # Validation Accuracy
    plt.subplot(2, 2, 4)
    plt.plot(epochs, val_accuracies, label="Validation Accuracy", color="purple")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Validation Accuracy")
    plt.legend()

    # Display subplots
    plt.tight_layout()
    plt.show()
    