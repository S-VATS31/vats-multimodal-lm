import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from configs.transformers.nlp.training_args import TrainingArgs

def compute_loss(
    logits: torch.Tensor,
    labels: torch.Tensor, 
    training_args: TrainingArgs,
    aux_loss: Optional[torch.Tensor] = None, 
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute training loss with optional auxiliary loss.
    
    Args:
        logits (torch.Tensor): Logits tensor of shape [B, T, V].
        labels (torch.Tensor): Labels tensor of shape [B, T].
        training_args (TrainingArgs): Training hyperparameters.
        aux_loss (torch.Tensor): aux_loss scalar tensor.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple containing total loss, lm loss, aux loss.
            - torch.Tensor: Total loss.
            - torch.Tensor: Language modeling loss.
            - torch.Tensor: Auxiliary loss.
    """
    # Shift logits/labels for CE
    shift_logits = logits.contiguous().view(-1, logits.size(-1))
    shift_labels = labels.contiguous().view(-1)

    # Tell model to ignore value of -100
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    lm_loss = criterion(shift_logits, shift_labels)

    if aux_loss is not None: 
        # Calculate total loss if aux_loss given
        total_loss = lm_loss + training_args.aux_loss_weight * aux_loss
        return total_loss, lm_loss, aux_loss
    else:
        # Initialize aux loss as 0 tensor
        return lm_loss, lm_loss, torch.tensor(0.0).to(lm_loss.device)

def compute_perplexity(loss: float) -> float:
    """Compute perplexity using the LM loss.
    
    Args:
        loss (float): LM loss used to compute perplexity.

    Returns:
        float: Perplexity computed by taking the exponent of the loss.
    """
    return math.exp(loss)
