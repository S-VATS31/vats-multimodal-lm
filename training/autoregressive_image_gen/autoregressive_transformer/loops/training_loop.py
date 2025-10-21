from configs.setup_env import device, dtype

from typing import Dict

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import GradScaler

from src.autoregressive_image_gen.autoregressive_transformer.model import AutoregressiveImageTransformer
from configs.autoregressive_image_gen.autoregressive_transformer.model_args.model_args_xsmall import ModelArgs
from configs.autoregressive_image_gen.autoregressive_transformer.training_args import TrainingArgs
from src.autoregressive_image_gen.vq_vae.vq_vae import VQVAE

class ImageGenTrainer:
    """Autoregressive image generation training module.
    
    Args:
        model_args (ModelArgs): Model hyparameters.
        training_args (TrainingArgs): Training hyperparameters.
        optimizer (Optimizer): PyTorch optimizer.
        scheduler (LambdaLR): PyTorch learning rate scheduler.
    """
    def __init__(
        self, 
        model_args: ModelArgs, 
        training_args: TrainingArgs,
        optimizer: Optimizer,
        scheduler: LambdaLR
    ):
        self.model_args = model_args
        self.training_args = training_args

        self.vq_vae = VQVAE(model_args).to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = GradScaler() if device == "cuda" else None

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Train for a single step.'
        
        Args:
            batch (Dict[str, torch.Tensor]): Batch containing tensors.

        """
        input_ids = batch["input_ids"].to(device)
        image_attention_mask = batch["image_attention_mask"].to(device)
        text_embeddings = batch["text_embeddings"].to(device)
        text_attention_mask = batch["text_attention_mask"].to(device)
        
        # Forward pass through VQVAE
        _, loss, _, _ = self.vq_vae(
            input_ids=input_ids,
            text_embeddings=text_embeddings,
            image_attention_mask=image_attention_mask,
            text_attention_mask=text_attention_mask,
            use_cache=False
        )
        loss /= self.training_args.grad_accum_steps

        # Backprop
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        self.scheduler.step()

        return loss.item() * self.training_args.grad_accum_steps

    def train():
        pass

