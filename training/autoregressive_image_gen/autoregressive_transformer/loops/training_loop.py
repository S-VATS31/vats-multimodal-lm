from configs.setup_env import device, dtype

from typing import Dict, Tuple
import logging

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from tqdm import tqdm

from src.autoregressive_image_gen.autoregressive_transformer.model import AutoregressiveImageTransformer
from configs.autoregressive_image_gen.autoregressive_transformer.model_args.model_args_xsmall import ModelArgs
from configs.autoregressive_image_gen.autoregressive_transformer.training_args import TrainingArgs
from src.autoregressive_image_gen.vq_vae.vq_vae import VQVAE
from utils.setup_logger import setup_logger

logger = setup_logger(
    name="train_logger", log_file="training.log", level=logging.INFO
)

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
        scheduler: LambdaLR,
        dataloader: DataLoader
    ):
        self.model_args = model_args
        self.training_args = training_args
        self.dataloader = dataloader

        self.model = AutoregressiveImageTransformer(model_args).to(device)
        self.vq_vae = VQVAE(model_args).to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = GradScaler() if device.type == "cuda" else None

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, bool]:
        """Train for a single step.'
        
        Args:
            batch (Dict[str, torch.Tensor]): Batch containing tensors.

        Returns:
            Tuple:
                - float: Average training loss over step.
                - bool: Whether the step was succesful or not.
        """
        try:
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
            else:
                loss.backward()

            return (loss.item() * self.training_args.grad_accum_steps, True)
    
        except Exception as e:
            if "out of memory" in str(e).lower():
                logger.info("Out of memory error occured, clearing cache")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            return (float("inf"), False) # return infinite loss if step failed
        
    def train(self) -> float:
        """Train for one epoch.
        
        Returns:
            float: Average training loss.
        """
        self.model.train()

        # Initialization
        total_loss = 0
        successful_steps = 0
        failed_steps = 0

        # progress bar for tracking
        pbar = tqdm(self.dataloader, desc="Training")
        self.optimizer.zero_grad(set_to_none=True)

        # training
        for step, batch in enumerate(pbar):
            loss, success = self.train_step(batch)

            # check if step was success
            if success:
                total_loss += loss
                successful_steps += 1
            else:
                logger.info(f"Step {step} failed.")
                failed_steps += 1
                if failed_steps > self.training_args.max_skipped_steps:
                    logger.info(f"Failed {failed_steps}, ending epoch")
                    break

            # Update weights
            if (step + 1) % self.training_args.grad_accum_steps == 0:
                if successful_steps > 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.training_args.clip_grad_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.training_args.clip_grad_norm)
                        self.optimizer.step()
                    self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

            # log
            if (step + 1) % self.training_args.logging_steps == 0 and successful_steps > 0:
                avg_loss = total_loss / successful_steps
                lr = self.scheduler.get_last_lr()[0]
                pbar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "lr": f"{lr:.2e}",
                    "skipped_steps": failed_steps,
                })

        # Final optimizer flush if partial gradient accumulation left
        if (successful_steps % self.training_args.grad_accum_steps) != 0 and successful_steps > 0:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.training_args.clip_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.training_args.clip_grad_norm)
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        if successful_steps == 0:
            logger.error("No successful training steps in epoch")
            return float("inf")

        return total_loss / successful_steps
