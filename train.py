from configs.setup_env import device, dtype, logger

from configs.training_args import TrainingArgs
from configs.model_args.model_args_medium import ModelArgs

import os
import math
from typing import Dict, List, Tuple, Optional, Union, Generator, Iterator

import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader, random_split
from torch.optim import AdamW
from torch.amp import GradScaler
from transformers import AutoTokenizer
from datasets import load_dataset

from tqdm import tqdm

from src.model import Transformer
from src.text_quality_filter import TextQualityFilter
from src.deduplication_filter import DeduplicationFilter

class TextDataset(IterableDataset):
    """Iterable text dataset for loading large datasets.

    Args:
        dataset_name (str): HuggingFace dataset name.
        tokenizer: HuggingFace tokenizer.
        max_length (int): Maximum sequence length.
        quality_filter (Optional[TextQualityFilter]): Text quality filter.
        dedup_filter (Optional[DeduplicationFilter]): Deduplication filter.
        buffer_size (int): Size of internal buffer for batch processing filters.
        max_samples (Optional[int]): Maximum number of samples to process (None means proecss all).
        skip_filtering (bool): Whether to skip quality/dedup filtering.
        dataset_config (Optional[str]): Dataset configuration name if needed.
        split (str): Dataset split to use.
        streaming (bool): Whether to use streaming mode.
    """
    def __init__(
        self,
        dataset_name: str,
        tokenizer,
        max_length: int,
        quality_filter: Optional[TextQualityFilter] = None,
        dedup_filter: Optional[DeduplicationFilter] = None,
        buffer_size: int = 1000,
        max_samples: Optional[int] = None,
        skip_filtering: bool = False,
        dataset_config: Optional[str] = None,
        split: str = "train",
        streaming: bool = True,
    ):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.quality_filter = quality_filter
        self.dedup_filter = dedup_filter
        self.buffer_size = buffer_size
        self.max_samples = max_samples
        self.skip_filtering = skip_filtering
        self.dataset_config = dataset_config
        self.split = split
        self.streaming = streaming

        # Prepare target length for padding
        self.target_length = max_length - 1 # Account for shifting for causal LM
        
        logger.info(f"Initialized TextDataset with streaming={streaming}")
        logger.info(f"Dataset: {dataset_name}, Max length: {max_length}")
    
    def load_dataset(self):
        """Load the raw dataset from HuggingFace."""
        try:
            if self.dataset_config is not None:
                # Dataset config given
                dataset = load_dataset(
                    self.dataset_name,
                    self.dataset_config,
                    split=self.split,
                    streaming=self.streaming,
                    trust_remote_code=True
                )
            else:
                # No dataset config given
                dataset = load_dataset(
                    self.dataset_name,
                    split=self.split,
                    streaming=self.streaming,
                    trust_remote_code=True
                )
            return dataset
        except Exception as e:
            logger.error(f"Failed to load dataset {self.dataset_name}: {e}")
            raise

    def process_batch(self, texts: List[str]) -> List[str]:
        """Process a batch of texts through quality and dedup filters.

        Args:
            texts (List[str]): All texts processed as a list.

        Returns:
            List[str]: Filtered and deduplicated texts if filtering is skipped.
        """
        # No filtering, return texts
        if self.skip_filtering:
            return texts

        # Convert to HF dataset format for batch processing
        from datasets import Dataset as HFDataset
        batch_dataset = HFDataset.from_dict({"text": texts})

        # Apply quality filter
        if self.quality_filter:
            batch_dataset = batch_dataset.map(
                self.quality_filter,
                batched=True,
                batch_size=512,
                num_proc=(os.cpu_count() or 8), # or is for case where os.cpu_count()=None
                desc="Quality filtering",
            ).filter(lambda x: x["text"] is not None)
        
        # Apply deduplication filter
        if self.dedup_filter:
            batch_dataset = batch_dataset.map(
                self.dedup_filter,
                batched=True,
                batch_size=512,
                num_proc=1, # Single threading for deduplicatoin
                desc="Deduplication",
            ).filter(lambda x: x["text"] is not None)
        
        return batch_dataset["text"]
    
    def tokenize(self, text: str, model_args: ModelArgs) -> Optional[Dict[str, torch.Tensor]]:
        """Tokenize a single text and format for causal language modeling.
        
        Args:
            text (str): String of input text.
            model_args (ModelArgs): Model hyperparameters.

        Returns:
            Optional[Dict[str, torch.Tensor]]: Dictionary containing input_ids, labels, and attention_mask.
        """
        try:
            # Tokenize the text
            tokens = self.tokenizer(
                text,
                max_length=model_args.max_seq_len,
                truncation=True,
                padding=False,
                add_special_tokens=True,
                return_attention_mask=False,
            )["input_ids"]
            
            # Add EOS token if not present
            if tokens[-1] != self.tokenizer.eos_token_id:
                tokens.append(self.tokenizer.eos_token_id)
            
            # Skip very short sequences
            if len(tokens) < 10:
                return None
            
            # Truncate if too long
            if len(tokens) > self.max_length:
                tokens = tokens[:self.max_length]
            
            # Create input_ids and labels for causal LM
            input_ids = tokens[:-1]
            labels = tokens[1:]
            
            # Pad to target length
            if len(input_ids) < self.target_length:
                pad_length = self.target_length - len(input_ids)
                input_ids.extend([self.tokenizer.pad_token_id] * pad_length)
                labels.extend([-100] * pad_length) # -100 is ignored in loss
            
            # Create attention mask
            real_length = min(len(tokens) - 1, self.target_length)
            attention_mask = [1.0] * real_length + [0.0] * (self.target_length - real_length)
            
            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.int64),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.float32),
                'labels': torch.tensor(labels, dtype=torch.int64),
            }
            
        except Exception as e:
            logger.warning(f"Failed to tokenize text: {e}")
            return None

    def worker_info(self) -> Tuple[int, int]:
        """Get worker information for multi-worker DataLoader support.
        
        Returns:
            Tuple[int, int]: A tuple containing worker id and and number of workers.
        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Single worker
            return 0, 1
        else:
            # Multiple workers - split data across workers
            return worker_info.id, worker_info.num_workers
    
    def stream_examples(self) -> Generator[Dict[str, torch.Tensor], None, None]:
        """Stream tokenized examples from the dataset.
        
        Returns:
            Generator[Dict[str, torch.Tensor], None, None]: Generator to lazily load examples into memory.
        """
        worker_id, num_workers = self.worker_info()
        
        # Load raw dataset
        dataset = self.load_dataset()
        
        # Buffer for batch processing filters
        text_buffer = []
        sample_count = 0
        processed_count = 0
        
        logger.info(f"Worker {worker_id}/{num_workers} starting to stream examples")
        
        try:
            for i, example in enumerate(dataset):
                # Skip examples for other workers
                if i % num_workers != worker_id:
                    continue
                
                # Check max samples limit
                if self.max_samples and sample_count >= self.max_samples:
                    break
                
                sample_count += 1
                
                # Extract text content
                text = example["text"].strip()
                
                # Skip empty texts
                if not text:
                    continue
                
                # Add to buffer for batch processing
                text_buffer.append(text)
                
                # Process buffer when it's full
                if len(text_buffer) >= self.buffer_size:
                    processed_texts = self.process_batch(text_buffer)
                    
                    # Tokenize and yield each processed text
                    for processed_text in processed_texts:
                        tokenized = self.tokenize(processed_text, ModelArgs())
                        if tokenized is not None:
                            processed_count += 1
                            yield tokenized
                    
                    # Clear buffer
                    text_buffer = []
                    
                    # Log progress periodically
                    if processed_count % 1000 == 0:
                        logger.info(f"Worker {worker_id}: processed {processed_count} examples")
            
            # Process remaining texts in buffer
            if text_buffer:
                processed_texts = self.process_batch(text_buffer)
                for processed_text in processed_texts:
                    tokenized = self.tokenize(processed_text)
                    if tokenized is not None:
                        processed_count += 1
                        yield tokenized

        except Exception as e:
            logger.error(f"Error in worker {worker_id}: {e}")
            raise
        
        logger.info(f"Worker {worker_id} finished: {processed_count} examples yielded from {sample_count} samples")
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterator function for IterableDataset.
        
        Returns:
            Iterator[Dict[str, torch.Tensor]]: Returning tokenized examples in the form of input_ids.
        """
        return self.stream_examples()

def create_dataloader(
    dataset_name: str,
    tokenizer,
    max_length: int,
    training_args: TrainingArgs,
    quality_filter: Optional[TextQualityFilter] = None,
    dedup_filter: Optional[DeduplicationFilter] = None,
    max_samples: Optional[int] = None,
    **dataset_kwargs
) -> DataLoader:
    """
    Create a DataLoader with lazy-loading TextDataset.
    
    Args:
        dataset_name (str): HuggingFace dataset name.
        tokenizer: HuggingFace tokenizer.
        max_length (int): Maximum sequence length.
        batch_size (int): Batch size.
        quality_filter (Optional[TextQualityFilter]): Text quality filter.
        dedup_filter (Optional[DeduplicationFilter]): Deduplication filter.
        max_samples (int): Maximum number of samples to process.
        num_workers (int): Number of DataLoader workers.
        pin_memory (bool): Whether to pin memory.
        drop_last (bool): Whether to drop last incomplete batch.
        **dataset_kwargs: Additional arguments for TextDataset.

    Returns:
        DataLoader: DataLoader with lazy-loading dataset.
    """
    # Initialize TextDataset
    dataset = TextDataset(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        max_length=max_length,
        quality_filter=quality_filter,
        dedup_filter=dedup_filter,
        max_samples=max_samples,
        **dataset_kwargs
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=training_args.batch_size,
        num_workers=training_args.num_workers,
        pin_memory=training_args.pin_memory,
        drop_last=training_args.drop_last,
    ) # shuffle=True not supported for IterableDataset
    
    return dataloader

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
        return lm_loss, lm_loss, torch.tensor(0.0, device=lm_loss.device)

def compute_perplexity(loss: float) -> float:
    """Compute perplexity using the LM loss.
    
    Args:
        loss (float): LM loss used to compute perplexity.

    Returns:
        float: Perplexity computed by taking the exponent of the loss.
    """
    return math.exp(loss)

def train_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    optimizer,
    training_args: TrainingArgs,
    device: torch.device,
    step: int,
    scaler: Optional[GradScaler] = None,
) -> Dict[str, Union[float, bool]]:
    """Single training step with AMP support.
    
    Args:
        model (nn.Module): Transformer architecture.
        batch (Dict[str, torch.Tensor]): Dictionary containing input_ids, labels, and attention mask.
        optimizer: PyTorch optimizer.
        training_args (TrainingArgs): Training hyperparameters.
        device (torch.device): Accelerator at use.
        step (int): Current step during training.
        scaler (Optional[GradScaler]): Gradient scaling for bf16/fp16 gradients.

    Returns:
        Dict[str, Union[float, bool]]: Dictionary containing loss and whether the step was a success.
    """
    try:
        # Get input_ids, labels and attention mask
        input_ids = batch['input_ids'].to(device, non_blocking=training_args.pin_memory)
        labels = batch['labels'].to(device, non_blocking=training_args.pin_memory)
        attention_mask = batch['attention_mask'].to(device, non_blocking=training_args.pin_memory)

        # Forward pass with AMP
        if scaler is not None:
            with torch.amp.autocast(device_type=device.type, dtype=dtype):
                logits, _, aux_loss = model(input_ids, padding_mask=attention_mask, use_cache=False)
                loss, lm_loss, aux_loss_val = compute_loss(logits, labels, training_args, aux_loss)
                loss = loss / training_args.grad_accum_steps # Scale loss by gradient accumulation steps
        else:
            # Standard FP32 forward pass
            logits, _, aux_loss = model(input_ids, padding_mask=attention_mask, use_cache=False)
            loss, lm_loss, aux_loss_val = compute_loss(logits, labels, training_args, aux_loss)
            loss = loss / training_args.grad_accum_steps # Scale loss by gradient accumulation steps

        # Backward pass with gradient scaling
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        return {
            'loss': loss.item() * training_args.grad_accum_steps,
            'lm_loss': lm_loss.item(),
            'aux_loss': aux_loss_val.item(),
            'success': True
        }

    # Catch OOM error
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.warning(f"OOM at step {step}, emptying CUDA cache.")
            if device.type == "cuda":
                torch.cuda.empty_cache()
            optimizer.zero_grad()
            return {'success': False, 'error': 'oom'}
        else:
            logger.error(f"Training step failed: {e}")
            return {'success': False, 'error': str(e)}

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader, 
    optimizer, 
    scheduler, 
    training_args: TrainingArgs, 
    device: torch.device, 
    epoch: int, 
    scaler: Optional[GradScaler] = None,
) -> Tuple[float, float, float]:
    """Train for one epoch with AMP support.
    
    Args:
        model (nn.Module): Transformer architecture.
        dataloader (DataLoader): PyTorch DataLoader.
        optimizer: PyTorch optimizer.
        scheduler: PyTorch scheduler.
        training_args (TrainingArgs): Training hyperparameters.
        device (torch.device): Accelerator at use.
        epoch (int): Current epoch during training.
        scaler (Optional[GradScaler]): Gradient scaler to scale bf16/fp16 gradients.

    Returns:
        Tuple[float, float, float]: Tuple containing total loss, lm loss, and aux loss.
            - float: Total loss scaled by succesful steps.
            - float: LM loss scaled by succesful steps.
            - float: Aux loss scaled by succesful steps.
    """
    model.train()

    # Initialization
    total_loss = 0
    total_lm_loss = 0
    total_aux_loss = 0
    successful_steps = 0
    skipped_steps = 0

    # Set up pbar
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
    optimizer.zero_grad()

    # Loop through all steps
    for step, batch in enumerate(progress_bar):
        result = train_step(model, batch, optimizer, training_args, device, step, scaler)

        # If step was successful, accumulate loss
        if result['success']:
            total_loss += result['loss']
            total_lm_loss += result['lm_loss']
            total_aux_loss += result['aux_loss']
            successful_steps += 1
        else:
            skipped_steps += 1
            # If 10% of steps fail, stop epoch
            if skipped_steps > len(dataloader) * 0.1:
                logger.error("Too many failed steps, stopping epoch")
                break

        # Gradient accumulation and optimizer step
        if (step + 1) % training_args.grad_accum_steps == 0:
            if successful_steps > 0:
                # AMP available
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    # Clip L2 Norm
                    nn.utils.clip_grad_norm_(model.parameters(), training_args.clip_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                # No AMP available
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), training_args.clip_grad_norm)
                    optimizer.step()
                scheduler.step()
            optimizer.zero_grad()

        # Logging
        if (step + 1) % training_args.logging_steps == 0 and successful_steps > 0:
            avg_loss = total_loss / successful_steps
            avg_lm_loss = total_lm_loss / successful_steps
            avg_aux_loss = total_aux_loss / successful_steps
            lr = scheduler.get_last_lr()[0]

            progress_bar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "lm_loss": f"{avg_lm_loss:.4f}",
                "aux_loss": f"{avg_aux_loss:.4f}",
                "lr": f"{lr:.2e}",
                "skipped_steps": skipped_steps
            })

    # Handle remaining gradients
    if len(dataloader) % training_args.grad_accum_steps != 0:
        if successful_steps > 0:
            # AMP available
            if scaler is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), training_args.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            # AMP not available
            else:
                nn.utils.clip_grad_norm_(model.parameters(), training_args.clip_grad_norm)
                optimizer.step()
        optimizer.zero_grad()

    # No succesful steps, all loss = inf
    if successful_steps == 0:
        logger.error("No successful training steps in epoch")
        return float('inf'), float('inf'), float('inf')

    return (total_loss / successful_steps,
            total_lm_loss / successful_steps,
            total_aux_loss / successful_steps)

def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    training_args: TrainingArgs,
    device: torch.device, 
    max_batches: Optional[int] = None,
) -> Tuple[float, float, float]:
    """Evaluate model with AMP support.
    
    Args:
        model (nn.Module): Transformer architecture
        dataloader (DataLoader): PyTorch DataLoader.
        training_args (TrainingArgs): Training hyperparameters.
        device (torch.device): Accelerator at use.
        max_batches (Optional[int]): Max batches to evaluate on.

    Returns:
    Tuple[float, float, float]: Tuple containing total loss, lm loss, and aux loss.
        - float: Total loss scaled by succesful steps.
        - float: LM loss scaled by succesful steps.
        - float: Aux loss scaled by succesful steps.
    """
    model.eval() # No dropout

    # Initialize loss and batches
    total_loss = 0
    total_lm_loss = 0
    total_aux_loss = 0
    successful_batches = 0

    # Turn off gradient tracking for evaluation
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            # Break if max batches reached
            if max_batches is not None and i >= max_batches:
                break

            try:
                # Get input_ids, labels, and attention_mask
                input_ids = batch['input_ids'].to(device, non_blocking=training_args.pin_memory)
                labels = batch['labels'].to(device, non_blocking=training_args.pin_memory)
                attention_mask = batch['attention_mask'].to(device, non_blocking=training_args.pin_memory)

                with torch.amp.autocast(device_type=device.type, dtype=dtype):
                    logits, _, aux_loss = model(input_ids, padding_mask=attention_mask, use_cache=False)
                    loss, lm_loss, aux_loss_val = compute_loss(logits, labels, training_args, aux_loss)

                # Accumulate loss
                total_loss += loss.item()
                total_lm_loss += lm_loss.item()
                total_aux_loss += aux_loss_val.item()
                successful_batches += 1
            
            # Catch OOM error
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning("OOM during evaluation, skipping batch")
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    continue
                else:
                    logger.warning(f"Evaluation error: {e}")
                    continue

    # No succesful batches, all loss = inf
    if successful_batches == 0:
        logger.error("No successful evaluation batches")
        return float('inf'), float('inf'), float('inf')

    return (total_loss / successful_batches,
            total_lm_loss / successful_batches,
            total_aux_loss / successful_batches)

def save_checkpoint(
    model: nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    step: int,
    loss: float,
    training_args: TrainingArgs,
    model_args: ModelArgs,
    scaler: Optional[GradScaler] = None,
    is_best: bool = False,
) -> str:
    """Save checkpoint to .pt file.
    
    Args:
        model (nn.Module): Transformer architecture.
        optimizer: PyTorch optimizer.
        scheduler: PyTorch scheduler.
        epoch (int): Current epoch to save checkpoint to.
        step (int): Current step to save checkpoint to.
        loss (float): Current loss to save checkpoint to.
        training_args (TrainingArgs): Training hyperparameters.
        model_args (ModelArgs): Model hyperparameters.
        scaler (Optional[GradScaler]): Save if GradScaler is not None.
        is_best (bool): Whether the current checkpoint contains the lowest validation loss or not.

    Returns:
        str: Returns path to save checkpoint so it can be loaded later.
    """
    try:
        # Create checkpoint data
        checkpoint_data = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
            'training_args': training_args.__dict__,
            'model_args': model_args.__dict__,
        }
        
        # Add scaler state if using AMP
        if scaler is not None:
            checkpoint_data['scaler_state_dict'] = scaler.state_dict()
        
        # Create filename
        filename = "best_model.pt" if is_best else f"checkpoint_step_{step}_epoch{epoch}.pt"
        
        # Load checkpoint data to filename
        torch.save(checkpoint_data, filename)
        logger.info(f"Succesfully saved checkpoint to {filename}")
        
        return filename

    except Exception as e:
        logger.error(f"Failed to save checkpoint as {filename}: {e}")
        raise # We don't want to load faulty checkpoints, so no return

def load_checkpoint(
    filename: str,
    model: nn.Module,
    optimizer,
    scheduler,
    scaler: Optional[GradScaler] = None,
    device: torch.device = None,
) -> Dict[str, Union[int, float]]:
    """Load checkpoint from saved .pt file.
    
    Args:
        filename (str): Filename where checkpoint is saved.
        model (nn.Module): Transformer architecture.
        optimizer: PyTorch optimizer.
        scheduler: PyTorch scheduler.
        scaler (Optional[GradScaler]): Gradient scaling for bf16/fp16 gradients.
        device (torch.device): Accelerator at use.

    Returns:
        Dict[str, Union[int, float]]: State dict returning current step, epoch, and loss.
            - int: Current epoch.
            - int: Current step.
            - float: Current loss.
    """
    try:
        # Load checkpoint
        checkpoint = torch.load(filename, map_location=device)
        
        # Load states
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load scaler state dict if using AMP
        if scaler is not None and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        logger.info(f"Succesfully loaded checkpoint from {filename}")
        
        return {
            'epoch': checkpoint['epoch'],
            'step': checkpoint['step'],
            'loss': checkpoint['loss'],
        }
        
    except Exception as e:
        logger.error(f"Failed to load checkpoint from {filename}: {e}")
        raise

def lr_scheduler(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create a linear learning rate schedule with warmup.
    
    Args:
        optimizer: PyTorch optimizer.
        num_warmup_steps (int): Number of warmup steps.
        num_training_steps (int): Number of training steps.
        last_epoch (int): Last epoch parameter for LambdaLR.
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, 
            float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

def setup_training_components(
    model: nn.Module,
    training_args: TrainingArgs,
    num_training_steps: int,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler, Optional[GradScaler]]:
    """Setup optimizer, scheduler, and scaler for training.
    
    Args:
        model (nn.Module): Transformer architecture.
        training_args (TrainingArgs): Training hyperparameters.
        num_training_steps (int): Number of training steps.

    Returns:
        Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler, Optional[GradScaler]]:
            - torch.optim.Optimizer: AdamW optimizer.
            - torch.optim.lr.scheduler._LRScheduler: Custom consine decay lr scheduler.
            - Optional[GradScaler]: Gradient scaling for bf16/fp16 gradients.
    """
    # Setup optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=training_args.learning_rate,
        betas=training_args.betas,
        eps=training_args.epsilon,
        weight_decay=training_args.weight_decay,
    )

    # Setup scheduler
    num_warmup_steps = int(training_args.warmup_ratio * num_training_steps)
    scheduler = lr_scheduler(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    
    # Setup gradient scaler for AMP
    scaler = GradScaler() if device.type == "cuda" else None
    
    return optimizer, scheduler, scaler

def main(
    dataset_name: str,
    resume_from_checkpoint: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> None:
    """Main training loop.
    
    Args:
        dataset_name (str): Name of the dataset to be downloaded.
        resume_from_checkpoint (Optional[str]): File path to resume from checkpoint.
        max_samples (Optional[int]): Number of samples to download (None for all samples).
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
    logger.info("Initialized model and training arguments.")

    # Initialize model
    model = Transformer(model_args).to(device)
    logger.info("Initialized model")

    # Count parameters (will be logged)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
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

    # Initialize start epoch and loss
    start_epoch = 0
    best_loss = float('inf') # Initialize loss

    # Resume from checkpoint if provided
    try:
        if resume_from_checkpoint is not None:
            logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
            checkpoint_info = load_checkpoint(
                resume_from_checkpoint, model, optimizer, scheduler, scaler, device
            )
            start_epoch = checkpoint_info['epoch'] + 1
            best_loss = checkpoint_info['loss']
    except Exception as e:
        logger.info(f"Failed to resume from {resume_from_checkpoint}: {e}")

    # Dataset info
    logger.info("DATASET INFORMATION")
    logger.info("=" * 50)
    logger.info(f"Training dataset: {dataset_name}")
    logger.info(f"Streaming mode: True")
    logger.info(f"Max samples per epoch: {max_samples if max_samples else 'All available'}")
    logger.info(f"Estimated steps per epoch: {estimated_steps_per_epoch}\n")

    # Tokenization info
    logger.info("TOKENIZATION INFORMATION")
    logger.info("=" * 50)
    logger.info(f"Pad token: {tokenizer.pad_token} | EOS token: {tokenizer.eos_token}")
    logger.info(f"Pad token id: {model_args.pad_token_id} | EOS token id: {model_args.eos_token_id}")
    logger.info(f"Vocab size: {model_args.vocab_size}")
    logger.info(f"Max sequence length: {model_args.max_seq_len}\n")

    # Model info
    logger.info("MODEL INFORMATION")
    logger.info("=" * 50)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}\n")

    # Training components
    logger.info("TRAINING COMPONENTS")
    logger.info("=" * 50)
    logger.info(f"Optimizer: {type(optimizer).__name__}")
    logger.info(f"Scheduler: {type(scheduler).__name__}")
    logger.info(f"Scaler available: {bool(scaler)}\n") # If device.type == cuda: True, else scaler=None: False

    # Training steps/epochs
    logger.info("TRAINING LENGTHS")
    logger.info("=" * 50)
    logger.info(f"Number of Epochs: {training_args.epochs}")
    logger.info(f"Estimated Number of Steps: {num_training_steps}")
    logger.info(f"Number of warmup steps: {int(training_args.warmup_ratio * num_training_steps)}\n")

    # Training loop
    logger.info("Training starting...")

    for epoch in range(start_epoch, training_args.epochs):
        logger.info(f"Starting epoch {epoch + 1}/{training_args.epochs}")

        # Train for one epoch
        train_loss, train_lm_loss, train_aux_loss = train_epoch(
            model, train_loader, optimizer, scheduler, training_args, device, epoch, scaler
        )

        # Evaluate for one epoch
        val_loss, val_lm_loss, val_aux_loss = evaluate_model(
            model, val_loader, training_args, device, training_args.max_eval_batches
        )

        # Log training loss and perplexity
        logger.info(f"Epoch {epoch + 1} Training Loss & Perplexity:")
        logger.info(f"Train Loss: {train_loss:.4f} | Train LM Loss: {train_lm_loss:.4f} | Train Aux Loss: {train_aux_loss:.4f}")
        logger.info(f"Train Perplexity: {compute_perplexity(train_lm_loss):.4f}")

        # Log validation loss and perplexity
        logger.info(f"Epoch {epoch + 1} Validation Loss & Perplexity:")
        logger.info(f"Val Loss: {val_loss:.4f} | Val LM Loss: {val_lm_loss:.4f} | Val Aux Loss: {val_aux_loss:.4f}")
        logger.info(f"Val Perplexity: {compute_perplexity(val_lm_loss):.4f}")

        # Save regular checkpoint
        if (epoch + 1) % training_args.save_freq == 0:
            checkpoint_path = save_checkpoint(
                model, optimizer, scheduler, epoch, 0, train_loss,
                training_args, model_args, scaler, is_best=False
            )
            logger.info(f"Saved checkpoint to {checkpoint_path}")

        # Save best model (lowest validation loss)
        if val_loss < best_loss:
            best_loss = val_loss
            best_checkpoint_path = save_checkpoint(
                model, optimizer, scheduler, epoch, 0, val_loss,
                training_args, model_args, scaler, is_best=True
            )
            logger.info(f"New best model saved to {best_checkpoint_path}")

        # Clean up GPU memory
        if device.type == "cuda":
            torch.cuda.empty_cache()
    
    logger.info("Training completed!")

if __name__ == "__main__":
    try:
        main(
            dataset_name="tiiuae/falcon-refinedweb",
            resume_from_checkpoint=None,
            max_samples=None
        )
    except Exception as e:
        logger.error(f"Failure occured when running main training loop: {e}")
