# TODO: Change logic to be GCP to local device
# TODO: Remove all XLA support, GPU only
# TODO: Add paths for saving dataset/tokenized dataset

from configs.setup_env import (
    device,
    dtype,
    logger,
)

from configs.training_args import TrainingArgs
from configs.model_args.model_args_medium import ModelArgs

import os
from typing import Dict, List, Tuple, Optional, Union, Generator
import gc
import gzip
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.amp import GradScaler
from transformers import AutoTokenizer
from datasets import (
    load_dataset,
    load_from_disk,
    Dataset as HFDataset,
)

from tqdm import tqdm

from src.model import Transformer
from src.text_quality_filter import TextQualityFilter
from src.deduplication_filter import DeduplicationFilter

class TextDataset(Dataset):
    """Enhanced dataset for GCP compatibility.
    
    Args:
        texts (Optional[List[str]]): List of input texts.
        tokenizer: HF tokenizer to convert text to input_ids.
        max_length (Optional[int]): Maximum sequence length.
        quality_filter (Optional[TextQualityFilter]): Text filtering for cleaner text.
        dedup_filter (Optional[DeduplicationFilter]): Deduplication filtering to prevent overfitting.
        skip_filtering (bool): Whether to use pre-filtered or filter new texts.
        save_filtered_texts_path (Optional[str]): Path to save filtered texts (supports gs:// URLs).
        save_tokenized_dataset_path (Optional[str]): Path to save tokenized texts (supports gs:// URLs).
    """
    def __init__(
        self,
        texts: Optional[List[str]] = None, 
        tokenizer=None, 
        max_length: Optional[int] = None, 
        quality_filter: Optional[TextQualityFilter] = None, 
        dedup_filter: Optional[DeduplicationFilter] = None, 
        skip_filtering: bool = False,
        save_filtered_texts_path: Optional[str] = None,
        save_tokenized_dataset_path: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.quality_filter = quality_filter
        self.dedup_filter = dedup_filter
        self.gcs_client = storage.Client()
        
        # Filtering step
        if skip_filtering:
            logger.info("Skipping filtering step - using pre-filtered texts")
            self.processed_texts = texts
        else:
            logger.info("Filtering texts in batches")
            self.processed_texts = self._process_texts_in_chunks(texts)
            logger.info(f"Kept {len(self.processed_texts)} texts after filtering from {len(texts)} original")

            # Save filtered texts
            if save_filtered_texts_path is not None:
                try:
                    hf_dataset = HFDataset.from_dict({"text": self.processed_texts})
                    self._save_dataset_to_path(hf_dataset, save_filtered_texts_path)
                    logger.info(f"Saved filtered texts to {save_filtered_texts_path}")
                except Exception as e:
                    logger.error(f"Failed to save filtered texts: {e}")

        # Tokenization and saving
        if save_tokenized_dataset_path is not None:
            try:
                logger.info("Tokenizing and saving dataset...")
                self._tokenize_and_save(self.processed_texts, save_tokenized_dataset_path)
            except Exception as e:
                logger.error(f"Tokenization failed: {e}")
                raise e
    
        # Optional for __getitem__ usage
        self.tokenized_texts = None

    def _is_gcs_path(self, path: str) -> bool:
        """Check if path is a Google Cloud Storage path."""
        return path is not None and path.startswith('gs://')

    def _parse_gcs_path(self, gcs_path: str) -> tuple:
        """Parse GCS path into bucket and blob name."""
        if not gcs_path.startswith('gs://'):
            raise ValueError(f"Invalid GCS path: {gcs_path}")
        
        path_parts = gcs_path[5:].split('/', 1)
        bucket_name = path_parts[0]
        blob_name = path_parts[1] if len(path_parts) > 1 else ""
        
        return bucket_name, blob_name

    def _save_dataset_to_path(self, dataset: HFDataset, path: str):
        """Save dataset to local path or GCS bucket."""
        if self._is_gcs_path(path):
            # Save to GCS
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dataset_path = os.path.join(temp_dir, "dataset")
                dataset.save_to_disk(temp_dataset_path)
                self._upload_directory_to_gcs(temp_dataset_path, path)
        else:
            # Save locally
            dataset.save_to_disk(path)

    def _upload_directory_to_gcs(self, local_dir: str, gcs_path: str):
        """Upload a directory to Google Cloud Storage."""
        bucket_name, blob_prefix = self._parse_gcs_path(gcs_path)
        bucket = self.gcs_client.bucket(bucket_name)
        
        for root, _, files in os.walk(local_dir):
            for file in files:
                local_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_file_path, local_dir)
                blob_name = os.path.join(blob_prefix, relative_path).replace('\\', '/')
                
                blob = bucket.blob(blob_name)
                blob.upload_from_filename(local_file_path)
                logger.info(f"Uploaded {local_file_path} to gs://{bucket_name}/{blob_name}")

    def _download_directory_from_gcs(self, gcs_path: str, local_dir: str) -> None:
        """Download a directory from Google Cloud Storage."""
        bucket_name, blob_prefix = self._parse_gcs_path(gcs_path)
        bucket = self.gcs_client.bucket(bucket_name)
        
        # List all blobs with the prefix
        blobs = bucket.list_blobs(prefix=blob_prefix)
        
        for blob in blobs:
            # Skip directories (blobs ending with '/')
            if blob.name.endswith('/'):
                continue
                
            # Create local file path
            relative_path = os.path.relpath(blob.name, blob_prefix)
            local_file_path = os.path.join(local_dir, relative_path)
            
            # Create directories if they don't exist
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            
            # Download the blob
            blob.download_to_filename(local_file_path)
            logger.info(f"Downloaded gs://{bucket_name}/{blob.name} to {local_file_path}")

    def _process_texts_in_chunks(self, texts: List[str], batch_size: int = 100000) -> List[str]:
        """Efficient batch processing for quality + dedup filtering."""
        filtered = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            dataset = HFDataset.from_dict({"text": batch})

            dataset = dataset.map(
                self.quality_filter,
                batched=True,
                batch_size=512,
                num_proc=4,
                desc="Quality filtering",
            ).filter(lambda x: x["text"] is not None)

            dataset = dataset.map(
                self.dedup_filter,
                batched=True,
                batch_size=512,
                num_proc=1,
                desc="Deduplication",
            ).filter(lambda x: x["text"] is not None)

            filtered.extend(dataset["text"])
        return filtered

    def _tokenize_and_save(self, texts: List[str], save_path: str):
        """Tokenize texts and save the dataset."""
        # Check if final dataset already exists
        if self._dataset_exists(save_path):
            logger.info(f"Final dataset already exists at {save_path}, skipping tokenization")
            return

        logger.info(f"Starting tokenization of {len(texts):,} texts...")

        # Tokenize all texts in batches
        all_tokenized_data = []
        batch_size = 5000  # Process in batches to manage memory

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]

            # Tokenize batch
            tokenized_batch = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=False,
                add_special_tokens=True,
                return_attention_mask=False,
            )["input_ids"]

            # Process each tokenized sequence
            for tokens in tokenized_batch:
                # Add EOS token if not present
                if tokens[-1] != self.tokenizer.eos_token_id:
                    tokens.append(self.tokenizer.eos_token_id)

                # Skip very short sequences
                if len(tokens) < 10:
                    continue

                # Truncate if too long
                if len(tokens) > self.max_length:
                    tokens = tokens[:self.max_length]

                # Create input_ids and labels for causal LM
                input_ids = tokens[:-1]
                labels = tokens[1:]

                # Pad to target length
                target_length = self.max_length - 1
                if len(input_ids) < target_length:
                    pad_length = target_length - len(input_ids)
                    input_ids.extend([self.tokenizer.pad_token_id] * pad_length)
                    labels.extend([-100] * pad_length)

                # Create attention mask
                attention_mask = [1.0] * (len(tokens) - 1) + [0.0] * (target_length - (len(tokens) - 1))

                all_tokenized_data.append({
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': labels,
                })

            # Log progress every 10 batches
            if (i // batch_size) % 10 == 0:
                logger.info(f"Processed {i + len(batch_texts):,} texts... (Current samples: {len(all_tokenized_data):,})")

        # Create and save the final dataset
        if all_tokenized_data:
            logger.info(f"Creating final dataset with {len(all_tokenized_data):,} samples...")
            final_dataset = HFDataset.from_list(all_tokenized_data)
            
            self._save_dataset_to_path(final_dataset, save_path)
            logger.info(f"Successfully saved tokenized dataset to {save_path}")
            
            # Clean up memory
            del final_dataset
            del all_tokenized_data
            gc.collect()
        else:
            logger.warning("No data was processed - all_tokenized_data is empty")

    def _dataset_exists(self, path: str) -> bool:
        """Check if dataset exists at the given path."""
        if self._is_gcs_path(path):
            # Check if GCS path exists
            bucket_name, blob_prefix = self._parse_gcs_path(path)
            bucket = self.gcs_client.bucket(bucket_name)
            
            # Check if any blobs exist with this prefix
            blobs = list(bucket.list_blobs(prefix=blob_prefix, max_results=1))
            return len(blobs) > 0
        else:
            # Check if local path exists
            return os.path.exists(path)

    def __len__(self) -> int:
        """Get the length of all tokenized texts."""
        if self.tokenized_texts is not None:
            return len(self.tokenized_texts)
        raise NotImplementedError("Dataset length unknown. Load a saved dataset or use custom logic.")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Function for allowing indexing in DataLoader."""
        if self.tokenized_texts is None:
            raise ValueError("tokenized_texts not loaded. Load from disk or fill manually.")

        tokens = self.tokenized_texts[idx]

        # Handle minimum length
        if len(tokens) < 2:
            tokens = tokens + [self.tokenizer.pad_token_id] * (2 - len(tokens))
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]

        # Create input_ids and labels for causal LM
        input_ids = tokens[:-1]
        labels = tokens[1:]

        # Pad to target length
        target_length = self.max_length - 1
        if len(input_ids) < target_length:
            pad_length = target_length - len(input_ids)
            input_ids.extend([self.tokenizer.pad_token_id] * pad_length)
            labels.extend([-100] * pad_length)

        # Convert to LongTensors (int64)
        input_ids = torch.tensor(input_ids, dtype=torch.int64)
        labels = torch.tensor(labels, dtype=torch.int64)

        # Create attention mask
        real_length = min(len(tokens) - 1, target_length)
        attention_mask = [1.0] * real_length + [0.0] * (target_length - real_length)

        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }

def download_falcon_refinedweb(
    max_samples: Optional[int] = None,
    batch_size: int = 1000,
    compress: bool = True,
) -> str:
    """Download and process Falcon RefinedWeb dataset, streaming directly to GCS bucket."""
    try:
        # Initialize GCS client
        client = storage.Client()
        bucket = client.bucket(GCP_BUCKET_NAME)
        
        # Load Falcon RefinedWeb dataset
        dataset = load_dataset(
            "tiiuae/falcon-refinedweb",
            split="train",
            streaming=True,
            trust_remote_code=True
        )

        logger.info(f"Starting Falcon RefinedWeb download and streaming to {GCP_FALCON_PATH}")

        # Initialization
        batch_texts = []
        file_counter = 0
        total_samples = 0

        # Create file extension based on compression
        file_ext = ".jsonl.gz" if compress else ".jsonl"
        
        with tqdm(dataset, desc="Processing samples") as pbar:
            for i, example in enumerate(dataset):
                if max_samples is not None and i >= max_samples:
                    break
                
                text = example['content'].strip()
                
                # Add to current batch
                batch_texts.append({
                    "id": i,
                    "content": text,
                    "url": example.get('url', ''),
                    "timestamp": example.get('timestamp', '')
                })
                
                # Write batch to GCS when batch_size is reached
                if len(batch_texts) >= batch_size:
                    _write_batch_to_gcs(
                        bucket, 
                        batch_texts, 
                        "falcon-refinedweb", 
                        file_counter, 
                        file_ext, 
                        compress
                    )
                    
                    total_samples += len(batch_texts)
                    batch_texts = []
                    file_counter += 1
                
                pbar.update(1)
        
        # Write remaining samples
        if batch_texts:
            _write_batch_to_gcs(
                bucket, 
                batch_texts, 
                "falcon-refinedweb", 
                file_counter, 
                file_ext, 
                compress
            )
            total_samples += len(batch_texts)
        
        logger.info(f"Successfully streamed {total_samples} samples to {GCP_FALCON_PATH}")
        
        # Write metadata file
        _write_metadata_to_gcs(
            bucket, 
            "falcon-refinedweb", 
            total_samples, 
            file_counter + 1, 
            batch_size
        )
        
        return GCP_FALCON_PATH

    except Exception as e:
        logger.error(f"Failed to process Falcon RefinedWeb: {e}")
        raise

def _write_batch_to_gcs(
    bucket: storage.Bucket,
    batch_texts: list,
    gcs_path_prefix: str,
    file_counter: int,
    file_ext: str,
    compress: bool
) -> None:
    """Write a batch of texts to GCS as JSONL file."""
    
    # Create blob path
    blob_name = f"{gcs_path_prefix}/batch_{file_counter:06d}{file_ext}"
    blob = bucket.blob(blob_name)
    
    # Prepare data as JSONL
    jsonl_data = '\n'.join(json.dumps(item) for item in batch_texts)
    
    # Compress if requested
    if compress:
        data_bytes = gzip.compress(jsonl_data.encode('utf-8'))
        content_type = "application/gzip"
    else:
        data_bytes = jsonl_data.encode('utf-8')
        content_type = "application/json"
    
    # Upload to GCS
    blob.upload_from_string(
        data_bytes,
        content_type=content_type
    )
    
    logger.debug(f"Uploaded batch {file_counter} with {len(batch_texts)} samples to {blob_name}")

def _write_metadata_to_gcs(
    bucket: storage.Bucket,
    gcs_path_prefix: str,
    total_samples: int,
    num_files: int,
    batch_size: int
) -> None:
    """Write metadata about the dataset to GCS."""
    
    metadata = {
        "dataset": "falcon-refinedweb",
        "total_samples": total_samples,
        "num_files": num_files,
        "batch_size": batch_size,
        "format": "jsonl",
        "compressed": True
    }
    
    blob_name = f"{gcs_path_prefix}/metadata.json"
    blob = bucket.blob(blob_name)
    blob.upload_from_string(
        json.dumps(metadata, indent=2),
        content_type="application/json"
    )
    
    logger.info(f"Uploaded metadata to {blob_name}")

def read_falcon_refinedweb_from_gcs() -> Generator[dict, None, None]:
    """Read Falcon RefinedWeb data from GCS bucket."""
    client = storage.Client()
    bucket = client.bucket(GCP_BUCKET_NAME)
    
    # List all batch files
    blobs = bucket.list_blobs(prefix="falcon-refinedweb/batch_")
    
    for blob in blobs:
        if blob.name.endswith('.jsonl.gz'):
            # Download and decompress
            compressed_data = blob.download_as_bytes()
            jsonl_data = gzip.decompress(compressed_data).decode('utf-8')
            
            # Parse JSONL
            for line in jsonl_data.strip().split('\n'):
                if line:
                    yield json.loads(line)
        elif blob.name.endswith('.jsonl'):
            # Download uncompressed
            jsonl_data = blob.download_as_text()
            
            # Parse JSONL
            for line in jsonl_data.strip().split('\n'):
                if line:
                    yield json.loads(line)

def load_tokenized_dataset_from_gcs(
    gcs_path: str,
    max_length: int,
    tokenizer
) -> 'TokenizedDataset':
    """Load tokenized dataset from GCS."""
    # Download dataset to temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        dataset_instance = TextDataset(
            texts=None,
            tokenizer=tokenizer,
            max_length=max_length
        )
        
        dataset_instance._download_directory_from_gcs(gcs_path, temp_dir)
        
        # Load the dataset
        dataset = load_from_disk(temp_dir)
        
        # Create a simple dataset wrapper
        return TokenizedDataset(dataset, max_length)

class TokenizedDataset(Dataset):
    """Simple wrapper for pre-tokenized dataset."""
    
    def __init__(self, hf_dataset, max_length):
        self.dataset = hf_dataset
        self.max_length = max_length
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            'input_ids': torch.tensor(item['input_ids'], dtype=torch.int64),
            'labels': torch.tensor(item['labels'], dtype=torch.int64),
            'attention_mask': torch.tensor(item['attention_mask'], dtype=torch.float32)
        }

def compute_loss(
    logits: torch.Tensor,
    labels: torch.Tensor, 
    aux_loss: Optional[torch.Tensor] = None, 
    aux_loss_weight: float = 0.01, 
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute training loss with optional auxiliary loss."""
    # Shift logits/labels for CE
    shift_logits = logits.view(-1, logits.size(-1))
    shift_labels = labels.view(-1)

    # Tell model to ignore value of -100
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    lm_loss = loss_fct(shift_logits, shift_labels)

    if aux_loss is not None:
        total_loss = lm_loss + aux_loss_weight * aux_loss
        return total_loss, lm_loss, aux_loss
    else:
        return lm_loss, lm_loss, torch.tensor(0.0, device=lm_loss.device)

def train_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    optimizer,
    training_args: TrainingArgs,
    device: torch.device,
    step: int,
    scaler: Optional[GradScaler] = None,
) -> Dict[str, Union[float, bool]]:
    """Single training step with AMP support."""
    try:
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)

        # Forward pass with AMP
        if scaler is not None:
            with device_specific_amp(device=device, dtype=dtype):
                logits, _, aux_loss = model(input_ids, padding_mask=attention_mask, use_cache=False)
                loss, lm_loss, aux_loss_val = compute_loss(logits, labels, aux_loss, aux_loss_weight=0.01)
                loss = loss / training_args.grad_accum_steps
        else:
            # Standard FP32 forward pass
            logits, _, aux_loss = model(input_ids, padding_mask=attention_mask, use_cache=False)
            loss, lm_loss, aux_loss_val = compute_loss(logits, labels, aux_loss, aux_loss_weight=0.01)
            loss = loss / training_args.grad_accum_steps

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

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.warning(f"OOM at step {step}.")
            if device.type == "cuda":
                torch.cuda.empty_cache()
            optimizer.zero_grad()
            return {'success': False, 'error': 'oom'}
        else:
            logger.error(f"Training step failed: {e}")
            return {'success': False, 'error': str(e)}
    except Exception as e:
        logger.error(f"Exception occurred during training step {step}: {e}")
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
    """Train for one epoch with AMP support."""
    model.train()

    # Initialization
    total_loss = 0
    total_lm_loss = 0
    total_aux_loss = 0
    successful_steps = 0
    skipped_steps = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
    optimizer.zero_grad()

    for step, batch in enumerate(progress_bar):
        result = train_step(model, batch, optimizer, training_args, device, step, scaler)

        if result['success']:
            total_loss += result['loss']
            total_lm_loss += result['lm_loss']
            total_aux_loss += result['aux_loss']
            successful_steps += 1
        else:
            skipped_steps += 1
            if skipped_steps > len(dataloader) * 0.2:
                logger.error("Too many failed steps, stopping epoch")
                break

        # Gradient accumulation and optimizer step
        if (step + 1) % training_args.grad_accum_steps == 0:
            if successful_steps > 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), training_args.clip_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
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
                'loss': f'{avg_loss:.4f}',
                'lm_loss': f'{avg_lm_loss:.4f}',
                'aux_loss': f'{avg_aux_loss:.4f}',
                'lr': f'{lr:.2e}',
                'skipped': skipped_steps
            })

    # Handle remaining gradients
    if len(dataloader) % training_args.grad_accum_steps != 0:
        if successful_steps > 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), training_args.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), training_args.clip_grad_norm)
                optimizer.step()
        optimizer.zero_grad()

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
    """Evaluate model with AMP support."""
    model.eval()

    total_loss = 0
    total_lm_loss = 0
    total_aux_loss = 0
    successful_batches = 0

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            if max_batches is not None and i >= max_batches:
                break

            try:
                input_ids = batch['input_ids'].to(device, non_blocking=training_args.pin_memory)
                labels = batch['labels'].to(device, non_blocking=training_args.pin_memory)
                attention_mask = batch['attention_mask'].to(device, non_blocking=training_args.pin_memory)

                with device_specific_amp(device=device, dtype=dtype):
                    logits, _, aux_loss = model(input_ids, padding_mask=attention_mask, use_cache=False)
                    loss, lm_loss, aux_loss_val = compute_loss(logits, labels, aux_loss)

                total_loss += loss.item()
                total_lm_loss += lm_loss.item()
                total_aux_loss += aux_loss_val.item()
                successful_batches += 1

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning("OOM during evaluation, skipping batch")
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    continue
                else:
                    logger.warning(f"Evaluation error: {e}")
                    continue
            except Exception as e:
                logger.error(f"Error occurred during evaluation: {e}")

    if successful_batches == 0:
        logger.error("No successful evaluation batches")
        return float('inf'), float('inf'), float('inf')

    return (total_loss / successful_batches,
            total_lm_loss / successful_batches,
            total_aux_loss / successful_batches)

def save_checkpoint_to_gcs(
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
    """Save checkpoint to Google Cloud Storage."""
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
        if is_best:
            filename = "checkpoint_best.pt"
        else:
            filename = f"checkpoint_epoch_{epoch}_step_{step}.pt"
        
        # Save to temporary file first
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as temp_file:
            torch.save(checkpoint_data, temp_file.name)
            temp_path = temp_file.name
        
        # Upload to GCS
        client = storage.Client()
        bucket = client.bucket(GCP_BUCKET_NAME)
        blob_name = f"checkpoints/{filename}"
        blob = bucket.blob(blob_name)
        
        blob.upload_from_filename(temp_path)
        
        # Clean up temporary file
        os.unlink(temp_path)
        
        gcs_path = f"gs://{GCP_BUCKET_NAME}/{blob_name}"
        logger.info(f"Saved checkpoint to {gcs_path}")
        
        return gcs_path
        
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        raise

def load_checkpoint_from_gcs(
    gcs_path: str,
    model: nn.Module,
    optimizer,
    scheduler,
    scaler: Optional[GradScaler] = None,
    device: torch.device = None,
) -> Dict[str, Union[int, float]]:
    """Load checkpoint from Google Cloud Storage."""
    try:
        # Parse GCS path
        if not gcs_path.startswith('gs://'):
            raise ValueError(f"Invalid GCS path: {gcs_path}")
        
        path_parts = gcs_path[5:].split('/', 1)
        bucket_name = path_parts[0]
        blob_name = path_parts[1]
        
        # Download checkpoint
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as temp_file:
            blob.download_to_filename(temp_file.name)
            temp_path = temp_file.name
        
        # Load checkpoint
        checkpoint = torch.load(temp_path, map_location=device)
        
        # Load states
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if scaler is not None and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Clean up temporary file
        os.unlink(temp_path)
        
        logger.info(f"Loaded checkpoint from {gcs_path}")
        
        return {
            'epoch': checkpoint['epoch'],
            'step': checkpoint['step'],
            'loss': checkpoint['loss'],
        }
        
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise

def get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create a linear learning rate schedule with warmup."""
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
    """Setup optimizer, scheduler, and scaler for training."""
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
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    
    # Setup gradient scaler for AMP
    scaler = GradScaler() if device.type == "cuda" else None
    
    return optimizer, scheduler, scaler

def main(
    resume_from_checkpoint: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> None:
    """Main training loop with GCS integration."""
    logger.info("Starting training loop...")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2") # Rust tokenizer
    logger.info(f"Initialized {tokenizer.name_or_path} tokenizer.")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Inintialize model and training arguments
    model_args = ModelArgs()
    model_args.vocab_size = tokenizer.vocab_size
    model_args.pad_token_id = tokenizer.pad_token_id
    model_args.eos_token_id = tokenizer.eos_token_id

    training_args = TrainingArgs()

    # Initialize model
    model = Transformer(model_args).to(device)
    
    # Setup filters
    quality_filter = TextQualityFilter()
    dedup_filter = DeduplicationFilter()
    
    # Check if we need to download and process data
    if not os.path.exists(GCP_TOKENIZED_PATH.replace('gs://', '').replace(GCP_BUCKET_NAME + '/', '')):
        logger.info("Downloading and processing Falcon RefinedWeb dataset...")
        
        # Download raw data
        download_falcon_refinedweb(max_samples=max_samples)
        
        # Read and process data
        logger.info("Reading data from GCS...")
        texts = []
        for sample in read_falcon_refinedweb_from_gcs():
            texts.append(sample['content'])
            if max_samples is not None and len(texts) >= max_samples:
                break
        
        logger.info(f"Loaded {len(texts)} texts from GCS")
        
        # Create dataset with filtering and tokenization
        dataset = TextDataset(
            texts=texts,
            tokenizer=tokenizer,
            max_length=model_args.max_seq_len,
            quality_filter=quality_filter,
            dedup_filter=dedup_filter,
            skip_filtering=False,
            save_filtered_texts_path=GCP_FILTERED_PATH,
            save_tokenized_dataset_path=GCP_TOKENIZED_PATH,
        )
        
        # Clean up texts from memory
        del texts
        gc.collect()
    
    # Load tokenized dataset
    logger.info("Loading tokenized dataset...")
    dataset = load_tokenized_dataset_from_gcs(
        GCP_TOKENIZED_PATH,
        model_args.max_position_embeddings,
        tokenizer
    )
    
    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=training_args.batch_size,
        shuffle=True,
        num_workers=training_args.num_workers,
        pin_memory=training_args.pin_memory,
        drop_last=True,
    )
    
    # Calculate training steps
    num_training_steps = len(dataloader) * training_args.epochs // training_args.grad_accum_steps
    
    # Setup training components
    optimizer, scheduler, scaler = setup_training_components(
        model, training_args, num_training_steps
    )
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_loss = float('inf') # Initialize loss
    
    if resume_from_checkpoint is not None:
        logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
        checkpoint_info = load_checkpoint_from_gcs(
            resume_from_checkpoint, model, optimizer, scheduler, scaler, device
        )
        start_epoch = checkpoint_info['epoch'] + 1
        best_loss = checkpoint_info['loss']
    
    # Training loop
    logger.info(f"Starting training for {training_args.epochs} epochs...")
    
    for epoch in range(start_epoch, training_args.epochs):
        logger.info(f"Starting epoch {epoch + 1}/{training_args.epochs}")
        
        # Train for one epoch
        train_loss, train_lm_loss, train_aux_loss = train_epoch(
            model, dataloader, optimizer, scheduler, training_args, device, epoch, scaler
        )
        
        logger.info(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, "
                   f"LM Loss: {train_lm_loss:.4f}, Aux Loss: {train_aux_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % training_args.save_steps == 0:
            checkpoint_path = save_checkpoint_to_gcs(
                model, optimizer, scheduler, epoch, 0, train_loss,
                training_args, model_args, scaler, is_best=False
            )
            logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model
        if train_loss < best_loss:
            best_loss = train_loss
            best_checkpoint_path = save_checkpoint_to_gcs(
                model, optimizer, scheduler, epoch, 0, train_loss,
                training_args, model_args, scaler, is_best=True
            )
            logger.info(f"New best model saved to {best_checkpoint_path}")
        
        # Clean up GPU memory
        if device.type == "cuda":
            torch.cuda.empty_cache()
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main(
        resume_from_checkpoint=None,
        max_samples=100000, # Limit samples for testing
    )
    