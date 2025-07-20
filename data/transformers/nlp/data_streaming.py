import os
import logging
from typing import Tuple, Optional, List, Dict, Iterator, Generator

import torch
from torch.utils.data import IterableDataset

from datasets import load_dataset

from configs.transformers.nlp.model_args.model_args_medium import ModelArgs
from src.transformers.nlp.text_cleaning.deduplication_filter import DeduplicationFilter
from src.transformers.nlp.text_cleaning.text_quality_filter import TextQualityFilter
from utils.setup_logger import setup_logger

# Set up logger
data_logger = setup_logger(name="data_logger", log_file="data_loading.log", level=logging.INFO)

class StreamingTextDataset(IterableDataset):
    """Iterable text dataset for loading large datasets.

    Args:
        dataset_name (str): HuggingFace dataset name.
        tokenizer: HuggingFace tokenizer.
        max_length (int): Maximum sequence length.
        quality_filter (Optional[TextQualityFilter]): Text quality filter.
        dedup_filter (Optional[DeduplicationFilter]): Deduplication filter.
        buffer_size (int): Size of internal buffer for batch processing filters.
        max_samples (Optional[int]): Maximum number of samples to process (None means process all).
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
        
        data_logger.info(f"Initialized TextDataset with streaming={streaming}")
        data_logger.info(f"Dataset: {dataset_name}, Max length: {max_length}")
    
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
            data_logger.error(f"Failed to load dataset {self.dataset_name}: {e}")
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
            data_logger.warning(f"Failed to tokenize text: {e}")
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
        
        data_logger.info(f"Worker {worker_id}/{num_workers} starting to stream examples")
        
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
                        data_logger.info(f"Worker {worker_id}: processed {processed_count} examples")
            
            # Process remaining texts in buffer
            if text_buffer:
                processed_texts = self.process_batch(text_buffer)
                for processed_text in processed_texts:
                    tokenized = self.tokenize(processed_text)
                    if tokenized is not None:
                        processed_count += 1
                        yield tokenized
        
        except Exception as e:
            data_logger.error(f"Error in worker {worker_id}: {e}")
            raise
        
        data_logger.info(f"Worker {worker_id} finished: {processed_count} examples yielded from {sample_count} samples")
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterator function for IterableDataset.
        
        Returns:
            Iterator[Dict[str, torch.Tensor]]: Returning tokenized examples in the form of input_ids.
        """
        return self.stream_examples()