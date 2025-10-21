import logging
import time

import torch
from torch.utils.data import Dataset
from datasets import load_dataset

from utils.setup_logger import setup_logger
from configs.transformers.nlp.model_args.model_args_xsmall import ModelArgs

# Set up logger
data_logger = setup_logger(
    name="data_logger", 
    log_file="data_loading.log", 
    level=logging.INFO
)

class TextDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        model_args: ModelArgs,
        dataset_name: str,
        split: str,
    ):
        self.tokenizer = tokenizer
        self.model_args = model_args

        # Load dataset with retry logic
        max_retries = 5
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                self.ds = load_dataset(
                    dataset_name, 
                    split="train", 
                    streaming=True
                )
                print(self.ds.column_names)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    data_logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    data_logger.error(f"Failed to load dataset after {max_retries} attempts")
                    raise

        self.dataset = []
        for i, example in enumerate(self.ds):
            if i >= 3_500_000:
                break
            if i % 500_000 == 0:
                print(len(self.dataset))
            
            # Retry logic for individual examples
            max_example_retries = 3
            for attempt in range(max_example_retries):
                try:
                    self.dataset.append(example)
                    break
                except Exception as e:
                    if attempt < max_example_retries - 1:
                        data_logger.warning(f"Failed to process example {i}, attempt {attempt + 1}/{max_example_retries}: {e}")
                        time.sleep(2)
                    else:
                        data_logger.error(f"Skipping example {i} after {max_example_retries} failed attempts")

        print(f"number of examples: {len(self.dataset)}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item['content']

        # tokenize with padding to max_length
        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=self.model_args.max_seq_len,
            padding='max_length',
            return_tensors='pt'
        )

        # squeeze to remove batch dimension
        input_ids = tokenized['input_ids'].squeeze(0)
        attention_mask = tokenized['attention_mask'].squeeze(0)

        # create labels, mask padding tokens
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100

        # shift input_ids and labels for causal LM
        shifted_input_ids = input_ids[:-1]
        shifted_labels = labels[1:]

        shifted_attention_mask = attention_mask[:-1]

        return {
            'input_ids': shifted_input_ids,
            'attention_mask': shifted_attention_mask,
            'labels': shifted_labels
        }
    