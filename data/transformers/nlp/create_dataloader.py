from typing import Optional

from torch.utils.data import DataLoader

from configs.transformers.nlp.training_args import TrainingArgs
from data.transformers.nlp.data_streaming import StreamingTextDataset
from src.transformers.nlp.text_cleaning.text_quality_filter import TextQualityFilter
from src.transformers.nlp.text_cleaning.deduplication_filter import DeduplicationFilter

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
        **dataset_kwargs: Additional arguments for StreamingTextDataset.

    Returns:
        DataLoader: DataLoader with lazy-loading dataset.
    """
    # Initialize TextDataset
    dataset = StreamingTextDataset(
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