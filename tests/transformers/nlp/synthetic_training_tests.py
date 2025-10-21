from configs.setup_env import device

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from configs.transformers.nlp.generation_args import GenerationArgs
from configs.transformers.nlp.training_args import TrainingArgs
from configs.transformers.nlp.model_args.model_args_xsmall import ModelArgs
from src.transformers.nlp.model import AutoregressiveTextTransformer
from training.transformers.nlp.loops.training_loop import train
from training.transformers.nlp.loops.validation_loop import validate
from training.transformers.nlp.setup_training_components import setup_training_components

class SyntheticDataset(Dataset):
    def __init__(self, num_samples, seq_len, vocab_size, mask_prob=0.0):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.mask_prob = mask_prob

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,), dtype=torch.long)
        labels = input_ids.clone()
        
        if self.mask_prob > 0.0:
            mask = torch.rand(self.seq_len) < self.mask_prob
            labels[mask] = -100 
            
        attention_mask = torch.ones(self.seq_len, dtype=torch.long)
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }

def main():
    model_args = ModelArgs()
    training_args = TrainingArgs()
    generation_args = GenerationArgs()
    B, T = 16, 12
    num_training_samples = 512
    num_validation_samples = 128
    num_training_steps = num_training_samples//B

    train_dataset = SyntheticDataset(num_training_samples, T, vocab_size=128, mask_prob=0.0)
    val_dataset = SyntheticDataset(num_validation_samples, T, vocab_size=128, mask_prob=0.0)
    train_loader = DataLoader(train_dataset, batch_size=B, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=B, shuffle=False)

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    # Initialize pad/eos token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    generation_args.pad_token_id = tokenizer.pad_token_id

    model = AutoregressiveTextTransformer(model_args).to(device)
    optimizer, scheduler, scaler = setup_training_components(
        model, training_args, num_training_steps
    )

    total_tokens_seen = 0
    stop_early = False
    last_logged_tokens = 0

    # training loop
    while total_tokens_seen < training_args.max_train_tokens and not stop_early:
        train_loss, train_lm_loss, train_aux_loss, train_perplexity, total_tokens, stop_early = train(
            model, train_loader, optimizer, scheduler, training_args, generation_args, scaler
        )
        val_loss, val_lm_loss, val_aux_loss, val_perplexity = validate(
            model, val_loader, training_args, max_batches=None
        )
        total_tokens_seen += total_tokens
        if total_tokens_seen - last_logged_tokens >= training_args.logging_tokens_freq:
            last_logged_tokens = total_tokens_seen
            print(f"Total tokens seen: {total_tokens_seen}")
            print(f"    Total Train Loss: {train_loss:.4f} | Total Validation Loss: {val_loss:.4f}")
            print(f"    Train LM Loss: {train_lm_loss:.4f} | Validation LM Loss: {val_lm_loss:.4f}")
            print(f"    Train Auxiliary Loss: {train_aux_loss:.4f} | Validation Auxiliary Loss: {val_aux_loss:.4f}")
            print(f"    Train Perplexity: {train_perplexity:.4f} | Validation Perplexity: {val_perplexity:.4f}")

if __name__ == "__main__": 
    main()
