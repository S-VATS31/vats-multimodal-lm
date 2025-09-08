from configs.setup_env import device, dtype

from typing import Optional

import torch
import torch.nn.functional as F
from torch.amp import autocast

from src.optimized_attention import KVCache
from src.transformers.nlp.model import AutoregressiveTextTransformer
from configs.transformers.nlp.model_args.model_args_medium import ModelArgs
from configs.transformers.nlp.generation_args import GenerationArgs

class AutoregressiveTokenGenerator:
    """Module use for autoregressive token generation.
    
    Args:
        model_args (ModelArgs): Dataclass containing model hyperparameters.
    """
    def __init__(self, model_args: ModelArgs):
        self.model_args = model_args

        self.model = AutoregressiveTextTransformer(model_args).to(device)
        self.model.eval()

        self.kv_cache = KVCache(
            max_batch_size=model_args.max_batch_size,
            max_seq_len=model_args.max_seq_len,
            num_heads=model_args.num_heads,
            head_dim=model_args.d_model // model_args.num_heads,
            num_layers=model_args.num_layers
        )

    def _generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        repetition_penalty: Optional[float] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Generate tokens autoregressively using decoding methods.

        Args:
            input_ids (torch.LongTensor): LongTensor containing tokenized ids.
            max_new_tokens (int): Maximum number of tokens the model can generate at a time.
            repetition_penalty (float): Penalty that scales logits to discourage repetitive tokens.
            temperature (float): Decoding method to encourage more randomness/determinism based on value.
            top_k (int): Top-k logits to be sampled.
            top_p (float): Top-p hyperparameter used as a threshold for masking out certain logits.
            do_sample (bool): Whether to apply sampling or greedy decoding.
            pad_token_id (Optional[int]): Special value of the padding token to be masked out.
            eos_token_id (Optional[int]): End of sequence token appended to the end of each token.
            attention_mask (Optional[torch.Tensor]): Padding mask of shape [B, T].
            use_cache (bool): Boolean to whether use the KV cache or not.

        Returns:
            torch.Tensor: Returns a tensor of generated tokens of shape [B, T].
        """
        B, T = input_ids.shape

        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = (input_ids != pad_token_id)

        # Check if initial sequence length plus max_new_tokens exceeds model limit
        max_total_length = min(self.model_args.max_seq_len, T + max_new_tokens)
        actual_max_new_tokens = max_total_length - T
        
        if actual_max_new_tokens <= 0:
            return input_ids

        generated_ids = input_ids.clone()
        unfinished_sequences = torch.ones(B, dtype=torch.bool).to(device) # All sequences start unfinished

        with torch.no_grad():
            # Reset and initialize cache
            if use_cache:
                self.kv_cache.reset()
                self.kv_cache.initialize(B)

            # Process initial sequence
            logits, _, _ = self.model(
                input_ids=generated_ids, padding_mask=attention_mask, use_cache=use_cache
            )

            # Generation loop
            for step in range(actual_max_new_tokens):
                current_seq_len = generated_ids.shape[1]

                # Check sequence length limit
                if current_seq_len >= self.model_args.max_seq_len:
                    break

                # Skip if all sequences are finished
                if not unfinished_sequences.any():
                    break

                # Get logits for next token prediction
                if use_cache and step > 0 and current_seq_len < self.model_args.max_seq_len - 1:
                    # Check if we can safely increment the cache
                    if current_seq_len >= self.model_args.max_seq_len:
                        break
                        
                    # For cached generation, only process the last token
                    last_token = generated_ids[:, -1:].contiguous()
                    last_attention = torch.ones(B, 1, dtype=torch.bool).to(device)
                    # Only process unfinished sequences
                    last_attention = last_attention & unfinished_sequences[:, None]

                    logits, _, _ = self.model(
                        input_ids=last_token, padding_mask=last_attention, use_cache=True
                    )
                else:
                    # For non-cached or first step, process full sequence
                    if attention_mask.shape[1] < current_seq_len:
                        # Extend attention mask for new tokens
                        new_attention = torch.cat([attention_mask,
                            unfinished_sequences[:, None].expand(-1, current_seq_len - attention_mask.shape[1])
                        ], dim=1)
                    else:
                        new_attention = attention_mask[:, :current_seq_len]

                    logits, _, _ = self.model(
                        input_ids=generated_ids, padding_mask=new_attention, use_cache=False
                    )

                # Get logits for the last position
                next_token_logits = logits[:, -1, :] # [B, V]

                # Apply repetition penatly
                if repetition_penalty is not None and repetition_penalty != 1.0:
                    logits_B, vocab_size = next_token_logits.size()
                    for batch_idx in range(logits_B):
                        # Get unique tokens
                        unique_tokens = torch.unique(generated_ids[batch_idx])
                        # Apply repetition penalty
                        for token_id in unique_tokens:
                            if 0 <= token_id <= vocab_size:
                                # Discourage repeating of token
                                if next_token_logits[batch_idx, token_id] > 0:
                                    next_token_logits[batch_idx, token_id] /= repetition_penalty
                                # Model confident in logit, increase probability
                                else:
                                    next_token_logits[batch_idx, token_id] *= repetition_penalty

                # Apply temperature
                if temperature is not None:
                    if temperature > 0 :
                        next_token_logits = next_token_logits / temperature
                    # Greedy decoding
                    elif temperature == 0:
                        do_sample = False
                    # Invalid values
                    else:
                        raise ValueError(f"Expected temperature >= 0, got temperature of {temperature}")

                # Apply top-k filtering
                if top_k is not None:
                    if top_k > 0 and top_k < self.model_args.vocab_size:
                        # Get the top-k values and set others to -inf
                        topk_values, _ = torch.topk(next_token_logits, top_k, dim=-1)
                        min_topk = topk_values[:, -1:].expand_as(next_token_logits)
                        next_token_logits = torch.where(
                            next_token_logits < min_topk,
                            torch.full_like(next_token_logits, float('-inf')), # mask to -inf if condition True
                            next_token_logits # Return normal logits if condition False
                        )
                    # Greedy decoding
                    elif top_k == 1:
                        do_sample = False
                    # Invalid values
                    else:
                        raise ValueError(f"Expected top_k >= 1, got top_k of {top_k}")

                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    if 0 < top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                        # Remove tokens with cumulative probability above the threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        # Keep at least one token
                        sorted_indices_to_remove[:, 0] = False
                        # Shift right to keep the first token above threshold
                        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()

                        # Convert back to original indices
                        indices_to_remove = torch.zeros_like(next_token_logits, dtype=torch.bool)
                        indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
                        next_token_logits[indices_to_remove] = float('-inf')
                    # Invalid values
                    else:
                        raise ValueError(f"Expected 0 < top_p < 1.0, got top_p  of {top_p}")

                # Apply sampling
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                # Greedy decoding
                else:
                    next_tokens = torch.argmax(next_token_logits, dim=-1)

                # Only update unfinished sequences
                next_tokens = torch.where(unfinished_sequences, next_tokens, pad_token_id)

                # Append new tokens
                generated_ids = torch.cat([generated_ids, next_tokens[:, None]], dim=1)

                # Update attention mask
                attention_mask = torch.cat([
                    attention_mask,
                    unfinished_sequences[:, None]
                ], dim=1)

                # Check for EOS tokens
                if eos_token_id is not None:
                    unfinished_sequences = unfinished_sequences & (next_tokens != eos_token_id)

        # Reset KV cache
        if use_cache:
            self.kv_cache.reset()

        return generated_ids

    def generate_tokens(
        self,
        prompt: str,
        generation_args: GenerationArgs,
        tokenizer,
    ) -> str:
        """Generate tokens using a HuggingFace tokenizer.
        
        Args:
            prompt (str): Input string of text to be tokenized.
            generation_args (GenerationArgs): Generation arguments.
            tokenizer: Hugging Face tokenizer.

        Returns:
            str: Generated text based on prompt.
        """
        # Handle empty input prompts
        if not prompt or not prompt.strip():
            return "Please enter a valid prompt."
        
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        # Turn off gradient computation for token generation
        with torch.no_grad():
            with autocast(device_type=device.type, dtype=dtype):
                generated_ids = self._generate(
                    input_ids=input_ids,
                    max_new_tokens=generation_args.max_new_tokens,
                    repetition_penalty=generation_args.repetition_penalty,
                    temperature=generation_args.temperature,
                    top_k=generation_args.top_k,
                    top_p=generation_args.top_p,
                    do_sample=generation_args.do_sample,
                    pad_token_id=generation_args.pad_token_id,
                    eos_token_id=generation_args.eos_token_id,
                    attention_mask=None,
                    use_cache=generation_args.use_cache
                )

        # [0] is to get the first generated sequence
        if generation_args.return_only_new_tokens:
            generated_text = tokenizer.decode(
                generated_ids[0][input_ids.shape[1]:], skip_special_tokens=True
            )
        else:
            generated_text = tokenizer.decode(
                generated_ids[0], skip_special_tokens=True
            )
        return generated_text

def main():
    from transformers import AutoTokenizer
    
    # Initialize model and generation arguments
    model_args = ModelArgs()
    generation_args = GenerationArgs()

    # Load Mistral tokenizer
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

    # Set special tokens
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    generation_args.pad_token_id = tokenizer.pad_token_id
    generation_args.eos_token_id = tokenizer.eos_token_id

    # Create generator
    generator = AutoregressiveTokenGenerator(model_args)

    # Input prompt
    prompt = input(f"Enter prompt: ")

    # Generate and print the output
    output = generator.generate_tokens(prompt, generation_args, tokenizer)
    print("--- Generated Text ---")
    print(output)

if __name__ == "__main__":
    main()
