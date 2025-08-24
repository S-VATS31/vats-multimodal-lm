from dataclasses import dataclass
from typing import Optional

@dataclass
class GenerationArgs:
    """Dataclass containing generation args.
    
    Args:
        max_new_tokens (int): Maximum number of tokens that can be generated per prompt.
        temperature (float): Temperature hyperparameter to encourage randomness or determinism.
        top_k (float): Top-k hyperparameter for token sampling.
        top_p (float): Top-p hyperparameter for token sampling.
        do_sample (bool): Whether to apply sampling or not.
        pad_token_id (Optional[int]): Pad token id to pad variable sequence lengths.
        eos_token_id (Optional[int]): Eos token id to mark when the sequence is over.
        use_cache (bool): Whether to use the KV cache or not.
        repetition_penalty (float): Penalty to discourage repetitive generated sequences.
        return_only_new_tokens (bool): Whether to return only generated tokens or prompt + generated tokens.
        generation_frequency (int): Test generation every N tokens seen to see if model can learn well.
    """
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.95
    do_sample: bool = True
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    use_cache: bool = True
    repetition_penalty: float = 1.7
    return_only_new_tokens: bool = True
    generation_frequency: int = 10_000_000_000
