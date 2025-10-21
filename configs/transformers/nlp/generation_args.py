from dataclasses import dataclass
from typing import Optional

@dataclass
class GenerationArgs:
    """Dataclass containing generation hyperparameters."""
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
    generation_frequency: int = 10_000
