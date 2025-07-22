from dataclasses import dataclass
from typing import Optional

@dataclass
class GenerationArgs:
    """Dataclass containing generation args."""
    max_new_tokens: int = 100
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.95
    do_sample: bool = True
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    use_cache: bool = True
    max_seq_len: int = 32768
    repetition_penalty: float = 1.5
    return_only_new_tokens: bool = True
    generation_frequency: int = 10_000_000_000
