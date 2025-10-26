from typing import Optional
from dataclasses import dataclass
import math

from configs.transformers.nlp.model_args.post_init import PostInitMixin

@dataclass
class ModelArgs(PostInitMixin):
    """Small configuration of model arguments."""
    d_model: int = 768
    num_heads: int = 32
    query_groups: int = 8
    softmax_scale: Optional[float] = None
    d_ffn: int = 768*4
    num_layers: int = 10
    dropout: float = 0.1
    rope_base: float = 10000.0
    rms_norm_eps: float = 1e-7
    left_window: int = 256
    right_window: int = 0
    vocab_size: int = 32768
    max_seq_len: int = 512
    tie_weights: bool = True
    max_batch_size: int = 1024
    gradient_checkpointing: bool = False
    use_proj_bias: bool = False
    use_qkv_proj: bool = True
    use_causal: bool = True
    use_mqa: bool = False
    num_experts: int = 1
    top_k: int = 1

    def __post_init__(self):
        """Post initialization to set softmax scale dynamically."""
        if self.softmax_scale is None:
            self.softmax_scale = 1 / math.sqrt(self.d_model // self.num_heads)

        # Call assertions
        super().__post_init__()
