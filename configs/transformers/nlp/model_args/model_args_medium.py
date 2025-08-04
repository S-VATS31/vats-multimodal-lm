import math
from typing import Optional
from dataclasses import dataclass

from configs.transformers.nlp.model_args.post_init import PostInitMixin

@dataclass
class ModelArgs(PostInitMixin):
    """Medium configuration of model arguments."""
    d_model: int = 1440
    num_heads: int = 24
    query_groups: int = 1
    softmax_scale: Optional[float] = None
    d_ffn: int = 5760
    num_layers: int = 20
    dropout: float = 0.2
    rope_base: float = 10000.0
    rms_norm_eps: float = 1e-7
    left_window: int = 384
    right_window: int = 0
    vocab_size: int = 65536
    max_seq_len: int = 2048
    tie_weights: bool = True
    gradient_checkpointing: bool = True
    use_proj_bias: bool = False
    use_qkv_proj: bool = True
    use_causal: bool = True
    use_mqa: bool = True
    use_cache: bool = False
    max_batch_size: int = 1024
    num_experts: int = 1
    top_k: int = 1

    def __post_init__(self):
        """Post initialization to set softmax scale dynamically."""
        if self.softmax_scale is None:
            self.softmax_scale = 1 / math.sqrt(self.d_model // self.num_heads)
