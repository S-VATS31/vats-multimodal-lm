import math
from typing import Optional
from dataclasses import dataclass

from configs.transformers.nlp.model_args.post_init import PostInitMixin

@dataclass
class ModelArgs(PostInitMixin):
    """Large configuration of model arguments."""
    d_model: int = 4096
    num_heads: int = 32
    query_groups: int = 8
    softmax_scale: Optional[float] = None
    d_ffn: int = 14336
    num_layers: int = 32
    dropout: float = 0.2
    rope_base: float = 10000.0
    rms_norm_eps: float = 1e-7
    left_window: int = 512
    right_window: int = 0
    vocab_size: int = 65536
    max_seq_len: int = 32768
    tie_weights: bool = True
    gradient_checkpointing: bool = True
    use_proj_bias: bool = False
    use_qkv_proj: bool = True
    use_causal: bool = True
    use_mqa: bool = True
    use_cache: bool = False
    max_batch_size: int = 2048
    num_experts: int = 32
    top_k: int = 2


    def __post_init__(self):
        """Post initialization to set softmax scale dynamically."""
        if self.softmax_scale is None:
            self.softmax_scale = 1 / math.sqrt(self.d_model // self.num_heads)
