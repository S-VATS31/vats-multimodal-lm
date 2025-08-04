from typing import Optional
from dataclasses import dataclass
import math

from configs.transformers.nlp.model_args.post_init import PostInitMixin

@dataclass
class ModelArgs(PostInitMixin):
    """Small configuration of model arguments."""
    d_model: int = 512
    num_heads: int = 8
    query_groups: int = 4
    softmax_scale: Optional[float] = None
    d_ffn: int = 2048
    num_layers: int = 12
    dropout: float = 0.2
    rope_base: float = 10000.0
    rms_norm_eps: float = 1e-7
    left_window: int = 256
    right_window: int = 0
    vocab_size: int = 65536
    max_seq_len: int = 2048
    tie_weights: bool = False
    max_batch_size: int = 2048
    gradient_checkpointing: bool = True
    use_proj_bias: bool = False
    use_qkv_proj: bool = True
    use_causal: bool = True
    use_mqa: bool = True
    use_cache: bool = False
    num_experts: int = 1
    top_k: int = 1

    def __post_init__(self):
        """Post initialization to set softmax scale dynamically."""
        if self.softmax_scale is None:
            self.softmax_scale = 1 / math.sqrt(self.d_model // self.num_heads)

        # Call assertions
        super().__post_init__()