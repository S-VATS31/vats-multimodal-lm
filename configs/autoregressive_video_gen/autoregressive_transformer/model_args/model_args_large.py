from typing import Optional, Tuple, Literal
from dataclasses import dataclass

@dataclass
class ModelArgs:
    """Large configuration of model arguments 1.2 billion parameters."""
    patch_size: Tuple[int, int, int] = (2, 8, 8)
    max_frames: int = 30
    d_model: int = 1792
    num_heads: int = 32
    query_groups: int = 4
    max_batch_size: int = 32
    softmax_scale: Optional[float] = None
    d_ffn: int = 7168
    num_layers: int = 20
    dropout: float = 0.2
    rope_theta: float = 10000.0
    use_ntk_rope: bool = True
    ntk_scale_factor: float = 0.7
    left_window: int = -1
    right_window: int = -1
    rms_norm_eps: float = 1e-12
    vocab_size: int = 65536
    max_position_embeddings: int = 2048
    use_checkpointing: bool = True
    use_proj_bias: bool = False
    use_qkv_proj: bool = True
    use_mqa: bool = False
    use_qk_norm: bool = True
    use_causal: bool = True
    use_windowed_attn: bool = True
    vae_encoder_activation: Literal["relu", "leaky_relu", "sigmoid"] = "relu"
    num_embeddings: int = 256
    commitment_beta: float = 0.7
    C_in_out: int = 3
