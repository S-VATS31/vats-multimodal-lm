from typing import Optional, Tuple, Literal
from dataclasses import dataclass

@dataclass
class ModelArgs:
    """Large configuration of model arguments 1.2 billion parameters."""
    patch_size: Tuple[int, int, int] = (2, 4, 4)
    max_frames: int = 10
    d_model: int = 128
    num_heads: int = 16
    query_groups: int = 4
    max_batch_size: int = 32
    softmax_scale = 1 / ((128//8) ** 0.5)
    d_ffn: int = 512
    num_layers: int = 4
    dropout: float = 0.1
    rope_theta: float = 10000.0
    use_ntk_rope: bool = True
    ntk_scale_factor: float = 0.7
    left_window: int = -1
    right_window: int = -1
    rms_norm_eps: float = 1e-5
    vocab_size: int = 4096
    max_position_embeddings: int = 256
    use_checkpointing: bool = True
    use_proj_bias: bool = False
    use_qkv_proj: bool = False
    use_mqa: bool = False
    use_qk_norm: bool = True
    use_causal: bool = True
    use_windowed_attn: bool = True
    vae_encoder_activation: Literal["relu", "leaky_relu", "sigmoid"] = "relu"
    num_embeddings: int = 512
    commitment_beta: float = 0.7
    C_in_out: int = 3
