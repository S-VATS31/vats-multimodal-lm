from dataclasses import dataclass

@dataclass
class ModelArgs:
    """Small configuration of model arguments, containing 190 million parameters."""
    img_size: int = 384
    patch_size: int = 16
    C_in: int = 3
    d_model: int = 1024
    num_heads: int = 16
    query_groups: int = 8
    d_ffn: int = 4096
    num_layers: int = 12
    dropout: float = 0.2
    rope_base: float = 30000.0
    rms_norm_eps: float = 1e-7
    num_classes = 1000 # change for different datasets
    use_checkpointing: bool = True
    