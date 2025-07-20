from dataclasses import dataclass

@dataclass
class ModelArgs:
    """Extra large configuration of model arguments, containing 1.5 billion parameters."""
    img_size: int = 384
    patch_size: int = 16
    C_in: int = 3
    d_model: int = 2048
    num_heads: int = 32
    query_groups: int = 16
    d_ffn: int = 8092
    num_layers: int = 24
    dropout: float = 0.2
    rope_base: float = 30000.0
    rms_norm_eps: float = 1e-7
    num_classes = 1000 # change for different datasets
    use_checkpointing: bool = True
    