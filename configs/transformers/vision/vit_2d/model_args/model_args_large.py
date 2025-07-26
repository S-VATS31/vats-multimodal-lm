from dataclasses import dataclass
from typing import Tuple

@dataclass
class ModelArgs:
    """Large configuration of model arguments, containing 625 million parameters.
    
    Args:
        img_size (int): Height and width of the input image.
        patch_size (int): Size of each square patch.
        C_in (int): Number of input channels.
        d_model (int): Dimensionality of the model's embeddings.
        num_heads (int): Number of attention heads.
        query_groups (int): Number of query groups for GQA.
        d_ffn (int): Dimensionality of the FFN.
        num_layers (int): Number of transformer layers to be stacked.
        window_size (Tuple[int, int]): Symmetrical window size for sliding window attention.
        dropout (float): Dropout probability.
        rope_base (float): Theta hyperparameter for RoPE.
        rms_norm_eps (float): Epsilon value for RMSNorm to prevent numerical instability.
        num_classes (int): Number of classes that are possible based on input image.
        use_checkpointing (bool): Whether to apply gradient checkpointing or not.
    """
    img_size: int = 384
    patch_size: int = 16
    C_in: int = 3
    d_model: int = 1440
    num_heads: int = 24
    query_groups: int = 12
    d_ffn: int = 5760
    num_layers: int = 20
    window_size: Tuple[int, int] = (512, 512)
    dropout: float = 0.2
    rope_base: float = 30000.0
    rms_norm_eps: float = 1e-7
    num_classes = 1000  # change for different datasets
    use_checkpointing: bool = True
    
    def __post_init__(self):
        """Post initialization for assertions."""
        if self.d_model % self.num_heads != 0:
            raise ValueError(f"Expected d_model to be divible by num_heads, got {self.d_model} % {self.num_heads} != 0.")
        if self.num_heads % self.query_groups != 0:
            raise ValueError(f"Expected num_heads to be divible by query_groups, got {self.num_heads} % {self.query_groups} != 0.")
        if self.d_model * 4 != self.d_ffn:
            raise ValueError(f"Expected d_ffn = d_model * 4, got {self.d_model} * 4 != {self.d_ffn}")
        if len(self.window_size) != 2:
            raise ValueError(f"Expected len(window_size) to be equal to 2, got {len(self.window_size)}")
        if self.window_size[0] != self.window_size[1]:
            raise ValueError(f"Expected left and right windows to be equal, got {self.window_size[0]} != {self.window_size[1]}")
        