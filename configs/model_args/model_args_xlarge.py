"""
Extra large configuration of model arguments.

This configuration contains ~13 billion parameters.
NOTE: This configuration has been not been tested due to the
computational power needed. This is just a rough estimate based
on the formula:
 - Total parameters ≈ L * (4 * d_model^2 + 2 * d_model * d_ffn) + 2 * d_model * V

Note that the training arguments will need to updated 
based on different ModelArgs parameter sizes for better
training and convergence.
"""

from dataclasses import dataclass

@dataclass
class ModelArgs:
    """
    d_model (int): Dimensionality of the model's embeddings.
    num_heads (int): Number of attention heads for the query tensors (GQA).
    query_groups (int): Number of query groups for the key and value tensors (GQA).
    d_ffn (int): Dimensionality of the feed forward network. d_ffn = 4 * d_model.
    num_layers (int): Number of times the transformer block will be stacked.
    dropout (float): Probability of random model components being dropped out.
    rope_base (float): Exponential base of the RoPE inverse frequency.
    rms_norm_eps (float): Small epsilon value to ensure numerical stability in RMSNorm.
    vocab_size (int): Number of unique tokens the model can recognize.
    max_seq_len (int): Longest input sequence the model can handle at once.
    tie_weights (bool): Flag to tie weights or not.
    pad_token_id (int): Number reserved for the padding token which will be masked out.
    eos_token_id (int): End of sequence token id which is typically vocab_size - 1.
    gradient_checkpointing (bool): Flag to apply gradient checkpointing or not.
    """
    d_model: int = 5120
    num_heads: int = 40
    query_groups: int = 10
    d_ffn: int = 20480
    num_layers: int = 40
    dropout: float = 0.2
    rope_base: float = 10000.0
    rms_norm_eps: float = 1e-7
    vocab_size: int = 65536
    max_seq_len: int = 32768
    tie_weights: bool = False
    pad_token_id: int = 0
    eos_token_id: int = 65535
    gradient_checkpointing: bool = False
    num_experts: int = 64
    top_k: int = 2