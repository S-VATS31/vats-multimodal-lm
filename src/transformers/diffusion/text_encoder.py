from configs.setup_env import device, dtype

from typing import Optional, Tuple, List

# We can use most parts from NLP Transformer for this
from src.transformers.nlp.optimized_attention import RoPE
from src.transformers.nlp.optimized_attention import Attention
from src.transformers.nlp.optimized_attention import AttentionBlock
from src.rms_norm import RMSNorm
from src.swiglu_activation import SwiGLUActivation
from src.transformers.vision.ffn_block import FFNBlock # TODO: move to src/ffn_block

# RoPE
# Attention
# FFN
# RMSNorm
# AttentionBlock
# FFNBlock
# TransformerBlock, encoder
# Transformer
# don't use sliding window attention -> global attention
# pass -1 for left and right window for text only encoder
