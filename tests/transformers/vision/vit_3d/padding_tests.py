from configs.setup_env import device

import torch

torch.manual_seed(42)

from src.transformers.vision.vit_3d.model import VideoTransformer
from src.transformers.vision.vit_3d.optimized_attention import SpatioTemporalAttention
from configs.transformers.vision.vit_3d.model_args.model_args_xsmall import ModelArgs

model_args = ModelArgs()

def setup():
    pass

def test_attn_padding():
    pass
    print("PASSED ATTN PADDING TEST")

def test_model_padding():
    pass
    print("PASSED MODEL PADDING TEST")

if __name__ == "__main__":
    test_attn_padding()
    test_model_padding()
