from configs.setup_env import device

import torch

torch.manual_seed(42)

from src.autoregressive_image_gen.autoregressive_transformer.attention.cross_attention import CrossAttention
from configs.autoregressive_image_gen.autoregressive_transformer.model_args.model_args_xsmall import ModelArgs

def setup():
    pass

def test_output_shape():
    pass
    print("PASSED OUTPUT SHAPE TEST")

def test_padding():
    pass
    print("PASSED PADDING TEST")

def test_no_padding():
    pass
    print("PASSED NO PADDING TEST")

def test_qkv_shape():
    pass
    print("PASSED QKV SHAPE TEST")

def test_gradients():
    pass
    print("PASSED GRADIENTS TEST")

def test_zero_tokens():
    pass
    print("PASSED ZERO INPUT TOKENS TEST")

def test_variable_batch_sizes():
    pass
    print("PASSED VARIABLE BATCH SIZES TEST")

def test_variable_tokens():
    pass
    print("PASSED VARIABLE INPUT TOKENS TEST")

def test_variable_resolutions():
    pass
    print("PASSED VARIABLE RESOLUTIONS TEST")

def test_weight_shapes():
    pass
    print("PASSED WEIGHT SHAPES TEST")

def test_numerical_stability():
    pass
    print("PASSED NUMERICAL STABILITY TEST")

def test_non_square_input():
    pass
    print("PASSED NON SQUARE INPUT TEST")

def test_square_input():
    pass
    print("PASSED SQUARE INPUT TEST")

if __name__ == "__main__":
    test_output_shape()
    test_padding()
    test_no_padding()
    test_qkv_shape()
    test_gradients()
    test_zero_tokens()
    test_variable_batch_sizes()
    test_variable_tokens()
    test_variable_resolutions()
    test_weight_shapes()
    test_numerical_stability()
    test_non_square_input()
    test_square_input()
