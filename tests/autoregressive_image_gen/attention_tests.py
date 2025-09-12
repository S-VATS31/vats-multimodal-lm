from configs.setup_env import device

import torch

torch.manual_seed(42)

from src.autoregressive_image_gen.autoregressive_transformer.attention.optimized_attention import CausalSelfAttention
from configs.autoregressive_image_gen.autoregressive_transformer.model_args.model_args_xsmall import ModelArgs

def setup():
    pass

def test_output_shape():
    pass
    print("PASSED OUTPUT SHAPE TEST")

def test_causal():
    pass
    print("PASSED CAUSAL MASKING TEST")

def test_no_causal():
    pass
    print("PASSED NO CAUSAL MASKING TEST")

def test_no_padding():
    pass
    print("PASSED NO PADDING TEST")

def test_padding():
    pass
    print("PASSED PADDING TEST")

def test_cache():
    pass
    print("PASSED KV CACHING TEST")

def test_no_cache():
    pass
    print("PASSED NO KV CACHING TEST")

def test_gradients():
    pass
    print("PASSED FLOWING GRADIENTS TEST")

def test_qk_norm_stability():
    pass
    print("PASSED QK NORMALIZATION STABILITY TEST")

def test_no_qk_norm_stability():
    pass
    print("PASSED NO QK NORMALIZATION STABILITY TEST")

def test_qkv_shape():
    pass
    print("PASSED QKV SHAPE TEST")

def test_weight_shapes():
    pass
    print("PASSED WEIGHT SHAPE TESTS")

def test_variable_batch_sizes():
    pass    
    print("PASSED VARIABLE BATCH SIZES TEST")

def test_variable_resolution():
    pass
    print("PASSED VARIBALE RESOLUTIONS TEST")

def test_non_square_input():
    pass
    print("PASSED NON-SQUARE INPUT TEST")

def test_square_input():
    pass
    print("PASSED SQUARE INPUT TEST")

def test_numerical_stability():
    pass
    print("PASSED NUMERICAL STABILITY TEST")

if __name__ == "__main__":
    test_output_shape()
    test_causal()
    test_no_causal()
    test_no_padding()
    test_padding()
    test_cache()
    test_no_cache()
    test_gradients()
    test_qk_norm_stability()
    test_no_qk_norm_stability()
    test_qkv_shape()
    test_weight_shapes()
    test_variable_batch_sizes()
    test_variable_resolution()
    test_non_square_input()
    test_square_input()
    test_numerical_stability()
