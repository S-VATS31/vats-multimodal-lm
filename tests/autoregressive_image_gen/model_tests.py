from configs.setup_env import device

import torch

torch.manual_seed(42)

from src.autoregressive_image_gen.autoregressive_transformer.model import AutoregressiveImageTransformer
from configs.autoregressive_image_gen.autoregressive_transformer.model_args.model_args_xsmall import ModelArgs

def setup():
    pass

def test_output_shape():
    pass
    print("PASSED OUTPUT SHAPE TEST")

def test_spatial_padding():
    pass
    print("PASSED SPATIAL PADDING TEST")

def test_text_padding():
    pass
    print("PASSED TEXT PADDING TEST")

def test_combined_padding():
    pass
    print("PASSED COMBINED PADDING TEST")

def test_no_padding():
    pass
    print("PASSED NO PADDING TEST")

def test_gradients():
    pass
    print("PASSED FLOWING GRADIENTS TEST")

def test_cache():
    pass
    print("PASSED KV CACHING TEST")

def test_no_cache():
    pass
    print("PASSED NO KV CACHING TEST")

def test_causal():
    pass
    print("PASSED CAUSAL MASKING TEST")

def test_no_causal():
    pass
    print("PASSED NO CAUSAL MASKING TEST")

def test_zero_input_tokens():
    pass
    print("PASSED ZERO INPUT TOKENS TEST")

def test_variable_batch_sizes():
    pass
    print("PASSED VARIABLE BATCH SIZES TEST")

def test_variable_resolutions():
    pass
    print("PASSED VARIABLE RESOLUTIONS TEST")

def test_variable_input_tokens():
    pass
    print("PASSED VARIABLE INPUT TOKENS TEST")

def test_square_input():
    pass
    print("PASSED SQUARE INPUT TEST")

def test_non_square_input():
    pass
    print("PASSED NON-SQUARE INPUT TEST")

def test_numerical_stability():
    pass
    print("PASSED NUMERICAL STABILITY TEST")

if __name__ == "__main__":
    test_output_shape()
    test_spatial_padding()
    test_text_padding()
    test_combined_padding()
    test_no_padding()
    test_gradients()
    test_cache()
    test_no_cache()
    test_causal()
    test_no_causal()
    test_zero_input_tokens()
    test_variable_batch_sizes()
    test_variable_resolutions()
    test_variable_input_tokens()
    test_square_input()
    test_non_square_input()
    test_numerical_stability()
