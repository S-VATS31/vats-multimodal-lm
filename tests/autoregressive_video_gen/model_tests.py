from configs.setup_env import device, dtype

import torch

torch.manual_seed(42)

from src.autoregressive_video_gen.autoregressive_transformer.model import AutoregressiveVideoTransformer
from configs.autoregressive_video_gen.autoregressive_transformer.model_args.model_args_xsmall import ModelArgs

model_args = ModelArgs()

def setup():
    pass

def test_output_shape():
    pass

def test_image_padding():
    pass

def test_text_padding():
    pass

def test_combined_padding():
    pass

def test_no_padding():
    pass

def test_cache():
    pass

def test_no_cache():
    pass

def test_zero_frames():
    pass

def test_zero_tokens():
    pass

def test_exceeding_max_frames():
    pass

def test_exceeding_max_tokens():
    pass

def test_square_input_hw():
    pass

def test_non_square_input_hw():
    pass

def test_gradients():
    pass

def test_weight_shapes():
    pass

def test_numerical_stability():
    pass

def test_variable_resolution():
    pass

def test_variable_batch_sizes():
    pass

def test_variable_input_frames():
    pass

def test_variable_input_tokens():
    pass

if __name__ == "__main__":
    test_output_shape()
    test_image_padding()
    test_text_padding()
    test_combined_padding()
    test_no_padding()
    test_cache()
    test_no_cache()
    test_zero_frames()
    test_zero_tokens()
    test_exceeding_max_frames()
    test_exceeding_max_tokens()
    test_square_input_hw()
    test_non_square_input_hw()
    test_gradients()
    test_weight_shapes()
    test_numerical_stability()
    test_variable_resolution()
    test_variable_batch_sizes()
    test_variable_input_frames()
    test_variable_input_tokens()
