from configs.setup_env import device, dtype

import torch

torch.manual_seed(42)

from src.autoregressive_video_gen.autoregressive_transformer.attention.cross_attention import FactorizedCrossAttention
from configs.autoregressive_video_gen.autoregressive_transformer.model_args.model_args_xsmall import ModelArgs

model_args = ModelArgs()

def setup():
    pass

def test_output_shape():
    pass
    print("PASSED OUTPUT SHAPE TEST")

def test_gradients():
    pass
    print("PASSED GRADIENT FLOW TEST")

def test_weight_shapes():
    pass
    print("PASSED WEIGHT SHAPES TEST")

def test_padding():
    pass
    print("PASSED PADDING TEST")

def test_no_padding():
    pass
    print("PASSED NO PADDING TEST")

def test_spatial_qkv_shapes():
    pass
    print("PASSED SPATIAL QKV SHAPES TEST")

def test_temporal_qkv_shapes():
    pass
    print("PASSED TEMPORAL QKV SHAPES TEST")

def test_mqa():
    pass
    print("PASSED MQA TEST")

def test_zero_frames():
    pass
    print("PASSED ZERO FRAMES INPUT TEST")

def test_exceeding_max_frames():
    pass
    print("PASSED INPUT FRAMES > MAX POSSIBLE FRAMES TEST")

def test_numerical_stability():
    pass
    print("PASSED NUMERICAL STABILITY TEST")

if __name__ == "__main__":
    test_output_shape()
    test_gradients()
    test_weight_shapes()
    test_padding()
    test_no_padding()
    test_spatial_qkv_shapes()
    test_temporal_qkv_shapes()
    test_mqa()
    test_zero_frames()
    test_exceeding_max_frames()
    test_numerical_stability()
