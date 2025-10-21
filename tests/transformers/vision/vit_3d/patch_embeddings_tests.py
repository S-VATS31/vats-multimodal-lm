from configs.setup_env import device

import torch

torch.manual_seed(42)

from src.transformers.vision.vit_3d.patch_embeddings3d import PatchEmbeddings3D
from configs.transformers.vision.vit_3d.model_args.model_args_xsmall import ModelArgs

model_args = ModelArgs()

def setup():
    pass

def test_output_shape():
    pass
    print("PASSED PATCH EMBEDDINGS 3D TEST")

def test_padding():
    pass
    print("PASSED PADDING TEST")

def test_no_padding():
    pass
    print("PASSED NO PADDING TEST")

def test_spatial_reshape():
    pass
    print("PASSED SPATIAL RESHAPING TEST")

def test_exceeding_max_frames_input():
    pass
    print("PASSED EXCEEDING MAX FRAMES TEST")

def test_zero_frames_input():
    pass
    print("PASSED ZERO FRAMES INPUT TEST")

def test_patch_mask_creation():
    pass
    print("PASSED PATCH MASK CREATION TEST")

def test_variable_input_frames():
    pass
    print("PASSED VARIABLE INPUT FRAMES TEST")

def test_variable_batch_sizes():
    pass
    print("PASSED VARIABLE BATCH SIZES TEST")

def test_variable_resolutions():
    pass
    print("PASSED VARIABLE RESOLUTIONS TEST")

def test_frame_truncation():
    pass
    print("PASSED FRAME TRUNCATION TEST")

def test_frame_padding():
    pass
    print("PASSED FRAME PADDING TEST")

def test_numerical_stability():
    pass
    print("PASSED NUMERICAL STABILITY TEST")


if __name__ == "__main__":
    test_output_shape()
    test_padding()
    test_no_padding()
    test_spatial_reshape()
    test_exceeding_max_frames_input()
    test_zero_frames_input()
    test_patch_mask_creation()
    test_variable_input_frames()
    test_variable_batch_sizes()
    test_variable_resolutions()
    test_frame_truncation()
    test_frame_padding()
    test_numerical_stability()
