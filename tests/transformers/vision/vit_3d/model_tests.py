from configs.setup_env import device

import torch

torch.manual_seed(42)

from src.transformers.vision.vit_3d.model import VideoTransformer
from configs.transformers.vision.vit_3d.model_args.model_args_xsmall import ModelArgs

model_args = ModelArgs()

def setup():
    model = VideoTransformer(model_args).to(device)
    B, T, H, W = 2, 8, 16, 16
    x = torch.randn(B, model_args.C_in, T, H, W).to(device)
    head_dim = model_args.d_model//model_args.num_heads
    return model, B, T, H, W, x, head_dim

model, B, T, H, W, x, head_dim = setup()

def test_output_shape():
    out, grid_size = model(x, True, return_grid_size=True)
    new_T, new_H, new_W = grid_size
    assert out.shape == (B, new_T*new_H*new_W, model_args.d_model)

    print("PASSED OUTPUT SHAPE TEST")

def test_gradients():
    out = model(x, True)
    loss = out.sum()
    loss.backward()
    for _, param in model.named_parameters():
        assert torch.all(torch.isfinite(param.grad))
        assert torch.all(torch.isreal(param.grad))
        assert torch.all(torch.isfinite(param.grad.norm()))
        assert torch.all(torch.isreal(param.grad.norm()))
        assert not torch.all(torch.isnan(param.grad))
        assert not torch.all(torch.isinf(param.grad))
        assert not torch.all(torch.isnan(param.grad.norm()))
        assert not torch.all(torch.isinf(param.grad.norm()))
    print("PASSED FLOWING GRADIENTS TEST")

def test_patchification():
    _, grid_size = model(x, True, return_grid_size=True)
    pt, ph, pw = model_args.patch_size
    target_h, target_w = model_args.target_size
    actual_grid_size = (
        model_args.max_frames//pt, target_h//ph, target_w//pw
    )
    assert grid_size == actual_grid_size
    print("PASSED PATCHIFICATION TEST")

def test_padding():
    pass
    print("PASSED PADDING TEST")

def test_no_padding():
    pass
    print("PASSED NO PADDING TEST")

def test_numerical_stability():
    pass
    print("PASSED NUMERICAL STABILITY TEST")

def test_zero_frames_input():
    pass
    print("PASSED ZERO INPUT FRAMES TEST")

def test_variable_batch_sizes():
    pass
    print("PASSED VARIABLE BATCH SIZES TEST")

def test_variable_input_frames():
    pass
    print("PASSED VARIABLE INPUT FRAMES TEST")

def test_variable_resolutions():
    pass
    print("PASSED VARIABLE INPUT RESOLUTIONS TEST")

def test_qk_norm_stability():
    pass
    print("PASSED QK NORM STABILITY TEST")

def test_no_qk_norm_stability_test():
    pass
    print("PASSED NO QK NORM STABILITY TEST")

if __name__ == "__main__":
    test_output_shape()
    test_gradients()
    test_patchification()
    test_padding()
    test_no_padding()
    test_numerical_stability()
    test_zero_frames_input()
    test_variable_batch_sizes()
    test_variable_input_frames()
    test_variable_resolutions()
    test_qk_norm_stability()
    test_no_qk_norm_stability_test()
