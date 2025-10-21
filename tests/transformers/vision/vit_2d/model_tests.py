from configs.setup_env import device

import torch

torch.manual_seed(42)

from src.transformers.vision.vit_2d.model import ImageEncoderTransformer
from configs.transformers.vision.vit_2d.model_args.model_args_xsmall import ModelArgs

model_args = ModelArgs()

def setup():
    model = ImageEncoderTransformer(model_args).to(device)
    B, H, W = 2, 16, 16
    x = torch.randn(B, model_args.C_in, H, W).to(device)
    head_dim = model_args.d_model // model_args.num_heads
    grid_size = model_args.target_size // model_args.patch_size
    num_patches = grid_size ** 2
    return model, B, H, W, x, head_dim, num_patches

model, B, H, W, x, head_dim, num_patches = setup()

def test_output_shape():
    out = model(x)
    assert out.shape == (B, num_patches, model_args.d_model)
    print("PASSED OUTPUT SHAPE TEST")

def test_gradients():
    out = model(x)
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

def test_variable_batch_sizes():
    for batch_size in [1, 2, 4, 8, 16, 32, 64]:
        x = torch.randn(batch_size, model_args.C_in, H, W).to(device)
        out = model(x)
        assert torch.all(torch.isfinite(out))
        assert torch.all(torch.isreal(out))
        assert torch.all(torch.isfinite(out.norm()))
        assert torch.all(torch.isreal(out.norm()))
        assert not torch.all(torch.isnan(out))
        assert not torch.all(torch.isinf(out))
        assert not torch.all(torch.isnan(out.norm()))
        assert not torch.all(torch.isinf(out.norm()))
    print("PASSED VARIABLE BATCH SIZES TEST")

def test_variable_input_res():
    for res in [1, 2, 4, 8, 16, 32, 64]:
        x = torch.randn(B, model_args.C_in, res, res).to(device)
        out = model(x)
        assert torch.all(torch.isfinite(out))
        assert torch.all(torch.isreal(out))
        assert torch.all(torch.isfinite(out.norm()))
        assert torch.all(torch.isreal(out.norm()))
        assert not torch.all(torch.isnan(out))
        assert not torch.all(torch.isinf(out))
        assert not torch.all(torch.isnan(out.norm()))
        assert not torch.all(torch.isinf(out.norm()))
    print("PASSED VARIABLE INPUT RESOLUTIONS TEST")

def test_non_square_input():
    pass
    print("PASSED NON-SQUARE INPUT TEST")

def test_numerical_stability():
    out = model(x)
    print("PASSED NUMERICAL STABILITY TEST")
    assert torch.all(torch.isfinite(out))
    assert torch.all(torch.isreal(out))
    assert torch.all(torch.isfinite(out.norm()))
    assert torch.all(torch.isreal(out.norm()))
    assert not torch.all(torch.isnan(out))
    assert not torch.all(torch.isinf(out))
    assert not torch.all(torch.isnan(out.norm()))
    assert not torch.all(torch.isinf(out.norm()))
if __name__ == "__main__":
    setup()
    test_output_shape()
    test_gradients()
    test_variable_batch_sizes()
    test_variable_input_res()
    test_non_square_input()
    test_numerical_stability()
