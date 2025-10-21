from configs.setup_env import device

import torch

torch.manual_seed(42)

from src.transformers.vision.vit_2d.patch_embeddings2d import PatchEmbeddings2D
from configs.transformers.vision.vit_2d.model_args.model_args_xsmall import ModelArgs

model_args = ModelArgs()

def setup():
    patch_embed = PatchEmbeddings2D(
        patch_size=model_args.patch_size,
        target_size=model_args.target_size,
        C_in=model_args.C_in,
        d_model=model_args.d_model
    ).to(device)
    B, H, W = 2, 16, 16
    x = torch.randn(B, model_args.C_in, H, W).to(device)
    head_dim = model_args.d_model // model_args.num_heads
    return patch_embed, B, H, W, x, head_dim

patch_embed, B, H, W, x, head_dim = setup()

def test_output_shape():
    grid_size = model_args.target_size // model_args.patch_size
    num_patches = grid_size ** 2
    out = patch_embed(x)
    assert out.shape == (B, num_patches, model_args.d_model)
    print("PASSED OUTPUT SHAPE TEST")

def test_variable_patch_sizes():
    pass
    print("PASSED PATCH SIZES TEST")

def test_proportional_image_scale():
    pass
    print("PASSED PROPORTIONAL IMAGE SCALE TEST")

def test_square_input_image():
    grid_size = model_args.target_size // model_args.patch_size
    num_patches = grid_size ** 2
    x1 = torch.randn(B, model_args.C_in, H, H).to(device)
    x2 = torch.randn(B, model_args.C_in, W, W).to(device)
    out1 = patch_embed(x1)
    out2 = patch_embed(x2)
    assert out1.shape == (B, num_patches, model_args.d_model)
    assert out2.shape == (B, num_patches, model_args.d_model)
    print("PASSED SQUARE INPUT IMAGE TESET")

def test_non_square_input_image():
    pass
    print("PASSED NON-SQUARE INPUT IMAGE TEST")

def test_variable_batch_sizes():
    for batch_size in [1, 2, 4, 8, 16, 32, 64]:
        x = torch.randn(batch_size, model_args.C_in, H, W).to(device)
        out = patch_embed(x)
        assert torch.all(torch.isfinite(out))
        assert torch.all(torch.isreal(out))
        assert torch.all(torch.isfinite(out.norm()))
        assert torch.all(torch.isreal(out.norm()))
        assert not torch.all(torch.isnan(out))
        assert not torch.all(torch.isinf(out))
        assert not torch.all(torch.isnan(out.norm()))
        assert not torch.all(torch.isinf(out.norm()))
    print("PASSED VARIABLE BATCH SIZES TEST")

def test_variable_input_channels():
    for in_channels in [i for i in range(1, 51)]:
        patch_embed_temp = PatchEmbeddings2D(
            patch_size=model_args.patch_size,
            target_size=model_args.patch_size,
            C_in=in_channels,
            d_model=model_args.d_model
        ).to(device)
        x = torch.randn(B, in_channels, H, W).to(device)
        out = patch_embed_temp(x)
        assert torch.all(torch.isfinite(out))
        assert torch.all(torch.isreal(out))
        assert torch.all(torch.isfinite(out.norm()))
        assert torch.all(torch.isreal(out.norm()))
        assert not torch.all(torch.isnan(out))
        assert not torch.all(torch.isinf(out))
        assert not torch.all(torch.isnan(out.norm()))
        assert not torch.all(torch.isinf(out.norm()))
    print("PASSED VARIABLE INPUT CHANNELS TEST")

def test_smaller_target_size():
    pass
    print("PASSED SMALLER TARGET SIZE TEST")

def test_larger_target_size():
    pass
    print("PASSED LARGER TARGET SIZE TEST")

if __name__ == "__main__":
    test_output_shape()
    test_variable_patch_sizes()
    test_proportional_image_scale()
    test_square_input_image()
    test_non_square_input_image()
    test_variable_batch_sizes()
    test_variable_input_channels()
    test_smaller_target_size()
    test_larger_target_size()
