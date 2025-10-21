from configs.setup_env import device

import torch

torch.manual_seed(42)

from src.transformers.vision.vit_3d.patch_embeddings3d import PatchEmbeddings3D
from src.transformers.vision.vit_3d.optimized_attention import SpatioTemporalAttention
from configs.transformers.vision.vit_3d.model_args.model_args_xsmall import ModelArgs

model_args = ModelArgs()

def setup():
    attn = SpatioTemporalAttention(
        d_model=model_args.d_model,
        num_heads=model_args.num_heads,
        query_groups=model_args.query_groups,
        rope_theta=model_args.rope_theta,
        patch_size=model_args.patch_size
    ).to(device)
    patch_embed = PatchEmbeddings3D(
        C_in=model_args.C_in,
        patch_size=model_args.patch_size,
        target_size=model_args.target_size,
        max_frames=model_args.max_frames,
        d_model=model_args.d_model
    ).to(device)
    B, T, H, W = 2, 8, 16, 16
    x = torch.randn(B, model_args.C_in, T, H, W).to(device)
    return attn, patch_embed, B, T, H, W, x, model_args.d_model//model_args.num_heads

attn, patch_embed, B, T, H, W, x, head_dim = setup()

def test_output_shape():
    attn_inp, padding_mask, grid_size = patch_embed(x)
    new_T, new_H, new_W = grid_size
    out = attn(
        attn_inp, grid_size, False, True, (-1, -1), padding_mask
    )
    assert out.shape == (B, new_T, new_H*new_W, model_args.d_model)
    print("PASSED OUTPUT SHAPE TEST")

def test_qkv_shapes():
    attn_inp, _, grid_size = patch_embed(x)
    new_T, new_H, new_W = grid_size
    spatial_q, spatial_k, spatial_v = attn._setup_qkv(
        attn_inp, False, True, grid_size, attn_mode="spatial"
    )
    temporal_q, temporal_k, temporal_v = attn._setup_qkv(
        attn_inp, False, True, grid_size, attn_mode="temporal"
    )
    assert spatial_q.shape == (B*new_T, new_H*new_W, model_args.num_heads, head_dim)
    assert (
        spatial_k.shape == spatial_v.shape == (B*new_T, new_H*new_W, model_args.num_heads, head_dim) or
        spatial_k.shape == spatial_v.shape == (B*new_T, new_H*new_W, 1, head_dim)
    )
    assert temporal_q.shape == (B*new_H*new_W, new_T, model_args.num_heads, head_dim)
    assert (
        temporal_k.shape == temporal_v.shape == (B*new_H*new_W, new_T, model_args.num_heads, head_dim) or
        temporal_k.shape == temporal_v.shape == (B*new_H*new_W, new_T, 1, head_dim)
    )
    print("PASSED QKV SHAPES TEST")

def test_no_padding():
    attn_inp, _, grid_size = patch_embed(x)
    new_T, new_H, new_W = grid_size
    out = attn(
        attn_inp, grid_size, False, True, (-1, -1), None
    )
    assert out.shape == (B, new_T, new_H*new_W, model_args.d_model)
    print("PASSED NO PADDING TEST")

def test_padding():
    attn_inp, padding_mask, grid_size = patch_embed(x)
    new_T, new_H, new_W = grid_size
    out = attn(
        attn_inp, grid_size, False, True, (-1, -1), padding_mask
    )
    assert out.shape == (B, new_T, new_H*new_W, model_args.d_model)
    print("PASSED PADDING TEST")

def test_weight_shapes():
    assert attn.w_qkv.weight.shape == (
        model_args.d_model+2*model_args.query_groups*head_dim,
        model_args.d_model
    )
    assert attn.w_o.weight.shape == (
        model_args.d_model, model_args.d_model
    )
    print("PASSED WEIGHT SHAPES TEST")

def test_qk_norm_stability():
    attn_inp, padding_mask, grid_size = patch_embed(x)
    new_T, new_H, new_W = grid_size
    out = attn(
        attn_inp, grid_size, False, True, (-1, -1), padding_mask
    )
    assert out.shape == (B, new_T, new_H*new_W, model_args.d_model)
    assert torch.all(torch.isfinite(out))
    assert torch.all(torch.isreal(out))
    assert torch.all(torch.isfinite(out.norm()))
    assert torch.all(torch.isreal(out.norm()))
    assert not torch.all(torch.isnan(out))
    assert not torch.all(torch.isinf(out))
    assert not torch.all(torch.isnan(out.norm()))
    assert not torch.all(torch.isinf(out.norm()))
    print("PASSED QK NORM STABILITY TEST")

def test_no_qk_norm_stability():
    attn_inp, padding_mask, grid_size = patch_embed(x)
    new_T, new_H, new_W = grid_size
    out = attn(
        attn_inp, grid_size, False, False, (-1, -1), padding_mask
    )
    assert out.shape == (B, new_T, new_H*new_W, model_args.d_model)
    assert torch.all(torch.isfinite(out))
    assert torch.all(torch.isreal(out))
    assert torch.all(torch.isfinite(out.norm()))
    assert torch.all(torch.isreal(out.norm()))
    assert not torch.all(torch.isnan(out))
    assert not torch.all(torch.isinf(out))
    assert not torch.all(torch.isnan(out.norm()))
    assert not torch.all(torch.isinf(out.norm()))
    print("PASSED NO QK NORM STABLITY TEST")

def test_windowed_attn():
    attn_inp, padding_mask, grid_size = patch_embed(x)
    new_T, new_H, new_W = grid_size
    out = attn(
        attn_inp, grid_size, False, True, (256, 256), padding_mask
    )
    assert out.shape == (B, new_T, new_H*new_W, model_args.d_model)
    print("PASSED WINDOWED ATTN TEST")

def test_no_windowed_attn():
    attn_inp, padding_mask, grid_size = patch_embed(x)
    new_T, new_H, new_W = grid_size
    out = attn(
        attn_inp, grid_size, False, True, (-1, -1), padding_mask
    )
    assert out.shape == (B, new_T, new_H*new_W, model_args.d_model)
    print("PASSED NO WINDOWED ATTN TEST")

def test_zero_input_frames():
    frames = 0
    x_temp = torch.randn(B, model_args.C_in, frames, H, W).to(device)
    attn_inp, padding_mask, grid_size = patch_embed(x_temp)
    new_T, new_H, new_W = grid_size
    out = attn(
        attn_inp, grid_size, False, True, (-1, -1), padding_mask
    )
    assert out.shape == (B, new_T, new_H*new_W, model_args.d_model)
    print("PASSED ZERO INPUT FRAMES TEST")

def test_gradients():
    attn_inp, padding_mask, grid_size = patch_embed(x)
    out = attn(
        attn_inp, grid_size, False, True, (-1, -1), padding_mask
    )
    loss = out.sum()
    loss.backward()
    for _, param in attn.named_parameters():
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
    for batch_size in [1, 2, 4, 8, 16]:
        x_temp = torch.randn(batch_size, model_args.C_in, T, H, W).to(device)
        attn_inp, padding_mask, grid_size = patch_embed(x_temp)
        out = attn(
            attn_inp, grid_size, False, True, (-1, -1), padding_mask
        )
        assert torch.all(torch.isfinite(out))
        assert torch.all(torch.isreal(out))
        assert torch.all(torch.isfinite(out.norm()))
        assert torch.all(torch.isreal(out.norm()))
        assert not torch.all(torch.isnan(out))
        assert not torch.all(torch.isinf(out))
        assert not torch.all(torch.isnan(out.norm()))
        assert not torch.all(torch.isinf(out.norm()))
    print("PASSED VARIABLE BATCH SIZES TEST")

def test_variable_resolutions():
    for res in [1, 2, 4, 8, 16, 32]:
        x_temp = torch.randn(B, model_args.C_in, T, res, res).to(device)
        attn_inp, padding_mask, grid_size = patch_embed(x_temp)
        out = attn(
            attn_inp, grid_size, False, True, (-1, -1), padding_mask
        )
        assert torch.all(torch.isfinite(out))
        assert torch.all(torch.isreal(out))
        assert torch.all(torch.isfinite(out.norm()))
        assert torch.all(torch.isreal(out.norm()))
        assert not torch.all(torch.isnan(out))
        assert not torch.all(torch.isinf(out))
        assert not torch.all(torch.isnan(out.norm()))
        assert not torch.all(torch.isinf(out.norm()))
    print("PASSED VARIABLE RESOLUTIONS TEST")

def test_variable_input_frames():
    for frames in [i for i in range(1, 26)]:
        x_temp = torch.randn(B, model_args.C_in, frames, H, W).to(device)
        attn_inp, padding_mask, grid_size = patch_embed(x_temp)
        out = attn(
            attn_inp, grid_size, False, True, (-1, -1), padding_mask
        )
        assert torch.all(torch.isfinite(out))
        assert torch.all(torch.isreal(out))
        assert torch.all(torch.isfinite(out.norm()))
        assert torch.all(torch.isreal(out.norm()))
        assert not torch.all(torch.isnan(out))
        assert not torch.all(torch.isinf(out))
        assert not torch.all(torch.isnan(out.norm()))
        assert not torch.all(torch.isinf(out.norm()))
    print("PASSED VARIABLE INPUT FRAMES TEST")

def test_numerical_stability():
    attn_inp, padding_mask, grid_size = patch_embed(x)
    out = attn(
            attn_inp, grid_size, False, True, (-1, -1), padding_mask
    )
    assert torch.all(torch.isfinite(out))
    assert torch.all(torch.isreal(out))
    assert torch.all(torch.isfinite(out.norm()))
    assert torch.all(torch.isreal(out.norm()))
    assert not torch.all(torch.isnan(out))
    assert not torch.all(torch.isinf(out))
    assert not torch.all(torch.isnan(out.norm()))
    assert not torch.all(torch.isinf(out.norm()))
    print("PASSED NUMERICAL STABILITY TEST")

if __name__ == "__main__":
    test_output_shape()
    test_qkv_shapes()
    test_no_padding()
    test_padding()
    test_weight_shapes()
    test_qk_norm_stability()
    test_no_qk_norm_stability()
    test_windowed_attn()
    test_no_windowed_attn()
    test_zero_input_frames()
    test_gradients()
    test_variable_batch_sizes()
    test_variable_resolutions()
    test_variable_input_frames()
    test_numerical_stability()
