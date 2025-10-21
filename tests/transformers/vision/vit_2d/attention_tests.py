from configs.setup_env import device

import torch

torch.manual_seed(42)

from src.transformers.vision.vit_2d.optimized_attention import SpatialAttention
from configs.transformers.vision.vit_2d.model_args.model_args_xsmall import ModelArgs

model_args = ModelArgs()

def setup():
    attn = SpatialAttention(
        d_model=model_args.d_model,
        num_heads=model_args.num_heads,
        query_groups=model_args.query_groups,
        rope_theta=model_args.rope_theta,
        target_size=model_args.target_size,
        patch_size=model_args.patch_size,
        softmax_scale=model_args.softmax_scale,
        use_windowed_attn=model_args.use_windowed_attn,
        use_proj_bias=model_args.use_proj_bias,
        use_fused_proj=model_args.use_fused_proj
    ).to(device)
    B = 2
    grid_size = model_args.target_size // model_args.patch_size
    num_patches = grid_size ** 2
    x = torch.randn(B, num_patches, model_args.d_model).to(device)
    head_dim = model_args.d_model//model_args.num_heads
    return attn, B, num_patches, x, head_dim

attn, B, num_patches, x, head_dim = setup()

def test_output_shape():
    out = attn(x, False, True, -1, -1)
    assert out.shape == (B, num_patches, model_args.d_model)
    print("PASSED OUTPUT SHAPE TEST")

def test_weight_shapes():
    assert attn.qkv_proj.weight.shape == (
        model_args.d_model+2*model_args.query_groups*head_dim, 
        model_args.d_model
    )
    assert attn.o_proj.weight.shape == (
        model_args.d_model, model_args.d_model
    )
    print("PASSED WEIGHT SHAPES TEST")

def test_qkv_shapes():
    q, k, v = attn._setup_qkv(x, False, True)
    assert q.shape == (B, num_patches, model_args.num_heads, head_dim)
    assert (
        k.shape == v.shape == (B, num_patches, model_args.num_heads, head_dim) or
        k.shape == v.shape == (B, num_patches, 1, head_dim)
    )
    print("PASSED QKV SHAPES TEST")

def test_gradients():
    out = attn(x, False, True, -1, -1)
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

def test_windowed_attn():
    out = attn(x, False, True, 256, 256)
    assert out.shape == (B, num_patches, model_args.d_model)
    print("PASSED WINDOWED ATTN TEST")

def test_qk_norm_stability():
    out = attn(x, False, True, -1, -1)
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
    out = attn(x, False, False, -1, -1)
    assert torch.all(torch.isfinite(out))
    assert torch.all(torch.isreal(out))
    assert torch.all(torch.isfinite(out.norm()))
    assert torch.all(torch.isreal(out.norm()))
    assert not torch.all(torch.isnan(out))
    assert not torch.all(torch.isinf(out))
    assert not torch.all(torch.isnan(out.norm()))
    assert not torch.all(torch.isinf(out.norm()))
    print("PASSED NO QK NORM STABILITY TEST")

def test_numerical_stability():
    out = attn(x, False, True, -1, -1)
    assert torch.all(torch.isfinite(out))
    assert torch.all(torch.isreal(out))
    assert torch.all(torch.isfinite(out.norm()))
    assert torch.all(torch.isreal(out.norm()))
    assert not torch.all(torch.isnan(out))
    assert not torch.all(torch.isinf(out))
    assert not torch.all(torch.isnan(out.norm()))
    assert not torch.all(torch.isinf(out.norm()))
    print("PASSED NUMERICAL STABILITY TEST")

def test_variable_batch_sizes():
    for batch_size in [1, 2, 4, 8, 16, 32, 64]:
        x = torch.randn(batch_size, num_patches, model_args.d_model).to(device)
        out = attn(x, False, True, -1, -1)
        assert torch.all(torch.isfinite(out))
        assert torch.all(torch.isreal(out))
        assert torch.all(torch.isfinite(out.norm()))
        assert torch.all(torch.isreal(out.norm()))
        assert not torch.all(torch.isnan(out))
        assert not torch.all(torch.isinf(out))
        assert not torch.all(torch.isnan(out.norm()))
        assert not torch.all(torch.isinf(out.norm()))
    print("PASSED VARIABLE BATCH SIZES TEST")

if __name__ == "__main__":
    test_output_shape()
    test_weight_shapes()
    test_qkv_shapes()
    test_gradients()
    test_windowed_attn()
    test_qk_norm_stability()
    test_no_qk_norm_stability()
    test_numerical_stability()
    test_variable_batch_sizes()
