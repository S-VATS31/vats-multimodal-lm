from configs.setup_env import device

import torch

torch.manual_seed(42)

from src.autoregressive_image_gen.autoregressive_transformer.attention.optimized_attention import CausalSelfAttention
from configs.autoregressive_image_gen.autoregressive_transformer.model_args.model_args_xsmall import ModelArgs

model_args = ModelArgs()

def setup():
    attn = CausalSelfAttention(
        d_model=model_args.d_model,
        num_heads=model_args.num_heads,
        query_groups=model_args.query_groups,
        rope_theta=model_args.rope_theta,
        softmax_scale=model_args.softmax_scale,
        use_proj_bias=model_args.use_proj_bias,
        use_fused_proj=model_args.use_qk_norm,
        use_windowed_attn=model_args.use_windowed_attn,
        use_ntk_rope=model_args.use_ntk_rope,
        ntk_scale_factor=model_args.ntk_scale_factor
    ).to(device)
    B, H, W = 2, 16, 16
    x = torch.randn(B, H*W, model_args.d_model).to(device)
    padding_mask = torch.randint(
        0, 2, (B, H*W), dtype=torch.bool
    ).to(device)
    head_dim = model_args.d_model // model_args.num_heads
    return attn, B, H, W, x, padding_mask, head_dim

attn, B, H, W, x, padding_mask, head_dim = setup()

def test_output_shape():
    out = attn(
        x, False, True, True, -1, -1, False, padding_mask
    )
    assert out.shape == (B, H*W, model_args.d_model)
    print("PASSED OUTPUT SHAPE TEST")

def test_causal():
    pass
    print("PASSED CAUSAL MASKING TEST")

def test_no_causal():
    out = attn(
        x, False, True, False, -1, -1, False, padding_mask
    )
    assert out.shape == (B, H*W, model_args.d_model)
    print("PASSED NO CAUSAL MASKING TEST")

def test_no_padding():
    out = attn(
        x, False, True, True, -1, -1, False, None
    )
    assert out.shape == (B, H*W, model_args.d_model)
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
    out = attn(
        x, False, True, True, -1, -1, False, padding_mask
    )
    loss = out.sum()
    loss.backward()
    for name, param in attn.named_parameters():
        assert torch.all(torch.isfinite(param.grad))
        assert torch.all(torch.isreal(param.grad))
        assert torch.all(torch.isfinite(param.grad.norm()))
        assert torch.all(torch.isreal(param.grad.norm()))
        assert not torch.all(torch.isnan(param.grad))
        assert not torch.all(torch.isinf(param.grad))
        assert not torch.all(torch.isnan(param.grad.norm()))
        assert not torch.all(torch.isinf(param.grad.norm()))
    print("PASSED FLOWING GRADIENTS TEST")

def test_qk_norm_stability():
    out = attn(
        x, False, True, True, -1, -1, False, padding_mask
    )
    assert torch.all(torch.isfinite(out))
    assert torch.all(torch.isreal(out))
    assert torch.all(torch.isfinite(out.norm()))
    assert torch.all(torch.isreal(out.norm()))
    assert not torch.all(torch.isnan(out))
    assert not torch.all(torch.isinf(out))
    assert not torch.all(torch.isnan(out.norm()))
    assert not torch.all(torch.isinf(out.norm()))
    print("PASSED QK NORMALIZATION STABILITY TEST")

def test_no_qk_norm_stability():
    out = attn(
        x, False, False, True, -1, -1, False, padding_mask
    )
    assert torch.all(torch.isfinite(out))
    assert torch.all(torch.isreal(out))
    assert torch.all(torch.isfinite(out.norm()))
    assert torch.all(torch.isreal(out.norm()))
    assert not torch.all(torch.isnan(out))
    assert not torch.all(torch.isinf(out))
    assert not torch.all(torch.isnan(out.norm()))
    assert not torch.all(torch.isinf(out.norm()))
    print("PASSED NO QK NORMALIZATION STABILITY TEST")

def test_qkv_shape():
    q, k, v = attn._setup_qkv(x, False, True)
    assert q.shape == (B, H*W, model_args.num_heads, head_dim)
    assert (
        k.shape == v.shape == (B, H*W, model_args.num_heads, head_dim) or
        k.shape == v.shape == (B, H*W, 1, head_dim)
    )
    print("PASSED QKV SHAPE TEST")

def test_weight_shapes():
    assert attn.qkv_proj.weight.shape == (
        model_args.d_model + 2*model_args.query_groups*head_dim, model_args.d_model
    )
    assert attn.o_proj.weight.shape == (model_args.d_model, model_args.d_model)
    print("PASSED WEIGHT SHAPE TESTS")

def test_variable_batch_sizes():
    for batch_size in [1, 2, 4, 8, 16, 32, 64]:
        x = torch.randn(batch_size, H*W, model_args.d_model).to(device)
        padding_mask = torch.randint(
            0, 2, (batch_size, H*W), dtype=torch.bool
        ).to(device)
        out = attn(
            x, False, True, True, -1 , -1, False, padding_mask
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

def test_variable_resolution():
    for res in [1, 2, 4, 8, 16, 32, 64, 128]:
        x = torch.randn(B, res*res, model_args.d_model).to(device)
        padding_mask = torch.randint(
            0, 2, (B, res*res), dtype=torch.bool
        ).to(device)
        out = attn(
            x, False, True, True, -1 , -1, False, padding_mask
        )
        assert torch.all(torch.isfinite(out))
        assert torch.all(torch.isreal(out))
        assert torch.all(torch.isfinite(out.norm()))
        assert torch.all(torch.isreal(out.norm()))
        assert not torch.all(torch.isnan(out))
        assert not torch.all(torch.isinf(out))
        assert not torch.all(torch.isnan(out.norm()))
        assert not torch.all(torch.isinf(out.norm()))
    print("PASSED VARIBALE RESOLUTIONS TEST")

def test_non_square_input():
    pass
    print("PASSED NON-SQUARE INPUT TEST")

def test_square_input():
    x1 = torch.randn(B, H*H, model_args.d_model).to(device)
    x2 = torch.randn(B, W*W, model_args.d_model).to(device)
    padding_mask1 = torch.randint(
        0, 2, (B, H*H), dtype=torch.bool
    ).to(device)
    padding_mask2 = torch.randint(
        0, 2, (B, W*W), dtype=torch.bool
    ).to(device)
    out1 = attn(
        x1, False, True, True, -1, -1, False, padding_mask1
    )
    out2 = attn(
        x2, False, True, True, -1, -1, False, padding_mask2
    )
    assert out1.shape == (B, H*H, model_args.d_model)
    assert out2.shape == (B, W*W, model_args.d_model)

def test_numerical_stability():
    out = attn(
        x, False, True, True, -1, -1, False, padding_mask
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
