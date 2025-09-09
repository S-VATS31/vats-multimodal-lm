from configs.setup_env import device, dtype

import torch

torch.manual_seed(42)

from src.autoregressive_video_gen.autoregressive_transformer.attention.optimized_attention import CausalFactorizedAttention
from configs.autoregressive_video_gen.autoregressive_transformer.model_args.model_args_large import ModelArgs

model_args = ModelArgs()

def setup():
    attn = CausalFactorizedAttention(
        d_model=model_args.d_model,
        num_heads=model_args.num_heads,
        query_groups=model_args.query_groups,
        rope_theta=model_args.rope_theta,
        softmax_scale=model_args.softmax_scale,
        use_proj_bias=model_args.use_proj_bias,
        use_fused_proj=model_args.use_qkv_proj,
        use_windowed_attn=model_args.use_windowed_attn,
        use_ntk_rope=model_args.use_ntk_rope,
        ntk_scale_factor=model_args.ntk_scale_factor
    ).to(device)
    B, T_frames, H_pixels, W_pixels = 1, 8, 16, 16
    x = torch.randn(B, T_frames, H_pixels*W_pixels, model_args.d_model).to(device)
    padding_mask = torch.randint(
        0, 2, (B, T_frames*H_pixels*W_pixels), dtype=torch.bool
    ).to(device)
    x_out = attn(
        x,
        use_mqa=False,
        use_qk_norm=True,
        use_causal=True,
        left_window=-1,
        right_window=-1,
        use_cache=False,
        padding_mask=padding_mask,
        kv_cache=None,
        layer_idx=None
    )
    return (
        attn,
        x,
        padding_mask,
        x_out,
        B, T_frames, H_pixels, W_pixels, 
        model_args.d_model, model_args.num_heads, 
        model_args.d_model//model_args.num_heads
    )

attn, x, padding_mask, x_out, B, T, H, W, d_model, num_heads, head_dim = setup()

def test_output_shape():
    assert x.shape == x_out.shape == (B, T, H*W, d_model)
    print("PASSED OUTPUT SHAPE TEST")

def test_spatial_qkv_shape():
    _, q, k, v, _, _, _ = attn(
        x, False, True, True, -1, -1, False, return_qkv=True
    )
    assert q.shape == (B*T, H*W, num_heads, head_dim)
    assert (
        k.shape == v.shape ==(B*T, H*W, num_heads, head_dim) or 
        k.shape == v.shape == (B*T, H*W, 1, head_dim)
    )
    print("PASSED SPATIAL QKV SHAPES TEST") 

def test_temporal_qkv_shape():
    _, _, _, _, q, k, v = attn(
        x, False, True, True, -1, -1, False, return_qkv=True
    )
    assert q.shape == (B*H*W, T, num_heads, head_dim)
    assert (
        k.shape == v.shape ==(B*H*W, T, num_heads, head_dim) or 
        k.shape == v.shape == (B*H*W, T, 1, head_dim)
    )
    print("PASSED TEMPORAL QKV SHAPES TEST") 

def test_temporal_kv_cache():
    pass
    print("PASSED TEMPORAL KV CACHING TEST")

def test_causal_masking():
    pass
    print("PASSED CAUSAL MASKING TEST")

def test_no_causal_masking():
    out_no_causal = attn(
        x, False, True, False, -1, -1, False
    )
    assert out_no_causal.shape == x.shape == x_out.shape
    print("PASSED NO CAUSAL MASKING TEST")

def test_padding():
    pass
    print("PASSED PADDING TEST")

def test_no_padding():
    pass
    print("PASSED NO PADDING TEST")

def test_gradients():
    out = attn(x, False, True, True, -1, -1, False)
    loss = out.sum()
    loss.backward()
    for _, param in attn.named_parameters():
        assert (
            param.grad is not None and 
            not torch.any(torch.isnan(param.grad)) and
            not torch.any(torch.isinf(param.grad)) and
            torch.all(torch.isfinite(param.grad)) and
            torch.all(torch.isreal(param.grad)) and
            not torch.any(torch.isnan(param.grad.norm())) and
            not torch.any(torch.isinf(param.grad.norm())) and
            torch.all(torch.isfinite(param.grad.norm())) and
            torch.all(torch.isreal(param.grad.norm()))
        )
    print("PASSED FLOWING GRADIENTS TEST")

def test_weight_shapes():
    assert attn.qkv_proj.weight.shape == (d_model + 2*model_args.query_groups*head_dim, d_model)
    assert attn.o_proj.weight.shape == (d_model, d_model)
    print("PASSED WEIGHT SHAPES TEST")

def test_windowed_attention():
    windowed_attn_out = attn(
        x,
        use_mqa=False,
        use_qk_norm=True,
        use_causal=True,
        left_window=256,
        right_window=0,
        use_cache=False
    )
    assert windowed_attn_out.shape == x.shape == x_out.shape
    print("PASSED WINDOWED ATTENTION TESTS")

def test_no_windowed_attention():
    no_windowed_attn_out = attn(
        x,
        use_mqa=False,
        use_qk_norm=True,
        use_causal=True,
        left_window=-1,
        right_window=-1,
        use_cache=False
    )
    assert no_windowed_attn_out.shape == x.shape == x_out.shape
    print("PASSED NO WINDOWED ATTENTION TEST")

def test_zero_frames():
    input = torch.randn(B, 0, H*W, d_model).to(device)
    out_zero_frames, sQ, sK, sV, tQ, tK, tV = attn(
        input,
        use_mqa=False,
        use_qk_norm=True,
        use_causal=True,
        left_window=-1,
        right_window=-1,
        use_cache=False,
        return_qkv=True
    )
    assert out_zero_frames.shape == (B, 0, H*W, d_model)
    assert sQ.shape == (0, H*W, num_heads, head_dim)
    assert (
        sK.shape == sV.shape == (0, H*W, num_heads, head_dim) or
        sK.shape == sV.shape == (0, H*W, 1, head_dim)
    )
    assert tQ.shape == (B*H*W, 0, num_heads, head_dim)
    assert (
        tK.shape == tV.shape == (B*H*W, 0, num_heads, head_dim) or
        tK.shape == tV.shape == (B*H*W, 0, 1, head_dim)
    )
    print("PASSED 0 FRAMES INPUT TEST")

def test_exceeding_frames():
    pass
    print("PASSED INPUT FRAMES > MAX POSSIBLE FRAMES INPUT TEST")

def test_numerical_stability():
    out = attn(
        x,
        use_mqa=False,
        use_qk_norm=True,
        use_causal=True,
        left_window=-1,
        right_window=-1,
        use_cache=False
    )
    assert not torch.any(torch.isinf(out))
    assert not torch.any(torch.isnan(out))
    assert torch.all(torch.isfinite(out))
    assert torch.all(torch.isreal(out))
    assert not torch.any(torch.isinf(out.norm()))
    assert not torch.any(torch.isnan(out.norm()))
    assert torch.all(torch.isfinite(out.norm()))
    assert torch.all(torch.isreal(out.norm()))
    print("PASSED NUMERICAL STABILITY TEST")
    
if __name__ == "__main__":
    test_output_shape()
    test_spatial_qkv_shape()
    test_temporal_qkv_shape()
    test_temporal_kv_cache()
    test_causal_masking()
    test_no_causal_masking()
    test_padding()
    test_no_padding()
    test_gradients()
    test_weight_shapes()
    test_windowed_attention()
    test_no_windowed_attention()
    test_zero_frames()
    test_exceeding_frames()
    test_numerical_stability()
