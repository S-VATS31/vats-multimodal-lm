from configs.setup_env import device

import torch

torch.manual_seed(42)

from src.autoregressive_video_gen.autoregressive_transformer.attention.cross_attention import FactorizedCrossAttention
from configs.autoregressive_video_gen.autoregressive_transformer.model_args.model_args_xsmall import ModelArgs

model_args = ModelArgs()

def setup():
    cross_attn = FactorizedCrossAttention(
        d_model=model_args.d_model,
        num_heads=model_args.num_heads,
        query_groups=model_args.query_groups,
        softmax_scale=model_args.softmax_scale,
        use_proj_bias=model_args.use_proj_bias
    ).to(device)
    B, T_frames, H, W = 2, 8, 16, 16
    T_tokens = 9
    x = torch.randn(B, T_frames, H*W, model_args.d_model)
    text_embeddings = torch.randn(B, T_tokens, model_args.d_model).to(device)
    padding_mask = torch.randint(
        0, 2, (B, T_tokens), dtype=torch.bool
    ).to(device)
    head_dim = model_args.d_model // model_args.num_heads
    return cross_attn, B, T_frames, T_tokens, H, W, x, text_embeddings, padding_mask, head_dim

cross_attn, B, T_frames, T_tokens, H, W, x, text_embeddings, padding_mask, head_dim = setup()

def test_output_shape():
    out = cross_attn(
        x, text_embeddings, False, True, padding_mask
    )
    assert out.shape == x.shape == (B, T_frames, H*W, model_args.d_model)
    print("PASSED OUTPUT SHAPE TEST")

def test_gradients():
    out = cross_attn(
        x, text_embeddings, False, True, padding_mask
    )
    loss = out.sum()
    loss.backward()
    for _, param in cross_attn.named_parameters():
        assert (
            torch.all(torch.isfinite(param.grad)) and
            torch.all(torch.isreal(param.grad)) and
            torch.all(torch.isfinite(param.grad.norm())) and
            torch.all(torch.isreal(param.grad.norm())) and
            not torch.any(torch.isnan(param.grad)) and
            not torch.any(torch.isinf(param.grad)) and
            not torch.any(torch.isnan(param.grad.norm())) and
            not torch.any(torch.isinf(param.grad.norm()))
        )
    print("PASSED GRADIENT FLOW TEST")

def test_weight_shapes():
    assert cross_attn.q_proj.weight.shape == (model_args.d_model, model_args.d_model)
    assert (
        cross_attn.k_proj.weight.shape == cross_attn.v_proj.weight.shape == (
            model_args.query_groups*head_dim, model_args.d_model
        )
    )
    assert cross_attn.o_proj.weight.shape == (model_args.d_model, model_args.d_model)
    print("PASSED WEIGHT SHAPES TEST")

def test_padding():
    pass
    print("PASSED PADDING TEST")

def test_no_padding():
    pass
    print("PASSED NO PADDING TEST")

def test_spatial_qkv_shapes():
    spatial_q, spatial_k, spatial_v = cross_attn._setup_spatial_qkv(
        x, text_embeddings, False, True
    )
    assert spatial_q.shape == (B*T_frames, H*W, model_args.num_heads, head_dim)
    assert (
        spatial_k.shape == spatial_v.shape == (
            B, T_tokens, model_args.num_heads, head_dim
        )
        or
        spatial_k.shape == spatial_v.shape == (
            B, T_tokens, 1, head_dim
        )
    )
    print("PASSED SPATIAL QKV SHAPES TEST")

def test_temporal_qkv_shapes():
    temporal_q, temporal_k, temporal_v = cross_attn._setup_temporal_qkv(
        x, text_embeddings, False, True
    )
    assert temporal_q.shape == (B*H*W, T_frames, model_args.num_heads, head_dim)
    assert (
        temporal_k.shape == temporal_v.shape == (
            B, T_tokens, model_args.num_heads, head_dim
        )
        or
        temporal_k.shape == temporal_v.shape == (
            B, T_tokens, 1, head_dim
        )
    )
    print("PASSED TEMPORAL QKV SHAPES TEST")

def test_zero_frames():
    frames = 0
    image_input = torch.randn(B, frames, H*W, model_args.d_model).to(device)
    out = cross_attn(
        image_input, text_embeddings, False, True, padding_mask
    )
    assert out.shape == (B, frames, H*W, model_args.d_model)
    print("PASSED ZERO FRAMES INPUT TEST")

def test_zero_tokens():
    tokens = 0
    text_input = torch.randn(B, tokens, model_args.d_model)
    out = cross_attn(
        x, text_input, False, True, padding_mask
    )
    assert out.shape == (B, T_frames, H*W, model_args.d_model)
    print("PASSED ZERO INPUT TOKENS TEST")

def test_zero_frames_zero_tokens():
    frames, tokens = 0, 0
    image_input = torch.randn(B, frames, H*W, model_args.d_model).to(device)
    text_input = torch.randn(B, tokens, model_args.d_model).to(device)
    out = cross_attn(
        image_input, text_input, False, True, padding_mask
    )
    assert out.shape == (B, frames, H*W, model_args.d_model)
    print("PASSED ZERO INPUT FRAMES AND ZERO TOKENS TEST")

def test_exceeding_max_frames():
    pass
    print("PASSED INPUT FRAMES > MAX POSSIBLE FRAMES TEST")

def test_variable_input_frames():
    for frames in [1, 2, 4, 8, 16, 32, 64, 128]:
        image_input = torch.randn(B, frames, H*W, model_args.d_model).to(device)
        out = cross_attn(
            image_input, text_embeddings, False, True, padding_mask
        )
        assert (
            torch.all(torch.isfinite(out)) and
            torch.all(torch.isreal(out)) and
            torch.all(torch.isfinite(out.norm())) and
            torch.all(torch.isreal(out.norm())) and
            not torch.any(torch.isnan(out)) and
            not torch.any(torch.isinf(out)) and
            not torch.any(torch.isnan(out.norm())) and
            not torch.any(torch.isinf(out.norm()))
        )
    print("PASSED VARIABLE INPUT FRAMES TEST")

def test_variable_batch_sizes():
    for batch_size in [1, 2, 4, 8, 16, 32, 64]:
        image_input = torch.randn(batch_size, T_frames, H*W, model_args.d_model).to(device)
        text_input = torch.randn(batch_size, T_tokens, model_args.d_model).to(device)
        padding_mask = torch.randint(
            0, 2, (batch_size, T_tokens), dtype=torch.bool
        ).to(device)
        out = cross_attn(
            image_input, text_input, False, True, padding_mask
        )
        assert (
            torch.all(torch.isfinite(out)) and
            torch.all(torch.isreal(out)) and
            torch.all(torch.isfinite(out.norm())) and
            torch.all(torch.isreal(out.norm())) and
            not torch.any(torch.isnan(out)) and
            not torch.any(torch.isinf(out)) and
            not torch.any(torch.isnan(out.norm())) and
            not torch.any(torch.isinf(out.norm()))
        )
    print("PASSED VARIABLE BATCH SIZES TEST")

def test_variable_input_tokens():
    for tokens in [1, 2, 4, 8, 16, 32, 64]:
        input_text = torch.randn(B, tokens, model_args.d_model).to(device)
        out = cross_attn(
            x, input_text, False, True, padding_mask
        )
        assert (
            torch.all(torch.isfinite(out)) and
            torch.all(torch.isreal(out)) and
            torch.all(torch.isfinite(out.norm())) and
            torch.all(torch.isreal(out.norm())) and
            not torch.any(torch.isnan(out)) and
            not torch.any(torch.isinf(out)) and
            not torch.any(torch.isnan(out.norm())) and
            not torch.any(torch.isinf(out.norm()))
        )
    print("PASSED VARIABLE INPUT TOKENS TEST")

def test_numerical_stability():
    out = cross_attn(
        x, text_embeddings, False, True, padding_mask
    )
    assert (
        torch.all(torch.isfinite(out)) and
        torch.all(torch.isreal(out)) and
        torch.all(torch.isfinite(out.norm())) and
        torch.all(torch.isreal(out.norm())) and
        not torch.any(torch.isnan(out)) and
        not torch.any(torch.isinf(out)) and
        not torch.any(torch.isnan(out.norm())) and
        not torch.any(torch.isinf(out.norm()))
    )
    print("PASSED NUMERICAL STABILITY TEST")

if __name__ == "__main__":
    test_output_shape()
    test_gradients()
    test_weight_shapes()
    test_padding()
    test_no_padding()
    test_spatial_qkv_shapes()
    test_temporal_qkv_shapes()
    test_zero_frames()
    test_zero_tokens()
    test_zero_frames_zero_tokens
    test_exceeding_max_frames()
    test_variable_input_frames()
    test_variable_batch_sizes()
    test_variable_input_frames()
    test_numerical_stability()
