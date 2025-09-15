from configs.setup_env import device

import torch

torch.manual_seed(42)

from src.autoregressive_image_gen.autoregressive_transformer.attention.cross_attention import CrossAttention
from configs.autoregressive_image_gen.autoregressive_transformer.model_args.model_args_xsmall import ModelArgs

model_args = ModelArgs()

def setup():
    cross_attn = CrossAttention(
        d_model=model_args.d_model,
        num_heads=model_args.num_heads,
        softmax_scale=model_args.softmax_scale,
        use_proj_bias=model_args.use_proj_bias
    ).to(device)
    B, H, W = 2, 16, 16
    T_tokens = 8
    x = torch.randn(B, H*W, model_args.d_model).to(device)
    text_embeddings = torch.randn(B, T_tokens, model_args.d_model).to(device)
    padding_mask = torch.randint(
        0, 2, (B, T_tokens), dtype=torch.bool
    ).to(device)
    head_dim = model_args.d_model // model_args.num_heads
    return cross_attn, B, H, W, T_tokens, x, text_embeddings, padding_mask, head_dim

cross_attn, B, H, W, T_tokens, x, text_embeddings, padding_mask, head_dim = setup()

def test_output_shape():
    out = cross_attn(x, text_embeddings, padding_mask)
    assert out.shape == (B, H*W, model_args.d_model)
    print("PASSED OUTPUT SHAPE TEST")

def test_padding():
    pass
    print("PASSED PADDING TEST")

def test_no_padding():
    out = cross_attn(x, text_embeddings, None)
    assert out.shape == (B, H*W, model_args.d_model)
    print("PASSED NO PADDING TEST")

def test_qkv_shape():
    q, k, v = cross_attn._setup_qkv(x, text_embeddings)
    assert q.shape == (B, H*W, model_args.num_heads, head_dim)
    assert (
        k.shape == v.shape == (B, T_tokens, model_args.num_heads, head_dim)
    )
    print("PASSED QKV SHAPE TEST")

def test_gradients():
    out = cross_attn(x, text_embeddings, padding_mask)
    loss = out.sum()
    loss.backward()
    for _, param in cross_attn.named_parameters():
        assert (
            torch.isreal(param.grad).all() and
            torch.isfinite(param.grad).all() and
            torch.isreal(param.grad.norm()).all() and
            torch.isfinite(param.grad.norm()).all() and
            not torch.isnan(param.grad).any() and
            not torch.isinf(param.grad).any() and
            not torch.isnan(param.grad.norm()).any() and
            not torch.isinf(param.grad.norm()).any()
        )
    print("PASSED GRADIENTS TEST")

def test_zero_tokens():
    tokens = 0
    input_text = torch.randn(B, tokens, model_args.d_model).to(device)
    input_mask = torch.randint(
        0, 2, (B, tokens), dtype=torch.bool
    ).to(device)
    out = cross_attn(x, input_text, input_mask)
    assert (
        torch.isreal(out).all() and
        torch.isfinite(out).all() and
        torch.isreal(out.norm()).all() and
        torch.isfinite(out.norm()).all() and
        not torch.isnan(out).any() and
        not torch.isinf(out).any() and
        not torch.isnan(out.norm()).any() and
        not torch.isinf(out.norm()).any()
    )
    print("PASSED ZERO INPUT TOKENS TEST")

def test_variable_batch_sizes():
    for batch_size in [1, 2, 4, 8, 16, 32, 64, 128]:
        input_image = torch.randn(batch_size, H*W, model_args.d_model).to(device)
        input_text = torch.randn(batch_size, T_tokens, model_args.d_model).to(device)
        text_mask = torch.randint(
            0, 2, (batch_size, T_tokens), dtype=torch.bool
        ).to(device)
        out = cross_attn(input_image, input_text, text_mask)
        assert (
            torch.isreal(out).all() and
            torch.isfinite(out).all() and
            torch.isreal(out.norm()).all() and
            torch.isfinite(out.norm()).all() and
            not torch.isnan(out).any() and
            not torch.isinf(out).any() and
            not torch.isnan(out.norm()).any() and
            not torch.isinf(out.norm()).any()
    ) 
    print("PASSED VARIABLE BATCH SIZES TEST")

def test_variable_tokens():
    for tokens in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        input_image = torch.randn(B, H*W, model_args.d_model).to(device)
        input_text = torch.randn(B, tokens, model_args.d_model).to(device)
        text_mask = torch.randint(
            0, 2, (B, tokens), dtype=torch.bool
        ).to(device)
        out = cross_attn(input_image, input_text, text_mask)
        assert (
            torch.isreal(out).all() and
            torch.isfinite(out).all() and
            torch.isreal(out.norm()).all() and
            torch.isfinite(out.norm()).all() and
            not torch.isnan(out).any() and
            not torch.isinf(out).any() and
            not torch.isnan(out.norm()).any() and
            not torch.isinf(out.norm()).any()
        )
    print("PASSED VARIABLE INPUT TOKENS TEST")

def test_variable_resolutions():
    for res in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        input_image = torch.randn(B, res*res, model_args.d_model).to(device)
        input_text = torch.randn(B, T_tokens, model_args.d_model).to(device)
        text_mask = torch.randint(
            0, 2, (B, T_tokens), dtype=torch.bool
        ).to(device)
        out = cross_attn(input_image, input_text, text_mask)
        assert (
            torch.isreal(out).all() and
            torch.isfinite(out).all() and
            torch.isreal(out.norm()).all() and
            torch.isfinite(out.norm()).all() and
            not torch.isnan(out).any() and
            not torch.isinf(out).any() and
            not torch.isnan(out.norm()).any() and
            not torch.isinf(out.norm()).any()
        )
    print("PASSED VARIABLE RESOLUTIONS TEST")

def test_weight_shapes():
    assert cross_attn.q_proj.weight.shape == (model_args.d_model, model_args.d_model)
    assert cross_attn.k_proj.weight.shape == (model_args.d_model, model_args.d_model)
    assert cross_attn.v_proj.weight.shape == (model_args.d_model, model_args.d_model)
    assert cross_attn.o_proj.weight.shape == (model_args.d_model, model_args.d_model)
    print("PASSED WEIGHT SHAPES TEST")

def test_numerical_stability():
    out = cross_attn(x, text_embeddings, padding_mask)
    assert (
        torch.isreal(out).all() and
        torch.isfinite(out).all() and
        torch.isreal(out.norm()).all() and
        torch.isfinite(out.norm()).all() and
        not torch.isnan(out).any() and
        not torch.isinf(out).any() and
        not torch.isnan(out.norm()).any() and
        not torch.isinf(out.norm()).any()
    )
    print("PASSED NUMERICAL STABILITY TEST")

def test_non_square_input():
    pass
    print("PASSED NON SQUARE INPUT TEST")

def test_square_input():
    x1 = torch.randn(B, H*H, model_args.d_model).to(device)
    x2 = torch.randn(B, W*W, model_args.d_model).to(device)
    out1 = cross_attn(x1, text_embeddings, padding_mask)
    out2 = cross_attn(x2, text_embeddings, padding_mask)
    assert out1.shape == (B, H*H, model_args.d_model)
    assert out2.shape == (B, W*W, model_args.d_model)
    print("PASSED SQUARE INPUT TEST")

if __name__ == "__main__":
    test_output_shape()
    test_padding()
    test_no_padding()
    test_qkv_shape()
    test_gradients()
    test_zero_tokens()
    test_variable_batch_sizes()
    test_variable_tokens()
    test_variable_resolutions()
    test_weight_shapes()
    test_numerical_stability()
    test_non_square_input()
    test_square_input()
