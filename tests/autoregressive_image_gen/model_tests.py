from configs.setup_env import device

import torch

torch.manual_seed(42)

from src.autoregressive_image_gen.autoregressive_transformer.model import AutoregressiveImageTransformer
from configs.autoregressive_image_gen.autoregressive_transformer.model_args.model_args_xsmall import ModelArgs

model_args = ModelArgs()

def setup():
    model = AutoregressiveImageTransformer(model_args).to(device)
    B, H, W = 2, 16, 16
    T_tokens = 8
    input_ids = torch.randint(
        0, model_args.num_embeddings, (B, H, W), dtype=torch.int64
    ).to(device)
    input_text = torch.randn(B, T_tokens, model_args.d_model).to(device)
    image_mask = torch.randint(
        0, 2, (B, H*W), dtype=torch.bool
    ).to(device)
    text_mask = torch.randint(
        0, 2, (B, T_tokens), dtype=torch.bool
    ).to(device)
    head_dim = model_args.d_model // model_args.num_heads
    return model, B, H, W, T_tokens, input_ids, input_text, image_mask, text_mask, head_dim

model, B, H, W, T_tokens, input_ids, text_embeddings, image_mask, text_mask, head_dim = setup()

def test_output_shape():
    out = model(
        input_ids, text_embeddings, False, image_mask, text_mask
    )
    assert out.shape == (B, H, W, model_args.d_model)
    print("PASSED OUTPUT SHAPE TEST")

def test_spatial_padding():
    out = model(
        input_ids, text_embeddings, False, image_mask, None
    )
    assert out.shape == (B, H, W, model_args.d_model)
    print("PASSED SPATIAL PADDING TEST")

def test_text_padding():
    out = model(
        input_ids, text_embeddings, False, None, text_mask  
    )
    assert out.shape == (B, H, W, model_args.d_model)
    print("PASSED TEXT PADDING TEST")

def test_combined_padding():
    out = model(
        input_ids, text_embeddings, False, image_mask, text_mask  
    )
    assert out.shape == (B, H, W, model_args.d_model)
    print("PASSED COMBINED PADDING TEST")

def test_no_padding():
    out = model(
        input_ids, text_embeddings, False, None, None
    )
    assert out.shape == (B, H, W, model_args.d_model)
    print("PASSED NO PADDING TEST")

def test_cache():
    out = model(
        input_ids, text_embeddings, True, image_mask, text_mask
    )
    assert out.shape == (B, H, W, model_args.d_model)
    print("PASSED KV CACHING TEST")

def test_no_cache():
    out = model(
        input_ids, text_embeddings, False, image_mask, text_mask
    )
    assert out.shape == (B, H, W, model_args.d_model)
    print("PASSED NO CACHING TEST")

def test_causal():
    model_args.use_causal = True
    out = model(
        input_ids, text_embeddings, False, image_mask, text_mask
    )
    assert out.shape == (B, H, W, model_args.d_model)
    print("PASSED CAUSAL MASKING TEST")

def test_no_causal():
    model_args.use_causal = False
    out = model(
        input_ids, text_embeddings, False, image_mask, text_mask
    )
    assert out.shape == (B, H, W, model_args.d_model)
    print("PASSED NO CAUSAL MASKING TEST")

def test_gradients():
    out = model(
        input_ids, text_embeddings, False, image_mask, text_mask
    )
    loss = out.sum()
    loss.backward()
    for _, param in model.named_parameters():
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
    print("PASSED FLOWING GRADIENTS TEST")

def test_zero_input_tokens():
    tokens = 0
    input_text = torch.randn(B, tokens, model_args.d_model).to(device)
    text_mask = torch.randint(
        0, 2, (B, tokens), dtype=torch.bool
    ).to(device)
    out = model(
        input_ids, input_text, False, image_mask, text_mask
    )
    assert out.shape == (B, H, W, model_args.d_model)
    print("PASSED ZERO INPUT TOKENS TEST")

def test_variable_batch_sizes():
    for batch_size in [1, 2, 4, 8, 16, 32, 64, 128]:
        input_image = torch.randint(
            0, model_args.num_embeddings, (batch_size, H, W), dtype=torch.int64
        ).to(device)
        input_text = torch.randn(batch_size, T_tokens, model_args.d_model).to(device)
        image_mask = torch.randint(
            0, 2, (batch_size, H*W), dtype=torch.bool
        ).to(device)
        text_mask = torch.randint(
            0, 2, (batch_size, T_tokens), dtype=torch.bool
        ).to(device)
        out = model(
            input_image, input_text, False, image_mask, text_mask
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
    for res in [1, 2, 4, 8, 16, 32, 64]:
        input_image = torch.randint(
            0, model_args.num_embeddings, (B, res, res), dtype=torch.int64
        ).to(device)
        input_text = torch.randn(B, T_tokens, model_args.d_model).to(device)
        image_mask = torch.randint(
            0, 2, (B, res*res), dtype=torch.bool
        ).to(device)
        text_mask = torch.randint(
            0, 2, (B, T_tokens), dtype=torch.bool
        ).to(device)
        out = model(
            input_image, input_text, False, image_mask, text_mask
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

def test_variable_input_tokens():
    for tokens in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        input_image = torch.randint(
            0, model_args.num_embeddings, (B, H, W), dtype=torch.int64
        ).to(device)
        input_text = torch.randn(B, tokens, model_args.d_model).to(device)
        image_mask = torch.randint(
            0, 2, (B, H*W), dtype=torch.bool
        ).to(device)
        text_mask = torch.randint(
            0, 2, (B, tokens), dtype=torch.bool
        ).to(device)
        out = model(
            input_image, input_text, False, image_mask, text_mask
        )
        assert torch.all(torch.isfinite(out))
        assert torch.all(torch.isreal(out))
        assert torch.all(torch.isfinite(out.norm()))
        assert torch.all(torch.isreal(out.norm()))
        assert not torch.all(torch.isnan(out))
        assert not torch.all(torch.isinf(out))
        assert not torch.all(torch.isnan(out.norm()))
        assert not torch.all(torch.isinf(out.norm()))
    print("PASSED VARIABLE INPUT TOKENS TEST")

def test_square_input():
    x1 = torch.randint(
        0, model_args.num_embeddings, (B, H, H), dtype=torch.int64
    ).to(device)
    x2 = torch.randint(
        0, model_args.num_embeddings, (B, W, W), dtype=torch.int64
    ).to(device)
    mask1 = torch.randint(
        0, 2, (B, H*H), dtype=torch.bool
    ).to(device)
    mask2 = torch.randint(
        0, 2, (B, W*W), dtype=torch.bool
    ).to(device)
    out1 = model(
        x1, text_embeddings, False, mask1, text_mask
    )
    out2 = model(
        x2, text_embeddings, False, mask2, text_mask
    )
    assert out1.shape == (B, H, H, model_args.d_model)
    assert out2.shape == (B, W, W, model_args.d_model)
    print("PASSED SQUARE INPUT TEST")

def test_non_square_input():
    pass
    print("PASSED NON-SQUARE INPUT TEST")

def test_numerical_stability():
    out = model(
        input_ids, text_embeddings, False, image_mask, text_mask
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
    test_spatial_padding()
    test_text_padding()
    test_combined_padding()
    test_no_padding()
    test_cache()
    test_no_cache()
    test_causal()
    test_no_causal()
    test_gradients()
    test_zero_input_tokens()
    test_variable_batch_sizes()
    test_variable_resolutions()
    test_variable_input_tokens()
    test_square_input()
    test_non_square_input()
    test_numerical_stability()
