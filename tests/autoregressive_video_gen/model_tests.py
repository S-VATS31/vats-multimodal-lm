from configs.setup_env import device, dtype

import torch

torch.manual_seed(42)

from src.autoregressive_video_gen.autoregressive_transformer.model import AutoregressiveVideoTransformer
from configs.autoregressive_video_gen.autoregressive_transformer.model_args.model_args_xsmall import ModelArgs

model_args = ModelArgs()

def setup():
    model = AutoregressiveVideoTransformer(model_args).to(device)
    B, T_frames, H, W = 2, 8, 16, 16
    T_tokens = 12
    x = torch.randint(
        0, model_args.num_embeddings, (B, T_frames, H, W), dtype=torch.int64
    ).to(device)
    text_embeddings = torch.randn(B, T_tokens, model_args.d_model).to(device)
    image_padding_mask = torch.randint(
        0, 2, (B, T_frames*H*W), dtype=torch.bool
    ).to(device)
    text_padding_mask = torch.randint(
        0, 2, (B, T_tokens), dtype=torch.bool
    ).to(device)
    d_model = model_args.d_model
    num_heads = model_args.num_heads
    head_dim = d_model // num_heads
    return (
        model, 
        B, T_frames, T_tokens, H, W, 
        x, text_embeddings, 
        image_padding_mask, text_padding_mask, 
        d_model, num_heads, head_dim
    )

(
    model, 
    B, T_frames, T_tokens, H, W, 
    x, text_embeddings, 
    image_padding_mask, text_padding_mask, 
    d_model, num_heads, head_dim
) = setup()

def test_output_shape():
    out = model(
        x, text_embeddings, False, image_padding_mask, text_padding_mask
    )
    assert out.shape == (B, T_frames, H, W, d_model)
    print("PASSED OUTPUT SHAPE TEST")

def test_image_padding():
    out = model(
        x, text_embeddings, False, image_padding_mask, None
    )
    assert out.shape == (B, T_frames, H, W, d_model)
    print("PASSED IMAGE PADDING TEST")

def test_text_padding():
    out = model(
        x, text_embeddings, False, None, text_padding_mask
    )
    assert out.shape == (B, T_frames, H, W, d_model)
    print("PASSED TEXT PADDING TEST")

def test_combined_padding():
    out = model(
        x, text_embeddings, False, image_padding_mask, text_padding_mask
    )
    assert out.shape == (B, T_frames, H, W, d_model)
    print("PASSED COMBINED PADDING TEST")

def test_no_padding():
    out = model(
        x, text_embeddings, False, None, None
    )
    assert out.shape == (B, T_frames, H, W, d_model)
    print("PASSED NO PADDING TEST")

def test_cache():
    pass

def test_zero_frames():
    frames = 0
    x = torch.randint(
        0, model_args.num_embeddings, (B, frames, H, W), dtype=torch.int64
    ).to(device)
    image_mask = torch.randint(
        0, 2, (B, frames*H*W), dtype=torch.bool
    ).to(device)
    out = model(
        x, text_embeddings, False, image_mask, None
    )
    assert out.shape == (B, frames, H, W, d_model)
    print("PASSED ZERO FRAMES TEST")

def test_zero_tokens():
    tokens = 0
    text_embeddings = torch.randn(B, tokens, d_model).to(device)
    text_mask = torch.randint(0, 2, (B, tokens), dtype=torch.bool).to(device)
    out = model(
        x, text_embeddings, False, None, text_mask
    )
    assert out.shape == (B, T_frames, H, W, d_model)
    print("PASSED ZERO TOKENS TEST")

def test_zero_frames_zero_tokens():
    frames, tokens = 0, 0
    x = torch.randint(
        0, 2, (B, frames, H, W), dtype=torch.bool
    ).to(device)
    image_mask = torch.randint(
        0, 2, (B, frames*H*W), dtype=torch.bool
    ).to(device)
    text_embeddings = torch.randn(B, tokens, d_model).to(device)
    text_mask = torch.randint(0, 2, (B, tokens), dtype=torch.bool).to(device)
    out = model(
        x, text_embeddings, False, image_mask, text_mask
    )
    assert out.shape == (B, frames, H, W, d_model)
    print("PASSED ZERO FRAMES AND TOKENS TEST")

def test_exceeding_max_frames():
    pass

def test_exceeding_max_tokens():
    pass

def test_exceeding_max_frames_max_tokens():
    pass

def test_square_input_hw():
    pass

def test_non_square_input_hw():
    pass

def test_gradients():
    out = model(
        x, text_embeddings, False, image_padding_mask, text_padding_mask
    )
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

def test_numerical_stability():
    out = model(
        x, text_embeddings, True, image_padding_mask, text_padding_mask
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

def test_square_variable_resolution():
    resolutions = [1, 2, 4, 8, 16, 32, 64]
    for res in resolutions:
        model = AutoregressiveVideoTransformer(model_args).to(device)
        image_input = torch.randint(
            0, model_args.num_embeddings, (B, T_frames, res, res), dtype=torch.int64
        ).to(device)
        image_mask = torch.randint(0, 2, (B, T_frames*res*res), dtype=torch.bool).to(device)
        text_input = torch.randn(B, T_tokens, d_model).to(device)
        text_mask = torch.randint(0, 2, (B, T_tokens), dtype=torch.bool).to(device)
        out = model(image_input, text_input, False, image_mask, text_mask)
        assert torch.all(torch.isfinite(out))
        assert torch.all(torch.isreal(out))
        assert torch.all(torch.isfinite(out.norm()))
        assert torch.all(torch.isreal(out.norm()))
        assert not torch.all(torch.isnan(out))
        assert not torch.all(torch.isinf(out))
        assert not torch.all(torch.isnan(out.norm()))
        assert not torch.all(torch.isinf(out.norm()))
    print("PASSED SQUARE VARIABLE RESOLUTION TEST")

def test_variable_batch_sizes():
    for batch_size in [1, 2, 4, 8, 16, 32]:
        model = AutoregressiveVideoTransformer(model_args).to(device)
        image_input = torch.randint(
            0, model_args.num_embeddings, (batch_size, T_frames, H, W), dtype=torch.int64
        ).to(device)
        image_mask = torch.randint(0, 2, (batch_size, T_frames*H*W), dtype=torch.bool).to(device)
        text_input = torch.randn(batch_size, T_tokens, d_model).to(device)
        text_mask = torch.randint(0, 2, (batch_size, T_tokens), dtype=torch.bool).to(device)
        out = model(image_input, text_input, False, image_mask, text_mask)
        assert torch.all(torch.isfinite(out))
        assert torch.all(torch.isreal(out))
        assert torch.all(torch.isfinite(out.norm()))
        assert torch.all(torch.isreal(out.norm()))
        assert not torch.all(torch.isnan(out))
        assert not torch.all(torch.isinf(out))
        assert not torch.all(torch.isnan(out.norm()))
        assert not torch.all(torch.isinf(out.norm()))
    print("PASSED VARIABLE BATCH SIZES TEST")

def test_variable_input_frames():
    for frames in [1, 2, 4, 8, 16, 32, 64, 128]:
        model = AutoregressiveVideoTransformer(model_args).to(device)
        image_input = torch.randint(
            0, model_args.num_embeddings, (B, frames, H, W), dtype=torch.int64
        ).to(device)
        image_mask = torch.randint(0, 2, (B, frames*H*W), dtype=torch.bool).to(device)
        text_input = torch.randn(B, T_tokens, d_model).to(device)
        text_mask = torch.randint(0, 2, (B, T_tokens), dtype=torch.bool).to(device)
        out = model(image_input, text_input, False, image_mask, text_mask)
        assert torch.all(torch.isfinite(out))
        assert torch.all(torch.isreal(out))
        assert torch.all(torch.isfinite(out.norm()))
        assert torch.all(torch.isreal(out.norm()))
        assert not torch.all(torch.isnan(out))
        assert not torch.all(torch.isinf(out))
        assert not torch.all(torch.isnan(out.norm()))
        assert not torch.all(torch.isinf(out.norm()))
    print("PASSED VARIABLE INPUT FRAMES TEST")

def test_variable_input_tokens():
    for tokens in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        model = AutoregressiveVideoTransformer(model_args).to(device)
        image_input = torch.randint(
            0, model_args.num_embeddings, (B, T_frames, H, W), dtype=torch.int64
        ).to(device)
        image_mask = torch.randint(0, 2, (B, T_frames*H*W), dtype=torch.bool).to(device)
        text_input = torch.randn(B, tokens, d_model).to(device)
        text_mask = torch.randint(0, 2, (B, tokens), dtype=torch.bool).to(device)
        out = model(image_input, text_input, False, image_mask, text_mask)
        assert torch.all(torch.isfinite(out))
        assert torch.all(torch.isreal(out))
        assert torch.all(torch.isfinite(out.norm()))
        assert torch.all(torch.isreal(out.norm()))
        assert not torch.all(torch.isnan(out))
        assert not torch.all(torch.isinf(out))
        assert not torch.all(torch.isnan(out.norm()))
        assert not torch.all(torch.isinf(out.norm()))
    print("PASSED VARIABLE INPUT TOKENS TEST")

if __name__ == "__main__":
    test_output_shape()
    test_image_padding()
    test_text_padding()
    test_combined_padding()
    test_no_padding()
    test_cache()
    test_zero_frames()
    test_zero_tokens()
    test_zero_frames_zero_tokens()
    test_exceeding_max_frames()
    test_exceeding_max_tokens()
    test_exceeding_max_frames_max_tokens()
    test_square_input_hw()
    test_non_square_input_hw()
    test_gradients()
    test_numerical_stability()
    test_square_variable_resolution()
    test_variable_batch_sizes()
    test_variable_input_frames()
    test_variable_input_tokens()
