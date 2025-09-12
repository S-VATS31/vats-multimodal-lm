from configs.setup_env import device, dtype

import torch

from src.optimized_attention import Attention, KVCache
from configs.transformers.nlp.training_args import TrainingArgs
from configs.transformers.nlp.model_args.model_args_xsmall import ModelArgs

model_args, training_args = ModelArgs(), TrainingArgs()

torch.manual_seed(42)

def setup():
    attn = Attention(
        model_args.d_model,
        model_args.num_heads,
        model_args.query_groups,
        model_args.rope_base,
        model_args.softmax_scale,
        model_args.use_proj_bias,
        model_args.use_qkv_proj,
    ).to(device)
    batch_size, seq_len = 8, 16
    x = torch.randn(
        batch_size, seq_len, model_args.d_model, requires_grad=True
    ).to(device)
    padding_mask = torch.randint(
        0, 2, (batch_size, seq_len), dtype=torch.bool
    ).to(device)
    x_out: torch.Tensor = attn(
        x,
        left_window=model_args.left_window,
        right_window=model_args.right_window,
        causal=model_args.use_causal,
        padding_mask=padding_mask,
        kv_cache=None,
        layer_idx=None,
        use_cache=None,
        use_mqa=False,
        use_qk_norm=True
    )[0]
    layer_idx = model_args.num_layers - 1
    kv_cache = KVCache(
        max_batch_size=model_args.max_batch_size,
        max_seq_len=model_args.max_seq_len,
        num_heads=model_args.num_experts,
        head_dim=model_args.d_model//model_args.num_heads,
        num_layers=model_args.num_layers
    )
    return (
        attn, 
        batch_size, 
        seq_len, 
        x, 
        padding_mask, 
        x_out, 
        model_args.d_model, 
        kv_cache, 
        layer_idx
    )

attn, B, T, x, padding_mask, x_out, d_model, kv_cache, layer_idx  = setup()

print("--------------------- NLP ATTENTION TESTING ---------------------")

def test_shape():
    assert x.shape == x_out.shape == (B, T, d_model)
    print("PASSED SHAPE TEST")

def test_no_padding():
    out_no_pad = attn(x, -1, -1, True, padding_mask=None)[0]
    assert out_no_pad.shape == x.shape == x_out.shape
    print("PASSED NO PADDING TEST")

def test_cache():
    kv_cache.initialize(batch_size=B)
    x_out1, cache_out1 = attn(
        x,
        left_window=-1,
        right_window=-1,
        causal=True,
        padding_mask=None,
        kv_cache=kv_cache,
        layer_idx=0,
        use_cache=True
    )
    assert x_out1.shape == x.shape, f"Output shape mismatch: {x_out1.shape} != {x.shape}"
    assert cache_out1['k'].shape[0] == B, "KV cache batch dimension mismatch"
    assert cache_out1['v'].shape[0] == B, "KV cache batch dimension mismatch"
    x_next = torch.randn(B, 4, d_model).to(device)
    _, _ = attn(
        x_next,
        left_window=-1,
        right_window=-1,
        causal=True,
        padding_mask=None,
        kv_cache=kv_cache,
        layer_idx=0,
        use_cache=True
    )
    k_total, v_total = kv_cache.get(layer_idx=0, seq_len=kv_cache.current_seq_len)
    assert k_total.shape[1] == kv_cache.current_seq_len, "KV cache sequence length mismatch"
    assert v_total.shape[1] == kv_cache.current_seq_len, "KV cache sequence length mismatch"
    print("PASSED KV CACHING TEST")

def test_gradients():
    loss = x_out.sum()
    loss.backward()
    for _, param in attn.named_parameters():
        assert param.grad is not None
    print("PASSED GRADIENTS TEST")

def test_causal_masking():
    full_out, _ = attn(
        x,
        left_window=-1,
        right_window=0,
        causal=True,
        padding_mask=None,
        use_cache=False
    )
    for t in range(1, T):
        truncated_out, _ = attn(
            x[:, :t],
            left_window=-1,
            right_window=0,
            causal=True,
            padding_mask=None,
            use_cache=False
        )
        diff = torch.abs(truncated_out[:, -1] - full_out[:, t-1])
        assert torch.allclose(diff, torch.zeros_like(diff), atol=1e-3)
    print("PASSED CAUSAL MASKING TEST")

def test_no_causal_masking():
    x_no_causal = attn(x, -1, -1, causal=False)[0]
    assert x_no_causal.shape == x_out.shape == x.shape
    print("PASSED NO CAUSAL MASKING TEST")

def test_windowed_attn():
    x_out_windowed = attn(x, model_args.left_window, model_args.right_window)[0]
    assert x_out_windowed.shape == x.shape == x_out.shape
    print("PASSED WINDOWED ATTENTION TEST")

def test_no_windowed_attn():
    x_no_windowed = attn(x, -1, -1)[0]
    assert x_no_windowed.shape == x.shape == x_out.shape
    print("PASSED NO WINDOWED ATTENTION TEST")

if __name__ == "__main__":
    test_shape()
    test_no_padding()
    test_cache()
    test_gradients()
    test_causal_masking()
    test_no_causal_masking()
    test_windowed_attn()
    test_no_windowed_attn()
