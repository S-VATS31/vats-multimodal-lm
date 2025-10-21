from configs.setup_env import device

import torch

from src.transformers.nlp.model import AutoregressiveTextTransformer
from configs.transformers.nlp.model_args.model_args_xsmall import ModelArgs
from configs.transformers.nlp.training_args import TrainingArgs

model_args, training_args = ModelArgs(), TrainingArgs()

def setup():
    model = AutoregressiveTextTransformer(model_args).to(device)
    B, T, V = 16, 8, model_args.vocab_size
    input_ids = torch.randint(
        0, V, (B, T), dtype=torch.int64
    ).to(device)
    padding_mask = torch.randint(
        0, 2, (B, T), dtype=torch.bool
    ).to(device)
    head_dim = model_args.d_model // model_args.num_heads
    return model, input_ids, padding_mask, B, T, V, head_dim

model, input_ids, padding_mask, B, T, V, head_dim = setup()

def test_output_shape():
    logits = model(input_ids, padding_mask, False)[0]
    assert logits.shape == (B, T, V)
    print("PASSED OUTPUT SHAPE TEST")

def test_no_padding():
    logits = model(input_ids, None, False)[0]
    assert logits.shape == (B, T, V)
    print("PASSED NO PADDING TEST")

def test_gradients():
    logits = model(input_ids, padding_mask, False)[0]
    loss = logits.sum()
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
    print("PASSED PADDING TEST")

def test_cache():
    logits = model(input_ids, padding_mask, True)[0]
    assert logits.shape == (B, T, V)
    assert torch.all(torch.isfinite(logits))
    assert torch.all(torch.isreal(logits))
    assert torch.all(torch.isfinite(logits.norm()))
    assert torch.all(torch.isreal(logits.norm()))
    assert not torch.all(torch.isnan(logits))
    assert not torch.all(torch.isinf(logits))
    assert not torch.all(torch.isnan(logits.norm()))
    assert not torch.all(torch.isinf(logits.norm()))
    print("PASSED CACHING TEST")

def test_zero_input_tokens():
    tokens = 0
    input_ids_temp = torch.randint(0, V, (B, tokens), dtype=torch.int64).to(device)
    padding_mask_temp = torch.randint(0, 2, (B, tokens), dtype=torch.bool).to(device)
    logits = model(input_ids_temp, padding_mask_temp, True)[0]
    assert logits.shape == (B, tokens, V)
    print("PASSED ZERO INPUT TOKENS TEST")

def test_variable_batch_sizes():
    for batch_size in [1, 2, 4, 8, 16, 32]:
        input_ids_temp = torch.randint(
            0, V, (batch_size, T), dtype=torch.int64
        ).to(device)
        padding_mask_temp = torch.randint(
            0, 2, (batch_size, T), dtype=torch.bool
        ).to(device)
        logits = model(input_ids_temp, padding_mask_temp, True)[0]
        assert logits.shape == (batch_size, T, V)
        assert torch.all(torch.isfinite(logits))
        assert torch.all(torch.isreal(logits))
        assert torch.all(torch.isfinite(logits.norm()))
        assert torch.all(torch.isreal(logits.norm()))
        assert not torch.all(torch.isnan(logits))
        assert not torch.all(torch.isinf(logits))
        assert not torch.all(torch.isnan(logits.norm()))
        assert not torch.all(torch.isinf(logits.norm()))
    print("PASSED VARIABLE BATCH SIZES TEST")

def test_variable_tokens():
    for tokens in [i for i in range(1, 51)]:
        input_ids_temp = torch.randint(
            0, V, (B, tokens), dtype=torch.int64
        ).to(device)
        padding_mask_temp = torch.randint(
            0, 2, (B, tokens), dtype=torch.bool
        ).to(device)
        logits = model(input_ids_temp, padding_mask_temp, True)[0]
        assert logits.shape == (B, tokens, V)
        assert torch.all(torch.isfinite(logits))
        assert torch.all(torch.isreal(logits))
        assert torch.all(torch.isfinite(logits.norm()))
        assert torch.all(torch.isreal(logits.norm()))
        assert not torch.all(torch.isnan(logits))
        assert not torch.all(torch.isinf(logits))
        assert not torch.all(torch.isnan(logits.norm()))
        assert not torch.all(torch.isinf(logits.norm()))
    print("PASSED VARIABLE TOKENS TEST")

def test_exceeding_mask_tokens_test():
    tokens = model_args.max_seq_len+1
    input_ids_temp = torch.randint(
        0, V, (B, tokens), dtype=torch.int64
    ).to(device)
    padding_mask_temp = torch.randint(
        0, 2, (B, tokens), dtype=torch.bool
    ).to(device)
    logits = model(input_ids_temp, padding_mask_temp, True)[0]
    assert logits.shape == (B, tokens, V)
    assert torch.all(torch.isfinite(logits))
    assert torch.all(torch.isreal(logits))
    assert torch.all(torch.isfinite(logits.norm()))
    assert torch.all(torch.isreal(logits.norm()))
    assert not torch.all(torch.isnan(logits))
    assert not torch.all(torch.isinf(logits))
    assert not torch.all(torch.isnan(logits.norm()))
    assert not torch.all(torch.isinf(logits.norm()))
    print("PASSED EXCEEDING MAX TOKENS TEST")

def test_wrong_input_ids_dtype():
    for dtype in [torch.int8, torch.int16, torch.int32]:
        V_temp = 120
        model_args.vocab_size = V_temp
        model_temp = AutoregressiveTextTransformer(model_args).to(device)
        input_ids_temp = torch.randint(
            0, V_temp, (B, T), dtype=dtype
        ).to(device)
        padding_mask_temp = torch.randint(
            0, 2, (B, T), dtype=torch.bool
        ).to(device)
        logits = model_temp(input_ids_temp, padding_mask_temp, True)[0]
        assert logits.shape == (B, T, V_temp)
        assert torch.all(torch.isfinite(logits))
        assert torch.all(torch.isreal(logits))
        assert torch.all(torch.isfinite(logits.norm()))
        assert torch.all(torch.isreal(logits.norm()))
        assert not torch.all(torch.isnan(logits))
        assert not torch.all(torch.isinf(logits))
        assert not torch.all(torch.isnan(logits.norm()))
        assert not torch.all(torch.isinf(logits.norm()))
    print("PASSED WRONG INPUT IDS DTYPE TEST")

def test_float_input_ids():
    for dtype in [torch.float16, torch.float32, torch.float64]:
        V_temp = 120
        model_args.vocab_size = V_temp
        model_temp = AutoregressiveTextTransformer(model_args).to(device)
        input_ids_temp = torch.randint(
            0, V_temp, (B, T), dtype=dtype
        ).to(device)
        padding_mask_temp = torch.randint(
            0, 2, (B, T), dtype=torch.bool
        ).to(device)
        logits = model_temp(input_ids_temp, padding_mask_temp, True)[0]
        assert logits.shape == (B, T, V_temp)
        assert torch.all(torch.isfinite(logits))
        assert torch.all(torch.isreal(logits))
        assert torch.all(torch.isfinite(logits.norm()))
        assert torch.all(torch.isreal(logits.norm()))
        assert not torch.all(torch.isnan(logits))
        assert not torch.all(torch.isinf(logits))
        assert not torch.all(torch.isnan(logits.norm()))
        assert not torch.all(torch.isinf(logits.norm()))
    print("PASSED FLOAT INPUT IDS TEST")

def test_numerical_stability():
    logits = model(input_ids, padding_mask, True)[0]
    assert logits.shape == (B, T, V)
    assert torch.all(torch.isfinite(logits))
    assert torch.all(torch.isreal(logits))
    assert torch.all(torch.isfinite(logits.norm()))
    assert torch.all(torch.isreal(logits.norm()))
    assert not torch.all(torch.isnan(logits))
    assert not torch.all(torch.isinf(logits))
    assert not torch.all(torch.isnan(logits.norm()))
    assert not torch.all(torch.isinf(logits.norm()))
    print("PASSED NUMERICAL STABILITY TEST")

if __name__ == "__main__":
    test_output_shape()
    test_no_padding()
    test_gradients()
    test_cache()
    test_zero_input_tokens()
    test_variable_batch_sizes()
    test_variable_tokens()
    test_exceeding_mask_tokens_test()
    test_wrong_input_ids_dtype()
    test_float_input_ids()
    test_numerical_stability()
