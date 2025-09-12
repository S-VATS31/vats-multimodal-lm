from configs.setup_env import device, dtype

import torch

from src.transformers.nlp.model import AutoregressiveTextTransformer
from configs.transformers.nlp.model_args.model_args_xsmall import ModelArgs
from configs.transformers.nlp.training_args import TrainingArgs

model_args, training_args = ModelArgs(), TrainingArgs()

def setup():
    model = AutoregressiveTextTransformer(model_args).to(device)
    B, T = 16, 8
    input_ids = torch.randint(
        0, model_args.vocab_size, (B, T), dtype=torch.int64
    ).to(device)
    padding_mask = torch.randint(
        0, 2, (B, T), dtype=torch.bool
    ).to(device)
    logits, cache_outs, aux_loss = model(
        input_ids, padding_mask=padding_mask, use_cache=False
    )
    return (
        model, 
        input_ids, 
        padding_mask, 
        logits, 
        B, 
        T,
        cache_outs,
        aux_loss
    )

model, input_ids, padding_mask, logits, B, T, cache_outs, aux_loss = setup()

def test_logits_shape():
    assert logits.shape == (B, T, model_args.vocab_size)
    print("PASSED LOGITS SHAPE TEST")

def test_aux_loss_dtype():
    assert aux_loss.dtype == torch.float32
    print("PASSED AUX LOSS DTYPE TEST")

def test_padding_shape():
    assert padding_mask.shape == input_ids.shape
    print("PASSED PADDING/INPUT_IDS TEST")

def test_with_cache():
    out_with_cache = model(input_ids, padding_mask, use_cache=True)[0]
    assert out_with_cache.shape == logits.shape == (B, T, model_args.vocab_size)
    print("PASSED CACHING TEST")

if __name__ == "__main__":
    test_logits_shape()
    test_padding_shape()
    test_with_cache()
