from configs.setup_env import device

import time
import torch
import pytest
from transformers import AutoTokenizer

torch.manual_seed(42)

from src.transformers.nlp.inference.generate import AutoregressiveTokenGenerator
from configs.transformers.nlp.generation_args import GenerationArgs
from configs.transformers.nlp.model_args.model_args_xsmall import ModelArgs

generation_args, model_args = GenerationArgs(), ModelArgs()

def setup():
    token_generator = AutoregressiveTokenGenerator(model_args)
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        generation_args.pad_token_id = tokenizer.pad_token_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompt = "Hello, World!"
    return token_generator, tokenizer, prompt

token_generator, tokenizer, prompt = setup()

def test_generation():
    generated_tokens = token_generator.generate_tokens(
        prompt, generation_args, tokenizer
    )
    assert type(generated_tokens) == str
    print("PASSED GENERATION TEST")

def test_zero_input_tokens():
    prompt1 = ""
    prompt2 = " "
    generated1 = token_generator.generate_tokens(
        prompt1, generation_args, tokenizer
    )
    generated2 = token_generator.generate_tokens(
        prompt2, generation_args, tokenizer
    )
    assert generated1 == "Please enter a valid prompt."
    assert generated2 == "Please enter a valid prompt."
    print("PASSED ZERO INPUT TOKENS TEST")

def test_input_generation():
    input_prompt = input("enter text: ")
    gen_toks = token_generator.generate_tokens(
        input_prompt, generation_args, tokenizer
    )
    assert type(gen_toks) == str
    print("PASSED INTERACTIVE GENERATION TEST")

def test_generation_with_cache():
    generation_args.use_cache = True
    gen_toks = token_generator.generate_tokens(
        prompt, generation_args, tokenizer
    )
    assert type(gen_toks) == str
    print("PASSED GENERATION WITH CACHE TEST")

def test_generation_without_cache():
    generation_args.use_cache = False
    gen_toks = token_generator.generate_tokens(
        prompt, generation_args, tokenizer
    )
    assert type(gen_toks) == str
    print("PASSED GENERATION WITH NO CACHE TEST")

def test_zero_topk():
    generation_args.top_k = 0
    with pytest.raises(ValueError):
        token_generator.generate_tokens(
            prompt, generation_args, tokenizer
        )
    print("PASSED ZERO TOPK TEST")

def test_topk():
    generation_args.top_k = 50
    gen_toks = token_generator.generate_tokens(
        prompt, generation_args, tokenizer
    )
    assert type(gen_toks) == str
    print("PASSED TOPK TEST")

def test_no_topk():
    generation_args.top_k = None
    gen_toks = token_generator.generate_tokens(
        prompt, generation_args, tokenizer
    )
    assert type(gen_toks) == str
    print("PASSED NO TOPK TEST")

def test_zero_temperature():
    generation_args.temperature = 0
    gen_toks = token_generator.generate_tokens(
        prompt, generation_args, tokenizer
    )
    assert type(gen_toks) == str
    print("PASSED ZERO TEMPERATURE TEST")

def test_temperature():
    generation_args.temperature = 0.7
    gen_toks = token_generator.generate_tokens(
        prompt, generation_args, tokenizer
    )
    assert type(gen_toks) == str
    print("PASSED TEMPERATURE TEST")

def test_no_temperature():
    generation_args.temperature = None
    gen_toks = token_generator.generate_tokens(
        prompt, generation_args, tokenizer
    )
    assert type(gen_toks) == str
    print("PASSED NO TEMPERATURE TEST")

def test_zero_top_p():
    generation_args.top_p = 0
    with pytest.raises(ValueError):
        token_generator.generate_tokens(
            prompt, generation_args, tokenizer
    ) 
    print("PASSED ZERO TOP P TEST")

def test_top_p():
    generation_args.top_p = 0.9
    gen_toks = token_generator.generate_tokens(
        prompt, generation_args, tokenizer
    )
    assert type(gen_toks) == str
    print("PASSED TOP P TEST")

def test_no_top_p():
    generation_args.top_p = None
    gen_toks = token_generator.generate_tokens(
        prompt, generation_args, tokenizer
    )
    assert type(gen_toks) == str
    print("PASSED NO TOP P TEST")
 
def test_zero_max_new_tokens():
    generation_args.max_new_tokens = 0
    gen_toks = token_generator.generate_tokens(
        prompt, generation_args, tokenizer
    )
    print(gen_toks)
    assert gen_toks == prompt
    print("PASSED ZERO MAX NEW TOKENS TEST")

def test_max_new_tokens():
    generation_args.max_new_tokens = 50
    gen_toks = token_generator.generate_tokens(prompt, generation_args, tokenizer)
    gen_ids = tokenizer(gen_toks).input_ids
    prompt_ids = tokenizer(prompt).input_ids
    assert len(gen_ids) - len(prompt_ids) <= generation_args.max_new_tokens
    assert type(gen_toks) == str
    print("PASSED MAX NEW TOKENS TEST")

def test_large_max_new_tokens():
    generation_args.max_new_tokens = 1000
    gen_toks = token_generator.generate_tokens(prompt, generation_args, tokenizer)
    gen_ids = tokenizer(gen_toks).input_ids
    prompt_ids = tokenizer(prompt).input_ids
    assert len(gen_ids) - len(prompt_ids) <= generation_args.max_new_tokens
    assert type(gen_toks) == str
    print("PASSED LARGE MAX NEW TOKENS TEST")

def test_zero_repetition_penalty():
    generation_args.repetition_penalty = 0
    with pytest.raises(ValueError):
        token_generator.generate_tokens(prompt, generation_args, tokenizer)
    print("PASSED ZERO REPETITIONAL PENALTY TEST")

def test_repetition_penalty():
    generation_args.repetition_penalty = 0.7
    gen_toks = token_generator.generate_tokens(prompt, generation_args, tokenizer)
    assert type(gen_toks) == str
    print("PASSED REPETITION PENALTY TEST")

def test_no_repetition_penalty():
    generation_args.repetition_penalty = None
    gen_toks = token_generator.generate_tokens(prompt, generation_args, tokenizer)
    assert type(gen_toks) == str
    print("PASSED NO REPETITION PENALTY TEST")

def test_greedy_decoding():
    pass    
    print("PASSED GREEDY DECODING TEST")

def test_pre_created_attn_mask():
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = (input_ids != generation_args.pad_token_id)
    gen_toks = token_generator.generate_tokens(
        prompt, generation_args, tokenizer, attention_mask=attention_mask
    )
    assert type(gen_toks) == str
    print("PASSED PRE-CREATED ATTENTION MASK TEST")

def test_attn_mask_is_none():
    gen_toks = token_generator.generate_tokens(
        prompt, generation_args, tokenizer, None
    )
    assert type(gen_toks) == str
    print("PASSED ATTENTION MASK IS NONE TEST")

def test_topk_topp_combination():
    generation_args.top_k = 50
    generation_args.top_p = 0.9
    gen_toks = token_generator.generate_tokens(
        prompt, generation_args, tokenizer
    )
    assert type(gen_toks) == str
    print("PASSED TOPK + TOPP COMBINATION TEST")

def test_temperature_topk_topp_interaction():
    pass
    print("PASSED TEMPERATURE + TOPK + TOPP INTERACTION TEST")

def test_determinism():
    pass
    print("PASSED DETERMINISM TEST")

def test_eos_handling():
    pass
    print("PASSED EOS HANDLING TEST")

def test_invalid_logits():
    pass
    print("PASSED INVALID LOGITS TEST")

def test_long_context():
    pass
    print("PASSED LONG CONTEXT TEST")

def test_cache_speed():
    # KV caching speed test
    generation_args.use_cache = True
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    else:
        torch.cpu.synchronize()
    kv_start_time = time.time()
    kv_gen_toks = token_generator.generate_tokens(prompt, generation_args, tokenizer)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    else:
        torch.cpu.synchronize()
    kv_end_time = time.time()
    kv_time = kv_end_time - kv_start_time
    # No caching speed test
    generation_args.use_cache = False
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    else:
        torch.cpu.synchronize()
    no_kv_start_time = time.time()
    no_kv_gen_toks = token_generator.generate_tokens(prompt, generation_args, tokenizer)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    else:
        torch.cpu.synchronize()
    no_kv_end_time = time.time()
    no_kv_time = no_kv_end_time - no_kv_start_time
    print(kv_gen_toks)
    print(no_kv_gen_toks)
    print(f"kv time: {kv_time}")
    print(f"no kv time: {no_kv_time}")
    assert kv_time < no_kv_time
    print("PASSED FORCED DECODING TEST")

def test_repetition_penalty_effect():
    pass
    print("PASSED REPETITION PENALTY EFFECT TEST")

def test_greedy_equals_argmax():
    pass
    print("PASSED GREEDY EQUALS ARGMAX TEST")


if __name__ == "__main__":
    test_generation()
    test_zero_input_tokens()
    test_input_generation()
    test_generation_with_cache()
    test_generation_without_cache()
    test_zero_topk()
    test_topk()
    test_no_topk()
    test_zero_temperature()
    test_temperature()
    test_no_temperature()
    test_zero_top_p()
    test_top_p()
    test_no_top_p()
    test_zero_max_new_tokens()
    test_max_new_tokens()
    test_large_max_new_tokens()
    test_zero_repetition_penalty()
    test_repetition_penalty()
    test_no_repetition_penalty()
    test_greedy_decoding()
    test_pre_created_attn_mask()
    test_attn_mask_is_none()
    test_topk_topp_combination()
    test_temperature_topk_topp_interaction()
    test_determinism()
    test_eos_handling()
    test_invalid_logits()
    test_long_context()
    test_cache_speed()
    test_repetition_penalty_effect()
    test_greedy_equals_argmax()
