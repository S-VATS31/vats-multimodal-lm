from configs.setup_env import device

from pathlib import Path

import torch
from transformers import AutoTokenizer

from src.transformers.nlp.inference.generate import AutoregressiveTokenGenerator
from configs.transformers.nlp.model_args.model_args_medium import ModelArgs

from configs.transformers.nlp.generation_args import GenerationArgs

def load_best_model_for_generation(
    checkpoints_dir: Path = Path("nlp_checkpoints"),
    model_filename: str = "best_model.pt"
):
    """
    Load the best model checkpoint and create a token generator ready for inference.
    
    Args:
        checkpoints_dir (Path): Directory containing checkpoints
        model_filename (str): Name of the best model file
    
    Returns:
        AutoregressiveTokenGenerator: Generator with loaded best model
        dict: Checkpoint information (tokens_seen, loss, etc.)
    """
    
    # Load checkpoint data to get model_args
    checkpoint_path = checkpoints_dir / model_filename
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Reconstruct model_args from saved dict
    model_args_dict = checkpoint['model_args']
    model_args = ModelArgs(**model_args_dict)
    
    # Create token generator with the model args
    generator = AutoregressiveTokenGenerator(model_args)
    
    # Get the state dict from checkpoint
    state_dict = checkpoint['model_state_dict']
    
    # Handle RoPE caches - remove them from checkpoint and let model initialize fresh ones
    keys_to_remove = []
    for key in state_dict.keys():
        if "rope.cos_cache" in key or "rope.sin_cache" in key:
            keys_to_remove.append(key)
    
    # Remove RoPE cache keys from state dict
    for key in keys_to_remove:
        del state_dict[key]
    
    # Load the state dict with strict=False to handle missing RoPE caches gracefully
    missing_keys, unexpected_keys = generator.model.load_state_dict(state_dict, strict=False)
    
    # Check if only RoPE caches are missing (which is expected and OK)
    rope_cache_keys = [k for k in missing_keys if "rope.cos_cache" in k or "rope.sin_cache" in k]
    non_rope_missing_keys = [k for k in missing_keys if k not in rope_cache_keys]
    
    if non_rope_missing_keys:
        print(f"Warning: Missing non-RoPE keys: {non_rope_missing_keys}")
    
    if unexpected_keys:
        print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")
    
    # Set model to evaluation mode
    generator.model.eval()
    
    # Extract checkpoint info
    checkpoint_info = {
        'tokens_seen': checkpoint['tokens_seen'],
        'loss': checkpoint['loss'],
        'training_args': checkpoint['training_args'],
        'model_args': checkpoint['model_args']
    }
    
    print(f"Successfully loaded best model from {checkpoint_path}")
    print(f"Model was trained for {checkpoint_info['tokens_seen']} tokens")
    print(f"Best validation loss: {checkpoint_info['loss']:.4f}")
    
    return generator, checkpoint_info

# Example usage:
if __name__ == "__main__":
    # Load the best model
    generator, checkpoint_info = load_best_model_for_generation()
    
    # Set up tokenizer
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set up generation args
    generation_args = GenerationArgs()
    generation_args.pad_token_id = tokenizer.pad_token_id
    generation_args.eos_token_id = tokenizer.eos_token_id
    
    # Generate text
    prompt = input("Enter prompt: ")
    generated_text = generator.generate_tokens(
        prompt=prompt,
        generation_args=generation_args,
        tokenizer=tokenizer
    )
    
    print(f"Generated text: {generated_text}")
