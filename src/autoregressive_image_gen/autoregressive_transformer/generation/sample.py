import torch

def temperature_sampling(
    logits: torch.Tensor, 
    temperature: float
) -> torch.Tensor:
    """Apply temperature sampling to input logits of shpape [B, V].
    
    Args:
        logits (torch.Tensor): Input logits to be scaled.
        temperature (float): Logits scaling factor.

    Returns:
        torch.Tensor: Scaled output tensor of shape [B, V].
    """
    if temperature <= 0:
        raise ValueError(f"expected temperature > 0, got {temperature}")
    else:
        return logits / temperature 

def top_k_sampling() -> torch.Tensor:
    pass

def top_p_sampling() -> torch.Tensor:
    pass

def repetition_penalty() -> torch.Tensor:
    pass

def greedy_sampling(logits: torch.Tensor) -> torch.LongTensor:
    """Apply greedy sampling to input logits of shape [B, V].
    
    Args:
        logits (torch.Tensor): Input logits of shape [B, V].

    Returns:
        torch.LongTensor: Returns indices of largest logits per row.
    """
    return torch.argmax(logits, dim=-1)

# -------------------------- TESTING ----------------------------------

B, V = 4, 65536

def _test_temp():
    logits = torch.randn(B, V)
    temp = 0.7
    return logits, temperature_sampling(logits=logits, temperature=temp)

def _test_top_k():
    pass

def _test_top_p():
    pass

def _test_repetition_penalty():
    pass

def _test_greedy():
    logits = torch.randn(B, V)
    return logits, greedy_sampling(logits)

if __name__ == "__main__":
    torch.manual_seed(0)
    logits, greedy_toks = _test_greedy()
    print(greedy_toks)
    print(greedy_toks.dtype)
    print(logits[:, 36885])
    print(logits[:, 15385])
    print(logits[:, 59837])
    print(logits[:, 57909])
