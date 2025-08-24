import torch

def temperature_sampling(
    logits: torch.Tensor, 
    temperature: float
) -> torch.Tensor:
    """Apply temperature sampling to input logits of shape [B, V].
    
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

def top_k_sampling(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Apply top-k sampling to input logits of shape [B, V].
    
    Args:
        logits (torch.Tensor): Logits tensor of shape [B, V].
        top_k (int): Logits to sample.

    Returns:
        torch.Tensor: Output tensor of shape [B, V] with masked logits.
    """
    if top_k <= 1 or top_k > logits.size(1):
        raise ValueError(f"expected 1 < top_k <= vocab size, got {top_k}")

    # Viable range for top_k
    top_k_logits, _ = torch.topk(logits, k=top_k, dim=-1, sorted=True)
    threshold = top_k_logits[:, -1, None] # Threshold (smallest kept logit)
    masked_logits = torch.where(logits < threshold, float("-inf"), logits)
    return masked_logits

def top_p_sampling() -> torch.Tensor:
    pass

def greedy_sampling(logits: torch.Tensor) -> torch.LongTensor:
    """Apply greedy sampling to input logits of shape [B, V].
    
    Args:
        logits (torch.Tensor): Input logits of shape [B, V].

    Returns:
        torch.LongTensor: Returns indices of largest logits per row.
    """
    return torch.argmax(logits, dim=-1)

# ---------------------------------- TESTING ---------------------------------- #

B, V = 4, 20

def _test_temp():
    logits = torch.randn(B, V)
    temp = 0.7
    return logits, temperature_sampling(logits, temp)

def _test_top_k():
    logits = torch.randn(B, V)
    top_k = 2
    return logits, top_k_sampling(logits, top_k)

def _test_top_p():
    pass

def _test_greedy():
    logits = torch.randn(B, V)
    return logits, greedy_sampling(logits)

if __name__ == "__main__":
    logits, top_k_logits = _test_top_k()
    print(logits)
    print(top_k_logits)