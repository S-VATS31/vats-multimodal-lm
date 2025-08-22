from typing import Tuple

import torch

def extend_kv_heads(
    input: torch.Tensor,
    repeats: int,
    dim: int,
    use_mqa: bool
) -> torch.Tensor:
    """Extend kv heads to be equal to query heads.

    Args:
        input (torch.Tensor): Input tensor.
        repeats (int): Number of repeats.
        dim (int): Dimension to be repeated.
        use_mqa (bool): Whether to use MQA or not.
            Notes: input.size(dim) == 1. Returns input tensor.

    Returns:
        torch.Tensor: Output tensor with specific dimension repeated.
    """
    if input.size(dim) == 1 and use_mqa:
        return input
    return input.repeat_interleave(repeats=repeats, dim=dim)
