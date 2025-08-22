from typing import Tuple, Union, Optional

import torch
import torch.nn as nn

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

def setup_projections(
    d_model: int,
    num_heads: int,
    head_dim: int,
    use_fused_proj: bool,
    use_gqa: bool,
    use_proj_bias: bool,
    query_groups: Optional[int] = None
) -> Union[
    Tuple[nn.Linear, nn.Linear], Tuple[nn.Linear, nn.Linear, nn.Linear, nn.Linear]
]:
    """Set up query, key, value, and output projections.
    
    Args:
        d_model (int): Dimensionality of model embeddings.
        num_heads (int): Number of attention heads.
        head_dim (int): Dimensionality of each attention head.
        use_fused_proj (bool): Whether to use fused qkv proj or not.
        use_gqa (bool): Whether to use GQA or not.
        use_proj_bias (bool): Whether to use projection bias or not.
        query_groups (Optional[int]): Query groups for GQA.
    """
    if use_gqa:
        assert query_groups is not None, "Must have query groups for GQA."

    # Set up output projection
    o_proj = nn.Linear(d_model, d_model, bias=use_proj_bias)

    # Set up fused projection
    if use_fused_proj:
        qkv_proj = nn.Linear(
            d_model,
            (num_heads*head_dim + 2*query_groups*head_dim) if use_gqa else 3*d_model,
            bias=use_proj_bias
        )
        return qkv_proj, o_proj
    # Set up seperate projections
    else:
        q_proj = nn.Linear(d_model, num_heads*head_dim, bias=use_proj_bias)
        k_proj = nn.Linear(
            d_model,
            (query_groups*head_dim) if use_gqa else num_heads*head_dim,
            bias=use_proj_bias
        )
        v_proj = nn.Linear(
            d_model,
            (query_groups*head_dim) if use_gqa else num_heads*head_dim,
            bias=use_proj_bias
        )
        return q_proj, k_proj, v_proj, o_proj
