from typing import Tuple, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

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

def apply_qk_norm(
    query: torch.Tensor, 
    key: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply QK normalization to input QK vectors over features dim.
    
    Args:
        query (torch.Tensor): Query tensor.
        key (torch.Tensor): Key tensor.

    Returns:
        Tuple:
            - torch.Tensor: Normalized query tensor.
            - torch.Tensor: Normalized key tensor.

    Formula:
        q_norm = q / sqrt(sum(x**2))
        This is equivalent to calculating the L2 Norm for high-dim tensors.
    """
    return (
        F.normalize(query, p=2, dim=-1),
        F.normalize(key, p=2, dim=-1)
    )

# ---------------------------- TESTING ---------------------------- # 

torch.manual_seed(42)

B, T, num_heads, d_model = 2, 16, 32, 512
head_dim = d_model // num_heads
q = torch.randn(B, T, num_heads, head_dim)
k = torch.randn(B, T, num_heads, head_dim)

def _test_qk_norm():
    query, key = apply_qk_norm(q, k)
    return query, key

def _test_norm(input: torch.Tensor):
    return input / torch.sqrt(torch.sum(input ** 2, dim=-1, keepdim=True))

if __name__ == "__main__":
    q_norm, k_norm = _test_qk_norm()
    q_calculated_norm = _test_norm(q)
    print(q[0, 0, 0, 0])
    print(q_norm[0, 0, 0, 0])
    print(q_calculated_norm[0, 0, 0, 0])
