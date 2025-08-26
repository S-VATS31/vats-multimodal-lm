import time

import torch
torch.manual_seed(0)  
import torch.nn as nn
import torch.nn.functional as F

class Norm(nn.Module):
    def __init__(self, d_model: int, eps: float):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor, use_func_norm: bool) -> torch.Tensor:
        if use_func_norm:
            return self.weight * (
                F.normalize(x, p=2, dim=-1, eps=self.eps)
            )
        else:
            return self.weight * (
                x / torch.sqrt(torch.sum(x ** 2, dim=-1, keepdim=True) + self.eps)
            )

def benchmark(n_iters: int):
    d_model, eps = 1024, 1e-12
    norm = Norm(d_model, eps)
    B, T = 32, 32
    x = torch.randn(B, T, d_model)

    # warmup
    for _ in range(n_iters):
        _ = norm(x, True)
    for _ in range(n_iters):
        _ = norm(x, False)

    # torch norm
    total_torch_time = 0.0
    for _ in range(n_iters):
        torch_start_time = time.time()
        _ = norm(x, True)
        torch_end_time = time.time()
        total_torch_time += (torch_end_time - torch_start_time)

    avg_torch_time = total_torch_time / n_iters

    # custom norm
    total_custom_time = 0.0
    for _ in range(n_iters):
        custom_start_time = time.time()
        _ = norm(x, False)
        custom_end_time = time.time()
        total_custom_time += (custom_end_time - custom_start_time)

    avg_custom_time = total_custom_time / n_iters

    # log
    print(f"torch.nn.functional.normalize time: {avg_torch_time*1000:.4f} ms")
    print(f"torch.sqrt(torch.sum(x**2)) time:   {avg_custom_time*1000:.4f} ms")

benchmark(50_000)