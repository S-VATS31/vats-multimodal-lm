from configs.setup_env import device

import torch
import torch.nn.functional as F

from src.rms_norm import RMSNorm

def setup():
    d_model, eps = 512, 1e-12
    B, T = 16, 8
    rms_norm = RMSNorm(d_model, eps)
    x = torch.randn(B, T, d_model).to(device)
    x_out = rms_norm(x)
    return rms_norm, x, x_out, B, T, d_model

rms_norm, x, x_out, B, T, d_model = setup()

def test_out_shape():
    assert x.shape == x_out.shape == (B, T, d_model)
    print("PASSED OUTPUT SHAPE TEST")

if __name__ == "__main__":
    test_out_shape()
