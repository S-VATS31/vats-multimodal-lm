# NOTE: since SwiGLU expects same input, we can create a single module
# and import it to multiple transformers.

from configs.setup_env import (
    device,
    dtype,
    swiglu,
    use_xformers_swiglu
)

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLUActivation(nn.Module):
    """SwiGLU expert layer.

    Args:
        d_model (int): Dimensionality of the model's embeddings.
        d_ffn (int): Dimensionality of the feed forward network.
        dropout (float): Probability of model components being dropped out.
    """
    def __init__(self, d_model: int, d_ffn: int, dropout: float):
        super().__init__()

        self.weight1 = nn.Linear(d_model, d_ffn, bias=False)
        self.weight2 = nn.Linear(d_model, d_ffn, bias=False)
        self.weight3 = nn.Linear(d_ffn, d_model, bias=False)
        self.dropout = nn.Dropout(p=dropout)
    
    def _optimized_swiglu(self, x: torch.Tensor) -> torch.Tensor:
        """Optimized SwiGLU activation using xformers.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, T, d_model].
            
        Returns:
            torch.Tensor: Output tensor of same shape.

        Notes:
            - We do self.matrix.weight.T as xformers func needs the matrix to be broadcastable
            with x raw (x @ matrix) (basically, without transpose we cannot apply matrix multiplication
            due to a shape error).
            - When we call SwiGLU, all the `None`s being passed, are `Optional[torch.Tensor]`s
            representing bias.
            - Requires a contiguous tensor, so we use .contiguous() on the input tensor.

        Requirements:
            xformers swiglu() import must be succesful.
            `device` must be cuda.
            x must live on `device` of cuda.
            x must have dtype of float16 or bfloat16.

        NOTE: OPTIMIZED SWIGLU HAS NOT BEEN TESTED DUE TO HARDWARE REQUIREMENTS.
        """
        # xformers SwiGLU
        if (
            use_xformers_swiglu
            and device.type == "cuda"
            and x.is_cuda
            and x.dtype in [torch.float16, torch.bfloat16]
        ):
            return self.dropout(swiglu(
                x.contiguous(),
                self.weight1.weight.T, None,
                self.weight2.weight.T, None,
                self.weight3.weight.T, None
            ))
        # PyTorch SwiGLU fallback
        else:
            warnings.warn("xformers SwiGLU not available, falling back to PyTorch SwiGLU.")
            return self._pytorch_swiglu(x)
    
    def _pytorch_swiglu(self, x: torch.Tensor) -> torch.Tensor:
        """Optimized SwiGLU activation using xformers.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, T, d_model].
            
        Returns:
            torch.Tensor: Output tensor of same shape.
        """
        return self.dropout(self.weight3(F.silu(self.weight1(x)) * self.weight2(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SwiGLU Expert layer.

        Formula:
            SwiGLU(x) = Dropout(Swish((W1 @ x) * (W3 @ x)) @ W2)

        Args:
            x (torch.Tensor): Input tensor of shape [B, T, d_model].

        Returns:
            torch.Tensor: Output tensor with the same shape.
        """
        with torch.amp.autocast(device_type=device.type, dtype=dtype):
            return self._optimized_swiglu(x)
        