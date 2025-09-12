import os

# Set up environment
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch

# Set up device and dtype
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

# Import Flash attention 2 function
try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
    use_flash_attn = (device.type == "cuda")
except ImportError:
    use_flash_attn = False
    flash_attn_varlen_qkvpacked_func = None

# Import xformers SwiGLU function
try:
    from xformers.ops import swiglu
    use_xformers_swiglu = (device.type == "cuda")
except:
    use_xformers_swiglu = False
    swiglu = None

# Ensure dtypes are float16 or bfloat16 for specific functions
gpu_dtypes = [torch.float16, torch.bfloat16]
