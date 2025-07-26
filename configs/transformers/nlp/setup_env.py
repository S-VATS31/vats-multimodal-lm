# TODO: change configs/transformers/(nlp or vision/(vit_2d or vit_3d))/setup_env.py to configs/setup_env.py

import os

# Set up environment
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch

# Set up device and dtype
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

# Import Flash attention 2 function
try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
    use_flash_attn = (device.type == "cuda")
except ImportError:
    use_flash_attn = False
    flash_attn_varlen_qkvpacked_func = None
