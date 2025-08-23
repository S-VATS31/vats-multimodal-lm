from configs.setup_env import device, dtype

from typing import Optional

import torch
import torch.nn as nn
from torch.amp import autocast

from src.ffn_block import FFNBlock
from src.optimized_attention import KVCache
# from configs.autoregressive_image_gen.autoregressive_transformer.model_args.model_args_large import ModelArgs
from src.autoregressive_image_gen.autoregressive_transformer.attention.cross_attention import CrossAttentionBlock
from src.autoregressive_image_gen.autoregressive_transformer.attention.optimized_attention import CausalSelfAttentionBlock

class AutoregressiveTransformerBlock(nn.Module):
    """Transformer block stacking causal self attention block, cross attention and FFN block.
    
    Args:
        d_model (int): Dimensionality of model embeddings.
        num_heads (int): Number of attention heads for GQA.
        query_groups (int): Number of query groups for GQA.
        rope_theta (float): Exponential base of inv freq for RoPE.
        softmax_scale (float): Value to scale softmax scores by.
        use_proj_bias (bool): Whether to use projection bias.
        used_fused_proj (bool): Whether to use fused projection or not.
        use_windowed_attn (bool): Whether to use windowed attention or not.
        use_ntk_rope (bool): Whether to use NTK or classic RoPE.
        dropout (float): Dropout probability for regularization.
        eps (float): Epsilon value to maintain numerical stability in RMSNorm.
        d_ffn (int): Dimensionality of FFN.
        max_position_embeddings (Optional[int]): Max sequence length to train RoPE on.
        ntk_scale_factor: (Optional[float]): Scale factor for NTK RoPE.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        query_groups: int,
        rope_theta: float,
        softmax_scale: float,
        use_proj_bias: bool,
        use_fused_proj: bool,
        use_windowed_attn: bool,
        use_ntk_rope: bool,
        dropout: float,
        eps: float,
        d_ffn: int,
        max_position_embeddings: Optional[int] = None,
        ntk_scale_factor: Optional[float] =  None,
    ):
        super().__init__()

        self.causal_attention_block = CausalSelfAttentionBlock(
            d_model=d_model,
            num_heads=num_heads,
            query_groups=query_groups,
            rope_theta=rope_theta,
            softmax_scale=softmax_scale,
            use_proj_bias=use_proj_bias,
            use_fused_proj=use_fused_proj,
            use_windowed_attn=use_windowed_attn,
            use_ntk_rope=use_ntk_rope,
            dropout=dropout,
            eps=eps,
            max_position_embeddings=max_position_embeddings,
            ntk_scale_factor=ntk_scale_factor
        )
        self.cross_attention_block = CrossAttentionBlock(
            d_model=d_model,
            num_heads=num_heads,
            softmax_scale=softmax_scale,
            use_proj_bias=use_proj_bias,
            eps=eps,
            dropout=dropout
        )
        self.ffn_block = FFNBlock(
            d_model=d_model,
            d_ffn=d_ffn,
            dropout=dropout,
            eps=eps
        )

    def forward(
        self,
        x: torch.Tensor,
        text_embeddings: torch.Tensor,
        use_mqa: bool,
        causal: bool,
        left_window: int,
        right_window: int,
        use_cache: bool,
        padding_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        layer_idx: Optional[int] = None
    ) -> torch.Tensor:
        """Forward pass of transformer block.
        
        Args:
            x (torch.Tensor): Image tokens of shape [B, H*W, d_model].
            text_embeddings (torch.Tensor): Text tokens of shape [B, T, d_model].
            use_mqa (bool): Whether to use MQA or not.
            causal (bool): Whether to use causal masking or not.
            left_window (int): Left window for SWA.
            right_window (int): Right window for SWA.
            use_cache (bool): Whether to use KV cache or not.
            padding_mask (Optional[torch.Tensor]): Padding tensor of shape [B, T].
            kv_cache (Optional[KVCache]): KVCache module.
            layer_idx (Optional[int]): Current layer to update KV cache for.
        """
        with autocast(device_type=device.type, dtype=dtype):
            # Causal self attention block
            causal_attention_out = self.causal_attention_block(
                x,
                use_mqa=use_mqa,
                causal=causal,
                left_window=left_window,
                right_window=right_window,
                use_cache=use_cache,
                padding_mask=padding_mask,
                kv_cache=kv_cache,
                layer_idx=layer_idx
            )

            # Cross attention block
            cross_attention_out = self.cross_attention_block(
                causal_attention_out, text_embeddings=text_embeddings
            )

            # FFN block
            block_out = self.ffn_block(cross_attention_out)

            return block_out

class AutoregressiveImageTransformer(nn.Module):
    """Autorgressive image transformer complete module.
    
    Args:
        model_args (ModelArgs): Model hyperparameters.
    """
    def __init__(self, model_args): # TODO: type hint ModelArgs here
        super().__init__()

        self.model_args = model_args

    def _init_weights(self, module) -> None:
        pass

    def forward(self) -> torch.Tensor:
        pass

def test_transformer_block():
    d_model, num_heads, query_groups, rope_theta = 512, 32, 8, 10000.0
    softmax_scale = 1 / (d_model // num_heads) ** 0.5
    use_proj_bias, use_fused_proj, use_windowed_attn, use_ntk_rope = False, True, True, True
    dropout, eps, max_position_embeddings, ntk_scale_factor = 0.15, 1e-7, 2048, 0.8
    d_ffn = d_model*4
    block = AutoregressiveTransformerBlock(
        d_model, num_heads, query_groups, rope_theta, softmax_scale,
        use_proj_bias, use_fused_proj, use_windowed_attn, use_ntk_rope,
        dropout, eps, d_ffn, max_position_embeddings, ntk_scale_factor
    ).to(device)
    kv_cache = KVCache(
        max_batch_size=32,
        max_seq_len=max_position_embeddings,
        num_heads=num_heads,
        head_dim=d_model//num_heads,
        num_layers=10
    )
    B, H, W = 1, 72, 144
    T_q = H*W
    T_k = 16
    x = torch.randn(B, T_q, d_model).to(device)
    text_embeddings = torch.randn(B, T_k, d_model).to(device)
    padding_mask = torch.randint(
        0, 2, (B, T_q), dtype=torch.bool
    ).to(device)
    x_out = block(
        x,
        text_embeddings,
        use_mqa=False,
        causal=True,
        left_window=-1,
        right_window=-1,
        use_cache=True,
        padding_mask=padding_mask,
        kv_cache=kv_cache,
    )

    return x_out

if __name__ == "__main__":
    x = test_transformer_block()
    print(x.shape) # [1, 72*144, 512]
