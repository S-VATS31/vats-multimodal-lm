from configs.setup_env import device, dtype

from typing import Optional

import torch
import torch.nn as nn
from torch.amp import autocast

from src.rms_norm import RMSNorm
from src.ffn_block import FFNBlock
from src.optimized_attention import KVCache
from src.autoregressive_image_gen.autoregressive_transformer.attention.cross_attention import CrossAttentionBlock
from src.autoregressive_video_gen.autoregressive_transformer.attention.optimized_attention import CausalFactorizedAttentionBlock


class AutoregressiveVideoTransformerBlock(nn.Module):
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
        eps: float,
        dropout: float,
        d_ffn: int,
        ntk_scale_factor: Optional[float] = None
    ):
        super().__init__()

        self.factorized_attention_block = CausalFactorizedAttentionBlock(
            d_model=d_model,
            num_heads=num_heads,
            query_groups=query_groups,
            rope_theta=rope_theta,
            softmax_scale=softmax_scale,
            use_proj_bias=use_proj_bias,
            use_fused_proj=use_fused_proj,
            use_windowed_attn=use_windowed_attn,
            use_ntk_rope=use_ntk_rope,
            eps=eps,
            dropout=dropout,
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
        use_qk_norm: bool,
        use_causal: bool,
        left_window: int,
        right_window: int,
        use_cache: bool,
        spatio_temporal_padding_mask: Optional[torch.Tensor] = None,
        text_padding_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        layer_idx: Optional[int] = None
    ) -> torch.Tensor:
        """Forward of transformer block layer.
        
        Args:
            x (torch.Tensor): Input spatio-temporal tensor of shape [B, T_frames, H*W, d_model].
            text_embeddings (torch.Tensor): Input text tokens of shape [B, T_tokens, d_model].
            use_mqa (bool): Whether to use MQA or not.
            use_qk_norm (bool): Whether to use QK normalization or not.
            use_causal (bool): Whether to use causal masking or not.
            left_window (int): Left window for SWA.
            right_window (int): Right window for SWA.
            use_cache (bool): Whether to use KV caching or not.
            spatio_temporal_padding_mask (Optional[torch.Tensor]): Padding tensor of shape [B, T_frames*H*W].
            text_padding_mask (Optional[torch.Tensor]): Padding tensor of shape [B, T_tokens].
            kv_cache (Optional[KVCache]): KV caching module.
            layer_idx (Optional[int]): Layer to update KVs for.

        Returns:
            torch.Tensor: Output tensor of shape [B, T_frames, H*W, d_model].
        """
        with autocast(device_type=device.type, dtype=dtype):
            # Apply factorized attention
            factorized_attn_out = self.factorized_attention_block(
                x,
                use_mqa=use_mqa,
                use_qk_norm=use_qk_norm,
                use_causal=use_causal,
                left_window=left_window,
                right_window=right_window,
                use_cache=use_cache,
                padding_mask=spatio_temporal_padding_mask,
                kv_cache=kv_cache,
                layer_idx=layer_idx
            )
            
            # Apply cross attention
            cross_attn_out = self.cross_attention_block(
                factorized_attn_out,
                text_embeddings=text_embeddings,
                padding_mask=text_padding_mask
            )

            # Apply FFN
            block_out = self.ffn_block(cross_attn_out)

            return block_out


class AutoregressiveVideoTransformer(nn.Module):
    def __init__(self, model_args):
        super().__init__()

        pass

    def _init_weights(self, module) -> None:
        pass

    def forward(self) -> torch.Tensor:
        pass

def test_transformer_block():
    d_model, num_heads, query_groups, rope_theta = 512, 32, 8, 10000.0
    softmax_scale, d_ffn = (d_model // num_heads) ** 0.5, 4*d_model
    use_proj_bias, use_fused_proj, use_windowed_attn = False, True, True
    use_ntk_rope, eps, dropout, ntk_scale_factor = True, 1e-12, 0.15, 0.7
    transformer_block = AutoregressiveVideoTransformerBlock(
        d_model, num_heads, query_groups,
        rope_theta, softmax_scale, use_proj_bias,
        use_fused_proj, use_windowed_attn, use_ntk_rope,
        eps, dropout, d_ffn, ntk_scale_factor
    ).to(device)
    B, T_frames, H, W = 1, 16, 32, 32
    T_tokens = 12
    x = torch.randn(B, T_frames, H*W, d_model).to(device)
    text_embeddings = torch.randn(B, T_tokens, d_model).to(device)
    factorized_padding_mask = torch.randint(
        0, 2, (B, T_frames*H*W), dtype=torch.bool
    ).to(device)
    text_padding_mask = torch.randint(
        0, 2, (B, T_tokens), dtype=torch.bool
    ).to(device)
    x_out = transformer_block(
        x,
        text_embeddings=text_embeddings,
        use_mqa=False,
        use_qk_norm=True,
        use_causal=True,
        left_window=-1,
        right_window=-1,
        use_cache=False,
        spatio_temporal_padding_mask=factorized_padding_mask,
        text_padding_mask=text_padding_mask,
        kv_cache=None,
        layer_idx=None
    )
    return x_out

if __name__ == "__main__":
    x_out = test_transformer_block()
    print(x_out.shape) # [1, 16, 32*32, 512]
