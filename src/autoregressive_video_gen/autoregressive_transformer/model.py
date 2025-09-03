from configs.setup_env import device, dtype

from typing import Optional

import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.checkpoint import checkpoint

from src.rms_norm import RMSNorm
from src.ffn_block import FFNBlock
from src.optimized_attention import KVCache
from configs.autoregressive_video_gen.autoregressive_transformer.model_args.model_args_large import ModelArgs
from src.autoregressive_video_gen.autoregressive_transformer.attention.cross_attention import FactorizedCrossAttentionBlock
from src.autoregressive_video_gen.autoregressive_transformer.attention.optimized_attention import CausalFactorizedAttentionBlock

class AutoregressiveVideoTransformerBlock(nn.Module):
    """Transformer block combining causal attn, cross attn, and ffn block.
       
    Args:
        d_model (int): Dimensionality of model embeddings.
        num_heads (int): Number of attention heads for queries.
        query_groups (int): Number of query groups for GQA.
        rope_theta (float): Exponential base of inv freq for RoPE.
        softmax_scale (float): Softmax scale for attention scores.
        use_proj_bias (bool): Whether to use projection bias or not.
        use_fused_proj (bool): Whether to use QKV projection or not.
        use_windowed_attn (bool): Whether to use windowed attention or not.
        use_ntk_rope (bool): Whether to use NTK RoPE or classic.
        eps (float): Epsilon value to maintain numerical stability in RMSNorm.
        dropout (float): Dropout probability for regularization.
        d_ffn (int): Dimensionality of the FFN.
        ntk_scale_factor (Optional[float]): Alpha hyperparameter for NTK RoPE.
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
        self.cross_attention_block = FactorizedCrossAttentionBlock(
            d_model=d_model,
            num_heads=num_heads,
            query_groups=query_groups,
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
                use_mqa=use_mqa,
                use_qk_norm=use_qk_norm,
                padding_mask=text_padding_mask
            )

            # Apply FFN
            block_out = self.ffn_block(cross_attn_out)

            return block_out


class AutoregressiveVideoTransformer(nn.Module):
    def __init__(self, model_args: ModelArgs):
        super().__init__()

        self.model_args = model_args

        # Set up dropout
        self.dropout = nn.Dropout(p=model_args.dropout).to(device)

        # Set up layers
        self.layers = nn.ModuleList([
            AutoregressiveVideoTransformerBlock(
                d_model=model_args.d_model,
                num_heads=model_args.num_heads,
                query_groups=model_args.query_groups,
                rope_theta=model_args.rope_theta,
                softmax_scale=model_args.softmax_scale,
                use_proj_bias=model_args.use_proj_bias,
                use_fused_proj=model_args.use_qkv_proj,
                use_windowed_attn=model_args.use_windowed_attn,
                use_ntk_rope=model_args.use_ntk_rope,
                eps=model_args.rms_norm_eps,
                dropout=model_args.dropout,
                d_ffn=model_args.d_ffn,
                ntk_scale_factor=model_args.ntk_scale_factor
            ).to(device) for _ in range(model_args.num_layers)
        ])

        # Set up RMSNorm
        self.rms_norm = RMSNorm(
            d_model=model_args.d_model, eps=model_args.rms_norm_eps
        ).to(device)

        # Set up KV Cache
        self.kv_cache = KVCache(
            max_batch_size=model_args.max_batch_size,
            max_seq_len=model_args.max_frames, # ONLY CACHING OVER TEMPORAL DIM
            num_heads=model_args.num_heads,
            head_dim=model_args.d_model // model_args.num_heads,
            num_layers=model_args.num_layers
        )

        # Initialize weights
        # self.apply(self._init_weights)

    def _init_weights(self, module) -> None:
        """Initialize weights based on given module.
        
        Args:
            module: Module to be initialized.
        """
        pass

    def forward(
        self,
        x: torch.Tensor,
        text_embeddings: torch.Tensor,
        use_cache: bool,
        spatio_temoral_padding_mask: Optional[torch.Tensor] = None,
        text_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of entire transformer.
        
        Args:
            x (torch.Tensor): Input image tokens of shape [B, T_frames, H*W, d_model].
            text_embeddings (torch.Tensor): Input text tokens of shape [B, T_tokens, d_model].
            use_cache (bool): Whether to use KV caching or not.
            spatio_temporal_padding_mask (Optional[torch.Tensor]): Padding tensor of shape [B, T_frames*H*W].
            text_padding_mask (Optional[torch.Tensor]): Padding tensor of shape [B, T_tokens].

        Returns:
            torch.Tensor: Output tensor of shape [B, T_frames, H*W, d_model].
        """
        assert x.dim() == 4, f"expected 4 dims, got {x.dim()} dims"
        assert text_embeddings.dim() == 3, f"expected 3 dims, got {text_embeddings.dim()} dims"
        assert (
            spatio_temoral_padding_mask.dim() == 2
        ), f"expected 2 dims, got {spatio_temoral_padding_mask.dim()} dims"
        assert (
            text_padding_mask.dim() == 2
        ), f"expected 2 dims, got {text_padding_mask.dim()} dims"
        assert (
            spatio_temoral_padding_mask.shape == (x.size(0), x.size(1)*x.size(2))
        ), f"expected {(x.size(0), x.size(1)*x.size(2))}, got {spatio_temoral_padding_mask.shape}"
        assert (
            text_padding_mask.shape == text_embeddings.shape[:-1]
        ), f"expected {text_embeddings.shape[:-1]}, got {text_padding_mask.shape}"

        x = self.dropout(x) # [B, T_frames, H*W, d_model]

        # Loop through layers
        for layer_idx, layer in enumerate(self.layers):
            if self.model_args.use_checkpointing:
                x = checkpoint(
                    layer,
                    x,
                    text_embeddings,
                    self.model_args.use_mqa,
                    self.model_args.use_qk_norm,
                    self.model_args.use_causal,
                    self.model_args.left_window,
                    self.model_args.right_window,
                    use_cache,
                    spatio_temoral_padding_mask,
                    text_padding_mask,
                    self.kv_cache,
                    layer_idx,
                    use_reentrant=False    
                )
            else:
                x = layer(
                    x,
                    text_embeddings=text_embeddings,
                    use_mqa=self.model_args.use_mqa,
                    use_qk_norm=self.model_args.use_qk_norm,
                    use_causal=self.model_args.use_causal,
                    left_window=self.model_args.left_window,
                    right_window=self.model_args.right_window,
                    use_cache=use_cache,
                    spatio_temoral_padding_mask=spatio_temoral_padding_mask,
                    text_padding_mask=text_padding_mask,
                    kv_cache=self.kv_cache,
                    layer_idx=layer_idx
                )

        assert x.dim() == 4, f"expected 4 dims, got {x.dim()} dims"

        # Apply RMSNorm
        x = self.rms_norm(x)

        return x # [B, T_frames, H*W, d_model]

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
    kv_cache = KVCache(
        max_batch_size=96,
        max_seq_len=2048,
        num_heads=num_heads,
        head_dim=d_model//num_heads,
        num_layers=4
    )
    x_out = transformer_block(
        x,
        text_embeddings=text_embeddings,
        use_mqa=False,
        use_qk_norm=True,
        use_causal=True,
        left_window=-1,
        right_window=-1,
        use_cache=True,
        spatio_temporal_padding_mask=factorized_padding_mask,
        text_padding_mask=text_padding_mask,
        kv_cache=kv_cache,
        layer_idx=1
    )
    return x_out

def test_transformer_forward():
    model_args = ModelArgs()
    model = AutoregressiveVideoTransformer(model_args).to(device)
    B, T_frames, H, W = 1, 15, 32, 32
    T_tokens = 16
    x = torch.randn(B, T_frames, H*W, model_args.d_model).to(device)
    text_embeddings = torch.randn(B, T_tokens, model_args.d_model).to(device)
    print(x.size(1))
    print(x.size(2))
    print(x.size(3))
    spatio_temporal_padding_mask = torch.randint(
        0, 2, (B, T_frames*H*W), dtype=torch.bool
    ).to(device)
    text_padding_mask = torch.randint(
        0, 2, (B, T_tokens), dtype=torch.bool
    ).to(device)
    x_out = model(
        x,
        text_embeddings=text_embeddings,
        use_cache=True,
        spatio_temoral_padding_mask=spatio_temporal_padding_mask,
        text_padding_mask=text_padding_mask
    )
    return x_out


if __name__ == "__main__":
    x_out = test_transformer_forward()
    # [1, 15, 1024, 1792]
    print(x_out.shape)

# if __name__ == "__main__":
#     x_out = test_transformer_block()
#     print(x_out.shape) # [1, 16, 32*32, 512]
