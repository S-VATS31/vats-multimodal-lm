from configs.setup_env import device, dtype

from typing import Optional

import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.checkpoint import checkpoint

from src.rms_norm import RMSNorm
from src.ffn_block import FFNBlock
from src.optimized_attention import KVCache
from configs.autoregressive_image_gen.autoregressive_transformer.model_args.model_args_xlarge import ModelArgs
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
        causal: bool,
        left_window: int,
        right_window: int,
        use_cache: bool,
        causal_padding_mask: Optional[torch.Tensor] = None,
        cross_padding_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        layer_idx: Optional[int] = None
    ) -> torch.Tensor:
        """Forward pass of transformer block.
        
        Args:
            x (torch.Tensor): Image tokens of shape [B, H*W, d_model].
            text_embeddings (torch.Tensor): Text tokens of shape [B, T, d_model].
            use_mqa (bool): Whether to use MQA or not.
            use_qk_norm (bool): Whether to use QK normalization or not.
            causal (bool): Whether to use causal masking or not.
            left_window (int): Left window for SWA.
            right_window (int): Right window for SWA.
            use_cache (bool): Whether to use KV cache or not.
            causal_padding_mask (Optional[torch.Tensor]): Padding mask for autoregressive image gen. Shape: [B, H*W].
            cross_padding_mask (Optional[torch.Tensor]): Padding mask for text encoder. Shape: [B, T_k].
            kv_cache (Optional[KVCache]): KVCache module.
            layer_idx (Optional[int]): Current layer to update KV cache for.

        Returns:
            torch.Tensor: Output tensor of shape [B, H*W, d_model].
        """
        with autocast(device_type=device.type, dtype=dtype):
            # Causal self attention block
            causal_attention_out = self.causal_attention_block(
                x,
                use_mqa=use_mqa,
                use_qk_norm=use_qk_norm,
                causal=causal,
                left_window=left_window,
                right_window=right_window,
                use_cache=use_cache,
                padding_mask=causal_padding_mask,
                kv_cache=kv_cache,
                layer_idx=layer_idx
            )

            # Cross attention block
            cross_attention_out = self.cross_attention_block(
                causal_attention_out, 
                text_embeddings=text_embeddings,
                padding_mask=cross_padding_mask
            )

            # FFN block
            block_out = self.ffn_block(cross_attention_out)

            return block_out

class AutoregressiveImageTransformer(nn.Module):
    """Autorgressive image transformer complete module.
    
    Args:
        model_args (ModelArgs): Model hyperparameters.
    """
    def __init__(self, model_args: ModelArgs):
        super().__init__()

        self.model_args = model_args

        # Set up dropout
        self.dropout = nn.Dropout(p=model_args.dropout).to(device)

        # Set up transformer blocks
        self.layers = nn.ModuleList([
            AutoregressiveTransformerBlock(
                d_model=model_args.d_model,
                num_heads=model_args.num_heads,
                query_groups=model_args.query_groups,
                rope_theta=model_args.rope_theta,
                softmax_scale=model_args.softmax_scale,
                use_proj_bias=model_args.use_proj_bias,
                use_fused_proj=model_args.use_qkv_proj,
                use_windowed_attn=model_args.use_windowed_attn,
                use_ntk_rope=model_args.use_ntk_rope,
                dropout=model_args.dropout,
                eps=model_args.rms_norm_eps,
                d_ffn=model_args.d_ffn,
                ntk_scale_factor=model_args.ntk_scale_factor
            ).to(device) for _ in range(model_args.num_layers)
        ])

        # Initialize KV cache
        # PRE-ALLOCATE USING NUM HEADS, WE WILL EXPAND QUERY GROUPS -> NUM HEADS
        self.kv_cache = KVCache(
            max_batch_size=model_args.max_batch_size,
            max_seq_len=model_args.max_position_embeddings,
            num_heads=model_args.num_heads,
            head_dim=model_args.d_model // model_args.num_heads,
            num_layers=model_args.num_layers
        )

        # Set up RMSNorm
        self.rms_norm = RMSNorm(
            d_model=model_args.d_model, eps=model_args.rms_norm_eps
        ).to(device)

        # TODO: work on weight initialization
        # TODO: add weight initialization based on module named and best init val
        
        # Initialize weights
        #self.apply(self._init_weights)

    def _init_weights(self, module) -> None:
        """Initialize weights for different modules.
        
        Args:
            module: Module to be initialized.
        """
        pass

    def forward(
        self,
        x: torch.Tensor,
        text_embeddings: torch.Tensor,
        use_cache: bool,
        causal_padding_mask: Optional[torch.Tensor] = None,
        cross_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of autoregressive image generation transformer.
        
        Args:
            x (torch.Tensor): Image tokens of shape [B, H*W, d_model].
            text_embeddings (torch.Tensor): Text tokens of shape [B, T, d_model].
            causal_padding_mask (Optional[torch.Tensor]): Causal padding tensor of shape [B, H*W].
            cross_padding_mask (Optional[torch.Tensor]): Encoder padding tensor of shape [B, T_k].
            use_cache (bool): Whether to use KV caching or not.
        
        Returns:
            torch.Tensor: Output tensor of shape [B, H*W, d_model].
        """
        assert x.dim() == 3, f"expected 3 dims, got {x.dim()} dims"
        assert text_embeddings.dim() == 3, f"expected 3 dims, got {text_embeddings.dim()} dims"
        assert causal_padding_mask.dim() == 2, f"expected 2 dims, got {causal_padding_mask.dim()} dims"
        assert cross_padding_mask.dim() == 2, f"expected 2 dims, got {cross_padding_mask.dim()} dims"
        assert x.size(1) == causal_padding_mask.size(1), (
            f"expected x.size(1) == causal_padding_mask.size(1), "
            f"got {x.size(1)}, {causal_padding_mask.size(1)}, respectively"
        )
        assert text_embeddings.size(1) == cross_padding_mask.size(1), (
            f"expected text_embeddings.size(1) == cross_padding_mask.size(1), "
            f"got {text_embeddings.size(1)}, {cross_padding_mask.size(1)} respectively"
        )

        # Apply final dropout
        x = self.dropout(x) # [B, H*W, d_model]

        # Loop through transformer blocks
        for layer_idx, layer in enumerate(self.layers):
            if self.model_args.use_checkpointing:
                x = checkpoint(
                    layer,
                    x,
                    text_embeddings,
                    self.model_args.enable_mqa,
                    self.model_args.use_qk_norm,
                    self.model_args.use_causal,
                    self.model_args.left_window,
                    self.model_args.right_window,
                    use_cache,
                    causal_padding_mask,
                    cross_padding_mask,
                    self.kv_cache,
                    layer_idx,
                    use_reentrant=False
                )
            else:
                x = layer(
                    x,
                    text_embeddings=text_embeddings,
                    use_mqa=self.model_args.enable_mqa,
                    use_qk_norm=self.model_args.use_qk_norm,
                    causal=self.model_args.use_causal,
                    left_window=self.model_args.left_window,
                    right_window=self.model_args.right_window,
                    use_cache=use_cache,
                    causal_padding_mask=causal_padding_mask,
                    cross_padding_mask=cross_padding_mask,
                    kv_cache=self.kv_cache,
                    layer_idx=layer_idx
                )

        # Apply final RMSNorm
        x = self.rms_norm(x)

        return x # [B, H*W, d_model]

def test_transformer_block():
    d_model, num_heads, query_groups, rope_theta = 512, 32, 8, 10000.0
    softmax_scale = 1 / (d_model // num_heads) ** 0.5
    use_proj_bias, use_fused_proj, use_windowed_attn, use_ntk_rope = False, True, True, True
    dropout, eps, max_position_embeddings, ntk_scale_factor = 0.15, 1e-7, 2048, 0.8
    d_ffn = d_model*4
    block = AutoregressiveTransformerBlock(
        d_model, num_heads, query_groups, rope_theta, softmax_scale,
        use_proj_bias, use_fused_proj, use_windowed_attn, use_ntk_rope,
        dropout, eps, d_ffn, ntk_scale_factor
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
    # 2 seperate masks
    image_padding_mask = torch.randint(
        0, 2, (B, T_q), dtype=torch.bool
    ).to(device)
    text_padding_mask = torch.randint(
        0, 2, (B, T_k), dtype=torch.bool
    ).to(device)
    x_out = block(
        x,
        text_embeddings,
        use_mqa=False,
        use_qk_norm=True,
        causal=True,
        left_window=-1,
        right_window=-1,
        use_cache=True,
        causal_padding_mask=image_padding_mask,
        cross_padding_mask=text_padding_mask,
        kv_cache=kv_cache,
    )

    return x_out

def test_model_forward():
    model_args = ModelArgs()
    model = AutoregressiveImageTransformer(model_args).to(device)
    B, H, W, d_model = 1, 12, 12, model_args.d_model
    T_q = H*W
    T_k = 4
    x = torch.randn(B, T_q, d_model).to(device)
    text_embeddings = torch.randn(B, T_k, d_model).to(device)
    image_padding_mask = torch.randint(
        0, 2, (B, T_q), dtype=torch.bool
    ).to(device)
    text_padding_mask = torch.randint(
        0, 2, (B, T_k), dtype=torch.bool
    ).to(device)
    x_out = model(
        x,
        text_embeddings=text_embeddings,
        use_cache=True,
        causal_padding_mask=image_padding_mask,
        cross_padding_mask=text_padding_mask
    )
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return x_out, params

if __name__ == "__main__":
    x, params = test_model_forward()
    print(x.shape) # [1, 144, d_model]
    print(f"{params:,}")
