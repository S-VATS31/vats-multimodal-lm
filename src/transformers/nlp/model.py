from configs.setup_env import device, dtype

import math
import warnings
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from src.rms_norm import RMSNorm
from src.transformers.nlp.moe import MoEBlock
from src.optimized_attention import AttentionBlock, KVCache
from configs.transformers.nlp.model_args.model_args_medium import ModelArgs

class TransformerBlock(nn.Module):
    """Transformer block that will be stacked in the final Transformer class.

    Args:
        d_model (int): Dimensionality of the model's embeddings.
        num_heads (int): Number of attention heads for KV vectors, shared across grouped query heads in GQA.
        query_groups (int): Number of query groups that partition the queries into subsets.
        softmax_scale (float): Float to scale the attention scores by.
        use_proj_bias (bool): Whether to use bias in q, k, v, o projections.
        use_qkv_proj (bool): Whether to use fused qkv proj or individual q, k, v projections.
        d_ffn (int): Dimensionality of the feed forward network.
        dropout (float): Probability that model components will be randomly dropped out.
        num_experts (int): Number of feed forward networks in the MoE layer.
        top_k (int): Number of experts each token is routed to.
        theta (float): Exponential base of the inverse frequency.
        eps (float): Small epsilon value to ensure numerical stability.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        query_groups: int,
        softmax_scale: float,
        use_proj_bias: bool,
        use_qkv_proj: bool,
        d_ffn: int,
        dropout: float,
        num_experts: int,
        top_k: int,
        theta: float,
        eps: float,
    ):
        super().__init__()

        self.attn_block = AttentionBlock(
            d_model=d_model, 
            num_heads=num_heads, 
            query_groups=query_groups,
            softmax_scale=softmax_scale,
            use_proj_bias=use_proj_bias,
            use_qkv_proj=use_qkv_proj,
            dropout=dropout, 
            theta=theta, 
            eps=eps
        )
        self.moe_block = MoEBlock(
            d_model=d_model, 
            d_ffn=d_ffn, 
            dropout=dropout, 
            num_experts=num_experts, 
            top_k=top_k, 
            eps=eps
        )

    def forward(
        self,
        x: torch.Tensor,
        left_window: int,
        right_window: int,
        causal: bool = True,
        padding_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        layer_idx: Optional[int] = None,
        use_cache: bool = False,
        use_mqa: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        """Forward pass of the Transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape [B, T, d_model].
            window_size (Tuple[int, int]): Window size for SWA.
            padding_mask (Optional[torch.Tensor]): Padding tensor of shape [B, T].
            kv_cache (Optional[KVCache]): Key-value cache for efficient generation.
            layer_idx (Optional[int]): Index of the current layer for cache access.
            use_cache (bool): Whether to use KV caching during forward pass.

        Returns:
            Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]], torch.Tensor]:
                - Output tensor of shape [B, T, d_model].
                - Cache dictionary with 'k' and 'v' tensors if use_cache is True, else None.
                - Auxiliary loss from the MoE layer.
        """
        with torch.amp.autocast(device_type=device.type, dtype=dtype):
            attn_out, cache_out = self.attn_block(
                x=x,
                left_window=left_window,
                right_window=right_window,
                causal=causal,
                padding_mask=padding_mask,
                kv_cache=kv_cache,
                layer_idx=layer_idx,
                use_cache=use_cache,
                use_mqa=use_mqa
            )
            moe_out, aux_loss = self.moe_block(attn_out)
            return moe_out, cache_out, aux_loss


class Transformer(nn.Module):
    """Complete Transformer class stacking all decoder blocks.

    Args:
        model_args (ModelArgs): Dataclass containing all model arguments.
    """
    def __init__(self, model_args: ModelArgs):
        super().__init__()

        self.model_args = model_args

        self.token_embed = nn.Embedding(model_args.vocab_size, model_args.d_model).to(device)
        self.dropout = nn.Dropout(p=model_args.dropout).to(device)

        # Stack transformer blocks with MoE
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=model_args.d_model,
                num_heads=model_args.num_heads,
                query_groups=model_args.query_groups,
                softmax_scale=model_args.softmax_scale,
                use_proj_bias=model_args.use_proj_bias,
                use_qkv_proj=model_args.use_qkv_proj,
                d_ffn=model_args.d_ffn,
                dropout=model_args.dropout,
                num_experts=model_args.num_experts,
                top_k=model_args.top_k,
                theta=model_args.rope_base,
                eps=model_args.rms_norm_eps
            ).to(device) for _ in range(model_args.num_layers)
        ])

        self.rms_norm = RMSNorm(model_args.d_model, model_args.rms_norm_eps).to(device)

        # Initialize KV Cache
        self.kv_cache = KVCache(
            max_batch_size=model_args.max_batch_size,
            max_seq_len=model_args.max_seq_len,
            num_heads=model_args.query_groups,
            head_dim=model_args.d_model // model_args.num_heads,
            num_layers=model_args.num_layers,
        )

        # Language modeling head, bias=False for weight tying
        self.lm_head = nn.Linear(model_args.d_model, model_args.vocab_size, bias=False).to(device)

        self.apply(self._init_weights)

        # Tie weights if flag is True after initialization
        if model_args.tie_weights:
            self.lm_head.weight = self.token_embed.weight

    def _init_weights(self, module) -> None:
        """Initializes model weights according to layer type.
        
        Args:
            module: PyTorch module to be initialized.
        """
        num_layers = self.model_args.num_layers
        init_std = 0.02

        # Initialize Embeddings
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=init_std)

        # Initialize linear layers
        elif isinstance(module, nn.Linear):
            # Identify linear layers
            is_qkv = hasattr(module, '_is_qkv') or any(hasattr(p, 'w_qkv') and p.w_qkv is module for p in self.modules())
            is_attn_out = hasattr(module, '_is_attn_out') or any(hasattr(p, 'w_o') and p.w_o is module for p in self.modules())
            is_ffn_gate = hasattr(module, '_is_ffn_gate') or any(hasattr(p, 'weight1') and p.weight1 is module for p in self.modules())
            is_ffn_up = hasattr(module, '_is_ffn_up') or any(hasattr(p, 'weight3') and p.weight3 is module for p in self.modules())
            is_ffn_down = hasattr(module, '_is_ffn_down') or any(hasattr(p, 'weight2') and p.weight2 is module for p in self.modules())
            is_lm_head = hasattr(self, 'lm_head') and self.lm_head is module
            is_router = any(hasattr(p, 'router') and hasattr(p.router, 'weight') and p.router.weight is module for p in self.modules())

            # Initialize input projections
            if is_qkv or is_ffn_gate or is_ffn_up or is_router:
                nn.init.xavier_uniform_(module.weight)
                if num_layers > 12:
                    module.weight.data *= (1.0 / math.sqrt(num_layers / 6.0))

            # Initialize output projections
            elif is_attn_out or is_ffn_down:
                scaled_std = init_std / math.sqrt(2 * num_layers)
                nn.init.normal_(module.weight, mean=0.0, std=scaled_std)

            # Initialize language modeling head
            elif is_lm_head:
                nn.init.normal_(module.weight, mean=0.0, std=init_std)

            # Default to Xavier initialization
            else:
                nn.init.xavier_uniform_(module.weight)

            # Initialize bias
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

        # Initialize RMSNorm weight (scaling factor)
        elif hasattr(module, 'weight') and module.__class__.__name__ == 'RMSNorm':
            nn.init.constant_(module.weight, 1.0)

    def forward(
        self,
        input_ids: torch.LongTensor,
        padding_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[Dict[str, torch.Tensor]]], torch.Tensor]:
        """Forward pass of the entire Transformer.

        Args:
            input_ids (torch.Tensor): Input tensor of shape [B, T].
            padding_mask (Optional[torch.Tensor]): Padding tensor of shape [B, T].
            use_cache (bool): Whether to use KV caching during forward pass.

        Returns:
            Tuple[torch.Tensor, Optional[List[Dict[str, torch.Tensor]]], torch.Tensor]:
                - torch.Tensor: Output logits of shape [B, T, vocab_size].
                - Optional[List[Dict[str, torch.Tensor]]]: List of cache dictionaries for each layer.
                - Sum of auxiliary losses from all MoE layers.
        """
        with torch.amp.autocast(device_type=device.type, dtype=dtype):
            assert (
                input_ids.dim() == 2
            ), f"input_ids must have 2 dimensions, got {input_ids.dim()}"
            # Ensure input_ids are int64 for nn.Embedding() layer
            if input_ids.dtype != torch.int64:
                warnings.warn(f"got input_ids of {input_ids.dtype}, casting to int64")
                input_ids = input_ids.to(torch.int64)

            # Ensure padding mask/input_ids are the same shape
            if padding_mask is not None:
                assert (
                    input_ids.shape == padding_mask.shape
                ), "input_ids and padding_mask must have the same shape for correct masking."

            # input_ids dimensions/dtype assertions
            assert (
                input_ids.dim() == 2
            ), f"input_ids must be a 2 dimensional tensor, got {input_ids.dim()}"
            assert (
                input_ids.dtype == torch.int64
            ), f"input_ids dtype must be int64, got {input_ids.dtype}"

            # Apply embeddings
            x = self.token_embed(input_ids) # [B, T, d_model]

            assert (
                x.dim() == 3
            ), f"x must be a 3 dimensional tensor, got {x.dim()}"

            # Final dropout
            x = self.dropout(x)

            # Initialize KV cache outputs as list
            cache_outs = [] if use_cache else None

            # Initialize aux loss as float32 tensor
            total_aux_loss = torch.tensor(0.0, dtype=torch.float32).to(device)

            assert (
                total_aux_loss.dtype == torch.float32
            ), f"total_aux_loss dtype must be float32, got {total_aux_loss.dtype}"

            # Stack transformer layers
            for i, layer in enumerate(self.layers):
                if self.model_args.gradient_checkpointing:
                    x, cache_out, aux_loss = checkpoint(
                        layer,
                        x,
                        self.model_args.left_window,
                        self.model_args.right_window,
                        self.model_args.use_causal,
                        padding_mask,
                        self.kv_cache,
                        i, # layer_idx
                        use_cache,
                        self.model_args.use_mqa,
                        use_reentrant=False
                    )
                else:
                    x, cache_out, aux_loss = layer(
                        x=x,
                        left_window=self.model_args.left_window,
                        right_window=self.model_args.right_window,
                        causal=self.model_args.use_causal,
                        padding_mask=padding_mask,
                        kv_cache=self.kv_cache,
                        layer_idx=i,
                        use_cache=use_cache,
                        use_mqa=self.model_args.use_mqa,
                    )
                # Accumulate auxiliary loss
                if aux_loss is not None:
                    total_aux_loss += aux_loss
                
                # Apped to cache
                if use_cache:
                    cache_outs.append(cache_out)

            # Final RMSNorm
            x = self.rms_norm(x)

            # Final projection to logits
            logits = self.lm_head(x) # [B, T, V]
            assert (
                logits.dim() == 3
            ), f"logits must have 3 dimenions, got {logits.dim()}"

            return logits, cache_outs, total_aux_loss
