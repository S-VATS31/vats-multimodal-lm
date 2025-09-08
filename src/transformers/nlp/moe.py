from configs.setup_env import device, dtype

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rms_norm import RMSNorm
from src.swiglu_activation import SwiGLUActivation

class TopKRouter(nn.Module):
    """TopK Routing for MoE layer.

    Args:
        d_model (int): Dimensionality of the model's embeddings.
        num_experts (int): Number of FFNs (SwiGLU in this case).
        top_k (int): Number of experts each token is routed to.
        use_aux_loss (bool): Flag to use auxiliary loss or not.
    """
    def __init__(
            self,
            d_model: int,
            num_experts: int,
            top_k: int,
            use_aux_loss: bool = True,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_aux_loss = use_aux_loss

        # Set up router (projects from d_model to num_experts)
        self.router = nn.Linear(d_model, num_experts)

    def forward(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  
        """Forward pass of the routing layer.

        Args:
            x (torch.Tensor): Input tensor of shape [B, T, d_model].

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - torch.Tensor: Tensor containing gating scores.
                - Torch.Tensor: Indices of gating scores.
                - torch.Tensor: Tensor containing auxiliary loss.
        """
        with torch.amp.autocast(device_type=device.type, dtype=dtype):
            assert (
                x.dim() == 3
            ), f"x must have 3 dimensions, got {x.dim()}"
            B, T, _ = x.shape

            # Flatten for efficiency
            x_flattened = x.view(-1, self.d_model) # [B * T, d_model]
            assert (
                x_flattened.shape == (B * T, self.d_model)
            ), f"x must have shape of {(B * T, self.d_model)}"

            # Compute logits
            logits = self.router(x_flattened) # [B * T, num_experts]
            assert (
                logits.shape == (B * T, self.num_experts)
            ), f"logits must have shape of {(B * T, self.num_experts)}, got {logits.shape}"

            # Get probabilities
            prob_scores = F.softmax(logits, dim=-1) # [B * T, num_experts]
            assert (
                prob_scores.shape == (B * T, self.num_experts)
            ), f"logits must have shape of {(B * T, self.num_experts)}, got {prob_scores.shape}"

            # Get top-k experts
            top_k_values, top_k_indices = torch.topk(prob_scores, self.top_k, dim=-1) # Both: [B * T, top_k]
            assert (
                top_k_values.shape == (B * T, self.top_k)
            ), f"top_k_values must have shape of {(B * T, self.top_k)}, got {top_k_values.shape}"
            assert (
                top_k_indices.shape == (B * T, self.top_k)
            ), f"top_k_indices must have shape of {(B * T, self.top_k)}, got {top_k_indices.shape}"

            # Get weights
            top_k_weights = top_k_values / torch.sum(top_k_values, dim=-1, keepdim=True)
            assert (
                top_k_weights.shape == (B * T, self.top_k)
            ), f"top_k_weights must have shape of {(B * T, self.top_k)}"

            # Reshape
            expert_weights = top_k_weights.view(B, T, self.top_k)
            expert_indices = top_k_indices.view(B, T, self.top_k)

            assert (
                expert_weights.shape == (B, T, self.top_k)
            ), f"expert_weights must have shape of {(B, T, self.top_k)}, got {expert_weights.shape}"
            assert (
                expert_indices.shape == (B, T, self.top_k)
            ), f"expert_indices must have shape of {(B, T, self.top_k)}, got {expert_indices.shape}"

            # Compute auxiliary loss
            aux_loss = torch.tensor(0.0).to(x.device) # Initialize aux loss
            if self.use_aux_loss and self.training:
                aux_loss = self._compute_aux_loss(prob_scores)

            return expert_weights, expert_indices, aux_loss

    def _compute_aux_loss(self, prob_scores: torch.Tensor) -> torch.Tensor:
        """Compute the auxiliary loss if applicable.

        Args:
            prob_scores (torch.Tensor): Probability of each expert being chosen over all tokens.

        Returns:
            cv (torch.Tensor): Coefficient of variation.
        """
        # [B * T, num_experts]
        experts = prob_scores.sum(dim=0) # Sum over total tokens
        experts_fractions = experts / experts.sum()

        # Compute coefficient of variation
        cv = experts_fractions.std(unbiased=False) / experts_fractions.mean()

        return cv

class MoELayer(nn.Module):
    """Mixture of Experts layer.

    Args:
        d_model (int): Dimensionality of the model's embeddings.
        d_ffn (int): Dimensionality of the feed forward network.
        dropout (float): Probability of model components being dropped out.
        num_experts (float): Number of feed forward networks.
        top_k (float): Number of experts each token is routed to.
        eps (float): Small value to maintain numerical stability in RMSNorm.
    """
    def __init__(
            self,
            d_model: int,
            d_ffn: int,
            dropout: float,
            num_experts: int,
            top_k: int,
            eps: float,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_ffn = d_ffn
        self.num_experts = num_experts
        self.top_k = top_k

        # Router
        self.router = TopKRouter(
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k,
            use_aux_loss=True
        )

        # Experts
        self.experts = nn.ModuleList([
            SwiGLUActivation(
                d_model, d_ffn, dropout
            ).to(device) for _ in range(num_experts)
        ])

        # Set up RMSNorm
        self.rms_norm = RMSNorm(d_model, eps)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the Mixture of Experts layer.

        Args
            x (torch.Tensor): Input tensor of shape [B, T, d_model].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - out: Ouput tensor of shape [B, T, d_model].
                - aux_loss: Auxiliary loss.
        """
        assert (
            x.dim() == 3
        ), f"x must be a 3 dimensional tensor, got {x.dim()}"
        B, T, _ = x.shape

        # Apply RMSNorm
        x = self.rms_norm(x)

        # Get routers
        expert_weights, expert_indices, aux_loss = self.router(x)

        assert (
           expert_weights.shape == (B, T, self.top_k)
        ), f"expert_weights must have shape of {(B, T, self.top_k)}, got {expert_weights.shape}"
        assert (
            expert_indices.shape == (B, T, self.top_k)
        ), f"expert_indices must have shape of {(B, T, self.top_k)}, got {expert_indices.shape}"

        # Flatten for efficiency
        x_flattened = x.view(-1, self.d_model) # [B * T, d_model]
        assert (
            x_flattened.shape == (B * T, self.d_model)
        ), f"x_flattened must have shape of {(B * T, self.d_model)}"

        # Initialize output
        out = torch.zeros_like(x_flattened)

        # Process all experts
        for expert_id in range(self.num_experts):
            expert_mask = (expert_indices == expert_id) # [B, T, top_k]
            assert (
                expert_mask.shape == (B, T, self.top_k)
            ), f"expert_mask must have shape of {(B, T, self.top_k)}, got {expert_mask.shape}"

            if expert_mask.any():
                # Get positions where this expert is used
                expert_positions = expert_mask.nonzero(as_tuple=False) # [num_matches, 3] [B, T, expert_id]

                if expert_positions.numel() > 0:
                    # Get the corresponding tokens
                    batch_indices = expert_positions[:, 0]
                    seq_indices = expert_positions[:, 1]
                    topk_indices = expert_positions[:, 2]

                    # Convert to flattened indices
                    flat_indices = batch_indices * T + seq_indices

                    # Get input tokens for this expert
                    expert_input = x_flattened[flat_indices] # [num_matches, d_model]

                    # Get weights for this expert
                    expert_weight_vals = expert_weights[batch_indices, seq_indices, topk_indices] # [num_matches]

                    # Forward through expert
                    expert_output = self.experts[expert_id](expert_input) # [num_matches, d_model]

                    # Apply weights and accumulate
                    weighted_output = expert_output * expert_weight_vals[..., None] # [num_matches, d_model]

                    # Add to output
                    out[flat_indices] += weighted_output

        # Reshape back to [B, T, d_model]
        out = out.view(B, T, self.d_model)
        assert (
            out.shape == (B, T, self.d_model)
        ), f"out must have shape of {(B, T, self.d_model)}, got {out.shape}"

        return out, aux_loss


class MoEBlock(nn.Module):
    """Mixture of Experts block where RMSNorm, Dropout, and residuals are applied.

    Args:
        d_model (int): Dimensionality of the model's embeddings.
        d_ffn (int): Dimensionality of the feed forward network.
        dropout (float): Probability that model components will be randomly dropped out.
        num_experts (int): Number of feed forward networks in the MoE layer.
        top_k (int): Number of experts each token is routed to.
        eps (float): Small epsilon value to ensure numerical stability.
    """
    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        dropout: float,
        num_experts: int,
        top_k: int,
        eps: float,
    ):
        super().__init__()

        self.rms_norm = RMSNorm(d_model, eps)
        self.moe = MoELayer(d_model, d_ffn, dropout, num_experts, top_k, eps)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the MoE Block.

        Args:
            x (torch.Tensor): Input tensor of shape [B, T, d_model].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Output tensor passed through the MoE Block with the same shape.
                - Auxiliary loss from the MoE layer.
        """
        with torch.amp.autocast(device_type=device.type, dtype=dtype):
            moe_out, aux_loss = self.moe(self.rms_norm(x))
            return x + self.dropout(moe_out), aux_loss
        