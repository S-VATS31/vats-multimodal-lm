from typing import Optional

import torch
import torch.nn.functional as F

class ImageGenerationSampler:
    """Image generation sampler featuring several methods."""
    @staticmethod
    def _temp_sampling(logits: torch.Tensor, temperature: float) -> torch.Tensor:
        """Apply temperature sampling to input logits of shape [B, V].
        
        Args:
            logits (torch.Tensor): Input logits to be scaled.
            temperature (float): Logits scaling factor.

        Returns:
            torch.Tensor: Scaled output tensor of shape [B, V].
        """
        if temperature <= 0:
            raise ValueError(f"expected temperature > 0, got {temperature}")
        else:
            return logits / temperature 

    @staticmethod
    def _top_k_sampling(logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """Apply top-k sampling to input logits of shape [B, V].
        
        Args:
            logits (torch.Tensor): Logits tensor of shape [B, V].
            top_k (int): Logits to sample.

        Returns:
            torch.Tensor: Output tensor of shape [B, V] with masked logits.
        """
        if top_k <= 1 or top_k > logits.size(1):
            raise ValueError(f"expected 1 < top_k <= vocab size, got {top_k}")

        # Viable range for top_k
        top_k_logits, _ = torch.topk(logits, k=top_k, dim=-1, sorted=True)
        threshold = top_k_logits[:, -1, None] # Threshold (smallest kept logit)
        masked_logits = torch.where(logits < threshold, float("-inf"), logits)
        return masked_logits

    @staticmethod
    def _top_p_sampling(logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Apply top-p sampling to input logits of shape [B, V].
        
        Args:
            logits (torch.Tensor): Logits tensor of shape [B, V].
            top_p (float): Top-p hyperparameter.

        Returns:
            torch.Tensor: Output tensor with top-p sampling applied.
        """
        return logits

    @staticmethod
    def _greedy_sampling(logits: torch.Tensor) -> torch.LongTensor:
        """Apply greedy sampling to input logits of shape [B, V].
        
        Args:
            logits (torch.Tensor): Input logits of shape [B, V].

        Returns:
            torch.LongTensor: Returns indices of largest logits per row.
        """
        return torch.argmax(logits, dim=-1)
    
    @staticmethod
    def sample_next_token(
        logits: torch.Tensor,
        use_sampling: bool,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """Autoregressively sample next token with or without using decoding methods.
        
        Args:
            logits (torch.Tensor): Logits tensor of shape [B, V].
            use_sampling (bool): Whether to apply sampling methods or not.
            temperature (Optional[float]): Temperature hyperparameter.
            top_k (Optional[int]): Top-k sampling hyperparameter.
            top_p (Optional[float]): Top-p sampling hyperparameter.

        Returns:
            torch.Tensor: Output tensor of same shape with predicted tokens.
        """
        # Apply sampling
        if use_sampling:
            logits = ImageGenerationSampler._temp_sampling(logits, temperature)
            logits = ImageGenerationSampler._top_k_sampling(logits, top_k)
            logits = ImageGenerationSampler._top_p_sampling(logits, top_p)
        else:
            return ImageGenerationSampler._greedy_sampling(logits)
        
        # Get probabities and predict next token
        probs = F.softmax(logits, dim=-1) # [B, V]
        next_token = probs.multinomial(num_samples=1) # [B, V]

        return next_token

def main():
    logits = torch.tensor([[1.0, 2.0, 3.0, 0.5]])
    print("Logits:", logits)

    # Temperature sampling
    temp_scaled = ImageGenerationSampler._temp_sampling(logits, temperature=2.0)
    print("Temperature scaled:", temp_scaled)

    # Top-k sampling
    top_k_masked = ImageGenerationSampler._top_k_sampling(logits, top_k=2)
    print("Top-k masked:", top_k_masked)

    # Greedy sampling
    greedy = ImageGenerationSampler._greedy_sampling(logits)
    print("Greedy sampling token:", greedy)

    # Full sample with sampling enabled
    sampled = ImageGenerationSampler.sample_next_token(
        logits,
        use_sampling=True,
        temperature=1.0,
        top_k=2,
        top_p=1.0,
    )
    print("Sampled token (use_sampling=True):", sampled)

    # Full sample with greedy decoding
    greedy_full = ImageGenerationSampler.sample_next_token(
        logits,
        use_sampling=False
    )
    print("Sampled token (use_sampling=False):", greedy_full)

if __name__ == "__main__":
    main()