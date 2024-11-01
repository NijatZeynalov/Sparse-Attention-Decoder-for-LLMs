import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
from ..config.config import AttentionConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


class TopKTokenSelector:
    """
    Implements dynamic token selection strategies for sparse attention.
    """

    def __init__(self, config: AttentionConfig):
        self.config = config
        self.selection_history = {}
        self.token_importance_cache = {}
        self.window_size = 1024  # Size of local attention window
        self.min_tokens = 32  # Minimum tokens to select
        self.sliding_window_size = 128  # Size of sliding window for dynamic selection

    def __call__(
            self,
            attention_scores: torch.Tensor,
            persistent_indices: Optional[torch.Tensor] = None,
            top_k: Optional[int] = None
    ) -> torch.Tensor:
        """
        Select top-k tokens based on attention scores and persistence.

        Args:
            attention_scores: Attention score matrix [batch_size, num_heads, seq_length, seq_length]
            persistent_indices: Previously selected important token indices
            top_k: Number of tokens to select (overrides config if provided)

        Returns:
            torch.Tensor: Selected token indices
        """
        batch_size, num_heads, seq_length, _ = attention_scores.shape
        k = top_k if top_k is not None else self.config.top_k

        # Compute importance scores
        importance_scores = self._compute_importance_scores(attention_scores)

        # Apply sliding window attention for local context
        local_indices = self._get_local_attention_indices(seq_length)

        # Combine with global attention
        global_indices = self._select_global_tokens(importance_scores, k // 2)

        # Merge local and global attention indices
        selected_indices = self._merge_attention_indices(
            local_indices,
            global_indices,
            persistent_indices,
            batch_size,
            k
        )

        return selected_indices

    def _compute_importance_scores(self, attention_scores: torch.Tensor) -> torch.Tensor:
        """
        Compute token importance scores using multiple criteria.
        """
        # Max attention across heads
        max_attention = torch.max(attention_scores, dim=1)[0]

        # Mean attention across sequence length
        mean_attention = torch.mean(attention_scores, dim=-1)

        # Gradient of attention scores
        attention_grad = torch.abs(
            attention_scores[..., 1:] - attention_scores[..., :-1]
        )
        grad_score = torch.mean(attention_grad, dim=(-1, -2))

        # Position bias (favor recent tokens)
        seq_length = attention_scores.shape[-1]
        position_bias = torch.arange(
            seq_length,
            device=attention_scores.device,
            dtype=torch.float32
        ) / seq_length
        position_bias = position_bias.unsqueeze(0).unsqueeze(0)

        # Combine scores
        importance_scores = (
                0.4 * max_attention +
                0.3 * mean_attention +
                0.2 * grad_score +
                0.1 * position_bias
        )

        return importance_scores

    def _get_local_attention_indices(self, seq_length: int) -> torch.Tensor:
        """
        Get indices for local sliding window attention.
        """
        if seq_length <= self.sliding_window_size:
            return torch.arange(seq_length)

        # Create sliding windows
        windows = []
        for i in range(0, seq_length - self.sliding_window_size + 1, self.sliding_window_size // 2):
            end = min(i + self.sliding_window_size, seq_length)
            windows.append(torch.arange(i, end))

        return torch.cat(windows)

    def _select_global_tokens(
            self,
            importance_scores: torch.Tensor,
            num_tokens: int
    ) -> torch.Tensor:
        """
        Select globally important tokens.
        """
        # Get top-k tokens based on importance scores
        _, indices = torch.topk(
            importance_scores,
            k=min(num_tokens, importance_scores.shape[-1]),
            dim=-1
        )

        # Ensure sufficient diversity in selection
        if num_tokens > self.min_tokens:
            # Add some random tokens for exploration
            random_indices = torch.randint(
                0,
                importance_scores.shape[-1],
                (importance_scores.shape[0], num_tokens // 8),
                device=importance_scores.device
            )
            indices = torch.cat([indices, random_indices], dim=-1)

        return indices

    def _merge_attention_indices(
            self,
            local_indices: torch.Tensor,
            global_indices: torch.Tensor,
            persistent_indices: Optional[torch.Tensor],
            batch_size: int,
            max_tokens: int
    ) -> torch.Tensor:
        """
        Merge different types of attention indices.
        """
        device = global_indices.device
        all_indices = [local_indices.to(device), global_indices]

        if persistent_indices is not None:
            all_indices.append(persistent_indices.to(device))

        # Combine all indices
        merged = torch.cat(all_indices, dim=-1)

        # Remove duplicates
        merged = torch.unique(merged, dim=-1)

        # Limit to max_tokens
        if merged.size(-1) > max_tokens:
            merged = merged[..., :max_tokens]

        # Expand to batch size if needed
        if merged.dim() == 1:
            merged = merged.unsqueeze(0).expand(batch_size, -1)

        return merged

    def update_importance_cache(
            self,
            layer_idx: int,
            token_indices: torch.Tensor,
            importance_scores: torch.Tensor
    ) -> None:
        """
        Update cache of token importance scores.
        """
        # Update exponential moving average of importance scores
        if layer_idx in self.token_importance_cache:
            old_scores = self.token_importance_cache[layer_idx]
            alpha = 0.8  # EMA decay rate
            self.token_importance_cache[layer_idx] = (
                    alpha * old_scores + (1 - alpha) * importance_scores
            )
        else:
            self.token_importance_cache[layer_idx] = importance_scores

        # Keep cache size bounded
        if len(self.token_importance_cache) > 100:  # Max layers to track
            min_layer = min(self.token_importance_cache.keys())
            del self.token_importance_cache[min_layer]

    def reset_cache(self) -> None:
        """Reset importance score cache."""
        self.token_importance_cache.clear()
        self.selection_history.clear()

    def get_selection_stats(self) -> Dict[str, float]:
        """
        Get statistics about token selection.
        """
        if not self.selection_history:
            return {}

        total_selections = sum(len(indices) for indices in self.selection_history.values())
        unique_selections = len(set().union(*[
            set(indices.cpu().numpy().flatten())
            for indices in self.selection_history.values()
        ]))

        return {
            "total_selections": total_selections,
            "unique_selections": unique_selections,
            "selection_diversity": unique_selections / total_selections
            if total_selections > 0 else 0.0
        }