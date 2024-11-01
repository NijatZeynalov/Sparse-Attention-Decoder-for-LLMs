import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple
from ..config.config import AttentionConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


class PositionPersistenceManager:
    """
    Manages position persistence across attention layers to optimize token selection.
    """

    def __init__(self, config: AttentionConfig):
        self.config = config
        self.persistent_indices: Dict[int, torch.Tensor] = {}
        self.attention_history: Dict[int, torch.Tensor] = {}
        self.importance_scores: Dict[int, torch.Tensor] = {}
        self.layer_count = 0

    def get_persistent_indices(
            self,
            layer_idx: int,
            batch_size: int,
            seq_length: int,
            device: torch.device
    ) -> torch.Tensor:
        """
        Get persistent token indices for current layer.

        Args:
            layer_idx: Current layer index
            batch_size: Batch size
            seq_length: Sequence length
            device: Target device

        Returns:
            torch.Tensor: Persistent token indices
        """
        if layer_idx in self.persistent_indices:
            return self.persistent_indices[layer_idx]

        # Initialize with uniform selection if no persistence available
        num_persistent = min(self.config.top_k, seq_length)
        indices = torch.arange(seq_length, device=device)
        indices = indices.unsqueeze(0).expand(batch_size, -1)

        # Randomly select initial persistent tokens
        if seq_length > num_persistent:
            perm = torch.randperm(seq_length, device=device)
            indices = indices[:, perm[:num_persistent]]

        return indices

    def update_persistent_indices(
            self,
            layer_idx: int,
            selected_indices: torch.Tensor,
            attention_scores: torch.Tensor
    ) -> None:
        """
        Update persistent indices based on attention scores.

        Args:
            layer_idx: Current layer index
            selected_indices: Selected token indices
            attention_scores: Attention score matrix
        """
        batch_size, num_heads, seq_length, _ = attention_scores.shape

        # Compute importance scores for selected tokens
        max_attention = torch.max(attention_scores, dim=-1)[0]  # [batch_size, num_heads, seq_length]
        mean_attention = torch.mean(max_attention, dim=1)  # [batch_size, seq_length]

        # Update importance scores
        if layer_idx in self.importance_scores:
            old_scores = self.importance_scores[layer_idx]
            decay_rate = self.config.recompute_threshold
            self.importance_scores[layer_idx] = (
                    decay_rate * old_scores + (1 - decay_rate) * mean_attention
            )
        else:
            self.importance_scores[layer_idx] = mean_attention

        # Store attention history
        self.attention_history[layer_idx] = attention_scores

        # Update persistent indices based on importance scores
        if self.config.use_token_reselection and (layer_idx % self.config.reselection_interval == 0):
            self._reselect_persistent_tokens(layer_idx, selected_indices)

        self.persistent_indices[layer_idx] = selected_indices
        self.layer_count += 1

    def _reselect_persistent_tokens(
            self,
            layer_idx: int,
            current_indices: torch.Tensor
    ) -> None:
        """
        Reselect persistent tokens based on accumulated importance.

        Args:
            layer_idx: Current layer index
            current_indices: Currently selected token indices
        """
        importance_scores = self.importance_scores[layer_idx]

        # Identify tokens with low importance scores
        threshold = torch.quantile(
            importance_scores,
            self.config.min_attention_score,
            dim=-1,
            keepdim=True
        )
        low_importance_mask = importance_scores < threshold

        if torch.any(low_importance_mask):
            # Replace low importance tokens with high attention tokens
            _, high_attention_indices = torch.topk(
                importance_scores,
                k=self.config.top_k,
                dim=-1
            )

            # Combine with some of the current tokens for stability
            num_stable = int(self.config.top_k * 0.7)  # Keep 70% of current tokens
            num_new = self.config.top_k - num_stable

            stable_indices = current_indices[:, :num_stable]
            new_indices = high_attention_indices[:, :num_new]

            combined_indices = torch.cat([stable_indices, new_indices], dim=-1)
            self.persistent_indices[layer_idx] = combined_indices

    def get_token_importance(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Get importance scores for tokens in a layer."""
        return self.importance_scores.get(layer_idx)

    def get_attention_history(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Get attention score history for a layer."""
        return self.attention_history.get(layer_idx)

    def reset(self) -> None:
        """Reset all persistence state."""
        self.persistent_indices.clear()
        self.attention_history.clear()
        self.importance_scores.clear()
        self.layer_count = 0

    def save_state(self) -> Dict:
        """Save persistence state for later restoration."""
        return {
            'persistent_indices': self.persistent_indices,
            'attention_history': self.attention_history,
            'importance_scores': self.importance_scores,
            'layer_count': self.layer_count
        }

    def load_state(self, state: Dict) -> None:
        """Load persistence state."""
        self.persistent_indices = state['persistent_indices']
        self.attention_history = state['attention_history']
        self.importance_scores = state['importance_scores']
        self.layer_count = state['layer_count']