import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
import numpy as np
from ..config.config import CacheConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CacheStats:
    """Statistics for cache monitoring."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_tokens: int = 0
    memory_used: int = 0


class KeyValueCache:
    """
    Manages the key-value cache for efficient token reuse in sparse attention.
    """

    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache: Dict[int, Dict[str, torch.Tensor]] = {}
        self.cache_age: Dict[int, int] = {}
        self.stats = CacheStats()
        self.current_size = 0
        self.token_frequency = {}
        self.last_access_time = {}

    def update(
            self,
            layer_idx: int,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            token_ids: torch.Tensor,
            position_ids: Optional[torch.Tensor] = None
    ) -> None:
        """
        Update cache with new key-value states.

        Args:
            layer_idx: Layer index
            key_states: Key tensor [batch_size, num_heads, seq_length, head_dim]
            value_states: Value tensor [batch_size, num_heads, seq_length, head_dim]
            token_ids: Token IDs corresponding to the states
            position_ids: Position IDs for the states
        """
        try:
            batch_size, num_heads, seq_length, head_dim = key_states.shape

            # Check cache size and evict if necessary
            required_space = seq_length * head_dim * 2  # For both key and value states
            if self.current_size + required_space > self.config.max_cache_size:
                self._evict_entries(required_space)

            # Update cache
            cache_entry = {
                'key_states': key_states.detach(),
                'value_states': value_states.detach(),
                'token_ids': token_ids.detach(),
                'position_ids': position_ids.detach() if position_ids is not None else None,
                'frequency': torch.ones(seq_length, device=key_states.device),
                'last_access': torch.full((seq_length,), self.stats.total_tokens,
                                          device=key_states.device)
            }

            self.cache[layer_idx] = cache_entry
            self.cache_age[layer_idx] = self.stats.total_tokens
            self.current_size += required_space

            # Update statistics
            self._update_token_statistics(token_ids)

        except Exception as e:
            logger.error(f"Error updating cache: {str(e)}")
            raise

    def get(
            self,
            layer_idx: int,
            token_ids: torch.Tensor,
            position_ids: Optional[torch.Tensor] = None
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Retrieve cached key-value states for given tokens.

        Args:
            layer_idx: Layer index
            token_ids: Token IDs to retrieve
            position_ids: Position IDs for the tokens

        Returns:
            tuple: (cached_keys, cached_values) or (None, None) if not found
        """
        if layer_idx not in self.cache:
            self.stats.misses += 1
            return None, None

        cache_entry = self.cache[layer_idx]
        cached_tokens = cache_entry['token_ids']

        # Find matching tokens
        matches = []
        for i, token_id in enumerate(token_ids):
            mask = (cached_tokens == token_id)
            if position_ids is not None:
                pos_id = position_ids[i]
                mask &= (cache_entry['position_ids'] == pos_id)

            matches.append(mask.nonzero().squeeze(-1))

        if not matches or all(m.numel() == 0 for m in matches):
            self.stats.misses += 1
            return None, None

        # Update access statistics
        self._update_access_stats(layer_idx, matches)

        # Gather matching states
        gathered_keys = []
        gathered_values = []

        for match in matches:
            if match.numel() > 0:
                gathered_keys.append(cache_entry['key_states'][:, :, match[0]])
                gathered_values.append(cache_entry['value_states'][:, :, match[0]])

        if gathered_keys and gathered_values:
            self.stats.hits += 1
            return (
                torch.stack(gathered_keys, dim=2),
                torch.stack(gathered_values, dim=2)
            )

        self.stats.misses += 1
        return None, None

    def _evict_entries(self, required_space: int) -> None:
        """
        Evict cache entries to make space.

        Args:
            required_space: Amount of space needed
        """
        while self.current_size + required_space > self.config.max_cache_size:
            if not self.cache:
                break

            # Score entries for eviction
            scores = self._compute_eviction_scores()
            if not scores:
                break

            # Evict entry with lowest score
            layer_idx = min(scores.items(), key=lambda x: x[1])[0]
            evicted_size = (
                                   self.cache[layer_idx]['key_states'].numel() +
                                   self.cache[layer_idx]['value_states'].numel()
                           ) * 4  # Assuming float32

            del self.cache[layer_idx]
            del self.cache_age[layer_idx]

            self.current_size -= evicted_size
            self.stats.evictions += 1

    def _compute_eviction_scores(self) -> Dict[int, float]:
        """
        Compute scores for cache eviction decisions.

        Returns:
            Dict[int, float]: Layer index to eviction score mapping
        """
        scores = {}
        current_time = self.stats.total_tokens

        for layer_idx, entry in self.cache.items():
            # Combine multiple factors for eviction decision
            frequency_score = torch.mean(entry['frequency']).item()
            recency_score = torch.mean(current_time - entry['last_access']).item()
            age_score = current_time - self.cache_age[layer_idx]

            # Weight the factors
            score = (
                    0.4 * frequency_score +
                    0.4 * (1.0 / (recency_score + 1)) +
                    0.2 * (1.0 / (age_score + 1))
            )

            scores[layer_idx] = score

        return scores

    def _update_token_statistics(self, token_ids: torch.Tensor) -> None:
        """Update token frequency statistics."""
        for token_id in token_ids.unique():
            token_id = token_id.item()
            self.token_frequency[token_id] = self.token_frequency.get(token_id, 0) + 1
            self.last_access_time[token_id] = self.stats.total_tokens

        self.stats.total_tokens += len(token_ids)

    def _update_access_stats(
            self,
            layer_idx: int,
            matches: List[torch.Tensor]
    ) -> None:
        """Update access statistics for cache entries."""
        entry = self.cache[layer_idx]
        current_time = self.stats.total_tokens

        for match in matches:
            if match.numel() > 0:
                idx = match[0].item()
                entry['frequency'][idx] += 1
                entry['last_access'][idx] = current_time

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        hit_rate = (
            self.stats.hits / (self.stats.hits + self.stats.misses)
            if (self.stats.hits + self.stats.misses) > 0
            else 0.0
        )

        return {
            "hit_rate": hit_rate,
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "evictions": self.stats.evictions,
            "current_size": self.current_size,
            "total_tokens": self.stats.total_tokens,
            "unique_tokens": len(self.token_frequency)
        }

    def reset(self) -> None:
        """Reset cache state."""
        self.cache.clear()
        self.cache_age.clear()
        self.current_size = 0
        self.token_frequency.clear()
        self.last_access_time.clear()
        self.stats = CacheStats()