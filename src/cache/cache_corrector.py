import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple, List
import math
from ..config.config import CacheConfig
from ..utils.logger import get_logger
from .kv_cache import KeyValueCache

logger = get_logger(__name__)


class CacheCorrector:
    """
    Implements cache correction mechanisms to prevent distribution shifts in long sequences.
    Handles the periodic refresh and adjustment of the key-value cache to maintain
    consistency and prevent degradation over long sequences.
    """

    def __init__(self, config: CacheConfig):
        self.config = config
        self.correction_counter = 0
        self.distribution_tracker = DistributionTracker()
        self.last_correction_step = 0
        self.correction_history = []
        self.drift_threshold = 0.15  # Threshold for significant distribution drift

    def correct_cache(
            self,
            cache: KeyValueCache,
            current_hidden_states: torch.Tensor,
            layer_idx: int,
            current_step: int
    ) -> None:
        """
        Apply cache correction based on distribution drift analysis.

        Args:
            cache: KeyValueCache instance
            current_hidden_states: Current layer hidden states
            layer_idx: Current layer index
            current_step: Current generation step
        """
        try:
            if not self._needs_correction(current_step, layer_idx):
                return

            logger.info(f"Performing cache correction for layer {layer_idx} at step {current_step}")

            # Get cached states
            cache_entry = cache.cache.get(layer_idx)
            if cache_entry is None:
                return

            # Compute distribution metrics
            current_stats = self._compute_distribution_metrics(current_hidden_states)
            cached_stats = self._compute_distribution_metrics(
                cache_entry['key_states'].reshape(-1, cache_entry['key_states'].size(-1))
            )

            # Check for distribution drift
            drift_detected = self._detect_distribution_drift(current_stats, cached_stats)

            if drift_detected:
                self._apply_correction(cache, layer_idx, current_stats)
                self._record_correction(layer_idx, current_step, current_stats, cached_stats)

        except Exception as e:
            logger.error(f"Error in cache correction: {str(e)}")

    def _needs_correction(self, current_step: int, layer_idx: int) -> bool:
        """
        Determine if cache correction is needed based on steps and layer.
        """
        steps_since_last = current_step - self.last_correction_step

        # Check basic interval
        if steps_since_last < self.config.correction_steps:
            return False

        # Check layer-specific conditions
        recent_corrections = [
            c for c in self.correction_history[-10:]
            if c['layer_idx'] == layer_idx
        ]

        if recent_corrections:
            avg_drift = sum(c['drift_magnitude'] for c in recent_corrections) / len(recent_corrections)
            return avg_drift > self.drift_threshold

        return True

    def _compute_distribution_metrics(self, tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute key distribution metrics for a tensor.
        """
        # Ensure tensor is 2D for consistent computation
        if tensor.dim() > 2:
            tensor = tensor.reshape(-1, tensor.size(-1))

        metrics = {
            'mean': torch.mean(tensor, dim=0),
            'std': torch.std(tensor, dim=0),
            'norms': torch.norm(tensor, dim=-1),
            'cosine_sim': self._compute_cosine_similarity(tensor)
        }

        return metrics

    def _compute_cosine_similarity(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute average cosine similarity between vectors.
        """
        normalized = tensor / (torch.norm(tensor, dim=-1, keepdim=True) + 1e-8)
        similarity = torch.mm(normalized, normalized.t())
        # Remove self-similarity from diagonal
        similarity = similarity - torch.eye(similarity.size(0), device=tensor.device)
        return similarity.mean()

    def _detect_distribution_drift(
            self,
            current_stats: Dict[str, torch.Tensor],
            cached_stats: Dict[str, torch.Tensor]
    ) -> bool:
        """
        Detect significant distribution drift between current and cached states.
        """
        # Compute normalized differences
        mean_drift = torch.norm(current_stats['mean'] - cached_stats['mean']) / (
                torch.norm(cached_stats['mean']) + 1e-8
        )
        std_drift = torch.norm(current_stats['std'] - cached_stats['std']) / (
                torch.norm(cached_stats['std']) + 1e-8
        )
        norm_drift = abs(current_stats['cosine_sim'] - cached_stats['cosine_sim'])

        # Combine drift metrics
        total_drift = 0.4 * mean_drift + 0.3 * std_drift + 0.3 * norm_drift

        return float(total_drift) > self.drift_threshold

    def _apply_correction(
            self,
            cache: KeyValueCache,
            layer_idx: int,
            current_stats: Dict[str, torch.Tensor]
    ) -> None:
        """
        Apply correction to cached states.
        """
        cache_entry = cache.cache[layer_idx]

        # Compute correction factors
        mean_correction = current_stats['mean'] / (
                torch.mean(cache_entry['key_states'], dim=(0, 1, 2)) + 1e-8
        )
        std_correction = current_stats['std'] / (
                torch.std(cache_entry['key_states'], dim=(0, 1, 2)) + 1e-8
        )

        # Apply corrections with decay
        decay = self.config.cache_decay_rate

        # Correct key states
        cache_entry['key_states'] = (
                cache_entry['key_states'] * mean_correction * decay +
                cache_entry['key_states'] * (1 - decay)
        )

        # Correct value states similarly
        cache_entry['value_states'] = (
                cache_entry['value_states'] * std_correction * decay +
                cache_entry['value_states'] * (1 - decay)
        )

        # Update cache entry
        cache.cache[layer_idx] = cache_entry

    def _record_correction(
            self,
            layer_idx: int,
            step: int,
            current_stats: Dict[str, torch.Tensor],
            cached_stats: Dict[str, torch.Tensor]
    ) -> None:
        """
        Record correction event and its metrics.
        """
        mean_drift = float(torch.norm(
            current_stats['mean'] - cached_stats['mean']
        ) / (torch.norm(cached_stats['mean']) + 1e-8))

        correction_info = {
            'layer_idx': layer_idx,
            'step': step,
            'drift_magnitude': mean_drift,
            'timestamp': torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        }

        self.correction_history.append(correction_info)

        # Keep history bounded
        if len(self.correction_history) > 1000:
            self.correction_history = self.correction_history[-1000:]

        self.last_correction_step = step
        self.correction_counter += 1

    def get_correction_stats(self) -> Dict:
        """
        Get statistics about cache corrections.
        """
        if not self.correction_history:
            return {}

        recent_corrections = self.correction_history[-100:]

        stats = {
            'total_corrections': self.correction_counter,
            'average_drift': sum(c['drift_magnitude'] for c in recent_corrections) / len(recent_corrections),
            'corrections_per_layer': {
                layer_idx: len([c for c in recent_corrections if c['layer_idx'] == layer_idx])
                for layer_idx in set(c['layer_idx'] for c in recent_corrections)
            },
            'last_correction_step': self.last_correction_step
        }

        return stats

    def reset(self) -> None:
        """
        Reset corrector state.
        """
        self.correction_counter = 0
        self.correction_history.clear()
        self.last_correction_step = 0
        self.distribution_tracker = DistributionTracker()


class DistributionTracker:
    """
    Tracks distribution statistics over time.
    """

    def __init__(self):
        self.stats_history = []
        self.max_history = 1000

    def update(self, stats: Dict[str, torch.Tensor]) -> None:
        """
        Update statistics history.
        """
        self.stats_history.append({
            k: v.detach().clone() if isinstance(v, torch.Tensor) else v
            for k, v in stats.items()
        })

        if len(self.stats_history) > self.max_history:
            self.stats_history = self.stats_history[-self.max_history:]

    def get_trend(self, metric: str) -> Optional[torch.Tensor]:
        """
        Get trend for a specific metric.
        """
        if not self.stats_history:
            return None

        values = [stats[metric] for stats in self.stats_history]
        return torch.stack(values)