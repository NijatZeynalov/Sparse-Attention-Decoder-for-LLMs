import torch
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from collections import defaultdict
import time, json
from dataclasses import dataclass
from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class AttentionStats:
    """Container for attention-related statistics."""
    top_k_tokens: int
    sparsity: float
    coverage: float
    entropy: float
    selected_indices: torch.Tensor
    attention_scores: torch.Tensor


class SparseAttentionMetrics:
    """
    Tracks and analyzes metrics for sparse attention mechanism.
    Focuses on attention patterns, efficiency, and performance metrics.
    """

    def __init__(self):
        self.reset()
        self.attention_history = []
        self.throughput_history = []
        self.memory_history = []

    def reset(self):
        """Reset all metrics."""
        self.start_time = time.time()
        self.total_tokens = 0
        self.total_sequences = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.attention_stats = defaultdict(list)
        self.performance_stats = {
            'latency': [],
            'memory': [],
            'throughput': []
        }

    def update_attention_stats(
            self,
            attention_weights: torch.Tensor,
            selected_indices: torch.Tensor,
            layer_idx: int
    ) -> AttentionStats:
        """
        Update attention statistics for a layer.

        Args:
            attention_weights: Attention weight matrix
            selected_indices: Selected token indices
            layer_idx: Current layer index

        Returns:
            AttentionStats: Computed statistics
        """
        try:
            batch_size, num_heads, seq_len, _ = attention_weights.shape

            # Calculate sparsity
            sparsity = self._calculate_sparsity(attention_weights)

            # Calculate coverage
            coverage = self._calculate_coverage(attention_weights, selected_indices)

            # Calculate entropy
            entropy = self._calculate_entropy(attention_weights)

            stats = AttentionStats(
                top_k_tokens=selected_indices.size(-1),
                sparsity=sparsity,
                coverage=coverage,
                entropy=entropy,
                selected_indices=selected_indices.detach(),
                attention_scores=attention_weights.detach()
            )

            self.attention_stats[layer_idx].append(stats)
            self.attention_history.append({
                'layer': layer_idx,
                'sparsity': sparsity,
                'coverage': coverage,
                'entropy': entropy,
                'timestamp': time.time()
            })

            return stats

        except Exception as e:
            logger.error(f"Error updating attention stats: {str(e)}")
            raise

    def update_performance_stats(
            self,
            latency: float,
            memory_used: float,
            num_tokens: int,
            cache_info: Optional[Dict] = None
    ):
        """
        Update performance statistics.

        Args:
            latency: Processing time in seconds
            memory_used: Memory usage in MB
            num_tokens: Number of tokens processed
            cache_info: Optional cache statistics
        """
        try:
            # Update basic stats
            self.total_tokens += num_tokens
            self.total_sequences += 1

            # Update cache stats if provided
            if cache_info:
                self.cache_hits += cache_info.get('hits', 0)
                self.cache_misses += cache_info.get('misses', 0)

            # Calculate throughput
            throughput = num_tokens / latency if latency > 0 else 0

            # Store performance metrics
            self.performance_stats['latency'].append(latency)
            self.performance_stats['memory'].append(memory_used)
            self.performance_stats['throughput'].append(throughput)

            # Store historical data
            self.throughput_history.append({
                'tokens': num_tokens,
                'latency': latency,
                'throughput': throughput,
                'timestamp': time.time()
            })

            self.memory_history.append({
                'memory_used': memory_used,
                'timestamp': time.time()
            })

        except Exception as e:
            logger.error(f"Error updating performance stats: {str(e)}")
            raise

    def get_summary(self) -> Dict:
        """
        Get comprehensive summary of all metrics.

        Returns:
            Dict: Summary statistics
        """
        try:
            total_time = time.time() - self.start_time

            summary = {
                "attention": self._get_attention_summary(),
                "performance": self._get_performance_summary(total_time),
                "cache": self._get_cache_summary(),
                "trends": self._get_trend_analysis()
            }

            return summary

        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return {}

    def _calculate_sparsity(self, attention_weights: torch.Tensor) -> float:
        """Calculate attention sparsity ratio."""
        threshold = 1e-4  # Threshold for considering attention weights significant
        total_elements = attention_weights.numel()
        significant_elements = torch.sum(attention_weights > threshold).item()
        return 1.0 - (significant_elements / total_elements)

    def _calculate_coverage(
            self,
            attention_weights: torch.Tensor,
            selected_indices: torch.Tensor
    ) -> float:
        """Calculate attention coverage of selected tokens."""
        # Gather attention weights for selected tokens
        batch_size, num_heads, _, seq_len = attention_weights.shape
        selected_attention = torch.gather(
            attention_weights,
            -1,
            selected_indices.unsqueeze(1).expand(-1, num_heads, -1)
        )

        # Calculate coverage ratio
        total_attention = attention_weights.sum(dim=-1, keepdim=True)
        selected_attention_sum = selected_attention.sum(dim=-1, keepdim=True)
        coverage = (selected_attention_sum / total_attention).mean().item()

        return coverage

    def _calculate_entropy(self, attention_weights: torch.Tensor) -> float:
        """Calculate attention entropy."""
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        probs = attention_weights + eps
        probs = probs / probs.sum(dim=-1, keepdim=True)

        entropy = -torch.sum(probs * torch.log2(probs), dim=-1)
        return float(entropy.mean().item())

    def _get_attention_summary(self) -> Dict:
        """Get summary of attention statistics."""
        summary = {}

        for layer_idx, stats_list in self.attention_stats.items():
            layer_metrics = {
                'sparsity': np.mean([s.sparsity for s in stats_list]),
                'coverage': np.mean([s.coverage for s in stats_list]),
                'entropy': np.mean([s.entropy for s in stats_list]),
                'top_k': np.mean([s.top_k_tokens for s in stats_list])
            }
            summary[f"layer_{layer_idx}"] = layer_metrics

        return summary

    def _get_performance_summary(self, total_time: float) -> Dict:
        """Get summary of performance statistics."""
        return {
            'total_time': total_time,
            'total_tokens': self.total_tokens,
            'tokens_per_second': self.total_tokens / total_time,
            'average_latency': np.mean(self.performance_stats['latency']),
            'average_memory': np.mean(self.performance_stats['memory']),
            'peak_memory': max(self.performance_stats['memory']),
            'average_throughput': np.mean(self.performance_stats['throughput'])
        }

    def _get_cache_summary(self) -> Dict:
        """Get summary of cache statistics."""
        total_cache_requests = self.cache_hits + self.cache_misses
        hit_rate = (
            self.cache_hits / total_cache_requests
            if total_cache_requests > 0 else 0
        )

        return {
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'hit_rate': hit_rate
        }

    def _get_trend_analysis(self) -> Dict:
        """Analyze trends in metrics over time."""
        if not self.attention_history:
            return {}

        # Calculate moving averages
        window_size = min(len(self.throughput_history), 100)
        if window_size < 2:
            return {}

        recent_throughput = [
            x['throughput'] for x in self.throughput_history[-window_size:]
        ]
        throughput_trend = np.polyfit(
            range(window_size),
            recent_throughput,
            1
        )[0]

        recent_memory = [
            x['memory_used'] for x in self.memory_history[-window_size:]
        ]
        memory_trend = np.polyfit(
            range(window_size),
            recent_memory,
            1
        )[0]

        return {
            'throughput_trend': float(throughput_trend),
            'memory_trend': float(memory_trend)
        }

    def save_metrics(self, filepath: str):
        """Save metrics to file."""
        try:
            summary = self.get_summary()
            with open(filepath, 'w') as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Metrics saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")