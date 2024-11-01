import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union
import time
import psutil
import numpy as np
from dataclasses import dataclass
from torch.cuda import Event
from ..config.config import BenchmarkConfig
from ..utils.logger import get_logger
from ..utils.metrics import track_cuda_memory

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    latency_ms: float
    throughput: float
    memory_used_mb: float
    peak_memory_mb: float
    cuda_memory_mb: Optional[float]
    cache_hit_rate: float
    tokens_per_second: float


class PerformanceBenchmark:
    """
    Benchmarks performance metrics for sparse attention implementation.
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results = {}
        self.baseline_results = {}

        # Initialize CUDA events for timing
        self.start_event = Event(enable_timing=True)
        self.end_event = Event(enable_timing=True)
        self.cuda_available = torch.cuda.is_available()

    def run_comparison(
            self,
            sparse_model: nn.Module,
            baseline_model: nn.Module,
            test_sequences: List[torch.Tensor],
            sequence_lengths: Optional[List[int]] = None,
    ) -> Dict:
        """
        Run comparative benchmark between sparse and baseline models.

        Args:
            sparse_model: Model with sparse attention
            baseline_model: Model with standard attention
            test_sequences: Input sequences for testing
            sequence_lengths: Optional specific sequence lengths to test

        Returns:
            Dict: Comparative benchmark results
        """
        logger.info("Starting performance comparison benchmark")

        if sequence_lengths is None:
            sequence_lengths = self.config.sequence_lengths

        comparative_results = {
            "sparse": {},
            "baseline": {},
            "comparison": {}
        }

        # Run benchmarks for each sequence length
        for seq_len in sequence_lengths:
            logger.info(f"Benchmarking sequence length: {seq_len}")

            # Prepare test data for current sequence length
            test_data = self._prepare_test_data(test_sequences, seq_len)

            # Benchmark sparse model
            sparse_metrics = self._benchmark_model(
                sparse_model,
                test_data,
                "sparse"
            )
            comparative_results["sparse"][seq_len] = sparse_metrics

            # Benchmark baseline model
            baseline_metrics = self._benchmark_model(
                baseline_model,
                test_data,
                "baseline"
            )
            comparative_results["baseline"][seq_len] = baseline_metrics

            # Compare results
            comparative_results["comparison"][seq_len] = self._compare_metrics(
                sparse_metrics,
                baseline_metrics
            )

        return comparative_results

    def _benchmark_model(
            self,
            model: nn.Module,
            test_data: torch.Tensor,
            model_type: str
    ) -> PerformanceMetrics:
        """
        Benchmark a single model.

        Args:
            model: Model to benchmark
            test_data: Input test data
            model_type: Type of model ("sparse" or "baseline")

        Returns:
            PerformanceMetrics: Collected performance metrics
        """
        metrics = []

        # Warmup runs
        logger.info(f"Performing warmup runs for {model_type} model")
        for _ in range(self.config.warmup_steps):
            self._run_inference(model, test_data)

        # Actual benchmark runs
        logger.info(f"Running {self.config.num_runs} benchmark iterations")
        for run in range(self.config.num_runs):
            # Clear cache before each run
            if self.cuda_available:
                torch.cuda.empty_cache()

            try:
                run_metrics = self._measure_single_run(model, test_data)
                metrics.append(run_metrics)
            except Exception as e:
                logger.error(f"Error in benchmark run {run}: {str(e)}")
                continue

        # Aggregate metrics
        return self._aggregate_metrics(metrics)

    def _measure_single_run(
            self,
            model: nn.Module,
            test_data: torch.Tensor
    ) -> PerformanceMetrics:
        """Measure performance for a single run."""
        start_memory = self._get_memory_usage()

        # Start timing
        if self.cuda_available:
            self.start_event.record()
            start_cuda_memory = torch.cuda.memory_allocated()
        start_cpu_time = time.perf_counter()

        # Run inference
        output = self._run_inference(model, test_data)

        # End timing
        if self.cuda_available:
            self.end_event.record()
            end_cuda_memory = torch.cuda.memory_allocated()
            torch.cuda.synchronize()
            gpu_time_ms = self.start_event.elapsed_time(self.end_event)
        else:
            gpu_time_ms = None

        end_cpu_time = time.perf_counter()
        end_memory = self._get_memory_usage()

        # Calculate metrics
        cpu_time_ms = (end_cpu_time - start_cpu_time) * 1000
        latency = gpu_time_ms if gpu_time_ms is not None else cpu_time_ms

        throughput = (test_data.size(1) / latency) * 1000  # tokens per second
        memory_used = end_memory - start_memory

        cuda_memory_used = None
        if self.cuda_available:
            cuda_memory_used = (end_cuda_memory - start_cuda_memory) / 1024 ** 2  # MB

        # Get cache stats if available
        cache_hit_rate = 0.0
        if hasattr(model, "get_cache_stats"):
            cache_stats = model.get_cache_stats()
            cache_hit_rate = cache_stats.get("hit_rate", 0.0)

        return PerformanceMetrics(
            latency_ms=latency,
            throughput=throughput,
            memory_used_mb=memory_used,
            peak_memory_mb=self._get_peak_memory(),
            cuda_memory_mb=cuda_memory_used,
            cache_hit_rate=cache_hit_rate,
            tokens_per_second=throughput
        )

    def _run_inference(self, model: nn.Module, test_data: torch.Tensor) -> torch.Tensor:
        """Run model inference."""
        with torch.no_grad():
            return model(test_data)

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 ** 2

    def _get_peak_memory(self) -> float:
        """Get peak memory usage in MB."""
        if self.cuda_available:
            return torch.cuda.max_memory_allocated() / 1024 ** 2
        return psutil.Process().memory_info().peak_wset / 1024 ** 2

    @staticmethod
    def _aggregate_metrics(metrics: List[PerformanceMetrics]) -> PerformanceMetrics:
        """Aggregate metrics across multiple runs."""
        return PerformanceMetrics(
            latency_ms=np.mean([m.latency_ms for m in metrics]),
            throughput=np.mean([m.throughput for m in metrics]),
            memory_used_mb=np.mean([m.memory_used_mb for m in metrics]),
            peak_memory_mb=max(m.peak_memory_mb for m in metrics),
            cuda_memory_mb=np.mean([m.cuda_memory_mb for m in metrics if m.cuda_memory_mb is not None]),
            cache_hit_rate=np.mean([m.cache_hit_rate for m in metrics]),
            tokens_per_second=np.mean([m.tokens_per_second for m in metrics])
        )

    @staticmethod
    def _prepare_test_data(
            sequences: List[torch.Tensor],
            target_length: int
    ) -> torch.Tensor:
        """Prepare test data for benchmarking."""
        if len(sequences) == 0:
            raise ValueError("Empty test sequence list")

        # Pad or truncate sequences to target length
        processed_sequences = []
        for seq in sequences:
            if seq.size(1) < target_length:
                padding = torch.zeros(
                    seq.size(0),
                    target_length - seq.size(1),
                    dtype=seq.dtype,
                    device=seq.device
                )
                seq = torch.cat([seq, padding], dim=1)
            else:
                seq = seq[:, :target_length]
            processed_sequences.append(seq)

        return torch.cat(processed_sequences, dim=0)

    @staticmethod
    def _compare_metrics(
            sparse_metrics: PerformanceMetrics,
            baseline_metrics: PerformanceMetrics
    ) -> Dict:
        """Compare sparse and baseline metrics."""
        return {
            "latency_reduction": (
                    (baseline_metrics.latency_ms - sparse_metrics.latency_ms) /
                    baseline_metrics.latency_ms * 100
            ),
            "throughput_improvement": (
                    (sparse_metrics.throughput - baseline_metrics.throughput) /
                    baseline_metrics.throughput * 100
            ),
            "memory_reduction": (
                    (baseline_metrics.memory_used_mb - sparse_metrics.memory_used_mb) /
                    baseline_metrics.memory_used_mb * 100
            ),
            "relative_efficiency": (
                    sparse_metrics.tokens_per_second / baseline_metrics.tokens_per_second
            )
        }