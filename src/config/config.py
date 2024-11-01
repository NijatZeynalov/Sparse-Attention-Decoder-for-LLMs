from dataclasses import dataclass
from typing import Optional, List, Dict

@dataclass
class AttentionConfig:
    """Configuration for sparse attention mechanism"""
    top_k: int = 64  # Number of tokens to select
    attention_window: int = 1024  # Local attention window size
    position_persistence: bool = True
    min_attention_score: float = 0.1
    recompute_threshold: float = 0.3
    use_token_reselection: bool = True
    reselection_interval: int = 4  # Layers between reselections

@dataclass
class CacheConfig:
    """Configuration for KV cache management"""
    max_cache_size: int = 16384
    correction_steps: int = 1000
    cache_decay_rate: float = 0.95
    min_cache_refresh_ratio: float = 0.2
    adaptive_correction: bool = True

@dataclass
class ModelConfig:
    """Configuration for model settings"""
    model_name: str = "meta-llama/Llama-2-7b"
    dtype: str = "float16"
    device: str = "cuda"
    max_sequence_length: int = 4096
    batch_size: int = 1
    gradient_checkpointing: bool = False

@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking"""
    num_samples: int = 100
    sequence_lengths: List[int] = None
    metrics: List[str] = None
    seed: int = 42
    num_runs: int = 3
    warmup_steps: int = 5

    def __post_init__(self):
        if self.sequence_lengths is None:
            self.sequence_lengths = [512, 1024, 2048, 4096]
        if self.metrics is None:
            self.metrics = ["latency", "memory", "perplexity", "accuracy"]

@dataclass
class LogConfig:
    """Configuration for logging"""
    log_level: str = "INFO"
    use_wandb: bool = False
    wandb_project: str = "sparse-attention"
    log_interval: int = 100
    profile_memory: bool = True