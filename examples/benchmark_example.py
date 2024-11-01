import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from src.benchmarks.performance_benchmark import PerformanceBenchmark
from src.benchmarks.accuracy_benchmark import AccuracyBenchmark
from src.attention import SparseAttention
from src.config.config import BenchmarkConfig, AttentionConfig, ModelConfig
from src.utils.logger import get_logger
from src.utils.visualization import AttentionVisualizer

logger = get_logger(__name__)


def load_models(model_config: ModelConfig):
    """Load sparse and baseline models."""
    # Load tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(model_config.model_name)

    # Load baseline model
    baseline_model = LlamaForCausalLM.from_pretrained(
        model_config.model_name,
        torch_dtype=torch.float16 if model_config.dtype == "float16" else torch.float32,
        device_map="auto"
    )

    # Load model with sparse attention
    sparse_model = LlamaForCausalLM.from_pretrained(
        model_config.model_name,
        torch_dtype=torch.float16 if model_config.dtype == "float16" else torch.float32,
        device_map="auto"
    )

    # Initialize and attach sparse attention
    attention_config = AttentionConfig(
        top_k=64,
        attention_window=1024,
        position_persistence=True
    )

    sparse_attention = SparseAttention(
        config=attention_config,
        hidden_size=sparse_model.config.hidden_size,
        num_attention_heads=sparse_model.config.num_attention_heads,
        max_position_embeddings=sparse_model.config.max_position_embeddings
    )

    # Replace attention in first layer (for demonstration)
    sparse_model.model.layers[0].self_attn = sparse_attention

    return tokenizer, baseline_model, sparse_model


def prepare_benchmark_data(tokenizer, num_samples: int = 100):
    """Prepare data for benchmarking."""
    # Example prompts for various tasks
    prompts = [
        "Explain the concept of quantum entanglement in simple terms.",
        "Write a creative story about a robot discovering human emotions.",
        "Describe the water cycle and its importance to Earth's ecosystem.",
        "Analyze the themes in Shakespeare's Hamlet.",
        "What are the main differences between classical and quantum computers?"
    ]

    # Create variations of prompts
    all_prompts = []
    for i in range(num_samples):
        prompt = prompts[i % len(prompts)]
        all_prompts.append(prompt + f" (Variation {i + 1})")

    # Tokenize prompts
    encodings = tokenizer(
        all_prompts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    return {
        "input_ids": encodings.input_ids,
        "attention_mask": encodings.attention_mask,
        "prompts": all_prompts
    }


def main():
    """Run comprehensive benchmark comparison."""
    try:
        # Initialize configurations
        model_config = ModelConfig(
            model_name="meta-llama/Llama-2-7b",
            dtype="float16",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        benchmark_config = BenchmarkConfig(
            num_samples=100,
            sequence_lengths=[128, 256, 512, 1024],
            metrics=["latency", "memory", "perplexity", "accuracy"],
            num_runs=3
        )

        # Load models
        logger.info("Loading models...")
        tokenizer, baseline_model, sparse_model = load_models(model_config)

        # Prepare benchmark data
        logger.info("Preparing benchmark data...")
        benchmark_data = prepare_benchmark_data(tokenizer, benchmark_config.num_samples)

        # Initialize benchmarks
        performance_benchmark = PerformanceBenchmark(benchmark_config)
        accuracy_benchmark = AccuracyBenchmark(benchmark_config, tokenizer)

        # Run performance benchmark
        logger.info("Running performance benchmark...")
        performance_results = performance_benchmark.run_comparison(
            sparse_model=sparse_model,
            baseline_model=baseline_model,
            test_sequences=benchmark_data["input_ids"],
            sequence_lengths=benchmark_config.sequence_lengths
        )

        # Run accuracy benchmark
        logger.info("Running accuracy benchmark...")
        accuracy_results = accuracy_benchmark.evaluate_model(
            model=sparse_model,
            test_dataset=benchmark_data,
            task_type="generation",
            reference_outputs=None  # Can provide reference outputs if available
        )

        # Visualize results
        logger.info("Generating visualizations...")
        visualizer = AttentionVisualizer()

        # Plot performance comparison
        visualizer.plot_metrics_overview({
            "performance": performance_results,
            "accuracy": accuracy_results
        })

        # Print summary
        print("\nPerformance Comparison Summary:")
        print("-" * 40)
        for seq_len in benchmark_config.sequence_lengths:
            comparison = performance_results["comparison"][seq_len]
            print(f"\nSequence Length: {seq_len}")
            print(f"Latency Reduction: {comparison['latency_reduction']:.2f}%")
            print(f"Memory Reduction: {comparison['memory_reduction']:.2f}%")
            print(f"Throughput Improvement: {comparison['throughput_improvement']:.2f}%")

        print("\nAccuracy Metrics:")
        print("-" * 40)
        print(f"Perplexity: {accuracy_results['perplexity']:.2f}")
        print(f"Token Accuracy: {accuracy_results['token_accuracy']:.2f}")
        if 'sequence_accuracy' in accuracy_results:
            print(f"Sequence Accuracy: {accuracy_results['sequence_accuracy']:.2f}")

        logger.info("Benchmark completed successfully")

    except Exception as e:
        logger.error(f"Error in benchmark example: {str(e)}")
        raise


if __name__ == "__main__":
    main()