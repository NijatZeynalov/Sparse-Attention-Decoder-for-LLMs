import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from src.attention import SparseAttention
from src.core.planner import ReasoningPlanner
from src.utils.logger import get_logger
from src.config.config import AttentionConfig, ModelConfig

logger = get_logger(__name__)


def main():
    """Basic example of using the Sparse Attention Decoder with Llama 2 7B."""
    try:
        # Initialize configurations
        attention_config = AttentionConfig(
            top_k=64,
            attention_window=1024,
            position_persistence=True,
            recompute_threshold=0.3
        )

        model_config = ModelConfig(
            model_name="meta-llama/Llama-2-7b",
            dtype="float16",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load model and tokenizer
        logger.info("Loading Llama 2 7B model...")
        tokenizer = LlamaTokenizer.from_pretrained(model_config.model_name)
        model = LlamaForCausalLM.from_pretrained(
            model_config.model_name,
            torch_dtype=torch.float16 if model_config.dtype == "float16" else torch.float32,
            device_map="auto"
        )

        # Initialize sparse attention
        logger.info("Initializing sparse attention...")
        sparse_attention = SparseAttention(
            config=attention_config,
            hidden_size=model.config.hidden_size,
            num_attention_heads=model.config.num_attention_heads,
            max_position_embeddings=model.config.max_position_embeddings
        )

        # Replace model's attention with sparse attention
        model.model.layers[0].self_attn = sparse_attention

        # Example prompts
        prompts = [
            "Explain the theory of relativity in simple terms:",
            "Write a story about a time traveler who visits ancient Rome:",
            "Describe the process of photosynthesis step by step:"
        ]

        # Generate text with sparse attention
        logger.info("Generating text with sparse attention...")
        for prompt in prompts:
            # Tokenize input
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(model.device)

            # Generate with sparse attention
            outputs = model.generate(
                inputs.input_ids,
                max_length=512,
                num_beams=4,
                temperature=0.7,
                no_repeat_ngram_size=3,
                do_sample=True
            )

            # Decode and print results
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"\nPrompt: {prompt}\n")
            print(f"Generated Text:\n{generated_text}\n")
            print("-" * 80)

        logger.info("Text generation completed successfully")

    except Exception as e:
        logger.error(f"Error in basic usage example: {str(e)}")
        raise


if __name__ == "__main__":
    main()