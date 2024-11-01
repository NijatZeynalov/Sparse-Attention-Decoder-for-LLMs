import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict
from torch.nn.functional import cross_entropy
from transformers import PreTrainedTokenizer
import evaluate
from ..config.config import BenchmarkConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


class AccuracyBenchmark:
    """
    Benchmarks accuracy and quality metrics for sparse attention implementation.
    Focuses on comparing sparse attention against baseline for various NLP tasks.
    """

    def __init__(
            self,
            config: BenchmarkConfig,
            tokenizer: PreTrainedTokenizer
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.metrics = {
            'rouge': evaluate.load('rouge'),
            'bertscore': evaluate.load('bertscore'),
            'perplexity': evaluate.load('perplexity')
        }
        self.results_history = defaultdict(list)

    def run_comparison(
            self,
            sparse_model: nn.Module,
            baseline_model: nn.Module,
            evaluation_data: Dict[str, torch.Tensor],
            task_configs: Optional[Dict] = None
    ) -> Dict:
        """
        Run comparative evaluation between sparse and baseline models.

        Args:
            sparse_model: Model with sparse attention
            baseline_model: Model with standard attention
            evaluation_data: Dict of evaluation datasets
            task_configs: Optional configurations for specific tasks

        Returns:
            Dict: Comparative evaluation results
        """
        results = {
            "sparse": {},
            "baseline": {},
            "comparison": {}
        }

        # Default task configurations if none provided
        if task_configs is None:
            task_configs = {
                "generation": {"max_length": 128, "num_beams": 4},
                "completion": {"top_k": [1, 5, 10]},
                "long_context": {"context_lengths": [512, 1024, 2048, 4096]}
            }

        logger.info("Starting accuracy comparison evaluation")

        for task_name, data in evaluation_data.items():
            logger.info(f"Evaluating task: {task_name}")

            # Evaluate sparse model
            sparse_results = self._evaluate_model(
                sparse_model,
                data,
                task_name,
                task_configs.get(task_name, {})
            )
            results["sparse"][task_name] = sparse_results

            # Evaluate baseline model
            baseline_results = self._evaluate_model(
                baseline_model,
                data,
                task_name,
                task_configs.get(task_name, {})
            )
            results["baseline"][task_name] = baseline_results

            # Compare results
            results["comparison"][task_name] = self._compare_results(
                sparse_results,
                baseline_results
            )

        return results

    def _evaluate_model(
            self,
            model: nn.Module,
            data: Dict[str, torch.Tensor],
            task_name: str,
            task_config: Dict
    ) -> Dict:
        """
        Evaluate model on specific task.
        """
        if task_name == "generation":
            return self._evaluate_generation(model, data, task_config)
        elif task_name == "completion":
            return self._evaluate_completion(model, data, task_config)
        elif task_name == "long_context":
            return self._evaluate_long_context(model, data, task_config)
        else:
            raise ValueError(f"Unsupported task: {task_name}")

    def _evaluate_generation(
            self,
            model: nn.Module,
            data: Dict[str, torch.Tensor],
            config: Dict
    ) -> Dict:
        """
        Evaluate text generation quality.
        """
        results = defaultdict(list)

        input_ids = data["input_ids"]
        target_texts = data["target_texts"]

        with torch.no_grad():
            # Generate text
            generated_ids = model.generate(
                input_ids,
                max_length=config.get("max_length", 128),
                num_beams=config.get("num_beams", 4),
                early_stopping=True
            )

            # Decode generated text
            generated_texts = self.tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )

            # Calculate metrics
            rouge_scores = self.metrics['rouge'].compute(
                predictions=generated_texts,
                references=target_texts
            )

            bert_scores = self.metrics['bertscore'].compute(
                predictions=generated_texts,
                references=target_texts,
                lang="en"
            )

            # Calculate perplexity
            perplexity = self._calculate_perplexity(model, data)

            # Compile results
            metrics = {
                "perplexity": perplexity,
                "rouge1": rouge_scores["rouge1"],
                "rouge2": rouge_scores["rouge2"],
                "rougeL": rouge_scores["rougeL"],
                "bertscore_f1": np.mean(bert_scores["f1"])
            }

            return metrics

    def _evaluate_completion(
            self,
            model: nn.Module,
            data: Dict[str, torch.Tensor],
            config: Dict
    ) -> Dict:
        """
        Evaluate completion task accuracy.
        """
        results = defaultdict(list)
        k_values = config.get("top_k", [1, 5, 10])

        input_ids = data["input_ids"]
        target_ids = data["target_ids"]

        with torch.no_grad():
            # Get model predictions
            outputs = model(input_ids)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs

            # Calculate accuracy metrics
            for k in k_values:
                top_k_acc = self._calculate_top_k_accuracy(logits, target_ids, k)
                results[f"top_{k}_accuracy"].append(top_k_acc)

            # Calculate mean rank
            mean_rank = self._calculate_mean_rank(logits, target_ids)
            results["mean_rank"].append(mean_rank)

        return {
            metric: np.mean(values)
            for metric, values in results.items()
        }

    def _evaluate_long_context(
            self,
            model: nn.Module,
            data: Dict[str, torch.Tensor],
            config: Dict
    ) -> Dict:
        """
        Evaluate performance on long context understanding.
        """
        results = defaultdict(dict)
        context_lengths = config.get("context_lengths", [512, 1024, 2048, 4096])

        for length in context_lengths:
            # Prepare data for current context length
            truncated_data = self._truncate_data(data, length)

            # Evaluate attention patterns
            attention_metrics = self._analyze_attention_patterns(
                model,
                truncated_data
            )

            # Calculate context retention
            retention_score = self._calculate_context_retention(
                model,
                truncated_data
            )

            results[f"context_{length}"] = {
                "attention_coverage": attention_metrics["coverage"],
                "attention_entropy": attention_metrics["entropy"],
                "context_retention": retention_score
            }

        return dict(results)

    def _calculate_perplexity(
            self,
            model: nn.Module,
            data: Dict[str, torch.Tensor]
    ) -> float:
        """Calculate model perplexity."""
        return float(self.metrics['perplexity'].compute(
            model=model,
            input_ids=data["input_ids"]
        )["perplexity"])

    def _calculate_top_k_accuracy(
            self,
            logits: torch.Tensor,
            targets: torch.Tensor,
            k: int
    ) -> float:
        """Calculate top-k accuracy."""
        top_k_preds = torch.topk(logits, k, dim=-1).indices
        targets_expanded = targets.unsqueeze(-1).expand_as(top_k_preds)
        correct = (top_k_preds == targets_expanded).any(dim=-1)
        return float(correct.float().mean().item())

    def _calculate_mean_rank(
            self,
            logits: torch.Tensor,
            targets: torch.Tensor
    ) -> float:
        """Calculate mean rank of correct tokens."""
        ranks = torch.argsort(logits, dim=-1, descending=True)
        target_ranks = torch.where(ranks == targets.unsqueeze(-1))[1] + 1
        return float(target_ranks.float().mean().item())

    def _analyze_attention_patterns(
            self,
            model: nn.Module,
            data: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Analyze attention pattern metrics."""
        with torch.no_grad():
            outputs = model(data["input_ids"], output_attentions=True)
            attentions = outputs.attentions if hasattr(outputs, 'attentions') else None

            if attentions is None:
                return {"coverage": 0.0, "entropy": 0.0}

            # Calculate attention coverage
            coverage = torch.mean(torch.sum(attentions[-1] > 0.1, dim=-1).float())

            # Calculate attention entropy
            entropy = -torch.mean(
                torch.sum(attentions[-1] * torch.log(attentions[-1] + 1e-9), dim=-1)
            )

            return {
                "coverage": float(coverage.item()),
                "entropy": float(entropy.item())
            }

    def _calculate_context_retention(
            self,
            model: nn.Module,
            data: Dict[str, torch.Tensor]
    ) -> float:
        """Calculate context retention score."""
        with torch.no_grad():
            # Generate with full context
            full_output = model.generate(
                data["input_ids"],
                max_length=50,
                num_beams=1
            )

            # Generate with partial context
            partial_input = data["input_ids"][:, :data["input_ids"].size(1) // 2]
            partial_output = model.generate(
                partial_input,
                max_length=50,
                num_beams=1
            )

            # Compare outputs using ROUGE
            full_text = self.tokenizer.batch_decode(full_output)
            partial_text = self.tokenizer.batch_decode(partial_output)

            rouge_scores = self.metrics['rouge'].compute(
                predictions=partial_text,
                references=full_text
            )

            return float(rouge_scores["rougeL"])

    @staticmethod
    def _compare_results(
            sparse_results: Dict,
            baseline_results: Dict
    ) -> Dict:
        """Compare sparse and baseline results."""
        comparison = {}

        for metric in sparse_results:
            if isinstance(sparse_results[metric], (int, float)):
                rel_diff = (
                    (sparse_results[metric] - baseline_results[metric]) /
                    baseline_results[metric] * 100
                    if baseline_results[metric] != 0 else 0
                )
                comparison[f"{metric}_relative_diff"] = rel_diff

        return comparison

    def get_summary_statistics(self) -> Dict:
        """Get summary statistics of all evaluations."""
        summary = {}

        for task, results in self.results_history.items():
            task_summary = {
                "mean": np.mean(results),
                "std": np.std(results),
                "min": np.min(results),
                "max": np.max(results)
            }
            summary[task] = task_summary

        return summary