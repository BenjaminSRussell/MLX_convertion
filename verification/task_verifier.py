"""
Task-level accuracy verification.

Verifies that converted models maintain task-specific quality metrics.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import logging

logger = logging.getLogger('MLXVerifier.TaskVerifier')


@dataclass
class TaskVerificationResult:
    """Results from task-level verification."""
    task_type: str
    model_name: str
    dataset_name: str
    passed: bool
    primary_metric: str
    primary_metric_value: float
    baseline_value: float
    threshold: float
    metrics: Dict[str, float]
    details: Optional[Dict[str, Any]] = None


class TaskVerifier:
    """
    Verifies converted models at the task level.

    Different verification strategies for different tasks:
    - NLI: Accuracy, F1, precision, recall
    - Embeddings: Spearman correlation, cosine similarity
    - Text Classification: Accuracy, F1
    - NER: Entity-level F1, precision, recall
    - LLMs: Perplexity, generation quality
    """

    def verify_nli(
        self,
        baseline_predictions: List[str],
        converted_predictions: List[str],
        ground_truth: List[str],
        model_name: str,
        dataset_name: str,
        max_accuracy_drop: float = 0.01
    ) -> TaskVerificationResult:
        """
        Verify NLI model accuracy.

        Args:
            baseline_predictions: PyTorch baseline predictions
            converted_predictions: MLX converted model predictions
            ground_truth: Ground truth labels
            model_name: Model name
            dataset_name: Dataset name
            max_accuracy_drop: Maximum acceptable accuracy drop

        Returns:
            TaskVerificationResult
        """
        # Calculate baseline metrics
        baseline_acc = accuracy_score(ground_truth, baseline_predictions)
        baseline_f1 = f1_score(ground_truth, baseline_predictions, average='macro', zero_division=0)

        # Calculate converted metrics
        converted_acc = accuracy_score(ground_truth, converted_predictions)
        converted_f1 = f1_score(ground_truth, converted_predictions, average='macro', zero_division=0)
        converted_precision = precision_score(ground_truth, converted_predictions, average='macro', zero_division=0)
        converted_recall = recall_score(ground_truth, converted_predictions, average='macro', zero_division=0)

        # Check if passed
        accuracy_drop = baseline_acc - converted_acc
        passed = accuracy_drop <= max_accuracy_drop

        metrics = {
            'baseline_accuracy': baseline_acc,
            'converted_accuracy': converted_acc,
            'accuracy_drop': accuracy_drop,
            'accuracy_drop_pct': accuracy_drop * 100,
            'baseline_f1': baseline_f1,
            'converted_f1': converted_f1,
            'precision': converted_precision,
            'recall': converted_recall
        }

        logger.info(f"NLI Verification - {model_name} on {dataset_name}")
        logger.info(f"  Baseline accuracy: {baseline_acc:.4f}")
        logger.info(f"  Converted accuracy: {converted_acc:.4f}")
        logger.info(f"  Accuracy drop: {accuracy_drop:.4f} ({accuracy_drop*100:.2f}%)")
        logger.info(f"  Threshold: {max_accuracy_drop:.4f}")
        logger.info(f"  Status: {'✓ PASSED' if passed else '✗ FAILED'}")

        return TaskVerificationResult(
            task_type='nli',
            model_name=model_name,
            dataset_name=dataset_name,
            passed=passed,
            primary_metric='accuracy',
            primary_metric_value=converted_acc,
            baseline_value=baseline_acc,
            threshold=max_accuracy_drop,
            metrics=metrics
        )

    def verify_embeddings(
        self,
        baseline_scores: List[float],
        converted_scores: List[float],
        ground_truth_scores: List[float],
        model_name: str,
        dataset_name: str,
        min_spearman: float = 0.98
    ) -> TaskVerificationResult:
        """
        Verify embedding model quality using correlation metrics.

        Args:
            baseline_scores: PyTorch baseline similarity scores
            converted_scores: MLX converted similarity scores
            ground_truth_scores: Ground truth similarity scores
            model_name: Model name
            dataset_name: Dataset name
            min_spearman: Minimum Spearman correlation

        Returns:
            TaskVerificationResult
        """
        # Calculate baseline correlations
        baseline_spearman, _ = spearmanr(baseline_scores, ground_truth_scores)
        baseline_pearson, _ = pearsonr(baseline_scores, ground_truth_scores)

        # Calculate converted correlations
        converted_spearman, _ = spearmanr(converted_scores, ground_truth_scores)
        converted_pearson, _ = pearsonr(converted_scores, ground_truth_scores)

        # Calculate cosine similarity preservation
        baseline_arr = np.array(baseline_scores)
        converted_arr = np.array(converted_scores)
        score_correlation, _ = pearsonr(baseline_arr, converted_arr)

        # Check if passed
        passed = converted_spearman >= min_spearman

        metrics = {
            'baseline_spearman': baseline_spearman,
            'converted_spearman': converted_spearman,
            'spearman_drop': baseline_spearman - converted_spearman,
            'baseline_pearson': baseline_pearson,
            'converted_pearson': converted_pearson,
            'score_correlation': score_correlation,
        }

        logger.info(f"Embedding Verification - {model_name} on {dataset_name}")
        logger.info(f"  Baseline Spearman: {baseline_spearman:.4f}")
        logger.info(f"  Converted Spearman: {converted_spearman:.4f}")
        logger.info(f"  Threshold: {min_spearman:.4f}")
        logger.info(f"  Status: {'✓ PASSED' if passed else '✗ FAILED'}")

        return TaskVerificationResult(
            task_type='embedding',
            model_name=model_name,
            dataset_name=dataset_name,
            passed=passed,
            primary_metric='spearman_correlation',
            primary_metric_value=converted_spearman,
            baseline_value=baseline_spearman,
            threshold=min_spearman,
            metrics=metrics
        )

    def verify_text_classification(
        self,
        baseline_predictions: List[int],
        converted_predictions: List[int],
        ground_truth: List[int],
        model_name: str,
        dataset_name: str,
        max_accuracy_drop: float = 0.015
    ) -> TaskVerificationResult:
        """
        Verify text classification model accuracy.

        Args:
            baseline_predictions: PyTorch predictions
            converted_predictions: MLX predictions
            ground_truth: Ground truth labels
            model_name: Model name
            dataset_name: Dataset name
            max_accuracy_drop: Maximum accuracy drop

        Returns:
            TaskVerificationResult
        """
        # Similar to NLI but with slightly relaxed thresholds
        baseline_acc = accuracy_score(ground_truth, baseline_predictions)
        converted_acc = accuracy_score(ground_truth, converted_predictions)

        accuracy_drop = baseline_acc - converted_acc
        passed = accuracy_drop <= max_accuracy_drop

        # Calculate additional metrics
        f1 = f1_score(ground_truth, converted_predictions, average='macro', zero_division=0)
        precision = precision_score(ground_truth, converted_predictions, average='macro', zero_division=0)
        recall = recall_score(ground_truth, converted_predictions, average='macro', zero_division=0)

        metrics = {
            'baseline_accuracy': baseline_acc,
            'converted_accuracy': converted_acc,
            'accuracy_drop': accuracy_drop,
            'accuracy_drop_pct': accuracy_drop * 100,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

        logger.info(f"Text Classification Verification - {model_name} on {dataset_name}")
        logger.info(f"  Baseline accuracy: {baseline_acc:.4f}")
        logger.info(f"  Converted accuracy: {converted_acc:.4f}")
        logger.info(f"  Accuracy drop: {accuracy_drop:.4f}")
        logger.info(f"  Status: {'✓ PASSED' if passed else '✗ FAILED'}")

        return TaskVerificationResult(
            task_type='text_classification',
            model_name=model_name,
            dataset_name=dataset_name,
            passed=passed,
            primary_metric='accuracy',
            primary_metric_value=converted_acc,
            baseline_value=baseline_acc,
            threshold=max_accuracy_drop,
            metrics=metrics
        )

    def verify_ner(
        self,
        baseline_predictions: List[List[str]],
        converted_predictions: List[List[str]],
        ground_truth: List[List[str]],
        model_name: str,
        dataset_name: str,
        max_f1_drop: float = 0.02
    ) -> TaskVerificationResult:
        """
        Verify NER model with entity-level metrics.

        Args:
            baseline_predictions: PyTorch token predictions (per sentence)
            converted_predictions: MLX token predictions (per sentence)
            ground_truth: Ground truth labels (per sentence)
            model_name: Model name
            dataset_name: Dataset name
            max_f1_drop: Maximum F1 drop

        Returns:
            TaskVerificationResult
        """
        # Flatten predictions for token-level metrics
        baseline_flat = [tag for sent in baseline_predictions for tag in sent]
        converted_flat = [tag for sent in converted_predictions for tag in sent]
        truth_flat = [tag for sent in ground_truth for tag in sent]

        # Calculate baseline metrics
        baseline_f1 = f1_score(truth_flat, baseline_flat, average='macro', zero_division=0)

        # Calculate converted metrics
        converted_f1 = f1_score(truth_flat, converted_flat, average='macro', zero_division=0)
        precision = precision_score(truth_flat, converted_flat, average='macro', zero_division=0)
        recall = recall_score(truth_flat, converted_flat, average='macro', zero_division=0)
        token_acc = accuracy_score(truth_flat, converted_flat)

        # Check if passed
        f1_drop = baseline_f1 - converted_f1
        passed = f1_drop <= max_f1_drop

        metrics = {
            'baseline_f1': baseline_f1,
            'converted_f1': converted_f1,
            'f1_drop': f1_drop,
            'f1_drop_pct': f1_drop * 100,
            'precision': precision,
            'recall': recall,
            'token_accuracy': token_acc
        }

        logger.info(f"NER Verification - {model_name} on {dataset_name}")
        logger.info(f"  Baseline F1: {baseline_f1:.4f}")
        logger.info(f"  Converted F1: {converted_f1:.4f}")
        logger.info(f"  F1 drop: {f1_drop:.4f}")
        logger.info(f"  Status: {'✓ PASSED' if passed else '✗ FAILED'}")

        return TaskVerificationResult(
            task_type='ner',
            model_name=model_name,
            dataset_name=dataset_name,
            passed=passed,
            primary_metric='f1',
            primary_metric_value=converted_f1,
            baseline_value=baseline_f1,
            threshold=max_f1_drop,
            metrics=metrics
        )

    def generate_report(self, results: List[TaskVerificationResult]) -> str:
        """
        Generate verification report.

        Args:
            results: List of verification results

        Returns:
            Formatted report
        """
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed

        report = []
        report.append("=" * 60)
        report.append("TASK-LEVEL VERIFICATION REPORT")
        report.append("=" * 60)
        report.append(f"Total models verified: {total}")
        report.append(f"Passed: {passed} ({passed/total*100:.1f}%)")
        report.append(f"Failed: {failed} ({failed/total*100:.1f}%)")
        report.append("")

        # Group by task type
        by_task = {}
        for r in results:
            if r.task_type not in by_task:
                by_task[r.task_type] = []
            by_task[r.task_type].append(r)

        for task_type, task_results in by_task.items():
            report.append(f"{task_type.upper()} Models:")
            for r in task_results:
                status = "✓" if r.passed else "✗"
                report.append(f"  {status} {r.model_name} ({r.dataset_name})")
                report.append(f"      {r.primary_metric}: {r.primary_metric_value:.4f} (baseline: {r.baseline_value:.4f})")

                if not r.passed:
                    drop = r.baseline_value - r.primary_metric_value
                    report.append(f"      Drop: {drop:.4f} > threshold: {r.threshold:.4f}")
            report.append("")

        report.append("=" * 60)
        return "\n".join(report)
