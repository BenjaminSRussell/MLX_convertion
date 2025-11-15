"""
metrics calc utils for model evaluation
"""

from typing import List, Dict, Any
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from scipy.stats import spearmanr, pearsonr
import logging

logger = logging.getLogger('MLX8BitTester.MetricsCalculator')


class MetricsCalculator:
    """calculates all the metrics for evaluation (accuracy, speed, latency, etc)"""

    @staticmethod
    def calculate_accuracy(predictions: List[Any], references: List[Any]) -> float:
        """calculates accuracy"""
        return accuracy_score(references, predictions)

    @staticmethod
    def calculate_f1_score(
        predictions: List[Any],
        references: List[Any],
        average: str = 'macro'
    ) -> float:
        """calculates F1 score"""
        return f1_score(references, predictions, average=average, zero_division=0)

    @staticmethod
    def calculate_precision(
        predictions: List[Any],
        references: List[Any],
        average: str = 'macro'
    ) -> float:
        """calculates precision"""

        Args:
            predictions: List of predicted labels
            references: List of ground truth labels
            average: Averaging method

        Returns:
            Precision score
        """
        return precision_score(references, predictions, average=average, zero_division=0)

    @staticmethod
    def calculate_recall(
        predictions: List[Any],
        references: List[Any],
        average: str = 'macro'
    ) -> float:
        """
        Calculate recall score.

        Args:
            predictions: List of predicted labels
            references: List of ground truth labels
            average: Averaging method

        Returns:
            Recall score
        """
        return recall_score(references, predictions, average=average, zero_division=0)

    @staticmethod
    def calculate_spearman_correlation(
        predictions: List[float],
        references: List[float]
    ) -> float:
        """
        Calculate Spearman rank correlation (for embedding/similarity tasks).

        Args:
            predictions: List of predicted scores
            references: List of ground truth scores

        Returns:
            Spearman correlation coefficient
        """
        corr, _ = spearmanr(predictions, references)
        return corr if not np.isnan(corr) else 0.0

    @staticmethod
    def calculate_pearson_correlation(
        predictions: List[float],
        references: List[float]
    ) -> float:
        """
        Calculate Pearson correlation (for embedding/similarity tasks).

        Args:
            predictions: List of predicted scores
            references: List of ground truth scores

        Returns:
            Pearson correlation coefficient
        """
        corr, _ = pearsonr(predictions, references)
        return corr if not np.isnan(corr) else 0.0

    @staticmethod
    def calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    @staticmethod
    def calculate_qpm(num_samples: int, duration_seconds: float) -> float:
        """
        Calculate Queries Per Minute (QPM).

        Args:
            num_samples: Number of samples processed
            duration_seconds: Time taken in seconds

        Returns:
            Queries per minute
        """
        if duration_seconds <= 0:
            return 0.0
        return (num_samples / duration_seconds) * 60

    @staticmethod
    def calculate_throughput(num_tokens: int, duration_seconds: float) -> float:
        """
        Calculate token throughput (tokens/second).

        Args:
            num_tokens: Total number of tokens processed
            duration_seconds: Time taken in seconds

        Returns:
            Tokens per second
        """
        if duration_seconds <= 0:
            return 0.0
        return num_tokens / duration_seconds

    @staticmethod
    def calculate_latency_percentiles(
        latencies: List[float],
        percentiles: List[int] = [50, 95, 99]
    ) -> Dict[str, float]:
        """
        Calculate latency percentiles.

        Args:
            latencies: List of latency measurements (in seconds)
            percentiles: List of percentiles to calculate

        Returns:
            Dictionary mapping percentile names to values (in milliseconds)
        """
        if not latencies:
            return {f'p{p}': 0.0 for p in percentiles}

        latency_ms = np.array(latencies) * 1000  # Convert to ms
        return {
            f'p{p}': np.percentile(latency_ms, p)
            for p in percentiles
        }

    @staticmethod
    def calculate_memory_stats(memory_samples: List[float]) -> Dict[str, float]:
        """
        Calculate memory usage statistics.

        Args:
            memory_samples: List of memory measurements (in MB)

        Returns:
            Dictionary with avg, peak, and min memory
        """
        if not memory_samples:
            return {'avg': 0.0, 'peak': 0.0, 'min': 0.0}

        return {
            'avg': np.mean(memory_samples),
            'peak': np.max(memory_samples),
            'min': np.min(memory_samples)
        }

    def calculate_all_performance_metrics(
        self,
        predictions: List[Any],
        references: List[Any],
        latencies: List[float],
        total_tokens: int,
        duration: float,
        model_size_mb: float,
        memory_samples: List[float]
    ) -> Dict[str, Any]:
        """
        Calculate all 6 core performance metrics.

        Args:
            predictions: List of predictions
            references: List of ground truth labels
            latencies: List of per-query latencies (seconds)
            total_tokens: Total tokens processed
            duration: Total inference time (seconds)
            model_size_mb: Model size in MB
            memory_samples: List of memory measurements (MB)

        Returns:
            Dictionary containing all metrics
        """
        # 1. Accuracy
        accuracy = self.calculate_accuracy(predictions, references)

        # 2. Speed (QPM - Queries Per Minute)
        qpm = self.calculate_qpm(len(predictions), duration)

        # 3. Size (MB)
        size_mb = model_size_mb

        # 4. Token throughput (tokens/sec)
        tokens_per_sec = self.calculate_throughput(total_tokens, duration)

        # 5. Memory usage (average, peak in MB)
        memory_stats = self.calculate_memory_stats(memory_samples)

        # 6. Latency percentiles (p50, p95, p99 in milliseconds)
        latency_percentiles = self.calculate_latency_percentiles(latencies)

        return {
            'accuracy': accuracy,
            'qpm': qpm,
            'size_mb': size_mb,
            'tokens_per_sec': tokens_per_sec,
            'memory_avg_mb': memory_stats['avg'],
            'memory_peak_mb': memory_stats['peak'],
            'latency_p50_ms': latency_percentiles['p50'],
            'latency_p95_ms': latency_percentiles['p95'],
            'latency_p99_ms': latency_percentiles['p99']
        }

    def log_metrics(self, metrics: Dict[str, Any], prefix: str = ""):
        """
        Log metrics in a formatted way.

        Args:
            metrics: Dictionary of metrics to log
            prefix: Optional prefix for log messages (e.g., "PyTorch" or "MLX")
        """
        logger.info(f"{prefix} Metrics:")
        logger.info(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
        logger.info(f"  Speed: {metrics.get('qpm', 0):.1f} QPM")
        logger.info(f"  Tokens/sec: {metrics.get('tokens_per_sec', 0):.1f}")
        logger.info(f"  Memory: {metrics.get('memory_peak_mb', 0):.1f}MB peak, "
                   f"{metrics.get('memory_avg_mb', 0):.1f}MB avg")
        logger.info(f"  Latency p50/p95/p99: "
                   f"{metrics.get('latency_p50_ms', 0):.1f}/"
                   f"{metrics.get('latency_p95_ms', 0):.1f}/"
                   f"{metrics.get('latency_p99_ms', 0):.1f}ms")
        if 'size_mb' in metrics and metrics['size_mb'] > 0:
            logger.info(f"  Model size: {metrics.get('size_mb', 0):.1f}MB")
