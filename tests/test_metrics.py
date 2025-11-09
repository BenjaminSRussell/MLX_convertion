"""
Unit tests for performance metrics calculation
"""
import unittest
import numpy as np
from sklearn.metrics import accuracy_score


def calculate_performance_metrics(predictions, references, latencies, total_tokens, duration, model_size_mb, memory_samples):
    """Calculate all 6 performance metrics"""
    # 1. Accuracy
    accuracy = accuracy_score(references, predictions)

    # 2. Speed (QPM - Queries Per Minute)
    qpm = len(predictions) / duration * 60 if duration > 0 else 0

    # 3. Size (MB)
    size_mb = model_size_mb

    # 4. Token throughput (tokens/sec)
    tokens_per_sec = total_tokens / duration if duration > 0 else 0

    # 5. Memory usage (average, peak in MB)
    memory_avg_mb = np.mean(memory_samples) if memory_samples else 0
    memory_peak_mb = np.max(memory_samples) if memory_samples else 0

    # 6. Latency percentiles (p50, p95, p99 in milliseconds)
    latency_p50 = np.percentile(latencies, 50) * 1000 if latencies else 0
    latency_p95 = np.percentile(latencies, 95) * 1000 if latencies else 0
    latency_p99 = np.percentile(latencies, 99) * 1000 if latencies else 0

    return {
        'accuracy': accuracy,
        'qpm': qpm,
        'size_mb': size_mb,
        'tokens_per_sec': tokens_per_sec,
        'memory_avg_mb': memory_avg_mb,
        'memory_peak_mb': memory_peak_mb,
        'latency_p50_ms': latency_p50,
        'latency_p95_ms': latency_p95,
        'latency_p99_ms': latency_p99
    }


class TestMetricsCalculation(unittest.TestCase):
    """Test all 6 performance metrics"""

    def test_accuracy_metric(self):
        """Test accuracy calculation"""
        predictions = [0, 1, 0, 1]
        references = [0, 1, 0, 1]
        metrics = calculate_performance_metrics(
            predictions, references, [0.1] * 4, 100, 1.0, 100, [500]
        )
        self.assertEqual(metrics['accuracy'], 1.0)

    def test_accuracy_with_errors(self):
        """Test accuracy with some incorrect predictions"""
        predictions = [0, 1, 0, 1]
        references = [0, 0, 0, 1]
        metrics = calculate_performance_metrics(
            predictions, references, [0.1] * 4, 100, 1.0, 100, [500]
        )
        self.assertEqual(metrics['accuracy'], 0.75)

    def test_qpm_metric(self):
        """Test queries per minute calculation"""
        predictions = [0] * 100
        references = [0] * 100
        duration = 60.0  # 60 seconds
        metrics = calculate_performance_metrics(
            predictions, references, [0.1] * 100, 1000, duration, 100, [500]
        )
        self.assertEqual(metrics['qpm'], 100.0)  # 100 queries in 60s = 100 QPM

    def test_size_metric(self):
        """Test model size metric"""
        metrics = calculate_performance_metrics(
            [0], [0], [0.1], 10, 1.0, 125.5, [500]
        )
        self.assertEqual(metrics['size_mb'], 125.5)

    def test_token_throughput(self):
        """Test token throughput calculation"""
        total_tokens = 1000
        duration = 10.0  # 10 seconds
        metrics = calculate_performance_metrics(
            [0] * 10, [0] * 10, [0.1] * 10, total_tokens, duration, 100, [500]
        )
        self.assertEqual(metrics['tokens_per_sec'], 100.0)  # 1000 tokens / 10 seconds

    def test_memory_metrics(self):
        """Test memory average and peak"""
        memory_samples = [100, 200, 150, 180, 120]
        metrics = calculate_performance_metrics(
            [0] * 5, [0] * 5, [0.1] * 5, 100, 1.0, 100, memory_samples
        )
        self.assertEqual(metrics['memory_avg_mb'], 150.0)
        self.assertEqual(metrics['memory_peak_mb'], 200)

    def test_latency_percentiles(self):
        """Test latency percentile calculations"""
        latencies = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]  # seconds
        metrics = calculate_performance_metrics(
            [0] * 10, [0] * 10, latencies, 100, 1.0, 100, [500]
        )

        # Should be in milliseconds
        self.assertGreater(metrics['latency_p50_ms'], 0)
        self.assertGreater(metrics['latency_p95_ms'], metrics['latency_p50_ms'])
        self.assertGreater(metrics['latency_p99_ms'], metrics['latency_p95_ms'])

    def test_empty_data_handling(self):
        """Test metrics calculation with empty data"""
        metrics = calculate_performance_metrics(
            [], [], [], 0, 0.0, 0, []
        )

        self.assertEqual(metrics['qpm'], 0)
        self.assertEqual(metrics['tokens_per_sec'], 0)
        self.assertEqual(metrics['memory_avg_mb'], 0)
        self.assertEqual(metrics['memory_peak_mb'], 0)

    def test_all_six_metrics_present(self):
        """Test that all 6 metrics are calculated and present"""
        predictions = [0, 1, 0, 1]
        references = [0, 1, 0, 1]
        latencies = [0.1, 0.2, 0.15, 0.18]
        total_tokens = 100
        duration = 2.0
        model_size_mb = 125
        memory_samples = [500, 550, 525]

        metrics = calculate_performance_metrics(
            predictions, references, latencies, total_tokens, duration, model_size_mb, memory_samples
        )

        # Verify all 6 metrics are present
        required_metrics = [
            'accuracy',
            'qpm',
            'size_mb',
            'tokens_per_sec',
            'memory_avg_mb',
            'memory_peak_mb',
            'latency_p50_ms',
            'latency_p95_ms',
            'latency_p99_ms'
        ]

        for metric in required_metrics:
            self.assertIn(metric, metrics, f"Missing metric: {metric}")
            self.assertIsNotNone(metrics[metric], f"Metric {metric} is None")


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestMetricsCalculation))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    import sys
    sys.exit(0 if success else 1)
