"""
Performance regression verification.

Ensures converted models meet performance targets and don't regress.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from pathlib import Path
import time
import json
import numpy as np
import logging

logger = logging.getLogger('MLXVerifier.PerformanceVerifier')


@dataclass
class PerformanceVerificationResult:
    """Results from performance verification."""
    metric_name: str
    current_value: float
    baseline_value: Optional[float]
    threshold: float
    passed: bool
    regression: bool  # True if performance degraded
    improvement_pct: float  # Positive if improved
    details: Optional[Dict[str, Any]] = None


class PerformanceVerifier:
    """
    Verifies performance metrics and detects regressions.

    Tracks:
    1. Latency (p50, p95, p99)
    2. Throughput (queries/second, tokens/second)
    3. Memory usage (peak, average)
    4. Model size (MB)
    """

    def __init__(self, baseline_dir: Optional[Path] = None):
        """
        Initialize performance verifier.

        Args:
            baseline_dir: Directory containing baseline performance metrics
        """
        self.baseline_dir = Path(baseline_dir) if baseline_dir else None
        self.baselines = {}

        if self.baseline_dir and self.baseline_dir.exists():
            self._load_baselines()

    def _load_baselines(self):
        """Load baseline metrics from disk."""
        for baseline_file in self.baseline_dir.glob("*.json"):
            model_name = baseline_file.stem
            with open(baseline_file, 'r') as f:
                self.baselines[model_name] = json.load(f)
            logger.info(f"Loaded baseline for {model_name}")

    def save_baseline(self, model_name: str, metrics: Dict[str, float]):
        """
        Save baseline metrics for a model.

        Args:
            model_name: Model name
            metrics: Performance metrics
        """
        if not self.baseline_dir:
            logger.warning("No baseline directory configured")
            return

        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        baseline_file = self.baseline_dir / f"{model_name}.json"

        with open(baseline_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        self.baselines[model_name] = metrics
        logger.info(f"Saved baseline for {model_name} to {baseline_file}")

    def verify_latency(
        self,
        current_latency: float,
        model_name: str,
        percentile: str = "p50",
        max_regression_pct: float = 10.0
    ) -> PerformanceVerificationResult:
        """
        Verify latency hasn't regressed.

        Args:
            current_latency: Current latency (ms)
            model_name: Model name
            percentile: Latency percentile (p50, p95, p99)
            max_regression_pct: Max acceptable regression %

        Returns:
            PerformanceVerificationResult
        """
        metric_name = f"latency_{percentile}"
        baseline = self.baselines.get(model_name, {}).get(metric_name)

        if baseline is None:
            logger.warning(f"No baseline for {model_name} {metric_name}")
            passed = True  # Pass if no baseline
            regression = False
            improvement_pct = 0.0
        else:
            # Calculate regression/improvement
            improvement_pct = ((baseline - current_latency) / baseline) * 100
            regression = improvement_pct < -max_regression_pct  # Negative = slower
            passed = not regression

        details = {
            'unit': 'ms',
            'lower_is_better': True
        }

        if not passed:
            logger.warning(f"Latency regression detected for {model_name}")
            logger.warning(f"  Current: {current_latency:.2f}ms")
            logger.warning(f"  Baseline: {baseline:.2f}ms")
            logger.warning(f"  Regression: {-improvement_pct:.1f}%")

        return PerformanceVerificationResult(
            metric_name=metric_name,
            current_value=current_latency,
            baseline_value=baseline,
            threshold=max_regression_pct,
            passed=passed,
            regression=regression,
            improvement_pct=improvement_pct,
            details=details
        )

    def verify_throughput(
        self,
        current_throughput: float,
        model_name: str,
        metric_type: str = "qpm",
        min_regression_pct: float = -10.0
    ) -> PerformanceVerificationResult:
        """
        Verify throughput hasn't regressed.

        Args:
            current_throughput: Current throughput
            model_name: Model name
            metric_type: Type (qpm, tokens_per_sec)
            min_regression_pct: Min acceptable regression %

        Returns:
            PerformanceVerificationResult
        """
        metric_name = f"throughput_{metric_type}"
        baseline = self.baselines.get(model_name, {}).get(metric_name)

        if baseline is None:
            passed = True
            regression = False
            improvement_pct = 0.0
        else:
            # For throughput, higher is better
            improvement_pct = ((current_throughput - baseline) / baseline) * 100
            regression = improvement_pct < min_regression_pct  # Negative = slower
            passed = not regression

        details = {
            'unit': 'queries/min' if metric_type == 'qpm' else 'tokens/sec',
            'higher_is_better': True
        }

        return PerformanceVerificationResult(
            metric_name=metric_name,
            current_value=current_throughput,
            baseline_value=baseline,
            threshold=abs(min_regression_pct),
            passed=passed,
            regression=regression,
            improvement_pct=improvement_pct,
            details=details
        )

    def verify_memory(
        self,
        current_memory: float,
        model_name: str,
        memory_type: str = "peak",
        max_regression_pct: float = 15.0
    ) -> PerformanceVerificationResult:
        """
        Verify memory usage hasn't regressed.

        Args:
            current_memory: Current memory (MB)
            model_name: Model name
            memory_type: Type (peak, average)
            max_regression_pct: Max acceptable regression %

        Returns:
            PerformanceVerificationResult
        """
        metric_name = f"memory_{memory_type}_mb"
        baseline = self.baselines.get(model_name, {}).get(metric_name)

        if baseline is None:
            passed = True
            regression = False
            improvement_pct = 0.0
        else:
            # For memory, lower is better
            improvement_pct = ((baseline - current_memory) / baseline) * 100
            regression = improvement_pct < -max_regression_pct
            passed = not regression

        details = {
            'unit': 'MB',
            'lower_is_better': True
        }

        return PerformanceVerificationResult(
            metric_name=metric_name,
            current_value=current_memory,
            baseline_value=baseline,
            threshold=max_regression_pct,
            passed=passed,
            regression=regression,
            improvement_pct=improvement_pct,
            details=details
        )

    def verify_all_metrics(
        self,
        model_name: str,
        current_metrics: Dict[str, float]
    ) -> List[PerformanceVerificationResult]:
        """
        Verify all performance metrics for a model.

        Args:
            model_name: Model name
            current_metrics: Current performance metrics

        Returns:
            List of verification results
        """
        results = []

        # Verify latency metrics
        for percentile in ['p50', 'p95', 'p99']:
            latency_key = f'latency_{percentile}_ms'
            if latency_key in current_metrics:
                result = self.verify_latency(
                    current_metrics[latency_key],
                    model_name,
                    percentile=percentile
                )
                results.append(result)

        # Verify throughput
        if 'qpm' in current_metrics:
            result = self.verify_throughput(
                current_metrics['qpm'],
                model_name,
                metric_type='qpm'
            )
            results.append(result)

        if 'tokens_per_sec' in current_metrics:
            result = self.verify_throughput(
                current_metrics['tokens_per_sec'],
                model_name,
                metric_type='tokens_per_sec'
            )
            results.append(result)

        # Verify memory
        for memory_type in ['peak', 'avg']:
            memory_key = f'memory_{memory_type}_mb'
            if memory_key in current_metrics:
                result = self.verify_memory(
                    current_metrics[memory_key],
                    model_name,
                    memory_type=memory_type
                )
                results.append(result)

        return results

    def generate_report(self, results: List[PerformanceVerificationResult]) -> str:
        """
        Generate performance verification report.

        Args:
            results: List of verification results

        Returns:
            Formatted report
        """
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed
        regressions = sum(1 for r in results if r.regression)

        report = []
        report.append("=" * 60)
        report.append("PERFORMANCE VERIFICATION REPORT")
        report.append("=" * 60)
        report.append(f"Total metrics checked: {total}")
        report.append(f"Passed: {passed} ({passed/total*100:.1f}%)")
        report.append(f"Failed: {failed} ({failed/total*100:.1f}%)")
        report.append(f"Regressions detected: {regressions}")
        report.append("")

        # Group by metric type
        by_type = {}
        for r in results:
            metric_type = r.metric_name.split('_')[0]
            if metric_type not in by_type:
                by_type[metric_type] = []
            by_type[metric_type].append(r)

        for metric_type, type_results in by_type.items():
            report.append(f"{metric_type.upper()} Metrics:")
            for r in type_results:
                status = "✓" if r.passed else "✗"
                direction = "↑" if r.improvement_pct > 0 else "↓" if r.improvement_pct < 0 else "="

                report.append(f"  {status} {r.metric_name}: {r.current_value:.2f}")

                if r.baseline_value is not None:
                    report.append(f"      Baseline: {r.baseline_value:.2f}")
                    report.append(f"      Change: {direction} {abs(r.improvement_pct):.1f}%")

                if r.regression:
                    report.append(f"      ⚠ REGRESSION DETECTED")
            report.append("")

        report.append("=" * 60)
        return "\n".join(report)
