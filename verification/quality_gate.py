"""
automated quality gate enforcement - checks all the quality requirements
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import logging

from .layer_verifier import LayerVerifier, LayerVerificationResult
from .task_verifier import TaskVerifier, TaskVerificationResult
from .parity_verifier import ParityVerifier, ParityVerificationResult
from .performance_verifier import PerformanceVerifier, PerformanceVerificationResult

logger = logging.getLogger('MLXVerifier.QualityGate')


@dataclass
class QualityGateResult:
    """results from quality gate checks"""
    model_name: str
    passed: bool
    gates_passed: Dict[str, bool]
    layer_results: Optional[List[LayerVerificationResult]] = None
    task_results: Optional[List[TaskVerificationResult]] = None
    parity_results: Optional[List[ParityVerificationResult]] = None
    performance_results: Optional[List[PerformanceVerificationResult]] = None
    summary: Optional[Dict[str, Any]] = None


class QualityGateEnforcer:
    """enforces quality gates: layer accuracy, task accuracy, parity, performance"""

    def __init__(
        self,
        layer_error_threshold: float = 0.01,
        performance_baseline_dir: Optional[Path] = None,
        parity_tolerance: float = 1e-3
    ):
        """sets up quality gate enforcer"""
        self.layer_verifier = LayerVerifier(error_threshold=layer_error_threshold)
        self.task_verifier = TaskVerifier()
        self.parity_verifier = ParityVerifier(tolerance=parity_tolerance)
        self.performance_verifier = PerformanceVerifier(baseline_dir=performance_baseline_dir)

    def enforce_all_gates(
        self,
        model_name: str,
        layer_data: Optional[Dict[str, Any]] = None,
        task_data: Optional[Dict[str, Any]] = None,
        parity_data: Optional[Dict[str, Any]] = None,
        performance_data: Optional[Dict[str, Any]] = None,
        required_gates: Optional[List[str]] = None
    ) -> QualityGateResult:
        """
        Enforce all quality gates.

        Args:
            model_name: Model name
            layer_data: Layer verification data
            task_data: Task verification data
            parity_data: Parity verification data
            performance_data: Performance verification data
            required_gates: List of required gates (default: all)

        Returns:
            QualityGateResult
        """
        if required_gates is None:
            required_gates = ['layer', 'task', 'parity', 'performance']

        gates_passed = {}
        layer_results = None
        task_results = None
        parity_results = None
        performance_results = None

        # Layer verification
        if 'layer' in required_gates and layer_data:
            layer_results = self.layer_verifier.verify_model(
                original_weights=layer_data['original_weights'],
                quantized_weights=layer_data['quantized_weights'],
                scales=layer_data['scales'],
                zero_points=layer_data.get('zero_points')
            )
            gates_passed['layer'] = all(r.passed for r in layer_results)
            logger.info(f"Layer gate: {'✓ PASSED' if gates_passed['layer'] else '✗ FAILED'}")

        # Task verification
        if 'task' in required_gates and task_data:
            task_type = task_data['task_type']

            if task_type == 'nli':
                task_result = self.task_verifier.verify_nli(
                    baseline_predictions=task_data['baseline_predictions'],
                    converted_predictions=task_data['converted_predictions'],
                    ground_truth=task_data['ground_truth'],
                    model_name=model_name,
                    dataset_name=task_data.get('dataset_name', 'unknown'),
                    max_accuracy_drop=task_data.get('max_accuracy_drop', 0.01)
                )
            elif task_type == 'embedding':
                task_result = self.task_verifier.verify_embeddings(
                    baseline_scores=task_data['baseline_scores'],
                    converted_scores=task_data['converted_scores'],
                    ground_truth_scores=task_data['ground_truth_scores'],
                    model_name=model_name,
                    dataset_name=task_data.get('dataset_name', 'unknown'),
                    min_spearman=task_data.get('min_spearman', 0.98)
                )
            elif task_type == 'text_classification':
                task_result = self.task_verifier.verify_text_classification(
                    baseline_predictions=task_data['baseline_predictions'],
                    converted_predictions=task_data['converted_predictions'],
                    ground_truth=task_data['ground_truth'],
                    model_name=model_name,
                    dataset_name=task_data.get('dataset_name', 'unknown'),
                    max_accuracy_drop=task_data.get('max_accuracy_drop', 0.015)
                )
            elif task_type == 'ner':
                task_result = self.task_verifier.verify_ner(
                    baseline_predictions=task_data['baseline_predictions'],
                    converted_predictions=task_data['converted_predictions'],
                    ground_truth=task_data['ground_truth'],
                    model_name=model_name,
                    dataset_name=task_data.get('dataset_name', 'unknown'),
                    max_f1_drop=task_data.get('max_f1_drop', 0.02)
                )
            else:
                logger.warning(f"Unknown task type: {task_type}")
                task_result = None

            if task_result:
                task_results = [task_result]
                gates_passed['task'] = task_result.passed
                logger.info(f"Task gate: {'✓ PASSED' if gates_passed['task'] else '✗ FAILED'}")

        # Parity verification
        if 'parity' in required_gates and parity_data:
            parity_results = []

            # Output parity
            if 'pytorch_outputs' in parity_data and 'mlx_outputs' in parity_data:
                parity_result = self.parity_verifier.verify_output_parity(
                    pytorch_outputs=parity_data['pytorch_outputs'],
                    mlx_outputs=parity_data['mlx_outputs'],
                    test_name='output_parity'
                )
                parity_results.append(parity_result)

            gates_passed['parity'] = all(r.passed for r in parity_results) if parity_results else True
            logger.info(f"Parity gate: {'✓ PASSED' if gates_passed['parity'] else '✗ FAILED'}")

        # Performance verification
        if 'performance' in required_gates and performance_data:
            performance_results = self.performance_verifier.verify_all_metrics(
                model_name=model_name,
                current_metrics=performance_data
            )
            gates_passed['performance'] = all(r.passed for r in performance_results)
            logger.info(f"Performance gate: {'✓ PASSED' if gates_passed['performance'] else '✗ FAILED'}")

        # Overall result
        all_passed = all(gates_passed.get(gate, True) for gate in required_gates)

        # Create summary
        summary = {
            'total_gates': len(required_gates),
            'gates_passed': sum(gates_passed.values()),
            'gates_failed': len(required_gates) - sum(gates_passed.values()),
            'required_gates': required_gates,
            'gate_status': gates_passed
        }

        result = QualityGateResult(
            model_name=model_name,
            passed=all_passed,
            gates_passed=gates_passed,
            layer_results=layer_results,
            task_results=task_results,
            parity_results=parity_results,
            performance_results=performance_results,
            summary=summary
        )

        logger.info(f"\n{'='*60}")
        logger.info(f"Quality Gate Result: {model_name}")
        logger.info(f"Overall: {'✓ PASSED' if all_passed else '✗ FAILED'}")
        logger.info(f"Gates: {sum(gates_passed.values())}/{len(required_gates)} passed")
        logger.info(f"{'='*60}\n")

        return result

    def generate_comprehensive_report(self, result: QualityGateResult) -> str:
        """
        Generate comprehensive quality gate report.

        Args:
            result: Quality gate result

        Returns:
            Formatted report
        """
        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE QUALITY GATE REPORT")
        report.append("=" * 80)
        report.append(f"Model: {result.model_name}")
        report.append(f"Overall Status: {'✓ PASSED' if result.passed else '✗ FAILED'}")
        report.append(f"Gates Passed: {result.summary['gates_passed']}/{result.summary['total_gates']}")
        report.append("")

        # Gate summary
        report.append("Gate Status:")
        for gate, passed in result.gates_passed.items():
            status = "✓ PASSED" if passed else "✗ FAILED"
            report.append(f"  {gate.upper()}: {status}")
        report.append("")

        # Layer verification details
        if result.layer_results:
            report.append("-" * 80)
            report.append(self.layer_verifier.generate_report(result.layer_results))

        # Task verification details
        if result.task_results:
            report.append("-" * 80)
            report.append(self.task_verifier.generate_report(result.task_results))

        # Parity verification details
        if result.parity_results:
            report.append("-" * 80)
            report.append(self.parity_verifier.generate_report(result.parity_results))

        # Performance verification details
        if result.performance_results:
            report.append("-" * 80)
            report.append(self.performance_verifier.generate_report(result.performance_results))

        report.append("=" * 80)
        return "\n".join(report)

    def save_result(self, result: QualityGateResult, output_path: Path):
        """
        Save quality gate result to JSON.

        Args:
            result: Quality gate result
            output_path: Path to save result
        """
        data = {
            'model_name': result.model_name,
            'passed': result.passed,
            'gates_passed': result.gates_passed,
            'summary': result.summary
        }

        # Add detailed results (simplified)
        if result.layer_results:
            data['layer_verification'] = {
                'total_layers': len(result.layer_results),
                'passed': sum(1 for r in result.layer_results if r.passed),
                'avg_relative_error': sum(r.relative_error for r in result.layer_results) / len(result.layer_results)
            }

        if result.task_results:
            data['task_verification'] = [
                {
                    'task_type': r.task_type,
                    'passed': r.passed,
                    'primary_metric': r.primary_metric,
                    'value': r.primary_metric_value,
                    'baseline': r.baseline_value
                }
                for r in result.task_results
            ]

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Quality gate result saved to {output_path}")
