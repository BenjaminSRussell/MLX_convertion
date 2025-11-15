"""
Model comparison utilities for evaluating MLX vs PyTorch models.
"""

from typing import Dict, Any
from pathlib import Path
import json
import time
import logging

logger = logging.getLogger('MLX8BitTester.ModelComparator')


class ModelComparator:
    """
    Compares PyTorch baseline and MLX quantized model results.
    Handles quality gate checking and result persistence.
    """

    def __init__(self, comparisons_dir: Path):
        """
        Initialize comparator with output directory.

        Args:
            comparisons_dir: Directory to save comparison results
        """
        self.comparisons_dir = Path(comparisons_dir)
        self.comparisons_dir.mkdir(parents=True, exist_ok=True)

    def compare(
        self,
        model_name: str,
        dataset_name: str,
        pytorch_results: Dict[str, Any],
        mlx_results: Dict[str, Any],
        model_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare PyTorch and MLX model results.

        Args:
            model_name: Name of the model being compared
            dataset_name: Name of the test dataset
            pytorch_results: Results from PyTorch baseline
            mlx_results: Results from MLX quantized model
            model_config: Model configuration dict

        Returns:
            Comparison dictionary with metrics and quality gates
        """
        # Calculate comparison metrics
        accuracy_drop = pytorch_results['accuracy'] - mlx_results['accuracy']
        speedup = mlx_results['qpm'] / pytorch_results['qpm'] if pytorch_results['qpm'] > 0 else 0.0
        size_reduction = pytorch_results.get('size_mb', 0) - mlx_results.get('size_mb', 0)

        # Check quality gates
        quant_config = model_config.get('quantization', {})
        quality_gates = self.check_quality_gates(
            accuracy_drop=accuracy_drop,
            speedup=speedup,
            mlx_size_mb=mlx_results.get('size_mb', 0),
            config=quant_config
        )

        # Build comparison result
        comparison = {
            'model_name': model_name,
            'dataset_name': dataset_name,
            'quantization_bits': quant_config.get('bits', 8),
            'pytorch_baseline': pytorch_results,
            'mlx_quantized': mlx_results,
            'comparison_metrics': {
                'accuracy_drop': accuracy_drop,
                'accuracy_drop_pct': accuracy_drop * 100,
                'speedup': speedup,
                'size_reduction_mb': size_reduction,
                'compression_ratio': pytorch_results.get('size_mb', 0) / mlx_results.get('size_mb', 1) if mlx_results.get('size_mb', 0) > 0 else 0
            },
            'quality_gates': quality_gates,
            'timestamp': time.time()
        }

        # Save comparison
        self.save_comparison(model_name, dataset_name, comparison)

        # Log results
        self.log_comparison(comparison)

        return comparison

    def check_quality_gates(
        self,
        accuracy_drop: float,
        speedup: float,
        mlx_size_mb: float,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Check if model passes quality gates.

        Args:
            accuracy_drop: Accuracy decrease (PyTorch - MLX)
            speedup: Speed improvement factor
            mlx_size_mb: Size of MLX model in MB
            config: Quantization configuration

        Returns:
            Dictionary with gate results
        """
        # Extract thresholds from config
        max_accuracy_drop = config.get('max_accuracy_drop', 0.015)  # Default 1.5%
        min_speedup = config.get('min_speedup', 1.2)  # Default 20% faster
        target_size_mb = config.get('target_size_mb', 0)
        size_tolerance = 1.1  # Allow 10% over target

        # Check each gate
        accuracy_passed = accuracy_drop <= max_accuracy_drop
        speed_passed = speedup >= min_speedup
        size_passed = mlx_size_mb <= (target_size_mb * size_tolerance) if target_size_mb > 0 else True

        return {
            'accuracy': {
                'passed': accuracy_passed,
                'value': accuracy_drop,
                'threshold': max_accuracy_drop,
                'description': f'Accuracy drop must be ≤ {max_accuracy_drop*100:.1f}%'
            },
            'speed': {
                'passed': speed_passed,
                'value': speedup,
                'threshold': min_speedup,
                'description': f'Speed must be ≥ {min_speedup:.1f}x faster'
            },
            'size': {
                'passed': size_passed,
                'value': mlx_size_mb,
                'threshold': target_size_mb * size_tolerance if target_size_mb > 0 else 0,
                'description': f'Size must be ≤ {target_size_mb * size_tolerance:.1f}MB'
            },
            'all_passed': accuracy_passed and speed_passed and size_passed
        }

    def save_comparison(
        self,
        model_name: str,
        dataset_name: str,
        comparison: Dict[str, Any]
    ):
        """
        Save comparison results to JSON file.

        Args:
            model_name: Name of the model
            dataset_name: Name of the dataset
            comparison: Comparison dictionary
        """
        quant_bits = comparison['quantization_bits']
        filename = f"{model_name}_q{quant_bits}_{dataset_name}_comparison.json"
        filepath = self.comparisons_dir / filename

        with open(filepath, 'w') as f:
            json.dump(comparison, f, indent=2)

        logger.info(f"Comparison saved to {filepath}")

    def log_comparison(self, comparison: Dict[str, Any]):
        """
        Log comparison results in a readable format.

        Args:
            comparison: Comparison dictionary
        """
        model_name = comparison['model_name']
        dataset_name = comparison['dataset_name']
        metrics = comparison['comparison_metrics']
        gates = comparison['quality_gates']

        logger.info(f"\n{'='*60}")
        logger.info(f"COMPARISON: {model_name} on {dataset_name}")
        logger.info(f"{'='*60}")

        # Metrics
        logger.info(f"Accuracy drop: {metrics['accuracy_drop_pct']:.2f}%")
        logger.info(f"Speed improvement: {metrics['speedup']:.2f}x")
        logger.info(f"Size reduction: {metrics['size_reduction_mb']:.1f}MB")
        logger.info(f"Compression ratio: {metrics['compression_ratio']:.1f}x")

        # Quality gates
        logger.info(f"\nQuality Gates:")
        for gate_name, gate_info in gates.items():
            if gate_name == 'all_passed':
                continue

            status = "✓ PASS" if gate_info['passed'] else "✗ FAIL"
            logger.info(f"  {status} {gate_name.upper()}: {gate_info['description']}")
            logger.info(f"        Value: {gate_info['value']:.4f}, Threshold: {gate_info['threshold']:.4f}")

        # Overall result
        if gates['all_passed']:
            logger.info(f"\n✓ ALL QUALITY GATES PASSED")
        else:
            logger.warning(f"\n✗ SOME QUALITY GATES FAILED")

        logger.info(f"{'='*60}\n")

    def load_comparison(
        self,
        model_name: str,
        dataset_name: str,
        quant_bits: int = 8
    ) -> Dict[str, Any]:
        """
        Load a previously saved comparison.

        Args:
            model_name: Name of the model
            dataset_name: Name of the dataset
            quant_bits: Quantization bits

        Returns:
            Comparison dictionary or None if not found
        """
        filename = f"{model_name}_q{quant_bits}_{dataset_name}_comparison.json"
        filepath = self.comparisons_dir / filename

        if not filepath.exists():
            logger.warning(f"Comparison file not found: {filepath}")
            return None

        with open(filepath, 'r') as f:
            return json.load(f)

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics across all comparisons.

        Returns:
            Dictionary with summary statistics
        """
        comparison_files = list(self.comparisons_dir.glob("*_comparison.json"))

        if not comparison_files:
            logger.warning("No comparison files found")
            return {}

        total = len(comparison_files)
        passed = 0
        failed = 0
        avg_accuracy_drop = 0.0
        avg_speedup = 0.0

        for filepath in comparison_files:
            with open(filepath, 'r') as f:
                comparison = json.load(f)

            if comparison['quality_gates']['all_passed']:
                passed += 1
            else:
                failed += 1

            avg_accuracy_drop += comparison['comparison_metrics']['accuracy_drop']
            avg_speedup += comparison['comparison_metrics']['speedup']

        return {
            'total_comparisons': total,
            'passed': passed,
            'failed': failed,
            'pass_rate': passed / total if total > 0 else 0,
            'avg_accuracy_drop': avg_accuracy_drop / total if total > 0 else 0,
            'avg_speedup': avg_speedup / total if total > 0 else 0
        }
