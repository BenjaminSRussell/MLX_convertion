"""
Cross-platform parity verification.

Ensures PyTorch and MLX models produce consistent outputs.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from scipy.stats import ks_2samp
import logging

logger = logging.getLogger('MLXVerifier.ParityVerifier')


@dataclass
class ParityVerificationResult:
    """Results from parity verification."""
    test_name: str
    passed: bool
    output_similarity: float  # Cosine similarity of outputs
    distribution_pvalue: float  # KS test p-value
    max_difference: float  # Maximum absolute difference
    tolerance: float  # Tolerance used
    details: Optional[Dict[str, Any]] = None


class ParityVerifier:
    """
    Verifies parity between PyTorch and MLX implementations.

    Checks:
    1. Output consistency (same input → similar output)
    2. Distribution similarity (statistical tests)
    3. Deterministic behavior (same input → same output)
    4. Batch vs single inference consistency
    """

    def __init__(self, tolerance: float = 1e-3):
        """
        Initialize parity verifier.

        Args:
            tolerance: Maximum acceptable difference
        """
        self.tolerance = tolerance

    def verify_output_parity(
        self,
        pytorch_outputs: np.ndarray,
        mlx_outputs: np.ndarray,
        test_name: str = "output_parity"
    ) -> ParityVerificationResult:
        """
        Verify that PyTorch and MLX produce similar outputs.

        Args:
            pytorch_outputs: Outputs from PyTorch model
            mlx_outputs: Outputs from MLX model
            test_name: Name of this test

        Returns:
            ParityVerificationResult
        """
        # Ensure same shape
        if pytorch_outputs.shape != mlx_outputs.shape:
            logger.error(f"Shape mismatch: PyTorch {pytorch_outputs.shape} vs MLX {mlx_outputs.shape}")
            return ParityVerificationResult(
                test_name=test_name,
                passed=False,
                output_similarity=0.0,
                distribution_pvalue=0.0,
                max_difference=float('inf'),
                tolerance=self.tolerance,
                details={'error': 'Shape mismatch'}
            )

        # Calculate cosine similarity
        pytorch_flat = pytorch_outputs.flatten()
        mlx_flat = mlx_outputs.flatten()

        dot_product = np.dot(pytorch_flat, mlx_flat)
        norm_pytorch = np.linalg.norm(pytorch_flat)
        norm_mlx = np.linalg.norm(mlx_flat)

        if norm_pytorch > 0 and norm_mlx > 0:
            cosine_sim = dot_product / (norm_pytorch * norm_mlx)
        else:
            cosine_sim = 0.0

        # Calculate max difference
        max_diff = np.max(np.abs(pytorch_outputs - mlx_outputs))

        # Kolmogorov-Smirnov test for distribution similarity
        ks_statistic, pvalue = ks_2samp(pytorch_flat, mlx_flat)

        # Check if passed (high similarity, low difference, similar distribution)
        passed = (cosine_sim > 0.99 and max_diff < self.tolerance and pvalue > 0.01)

        details = {
            'cosine_similarity': float(cosine_sim),
            'ks_statistic': float(ks_statistic),
            'mean_absolute_difference': float(np.mean(np.abs(pytorch_outputs - mlx_outputs))),
            'std_difference': float(np.std(pytorch_outputs - mlx_outputs)),
            'pytorch_mean': float(np.mean(pytorch_outputs)),
            'pytorch_std': float(np.std(pytorch_outputs)),
            'mlx_mean': float(np.mean(mlx_outputs)),
            'mlx_std': float(np.std(mlx_outputs))
        }

        logger.info(f"Parity Check: {test_name}")
        logger.info(f"  Cosine similarity: {cosine_sim:.6f}")
        logger.info(f"  Max difference: {max_diff:.6f}")
        logger.info(f"  KS test p-value: {pvalue:.6f}")
        logger.info(f"  Status: {'✓ PASSED' if passed else '✗ FAILED'}")

        return ParityVerificationResult(
            test_name=test_name,
            passed=passed,
            output_similarity=float(cosine_sim),
            distribution_pvalue=float(pvalue),
            max_difference=float(max_diff),
            tolerance=self.tolerance,
            details=details
        )

    def verify_deterministic_inference(
        self,
        inference_fn: callable,
        inputs: Any,
        num_runs: int = 3
    ) -> ParityVerificationResult:
        """
        Verify that model produces deterministic outputs.

        Args:
            inference_fn: Function that runs inference
            inputs: Input data
            num_runs: Number of runs to test

        Returns:
            ParityVerificationResult
        """
        outputs = []

        for i in range(num_runs):
            output = inference_fn(inputs)
            outputs.append(output)

        # Check that all outputs are identical
        first_output = outputs[0]
        all_identical = all(np.allclose(first_output, out, atol=1e-6) for out in outputs[1:])

        if all_identical:
            max_diff = 0.0
        else:
            # Calculate maximum difference across runs
            diffs = [np.max(np.abs(first_output - out)) for out in outputs[1:]]
            max_diff = max(diffs)

        passed = all_identical

        details = {
            'num_runs': num_runs,
            'all_identical': all_identical,
            'max_diff_across_runs': float(max_diff)
        }

        logger.info(f"Deterministic Inference Check:")
        logger.info(f"  Runs: {num_runs}")
        logger.info(f"  All identical: {all_identical}")
        logger.info(f"  Status: {'✓ PASSED' if passed else '✗ FAILED'}")

        return ParityVerificationResult(
            test_name='deterministic_inference',
            passed=passed,
            output_similarity=1.0 if all_identical else 0.0,
            distribution_pvalue=1.0,
            max_difference=float(max_diff),
            tolerance=1e-6,
            details=details
        )

    def verify_batch_consistency(
        self,
        inference_fn: callable,
        inputs: List[Any],
        batch_size: int = 8
    ) -> ParityVerificationResult:
        """
        Verify that batch inference produces same results as single inference.

        Args:
            inference_fn: Inference function
            inputs: List of input samples
            batch_size: Batch size to test

        Returns:
            ParityVerificationResult
        """
        # Run single inference
        single_outputs = []
        for inp in inputs:
            out = inference_fn([inp])  # Batch of 1
            single_outputs.append(out[0] if isinstance(out, list) else out)

        # Run batch inference
        batch_outputs = []
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i+batch_size]
            out = inference_fn(batch)
            batch_outputs.extend(out if isinstance(out, list) else [out])

        # Compare outputs
        differences = []
        for single, batch in zip(single_outputs[:len(batch_outputs)], batch_outputs):
            diff = np.max(np.abs(np.array(single) - np.array(batch)))
            differences.append(diff)

        max_diff = max(differences) if differences else 0.0
        passed = max_diff < self.tolerance

        details = {
            'num_samples': len(inputs),
            'batch_size': batch_size,
            'mean_difference': float(np.mean(differences)) if differences else 0.0,
            'std_difference': float(np.std(differences)) if differences else 0.0
        }

        logger.info(f"Batch Consistency Check:")
        logger.info(f"  Samples: {len(inputs)}, Batch size: {batch_size}")
        logger.info(f"  Max difference: {max_diff:.6f}")
        logger.info(f"  Status: {'✓ PASSED' if passed else '✗ FAILED'}")

        return ParityVerificationResult(
            test_name='batch_consistency',
            passed=passed,
            output_similarity=1.0 if passed else 0.0,
            distribution_pvalue=1.0,
            max_difference=float(max_diff),
            tolerance=self.tolerance,
            details=details
        )

    def generate_report(self, results: List[ParityVerificationResult]) -> str:
        """
        Generate parity verification report.

        Args:
            results: List of parity results

        Returns:
            Formatted report
        """
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed

        report = []
        report.append("=" * 60)
        report.append("CROSS-PLATFORM PARITY VERIFICATION REPORT")
        report.append("=" * 60)
        report.append(f"Total tests: {total}")
        report.append(f"Passed: {passed} ({passed/total*100:.1f}%)")
        report.append(f"Failed: {failed} ({failed/total*100:.1f}%)")
        report.append("")

        for r in results:
            status = "✓" if r.passed else "✗"
            report.append(f"{status} {r.test_name}")
            report.append(f"    Output similarity: {r.output_similarity:.6f}")
            report.append(f"    Max difference: {r.max_difference:.6f} (tolerance: {r.tolerance:.6f})")
            report.append(f"    Distribution p-value: {r.distribution_pvalue:.6f}")
            if r.details:
                report.append(f"    Details: {r.details}")
            report.append("")

        report.append("=" * 60)
        return "\n".join(report)
