"""
Layer-level weight verification.

Verifies that quantized weights can be accurately reconstructed
and that quantization error is within acceptable bounds.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from pathlib import Path
import numpy as np
import logging

logger = logging.getLogger('MLXVerifier.LayerVerifier')


@dataclass
class LayerVerificationResult:
    """Results from layer-level verification."""
    layer_name: str
    passed: bool
    mse: float  # Mean Squared Error
    mae: float  # Mean Absolute Error
    max_error: float  # Maximum absolute error
    relative_error: float  # Relative error (%)
    error_threshold: float  # Threshold used
    details: Optional[Dict[str, Any]] = None


class LayerVerifier:
    """
    Verifies quantized weights at the layer level.

    Checks:
    1. Weight reconstruction accuracy (MSE, MAE)
    2. Maximum absolute error per layer
    3. Relative error percentage
    4. Distribution preservation
    """

    def __init__(self, error_threshold: float = 0.01):
        """
        Initialize layer verifier.

        Args:
            error_threshold: Maximum acceptable relative error (default 1%)
        """
        self.error_threshold = error_threshold

    def verify_layer(
        self,
        original_weight: np.ndarray,
        quantized_weight: np.ndarray,
        scale: Any,
        zero_point: Optional[Any] = None,
        layer_name: str = "unknown"
    ) -> LayerVerificationResult:
        """
        Verify a single layer's quantization accuracy.

        Args:
            original_weight: Original FP32 weights
            quantized_weight: Quantized weights (INT8/INT4)
            scale: Quantization scale factor(s)
            zero_point: Zero point for asymmetric quantization
            layer_name: Name of the layer

        Returns:
            LayerVerificationResult
        """
        # Dequantize the weight
        if zero_point is not None:
            # Asymmetric dequantization
            if isinstance(scale, np.ndarray):
                # Per-channel
                reconstructed = (quantized_weight.astype(np.float32) - zero_point) * scale
            else:
                # Per-tensor
                reconstructed = (quantized_weight.astype(np.float32) - zero_point) * scale
        else:
            # Symmetric dequantization
            if isinstance(scale, np.ndarray):
                # Per-channel or grouped
                if scale.size == quantized_weight.shape[-1]:
                    # Per-channel (broadcast along last dimension)
                    reconstructed = quantized_weight.astype(np.float32) * scale
                else:
                    # Grouped quantization
                    reconstructed = self._dequantize_grouped(quantized_weight, scale)
            else:
                # Per-tensor
                reconstructed = quantized_weight.astype(np.float32) * scale

        # Calculate error metrics
        mse = np.mean((original_weight - reconstructed) ** 2)
        mae = np.mean(np.abs(original_weight - reconstructed))
        max_error = np.max(np.abs(original_weight - reconstructed))

        # Relative error (avoid division by zero)
        original_range = np.max(np.abs(original_weight))
        if original_range > 0:
            relative_error = max_error / original_range
        else:
            relative_error = 0.0

        # Check if passed
        passed = relative_error <= self.error_threshold

        # Additional statistics
        details = {
            'original_min': float(np.min(original_weight)),
            'original_max': float(np.max(original_weight)),
            'original_mean': float(np.mean(original_weight)),
            'original_std': float(np.std(original_weight)),
            'reconstructed_min': float(np.min(reconstructed)),
            'reconstructed_max': float(np.max(reconstructed)),
            'reconstructed_mean': float(np.mean(reconstructed)),
            'reconstructed_std': float(np.std(reconstructed)),
            'weight_shape': original_weight.shape,
            'num_params': original_weight.size
        }

        result = LayerVerificationResult(
            layer_name=layer_name,
            passed=passed,
            mse=float(mse),
            mae=float(mae),
            max_error=float(max_error),
            relative_error=float(relative_error),
            error_threshold=self.error_threshold,
            details=details
        )

        if not passed:
            logger.warning(f"Layer {layer_name} failed verification: "
                          f"relative_error={relative_error:.4f} > {self.error_threshold:.4f}")
        else:
            logger.debug(f"Layer {layer_name} passed: relative_error={relative_error:.4f}")

        return result

    def _dequantize_grouped(
        self,
        quantized: np.ndarray,
        scales: np.ndarray,
        group_size: int = 64
    ) -> np.ndarray:
        """
        Dequantize weights with grouped quantization.

        Args:
            quantized: Quantized weights
            scales: Scale factors per group
            group_size: Group size

        Returns:
            Dequantized weights
        """
        original_shape = quantized.shape
        quantized_flat = quantized.flatten()

        # Ensure we have the right number of scales
        num_groups = (len(quantized_flat) + group_size - 1) // group_size

        if len(scales) != num_groups:
            logger.warning(f"Scale count mismatch: {len(scales)} vs {num_groups} groups")
            # Fallback to per-tensor
            return quantized.astype(np.float32) * scales.mean()

        # Dequantize each group
        dequantized_flat = np.zeros_like(quantized_flat, dtype=np.float32)

        for i in range(num_groups):
            start = i * group_size
            end = min((i + 1) * group_size, len(quantized_flat))
            dequantized_flat[start:end] = quantized_flat[start:end].astype(np.float32) * scales[i]

        return dequantized_flat[:np.prod(original_shape)].reshape(original_shape)

    def verify_model(
        self,
        original_weights: Dict[str, np.ndarray],
        quantized_weights: Dict[str, np.ndarray],
        scales: Dict[str, Any],
        zero_points: Optional[Dict[str, Any]] = None
    ) -> List[LayerVerificationResult]:
        """
        Verify all layers in a model.

        Args:
            original_weights: Original FP32 weights
            quantized_weights: Quantized weights
            scales: Scale factors per layer
            zero_points: Zero points per layer (if asymmetric)

        Returns:
            List of verification results
        """
        results = []

        for layer_name in original_weights.keys():
            # Skip if not quantized
            if layer_name not in quantized_weights:
                continue

            original = original_weights[layer_name]
            quantized = quantized_weights[layer_name]
            scale = scales.get(layer_name)
            zero_point = zero_points.get(layer_name) if zero_points else None

            if scale is None:
                logger.warning(f"No scale found for layer {layer_name}, skipping")
                continue

            result = self.verify_layer(
                original_weight=original,
                quantized_weight=quantized,
                scale=scale,
                zero_point=zero_point,
                layer_name=layer_name
            )

            results.append(result)

        return results

    def generate_report(self, results: List[LayerVerificationResult]) -> str:
        """
        Generate a text report from verification results.

        Args:
            results: List of verification results

        Returns:
            Formatted report string
        """
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed

        # Calculate statistics
        avg_mse = np.mean([r.mse for r in results])
        avg_mae = np.mean([r.mae for r in results])
        avg_relative_error = np.mean([r.relative_error for r in results])
        max_relative_error = np.max([r.relative_error for r in results])

        report = []
        report.append("="* 60)
        report.append("LAYER-LEVEL VERIFICATION REPORT")
        report.append("=" * 60)
        report.append(f"Total layers verified: {total}")
        report.append(f"Passed: {passed} ({passed/total*100:.1f}%)")
        report.append(f"Failed: {failed} ({failed/total*100:.1f}%)")
        report.append(f"Error threshold: {self.error_threshold:.4f}")
        report.append("")
        report.append("Average Metrics:")
        report.append(f"  MSE: {avg_mse:.6f}")
        report.append(f"  MAE: {avg_mae:.6f}")
        report.append(f"  Avg Relative Error: {avg_relative_error:.4f} ({avg_relative_error*100:.2f}%)")
        report.append(f"  Max Relative Error: {max_relative_error:.4f} ({max_relative_error*100:.2f}%)")
        report.append("")

        if failed > 0:
            report.append("Failed Layers:")
            for r in results:
                if not r.passed:
                    report.append(f"  {r.layer_name}:")
                    report.append(f"    Relative Error: {r.relative_error:.4f} ({r.relative_error*100:.2f}%)")
                    report.append(f"    MSE: {r.mse:.6f}, MAE: {r.mae:.6f}")
            report.append("")

        report.append("Top 5 Layers by Error:")
        sorted_results = sorted(results, key=lambda r: r.relative_error, reverse=True)[:5]
        for r in sorted_results:
            status = "✓" if r.passed else "✗"
            report.append(f"  {status} {r.layer_name}: {r.relative_error:.4f} ({r.relative_error*100:.2f}%)")

        report.append("=" * 60)

        return "\n".join(report)

    def save_report(self, results: List[LayerVerificationResult], output_path: Path):
        """
        Save verification report to file.

        Args:
            results: Verification results
            output_path: Path to save report
        """
        report = self.generate_report(results)

        with open(output_path, 'w') as f:
            f.write(report)

        logger.info(f"Verification report saved to {output_path}")
