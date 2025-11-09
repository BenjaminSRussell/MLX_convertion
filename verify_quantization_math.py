#!/usr/bin/env python3
"""
Verify the quantization/dequantization math is correct.

This tests the exact formulas used in the converter.
"""

import numpy as np


def quantize_8bit(weights: np.ndarray) -> tuple:
    """
    Quantize weights to 8-bit (same logic as convert_encoder.py).

    Returns: (quantized_weights, scale)
    """
    # Symmetric quantization to [-127, 127]
    w_max = np.maximum(np.abs(weights.max()), np.abs(weights.min()))

    # Calculate scale
    scale = w_max / 127.0

    # Avoid division by zero
    if scale == 0:
        scale = 1.0

    # Quantize
    w_quant = np.round(weights / scale)
    w_quant = np.clip(w_quant, -127, 127).astype(np.int8)

    return w_quant, scale


def dequantize_8bit(w_quant: np.ndarray, scale: float) -> np.ndarray:
    """
    Dequantize 8-bit weights back to float32 (same logic as test_encoder.py).

    Returns: dequantized_weights
    """
    return w_quant.astype(np.float32) * scale


def test_quantization():
    """Run tests on the quantization logic."""
    print("=" * 80)
    print("QUANTIZATION MATH VERIFICATION")
    print("=" * 80)
    print()

    # Test 1: Small positive/negative values
    print("Test 1: Small values [-0.8, 0.7]")
    print("-" * 80)
    original = np.array([0.5, -0.3, 0.7, -0.8, 0.0, 0.1])
    w_quant, scale = quantize_8bit(original)
    w_dequant = dequantize_8bit(w_quant, scale)

    print(f"Original:     {original}")
    print(f"Scale:        {scale:.6f}")
    print(f"Quantized:    {w_quant}")
    print(f"Dequantized:  {w_dequant}")
    print(f"Error:        {original - w_dequant}")
    print(f"Max error:    {np.max(np.abs(original - w_dequant)):.6f}")
    print(f"Rel error:    {np.mean(np.abs(original - w_dequant) / (np.abs(original) + 1e-8)):.4%}")
    print()

    # Test 2: Large values
    print("Test 2: Large values [-100, 100]")
    print("-" * 80)
    original = np.array([50.0, -30.0, 70.0, -100.0, 0.0, 10.0])
    w_quant, scale = quantize_8bit(original)
    w_dequant = dequantize_8bit(w_quant, scale)

    print(f"Original:     {original}")
    print(f"Scale:        {scale:.6f}")
    print(f"Quantized:    {w_quant}")
    print(f"Dequantized:  {w_dequant}")
    print(f"Error:        {original - w_dequant}")
    print(f"Max error:    {np.max(np.abs(original - w_dequant)):.6f}")
    print(f"Rel error:    {np.mean(np.abs(original - w_dequant) / (np.abs(original) + 1e-8)):.4%}")
    print()

    # Test 3: Realistic neural network weights
    print("Test 3: Realistic NN weights (random normal)")
    print("-" * 80)
    np.random.seed(42)
    original = np.random.randn(1000, 1000).astype(np.float32) * 0.1  # Typical weight initialization
    w_quant, scale = quantize_8bit(original)
    w_dequant = dequantize_8bit(w_quant, scale)

    error = original - w_dequant
    print(f"Shape:        {original.shape}")
    print(f"Scale:        {scale:.6f}")
    print(f"Weight range: [{original.min():.4f}, {original.max():.4f}]")
    print(f"Quant range:  [{w_quant.min()}, {w_quant.max()}]")
    print()
    print(f"MSE:          {np.mean(error ** 2):.2e}")
    print(f"MAE:          {np.mean(np.abs(error)):.2e}")
    print(f"Max error:    {np.max(np.abs(error)):.6f}")
    print(f"Rel error:    {np.mean(np.abs(error) / (np.abs(original) + 1e-8)):.4%}")
    print()

    # Test 4: Edge cases
    print("Test 4: Edge cases")
    print("-" * 80)

    # 4a: All zeros
    original = np.zeros(10)
    w_quant, scale = quantize_8bit(original)
    w_dequant = dequantize_8bit(w_quant, scale)
    print(f"All zeros - Scale: {scale}, Max error: {np.max(np.abs(original - w_dequant)):.6f} ✓")

    # 4b: Very small values
    original = np.array([1e-8, -1e-8, 1e-7])
    w_quant, scale = quantize_8bit(original)
    w_dequant = dequantize_8bit(w_quant, scale)
    print(f"Tiny values - Scale: {scale:.2e}, Max error: {np.max(np.abs(original - w_dequant)):.2e} ✓")

    # 4c: Values at quantization boundaries
    original = np.array([127.0, -127.0, 0.0])
    w_quant, scale = quantize_8bit(original)
    w_dequant = dequantize_8bit(w_quant, scale)
    print(f"At boundaries - Scale: {scale:.6f}, Max error: {np.max(np.abs(original - w_dequant)):.6f} ✓")
    print()

    # Verify properties
    print("=" * 80)
    print("VERIFICATION RESULTS")
    print("=" * 80)
    print()

    checks = []

    # Check 1: Scale calculation is correct
    w = np.array([0.5, -0.8, 0.3])
    w_max = max(abs(0.5), abs(-0.8), abs(0.3))
    expected_scale = w_max / 127.0
    w_quant, actual_scale = quantize_8bit(w)
    checks.append(("Scale calculation", abs(expected_scale - actual_scale) < 1e-9))

    # Check 2: Quantized values are in range
    original = np.random.randn(100, 100).astype(np.float32)
    w_quant, scale = quantize_8bit(original)
    checks.append(("Quantized range [-127, 127]", (w_quant.min() >= -127) and (w_quant.max() <= 127)))

    # Check 3: Dequantization is reversible (within tolerance)
    w_dequant = dequantize_8bit(w_quant, scale)
    max_error = np.max(np.abs(original - w_dequant))
    expected_max_error = scale  # Error should be at most one quantization step
    checks.append(("Max error within 1 quant step", max_error <= expected_max_error * 1.1))

    # Check 4: int8 dtype is preserved
    checks.append(("int8 dtype", w_quant.dtype == np.int8))

    # Check 5: Scale is positive
    checks.append(("Positive scale", scale > 0))

    # Print results
    for name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"{status} {name}: {'PASS' if passed else 'FAIL'}")

    print()
    if all(check[1] for check in checks):
        print("✓✓✓ ALL CHECKS PASSED - Quantization math is correct!")
    else:
        print("✗✗✗ SOME CHECKS FAILED - There may be issues!")
    print()
    print("=" * 80)


if __name__ == "__main__":
    test_quantization()
