#!/usr/bin/env python3
"""
Verify Quantization Accuracy - Weight-Level Comparison

This script properly tests quantization accuracy by comparing weights
at the same level (HF original weights vs MLX dequantized weights).

Unlike compare_encoder_models.py, this compares the RIGHT things:
- Original float32 weights from HuggingFace
- Dequantized float32 weights from MLX
- Both are weight matrices, not model outputs
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
import sys

from transformers import AutoModel


def load_mlx_weights(model_path: Path) -> Dict[str, np.ndarray]:
    """
    Load and dequantize MLX weights.

    Returns dequantized float32 weights for comparison.
    """
    weights_file = model_path / "weights.npz"
    if not weights_file.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_file}")

    np_weights = np.load(weights_file)
    dequantized_weights = {}

    # Get metadata
    metadata_file = model_path / "conversion_metadata.json"
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    # Dequantize INT8 weights
    weight_names = [k for k in np_weights.keys() if not k.endswith('.__scale__')]

    for name in weight_names:
        weight = np_weights[name]

        # Check if quantized
        scale_name = f"{name}.__scale__"
        if scale_name in np_weights:
            # Dequantize: w_fp32 = w_int8 * scale
            scale = float(np_weights[scale_name])
            weight_dequant = weight.astype(np.float32) * scale
            dequantized_weights[name] = weight_dequant
        else:
            # Not quantized (embeddings, norms, biases)
            dequantized_weights[name] = weight.astype(np.float32)

    return dequantized_weights, metadata


def load_hf_weights(hf_name: str) -> Dict[str, np.ndarray]:
    """Load original HuggingFace weights."""
    model = AutoModel.from_pretrained(hf_name)

    hf_weights = {}
    for name, param in model.state_dict().items():
        hf_weights[name] = param.detach().cpu().numpy()

    return hf_weights


def compare_weights(
    hf_weights: Dict[str, np.ndarray],
    mlx_weights: Dict[str, np.ndarray]
) -> Dict[str, Any]:
    """
    Compare weights layer by layer.

    Returns detailed error metrics for each layer.
    """
    results = {
        'per_layer': [],
        'overall': {},
        'quantized_layers': [],
        'unquantized_layers': []
    }

    all_mse = []
    all_mae = []
    all_max_errors = []
    all_rel_errors = []

    # Find common keys
    hf_keys = set(hf_weights.keys())
    mlx_keys = set(mlx_weights.keys())
    common_keys = hf_keys.intersection(mlx_keys)

    print(f"\nComparing {len(common_keys)} weight matrices...")
    print("-" * 80)

    for name in sorted(common_keys):
        hf_w = hf_weights[name]
        mlx_w = mlx_weights[name]

        # Check shapes match
        if hf_w.shape != mlx_w.shape:
            print(f"⚠️  Shape mismatch for {name}: {hf_w.shape} vs {mlx_w.shape}")
            continue

        # Calculate errors
        diff = hf_w - mlx_w
        mse = float(np.mean(diff ** 2))
        mae = float(np.mean(np.abs(diff)))
        max_error = float(np.max(np.abs(diff)))

        # Relative error (avoid division by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_error = np.abs(diff) / (np.abs(hf_w) + 1e-10)
            rel_error = float(np.mean(rel_error[np.isfinite(rel_error)]))

        # Determine if this layer was quantized
        # (quantized layers will have non-zero error)
        is_quantized = mse > 1e-10

        layer_result = {
            'name': name,
            'shape': hf_w.shape,
            'mse': mse,
            'mae': mae,
            'max_error': max_error,
            'relative_error': rel_error,
            'is_quantized': is_quantized
        }

        results['per_layer'].append(layer_result)

        if is_quantized:
            results['quantized_layers'].append(name)
            all_mse.append(mse)
            all_mae.append(mae)
            all_max_errors.append(max_error)
            all_rel_errors.append(rel_error)
        else:
            results['unquantized_layers'].append(name)

    # Overall statistics (only for quantized layers)
    if all_mse:
        results['overall'] = {
            'num_quantized_layers': len(all_mse),
            'num_unquantized_layers': len(results['unquantized_layers']),
            'mse_mean': float(np.mean(all_mse)),
            'mse_max': float(np.max(all_mse)),
            'mae_mean': float(np.mean(all_mae)),
            'mae_max': float(np.max(all_mae)),
            'max_error_mean': float(np.mean(all_max_errors)),
            'max_error_max': float(np.max(all_max_errors)),
            'relative_error_mean': float(np.mean(all_rel_errors)),
            'relative_error_max': float(np.max(all_rel_errors))
        }

    return results


def print_comparison_report(results: Dict[str, Any], metadata: Dict[str, Any]):
    """Print detailed comparison report."""
    print("\n" + "=" * 80)
    print("QUANTIZATION ACCURACY REPORT (Weight-Level Comparison)")
    print("=" * 80)
    print()

    print(f"Model: {metadata['model_name']}")
    print(f"Architecture: {metadata['architecture']}")
    print(f"Quantization: {metadata['quantization']['bits']}-bit ({metadata['quantization']['dtype']})")
    method = metadata['quantization'].get('method', 'symmetric')  # Default to symmetric for old metadata
    print(f"Method: {method}")
    print()

    overall = results['overall']
    print("-" * 80)
    print("OVERALL QUANTIZATION ERROR (Quantized Layers Only)")
    print("-" * 80)
    print(f"Quantized layers: {overall['num_quantized_layers']}")
    print(f"Unquantized layers: {overall['num_unquantized_layers']}")
    print()
    print(f"Mean Squared Error (MSE):")
    print(f"  Mean: {overall['mse_mean']:.2e}")
    print(f"  Max:  {overall['mse_max']:.2e}")
    print()
    print(f"Mean Absolute Error (MAE):")
    print(f"  Mean: {overall['mae_mean']:.2e}")
    print(f"  Max:  {overall['mae_max']:.2e}")
    print()
    print(f"Max Absolute Error:")
    print(f"  Mean: {overall['max_error_mean']:.4f}")
    print(f"  Max:  {overall['max_error_max']:.4f}")
    print()
    print(f"Relative Error:")
    print(f"  Mean: {overall['relative_error_mean']:.4f} ({overall['relative_error_mean']*100:.2f}%)")
    print(f"  Max:  {overall['relative_error_max']:.4f} ({overall['relative_error_max']*100:.2f}%)")
    print()

    # Quality assessment
    print("-" * 80)
    print("QUALITY ASSESSMENT")
    print("-" * 80)

    # 8-bit quantization typical errors
    expected_8bit = {
        'mse': 1e-4,
        'relative_error': 0.01  # 1%
    }

    status = []
    if overall['mse_mean'] < expected_8bit['mse']:
        status.append("✓ MSE is within expected range for 8-bit quantization")
    else:
        status.append("⚠️  MSE is higher than expected")

    if overall['relative_error_mean'] < expected_8bit['relative_error']:
        status.append("✓ Relative error is within expected range (<1%)")
    else:
        status.append("⚠️  Relative error is higher than expected")

    if overall['max_error_max'] < 0.1:
        status.append("✓ Max absolute error is acceptably small")
    else:
        status.append("⚠️  Max absolute error is larger than expected")

    for s in status:
        print(s)
    print()

    # Overall verdict
    if overall['relative_error_mean'] < 0.01 and overall['mse_mean'] < 1e-4:
        print("✓✓✓ EXCELLENT: Quantization is high quality with minimal error")
    elif overall['relative_error_mean'] < 0.02:
        print("✓✓ GOOD: Quantization error is acceptable")
    elif overall['relative_error_mean'] < 0.05:
        print("✓ ACCEPTABLE: Quantization error is within reasonable bounds")
    else:
        print("✗ POOR: Quantization error is too high")
    print()

    # Show worst layers
    print("-" * 80)
    print("TOP 5 LAYERS WITH HIGHEST ERROR")
    print("-" * 80)

    quantized_layers = [l for l in results['per_layer'] if l['is_quantized']]
    sorted_layers = sorted(quantized_layers, key=lambda x: x['mse'], reverse=True)[:5]

    for i, layer in enumerate(sorted_layers, 1):
        print(f"{i}. {layer['name']}")
        print(f"   Shape: {layer['shape']}")
        print(f"   MSE: {layer['mse']:.2e}, MAE: {layer['mae']:.2e}, Max: {layer['max_error']:.4f}")
        print(f"   Relative error: {layer['relative_error']:.2%}")
        print()

    # Show unquantized layers
    if results['unquantized_layers']:
        print("-" * 80)
        print(f"UNQUANTIZED LAYERS ({len(results['unquantized_layers'])})")
        print("-" * 80)
        print("These layers kept full precision (embeddings, norms, biases):")
        for name in results['unquantized_layers'][:10]:
            print(f"  - {name}")
        if len(results['unquantized_layers']) > 10:
            print(f"  ... and {len(results['unquantized_layers']) - 10} more")
        print()

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Verify quantization accuracy by comparing weights'
    )
    parser.add_argument(
        'mlx_model_path',
        help='Path to MLX model directory'
    )
    parser.add_argument(
        '--output',
        help='Path to save JSON results'
    )
    parser.add_argument(
        '--show-all',
        action='store_true',
        help='Show all layers (not just top 5)'
    )

    args = parser.parse_args()

    mlx_path = Path(args.mlx_model_path)
    if not mlx_path.exists():
        print(f"ERROR: MLX model path does not exist: {mlx_path}", file=sys.stderr)
        return 1

    try:
        # Load metadata to get HF model name
        metadata_file = mlx_path / "conversion_metadata.json"
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        print(f"Loading models...")
        print(f"  MLX: {args.mlx_model_path}")
        print(f"  HF:  {metadata['hf_name']}")

        # Load weights
        mlx_weights, metadata = load_mlx_weights(mlx_path)
        hf_weights = load_hf_weights(metadata['hf_name'])

        print(f"✓ Loaded {len(hf_weights)} HF weights")
        print(f"✓ Loaded {len(mlx_weights)} MLX weights")

        # Compare
        results = compare_weights(hf_weights, mlx_weights)

        # Print report
        print_comparison_report(results, metadata)

        # Save results
        if args.output:
            output_data = {
                'metadata': metadata,
                'results': results
            }
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\n✓ Results saved to: {args.output}")

        return 0

    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
