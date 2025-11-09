#!/usr/bin/env python3
"""
Test MLX Encoder Model

This script loads and tests MLX encoder models with TRUE INT8 quantization.
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any

import mlx.core as mx
from transformers import AutoTokenizer


def load_quantized_weights(weights_file: Path) -> Dict[str, mx.array]:
    """
    Load quantized weights and dequantize them to MLX arrays.

    Args:
        weights_file: Path to weights.npz file

    Returns:
        Dictionary of dequantized MLX arrays ready for inference
    """
    # Load all weights
    np_weights = np.load(weights_file)
    mlx_weights = {}

    # Separate regular weights from scale/zero-point metadata
    weight_names = [k for k in np_weights.keys() if not k.endswith('.__scale__')]

    for name in weight_names:
        weight = np_weights[name]

        # Check if this weight was quantized
        scale_name = f"{name}.__scale__"
        if scale_name in np_weights:
            # This was a quantized weight, dequantize it
            scale = float(np_weights[scale_name])

            # Dequantize: w_fp32 = w_int8 * scale
            weight_dequant = weight.astype(np.float32) * scale
            mlx_weights[name] = mx.array(weight_dequant)
        else:
            # Not quantized, use as-is
            mlx_weights[name] = mx.array(weight)

    return mlx_weights


def load_mlx_model(model_path: Path) -> tuple:
    """
    Load MLX encoder model from disk with dequantization.

    Args:
        model_path: Path to MLX model directory

    Returns:
        Tuple of (weights, config, tokenizer)
    """
    # Load weights with dequantization
    weights_file = model_path / "weights.npz"
    if not weights_file.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_file}")

    mlx_weights = load_quantized_weights(weights_file)

    # Load config
    config_file = model_path / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, 'r') as f:
        config = json.load(f)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    return mlx_weights, config, tokenizer


def analyze_quantization(model_path: Path):
    """Analyze the quantization of the model."""
    weights_file = model_path / "weights.npz"
    np_weights = np.load(weights_file)

    print("\nQuantization Analysis:")
    print("-" * 60)

    # Count different types of weights
    int8_count = 0
    float32_count = 0
    scale_count = 0
    int8_size_mb = 0
    float32_size_mb = 0
    total_size_mb = 0

    for name, weight in np_weights.items():
        size_mb = weight.nbytes / (1024 * 1024)
        total_size_mb += size_mb

        if name.endswith('.__scale__'):
            scale_count += 1
        elif weight.dtype == np.int8:
            int8_count += 1
            int8_size_mb += size_mb
        else:
            float32_count += 1
            float32_size_mb += size_mb

    print(f"Total parameters: {len(np_weights)}")
    print(f"INT8 quantized layers: {int8_count}")
    print(f"Float32 layers: {float32_count}")
    print(f"Scale parameters: {scale_count}")
    print()
    print(f"INT8 layers size: {int8_size_mb:.2f} MB")
    print(f"Float32 layers size: {float32_size_mb:.2f} MB")
    print(f"Total size: {total_size_mb:.2f} MB")
    print()

    # Show sample quantized weights
    print("Sample quantized weights:")
    shown = 0
    for name, weight in np_weights.items():
        if weight.dtype == np.int8 and shown < 3:
            scale_name = f"{name}.__scale__"
            if scale_name in np_weights:
                scale = np_weights[scale_name]
                print(f"  {name}:")
                print(f"    Shape: {weight.shape}")
                print(f"    Dtype: {weight.dtype}")
                print(f"    Range: [{weight.min()}, {weight.max()}]")
                print(f"    Scale: {float(scale):.6f}")
                shown += 1


def test_model_loading(model_path: str, verbose: bool = False):
    """Test loading the MLX model."""
    print(f"Testing model loading from: {model_path}")

    model_path = Path(model_path)
    if not model_path.exists():
        print(f"ERROR: Model path does not exist: {model_path}")
        return False

    try:
        weights, config, tokenizer = load_mlx_model(model_path)

        print(f"✓ Successfully loaded model")
        print(f"  - Config: {config.get('model_type', 'unknown')} ({config.get('_name_or_path', 'N/A')})")
        print(f"  - Weights: {len(weights)} parameters")
        print(f"  - Tokenizer: {len(tokenizer)} tokens")

        # Load metadata if available
        metadata_file = model_path / "conversion_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            print(f"  - Quantization: {metadata['quantization']['bits']}-bit ({metadata['quantization']['dtype']})")
            print(f"  - Original size: {metadata['quantization']['original_size_mb']:.1f}MB")
            print(f"  - Quantized size: {metadata['quantization']['actual_size_mb']:.1f}MB")
            print(f"  - Compression: {metadata['quantization']['compression_ratio']:.2f}x")
            print(f"  - Converted: {metadata['timestamp']}")
            print(f"  - Converter: {metadata.get('converter_version', 'v1')}")

        if verbose:
            analyze_quantization(model_path)

        return True

    except Exception as e:
        print(f"ERROR: Failed to load model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_tokenization(model_path: str, text: str = "This is a test sentence."):
    """Test tokenization with the model's tokenizer."""
    print(f"\nTesting tokenization with text: '{text}'")

    model_path = Path(model_path)

    try:
        _, _, tokenizer = load_mlx_model(model_path)

        # Tokenize
        inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True)

        print(f"✓ Tokenization successful")
        print(f"  - Input IDs shape: {inputs['input_ids'].shape}")
        print(f"  - Tokens: {tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])[:10]}...")

        return True

    except Exception as e:
        print(f"ERROR: Tokenization failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_embedding_extraction(model_path: str, text: str = "This is a test sentence."):
    """
    Test basic embedding extraction from the dequantized model.
    """
    print(f"\nTesting embedding extraction (with dequantization)")

    model_path = Path(model_path)

    try:
        weights, config, tokenizer = load_mlx_model(model_path)

        # Tokenize
        inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True)
        input_ids = mx.array(inputs['input_ids'])

        # Try to extract word embeddings (if available)
        embedding_keys = [k for k in weights.keys() if 'embedding' in k.lower() and 'word' in k.lower()]

        if not embedding_keys:
            print("  - No word embeddings found in weights")
            return True

        # Get first embedding layer
        embedding_key = embedding_keys[0]
        embeddings = weights[embedding_key]

        print(f"✓ Found embeddings layer: {embedding_key}")
        print(f"  - Embedding shape: {embeddings.shape}")
        print(f"  - Vocab size: {embeddings.shape[0]}")
        print(f"  - Hidden size: {embeddings.shape[1]}")
        print(f"  - Dtype: {embeddings.dtype}")

        # Extract embeddings for input
        token_embeddings = embeddings[input_ids[0]]
        print(f"  - Extracted embeddings shape: {token_embeddings.shape}")

        # Show sample values
        print(f"  - Sample embedding values: {np.array(token_embeddings[0])[:5]}")

        return True

    except Exception as e:
        print(f"ERROR: Embedding extraction failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Test MLX encoder models')
    parser.add_argument(
        'model_path',
        help='Path to MLX model directory'
    )
    parser.add_argument(
        '--text',
        default="This is a test sentence.",
        help='Test text for tokenization'
    )
    parser.add_argument(
        '--full-test',
        action='store_true',
        help='Run all tests'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed quantization analysis'
    )

    args = parser.parse_args()

    print("="*60)
    print("MLX Encoder Model Test")
    print("="*60)

    # Run tests
    success = True

    # Test 1: Loading
    if not test_model_loading(args.model_path, args.verbose):
        success = False

    if args.full_test:
        # Test 2: Tokenization
        if not test_tokenization(args.model_path, args.text):
            success = False

        # Test 3: Embedding extraction
        if not test_embedding_extraction(args.model_path, args.text):
            success = False

    print("\n" + "="*60)
    if success:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
    print("="*60)

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
