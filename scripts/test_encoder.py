#!/usr/bin/env python3
"""
Test MLX Encoder Model

This script loads and tests MLX encoder models to verify they work correctly.
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any

import mlx.core as mx
from transformers import AutoTokenizer


def load_mlx_model(model_path: Path) -> tuple:
    """
    Load MLX encoder model from disk.

    Args:
        model_path: Path to MLX model directory

    Returns:
        Tuple of (weights, config, tokenizer)
    """
    # Load weights
    weights_file = model_path / "weights.npz"
    if not weights_file.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_file}")

    # Load numpy weights and convert to MLX
    np_weights = np.load(weights_file)
    mlx_weights = {k: mx.array(v) for k, v in np_weights.items()}

    # Load config
    config_file = model_path / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, 'r') as f:
        config = json.load(f)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    return mlx_weights, config, tokenizer


def test_model_loading(model_path: str):
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
            print(f"  - Quantization: {metadata['quantization']['bits']}-bit")
            print(f"  - Model size: {metadata['quantization']['actual_size_mb']:.1f}MB")
            print(f"  - Converted: {metadata['timestamp']}")

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
    Test basic embedding extraction from the model.

    Note: This is a simple forward pass test - full model inference
    would require implementing the full model architecture in MLX.
    """
    print(f"\nTesting embedding extraction")

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

        # Extract embeddings for input
        token_embeddings = embeddings[input_ids[0]]
        print(f"  - Extracted embeddings shape: {token_embeddings.shape}")

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

    args = parser.parse_args()

    print("="*60)
    print("MLX Encoder Model Test")
    print("="*60)

    # Run tests
    success = True

    # Test 1: Loading
    if not test_model_loading(args.model_path):
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
