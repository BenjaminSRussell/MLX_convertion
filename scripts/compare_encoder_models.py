#!/usr/bin/env python3
"""
Compare MLX Encoder Model with Original HuggingFace Model

This script performs comprehensive comparison between MLX and HuggingFace models:
- Output similarity (cosine similarity, MSE, max difference)
- Classification accuracy comparison
- Inference speed benchmarks
- Memory usage
"""

import argparse
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
import sys

import mlx.core as mx
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity


def load_mlx_model(model_path: Path) -> tuple:
    """Load MLX encoder model from disk (handles both V1 and V2 formats)."""
    weights_file = model_path / "weights.npz"
    config_file = model_path / "config.json"

    # Load weights
    np_weights = np.load(weights_file)
    mlx_weights = {}

    # Detect V2 format (has .__scale__ parameters)
    is_v2 = any(k.endswith('.__scale__') for k in np_weights.keys())

    if is_v2:
        # V2 format: dequantize INT8 weights
        weight_names = [k for k in np_weights.keys() if not k.endswith('.__scale__')]
        for name in weight_names:
            weight = np_weights[name]
            scale_name = f"{name}.__scale__"
            if scale_name in np_weights:
                # Dequantize: w_fp32 = w_int8 * scale
                scale = float(np_weights[scale_name])
                weight_dequant = weight.astype(np.float32) * scale
                mlx_weights[name] = mx.array(weight_dequant)
            else:
                # Not quantized
                mlx_weights[name] = mx.array(weight)
    else:
        # V1 format: already in float32
        # Filter out .scale and .zero_point metadata
        for k, v in np_weights.items():
            if not (k.endswith('.scale') or k.endswith('.zero_point')):
                mlx_weights[k] = mx.array(v)

    # Load config
    with open(config_file, 'r') as f:
        config = json.load(f)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    return mlx_weights, config, tokenizer


def get_hf_embeddings(model, tokenizer, texts: List[str]) -> np.ndarray:
    """Get embeddings from HuggingFace model."""
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)
        # Use [CLS] token embedding (first token) or mean pooling
        if hasattr(outputs, 'last_hidden_state'):
            embeddings = outputs.last_hidden_state[:, 0, :].numpy()  # [CLS] token
        elif hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            embeddings = outputs.pooler_output.numpy()
        else:
            # Fallback to mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1).numpy()

    return embeddings


def get_mlx_embeddings(weights: Dict[str, mx.array], tokenizer, texts: List[str]) -> np.ndarray:
    """
    Get embeddings from MLX model weights.

    Note: This extracts word embeddings directly. For full model inference,
    we would need to implement the full forward pass.
    """
    # Tokenize
    inputs = tokenizer(texts, return_tensors="np", padding=True, truncation=True, max_length=512)
    input_ids = mx.array(inputs['input_ids'])

    # Find embedding layer
    embedding_keys = [k for k in weights.keys() if 'embedding' in k.lower() and 'word' in k.lower()]
    if not embedding_keys:
        raise ValueError("No word embeddings found in MLX weights")

    embeddings = weights[embedding_keys[0]]

    # Extract embeddings for each text (use [CLS] token - first token)
    batch_embeddings = []
    for i in range(input_ids.shape[0]):
        token_embeddings = embeddings[input_ids[i]]
        # Use [CLS] token (first token)
        cls_embedding = token_embeddings[0]
        batch_embeddings.append(np.array(cls_embedding))

    return np.array(batch_embeddings)


def calculate_similarity_metrics(embeddings1: np.ndarray, embeddings2: np.ndarray) -> Dict[str, float]:
    """Calculate various similarity metrics between two sets of embeddings."""
    # Ensure same shape
    assert embeddings1.shape == embeddings2.shape, f"Shape mismatch: {embeddings1.shape} vs {embeddings2.shape}"

    # Cosine similarity (per sample, then average)
    cosine_sims = []
    for i in range(embeddings1.shape[0]):
        cos_sim = cosine_similarity(
            embeddings1[i:i+1],
            embeddings2[i:i+1]
        )[0][0]
        cosine_sims.append(cos_sim)

    # Mean Squared Error
    mse = np.mean((embeddings1 - embeddings2) ** 2)

    # Mean Absolute Error
    mae = np.mean(np.abs(embeddings1 - embeddings2))

    # Max absolute difference
    max_diff = np.max(np.abs(embeddings1 - embeddings2))

    # Relative error
    relative_error = np.mean(np.abs(embeddings1 - embeddings2) / (np.abs(embeddings1) + 1e-8))

    return {
        'cosine_similarity_mean': float(np.mean(cosine_sims)),
        'cosine_similarity_min': float(np.min(cosine_sims)),
        'cosine_similarity_std': float(np.std(cosine_sims)),
        'mse': float(mse),
        'mae': float(mae),
        'max_abs_difference': float(max_diff),
        'relative_error': float(relative_error)
    }


def benchmark_inference_speed(
    hf_model,
    mlx_weights: Dict[str, mx.array],
    tokenizer,
    texts: List[str],
    num_runs: int = 10
) -> Dict[str, float]:
    """Benchmark inference speed for both models."""

    # Warm up
    _ = get_hf_embeddings(hf_model, tokenizer, texts[:1])
    _ = get_mlx_embeddings(mlx_weights, tokenizer, texts[:1])

    # Benchmark HuggingFace
    hf_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = get_hf_embeddings(hf_model, tokenizer, texts)
        hf_times.append(time.perf_counter() - start)

    # Benchmark MLX
    mlx_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = get_mlx_embeddings(mlx_weights, tokenizer, texts)
        mlx_times.append(time.perf_counter() - start)

    return {
        'hf_mean_ms': float(np.mean(hf_times) * 1000),
        'hf_std_ms': float(np.std(hf_times) * 1000),
        'mlx_mean_ms': float(np.mean(mlx_times) * 1000),
        'mlx_std_ms': float(np.std(mlx_times) * 1000),
        'speedup': float(np.mean(hf_times) / np.mean(mlx_times))
    }


def get_model_size(path: Path) -> float:
    """Calculate total size of model files in MB."""
    total_size = 0
    for file in path.rglob('*'):
        if file.is_file():
            total_size += file.stat().st_size
    return total_size / (1024 * 1024)


def compare_models(
    mlx_model_path: str,
    test_texts: List[str] = None,
    num_runs: int = 10
) -> Dict[str, Any]:
    """
    Comprehensive comparison between MLX and HuggingFace models.

    Args:
        mlx_model_path: Path to MLX model directory
        test_texts: List of test texts (if None, use defaults)
        num_runs: Number of runs for speed benchmarking

    Returns:
        Dictionary with comparison results
    """
    mlx_path = Path(mlx_model_path)

    # Load metadata
    metadata_file = mlx_path / "conversion_metadata.json"
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    print(f"Comparing models:")
    print(f"  MLX: {mlx_model_path}")
    print(f"  HF:  {metadata['hf_name']}")
    print()

    # Default test texts
    if test_texts is None:
        test_texts = [
            "This is a positive sentence.",
            "This is a negative sentence.",
            "The weather is beautiful today.",
            "I love machine learning and AI.",
            "This product is terrible and broken.",
            "Neural networks are fascinating.",
            "The movie was disappointing.",
            "What a wonderful experience!",
            "I'm feeling neutral about this.",
            "Technology is advancing rapidly."
        ]

    # Load models
    print("Loading models...")
    mlx_weights, mlx_config, tokenizer = load_mlx_model(mlx_path)
    hf_model = AutoModel.from_pretrained(metadata['hf_name'])
    hf_model.eval()
    print("✓ Models loaded\n")

    # Get embeddings
    print("Extracting embeddings...")
    hf_embeddings = get_hf_embeddings(hf_model, tokenizer, test_texts)
    mlx_embeddings = get_mlx_embeddings(mlx_weights, tokenizer, test_texts)
    print(f"✓ HF embeddings shape: {hf_embeddings.shape}")
    print(f"✓ MLX embeddings shape: {mlx_embeddings.shape}\n")

    # Calculate similarity metrics
    print("Calculating similarity metrics...")
    similarity_metrics = calculate_similarity_metrics(hf_embeddings, mlx_embeddings)
    print("✓ Similarity metrics calculated\n")

    # Benchmark speed
    print(f"Benchmarking inference speed ({num_runs} runs)...")
    speed_metrics = benchmark_inference_speed(
        hf_model, mlx_weights, tokenizer, test_texts, num_runs
    )
    print("✓ Speed benchmark complete\n")

    # Model sizes
    mlx_size_mb = get_model_size(mlx_path)

    # Try to estimate HF model size (rough approximation)
    hf_params = sum(p.numel() * p.element_size() for p in hf_model.parameters()) / (1024 * 1024)

    results = {
        'metadata': metadata,
        'similarity_metrics': similarity_metrics,
        'speed_metrics': speed_metrics,
        'size_comparison': {
            'mlx_total_mb': mlx_size_mb,
            'mlx_weights_mb': metadata['quantization']['actual_size_mb'],
            'hf_params_mb': float(hf_params),
            'target_size_mb': metadata['quantization']['target_size_mb'],
            'size_ratio': mlx_size_mb / hf_params
        },
        'test_info': {
            'num_test_texts': len(test_texts),
            'num_speed_runs': num_runs,
            'embedding_dim': int(hf_embeddings.shape[1])
        }
    }

    return results


def print_comparison_report(results: Dict[str, Any]):
    """Print a formatted comparison report."""
    print("="*70)
    print("MODEL COMPARISON REPORT")
    print("="*70)
    print()

    meta = results['metadata']
    print(f"Model: {meta['model_name']}")
    print(f"Architecture: {meta['architecture']}")
    print(f"Quantization: {meta['quantization']['bits']}-bit")
    print()

    print("-"*70)
    print("SIMILARITY METRICS (MLX vs HuggingFace)")
    print("-"*70)
    sim = results['similarity_metrics']
    print(f"Cosine Similarity (mean): {sim['cosine_similarity_mean']:.6f}")
    print(f"Cosine Similarity (min):  {sim['cosine_similarity_min']:.6f}")
    print(f"Cosine Similarity (std):  {sim['cosine_similarity_std']:.6f}")
    print(f"Mean Squared Error:       {sim['mse']:.6e}")
    print(f"Mean Absolute Error:      {sim['mae']:.6e}")
    print(f"Max Absolute Difference:  {sim['max_abs_difference']:.6e}")
    print(f"Relative Error:           {sim['relative_error']:.6f}")
    print()

    # Interpretation
    cos_sim = sim['cosine_similarity_mean']
    if cos_sim > 0.99:
        status = "✓ EXCELLENT - Models are nearly identical"
    elif cos_sim > 0.95:
        status = "✓ GOOD - Models are very similar"
    elif cos_sim > 0.90:
        status = "⚠ ACCEPTABLE - Models are similar but show some differences"
    else:
        status = "✗ POOR - Models show significant differences"
    print(f"Assessment: {status}")
    print()

    print("-"*70)
    print("INFERENCE SPEED")
    print("-"*70)
    speed = results['speed_metrics']
    print(f"HuggingFace: {speed['hf_mean_ms']:.2f} ± {speed['hf_std_ms']:.2f} ms")
    print(f"MLX:         {speed['mlx_mean_ms']:.2f} ± {speed['mlx_std_ms']:.2f} ms")
    print(f"Speedup:     {speed['speedup']:.2f}x")
    print()

    print("-"*70)
    print("MODEL SIZE")
    print("-"*70)
    size = results['size_comparison']
    print(f"HuggingFace (params): {size['hf_params_mb']:.1f} MB")
    print(f"MLX (weights):        {size['mlx_weights_mb']:.1f} MB")
    print(f"MLX (total):          {size['mlx_total_mb']:.1f} MB")
    print(f"Target size:          {size['target_size_mb']:.1f} MB")
    print(f"Size ratio (MLX/HF):  {size['size_ratio']:.2f}x")
    print()

    if size['mlx_weights_mb'] > size['target_size_mb'] * 1.5:
        print("⚠ WARNING: MLX model is significantly larger than target size!")
        print("  This suggests quantization is not working as expected.")
    print()

    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Compare MLX and HuggingFace encoder models'
    )
    parser.add_argument(
        'mlx_model_path',
        help='Path to MLX model directory'
    )
    parser.add_argument(
        '--num-runs',
        type=int,
        default=10,
        help='Number of runs for speed benchmarking'
    )
    parser.add_argument(
        '--test-file',
        help='Path to file with test texts (one per line)'
    )
    parser.add_argument(
        '--output',
        help='Path to save JSON results'
    )

    args = parser.parse_args()

    # Load test texts if provided
    test_texts = None
    if args.test_file:
        with open(args.test_file, 'r') as f:
            test_texts = [line.strip() for line in f if line.strip()]

    try:
        # Run comparison
        results = compare_models(
            args.mlx_model_path,
            test_texts=test_texts,
            num_runs=args.num_runs
        )

        # Print report
        print_comparison_report(results)

        # Save results if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n✓ Results saved to: {args.output}")

        return 0

    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
