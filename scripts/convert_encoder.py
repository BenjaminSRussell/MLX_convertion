#!/usr/bin/env python3
"""
MLX Encoder Model Converter - With Proper Quantization

Implements TRUE quantization with INT8 storage for actual size reduction.

Features:
- Stores quantized weights as INT8 (1 byte) instead of float32 (4 bytes)
- Achieves 4x size reduction with 8-bit quantization
- Maintains accuracy with proper scale/zero-point handling
"""

import argparse
import json
import logging
import os
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import mlx.core as mx
import numpy as np
import yaml
from transformers import AutoModel, AutoTokenizer, AutoConfig


def setup_logging(model_name=None):
    """Set up logging for the conversion process."""
    temp_dir = os.environ.get('MLX_TEMP_DIR', tempfile.gettempdir())
    log_dir = Path(temp_dir) / 'mlx_conversion' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"encoder_conversion_{model_name}.log" if model_name else log_dir / "encoder_conversion.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(f'MLXEncoderConverter_{model_name}' if model_name else 'MLXEncoderConverter')


def quantize_weights_int8(
    weights: Dict[str, mx.array],
    bits: int = 8
) -> Dict[str, np.ndarray]:
    """
    Quantize model weights to int8/int4 for size reduction.

    Args:
        weights: Dictionary of weight tensors
        bits: Number of bits for quantization (4 or 8)

    Returns:
        Dictionary with quantized weights and scales
    """
    quantized_data = {}

    for name, weight in weights.items():
        weight_np = np.array(weight)

        # Skip embeddings, layer norms, biases, and problematic ffn layers
        if any(skip in name.lower() for skip in ['embedding', 'layernorm', 'bias', 'norm', 'ln', 'ffn.lin2']):
            quantized_data[name] = weight_np.astype(np.float32)
            continue

        # Only quantize 2D matrices
        if len(weight_np.shape) != 2:
            quantized_data[name] = weight_np.astype(np.float32)
            continue

        if bits == 8:
            # Symmetric quantization to [-127, 127]
            w_max = np.abs(weight_np).max()
            scale = w_max / 127.0 if w_max > 0 else 1.0

            w_quant = np.round(weight_np / scale)
            w_quant = np.clip(w_quant, -127, 127).astype(np.int8)

            quantized_data[name] = w_quant
            quantized_data[f"{name}.__scale__"] = np.array(scale, dtype=np.float32)

        elif bits == 4:
            # Symmetric quantization to [-7, 7]
            w_max = np.abs(weight_np).max()
            scale = w_max / 7.0 if w_max > 0 else 1.0

            w_quant = np.round(weight_np / scale)
            w_quant = np.clip(w_quant, -7, 7).astype(np.int8)

            quantized_data[name] = w_quant
            quantized_data[f"{name}.__scale__"] = np.array(scale, dtype=np.float32)

        else:
            quantized_data[name] = weight_np.astype(np.float32)

    return quantized_data


def convert_pytorch_to_mlx(state_dict: Dict[str, Any]) -> Dict[str, mx.array]:
    """
    Convert PyTorch state dict to MLX arrays.

    Args:
        state_dict: PyTorch model state dictionary

    Returns:
        Dictionary of MLX arrays
    """
    mlx_weights = {}

    for name, param in state_dict.items():
        # Convert PyTorch tensor to numpy, then to MLX
        if hasattr(param, 'detach'):
            param_np = param.detach().cpu().numpy()
        else:
            param_np = np.array(param)

        # Convert to MLX array
        mlx_weights[name] = mx.array(param_np)

    return mlx_weights


def calculate_model_size(weights: Dict[str, np.ndarray]) -> float:
    """Calculate total model size in MB."""
    total_bytes = sum(w.nbytes for w in weights.values())
    return total_bytes / (1024 * 1024)


def migrate_old_metadata(metadata: Dict[str, Any], output_path: Path) -> Dict[str, Any]:
    """
    Migrate old metadata to include missing fields.

    Args:
        metadata: Existing metadata dictionary
        output_path: Path to the converted model

    Returns:
        Updated metadata dictionary
    """
    updated = False

    # Ensure 'method' field exists
    if 'method' not in metadata.get('quantization', {}):
        if 'quantization' not in metadata:
            metadata['quantization'] = {}
        metadata['quantization']['method'] = 'symmetric'
        updated = True

    # Calculate original_size_mb if missing
    if 'original_size_mb' not in metadata.get('quantization', {}):
        weights_file = output_path / 'weights.npz'
        if weights_file.exists():
            try:
                with np.load(weights_file) as np_weights:
                    quantized_params = 0
                    unquantized_params = 0

                    for key in np_weights.keys():
                        if key.endswith('.__scale__'):
                            continue
                        weight = np_weights[key]
                        if f"{key}.__scale__" in np_weights:
                            quantized_params += weight.size
                        else:
                            unquantized_params += weight.size

                    # All params are fp32 originally (4 bytes each)
                    original_bytes = (quantized_params + unquantized_params) * 4
                    original_size_mb = original_bytes / (1024 * 1024)

                metadata['quantization']['original_size_mb'] = original_size_mb
                updated = True
            except Exception:
                pass

    # Recalculate compression ratio if both sizes available
    if 'original_size_mb' in metadata['quantization'] and 'actual_size_mb' in metadata['quantization']:
        original = metadata['quantization']['original_size_mb']
        actual = metadata['quantization']['actual_size_mb']
        if actual > 0:
            new_ratio = original / actual
            if 'compression_ratio' not in metadata['quantization'] or \
               abs(metadata['quantization']['compression_ratio'] - new_ratio) > 0.01:
                metadata['quantization']['compression_ratio'] = new_ratio
                updated = True

    if 'converter_version' not in metadata:
        metadata['converter_version'] = 'v1'
        updated = True

    # Save updated metadata if changes were made
    if updated:
        metadata_file = output_path / 'conversion_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    return metadata


def save_mlx_model(
    weights: Dict[str, np.ndarray],
    config: Dict[str, Any],
    tokenizer,
    output_path: Path,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Save MLX model with INT8 quantized weights.

    Args:
        weights: Quantized model weights (numpy arrays, including int8)
        config: Model configuration
        tokenizer: HuggingFace tokenizer
        output_path: Output directory
        metadata: Additional metadata to save
    """
    output_path.mkdir(parents=True, exist_ok=True)

    # Save weights in compressed NPZ format
    weights_file = output_path / "weights.npz"
    np.savez_compressed(weights_file, **weights)

    # Save config
    config_file = output_path / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    # Save tokenizer
    tokenizer.save_pretrained(output_path)

    # Save metadata
    if metadata:
        metadata_file = output_path / "conversion_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)


def load_quantized_weights(weights_file: Path) -> Dict[str, mx.array]:
    """
    Load quantized weights and dequantize them to MLX arrays.

    Args:
        weights_file: Path to weights.npz file

    Returns:
        Dictionary of dequantized MLX arrays ready for inference
    """
    mlx_weights = {}

    # Load all weights with proper resource management
    with np.load(weights_file) as np_weights:
        # Separate regular weights from scale/zero-point metadata
        weight_names = [k for k in np_weights.keys() if not k.endswith('.__scale__')]

        for name in weight_names:
            weight = np_weights[name]

            # Check if this weight was quantized
            scale_name = f"{name}.__scale__"
            if scale_name in np_weights:
                # This was a quantized weight, dequantize it
                scale = float(np_weights[scale_name])

                weight_dequant = weight.astype(np.float32) * scale
                mlx_weights[name] = mx.array(weight_dequant)
            else:
                mlx_weights[name] = mx.array(weight)

    return mlx_weights


class EncoderModelConverter:
    """Encoder model converter with int8 quantization"""

    def __init__(
        self,
        config_path: str = 'config/models.yaml',
        output_dir: str = 'models/mlx_converted',
        dry_run: bool = False,
        skip_existing: bool = True
    ):
        # Load config
        self.config = {'models': []}
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                cfg_data = yaml.safe_load(f)
                if 'models' in cfg_data:
                    self.config['models'] = cfg_data['models']

        temp_dir = os.environ.get('MLX_TEMP_DIR', tempfile.gettempdir())
        self.results_dir = Path(temp_dir) / 'mlx_conversion' / 'results'
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.output_dir = Path(output_dir)
        self.logger = setup_logging()
        self.dry_run = dry_run
        self.skip_existing = skip_existing

    def convert_single_model(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a single encoder model to MLX format with TRUE quantization."""
        model_name = model_config['name']
        logger = setup_logging(model_name)
        start_time = time.time()
        quant_config = model_config['quantization']

        logger.info(f"Starting encoder model conversion: {model_name}")
        logger.info(f"Target quantization: {quant_config['bits']}-bit")

        # Create output directory
        output_path = self.output_dir / f"{model_name}-mlx-q{quant_config['bits']}"

        # Check if model already exists
        if self.skip_existing and output_path.exists():
            metadata_file = output_path / 'conversion_metadata.json'
            if metadata_file.exists():
                logger.info(f"Skipping {model_name} - already converted at {output_path}")
                with open(metadata_file, 'r') as f:
                    existing_metadata = json.load(f)

                # Migrate old metadata to include missing fields
                existing_metadata = migrate_old_metadata(existing_metadata, output_path)

                return {
                    'success': True,
                    'skipped': True,
                    'metadata': existing_metadata,
                    'output_path': str(output_path)
                }

        # Remove incomplete conversions
        if output_path.exists():
            logger.info(f"Removing incomplete conversion at {output_path}")
            shutil.rmtree(output_path)

        if self.dry_run:
            logger.info("Dry-run enabled, skipping execution")
            return {
                'success': True,
                'dry_run': True,
                'model_name': model_name
            }

        try:
            # Load model and tokenizer from HuggingFace
            logger.info(f"Loading model from HuggingFace: {model_config['hf_name']}")
            hf_model = AutoModel.from_pretrained(model_config['hf_name'])
            tokenizer = AutoTokenizer.from_pretrained(model_config['hf_name'])
            config = AutoConfig.from_pretrained(model_config['hf_name'])

            # Convert to MLX format
            logger.info("Converting PyTorch weights to MLX format")
            mlx_weights = convert_pytorch_to_mlx(hf_model.state_dict())

            original_size_mb = sum(
                np.array(v).nbytes for v in mlx_weights.values()
            ) / (1024 * 1024)

            if quant_config['bits'] < 32:
                logger.info(f"Applying {quant_config['bits']}-bit quantization")
                quantized_weights = quantize_weights_int8(
                    mlx_weights,
                    bits=quant_config['bits']
                )
            else:
                logger.info("No quantization")
                quantized_weights = {k: np.array(v) for k, v in mlx_weights.items()}

            quantized_size_mb = calculate_model_size(quantized_weights)

            config_dict = config.to_dict()
            config_dict['quantization'] = {
                'bits': quant_config['bits'],
                'dtype': 'int8' if quant_config['bits'] <= 8 else quant_config['dtype'],
                'method': 'symmetric'
            }

            logger.info(f"Saving MLX model to {output_path}")

            duration = time.time() - start_time

            metadata = {
                'model_name': model_name,
                'hf_name': model_config['hf_name'],
                'model_type': 'encoder',
                'architecture': config_dict.get('model_type', 'unknown'),
                'quantization': {
                    'bits': quant_config['bits'],
                    'dtype': 'int8' if quant_config['bits'] <= 8 else 'float32',
                    'method': 'symmetric',
                    'target_size_mb': quant_config['target_size_mb'],
                    'actual_size_mb': quantized_size_mb,
                    'original_size_mb': original_size_mb,
                    'compression_ratio': original_size_mb / quantized_size_mb if quantized_size_mb > 0 else 1.0
                },
                'conversion_time_seconds': duration,
                'timestamp': datetime.now().isoformat(),
                'mlx_version': mx.__version__,
                'device': str(mx.default_device()),
                'size_target_ratio': quantized_size_mb / quant_config['target_size_mb'] if quant_config['target_size_mb'] > 0 else 0.0,
                'converter_version': 'v2'
            }

            save_mlx_model(quantized_weights, config_dict, tokenizer, output_path, metadata)

            logger.info(f"Conversion successful in {duration:.2f} seconds")
            logger.info(f"Original size: {original_size_mb:.1f}MB")
            logger.info(f"Quantized size: {quantized_size_mb:.1f}MB")
            logger.info(f"Compression ratio: {metadata['quantization']['compression_ratio']:.2f}x")
            logger.info(f"Target size: {quant_config['target_size_mb']}MB")

            return {
                'success': True,
                'metadata': metadata,
                'output_path': str(output_path)
            }

        except Exception as e:
            logger.error(f"Unexpected error during conversion: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e)
            }


def main():
    parser = argparse.ArgumentParser(
        description='Convert encoder models to MLX format with TRUE INT8 quantization'
    )
    parser.add_argument(
        '--config',
        default='config/models.yaml',
        help='Path to models config'
    )
    parser.add_argument(
        '--model',
        help='Specific model to convert (name from config)'
    )
    parser.add_argument(
        '--output-dir',
        default='models/mlx_converted',
        help='Output directory for converted models'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print conversion plan without executing'
    )
    parser.add_argument(
        '--no-skip',
        action='store_true',
        help='Force re-conversion even if model already exists'
    )

    args = parser.parse_args()

    logger = setup_logging()

    converter = EncoderModelConverter(
        args.config,
        args.output_dir,
        args.dry_run,
        skip_existing=not args.no_skip
    )

    if args.model:
        # Convert specific model
        model_config = next(
            (m for m in converter.config['models'] if m['name'] == args.model),
            None
        )
        if not model_config:
            logger.error(f"Model '{args.model}' not found in config")
            return 1

        result = converter.convert_single_model(model_config)
        print(json.dumps(result, indent=2))
        return 0 if result.get('success', False) else 1
    else:
        logger.error("Please specify a model with --model")
        return 1


if __name__ == "__main__":
    exit(main())
