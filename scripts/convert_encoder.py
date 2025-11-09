#!/usr/bin/env python3
"""
MLX Encoder Model Converter

This script converts encoder models (BERT, DistilBERT, RoBERTa, DeBERTa, etc.)
from HuggingFace to MLX format with quantization support.

Since mlx_lm only supports decoder-only and encoder-decoder language models,
this script uses the low-level MLX API to convert encoder models.
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
import mlx.nn as nn
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


def quantize_weights(weights: Dict[str, mx.array], bits: int = 8, group_size: int = 64) -> Dict[str, mx.array]:
    """
    Quantize model weights to the specified bit precision.

    Args:
        weights: Dictionary of weight tensors
        bits: Number of bits for quantization (4, 8)
        group_size: Group size for quantization

    Returns:
        Dictionary of quantized weights
    """
    quantized_weights = {}

    for name, weight in weights.items():
        # Skip quantization for certain layers (embeddings, layer norms, biases)
        if any(skip in name.lower() for skip in ['embedding', 'layernorm', 'bias', 'norm']):
            quantized_weights[name] = weight
            continue

        # Only quantize 2D weight matrices
        if len(weight.shape) != 2:
            quantized_weights[name] = weight
            continue

        # Perform quantization
        if bits == 8:
            # INT8 quantization: scale to [-127, 127]
            w_min = mx.min(weight)
            w_max = mx.max(weight)

            # Calculate scale and zero point
            scale = (w_max - w_min) / 255.0
            zero_point = -128 - (w_min / scale)

            # Quantize
            w_quant = mx.round((weight / scale) + zero_point)
            w_quant = mx.clip(w_quant, -128, 127)

            # Dequantize for storage (MLX doesn't have native int8 storage yet)
            w_dequant = (w_quant - zero_point) * scale

            # Store with metadata for later use
            quantized_weights[name] = w_dequant
            quantized_weights[f"{name}.scale"] = scale
            quantized_weights[f"{name}.zero_point"] = zero_point

        elif bits == 4:
            # 4-bit quantization (group-wise)
            # This is a simplified version - proper implementation would use group-wise quantization
            w_min = mx.min(weight)
            w_max = mx.max(weight)

            scale = (w_max - w_min) / 15.0
            zero_point = -(w_min / scale)

            w_quant = mx.round((weight / scale) + zero_point)
            w_quant = mx.clip(w_quant, 0, 15)

            w_dequant = (w_quant - zero_point) * scale

            quantized_weights[name] = w_dequant
            quantized_weights[f"{name}.scale"] = scale
            quantized_weights[f"{name}.zero_point"] = zero_point
        else:
            quantized_weights[name] = weight

    return quantized_weights


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


def save_mlx_model(
    weights: Dict[str, mx.array],
    config: Dict[str, Any],
    tokenizer,
    output_path: Path,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Save MLX model, config, and tokenizer.

    Args:
        weights: MLX model weights
        config: Model configuration
        tokenizer: HuggingFace tokenizer
        output_path: Output directory
        metadata: Additional metadata to save
    """
    output_path.mkdir(parents=True, exist_ok=True)

    # Save weights in MLX format (as .npz)
    weights_file = output_path / "weights.npz"

    # Convert MLX arrays to numpy for saving
    np_weights = {k: np.array(v) for k, v in weights.items()}
    np.savez(weights_file, **np_weights)

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


class EncoderModelConverter:
    """Converter for encoder models (BERT, DistilBERT, RoBERTa, etc.)"""

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
        """Convert a single encoder model to MLX format."""
        model_name = model_config['name']
        logger = setup_logging(model_name)
        start_time = time.time()
        quant_config = model_config['quantization']

        logger.info(f"Starting encoder model conversion: {model_name}")

        # Create output directory
        output_path = self.output_dir / f"{model_name}-mlx-q{quant_config['bits']}"

        # Check if model already exists and skip if requested
        if self.skip_existing and output_path.exists():
            metadata_file = output_path / 'conversion_metadata.json'
            if metadata_file.exists():
                logger.info(f"Skipping {model_name} - already converted at {output_path}")
                with open(metadata_file, 'r') as f:
                    existing_metadata = json.load(f)
                return {
                    'success': True,
                    'skipped': True,
                    'metadata': existing_metadata,
                    'output_path': str(output_path)
                }

        # If directory exists but has no metadata (incomplete conversion), remove it
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

            # Apply quantization
            if quant_config['bits'] < 32:
                logger.info(f"Quantizing to {quant_config['bits']}-bit")
                mlx_weights = quantize_weights(
                    mlx_weights,
                    bits=quant_config['bits'],
                    group_size=quant_config.get('group_size', 64)
                )

            # Prepare config dict
            config_dict = config.to_dict()
            config_dict['quantization'] = {
                'bits': quant_config['bits'],
                'group_size': quant_config.get('group_size', 64),
                'dtype': quant_config['dtype']
            }

            # Save model
            logger.info(f"Saving MLX model to {output_path}")

            # Create metadata
            model_size_mb = sum(
                np.array(v).nbytes for v in mlx_weights.values()
            ) / (1024 * 1024)

            duration = time.time() - start_time

            metadata = {
                'model_name': model_name,
                'hf_name': model_config['hf_name'],
                'model_type': 'encoder',
                'architecture': config_dict.get('model_type', 'unknown'),
                'quantization': {
                    'bits': quant_config['bits'],
                    'dtype': quant_config['dtype'],
                    'target_size_mb': quant_config['target_size_mb'],
                    'actual_size_mb': model_size_mb
                },
                'conversion_time_seconds': duration,
                'timestamp': datetime.now().isoformat(),
                'mlx_version': mx.__version__,
                'device': str(mx.default_device()),
                'size_accuracy_ratio': model_size_mb / quant_config['target_size_mb']
            }

            save_mlx_model(mlx_weights, config_dict, tokenizer, output_path, metadata)

            logger.info(f"Conversion successful in {duration:.2f} seconds")
            logger.info(f"Model size: {model_size_mb:.1f}MB (target: {quant_config['target_size_mb']}MB)")

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
        description='Convert encoder models to MLX format with quantization'
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
