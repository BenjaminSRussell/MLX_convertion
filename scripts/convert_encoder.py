#!/usr/bin/env python3
"""
converts encoder models to MLX with proper quantization
stores weights as INT8 for real size savings (4x smaller with 8-bit)
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


def should_quantize_weight(name: str, weight_np: np.ndarray) -> bool:
    """checks if we should quantize this weight or skip it"""
    if any(skip in name.lower() for skip in ['layernorm', 'bias', 'norm', 'ln']):
        return False
        
    if len(weight_np.shape) != 2:
        return False
        
    return True


def quantize_8bit(weight_np: np.ndarray) -> tuple:
    """quantizes weights to 8-bit with scale factor"""
    w_max = np.abs(weight_np).max()
    scale = w_max / 127.0 if w_max > 0 else 1.0
    
    w_quant = np.round(weight_np / scale)
    w_quant = np.clip(w_quant, -127, 127).astype(np.int8)
    
    return w_quant, scale


def quantize_4bit(weight_np: np.ndarray) -> tuple:
    """quantizes weights to 4-bit with scale factor"""
    w_max = np.abs(weight_np).max()
    scale = w_max / 7.0 if w_max > 0 else 1.0
    
    w_quant = np.round(weight_np / scale)
    w_quant = np.clip(w_quant, -7, 7).astype(np.int8)
    
    return w_quant, scale


def quantize_weights_int8(
    weights: Dict[str, mx.array],
    bits: int = 8
) -> Dict[str, np.ndarray]:
    """quantizes model weights to save space"""
    quantized_data = {}

    for name, weight in weights.items():
        weight_np = np.array(weight)
        
        if not should_quantize_weight(name, weight_np):
            quantized_data[name] = weight_np.astype(np.float32)
            continue

        if bits == 8:
            w_quant, scale = quantize_8bit(weight_np)
            quantized_data[name] = w_quant
            quantized_data[f"{name}.__scale__"] = np.array(scale, dtype=np.float32)
            
        elif bits == 4:
            w_quant, scale = quantize_4bit(weight_np)
            quantized_data[name] = w_quant
            quantized_data[f"{name}.__scale__"] = np.array(scale, dtype=np.float32)
            
        else:
            quantized_data[name] = weight_np.astype(np.float32)

    return quantized_data


def convert_pytorch_to_mlx(state_dict: Dict[str, Any]) -> Dict[str, mx.array]:
    """converts PyTorch weights to MLX format"""
    mlx_weights = {}

    for name, param in state_dict.items():
        if hasattr(param, 'detach'):
            param_np = param.detach().cpu().numpy()
        else:
            param_np = np.array(param)

        mlx_weights[name] = mx.array(param_np)

    return mlx_weights


def calculate_model_size(weights: Dict[str, np.ndarray]) -> float:
    """calculates total model size in MB"""
    total_bytes = sum(w.nbytes for w in weights.values())
    return total_bytes / (1024 * 1024)


def migrate_old_metadata(metadata: Dict[str, Any], output_path: Path) -> Dict[str, Any]:
    """updates old metadata files to new format"""
    updated = False

    if 'method' not in metadata.get('quantization', {}):
        if 'quantization' not in metadata:
            metadata['quantization'] = {}
        metadata['quantization']['method'] = 'symmetric'
        updated = True

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

                    original_bytes = (quantized_params + unquantized_params) * 4
                    original_size_mb = original_bytes / (1024 * 1024)

                metadata['quantization']['original_size_mb'] = original_size_mb
                updated = True
            except Exception:
                pass

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
    """saves the MLX model to disk"""
    output_path.mkdir(parents=True, exist_ok=True)

    weights_file = output_path / "weights.npz"
    np.savez_compressed(weights_file, **weights)

    config_file = output_path / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    tokenizer.save_pretrained(output_path)

    if metadata:
        metadata_file = output_path / "conversion_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)


def load_quantized_weights(weights_file: Path) -> Dict[str, mx.array]:
    """loads and dequantizes weights from file"""
    mlx_weights = {}

    with np.load(weights_file) as np_weights:
        weight_names = [k for k in np_weights.keys() if not k.endswith('.__scale__')]

        for name in weight_names:
            weight = np_weights[name]

            scale_name = f"{name}.__scale__"
            if scale_name in np_weights:
                scale = float(np_weights[scale_name])
                weight_dequant = weight.astype(np.float32) * scale
                mlx_weights[name] = mx.array(weight_dequant)
            else:
                mlx_weights[name] = mx.array(weight)

    return mlx_weights


class EncoderModelConverter:
    def __init__(
        self,
        config_path: str = 'config/models.yaml',
        output_dir: str = 'models/mlx_converted',
        dry_run: bool = False,
        skip_existing: bool = True
    ):
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

    def convert_single_model(self, model_cfg: Dict[str, Any]) -> Dict[str, Any]:
        model_name = model_cfg['name']
        logger = setup_logging(model_name)
        t_start = time.time()
        quant_cfg = model_cfg['quantization']

        logger.info(f"converting encoder model {model_name}...")
        logger.info(f"using {quant_cfg['bits']}-bit quantization")

        out_path = self.output_dir / f"{model_name}-mlx-q{quant_cfg['bits']}"

        if self.skip_existing and out_path.exists():
            meta_file = out_path / 'conversion_metadata.json'
            if meta_file.exists():
                logger.info(f"skipping {model_name} - already done at {out_path}")
                with open(meta_file, 'r') as f:
                    existing_meta = json.load(f)

                existing_meta = migrate_old_metadata(existing_meta, out_path)

                return {
                    'success': True,
                    'skipped': True,
                    'metadata': existing_meta,
                    'output_path': str(out_path)
                }

        if out_path.exists():
            logger.info(f"cleaning up incomplete conversion at {out_path}")
            shutil.rmtree(out_path)

        if self.dry_run:
            logger.info("dry-run mode - not actually converting")
            return {
                'success': True,
                'dry_run': True,
                'model_name': model_name
            }

        try:
            hf_model = AutoModel.from_pretrained(model_cfg['hf_name'])
            tokenizer = AutoTokenizer.from_pretrained(model_cfg['hf_name'])
            config = AutoConfig.from_pretrained(model_cfg['hf_name'])

            logger.info("converting PyTorch weights to MLX...")
            mlx_weights = convert_pytorch_to_mlx(hf_model.state_dict())

            orig_size_mb = sum(
                np.array(v).nbytes for v in mlx_weights.values()
            ) / (1024 * 1024)

            if quant_cfg['bits'] < 32:
                logger.info(f"applying {quant_cfg['bits']}-bit quantization...")
                quant_weights = quantize_weights_int8(
                    mlx_weights,
                    bits=quant_cfg['bits']
                )
            else:
                logger.info("skipping quantization")
                quant_weights = {k: np.array(v) for k, v in mlx_weights.items()}

            quant_size_mb = calculate_model_size(quant_weights)

            cfg_dict = config.to_dict()
            cfg_dict['quantization'] = {
                'bits': quant_cfg['bits'],
                'dtype': 'int8' if quant_cfg['bits'] <= 8 else quant_cfg['dtype'],
                'method': 'symmetric'
            }

            logger.info(f"saving model to {out_path}...")

            dur = time.time() - t_start

            meta = {
                'model_name': model_name,
                'hf_name': model_cfg['hf_name'],
                'model_type': 'encoder',
                'architecture': cfg_dict.get('model_type', 'unknown'),
                'quantization': {
                    'bits': quant_cfg['bits'],
                    'dtype': 'int8' if quant_cfg['bits'] <= 8 else 'float32',
                    'method': 'symmetric',
                    'target_size_mb': quant_cfg['target_size_mb'],
                    'actual_size_mb': quant_size_mb,
                    'original_size_mb': orig_size_mb,
                    'compression_ratio': orig_size_mb / quant_size_mb if quant_size_mb > 0 else 1.0
                },
                'conversion_time_seconds': dur,
                'timestamp': datetime.now().isoformat(),
                'mlx_version': mx.__version__,
                'device': str(mx.default_device()),
                'size_target_ratio': quant_size_mb / quant_cfg['target_size_mb'] if quant_cfg['target_size_mb'] > 0 else 0.0,
                'converter_version': 'v2'
            }

            save_mlx_model(quant_weights, cfg_dict, tokenizer, out_path, meta)

            logger.info(f"done! took {dur:.2f}s")
            logger.info(f"original: {orig_size_mb:.1f}MB")
            logger.info(f"compressed: {quant_size_mb:.1f}MB")
            logger.info(f"that's {meta['quantization']['compression_ratio']:.2f}x smaller")
            logger.info(f"target was {quant_cfg['target_size_mb']}MB")

            return {
                'success': True,
                'metadata': meta,
                'output_path': str(out_path)
            }

        except Exception as e:
            logger.error(f"oops, conversion failed: {str(e)}")
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
        model_cfg = next(
            (m for m in converter.config['models'] if m['name'] == args.model),
            None
        )
        if not model_cfg:
            logger.error(f"can't find model '{args.model}' in config")
            return 1

        result = converter.convert_single_model(model_cfg)
        print(json.dumps(result, indent=2))
        return 0 if result.get('success', False) else 1
    else:
        logger.error("need to specify a model with --model")
        return 1


if __name__ == "__main__":
    exit(main())
