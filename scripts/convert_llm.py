#!/usr/bin/env python3
"""
hey! this converts LLMs to MLX format with quantization
uses mlx_lm's built-in convert function - pretty straightforward
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
from mlx_lm import convert as mlx_convert


def setup_logging(model_name=None):
    """sets up logging so we can track what's happening"""
    temp_dir = os.environ.get('MLX_TEMP_DIR', tempfile.gettempdir())
    log_dir = Path(temp_dir) / 'mlx_conversion' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"llm_conversion_{model_name}.log" if model_name else log_dir / "llm_conversion.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(f'MLXLLMConverter_{model_name}' if model_name else 'MLXLLMConverter')


def calculate_directory_size(path: Path) -> float:
    """figures out how big a directory is (in MB)"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    return total_size / (1024 * 1024)


class LLMConverter:
    """converts LLMs using mlx_lm's tools"""

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

    def convert_single_model(self, model_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """converts one LLM to MLX with quantization"""
        model_name = model_cfg['name']
        logger = setup_logging(model_name)
        t_start = time.time()
        quant_cfg = model_cfg.get('quantization', {})

        logger.info(f"converting {model_name} to MLX...")
        logger.info(f"using {quant_cfg.get('bits', 4)}-bit quantization")

        # Get HuggingFace model name
        hf_name = model_cfg.get('hf_name', model_name)

        # Create output directory
        bits = quant_cfg.get('bits', 4)
        out_path = self.output_dir / f"{model_name}-mlx-q{bits}"

        # Check if model already exists
        if self.skip_existing and out_path.exists():
            meta_file = out_path / 'conversion_metadata.json'
            if meta_file.exists():
                logger.info(f"skipping {model_name} - already done at {out_path}")
                with open(meta_file, 'r') as f:
                    existing_meta = json.load(f)

                return {
                    'success': True,
                    'skipped': True,
                    'metadata': existing_meta,
                    'output_path': str(out_path)
                }

        # Remove incomplete conversions
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
            # Use mlx_lm.convert for efficient conversion
            logger.info(f"converting {hf_name} with mlx_lm...")
            logger.info(f"  using {bits}-bit quantization")
            logger.info(f"  saving to {out_path}")

            # Convert using mlx_lm built-in function
            mlx_convert(
                hf_path=hf_name,
                mlx_path=str(out_path),
                quantize=True,
                q_bits=bits,
                q_group_size=64,
                dtype='float16'
            )

            # Calculate sizes
            actual_size_mb = calculate_directory_size(out_path)

            # Estimate original size (rough approximation)
            # 4-bit saves ~8x space, 8-bit saves ~4x (vs fp32)
            comp_factor = 8 if bits == 4 else 4 if bits == 8 else 2
            orig_size_mb = actual_size_mb * comp_factor

            dur = time.time() - t_start

            meta = {
                'model_name': model_name,
                'hf_name': hf_name,
                'model_type': 'causal-lm',
                'architecture': model_cfg.get('architecture', 'unknown'),
                'quantization': {
                    'bits': bits,
                    'dtype': f'int{bits}',
                    'method': 'group',
                    'group_size': 64,
                    'target_size_mb': quant_cfg.get('target_size_mb', 0),
                    'actual_size_mb': actual_size_mb,
                    'original_size_mb': orig_size_mb,
                    'compression_ratio': comp_factor
                },
                'conversion_time_seconds': dur,
                'timestamp': datetime.now().isoformat(),
                'mlx_version': mx.__version__,
                'device': str(mx.default_device()),
                'converter_version': 'v3_mlx_lm',
                'conversion_method': 'mlx_lm.convert'
            }

            # Save metadata
            meta_file = out_path / 'conversion_metadata.json'
            with open(meta_file, 'w') as f:
                json.dump(meta, f, indent=2)

            logger.info(f"done! took {dur:.2f}s")
            logger.info(f"original size would be ~{orig_size_mb:.1f}MB")
            logger.info(f"compressed to {actual_size_mb:.1f}MB")
            logger.info(f"that's ~{comp_factor:.1f}x smaller")

            return {
                'success': True,
                'metadata': meta,
                'output_path': str(out_path)
            }

        except Exception as e:
            logger.error(f"uh oh, conversion failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e)
            }


def main():
    parser = argparse.ArgumentParser(
        description='Convert LLMs to MLX format with quantization using mlx_lm'
    )
    parser.add_argument(
        '--config',
        default='mlx_conversion/config/models.yaml',
        help='Path to models config'
    )
    parser.add_argument(
        '--model',
        help='Specific model to convert (name from config)'
    )
    parser.add_argument(
        '--hf-model',
        help='HuggingFace model name (if not using config)'
    )
    parser.add_argument(
        '--output-dir',
        default='models/mlx_converted',
        help='Output directory for converted models'
    )
    parser.add_argument(
        '--bits',
        type=int,
        default=4,
        choices=[4, 8],
        help='Quantization bits (4 or 8)'
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

    converter = LLMConverter(
        args.config,
        args.output_dir,
        args.dry_run,
        skip_existing=not args.no_skip
    )

    if args.hf_model:
        # Convert from HuggingFace model name directly
        model_cfg = {
            'name': args.hf_model.split('/')[-1],
            'hf_name': args.hf_model,
            'quantization': {'bits': args.bits}
        }
        result = converter.convert_single_model(model_cfg)
        print(json.dumps(result, indent=2))
        return 0 if result.get('success', False) else 1

    elif args.model:
        # Convert specific model from config
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
        logger.error("need to specify a model with --model or --hf-model")
        return 1


if __name__ == "__main__":
    exit(main())
