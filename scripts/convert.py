import argparse
import yaml
import json
import os
import time
import tempfile
from pathlib import Path
import subprocess
import logging
from datetime import datetime
import mlx.core as mx
import concurrent.futures
import glob
import shutil
from mlx_lm import convert as mlx_convert
import mlx.nn as nn


def setup_logging(model_name=None):
    """Set up logging for the conversion process."""
    temp_dir = os.environ.get('MLX_TEMP_DIR', tempfile.gettempdir())
    log_dir = Path(temp_dir) / 'mlx_conversion' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"conversion_{model_name}.log" if model_name else log_dir / "conversion_summary.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(f'MLXConverter_{model_name}' if model_name else 'MLXConverter')

class Bit8ModelConverter:
    def __init__(self, config_path='config/models.yaml', output_dir='models/mlx_converted', dry_run=False, skip_existing=True):
        # Support multiple yaml files (glob pattern or single file)
        self.config = {'models': []}
        config_files = []

        if isinstance(config_path, list):
            config_files = config_path
        elif '*' in config_path or '?' in config_path:
            config_files = glob.glob(config_path)
        else:
            config_files = [config_path]

        # Load all config files and merge
        for cfg_file in config_files:
            if Path(cfg_file).exists():
                with open(cfg_file, 'r') as f:
                    cfg_data = yaml.safe_load(f)
                    if 'models' in cfg_data:
                        self.config['models'].extend(cfg_data['models'])

        temp_dir = os.environ.get('MLX_TEMP_DIR', tempfile.gettempdir())
        self.results_dir = Path(temp_dir) / 'mlx_conversion' / 'results'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = Path(output_dir)
        self.logger = setup_logging()
        self.dry_run = dry_run
        self.skip_existing = skip_existing
        
    def convert_single_model(self, model_config):
        """Convert a single model to MLX format"""
        model_name = model_config['name']
        logger = setup_logging(model_name)
        start_time = time.time()
        quant_config = model_config['quantization']

        logger.info(f"Starting conversion: {model_name}")

        output_path = self.output_dir / f"{model_name}-mlx-q{quant_config['bits']}"

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

        if output_path.exists():
            logger.info(f"Removing incomplete conversion at {output_path}")
            shutil.rmtree(output_path)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        def exclude_predicate(path: str, module: nn.Module) -> bool:
            if "ffn.lin2" in path:
                logger.info(f"Excluding layer from quantization: {path}")
                return False
            return True

        if self.dry_run:
            logger.info("Dry-run enabled, skipping execution")
            cmd_str = (
                f"mlx_lm.convert.convert(hf_path='{model_config['hf_name']}', "
                f"mlx_path='{str(output_path)}', quantize=True, "
                f"q_bits={quant_config['bits']}, "
                f"q_group_size={quant_config.get('group_size', 64)}, "
                f"quant_predicate=exclude_predicate)"
            )
            return {'success': True, 'dry_run': True, 'command': cmd_str}

        try:
            logger.info("Starting model conversion via Python API with layer exclusion")
            mlx_convert(
                hf_path=model_config['hf_name'],
                mlx_path=str(output_path),
                quantize=True,
                q_bits=quant_config['bits'],
                q_group_size=quant_config.get('group_size', 64),
                quant_predicate=exclude_predicate,
            )
            
            duration = time.time() - start_time
            model_size_mb = sum(f.stat().st_size for f in output_path.glob('**/*') if f.is_file()) / (1024 * 1024)
            
            metadata = {
                'model_name': model_name,
                'hf_name': model_config['hf_name'],
                'quantization': {
                    'bits': quant_config['bits'],
                    'dtype': quant_config['dtype'],
                    'target_size_mb': quant_config['target_size_mb'],
                    'actual_size_mb': model_size_mb,
                    'excluded_layers': ["ffn.lin2"],
                },
                'conversion_time_seconds': duration,
                'timestamp': datetime.now().isoformat(),
                'mlx_version': mx.__version__,
                'device': str(mx.default_device()),
                'size_target_ratio': model_size_mb / quant_config['target_size_mb'] if quant_config['target_size_mb'] > 0 else 0.0
            }
            
            metadata_file = output_path / 'conversion_metadata.json'
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Conversion successful in {duration:.2f} seconds")
            logger.info(f"Model size: {model_size_mb:.1f}MB (target: {quant_config['target_size_mb']}MB)")
            
            return {
                'success': True,
                'metadata': metadata,
                'output_path': str(output_path)
            }
            
        except Exception as e:
            logger.error(f"Unexpected error during conversion: {str(e)}")
            if output_path.exists():
                shutil.rmtree(output_path)
            return {
                'success': False,
                'error': str(e)
            }
    
    def convert_all_models(self):
        """Convert all configured models to MLX format in parallel."""
        self.logger.info("Starting parallel conversion batch for all models")
        
        results = {}
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_model = {executor.submit(self.convert_single_model, model_config): model_config for model_config in self.config['models']}
            
            for future in concurrent.futures.as_completed(future_to_model):
                model_config = future_to_model[future]
                model_name = model_config['name']
                try:
                    result = future.result()
                    results[model_name] = result
                    
                    # Save intermediate results
                    with open(self.results_dir / f"{model_name}_conversion.json", 'w') as f:
                        json.dump(result, f, indent=2)
                        
                except Exception as exc:
                    self.logger.error(f"{model_name} generated an exception: {exc}")
                    results[model_name] = {'success': False, 'error': str(exc)}

        # Save final summary
        summary_file = self.results_dir / 'conversion_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Conversion summary saved to {summary_file}")
        
        # Print summary
        successful = [name for name, result in results.items() if result.get('success', False)]
        failed = [name for name, result in results.items() if not result.get('success', False)]
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info("CONVERSION SUMMARY")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Successful: {len(successful)} models")
        self.logger.info(f"Failed: {len(failed)} models")
        
        if successful:
            self.logger.info("Successful conversions:")
            for name in successful:
                self.logger.info(f"  - {name}")
        
        if failed:
            self.logger.error("Failed conversions:")
            for name in failed:
                self.logger.error(f"  - {name}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Convert models to MLX format')
    parser.add_argument('--config', default='config/models.yaml', help='Path to models config (supports glob patterns like config/*.yaml)')
    parser.add_argument('--model', help='Specific model to convert (all if not specified)')
    parser.add_argument('--output-dir', default='models/mlx_converted', help='Output directory for converted models')
    parser.add_argument('--dry-run', action='store_true', help='Print conversion commands without executing them')
    parser.add_argument('--no-skip', action='store_true', help='Force re-conversion even if model already exists')

    args = parser.parse_args()

    logger = setup_logging()

    converter = Bit8ModelConverter(args.config, args.output_dir, args.dry_run, skip_existing=not args.no_skip)
    
    if args.model:
        # Convert specific model
        model_config = next((m for m in converter.config['models'] if m['name'] == args.model), None)
        if not model_config:
            logger.error(f"Model '{args.model}' not found in config")
            return 1
        
        result = converter.convert_single_model(model_config)
        print(json.dumps(result, indent=2))
        return 0 if result.get('success', False) else 1
        
    else:
        # Convert all models
        results = converter.convert_all_models()
        return 0 if all(result.get('success', False) for result in results.values()) else 1

if __name__ == "__main__":
    exit(main())
