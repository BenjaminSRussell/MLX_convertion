"""
PyTorch-based 8-bit quantization converter
Works on x86_64 Linux (alternative to MLX which requires Apple Silicon)
"""
import argparse
import yaml
import json
import os
import time
import tempfile
from pathlib import Path
import logging
from datetime import datetime
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import concurrent.futures
import glob

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
    return logging.getLogger(f'PyTorchConverter_{model_name}' if model_name else 'PyTorchConverter')

class PyTorchInt8Converter:
    """Convert models to 8-bit quantized PyTorch format"""

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

    def quantize_model(self, model):
        """Quantize model to int8 using PyTorch dynamic quantization"""
        self.logger.info("Applying dynamic int8 quantization...")

        # Dynamic quantization for linear layers
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )

        return quantized_model

    def convert_single_model(self, model_config):
        """Convert a single model to quantized PyTorch format"""
        model_name = model_config['name']
        logger = setup_logging(model_name)
        start_time = time.time()
        quant_config = model_config['quantization']

        logger.info(f"Starting conversion: {model_name}")

        # Create output directory
        output_path = self.output_dir / f"{model_name}-mlx-q{quant_config['bits']}"

        # Check if model already exists and skip if requested
        if self.skip_existing and output_path.exists():
            metadata_file = output_path / 'conversion_metadata.json'
            if metadata_file.exists():
                logger.info(f"⏭️  Skipping {model_name} - already converted at {output_path}")
                with open(metadata_file, 'r') as f:
                    existing_metadata = json.load(f)
                return {
                    'success': True,
                    'skipped': True,
                    'metadata': existing_metadata,
                    'output_path': str(output_path)
                }

        output_path.mkdir(parents=True, exist_ok=True)

        if self.dry_run:
            logger.info("🌵 Dry-run enabled, skipping execution")
            return {
                'success': True,
                'dry_run': True,
                'model_name': model_name
            }

        try:
            # Load model and tokenizer
            logger.info(f"📥 Loading model from HuggingFace: {model_config['hf_name']}")
            tokenizer = AutoTokenizer.from_pretrained(model_config['hf_name'])
            model = AutoModelForSequenceClassification.from_pretrained(model_config['hf_name'])

            # Quantize model
            logger.info("🔧 Quantizing model to int8...")
            quantized_model = self.quantize_model(model)

            # Save quantized model and tokenizer
            logger.info(f"💾 Saving quantized model to {output_path}")
            # Save tokenizer first
            tokenizer.save_pretrained(output_path)

            # Save quantized model using torch.save (quantized models can't use save_pretrained)
            model_path = output_path / 'pytorch_model.bin'
            torch.save(quantized_model.state_dict(), model_path)

            # Also save the original config
            model.config.save_pretrained(output_path)

            duration = time.time() - start_time

            # Calculate model size
            model_size_mb = sum(f.stat().st_size for f in output_path.glob('**/*') if f.is_file()) / (1024 * 1024)

            # Save conversion metadata
            metadata = {
                'model_name': model_name,
                'hf_name': model_config['hf_name'],
                'quantization': {
                    'bits': quant_config['bits'],
                    'dtype': quant_config['dtype'],
                    'target_size_mb': quant_config['target_size_mb'],
                    'actual_size_mb': model_size_mb,
                    'method': 'pytorch_dynamic_quantization'
                },
                'conversion_time_seconds': duration,
                'timestamp': datetime.now().isoformat(),
                'pytorch_version': torch.__version__,
                'size_accuracy_ratio': model_size_mb / quant_config['target_size_mb']
            }

            metadata_file = output_path / 'conversion_metadata.json'
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"✅ Conversion successful in {duration:.2f} seconds")
            logger.info(f"📊 Model size: {model_size_mb:.1f}MB (target: {quant_config['target_size_mb']}MB)")

            return {
                'success': True,
                'metadata': metadata,
                'output_path': str(output_path)
            }

        except Exception as e:
            logger.error(f"❌ Conversion failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e)
            }

    def convert_all_models(self):
        """Convert all configured models"""
        self.logger.info(f"Starting batch conversion for {len(self.config['models'])} models")

        results = {}

        for model_config in self.config['models']:
            model_name = model_config['name']
            try:
                result = self.convert_single_model(model_config)
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

        self.logger.info(f"📊 Conversion summary saved to {summary_file}")

        # Print summary
        successful = [name for name, result in results.items() if result.get('success', False)]
        failed = [name for name, result in results.items() if not result.get('success', False)]

        self.logger.info(f"\n{'='*60}")
        self.logger.info("CONVERSION SUMMARY")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"✅ Successful: {len(successful)} models")
        self.logger.info(f"❌ Failed: {len(failed)} models")

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
    parser = argparse.ArgumentParser(description='Convert models to 8-bit PyTorch quantized format')
    parser.add_argument('--config', default='config/models.yaml', help='Path to models config (supports glob patterns)')
    parser.add_argument('--model', help='Specific model to convert (all if not specified)')
    parser.add_argument('--output-dir', default='models/mlx_converted', help='Output directory for converted models')
    parser.add_argument('--dry-run', action='store_true', help='Print conversion plan without executing')
    parser.add_argument('--no-skip', action='store_true', help='Force re-conversion even if model already exists')

    args = parser.parse_args()

    logger = setup_logging()

    converter = PyTorchInt8Converter(args.config, args.output_dir, args.dry_run, skip_existing=not args.no_skip)

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
