"""
Refactored model testing script using handler architecture.
This replaces the god functions in test.py with modular, maintainable code.
"""

import argparse
import yaml
import json
import os
import tempfile
import numpy as np
from pathlib import Path
from datasets import load_dataset
import logging
import glob
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mlx_conversion.handlers import (
    NLIHandler,
    TextClassificationHandler,
    SemanticSimilarityHandler
)
from mlx_conversion.utils import (
    MetricsCalculator,
    MemoryTracker,
    ModelComparator
)


def setup_logging():
    """Setup logging configuration."""
    temp_dir = os.environ.get('MLX_TEMP_DIR', tempfile.gettempdir())
    log_dir = Path(temp_dir) / 'mlx_conversion' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'testing_refactored.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('MLX8BitTester')


logger = setup_logging()


class ModelTester:
    """
    Refactored model tester using handler architecture.
    No more god functions - each task type has its own handler.
    """

    # Map task types to handler classes
    TASK_HANDLERS = {
        'zero-shot-classification': NLIHandler,
        'text-classification': TextClassificationHandler,
        'classification': TextClassificationHandler,
        'sentiment-analysis': TextClassificationHandler,
        'semantic-similarity': SemanticSimilarityHandler,
        'sentence-similarity': SemanticSimilarityHandler,
    }

    def __init__(
        self,
        models_config='config/models.yaml',
        datasets_config='config/datasets.yaml',
        results_dir=None,
        comparisons_dir=None
    ):
        """
        Initialize model tester.

        Args:
            models_config: Path(s) to models YAML config
            datasets_config: Path(s) to datasets YAML config
            results_dir: Directory for test results
            comparisons_dir: Directory for comparison results
        """
        # Load model configs
        self.models_config = self._load_configs(models_config, 'models')

        # Load dataset configs
        self.datasets_config = self._load_configs(datasets_config, 'datasets')

        # Setup directories
        temp_dir = os.environ.get('MLX_TEMP_DIR', tempfile.gettempdir())
        self.results_dir = Path(results_dir) if results_dir else Path(temp_dir) / 'mlx_conversion' / 'test_results'
        self.results_dir.mkdir(parents=True, exist_ok=True)

        comparisons_path = Path(comparisons_dir) if comparisons_dir else Path(temp_dir) / 'mlx_conversion' / 'comparisons'

        # Initialize utilities
        self.metrics_calculator = MetricsCalculator()
        self.memory_tracker = MemoryTracker()
        self.comparator = ModelComparator(comparisons_path)

    def _load_configs(self, config_path, key):
        """
        Load configuration files (supports glob patterns).

        Args:
            config_path: Path or pattern to config file(s)
            key: Key to extract from configs ('models' or 'datasets')

        Returns:
            Dictionary with merged configs
        """
        config_data = {key: [] if key == 'models' else {}}
        config_files = []

        if isinstance(config_path, list):
            config_files = config_path
        elif '*' in config_path or '?' in config_path:
            config_files = glob.glob(config_path)
        else:
            config_files = [config_path]

        for cfg_file in config_files:
            if Path(cfg_file).exists():
                with open(cfg_file, 'r') as f:
                    cfg = yaml.safe_load(f)
                    if key in cfg:
                        if key == 'models':
                            config_data[key].extend(cfg[key])
                        else:
                            config_data[key].update(cfg[key])

        return config_data

    def get_handler(self, model_config: dict, dataset_config: dict):
        """
        Get appropriate handler for model task type.

        Args:
            model_config: Model configuration
            dataset_config: Dataset configuration

        Returns:
            Handler instance

        Raises:
            ValueError: If task type not supported
        """
        task = model_config['task']

        handler_class = self.TASK_HANDLERS.get(task)
        if not handler_class:
            raise ValueError(
                f"Unsupported task type: {task}. "
                f"Supported tasks: {list(self.TASK_HANDLERS.keys())}"
            )

        return handler_class(model_config, dataset_config)

    def load_test_dataset(self, dataset_name: str, max_samples: int = 500):
        """
        Load dataset for testing.

        Args:
            dataset_name: Name of dataset
            max_samples: Maximum number of samples

        Returns:
            HuggingFace dataset
        """
        config = self.datasets_config['datasets'][dataset_name]

        logger.info(f"Loading test dataset: {dataset_name}")

        try:
            # Determine cache directory
            cache_dir = config.get('preprocessing', {}).get('cache_dir') or \
                        self.datasets_config.get('cache_dir') or \
                        os.environ.get('MLX_TEMP_DIR', tempfile.gettempdir()) + '/hf_datasets'

            # Get validation split
            val_split = config['splits'].get('validation', list(config['splits'].values())[0])

            # Load parameters
            load_params = {
                'path': config['name'],
                'split': val_split,
                'cache_dir': cache_dir,
                'download_mode': 'reuse_dataset_if_exists'
            }

            # Add subset if specified
            if 'subset' in config:
                load_params['name'] = config['subset']

            dataset = load_dataset(**load_params)

            # Sample for faster testing
            sample_size = min(max_samples, len(dataset))
            if sample_size < len(dataset):
                indices = np.random.choice(len(dataset), sample_size, replace=False)
                dataset = dataset.select(indices)

            logger.info(f"Loaded {len(dataset)} examples from {dataset_name}")
            return dataset

        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {str(e)}")
            raise

    def compare_models(self, model_name: str, dataset_name: str, max_samples: int = 200):
        """
        Compare PyTorch baseline with MLX quantized model.

        This is the main entry point - no god function, just orchestration!

        Args:
            model_name: Name of model to test
            dataset_name: Name of dataset
            max_samples: Maximum samples to test

        Returns:
            Comparison results dictionary
        """
        logger.info(f"Comparing {model_name} on {dataset_name}")

        # Find model config
        model_config = next(
            (m for m in self.models_config['models'] if m['name'] == model_name),
            None
        )
        if not model_config:
            logger.error(f"Model '{model_name}' not found in config")
            return None

        # Get dataset config
        dataset_config = self.datasets_config['datasets'].get(dataset_name)
        if not dataset_config:
            logger.error(f"Dataset '{dataset_name}' not found in config")
            return None

        # Load dataset
        dataset = self.load_test_dataset(dataset_name, max_samples=max_samples)

        # Get appropriate handler for this task
        handler = self.get_handler(model_config, dataset_config)
        logger.info(f"Using handler: {handler.__class__.__name__}")

        # Test PyTorch baseline
        logger.info("Testing PyTorch baseline...")
        pt_results = handler.test_pytorch_baseline(
            model_name=model_config['hf_name'],
            dataset_name=dataset_name,
            dataset=dataset,
            memory_tracker=self.memory_tracker,
            metrics_calculator=self.metrics_calculator
        )

        # Test MLX quantized model
        quant_bits = model_config['quantization']['bits']
        mlx_path = Path(f"models/mlx_converted/{model_name}-mlx-q{quant_bits}")

        if not mlx_path.exists():
            logger.error(f"MLX model not found at {mlx_path}")
            logger.info(f"Run conversion first: python scripts/convert_encoder.py --model {model_name}")
            return None

        logger.info("Testing MLX quantized model...")
        mlx_results = handler.test_mlx_model(
            model_path=mlx_path,
            dataset_name=dataset_name,
            dataset=dataset,
            memory_tracker=self.memory_tracker,
            metrics_calculator=self.metrics_calculator
        )

        # Compare using ModelComparator
        comparison = self.comparator.compare(
            model_name=model_name,
            dataset_name=dataset_name,
            pytorch_results=pt_results,
            mlx_results=mlx_results,
            model_config=model_config
        )

        return comparison

    def test_all_models(self, max_samples: int = 200):
        """
        Test all configured models against their benchmarks.

        Args:
            max_samples: Maximum samples per test

        Returns:
            Dictionary of all results
        """
        logger.info("Starting comprehensive model testing")

        all_results = {}

        for model_config in self.models_config['models']:
            model_name = model_config['name']
            all_results[model_name] = {}

            logger.info(f"\n{'='*60}")
            logger.info(f"TESTING MODEL: {model_name}")
            logger.info(f"{'='*60}")

            for dataset_name in model_config['benchmarks']:
                logger.info(f"Testing on dataset: {dataset_name}")

                try:
                    comparison = self.compare_models(model_name, dataset_name, max_samples=max_samples)
                    if comparison:
                        all_results[model_name][dataset_name] = comparison

                        # Save intermediate results
                        with open(self.results_dir / f"{model_name}_results.json", 'w') as f:
                            json.dump(all_results, f, indent=2)

                except Exception as e:
                    logger.error(f"Failed to test {model_name} on {dataset_name}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    all_results[model_name][dataset_name] = {
                        'error': str(e),
                        'failed': True
                    }

        # Save final summary
        summary_file = self.results_dir / 'testing_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        logger.info(f"Testing summary saved to {summary_file}")

        # Print summary
        self._print_summary(all_results)

        return all_results

    def _print_summary(self, all_results: dict):
        """
        Print testing summary.

        Args:
            all_results: Dictionary of all test results
        """
        logger.info(f"\n{'='*60}")
        logger.info("TESTING SUMMARY")
        logger.info(f"{'='*60}")

        for model_name, datasets in all_results.items():
            logger.info(f"\nModel: {model_name}")
            for dataset_name, result in datasets.items():
                if 'quality_gates' in result:
                    gates = result['quality_gates']
                    status = "✓ PASSED" if gates['all_passed'] else "✗ FAILED"
                    logger.info(f"  {dataset_name}: {status}")

                    metrics = result.get('comparison_metrics', {})
                    logger.info(f"    Accuracy drop: {metrics.get('accuracy_drop_pct', 0):.2f}%")
                    logger.info(f"    Speedup: {metrics.get('speedup', 0):.2f}x")
                    logger.info(f"    Compression: {metrics.get('compression_ratio', 0):.1f}x")
                elif 'error' in result:
                    logger.info(f"  {dataset_name}: ✗ ERROR - {result['error']}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Test MLX models against originals (refactored)')
    parser.add_argument('--models', default='config/models.yaml', help='Models config')
    parser.add_argument('--datasets', default='config/datasets.yaml', help='Datasets config')
    parser.add_argument('--model', help='Specific model to test')
    parser.add_argument('--dataset', help='Specific dataset to test on')
    parser.add_argument('--max-samples', type=int, default=200, help='Max samples per test')
    parser.add_argument('--results-dir', default='results/test_results', help='Output directory')
    parser.add_argument('--comparisons-dir', default='results/comparisons', help='Comparisons directory')

    args = parser.parse_args()

    tester = ModelTester(
        models_config=args.models,
        datasets_config=args.datasets,
        results_dir=args.results_dir,
        comparisons_dir=args.comparisons_dir
    )

    if args.model and args.dataset:
        # Test specific model/dataset
        comparison = tester.compare_models(args.model, args.dataset, max_samples=args.max_samples)
        if comparison:
            print(json.dumps(comparison, indent=2))
            return 0 if comparison['quality_gates']['all_passed'] else 1
        return 1

    else:
        # Test all models
        results = tester.test_all_models(max_samples=args.max_samples)

        # Check if all quality gates passed
        all_passed = True
        for model_results in results.values():
            for dataset_results in model_results.values():
                if not dataset_results.get('quality_gates', {}).get('all_passed', False):
                    all_passed = False

        return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
