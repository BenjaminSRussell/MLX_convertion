"""
Unit tests for PyTorch converter
"""
import unittest
import tempfile
import yaml
import json
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.convert_pytorch import PyTorchInt8Converter, setup_logging


class TestConfigLoading(unittest.TestCase):
    """Test YAML configuration loading"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = Path(self.temp_dir) / 'test_models.yaml'

    def test_single_config_file(self):
        """Test loading a single config file"""
        config_data = {
            'models': [
                {
                    'name': 'test-model',
                    'hf_name': 'distilbert-base-uncased',
                    'task': 'classification',
                    'quantization': {
                        'bits': 8,
                        'dtype': 'int8',
                        'target_size_mb': 100,
                        'max_accuracy_drop': 0.02
                    },
                    'benchmarks': ['test']
                }
            ]
        }

        with open(self.config_file, 'w') as f:
            yaml.dump(config_data, f)

        converter = PyTorchInt8Converter(config_path=str(self.config_file))
        self.assertEqual(len(converter.config['models']), 1)
        self.assertEqual(converter.config['models'][0]['name'], 'test-model')

    def test_empty_config(self):
        """Test handling of empty config"""
        config_data = {'models': []}

        with open(self.config_file, 'w') as f:
            yaml.dump(config_data, f)

        converter = PyTorchInt8Converter(config_path=str(self.config_file))
        self.assertEqual(len(converter.config['models']), 0)

    def test_missing_config_file(self):
        """Test handling of missing config file"""
        converter = PyTorchInt8Converter(config_path='/nonexistent/path.yaml')
        self.assertEqual(len(converter.config['models']), 0)


class TestLogging(unittest.TestCase):
    """Test logging setup"""

    def test_logging_setup(self):
        """Test that logging is configured correctly"""
        logger = setup_logging('test_model')
        self.assertIsNotNone(logger)
        self.assertEqual(logger.name, 'PyTorchConverter_test_model')

    def test_logging_without_model_name(self):
        """Test logging setup without model name"""
        logger = setup_logging()
        self.assertIsNotNone(logger)
        self.assertEqual(logger.name, 'PyTorchConverter')


class TestOutputPaths(unittest.TestCase):
    """Test output path generation"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = Path(self.temp_dir) / 'test_models.yaml'

        config_data = {
            'models': [
                {
                    'name': 'test-model',
                    'hf_name': 'distilbert-base-uncased',
                    'task': 'classification',
                    'quantization': {
                        'bits': 8,
                        'dtype': 'int8',
                        'target_size_mb': 100,
                        'max_accuracy_drop': 0.02
                    },
                    'benchmarks': ['test']
                }
            ]
        }

        with open(self.config_file, 'w') as f:
            yaml.dump(config_data, f)

    def test_output_directory_creation(self):
        """Test that output directories are created"""
        output_dir = Path(self.temp_dir) / 'output'
        converter = PyTorchInt8Converter(
            config_path=str(self.config_file),
            output_dir=str(output_dir)
        )

        self.assertEqual(str(converter.output_dir), str(output_dir))

    def test_results_directory_creation(self):
        """Test that results directory is created in temp"""
        converter = PyTorchInt8Converter(config_path=str(self.config_file))
        self.assertTrue(converter.results_dir.exists())


class TestDryRun(unittest.TestCase):
    """Test dry-run functionality"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = Path(self.temp_dir) / 'test_models.yaml'

        config_data = {
            'models': [
                {
                    'name': 'tiny-model',
                    'hf_name': 'prajjwal1/bert-tiny',
                    'task': 'classification',
                    'quantization': {
                        'bits': 8,
                        'dtype': 'int8',
                        'target_size_mb': 20,
                        'max_accuracy_drop': 0.02
                    },
                    'benchmarks': ['test']
                }
            ]
        }

        with open(self.config_file, 'w') as f:
            yaml.dump(config_data, f)

    def test_dry_run_mode(self):
        """Test that dry-run doesn't actually convert"""
        converter = PyTorchInt8Converter(
            config_path=str(self.config_file),
            output_dir=str(Path(self.temp_dir) / 'output'),
            dry_run=True
        )

        model_config = converter.config['models'][0]
        result = converter.convert_single_model(model_config)

        self.assertTrue(result['success'])
        self.assertTrue(result.get('dry_run', False))


class TestSkipExisting(unittest.TestCase):
    """Test duplicate detection"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = Path(self.temp_dir) / 'test_models.yaml'
        self.output_dir = Path(self.temp_dir) / 'output'

        config_data = {
            'models': [
                {
                    'name': 'test-model',
                    'hf_name': 'prajjwal1/bert-tiny',
                    'task': 'classification',
                    'quantization': {
                        'bits': 8,
                        'dtype': 'int8',
                        'target_size_mb': 20,
                        'max_accuracy_drop': 0.02
                    },
                    'benchmarks': ['test']
                }
            ]
        }

        with open(self.config_file, 'w') as f:
            yaml.dump(config_data, f)

    def test_skip_existing_model(self):
        """Test that existing models are skipped"""
        # Create fake existing model
        existing_path = self.output_dir / 'test-model-mlx-q8'
        existing_path.mkdir(parents=True, exist_ok=True)

        metadata = {
            'model_name': 'test-model',
            'conversion_time_seconds': 10.0
        }

        with open(existing_path / 'conversion_metadata.json', 'w') as f:
            json.dump(metadata, f)

        converter = PyTorchInt8Converter(
            config_path=str(self.config_file),
            output_dir=str(self.output_dir),
            skip_existing=True
        )

        model_config = converter.config['models'][0]
        result = converter.convert_single_model(model_config)

        self.assertTrue(result['success'])
        self.assertTrue(result.get('skipped', False))


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestConfigLoading))
    suite.addTests(loader.loadTestsFromTestCase(TestLogging))
    suite.addTests(loader.loadTestsFromTestCase(TestOutputPaths))
    suite.addTests(loader.loadTestsFromTestCase(TestDryRun))
    suite.addTests(loader.loadTestsFromTestCase(TestSkipExisting))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
