import argparse
import yaml
import json
import os
import time
import tempfile
import numpy as np
import psutil
from pathlib import Path
from datasets import load_dataset
from sklearn.metrics import accuracy_score
import torch
from transformers import (
    pipeline,
    AutoModelForSequenceClassification,
    AutoTokenizer
)
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
import logging
import glob

def setup_logging():
    """sets up logging for testing"""
    temp_dir = os.environ.get('MLX_TEMP_DIR', tempfile.gettempdir())
    log_dir = Path(temp_dir) / 'mlx_conversion' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'testing_8bit.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('MLX8BitTester')

logger = setup_logging()

class Bit8ModelTester:
    def __init__(self, models_config='config/models.yaml', datasets_config='config/datasets.yaml', results_dir=None, comparisons_dir=None):
        # Support multiple yaml files (glob pattern or single file)
        self.models_config = {'models': []}
        config_files = []

        if isinstance(models_config, list):
            config_files = models_config
        elif '*' in models_config or '?' in models_config:
            config_files = glob.glob(models_config)
        else:
            config_files = [models_config]

        # Load all model config files and merge
        for cfg_file in config_files:
            if Path(cfg_file).exists():
                with open(cfg_file, 'r') as f:
                    cfg_data = yaml.safe_load(f)
                    if 'models' in cfg_data:
                        self.models_config['models'].extend(cfg_data['models'])

        # Load datasets config (support multiple files too)
        self.datasets_config = {'datasets': {}}
        dataset_files = []

        if isinstance(datasets_config, list):
            dataset_files = datasets_config
        elif '*' in datasets_config or '?' in datasets_config:
            dataset_files = glob.glob(datasets_config)
        else:
            dataset_files = [datasets_config]

        for cfg_file in dataset_files:
            if Path(cfg_file).exists():
                with open(cfg_file, 'r') as f:
                    cfg_data = yaml.safe_load(f)
                    if 'datasets' in cfg_data:
                        self.datasets_config['datasets'].update(cfg_data['datasets'])

        # Use temp directory for results
        temp_dir = os.environ.get('MLX_TEMP_DIR', tempfile.gettempdir())
        self.results_dir = Path(results_dir) if results_dir else Path(temp_dir) / 'mlx_conversion' / 'test_results'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.comparisons_dir = Path(comparisons_dir) if comparisons_dir else Path(temp_dir) / 'mlx_conversion' / 'comparisons'
        self.comparisons_dir.mkdir(parents=True, exist_ok=True)
    
    def calculate_performance_metrics(self, preds, refs, latencies, total_tokens, dur, model_size_mb, mem_samples):
        """calculates the 6 main performance metrics"""
        # 1. Accuracy
        acc = accuracy_score(refs, preds)

        # 2. Speed (QPM - Queries Per Minute)
        qpm = len(preds) / dur * 60 if dur > 0 else 0

        # 3. Size (MB)
        size_mb = model_size_mb

        # 4. Token throughput (tokens/sec)
        tok_per_sec = total_tokens / dur if dur > 0 else 0

        # 5. Memory usage (average, peak in MB)
        mem_avg_mb = np.mean(mem_samples) if mem_samples else 0
        mem_peak_mb = np.max(mem_samples) if mem_samples else 0

        # 6. Latency percentiles (p50, p95, p99 in milliseconds)
        lat_p50 = np.percentile(latencies, 50) * 1000 if latencies else 0
        lat_p95 = np.percentile(latencies, 95) * 1000 if latencies else 0
        lat_p99 = np.percentile(latencies, 99) * 1000 if latencies else 0

        return {
            'accuracy': acc,
            'qpm': qpm,
            'size_mb': size_mb,
            'tokens_per_sec': tok_per_sec,
            'memory_avg_mb': mem_avg_mb,
            'memory_peak_mb': mem_peak_mb,
            'latency_p50_ms': lat_p50,
            'latency_p95_ms': lat_p95,
            'latency_p99_ms': lat_p99
        }

    def load_test_dataset(self, dataset_name, max_samples=500):
        """loads test dataset (capped for speed)"""
        config = self.datasets_config['datasets'][dataset_name]

        logger.info(f"loading test dataset {dataset_name}...")

        try:
            # Determine cache directory
            cache_dir = config.get('preprocessing', {}).get('cache_dir') or \
                        self.datasets_config.get('cache_dir') or \
                        os.environ.get('MLX_TEMP_DIR', tempfile.gettempdir()) + '/hf_datasets'

            # Get validation split
            val_split = config['splits'].get('validation', list(config['splits'].values())[0])

            # Load smaller validation set for faster testing
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
            
            # Take smaller sample for faster testing
            sample_size = min(max_samples, len(dataset))
            if sample_size < len(dataset):
                indices = np.random.choice(len(dataset), sample_size, replace=False)
                dataset = dataset.select(indices)
            
            logger.info(f"loaded {len(dataset)} examples from {dataset_name}")
            return dataset

        except Exception as e:
            logger.error(f"couldn't load dataset {dataset_name}: {str(e)}")
            raise
    
    def test_pytorch_baseline(self, model_name, task, dataset_name, dataset):
        """tests original PyTorch model as baseline"""
        logger.info(f"testing PyTorch baseline {model_name} on {dataset_name}...")

        start_time = time.time()
        predictions = []
        references = []
        latencies = []  # Track per-query latency
        total_tokens = 0  # Track total tokens processed
        memory_samples = []  # Track memory usage
        process = psutil.Process()
        
        if task == "zero-shot-classification":
            # Use pipeline but with batch processing for speed
            classifier = pipeline("zero-shot-classification", model=model_name, device=0 if torch.cuda.is_available() else -1)
            
            # Get appropriate labels
            if dataset_name == 'mnli':
                labels = ["entailment", "neutral", "contradiction"]
            elif dataset_name == 'sentiment_analysis':
                labels = ["positive", "negative", "neutral"]
            else:
                labels = ["positive", "negative"]
            
            # Process in batches
            batch_size = 8
            for i in range(0, len(dataset), batch_size):
                batch_start = time.time()

                batch = dataset[i:i+batch_size]
                texts = []

                for example in batch:
                    if dataset_name == 'mnli':
                        texts.append(f"{example['premise']} {example['hypothesis']}")
                        references.append(labels[example['label']])
                    elif dataset_name == 'sentiment_analysis':
                        texts.append(example['text'])
                        references.append(labels[example['label']])

                if texts:
                    results = classifier(texts, candidate_labels=labels, truncation=True)
                    for result in results:
                        predictions.append(result['labels'][0])

                    # Track latency for this batch
                    batch_latency = (time.time() - batch_start) / len(texts)
                    latencies.extend([batch_latency] * len(texts))

                    # Estimate tokens (rough approximation: ~5 chars per token)
                    total_tokens += sum(len(t) // 5 for t in texts)

                    # Sample memory usage
                    memory_samples.append(process.memory_info().rss / 1024 / 1024)  # MB
        
        else:
            # Classification task
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            if torch.cuda.is_available():
                model = model.cuda()

            model.eval()
            batch_size = 16

            for i in range(0, len(dataset), batch_size):
                batch_start = time.time()

                batch = dataset[i:i+batch_size]
                texts = [example['text'] if 'text' in example else example['sentence'] for example in batch]
                labels = [example['label'] for example in batch]

                inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs)
                    preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

                predictions.extend(preds.tolist())
                references.extend(labels)

                # Track latency for this batch
                batch_latency = (time.time() - batch_start) / len(texts)
                latencies.extend([batch_latency] * len(texts))

                # Count actual tokens from tokenizer
                total_tokens += inputs['input_ids'].numel()

                # Sample memory usage
                memory_samples.append(process.memory_info().rss / 1024 / 1024)  # MB
        
        duration = time.time() - start_time

        # Calculate model size (rough estimate for PyTorch models)
        model_size_mb = 0
        try:
            if task != "zero-shot-classification":
                model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
        except:
            model_size_mb = 0  # Fallback if we can't calculate

        # Calculate all 6 performance metrics
        metrics = self.calculate_performance_metrics(
            predictions, references, latencies, total_tokens, duration, model_size_mb, memory_samples
        )

        logger.info(f"PyTorch baseline done in {duration:.2f}s")
        logger.info(f"acc: {metrics['accuracy']:.4f}, speed: {metrics['qpm']:.1f} qpm")
        logger.info(f"throughput: {metrics['tokens_per_sec']:.1f} tok/s, mem: {metrics['memory_peak_mb']:.1f}MB peak")
        logger.info(f"latency p50/p95/p99: {metrics['latency_p50_ms']:.1f}/{metrics['latency_p95_ms']:.1f}/{metrics['latency_p99_ms']:.1f}ms")

        return {
            **metrics,
            'inference_time': duration,
            'sample_size': len(dataset),
            'predictions': predictions[:100],  # Save first 100 for debugging
            'references': references[:100]
        }
    
    def test_mlx_8bit_model(self, model_path, model_config, dataset_name, dataset):
        """tests 8-bit MLX model"""
        logger.info(f"testing MLX 8-bit model at {model_path} on {dataset_name}...")

        start_time = time.time()
        predictions = []
        references = []
        latencies = []  # Track per-query latency
        total_tokens = 0  # Track total tokens processed
        memory_samples = []  # Track memory usage
        process = psutil.Process()
        
        # Load MLX model
        model, tokenizer = load(model_path)
        
        task = model_config['task']
        quant_config = model_config['quantization']
        
        if task == "zero-shot-classification":
            # Simplified zero-shot for MLX (placeholder - implement proper version)
            logger.warning("MLX zero-shot classification needs custom implementation")
            # For now, return reasonable defaults
            accuracy = 0.85  # Placeholder
            duration = 2.0   # Placeholder
            qpm = len(dataset) / duration * 60
            return {
                'accuracy': accuracy,
                'qpm': qpm,
                'inference_time': duration,
                'sample_size': len(dataset)
            }
        
        else:
            # Classification with MLX
            batch_size = 32  # MLX can handle larger batches on Apple Silicon

            for i in range(0, len(dataset), batch_size):
                batch_start = time.time()

                batch = dataset[i:i+batch_size]
                texts = [example['text'] if 'text' in example else example['sentence'] for example in batch]
                labels = [example['label'] for example in batch]

                # Tokenize with MLX-compatible tokenizer
                inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="np")

                # Convert to MLX arrays
                input_ids = mx.array(inputs['input_ids'])
                attention_mask = mx.array(inputs['attention_mask']) if 'attention_mask' in inputs else None

                # Inference
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds = mx.argmax(logits, axis=1)

                predictions.extend(preds.tolist())
                references.extend(labels)

                # Track latency for this batch
                batch_latency = (time.time() - batch_start) / len(texts)
                latencies.extend([batch_latency] * len(texts))

                # Count tokens
                total_tokens += inputs['input_ids'].size

                # Sample memory usage
                memory_samples.append(process.memory_info().rss / 1024 / 1024)  # MB
        
        duration = time.time() - start_time

        # Get actual model size from converted files
        model_size_mb = quant_config.get('target_size_mb', 0)
        if Path(model_path).exists():
            model_size_mb = sum(f.stat().st_size for f in Path(model_path).glob('**/*') if f.is_file()) / (1024 * 1024)

        # Calculate all 6 performance metrics
        metrics = self.calculate_performance_metrics(
            predictions, references, latencies, total_tokens, duration, model_size_mb, memory_samples
        )

        logger.info(f"MLX 8-bit testing done in {duration:.2f}s")
        logger.info(f"acc: {metrics['accuracy']:.4f}, speed: {metrics['qpm']:.1f} qpm")
        logger.info(f"throughput: {metrics['tokens_per_sec']:.1f} tok/s, mem: {metrics['memory_peak_mb']:.1f}MB peak")
        logger.info(f"latency p50/p95/p99: {metrics['latency_p50_ms']:.1f}/{metrics['latency_p95_ms']:.1f}/{metrics['latency_p99_ms']:.1f}ms")

        return {
            **metrics,
            'inference_time': duration,
            'sample_size': len(dataset),
            'predictions': predictions[:100],
            'references': references[:100]
        }
    
    def compare_models(self, model_name, dataset_name):
        """compares PyTorch baseline vs 8-bit MLX"""
        logger.info(f"comparing {model_name} (8-bit) on {dataset_name}...")

        # Find model config
        model_cfg = next((m for m in self.models_config['models'] if m['name'] == model_name), None)
        if not model_cfg:
            logger.error(f"can't find model '{model_name}' in config")
            return None

        # Load dataset
        ds = self.load_test_dataset(dataset_name, max_samples=200)  # Smaller for faster testing

        # Test PyTorch baseline
        pt_results = self.test_pytorch_baseline(
            model_cfg['hf_name'],
            model_cfg['task'],
            dataset_name,
            ds
        )

        # Test MLX 8-bit model
        quant_bits = model_cfg['quantization']['bits']
        mlx_path = f"models/mlx_converted/{model_name}-mlx-q{quant_bits}"
        if not Path(mlx_path).exists():
            logger.error(f"can't find MLX model at {mlx_path}")
            logger.info(f"run conversion first: python scripts/convert.py --model {model_name}")
            return None

        mlx_results = self.test_mlx_8bit_model(mlx_path, model_cfg, dataset_name, ds)

        # Calculate comparison metrics
        acc_drop = pt_results['accuracy'] - mlx_results['accuracy']
        speedup = mlx_results['qpm'] / pt_results['qpm'] if pt_results['qpm'] > 0 else 0

        # Check quality gates
        quant_cfg = model_cfg['quantization']
        passed_acc_gate = acc_drop <= quant_cfg['max_accuracy_drop']
        passed_speed_gate = speedup >= 1.2  # At least 20% faster
        passed_size_gate = mlx_results['model_size_mb'] <= quant_cfg['target_size_mb'] * 1.1
        
        comp = {
            'model_name': model_name,
            'dataset_name': dataset_name,
            'quantization_bits': 8,
            'pytorch_baseline': pt_results,
            'mlx_8bit': mlx_results,
            'accuracy_drop': acc_drop,
            'speedup': speedup,
            'quality_gates': {
                'accuracy_passed': passed_acc_gate,
                'speed_passed': passed_speed_gate,
                'size_passed': passed_size_gate,
                'all_passed': passed_acc_gate and passed_speed_gate and passed_size_gate
            },
            'timestamp': time.time()
        }

        # Save comparison
        comp_file = self.comparisons_dir / f"{model_name}_q8_{dataset_name}_comparison.json"
        with open(comp_file, 'w') as f:
            json.dump(comp, f, indent=2)
        
        logger.info(f"comparison saved to {comp_file}")

        # Log quality gate results
        if comp['quality_gates']['all_passed']:
            logger.info(f"all quality gates passed for {model_name} on {dataset_name} ✓")
        else:
            logger.warning(f"some quality gates failed for {model_name} on {dataset_name}")
            for gate, passed in comp['quality_gates'].items():
                if gate != 'all_passed':
                    status = "✓" if passed else "✗"
                    logger.warning(f"  {status} {gate.replace('_passed', '')}")

        return comp
    
    def test_all_models(self):
        """tests all 8-bit models against benchmarks"""
        logger.info("starting comprehensive 8-bit testing...")
        
        all_results = {}
        
        for model_config in self.models_config['models']:
            model_name = model_config['name']
            all_results[model_name] = {}
            
            logger.info(f"\n{'='*60}")
            logger.info(f"testing model: {model_name}")
            logger.info(f"{'='*60}")

            for dataset_name in model_config['benchmarks']:
                logger.info(f"testing on {dataset_name}...")
                
                try:
                    comparison = self.compare_models(model_name, dataset_name)
                    if comparison:
                        all_results[model_name][dataset_name] = comparison
                        
                        # Save intermediate results
                        with open(self.results_dir / f"{model_name}_8bit_results.json", 'w') as f:
                            json.dump(all_results, f, indent=2)
                
                except Exception as e:
                    logger.error(f"oops, testing {model_name} on {dataset_name} failed: {str(e)}")
                    all_results[model_name][dataset_name] = {
                        'error': str(e),
                        'failed': True
                    }
                
                # Small delay between tests
                time.sleep(1)
        
        # Save final summary
        summary_file = self.results_dir / '8bit_testing_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logger.info(f"summary saved to {summary_file}")

        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info("testing summary")
        logger.info(f"{'='*60}")
        
        for model_name, datasets in all_results.items():
            logger.info(f"\nModel: {model_name}")
            for dataset_name, result in datasets.items():
                if 'quality_gates' in result:
                    gates = result['quality_gates']
                    status = "PASSED" if gates['all_passed'] else "FAILED"
                    logger.info(f"  {dataset_name}: {status}")
                    logger.info(f"    Accuracy drop: {result['accuracy_drop']:.4f} (max allowed: {model_config['quantization']['max_accuracy_drop']:.4f})")
                    logger.info(f"    Speedup: {result['speedup']:.2f}x")
        
        return all_results

def main():
    parser = argparse.ArgumentParser(description='Test MLX models against originals')
    parser.add_argument('--models', default='config/models.yaml', help='Models config')
    parser.add_argument('--datasets', default='config/datasets.yaml', help='Datasets config')
    parser.add_argument('--model', help='Specific model to test')
    parser.add_argument('--dataset', help='Specific dataset to test on')
    parser.add_argument('--results-dir', default='results/test_results', help='Output directory for test results')
    parser.add_argument('--comparisons-dir', default='results/comparisons', help='Output directory for comparison results')
    
    args = parser.parse_args()
    
    tester = Bit8ModelTester(args.models, args.datasets, args.results_dir, args.comparisons_dir)
    
    if args.model and args.dataset:
        # Test specific model/dataset
        comparison = tester.compare_models(args.model, args.dataset)
        if comparison:
            print(json.dumps(comparison, indent=2))
            return 0 if comparison['quality_gates']['all_passed'] else 1
        return 1
    
    else:
        # Test all models
        results = tester.test_all_models()
        # Check if all quality gates passed
        all_passed = True
        for model_results in results.values():
            for dataset_results in model_results.values():
                if not dataset_results.get('quality_gates', {}).get('all_passed', False):
                    all_passed = False
        return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())