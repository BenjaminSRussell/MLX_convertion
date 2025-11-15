"""
handles standard text classification (sentiment, topics, etc)
"""

from typing import Dict, List, Any, Tuple
from pathlib import Path
import time
import logging
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import mlx.core as mx
from mlx_lm import load

from .base_handler import BaseHandler
from ..utils.inference import InferenceEngine

logger = logging.getLogger('MLX8BitTester.TextClassificationHandler')


class TextClassificationHandler(BaseHandler):
    """handles text classification (e.g. BERT for sentiment)"""

    def prepare_data(self, dataset: Any) -> Tuple[List[str], List[Any]]:
        """
        Prepare text classification dataset.

        Args:
            dataset: HuggingFace dataset

        Returns:
            Tuple of (texts, labels)
        """
        texts = []
        labels = []

        for example in dataset:
            # Try different field names
            if 'text' in example:
                text = example['text']
            elif 'sentence' in example:
                text = example['sentence']
            elif 'content' in example:
                text = example['content']
            else:
                logger.warning(f"Could not find text field in example: {example.keys()}")
                continue

            texts.append(text)
            labels.append(example['label'])

        return texts, labels

    def test_pytorch_baseline(
        self,
        model_name: str,
        dataset_name: str,
        dataset: Any,
        memory_tracker: Any,
        metrics_calculator: Any
    ) -> Dict[str, Any]:
        """
        Test PyTorch text classification model.

        Args:
            model_name: HuggingFace model name
            dataset_name: Name of the dataset
            dataset: HuggingFace dataset
            memory_tracker: MemoryTracker instance
            metrics_calculator: MetricsCalculator instance

        Returns:
            Dictionary with test results
        """
        logger.info(f"Testing PyTorch baseline: {model_name} on {dataset_name}")

        memory_tracker.start()
        start_time = time.time()

        # Prepare data
        texts, references = self.prepare_data(dataset)

        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

        # Move to device
        device = InferenceEngine.get_device()
        if device == 'cuda':
            model = model.cuda()
        elif device == 'mps':
            model = model.to('mps')

        model.eval()

        # Run inference in batches
        predictions = []
        latencies = []
        total_tokens = 0
        batch_size = self.get_batch_size_pytorch()
        max_length = self.get_max_length()

        for i in range(0, len(texts), batch_size):
            batch_start = time.time()

            batch_texts = texts[i:i+batch_size]
            batch_labels = references[i:i+batch_size]

            # Tokenize
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )

            # Move to device
            if device == 'cuda':
                inputs = {k: v.cuda() for k, v in inputs.items()}
            elif device == 'mps':
                inputs = {k: v.to('mps') for k, v in inputs.items()}

            # Inference
            with torch.no_grad():
                outputs = model(**inputs)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

            predictions.extend(preds.tolist())

            # Track metrics
            batch_latency = (time.time() - batch_start) / len(batch_texts)
            latencies.extend([batch_latency] * len(batch_texts))
            total_tokens += inputs['input_ids'].numel()

            # Sample memory
            memory_tracker.sample()

        duration = time.time() - start_time

        # Calculate model size
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024

        # Calculate metrics
        metrics = metrics_calculator.calculate_all_performance_metrics(
            predictions=predictions,
            references=references,
            latencies=latencies,
            total_tokens=total_tokens,
            duration=duration,
            model_size_mb=model_size_mb,
            memory_samples=memory_tracker.get_samples()
        )

        logger.info(f"PyTorch baseline completed in {duration:.2f}s")
        metrics_calculator.log_metrics(metrics, prefix="PyTorch")

        # Clean up
        del model, tokenizer
        if device == 'cuda':
            torch.cuda.empty_cache()

        return {
            **metrics,
            'inference_time': duration,
            'sample_size': len(texts),
            'predictions': predictions[:100],
            'references': references[:100]
        }

    def test_mlx_model(
        self,
        model_path: Path,
        dataset_name: str,
        dataset: Any,
        memory_tracker: Any,
        metrics_calculator: Any
    ) -> Dict[str, Any]:
        """
        Test MLX text classification model.

        Args:
            model_path: Path to MLX model
            dataset_name: Name of the dataset
            dataset: HuggingFace dataset
            memory_tracker: MemoryTracker instance
            metrics_calculator: MetricsCalculator instance

        Returns:
            Dictionary with test results
        """
        logger.info(f"Testing MLX model: {model_path} on {dataset_name}")

        memory_tracker.start()
        start_time = time.time()

        # Prepare data
        texts, references = self.prepare_data(dataset)

        # Load MLX model
        try:
            model, tokenizer = load(str(model_path))
        except Exception as e:
            logger.error(f"Failed to load MLX model: {e}")
            raise

        # Run inference in batches
        predictions = []
        latencies = []
        total_tokens = 0
        batch_size = self.get_batch_size_mlx()
        max_length = self.get_max_length()

        for i in range(0, len(texts), batch_size):
            batch_start = time.time()

            batch_texts = texts[i:i+batch_size]

            # Tokenize
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="np"
            )

            # Convert to MLX arrays
            input_ids = mx.array(inputs['input_ids'])
            attention_mask = mx.array(inputs['attention_mask']) if 'attention_mask' in inputs else None

            # Inference
            if attention_mask is not None:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                outputs = model(input_ids=input_ids)

            # Get predictions
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            preds = mx.argmax(logits, axis=1).tolist()

            predictions.extend(preds)

            # Track metrics
            batch_latency = (time.time() - batch_start) / len(batch_texts)
            latencies.extend([batch_latency] * len(batch_texts))
            total_tokens += inputs['input_ids'].size

            # Sample memory
            memory_tracker.sample()

        duration = time.time() - start_time

        # Get model size
        quant_config = self.model_config.get('quantization', {})
        model_size_mb = quant_config.get('target_size_mb', 0)

        # Calculate actual size from files
        if model_path.exists():
            model_size_mb = sum(f.stat().st_size for f in model_path.glob('**/*') if f.is_file()) / (1024 * 1024)

        # Calculate metrics
        metrics = metrics_calculator.calculate_all_performance_metrics(
            predictions=predictions,
            references=references,
            latencies=latencies,
            total_tokens=total_tokens,
            duration=duration,
            model_size_mb=model_size_mb,
            memory_samples=memory_tracker.get_samples()
        )

        logger.info(f"MLX testing completed in {duration:.2f}s")
        metrics_calculator.log_metrics(metrics, prefix="MLX")

        return {
            **metrics,
            'inference_time': duration,
            'sample_size': len(texts),
            'predictions': predictions[:100],
            'references': references[:100]
        }

    def get_quality_metrics(self) -> List[str]:
        """
        Get quality metrics for text classification.

        Returns:
            List of metric names
        """
        return ['accuracy', 'f1', 'precision', 'recall']

    def validate_quality(
        self,
        pytorch_results: Dict[str, Any],
        mlx_results: Dict[str, Any]
    ) -> Dict[str, bool]:
        """
        Validate MLX model against PyTorch baseline.

        Args:
            pytorch_results: PyTorch test results
            mlx_results: MLX test results

        Returns:
            Dictionary of quality gate results
        """
        quant_config = self.model_config.get('quantization', {})
        max_accuracy_drop = quant_config.get('max_accuracy_drop', 0.015)  # 1.5% default

        accuracy_drop = pytorch_results['accuracy'] - mlx_results['accuracy']
        accuracy_passed = accuracy_drop <= max_accuracy_drop

        # Speed gate
        speedup = mlx_results['qpm'] / pytorch_results['qpm'] if pytorch_results['qpm'] > 0 else 0.0
        speed_passed = speedup >= 1.2

        # Size gate
        target_size_mb = quant_config.get('target_size_mb', 0)
        size_passed = mlx_results['size_mb'] <= target_size_mb * 1.1 if target_size_mb > 0 else True

        return {
            'accuracy_passed': accuracy_passed,
            'speed_passed': speed_passed,
            'size_passed': size_passed,
            'all_passed': accuracy_passed and speed_passed and size_passed
        }

    def get_batch_size_pytorch(self) -> int:
        """batch size for PyTorch"""
        return 16

    def get_batch_size_mlx(self) -> int:
        """batch size for MLX (handles bigger batches)"""
        return 32
