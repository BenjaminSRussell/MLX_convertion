"""
Inference engine utilities for running model predictions.
"""

import time
from typing import List, Tuple, Any, Dict
import torch
import mlx.core as mx
import numpy as np
import logging

logger = logging.getLogger('MLX8BitTester.InferenceEngine')


class InferenceEngine:
    """
    Handles batch inference for both PyTorch and MLX models.
    """

    @staticmethod
    def run_pytorch_batch(
        model: Any,
        tokenizer: Any,
        texts: List[str],
        max_length: int = 512,
        device: str = 'cpu'
    ) -> Tuple[np.ndarray, int, float]:
        """
        Run inference on a batch of texts using PyTorch model.

        Args:
            model: PyTorch model
            tokenizer: HuggingFace tokenizer
            texts: List of input texts
            max_length: Maximum sequence length
            device: Device to run on ('cpu', 'cuda')

        Returns:
            Tuple of (predictions, num_tokens, latency_seconds)
        """
        start_time = time.time()

        # Tokenize
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        # Move to device
        if device == 'cuda':
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=1).cpu().numpy()

        # Count tokens
        num_tokens = inputs['input_ids'].numel()
        latency = time.time() - start_time

        return predictions, num_tokens, latency

    @staticmethod
    def run_pytorch_zero_shot_batch(
        classifier: Any,
        texts: List[str],
        candidate_labels: List[str]
    ) -> Tuple[List[str], float]:
        """
        Run zero-shot classification on a batch of texts.

        Args:
            classifier: HuggingFace zero-shot pipeline
            texts: List of input texts
            candidate_labels: List of candidate labels

        Returns:
            Tuple of (predictions, latency_seconds)
        """
        start_time = time.time()

        # Run zero-shot classification
        results = classifier(texts, candidate_labels=candidate_labels, truncation=True)

        # Extract top predictions
        if isinstance(results, list):
            predictions = [result['labels'][0] for result in results]
        else:
            predictions = [results['labels'][0]]

        latency = time.time() - start_time

        return predictions, latency

    @staticmethod
    def run_mlx_batch(
        model: Any,
        tokenizer: Any,
        texts: List[str],
        max_length: int = 512
    ) -> Tuple[np.ndarray, int, float]:
        """
        Run inference on a batch of texts using MLX model.

        Args:
            model: MLX model
            tokenizer: HuggingFace tokenizer (MLX-compatible)
            texts: List of input texts
            max_length: Maximum sequence length

        Returns:
            Tuple of (predictions, num_tokens, latency_seconds)
        """
        start_time = time.time()

        # Tokenize
        inputs = tokenizer(
            texts,
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
        predictions = mx.argmax(logits, axis=1).tolist()

        # Count tokens
        num_tokens = inputs['input_ids'].size
        latency = time.time() - start_time

        return np.array(predictions), num_tokens, latency

    @staticmethod
    def estimate_tokens_from_text(texts: List[str]) -> int:
        """
        Estimate number of tokens from text (rough approximation).
        Uses ~5 characters per token as heuristic.

        Args:
            texts: List of input texts

        Returns:
            Estimated number of tokens
        """
        total_chars = sum(len(t) for t in texts)
        return total_chars // 5

    @staticmethod
    def batch_generator(items: List[Any], batch_size: int):
        """
        Generate batches from a list of items.

        Args:
            items: List of items to batch
            batch_size: Size of each batch

        Yields:
            Batches of items
        """
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]

    @staticmethod
    def get_device() -> str:
        """
        Get the best available device for PyTorch.

        Returns:
            Device string ('cuda' or 'cpu')
        """
        if torch.cuda.is_available():
            logger.info("CUDA available, using GPU")
            return 'cuda'
        elif torch.backends.mps.is_available():
            logger.info("MPS available, using Apple Silicon GPU")
            return 'mps'
        else:
            logger.info("No GPU available, using CPU")
            return 'cpu'
