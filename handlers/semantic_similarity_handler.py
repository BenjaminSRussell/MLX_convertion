"""
Handler for semantic similarity / sentence embedding models.
Supports STS tasks and embedding extraction.
"""

from typing import Dict, List, Any, Tuple
from pathlib import Path
import time
import logging
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
import mlx.core as mx
from mlx_lm import load

from .base_handler import BaseHandler
from ..utils.inference import InferenceEngine

logger = logging.getLogger('MLX8BitTester.SemanticSimilarityHandler')


class SemanticSimilarityHandler(BaseHandler):
    """
    Handler for semantic similarity and sentence embedding models.

    Examples: all-MiniLM-L6-v2, all-mpnet-base-v2, sentence-transformers models
    """

    def mean_pooling(self, model_output: Any, attention_mask: Any) -> np.ndarray:
        """
        Mean pooling - take attention mask into account for correct averaging.

        Args:
            model_output: Model outputs (last_hidden_state)
            attention_mask: Attention mask

        Returns:
            Pooled embeddings
        """
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings

        if isinstance(token_embeddings, torch.Tensor):
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return (sum_embeddings / sum_mask).cpu().numpy()
        else:
            # MLX arrays
            input_mask_expanded = mx.expand_dims(attention_mask, -1)
            input_mask_expanded = mx.broadcast_to(input_mask_expanded, token_embeddings.shape)
            sum_embeddings = mx.sum(token_embeddings * input_mask_expanded, axis=1)
            sum_mask = mx.maximum(mx.sum(input_mask_expanded, axis=1), 1e-9)
            return np.array(sum_embeddings / sum_mask)

    def prepare_data(self, dataset: Any) -> Tuple[List[Tuple[str, str]], List[float]]:
        """
        Prepare STS (Semantic Textual Similarity) dataset.

        Args:
            dataset: HuggingFace dataset

        Returns:
            Tuple of (sentence_pairs, similarity_scores)
        """
        sentence_pairs = []
        scores = []

        for example in dataset:
            # STS format
            if 'sentence1' in example and 'sentence2' in example:
                sentence_pairs.append((example['sentence1'], example['sentence2']))
                scores.append(float(example.get('score', example.get('label', 0))))

            # Alternate format
            elif 'text1' in example and 'text2' in example:
                sentence_pairs.append((example['text1'], example['text2']))
                scores.append(float(example.get('score', example.get('label', 0))))

            else:
                logger.warning(f"Could not parse sentence pair from example: {example.keys()}")
                continue

        return sentence_pairs, scores

    def compute_embeddings_pytorch(
        self,
        model: Any,
        tokenizer: Any,
        sentences: List[str],
        device: str,
        batch_size: int,
        max_length: int
    ) -> np.ndarray:
        """
        Compute sentence embeddings using PyTorch model.

        Args:
            model: PyTorch model
            tokenizer: Tokenizer
            sentences: List of sentences
            device: Device ('cpu', 'cuda', 'mps')
            batch_size: Batch size
            max_length: Max sequence length

        Returns:
            Numpy array of embeddings
        """
        all_embeddings = []

        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i+batch_size]

            # Tokenize
            inputs = tokenizer(
                batch_sentences,
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

            # Get embeddings
            with torch.no_grad():
                outputs = model(**inputs)
                embeddings = self.mean_pooling(outputs, inputs['attention_mask'])

            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)

    def compute_embeddings_mlx(
        self,
        model: Any,
        tokenizer: Any,
        sentences: List[str],
        batch_size: int,
        max_length: int
    ) -> np.ndarray:
        """
        Compute sentence embeddings using MLX model.

        Args:
            model: MLX model
            tokenizer: Tokenizer
            sentences: List of sentences
            batch_size: Batch size
            max_length: Max sequence length

        Returns:
            Numpy array of embeddings
        """
        all_embeddings = []

        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i+batch_size]

            # Tokenize
            inputs = tokenizer(
                batch_sentences,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="np"
            )

            # Convert to MLX
            input_ids = mx.array(inputs['input_ids'])
            attention_mask = mx.array(inputs['attention_mask'])

            # Get embeddings
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = self.mean_pooling((outputs,), attention_mask)

            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)

    def test_pytorch_baseline(
        self,
        model_name: str,
        dataset_name: str,
        dataset: Any,
        memory_tracker: Any,
        metrics_calculator: Any
    ) -> Dict[str, Any]:
        """
        Test PyTorch semantic similarity model.

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
        sentence_pairs, reference_scores = self.prepare_data(dataset)

        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        device = InferenceEngine.get_device()
        if device == 'cuda':
            model = model.cuda()
        elif device == 'mps':
            model = model.to('mps')

        model.eval()

        # Compute embeddings for all sentences
        batch_size = self.get_batch_size_pytorch()
        max_length = self.get_max_length()

        # Extract all unique sentences
        all_sentences = []
        for s1, s2 in sentence_pairs:
            all_sentences.extend([s1, s2])

        # Compute embeddings
        embeddings = self.compute_embeddings_pytorch(
            model, tokenizer, all_sentences, device, batch_size, max_length
        )

        # Calculate cosine similarities
        predicted_scores = []
        for i in range(len(sentence_pairs)):
            emb1 = embeddings[i * 2]
            emb2 = embeddings[i * 2 + 1]
            cosine_sim = metrics_calculator.calculate_cosine_similarity(emb1, emb2)
            predicted_scores.append(cosine_sim)

        duration = time.time() - start_time

        # Calculate model size
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024

        # For embedding models, use Spearman correlation as primary metric
        spearman_corr = metrics_calculator.calculate_spearman_correlation(
            predicted_scores, reference_scores
        )
        pearson_corr = metrics_calculator.calculate_pearson_correlation(
            predicted_scores, reference_scores
        )

        # Calculate throughput metrics
        total_tokens = len(all_sentences) * max_length // 2  # Rough estimate
        qpm = len(sentence_pairs) / duration * 60

        logger.info(f"PyTorch baseline completed in {duration:.2f}s")
        logger.info(f"Spearman correlation: {spearman_corr:.4f}")
        logger.info(f"Pearson correlation: {pearson_corr:.4f}")

        # Clean up
        del model, tokenizer
        if device == 'cuda':
            torch.cuda.empty_cache()

        return {
            'spearman_correlation': spearman_corr,
            'pearson_correlation': pearson_corr,
            'accuracy': spearman_corr,  # Use correlation as "accuracy" for compatibility
            'qpm': qpm,
            'size_mb': model_size_mb,
            'tokens_per_sec': total_tokens / duration,
            'memory_avg_mb': memory_tracker.get_average_memory_mb(),
            'memory_peak_mb': memory_tracker.get_peak_memory_mb(),
            'latency_p50_ms': (duration / len(sentence_pairs)) * 1000,
            'latency_p95_ms': 0.0,
            'latency_p99_ms': 0.0,
            'inference_time': duration,
            'sample_size': len(sentence_pairs),
            'predictions': predicted_scores[:100],
            'references': reference_scores[:100]
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
        Test MLX semantic similarity model.

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
        sentence_pairs, reference_scores = self.prepare_data(dataset)

        # Load MLX model
        try:
            model, tokenizer = load(str(model_path))
        except Exception as e:
            logger.error(f"Failed to load MLX model: {e}")
            raise

        # Compute embeddings for all sentences
        batch_size = self.get_batch_size_mlx()
        max_length = self.get_max_length()

        # Extract all unique sentences
        all_sentences = []
        for s1, s2 in sentence_pairs:
            all_sentences.extend([s1, s2])

        # Compute embeddings
        embeddings = self.compute_embeddings_mlx(
            model, tokenizer, all_sentences, batch_size, max_length
        )

        # Calculate cosine similarities
        predicted_scores = []
        for i in range(len(sentence_pairs)):
            emb1 = embeddings[i * 2]
            emb2 = embeddings[i * 2 + 1]
            cosine_sim = metrics_calculator.calculate_cosine_similarity(emb1, emb2)
            predicted_scores.append(cosine_sim)

        duration = time.time() - start_time

        # Get model size
        quant_config = self.model_config.get('quantization', {})
        model_size_mb = quant_config.get('target_size_mb', 0)
        if model_path.exists():
            model_size_mb = sum(f.stat().st_size for f in model_path.glob('**/*') if f.is_file()) / (1024 * 1024)

        # Calculate correlation metrics
        spearman_corr = metrics_calculator.calculate_spearman_correlation(
            predicted_scores, reference_scores
        )
        pearson_corr = metrics_calculator.calculate_pearson_correlation(
            predicted_scores, reference_scores
        )

        # Calculate throughput metrics
        total_tokens = len(all_sentences) * max_length // 2
        qpm = len(sentence_pairs) / duration * 60

        logger.info(f"MLX testing completed in {duration:.2f}s")
        logger.info(f"Spearman correlation: {spearman_corr:.4f}")
        logger.info(f"Pearson correlation: {pearson_corr:.4f}")

        return {
            'spearman_correlation': spearman_corr,
            'pearson_correlation': pearson_corr,
            'accuracy': spearman_corr,  # Use correlation as "accuracy"
            'qpm': qpm,
            'size_mb': model_size_mb,
            'tokens_per_sec': total_tokens / duration,
            'memory_avg_mb': memory_tracker.get_average_memory_mb(),
            'memory_peak_mb': memory_tracker.get_peak_memory_mb(),
            'latency_p50_ms': (duration / len(sentence_pairs)) * 1000,
            'latency_p95_ms': 0.0,
            'latency_p99_ms': 0.0,
            'inference_time': duration,
            'sample_size': len(sentence_pairs),
            'predictions': predicted_scores[:100],
            'references': reference_scores[:100]
        }

    def get_quality_metrics(self) -> List[str]:
        """
        Get quality metrics for semantic similarity.

        Returns:
            List of metric names
        """
        return ['spearman_correlation', 'pearson_correlation', 'cosine_similarity']

    def validate_quality(
        self,
        pytorch_results: Dict[str, Any],
        mlx_results: Dict[str, Any]
    ) -> Dict[str, bool]:
        """
        Validate MLX model against PyTorch baseline.

        For embeddings, we check Spearman correlation preservation.

        Args:
            pytorch_results: PyTorch test results
            mlx_results: MLX test results

        Returns:
            Dictionary of quality gate results
        """
        # For embedding models, correlation should be > 0.98
        min_correlation = 0.98

        correlation_passed = mlx_results['spearman_correlation'] >= min_correlation

        # Speed gate
        speedup = mlx_results['qpm'] / pytorch_results['qpm'] if pytorch_results['qpm'] > 0 else 0.0
        speed_passed = speedup >= 1.2

        # Size gate
        quant_config = self.model_config.get('quantization', {})
        target_size_mb = quant_config.get('target_size_mb', 0)
        size_passed = mlx_results['size_mb'] <= target_size_mb * 1.1 if target_size_mb > 0 else True

        return {
            'correlation_passed': correlation_passed,
            'speed_passed': speed_passed,
            'size_passed': size_passed,
            'all_passed': correlation_passed and speed_passed and size_passed
        }
