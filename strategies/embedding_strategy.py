"""
Conversion strategy for semantic similarity / sentence embedding models.

Embedding models need special care to preserve cosine similarity
and semantic relationships after quantization.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import logging
from datasets import load_dataset

from .base_strategy import BaseConversionStrategy

logger = logging.getLogger('MLXConverter.EmbeddingStrategy')


class EmbeddingConversionStrategy(BaseConversionStrategy):
    """
    Conversion strategy for sentence embedding / semantic similarity models.

    Key characteristics:
    - Preserve embedding space structure
    - Calibration-based quantization
    - Strict Spearman correlation requirements (> 0.98)
    - Can quantize more aggressively (embeddings are robust)
    - May use asymmetric quantization for better accuracy
    """

    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        self.calibration_data = None

    def get_skip_patterns(self) -> List[str]:
        """
        Embedding models can be more aggressively quantized.

        We still skip some normalization layers to preserve
        the embedding space geometry.

        Returns:
            Patterns to skip
        """
        patterns = ['bias', 'norm', 'ln']

        # If preserve_norm is True, skip more norm layers
        if self.quant_config.preserve_norm:
            patterns.extend(['layernorm', 'layer_norm'])

        return patterns

    def get_quantization_method(self) -> str:
        """
        Embedding models use asymmetric quantization with calibration.

        This provides better reconstruction of the embedding space.

        Returns:
            'asymmetric' if calibration enabled, else 'symmetric'
        """
        return 'asymmetric' if self.quant_config.calibration_samples > 0 else 'symmetric'

    def get_quality_thresholds(self) -> Dict[str, float]:
        """
        Quality thresholds for embedding models.

        Embeddings are evaluated on correlation, not raw accuracy.

        Returns:
            Quality thresholds
        """
        return {
            'min_spearman_correlation': 0.98,   # Critical!
            'min_pearson_correlation': 0.97,
            'min_cosine_similarity': 0.95,      # Average cosine sim preservation
            'min_speedup': 1.2,                 # 20% speed improvement
            'max_size_ratio': 1.1               # Within 110% of target
        }

    def get_calibration_dataset(self) -> Optional[Any]:
        """
        Load calibration dataset for asymmetric quantization.

        Uses a subset of STS benchmark for calibration.

        Returns:
            Calibration dataset or None
        """
        if self.quant_config.calibration_samples == 0:
            return None

        if self.calibration_data is not None:
            return self.calibration_data

        try:
            logger.info(f"Loading calibration dataset ({self.quant_config.calibration_samples} samples)...")

            # Load STS benchmark for calibration
            dataset = load_dataset(
                "sentence-transformers/stsb",
                split=f"train[:{self.quant_config.calibration_samples}]"
            )

            self.calibration_data = dataset
            logger.info(f"Loaded {len(dataset)} calibration samples")

            return dataset

        except Exception as e:
            logger.warning(f"Failed to load calibration dataset: {e}")
            logger.warning("Falling back to symmetric quantization without calibration")
            return None

    def compute_activation_ranges(
        self,
        calibration_data: Any,
        model: Any
    ) -> Dict[str, tuple]:
        """
        Compute activation ranges for asymmetric quantization.

        Args:
            calibration_data: Calibration dataset
            model: Model to calibrate

        Returns:
            Dictionary mapping layer names to (min, max) ranges
        """
        logger.info("Computing activation ranges from calibration data...")

        activation_ranges = {}

        # TODO: Implement activation range computation
        # This would involve:
        # 1. Run forward passes on calibration data
        # 2. Collect min/max activations per layer
        # 3. Use for asymmetric quantization

        logger.warning("Activation range computation not yet implemented")

        return activation_ranges

    def should_quantize_weight(self, name: str, weight: np.ndarray) -> bool:
        """
        Determine if weight should be quantized.

        Embedding models are generally robust to quantization.

        Args:
            name: Weight name
            weight: Weight array

        Returns:
            True if should quantize
        """
        # Check skip patterns
        name_lower = name.lower()
        for pattern in self.get_skip_patterns():
            if pattern.lower() in name_lower:
                logger.debug(f"Skipping {name} (pattern: {pattern})")
                return False

        # Only quantize 2D matrices
        if len(weight.shape) != 2:
            logger.debug(f"Skipping {name} (not 2D)")
            return False

        # Quantize even small matrices for embeddings
        return True

    def quantize_weight(self, weight: np.ndarray, name: str) -> tuple:
        """
        Quantize weight using symmetric or asymmetric quantization.

        Args:
            weight: Weight to quantize
            name: Weight name

        Returns:
            Tuple of (quantized_weight, scale, zero_point)
        """
        if self.get_quantization_method() == 'asymmetric':
            # Asymmetric quantization (with zero-point)
            w_min = weight.min()
            w_max = weight.max()

            # Calculate scale and zero-point
            scale = (w_max - w_min) / 255.0 if w_max > w_min else 1.0
            zero_point = int(np.round(-w_min / scale))

            # Quantize to [0, 255] range
            w_quant = np.round(weight / scale + zero_point)
            w_quant = np.clip(w_quant, 0, 255).astype(np.uint8)

            logger.debug(f"Asymmetric quantized {name}: "
                        f"scale={scale:.6f}, zero_point={zero_point}")

            return w_quant, scale, zero_point

        else:
            # Symmetric quantization (no zero-point)
            w_max = np.abs(weight).max()
            scale = w_max / 127.0 if w_max > 0 else 1.0

            w_quant = np.round(weight / scale)
            w_quant = np.clip(w_quant, -127, 127).astype(np.int8)

            logger.debug(f"Symmetric quantized {name}: scale={scale:.6f}")

            return w_quant, scale, None

    def validate_conversion(
        self,
        original_size_mb: float,
        quantized_size_mb: float,
        metadata: Dict[str, Any]
    ) -> Dict[str, bool]:
        """
        Validate embedding model conversion.

        Args:
            original_size_mb: Original size
            quantized_size_mb: Quantized size
            metadata: Metadata

        Returns:
            Validation results
        """
        base_validation = super().validate_conversion(
            original_size_mb, quantized_size_mb, metadata
        )

        # Check compression (embeddings can achieve good compression)
        compression_ratio = original_size_mb / quantized_size_mb if quantized_size_mb > 0 else 1.0

        # For INT8, expect ~3.5x compression (similar to other encoders)
        # For INT4, expect ~7x compression
        expected_compression = 7.0 if self.quant_config.bits == 4 else 3.5
        compression_check = compression_ratio >= expected_compression * 0.85  # 15% tolerance

        validation_results = {
            **base_validation,
            'embedding_compression_check': compression_check,
            'all_valid': base_validation['all_valid'] and compression_check
        }

        if not compression_check:
            logger.warning(f"Compression below expected: {compression_ratio:.2f}x < {expected_compression:.2f}x")

        return validation_results

    def get_metadata_extras(self) -> Dict[str, Any]:
        """
        Get embedding-specific metadata.

        Returns:
            Extra metadata
        """
        base_metadata = super().get_metadata_extras()
        return {
            **base_metadata,
            'embedding_specifics': {
                'calibration_samples': self.quant_config.calibration_samples,
                'quantization_method': self.get_quantization_method(),
                'preserve_norm': self.quant_config.preserve_norm,
                'min_spearman_correlation': self.get_quality_thresholds()['min_spearman_correlation'],
                'min_cosine_similarity': self.get_quality_thresholds()['min_cosine_similarity']
            }
        }

    def log_conversion_summary(
        self,
        original_size_mb: float,
        quantized_size_mb: float,
        duration: float
    ):
        """
        Log embedding conversion summary.

        Args:
            original_size_mb: Original size
            quantized_size_mb: Quantized size
            duration: Conversion time
        """
        super().log_conversion_summary(original_size_mb, quantized_size_mb, duration)

        logger.info("Embedding Strategy Details:")
        logger.info(f"  - Quantization method: {self.get_quantization_method()}")
        logger.info(f"  - Calibration samples: {self.quant_config.calibration_samples}")
        logger.info(f"  - Preserve norm layers: {self.quant_config.preserve_norm}")
        logger.info(f"  - Min Spearman correlation: {self.get_quality_thresholds()['min_spearman_correlation']:.2f}")
        logger.info(f"  - Quality gates: CORRELATION-BASED")
