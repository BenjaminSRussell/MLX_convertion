"""
Conversion strategy for standard text classification models.

This is a balanced strategy between NLI (very strict) and embeddings (aggressive).
"""

from typing import Dict, Any, List
import numpy as np
import logging

from .base_strategy import BaseConversionStrategy

logger = logging.getLogger('MLXConverter.TextClassificationStrategy')


class TextClassificationStrategy(BaseConversionStrategy):
    """
    Conversion strategy for standard text classification tasks.

    Examples: Sentiment analysis, topic classification, spam detection

    Key characteristics:
    - Moderate accuracy requirements (< 1.5% drop)
    - Preserve classification head in FP32
    - Symmetric INT8 quantization
    - Balance between speed and accuracy
    """

    def get_skip_patterns(self) -> List[str]:
        """
        Skip patterns for text classification.

        Similar to NLI but slightly less conservative.

        Returns:
            Patterns to skip
        """
        return [
            'classifier',
            'layernorm',
            'layer_norm',
            'bias',
            'norm',
            'ln'
        ]

    def get_quantization_method(self) -> str:
        """
        Text classification uses symmetric INT8.

        Returns:
            'symmetric'
        """
        return 'symmetric'

    def get_quality_thresholds(self) -> Dict[str, float]:
        """
        Quality thresholds for text classification.

        Slightly more relaxed than NLI.

        Returns:
            Quality thresholds
        """
        return {
            'max_accuracy_drop': 0.015,  # 1.5% maximum drop
            'min_f1': 0.83,              # F1 minimum
            'min_precision': 0.80,       # Precision minimum
            'min_speedup': 1.2,          # 20% speed improvement
            'max_size_ratio': 1.1        # Within 110% of target
        }

    def should_quantize_weight(self, name: str, weight: np.ndarray) -> bool:
        """
        Determine if weight should be quantized.

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
                return False

        # Only quantize 2D matrices
        if len(weight.shape) != 2:
            return False

        # Skip very small matrices
        if weight.size < 100:
            return False

        return True

    def quantize_weight(self, weight: np.ndarray, name: str) -> tuple:
        """
        Quantize using symmetric INT8.

        Args:
            weight: Weight to quantize
            name: Weight name

        Returns:
            Tuple of (quantized_weight, scale, None)
        """
        w_max = np.abs(weight).max()
        scale = w_max / 127.0 if w_max > 0 else 1.0

        w_quant = np.round(weight / scale)
        w_quant = np.clip(w_quant, -127, 127).astype(np.int8)

        return w_quant, scale, None

    def get_metadata_extras(self) -> Dict[str, Any]:
        """
        Get text classification specific metadata.

        Returns:
            Extra metadata
        """
        base_metadata = super().get_metadata_extras()
        return {
            **base_metadata,
            'text_classification_specifics': {
                'classification_head_preserved': True,
                'max_accuracy_drop_pct': self.get_quality_thresholds()['max_accuracy_drop'] * 100
            }
        }
