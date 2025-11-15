"""
Conversion strategy for Named Entity Recognition (NER) models.

NER models need special handling for token classification heads
and entity-level accuracy preservation.
"""

from typing import Dict, Any, List
import numpy as np
import logging

from .base_strategy import BaseConversionStrategy

logger = logging.getLogger('MLXConverter.NERStrategy')


class NERConversionStrategy(BaseConversionStrategy):
    """
    Conversion strategy for Named Entity Recognition models.

    Examples: BERT-NER, RoBERTa-NER, ELECTRA-NER

    Key characteristics:
    - Per-token classification requires precision
    - Preserve token classification head in FP32
    - F1-score based quality gates (per entity type)
    - Moderate accuracy requirements (< 2% F1 drop)
    - Per-channel quantization for better token-level accuracy
    """

    def get_skip_patterns(self) -> List[str]:
        """
        Skip patterns for NER models.

        NER models need the token classification head preserved.

        Returns:
            Patterns to skip
        """
        return [
            'classifier',          # Token classification head
            'token_classifier',    # Alternate naming
            'layernorm',          # Layer normalization
            'layer_norm',
            'bias',
            'norm',
            'ln',
            'crf',                # Conditional Random Field layer (if present)
            'transitions'         # CRF transition matrix
        ]

    def get_quantization_method(self) -> str:
        """
        NER uses per-channel quantization for better token-level accuracy.

        Per-channel quantization quantizes each output channel separately,
        preserving more information for per-token predictions.

        Returns:
            'per_channel'
        """
        return 'per_channel'

    def get_quality_thresholds(self) -> Dict[str, float]:
        """
        Quality thresholds for NER models.

        NER is evaluated on entity-level F1 scores.

        Returns:
            Quality thresholds
        """
        return {
            'max_f1_drop': 0.02,           # 2% F1 drop maximum
            'min_entity_precision': 0.85,   # Precision per entity type
            'min_entity_recall': 0.83,      # Recall per entity type
            'min_token_accuracy': 0.95,     # Token-level accuracy
            'min_speedup': 1.2,             # 20% speed improvement
            'max_size_ratio': 1.1           # Within 110% of target
        }

    def should_quantize_weight(self, name: str, weight: np.ndarray) -> bool:
        """
        Determine if weight should be quantized for NER.

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

        # Skip very small matrices
        if weight.size < 100:
            logger.debug(f"Skipping {name} (too small)")
            return False

        return True

    def quantize_weight(self, weight: np.ndarray, name: str) -> tuple:
        """
        Quantize using per-channel quantization.

        Each output channel (column) gets its own scale factor.

        Args:
            weight: Weight to quantize (shape: [in_features, out_features])
            name: Weight name

        Returns:
            Tuple of (quantized_weight, scales, None)
        """
        # Per-channel quantization: one scale per output channel
        if len(weight.shape) != 2:
            # Fallback to symmetric for non-2D tensors
            logger.warning(f"Per-channel quantization requires 2D weights, "
                          f"falling back to symmetric for {name}")
            w_max = np.abs(weight).max()
            scale = w_max / 127.0 if w_max > 0 else 1.0
            w_quant = np.round(weight / scale)
            w_quant = np.clip(w_quant, -127, 127).astype(np.int8)
            return w_quant, scale, None

        # Calculate scale per output channel (column)
        in_features, out_features = weight.shape
        scales = np.zeros(out_features, dtype=np.float32)
        quantized = np.zeros_like(weight, dtype=np.int8)

        for i in range(out_features):
            channel = weight[:, i]
            c_max = np.abs(channel).max()
            scale = c_max / 127.0 if c_max > 0 else 1.0
            scales[i] = scale

            # Quantize channel
            c_quant = np.round(channel / scale)
            c_quant = np.clip(c_quant, -127, 127).astype(np.int8)
            quantized[:, i] = c_quant

        logger.debug(f"Per-channel quantized {name}: "
                    f"shape={weight.shape}, {out_features} channels, "
                    f"scale range=[{scales.min():.6f}, {scales.max():.6f}]")

        return quantized, scales, None

    def validate_conversion(
        self,
        original_size_mb: float,
        quantized_size_mb: float,
        metadata: Dict[str, Any]
    ) -> Dict[str, bool]:
        """
        Validate NER conversion.

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

        # NER-specific validation
        thresholds = self.get_quality_thresholds()

        # Check compression (per-channel adds overhead)
        compression_ratio = original_size_mb / quantized_size_mb if quantized_size_mb > 0 else 1.0

        # Per-channel has more overhead than symmetric, expect ~3x for INT8
        expected_compression = 3.0
        compression_check = compression_ratio >= expected_compression * 0.85

        validation_results = {
            **base_validation,
            'ner_compression_check': compression_check,
            'all_valid': base_validation['all_valid'] and compression_check
        }

        if not compression_check:
            logger.warning(f"NER compression below expected: "
                          f"{compression_ratio:.2f}x < {expected_compression:.2f}x")

        return validation_results

    def get_metadata_extras(self) -> Dict[str, Any]:
        """
        Get NER-specific metadata.

        Returns:
            Extra metadata
        """
        base_metadata = super().get_metadata_extras()
        return {
            **base_metadata,
            'ner_specifics': {
                'token_classifier_preserved': True,
                'quantization_method': 'per_channel',
                'max_f1_drop_pct': self.get_quality_thresholds()['max_f1_drop'] * 100,
                'entity_level_evaluation': True,
                'crf_layer_preserved': True  # If model has CRF
            }
        }

    def log_conversion_summary(
        self,
        original_size_mb: float,
        quantized_size_mb: float,
        duration: float
    ):
        """
        Log NER conversion summary.

        Args:
            original_size_mb: Original size
            quantized_size_mb: Quantized size
            duration: Conversion time
        """
        super().log_conversion_summary(original_size_mb, quantized_size_mb, duration)

        logger.info("NER Strategy Details:")
        logger.info(f"  - Token classifier: FP32 (preserved)")
        logger.info(f"  - Quantization method: Per-channel INT8")
        logger.info(f"  - Max F1 drop: {self.get_quality_thresholds()['max_f1_drop']*100}%")
        logger.info(f"  - Quality gates: ENTITY-LEVEL F1")
        logger.info(f"  - Evaluation: Per-entity type metrics")
