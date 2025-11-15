"""
Conversion strategy for Natural Language Inference (NLI) models.

NLI models require strict accuracy preservation due to the critical
nature of entailment classification.
"""

from typing import Dict, Any, List
import numpy as np
import logging

from .base_strategy import BaseConversionStrategy

logger = logging.getLogger('MLXConverter.NLIStrategy')


class NLIConversionStrategy(BaseConversionStrategy):
    """
    Conversion strategy for NLI/zero-shot classification models.

    Key characteristics:
    - Strict accuracy requirements (< 1% drop)
    - Preserve classification head in FP32
    - Conservative quantization on attention layers
    - Symmetric INT8 quantization
    """

    def get_skip_patterns(self) -> List[str]:
        """
        Skip quantization for critical layers in NLI models.

        NLI models need the classification head to remain precise
        for accurate 3-way entailment decisions.

        Returns:
            Patterns to skip
        """
        return [
            'classifier',      # Final classification layer (critical!)
            'layernorm',       # Layer normalization
            'layer_norm',      # Alternate naming
            'bias',            # Bias terms
            'norm',            # General norm layers
            'ln',              # Shortened layer norm
            'embeddings.LayerNorm',  # BERT-style
            'pooler'           # Pooling layer (if present)
        ]

    def get_quantization_method(self) -> str:
        """
        NLI uses symmetric INT8 quantization.

        Returns:
            'symmetric'
        """
        return 'symmetric'

    def get_quality_thresholds(self) -> Dict[str, float]:
        """
        Strict quality thresholds for NLI models.

        NLI requires high accuracy for reliable entailment detection.

        Returns:
            Quality thresholds
        """
        return {
            'max_accuracy_drop': 0.01,  # 1% maximum drop (strict!)
            'min_f1': 0.85,             # F1 score minimum
            'min_precision': 0.83,      # Precision minimum
            'min_speedup': 1.2,         # 20% speed improvement required
            'max_size_ratio': 1.1       # Within 110% of target size
        }

    def should_quantize_weight(self, name: str, weight: np.ndarray) -> bool:
        """
        Determine if weight should be quantized for NLI models.

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
                logger.debug(f"Skipping quantization for {name} (matches pattern '{pattern}')")
                return False

        # Only quantize 2D weight matrices (linear layers)
        if len(weight.shape) != 2:
            logger.debug(f"Skipping quantization for {name} (not 2D, shape: {weight.shape})")
            return False

        # Skip very small matrices (< 100 params)
        if weight.size < 100:
            logger.debug(f"Skipping quantization for {name} (too small: {weight.size} params)")
            return False

        return True

    def quantize_weight(self, weight: np.ndarray, name: str) -> tuple:
        """
        Quantize weight using symmetric INT8 quantization.

        Symmetric quantization:
        - Simpler than asymmetric (no zero-point)
        - Better for weights centered around zero
        - Range: [-127, 127]

        Args:
            weight: Weight to quantize
            name: Weight name

        Returns:
            Tuple of (quantized_weight, scale, None)
        """
        # Calculate scale from maximum absolute value
        w_max = np.abs(weight).max()
        scale = w_max / 127.0 if w_max > 0 else 1.0

        # Quantize to INT8 range
        w_quant = np.round(weight / scale)
        w_quant = np.clip(w_quant, -127, 127).astype(np.int8)

        logger.debug(f"Quantized {name}: shape={weight.shape}, "
                    f"scale={scale:.6f}, range=[{w_quant.min()}, {w_quant.max()}]")

        return w_quant, scale, None  # No zero-point for symmetric

    def validate_conversion(
        self,
        original_size_mb: float,
        quantized_size_mb: float,
        metadata: Dict[str, Any]
    ) -> Dict[str, bool]:
        """
        Validate NLI conversion with strict requirements.

        Args:
            original_size_mb: Original model size
            quantized_size_mb: Quantized model size
            metadata: Conversion metadata

        Returns:
            Validation results
        """
        # Call base validation
        base_validation = super().validate_conversion(
            original_size_mb, quantized_size_mb, metadata
        )

        # NLI-specific validations
        thresholds = self.get_quality_thresholds()

        # Check target size
        target_size = self.quant_config.target_size_mb
        size_check = quantized_size_mb <= target_size * thresholds['max_size_ratio'] if target_size > 0 else True

        # Expected INT8 compression: ~3.5-4x (accounting for unquantized layers)
        compression_ratio = original_size_mb / quantized_size_mb if quantized_size_mb > 0 else 1.0
        compression_check = compression_ratio >= 3.0  # Minimum 3x for INT8

        validation_results = {
            **base_validation,
            'nli_size_check': size_check,
            'nli_compression_check': compression_check,
            'all_valid': base_validation['all_valid'] and size_check and compression_check
        }

        if not validation_results['all_valid']:
            logger.warning("NLI conversion validation failed:")
            if not size_check:
                logger.warning(f"  - Size exceeds target: {quantized_size_mb:.1f}MB > "
                             f"{target_size * thresholds['max_size_ratio']:.1f}MB")
            if not compression_check:
                logger.warning(f"  - Insufficient compression: {compression_ratio:.2f}x < 3.0x")

        return validation_results

    def get_metadata_extras(self) -> Dict[str, Any]:
        """
        Get NLI-specific metadata.

        Returns:
            Extra metadata
        """
        base_metadata = super().get_metadata_extras()
        return {
            **base_metadata,
            'nli_specifics': {
                'classification_head_preserved': True,
                'attention_quantized': True,
                'strict_accuracy_mode': True,
                'max_accuracy_drop_pct': self.get_quality_thresholds()['max_accuracy_drop'] * 100
            }
        }

    def log_conversion_summary(
        self,
        original_size_mb: float,
        quantized_size_mb: float,
        duration: float
    ):
        """
        Log NLI conversion summary.

        Args:
            original_size_mb: Original size
            quantized_size_mb: Quantized size
            duration: Conversion time
        """
        super().log_conversion_summary(original_size_mb, quantized_size_mb, duration)

        # Additional NLI-specific logging
        logger.info("NLI Strategy Details:")
        logger.info(f"  - Classification head: FP32 (preserved)")
        logger.info(f"  - Attention layers: INT8 (quantized)")
        logger.info(f"  - Max accuracy drop: {self.get_quality_thresholds()['max_accuracy_drop']*100}%")
        logger.info(f"  - Quality gates: STRICT")
