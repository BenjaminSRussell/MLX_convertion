"""
Conversion strategy for Large Language Models (LLMs).

LLMs use grouped quantization for better compression while maintaining quality.
"""

from typing import Dict, Any, List
import numpy as np
import logging

from .base_strategy import BaseConversionStrategy

logger = logging.getLogger('MLXConverter.LLMStrategy')


class LLMConversionStrategy(BaseConversionStrategy):
    """
    Conversion strategy for Large Language Models.

    Examples: Phi-2, Llama-2, Mistral, Gemma

    Key characteristics:
    - Grouped INT4 quantization (64 elements per group)
    - Achieves 8x compression
    - Perplexity-based quality gates
    - Optimized for generation tasks
    - Uses mlx_lm built-in conversion tools
    """

    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        # LLMs typically use 4-bit by default
        if self.quant_config.bits not in [4, 8]:
            logger.warning(f"LLMs typically use 4 or 8-bit, got {self.quant_config.bits}")

        # Set default group size if not specified
        if self.quant_config.group_size is None:
            self.quant_config.group_size = 64

    def get_skip_patterns(self) -> List[str]:
        """
        Skip patterns for LLMs.

        LLMs can be more aggressively quantized, but we preserve
        embeddings and final layer norm.

        Returns:
            Patterns to skip
        """
        return [
            'embed_tokens',      # Input embeddings
            'lm_head',           # Output head (sometimes)
            'final_layernorm',   # Final layer norm
            'norm',              # Layer norms
            'ln'                 # Shortened layer norm
        ]

    def get_quantization_method(self) -> str:
        """
        LLMs use grouped quantization.

        This quantizes weights in groups (e.g., 64 elements per group)
        for better accuracy than full-tensor quantization.

        Returns:
            'grouped'
        """
        return 'grouped'

    def get_quality_thresholds(self) -> Dict[str, float]:
        """
        Quality thresholds for LLMs.

        LLMs are evaluated on perplexity and generation quality.

        Returns:
            Quality thresholds
        """
        return {
            'max_perplexity_increase': 0.05,  # 5% perplexity increase max
            'min_hellaswag_score': 0.70,      # HellaSwag accuracy minimum
            'min_speedup': 1.5,               # 50% speed improvement (bigger models)
            'max_size_ratio': 1.1             # Within 110% of target
        }

    def should_quantize_weight(self, name: str, weight: np.ndarray) -> bool:
        """
        Determine if weight should be quantized.

        For LLMs, we quantize most weights except embeddings and norms.

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

        # Quantize 2D matrices (linear layers)
        if len(weight.shape) == 2:
            return True

        # For LLMs, we might also quantize 3D tensors (attention)
        if len(weight.shape) == 3:
            logger.debug(f"Considering 3D tensor for quantization: {name}")
            return True

        return False

    def quantize_weight(self, weight: np.ndarray, name: str) -> tuple:
        """
        Quantize using grouped quantization.

        Groups of elements share a scale factor for better accuracy.

        Args:
            weight: Weight to quantize
            name: Weight name

        Returns:
            Tuple of (quantized_weight, scales, None)
        """
        group_size = self.quant_config.group_size
        bits = self.quant_config.bits

        # Flatten weight for group quantization
        original_shape = weight.shape
        weight_flat = weight.flatten()

        # Pad to multiple of group_size
        remainder = len(weight_flat) % group_size
        if remainder != 0:
            pad_size = group_size - remainder
            weight_flat = np.pad(weight_flat, (0, pad_size), mode='constant')

        # Reshape into groups
        num_groups = len(weight_flat) // group_size
        weight_grouped = weight_flat.reshape(num_groups, group_size)

        # Quantize each group
        quantized_groups = []
        scales = []

        quant_range = 7 if bits == 4 else 127  # Max value for signed int

        for group in weight_grouped:
            # Calculate scale for this group
            g_max = np.abs(group).max()
            scale = g_max / quant_range if g_max > 0 else 1.0
            scales.append(scale)

            # Quantize group
            g_quant = np.round(group / scale)
            g_quant = np.clip(g_quant, -quant_range, quant_range).astype(np.int8)
            quantized_groups.append(g_quant)

        # Combine groups and reshape
        quantized_flat = np.concatenate(quantized_groups)

        # Remove padding and reshape to original shape
        total_elements = np.prod(original_shape)
        quantized_flat = quantized_flat[:total_elements]
        quantized = quantized_flat.reshape(original_shape)

        scales_array = np.array(scales, dtype=np.float32)

        logger.debug(f"Group quantized {name}: {num_groups} groups of {group_size}, "
                    f"{bits}-bit, scales shape={scales_array.shape}")

        return quantized, scales_array, None

    def validate_conversion(
        self,
        original_size_mb: float,
        quantized_size_mb: float,
        metadata: Dict[str, Any]
    ) -> Dict[str, bool]:
        """
        Validate LLM conversion.

        LLMs should achieve high compression ratios.

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

        # Check compression
        compression_ratio = original_size_mb / quantized_size_mb if quantized_size_mb > 0 else 1.0

        # For INT4, expect ~7-8x compression
        # For INT8, expect ~3.5-4x compression
        expected_compression = 7.0 if self.quant_config.bits == 4 else 3.5
        compression_check = compression_ratio >= expected_compression * 0.85

        validation_results = {
            **base_validation,
            'llm_compression_check': compression_check,
            'all_valid': base_validation['all_valid'] and compression_check
        }

        if not compression_check:
            logger.warning(f"LLM compression below expected: "
                          f"{compression_ratio:.2f}x < {expected_compression:.2f}x")

        return validation_results

    def get_metadata_extras(self) -> Dict[str, Any]:
        """
        Get LLM-specific metadata.

        Returns:
            Extra metadata
        """
        base_metadata = super().get_metadata_extras()
        return {
            **base_metadata,
            'llm_specifics': {
                'group_size': self.quant_config.group_size,
                'quantization_bits': self.quant_config.bits,
                'method': 'grouped',
                'max_perplexity_increase': self.get_quality_thresholds()['max_perplexity_increase'],
                'uses_mlx_lm_converter': True  # LLMs use mlx_lm built-in tools
            }
        }

    def log_conversion_summary(
        self,
        original_size_mb: float,
        quantized_size_mb: float,
        duration: float
    ):
        """
        Log LLM conversion summary.

        Args:
            original_size_mb: Original size
            quantized_size_mb: Quantized size
            duration: Conversion time
        """
        super().log_conversion_summary(original_size_mb, quantized_size_mb, duration)

        logger.info("LLM Strategy Details:")
        logger.info(f"  - Quantization: {self.quant_config.bits}-bit grouped")
        logger.info(f"  - Group size: {self.quant_config.group_size}")
        logger.info(f"  - Max perplexity increase: {self.get_quality_thresholds()['max_perplexity_increase']*100}%")
        logger.info(f"  - Quality gates: PERPLEXITY-BASED")
        logger.info(f"  - Conversion method: mlx_lm.convert")
