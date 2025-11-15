"""
base conversion strategy - all model-specific strategies inherit this
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import logging

logger = logging.getLogger('MLXConverter.BaseStrategy')


@dataclass
class QuantizationConfig:
    """quantization settings"""
    bits: int = 8
    method: str = 'symmetric'  # symmetric, asymmetric, per_channel
    group_size: Optional[int] = None  # For grouped quantization (LLMs)
    calibration_samples: int = 0  # Number of calibration samples (embeddings)
    preserve_norm: bool = False  # Preserve layer normalization (embeddings)
    skip_patterns: List[str] = None  # Patterns to skip quantization
    target_size_mb: float = 0.0  # Target model size
    max_accuracy_drop: float = 0.015  # Maximum acceptable accuracy drop (1.5%)

    def __post_init__(self):
        if self.skip_patterns is None:
            self.skip_patterns = ['layernorm', 'bias', 'norm', 'ln']


class BaseConversionStrategy(ABC):
    """base for model-specific conversion strategies (NLI, embeddings, LLMs, etc)"""

    def __init__(self, model_config: Dict[str, Any]):
        """sets up conversion strategy from config"""
        self.model_config = model_config
        self.model_name = model_config['name']
        self.hf_name = model_config['hf_name']
        self.task = model_config['task']

        # Create quantization config from model config
        quant_dict = model_config.get('quantization', {})
        self.quant_config = QuantizationConfig(
            bits=quant_dict.get('bits', 8),
            method=quant_dict.get('method', 'symmetric'),
            group_size=quant_dict.get('group_size'),
            calibration_samples=quant_dict.get('calibration_samples', 0),
            preserve_norm=quant_dict.get('preserve_norm', False),
            skip_patterns=quant_dict.get('skip_patterns'),
            target_size_mb=quant_dict.get('target_size_mb', 0.0),
            max_accuracy_drop=quant_dict.get('max_accuracy_drop', 0.015)
        )

    @abstractmethod
    def get_skip_patterns(self) -> List[str]:
        """
        Get patterns of layer names to skip during quantization.

        Different model types need different layers preserved.

        Returns:
            List of patterns to skip (e.g., ['classifier', 'layernorm'])
        """
        pass

    @abstractmethod
    def get_quantization_method(self) -> str:
        """
        Get quantization method for this model type.

        Returns:
            Method name: 'symmetric', 'asymmetric', 'per_channel', 'grouped'
        """
        pass

    @abstractmethod
    def get_quality_thresholds(self) -> Dict[str, float]:
        """
        Get quality thresholds for this model type.

        Returns:
            Dictionary with metric names and threshold values
        """
        pass

    @abstractmethod
    def should_quantize_weight(self, name: str, weight: np.ndarray) -> bool:
        """
        Determine if a specific weight should be quantized.

        Args:
            name: Weight tensor name
            weight: Weight array

        Returns:
            True if should quantize, False to keep as FP32
        """
        pass

    @abstractmethod
    def quantize_weight(self, weight: np.ndarray, name: str) -> tuple:
        """
        Quantize a weight tensor using strategy-specific method.

        Args:
            weight: Weight array to quantize
            name: Weight tensor name

        Returns:
            Tuple of (quantized_weight, scale, zero_point)
        """
        pass

    def validate_conversion(
        self,
        original_size_mb: float,
        quantized_size_mb: float,
        metadata: Dict[str, Any]
    ) -> Dict[str, bool]:
        """
        Validate that conversion meets quality standards.

        Args:
            original_size_mb: Original model size
            quantized_size_mb: Quantized model size
            metadata: Conversion metadata

        Returns:
            Dictionary of validation results
        """
        # Size validation
        target_size = self.quant_config.target_size_mb
        size_valid = quantized_size_mb <= target_size * 1.1 if target_size > 0 else True

        # Compression ratio validation
        compression_ratio = original_size_mb / quantized_size_mb if quantized_size_mb > 0 else 1.0

        # Expected compression for different bit widths
        expected_compression = {
            4: 7.0,  # INT4: ~8x (with overhead)
            8: 3.5,  # INT8: ~4x (with overhead)
            16: 1.8  # FP16: ~2x
        }

        min_compression = expected_compression.get(self.quant_config.bits, 1.0)
        compression_valid = compression_ratio >= min_compression * 0.9  # Allow 10% variance

        return {
            'size_valid': size_valid,
            'compression_valid': compression_valid,
            'all_valid': size_valid and compression_valid,
            'compression_ratio': compression_ratio,
            'target_compression': min_compression
        }

    def get_calibration_dataset(self) -> Optional[Any]:
        """
        Get calibration dataset for quantization (if needed).

        Override this for strategies that need calibration (e.g., embeddings).

        Returns:
            Dataset or None
        """
        return None

    def preprocess_weights(self, weights: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess weights before quantization.

        Override for strategy-specific preprocessing.

        Args:
            weights: Original model weights

        Returns:
            Preprocessed weights
        """
        return weights

    def postprocess_weights(self, weights: Dict[str, Any]) -> Dict[str, Any]:
        """
        Postprocess weights after quantization.

        Override for strategy-specific postprocessing.

        Args:
            weights: Quantized weights

        Returns:
            Postprocessed weights
        """
        return weights

    def get_metadata_extras(self) -> Dict[str, Any]:
        """
        Get strategy-specific metadata to save with converted model.

        Returns:
            Dictionary of extra metadata
        """
        return {
            'strategy': self.__class__.__name__,
            'task': self.task,
            'quantization_method': self.get_quantization_method(),
            'skip_patterns': self.get_skip_patterns(),
            'quality_thresholds': self.get_quality_thresholds()
        }

    def log_conversion_summary(
        self,
        original_size_mb: float,
        quantized_size_mb: float,
        duration: float
    ):
        """
        Log conversion summary with strategy-specific details.

        Args:
            original_size_mb: Original size
            quantized_size_mb: Quantized size
            duration: Conversion time
        """
        compression_ratio = original_size_mb / quantized_size_mb if quantized_size_mb > 0 else 1.0

        logger.info(f"\n{'='*60}")
        logger.info(f"Conversion Summary ({self.__class__.__name__})")
        logger.info(f"{'='*60}")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Task: {self.task}")
        logger.info(f"Quantization: {self.quant_config.bits}-bit {self.quant_config.method}")
        logger.info(f"Original size: {original_size_mb:.1f}MB")
        logger.info(f"Quantized size: {quantized_size_mb:.1f}MB")
        logger.info(f"Compression ratio: {compression_ratio:.2f}x")
        logger.info(f"Target size: {self.quant_config.target_size_mb:.1f}MB")
        logger.info(f"Conversion time: {duration:.2f}s")
        logger.info(f"{'='*60}\n")
