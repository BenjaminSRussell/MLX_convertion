"""
Conversion strategies for different model types.
Each strategy knows how to optimally convert its model type.
"""

from .base_strategy import BaseConversionStrategy, QuantizationConfig
from .nli_strategy import NLIConversionStrategy
from .embedding_strategy import EmbeddingConversionStrategy
from .text_classification_strategy import TextClassificationStrategy
from .llm_strategy import LLMConversionStrategy
from .ner_strategy import NERConversionStrategy

__all__ = [
    'BaseConversionStrategy',
    'QuantizationConfig',
    'NLIConversionStrategy',
    'EmbeddingConversionStrategy',
    'TextClassificationStrategy',
    'LLMConversionStrategy',
    'NERConversionStrategy',
]
