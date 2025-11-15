"""
Strategy selector for choosing the right conversion strategy based on task type.
"""

from typing import Dict, Any
import logging

from .base_strategy import BaseConversionStrategy
from .nli_strategy import NLIConversionStrategy
from .embedding_strategy import EmbeddingConversionStrategy
from .text_classification_strategy import TextClassificationStrategy
from .llm_strategy import LLMConversionStrategy
from .ner_strategy import NERConversionStrategy

logger = logging.getLogger('MLXConverter.StrategySelector')


class StrategySelector:
    """
    Selects the appropriate conversion strategy based on model task type.
    """

    # Map task types to strategy classes
    TASK_TO_STRATEGY = {
        # NLI / Zero-shot
        'zero-shot-classification': NLIConversionStrategy,
        'nli': NLIConversionStrategy,
        'natural-language-inference': NLIConversionStrategy,

        # Text Classification
        'text-classification': TextClassificationStrategy,
        'classification': TextClassificationStrategy,
        'sentiment-analysis': TextClassificationStrategy,
        'sentiment': TextClassificationStrategy,

        # Embeddings / Semantic Similarity
        'semantic-similarity': EmbeddingConversionStrategy,
        'sentence-similarity': EmbeddingConversionStrategy,
        'embedding': EmbeddingConversionStrategy,
        'sentence-embedding': EmbeddingConversionStrategy,
        'feature-extraction': EmbeddingConversionStrategy,

        # Named Entity Recognition
        'ner': NERConversionStrategy,
        'named-entity-recognition': NERConversionStrategy,
        'token-classification': NERConversionStrategy,

        # Large Language Models
        'llm': LLMConversionStrategy,
        'text-generation': LLMConversionStrategy,
        'causal-lm': LLMConversionStrategy,
        'causal-language-modeling': LLMConversionStrategy,
    }

    @classmethod
    def get_strategy(cls, model_config: Dict[str, Any]) -> BaseConversionStrategy:
        """
        Get the appropriate conversion strategy for a model.

        Args:
            model_config: Model configuration from models.yaml

        Returns:
            Instantiated strategy object

        Raises:
            ValueError: If task type not recognized
        """
        task = model_config.get('task', '').lower()

        if not task:
            raise ValueError(f"Model config missing 'task' field: {model_config.get('name')}")

        strategy_class = cls.TASK_TO_STRATEGY.get(task)

        if not strategy_class:
            # Try to guess from model architecture or name
            strategy_class = cls._guess_strategy(model_config)

            if not strategy_class:
                raise ValueError(
                    f"Unknown task type: '{task}'. "
                    f"Supported tasks: {list(cls.TASK_TO_STRATEGY.keys())}"
                )

        logger.info(f"Selected strategy: {strategy_class.__name__} for task '{task}'")

        return strategy_class(model_config)

    @classmethod
    def _guess_strategy(cls, model_config: Dict[str, Any]) -> type:
        """
        Attempt to guess strategy from model name or architecture.

        Args:
            model_config: Model configuration

        Returns:
            Strategy class or None
        """
        model_name = model_config.get('name', '').lower()
        hf_name = model_config.get('hf_name', '').lower()

        # Check for patterns in names
        if 'mnli' in model_name or 'mnli' in hf_name:
            logger.info(f"Guessing NLI strategy based on model name")
            return NLIConversionStrategy

        if 'ner' in model_name or 'ner' in hf_name:
            logger.info(f"Guessing NER strategy based on model name")
            return NERConversionStrategy

        if any(x in model_name for x in ['minilm', 'mpnet', 'sentence']):
            logger.info(f"Guessing Embedding strategy based on model name")
            return EmbeddingConversionStrategy

        if any(x in model_name for x in ['phi', 'llama', 'mistral', 'gemma', 'qwen']):
            logger.info(f"Guessing LLM strategy based on model name")
            return LLMConversionStrategy

        logger.warning(f"Could not guess strategy for model: {model_name}")
        return None

    @classmethod
    def list_supported_tasks(cls) -> Dict[str, str]:
        """
        Get list of supported tasks and their strategies.

        Returns:
            Dictionary mapping tasks to strategy names
        """
        return {
            task: strategy.__name__
            for task, strategy in cls.TASK_TO_STRATEGY.items()
        }

    @classmethod
    def register_strategy(cls, task: str, strategy_class: type):
        """
        Register a custom strategy for a task type.

        Useful for adding new strategies at runtime.

        Args:
            task: Task type name
            strategy_class: Strategy class to use
        """
        if not issubclass(strategy_class, BaseConversionStrategy):
            raise ValueError(f"{strategy_class} must inherit from BaseConversionStrategy")

        cls.TASK_TO_STRATEGY[task.lower()] = strategy_class
        logger.info(f"Registered custom strategy {strategy_class.__name__} for task '{task}'")
