"""
base handler for model testing - defines what all handlers need to implement
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple
from pathlib import Path
import logging

logger = logging.getLogger('MLX8BitTester.BaseHandler')


class BaseHandler(ABC):
    """base class for task-specific handlers (NLI, classification, etc)"""

    def __init__(self, model_cfg: Dict[str, Any], dataset_cfg: Dict[str, Any]):
        """sets up handler with model and dataset configs"""
        self.model_cfg = model_cfg
        self.dataset_cfg = dataset_cfg
        self.model_name = model_cfg['name']
        self.task = model_cfg['task']

    @abstractmethod
    def prepare_data(self, dataset: Any) -> Tuple[List[str], List[Any]]:
        """prepares dataset for testing - returns (texts, labels)"""
        pass

    @abstractmethod
    def test_pytorch_baseline(
        self,
        model_name: str,
        dataset_name: str,
        dataset: Any,
        mem_tracker: Any,
        metrics_calc: Any
    ) -> Dict[str, Any]:
        """tests original PyTorch model as baseline"""
        pass

    @abstractmethod
    def test_mlx_model(
        self,
        model_path: Path,
        dataset_name: str,
        dataset: Any,
        mem_tracker: Any,
        metrics_calc: Any
    ) -> Dict[str, Any]:
        """tests quantized MLX model"""
        pass

    @abstractmethod
    def get_quality_metrics(self) -> List[str]:
        """returns list of metrics for this task (e.g. accuracy, f1)"""
        pass

    @abstractmethod
    def validate_quality(
        self,
        pt_results: Dict[str, Any],
        mlx_results: Dict[str, Any]
    ) -> Dict[str, bool]:
        """checks if MLX model meets quality gates"""
        pass

    def get_batch_size_pytorch(self) -> int:
        """batch size for PyTorch (default: 16)"""
        return 16

    def get_batch_size_mlx(self) -> int:
        """batch size for MLX (default: 32, handles bigger batches well)"""
        return 32

    def get_max_length(self) -> int:
        """max sequence length for tokenization (default: 512)"""
        return self.dataset_cfg.get('preprocessing', {}).get('max_length', 512)

    def should_truncate(self) -> bool:
        """whether to truncate long sequences"""
        return self.dataset_cfg.get('preprocessing', {}).get('truncation', True)

    def should_pad(self) -> bool:
        """whether to pad sequences"""
        return self.dataset_cfg.get('preprocessing', {}).get('padding', True)
