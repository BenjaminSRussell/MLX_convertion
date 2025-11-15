"""
Memory tracking utilities for model testing.
"""

import psutil
from typing import List
import logging

logger = logging.getLogger('MLX8BitTester.MemoryTracker')


class MemoryTracker:
    """
    Tracks memory usage during model inference.
    """

    def __init__(self):
        """Initialize memory tracker with current process."""
        self.process = psutil.Process()
        self.samples: List[float] = []
        self.baseline_mb: float = 0.0

    def start(self):
        """
        Start tracking memory by recording baseline.
        """
        self.baseline_mb = self.get_current_memory_mb()
        self.samples = []
        logger.debug(f"Memory tracking started. Baseline: {self.baseline_mb:.1f}MB")

    def sample(self):
        """
        Take a memory measurement sample.
        """
        current_mb = self.get_current_memory_mb()
        self.samples.append(current_mb)

    def get_current_memory_mb(self) -> float:
        """
        Get current memory usage in MB.

        Returns:
            Memory usage in megabytes
        """
        return self.process.memory_info().rss / 1024 / 1024

    def get_samples(self) -> List[float]:
        """
        Get all memory samples.

        Returns:
            List of memory measurements in MB
        """
        return self.samples

    def get_peak_memory_mb(self) -> float:
        """
        Get peak memory usage.

        Returns:
            Peak memory in MB, or 0 if no samples
        """
        return max(self.samples) if self.samples else 0.0

    def get_average_memory_mb(self) -> float:
        """
        Get average memory usage.

        Returns:
            Average memory in MB, or 0 if no samples
        """
        return sum(self.samples) / len(self.samples) if self.samples else 0.0

    def get_memory_increase_mb(self) -> float:
        """
        Get memory increase from baseline.

        Returns:
            Memory increase in MB
        """
        current = self.get_current_memory_mb()
        return current - self.baseline_mb

    def reset(self):
        """
        Reset tracker and clear samples.
        """
        self.samples = []
        self.baseline_mb = 0.0

    def log_summary(self):
        """
        Log memory usage summary.
        """
        if self.samples:
            logger.info(f"Memory Usage Summary:")
            logger.info(f"  Baseline: {self.baseline_mb:.1f}MB")
            logger.info(f"  Peak: {self.get_peak_memory_mb():.1f}MB")
            logger.info(f"  Average: {self.get_average_memory_mb():.1f}MB")
            logger.info(f"  Increase from baseline: {self.get_memory_increase_mb():.1f}MB")
        else:
            logger.warning("No memory samples collected")
