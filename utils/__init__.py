"""
Utility modules for model testing and evaluation.

Heavy modules (psutil / torch / mlx) are imported lazily so lightweight
helpers such as ``utils.run_registry`` and ``utils.artifacts`` work on Linux
CI without the full Apple/ML stack (#10).
"""

from importlib import import_module

_LAZY = {
    'MetricsCalculator': '.metrics',
    'MemoryTracker': '.memory_tracker',
    'InferenceEngine': '.inference',
    'ModelComparator': '.comparison',
}

__all__ = list(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        return getattr(import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
