"""
Verification framework for ensuring model conversion quality.

This module provides comprehensive testing and verification tools:
- Layer-level weight accuracy verification
- Activation distribution verification
- Task-level accuracy verification
- Cross-platform parity checks
- Performance regression testing
"""

from .layer_verifier import LayerVerifier, LayerVerificationResult
from .task_verifier import TaskVerifier, TaskVerificationResult
from .parity_verifier import ParityVerifier, ParityVerificationResult
from .performance_verifier import PerformanceVerifier, PerformanceVerificationResult
from .quality_gate import QualityGateEnforcer, QualityGateResult

__all__ = [
    'LayerVerifier',
    'LayerVerificationResult',
    'TaskVerifier',
    'TaskVerificationResult',
    'ParityVerifier',
    'ParityVerificationResult',
    'PerformanceVerifier',
    'PerformanceVerificationResult',
    'QualityGateEnforcer',
    'QualityGateResult',
]
