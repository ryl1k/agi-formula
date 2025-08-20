"""
AGI-Formula Advanced GPU Computing Module

Phase 2 GPU optimization with custom CUDA kernels, multi-GPU support,
and advanced memory management for high-performance AGI computing.
"""

from .advanced_kernels import AdvancedGPUKernels
from .multi_gpu_manager import MultiGPUManager

__all__ = [
    'AdvancedGPUKernels',
    'MultiGPUManager'
]