"""
AGI-Formula Benchmarking Suite

Comprehensive benchmarks for evaluating AGI capabilities including:
- Standard ML benchmarks adapted for AGI
- AGI-specific capability tests
- Performance profiling
- Comparative analysis
"""

from .agi_benchmarks import AGIBenchmarks
from .performance_profiler import PerformanceProfiler

__all__ = [
    'AGIBenchmarks',
    'PerformanceProfiler'
]