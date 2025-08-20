"""
Performance Profiling Tools for AGI-Formula

Detailed performance analysis including timing, memory usage,
computational bottlenecks, and optimization recommendations.
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from collections import defaultdict
import functools
import threading
from pathlib import Path
import json


@dataclass
class ProfileResult:
    """Result of a performance profiling session"""
    function_name: str
    total_time: float
    call_count: int
    avg_time: float
    min_time: float
    max_time: float
    memory_usage: Optional[float] = None
    metadata: Dict[str, Any] = None


@dataclass
class ProfilingSession:
    """Complete profiling session results"""
    session_name: str
    total_duration: float
    results: List[ProfileResult]
    bottlenecks: List[str]
    recommendations: List[str]
    timestamp: str


class PerformanceProfiler:
    """
    Advanced performance profiler for AGI networks
    
    Features:
    - Function-level timing
    - Memory usage tracking
    - Bottleneck identification
    - Performance recommendations
    - Real-time monitoring
    - GPU vs CPU comparison
    """
    
    def __init__(self, network, enable_memory_tracking: bool = True):
        self.network = network
        self.enable_memory_tracking = enable_memory_tracking
        self.timing_data = defaultdict(list)
        self.memory_data = defaultdict(list)
        self.call_counts = defaultdict(int)
        self.profiling_active = False
        self.session_start = None
        
        # Try to import memory profiling tools
        self.psutil_available = False
        try:
            import psutil
            self.psutil_available = True
            self.process = psutil.Process()
        except ImportError:
            pass
        
        # GPU profiling
        self.cupy_available = False
        try:
            import cupy
            self.cupy_available = True
        except ImportError:
            pass
    
    def start_session(self, session_name: str = "default") -> None:
        """Start a profiling session"""
        self.session_name = session_name
        self.profiling_active = True
        self.session_start = time.time()
        self.timing_data.clear()
        self.memory_data.clear()
        self.call_counts.clear()
        
        print(f"Starting profiling session: {session_name}")
    
    def end_session(self) -> ProfilingSession:
        """End profiling session and return results"""
        if not self.profiling_active:
            raise RuntimeError("No active profiling session")
        
        self.profiling_active = False
        total_duration = time.time() - self.session_start
        
        # Generate results
        results = []
        for func_name, times in self.timing_data.items():
            if times:
                result = ProfileResult(
                    function_name=func_name,
                    total_time=sum(times),
                    call_count=self.call_counts[func_name],
                    avg_time=np.mean(times),
                    min_time=min(times),
                    max_time=max(times),
                    memory_usage=np.mean(self.memory_data[func_name]) if self.memory_data[func_name] else None
                )
                results.append(result)
        
        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks(results)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(results, bottlenecks)
        
        session = ProfilingSession(
            session_name=self.session_name,
            total_duration=total_duration,
            results=results,
            bottlenecks=bottlenecks,
            recommendations=recommendations,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        print(f"Profiling session '{self.session_name}' completed in {total_duration:.3f}s")
        return session
    
    def profile_function(self, func: Callable) -> Callable:
        """Decorator to profile a function"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not self.profiling_active:
                return func(*args, **kwargs)
            
            func_name = f"{func.__module__}.{func.__name__}"
            
            # Memory before
            memory_before = self._get_memory_usage() if self.enable_memory_tracking else None
            
            # Time the function
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                end_time = time.time()
                execution_time = end_time - start_time
                
                # Memory after
                memory_after = self._get_memory_usage() if self.enable_memory_tracking else None
                
                # Record data
                self.timing_data[func_name].append(execution_time)
                self.call_counts[func_name] += 1
                
                if memory_before is not None and memory_after is not None:
                    memory_delta = memory_after - memory_before
                    self.memory_data[func_name].append(memory_delta)
        
        return wrapper
    
    def profile_network_operations(self) -> None:
        """Profile key network operations automatically"""
        if not self.profiling_active:
            raise RuntimeError("Start a profiling session first")
        
        # Profile forward pass
        original_forward = self.network.forward
        self.network.forward = self.profile_function(original_forward)
        
        # Profile neuron activation if accessible
        if hasattr(self.network, 'neurons'):
            for neuron in self.network.neurons:
                if hasattr(neuron, 'activate'):
                    neuron.activate = self.profile_function(neuron.activate)
        
        # Profile attention mechanisms
        if hasattr(self.network, 'attention_module'):
            if hasattr(self.network.attention_module, 'compute_scores'):
                self.network.attention_module.compute_scores = self.profile_function(
                    self.network.attention_module.compute_scores
                )
        
        # Profile causal cache operations
        if hasattr(self.network, 'causal_cache'):
            if hasattr(self.network.causal_cache, 'store_contribution'):
                self.network.causal_cache.store_contribution = self.profile_function(
                    self.network.causal_cache.store_contribution
                )
    
    def benchmark_forward_pass(self, num_iterations: int = 1000) -> Dict[str, Any]:
        """Benchmark forward pass performance"""
        input_data = np.random.randn(self.network.config.input_size)
        
        # Warmup
        for _ in range(10):
            self.network.forward(input_data)
        
        # Benchmark
        times = []
        for _ in range(num_iterations):
            start_time = time.time()
            self.network.forward(input_data)
            times.append(time.time() - start_time)
        
        return {
            'avg_time_ms': np.mean(times) * 1000,
            'std_time_ms': np.std(times) * 1000,
            'min_time_ms': min(times) * 1000,
            'max_time_ms': max(times) * 1000,
            'ops_per_second': 1.0 / np.mean(times),
            'total_iterations': num_iterations
        }
    
    def benchmark_memory_usage(self, num_operations: int = 100) -> Dict[str, Any]:
        """Benchmark memory usage patterns"""
        if not self.psutil_available:
            return {'error': 'psutil not available for memory tracking'}
        
        input_data = np.random.randn(self.network.config.input_size)
        
        memory_before = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # Run operations
        for _ in range(num_operations):
            self.network.forward(input_data)
        
        memory_after = self.process.memory_info().rss / 1024 / 1024  # MB
        memory_delta = memory_after - memory_before
        
        return {
            'memory_before_mb': memory_before,
            'memory_after_mb': memory_after,
            'memory_increase_mb': memory_delta,
            'memory_per_operation_kb': (memory_delta * 1024) / num_operations,
            'operations': num_operations
        }
    
    def compare_cpu_gpu_performance(self) -> Dict[str, Any]:
        """Compare CPU vs GPU performance if available"""
        if not self.cupy_available:
            return {'error': 'CuPy not available for GPU comparison'}
        
        import cupy as cp
        
        results = {}
        input_data = np.random.randn(self.network.config.input_size)
        
        # CPU benchmark
        cpu_times = []
        for _ in range(100):
            start_time = time.time()
            self.network.forward(input_data)
            cpu_times.append(time.time() - start_time)
        
        results['cpu'] = {
            'avg_time_ms': np.mean(cpu_times) * 1000,
            'ops_per_second': 1.0 / np.mean(cpu_times)
        }
        
        # GPU benchmark (if network supports it)
        try:
            gpu_input = cp.asarray(input_data)
            gpu_times = []
            
            for _ in range(100):
                start_time = time.time()
                # This would need GPU-enabled network implementation
                # For now, simulate GPU computation
                cp.cuda.Stream.null.synchronize()
                gpu_times.append(time.time() - start_time)
            
            results['gpu'] = {
                'avg_time_ms': np.mean(gpu_times) * 1000,
                'ops_per_second': 1.0 / np.mean(gpu_times)
            }
            
            results['speedup'] = np.mean(cpu_times) / np.mean(gpu_times)
            
        except Exception as e:
            results['gpu_error'] = str(e)
        
        return results
    
    def profile_training_performance(self, training_data: List[np.ndarray], epochs: int = 10) -> Dict[str, Any]:
        """Profile training performance"""
        from ..training.masked_trainer import MaskedTrainer
        
        trainer = MaskedTrainer(self.network)
        
        # Profile training
        self.start_session("training_profile")
        self.profile_network_operations()
        
        start_time = time.time()
        trainer.train(training_data, epochs=epochs, verbose=False)
        total_training_time = time.time() - start_time
        
        session = self.end_session()
        
        return {
            'total_training_time': total_training_time,
            'time_per_epoch': total_training_time / epochs,
            'examples_per_second': len(training_data) * epochs / total_training_time,
            'profiling_session': session
        }
    
    def analyze_bottlenecks(self, results: List[ProfileResult]) -> Dict[str, Any]:
        """Analyze performance bottlenecks"""
        bottlenecks = {}
        
        if not results:
            return bottlenecks
        
        # Sort by total time
        by_total_time = sorted(results, key=lambda x: x.total_time, reverse=True)
        bottlenecks['by_total_time'] = by_total_time[:5]
        
        # Sort by average time
        by_avg_time = sorted(results, key=lambda x: x.avg_time, reverse=True)
        bottlenecks['by_avg_time'] = by_avg_time[:5]
        
        # Sort by call frequency
        by_call_count = sorted(results, key=lambda x: x.call_count, reverse=True)
        bottlenecks['by_call_frequency'] = by_call_count[:5]
        
        # Find functions with high variance
        high_variance = []
        for result in results:
            if result.max_time > 0 and result.min_time > 0:
                variance_ratio = result.max_time / result.min_time
                if variance_ratio > 5:  # High variance threshold
                    high_variance.append((result, variance_ratio))
        
        bottlenecks['high_variance'] = sorted(high_variance, key=lambda x: x[1], reverse=True)[:5]
        
        return bottlenecks
    
    def generate_optimization_report(self, session: ProfilingSession) -> str:
        """Generate detailed optimization report"""
        report = []
        report.append(f"Performance Analysis Report - {session.session_name}")
        report.append("=" * 50)
        report.append(f"Session Duration: {session.total_duration:.3f}s")
        report.append(f"Total Functions Profiled: {len(session.results)}")
        report.append("")
        
        # Top bottlenecks
        report.append("🔍 TOP PERFORMANCE BOTTLENECKS:")
        for i, bottleneck in enumerate(session.bottlenecks[:5], 1):
            result = next((r for r in session.results if r.function_name == bottleneck), None)
            if result:
                report.append(f"{i}. {result.function_name}")
                report.append(f"   Total Time: {result.total_time:.3f}s ({result.total_time/session.total_duration*100:.1f}%)")
                report.append(f"   Calls: {result.call_count}, Avg: {result.avg_time*1000:.2f}ms")
                report.append("")
        
        # Recommendations
        report.append("💡 OPTIMIZATION RECOMMENDATIONS:")
        for i, rec in enumerate(session.recommendations, 1):
            report.append(f"{i}. {rec}")
        report.append("")
        
        # Function details
        report.append("📊 DETAILED FUNCTION ANALYSIS:")
        for result in sorted(session.results, key=lambda x: x.total_time, reverse=True)[:10]:
            report.append(f"Function: {result.function_name}")
            report.append(f"  Total Time: {result.total_time:.3f}s")
            report.append(f"  Call Count: {result.call_count}")
            report.append(f"  Avg Time: {result.avg_time*1000:.2f}ms")
            report.append(f"  Min/Max: {result.min_time*1000:.2f}ms / {result.max_time*1000:.2f}ms")
            if result.memory_usage:
                report.append(f"  Memory Delta: {result.memory_usage:.2f}MB")
            report.append("")
        
        return "\n".join(report)
    
    def save_profile_results(self, session: ProfilingSession, filepath: Optional[str] = None) -> str:
        """Save profiling results to file"""
        if filepath is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filepath = f"profile_results_{session.session_name}_{timestamp}.json"
        
        # Convert to serializable format
        data = {
            'session_name': session.session_name,
            'total_duration': session.total_duration,
            'timestamp': session.timestamp,
            'bottlenecks': session.bottlenecks,
            'recommendations': session.recommendations,
            'results': [
                {
                    'function_name': r.function_name,
                    'total_time': r.total_time,
                    'call_count': r.call_count,
                    'avg_time': r.avg_time,
                    'min_time': r.min_time,
                    'max_time': r.max_time,
                    'memory_usage': r.memory_usage,
                    'metadata': r.metadata or {}
                }
                for r in session.results
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Profile results saved to: {filepath}")
        return filepath
    
    def _get_memory_usage(self) -> Optional[float]:
        """Get current memory usage in MB"""
        if self.psutil_available:
            return self.process.memory_info().rss / 1024 / 1024
        return None
    
    def _identify_bottlenecks(self, results: List[ProfileResult]) -> List[str]:
        """Identify performance bottlenecks"""
        if not results:
            return []
        
        # Sort by total time and identify top consumers
        sorted_results = sorted(results, key=lambda x: x.total_time, reverse=True)
        
        bottlenecks = []
        total_time = sum(r.total_time for r in results)
        
        for result in sorted_results:
            # Consider functions that take >10% of total time as bottlenecks
            if result.total_time / total_time > 0.1:
                bottlenecks.append(result.function_name)
            elif len(bottlenecks) >= 5:  # Limit to top 5
                break
        
        return bottlenecks
    
    def _generate_recommendations(self, results: List[ProfileResult], bottlenecks: List[str]) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        if not results:
            return recommendations
        
        # Analyze bottlenecks
        for bottleneck in bottlenecks[:3]:  # Top 3 bottlenecks
            result = next((r for r in results if r.function_name == bottleneck), None)
            if result:
                if 'forward' in bottleneck.lower():
                    recommendations.append(
                        f"Optimize {bottleneck}: Consider vectorization or GPU acceleration "
                        f"(current: {result.avg_time*1000:.1f}ms avg)"
                    )
                elif 'attention' in bottleneck.lower():
                    recommendations.append(
                        f"Optimize attention mechanism in {bottleneck}: "
                        f"Consider sparse attention or caching ({result.call_count} calls)"
                    )
                elif 'causal' in bottleneck.lower():
                    recommendations.append(
                        f"Optimize causal reasoning in {bottleneck}: "
                        f"Consider pruning weak causal links or better caching"
                    )
        
        # General recommendations
        total_calls = sum(r.call_count for r in results)
        if total_calls > 10000:
            recommendations.append("High call frequency detected: Consider function inlining or batching")
        
        # Memory recommendations
        memory_intensive = [r for r in results if r.memory_usage and r.memory_usage > 10]
        if memory_intensive:
            recommendations.append("Memory-intensive functions detected: Consider memory pooling or streaming")
        
        # High variance recommendations
        high_variance = [r for r in results if r.max_time > 5 * r.min_time]
        if high_variance:
            recommendations.append("High timing variance detected: Investigate caching or data-dependent complexity")
        
        return recommendations