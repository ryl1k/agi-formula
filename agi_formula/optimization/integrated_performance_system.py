"""
Integrated Performance Optimization System

Combines all optimization techniques into a cohesive high-performance system:
- Neural network optimizations with object pooling
- JIT compilation for critical paths  
- Intelligent caching for computational results
- Parallel processing coordination
- Memory management and profiling
"""

import numpy as np
import time
import psutil
import threading
from typing import Dict, List, Any, Optional, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

from .neural_optimization import (
    NeuronPool, VectorizedNeuronOperations, NetworkMemoryOptimizer,
    ParallelNeuronProcessor, OptimizedNetworkBuilder
)
from .jit_optimization import JITOptimizedOperations, create_jit_optimizer
from .intelligent_caching import IntelligentCacheManager, create_cache_manager

class PerformanceMetrics:
    """Comprehensive performance monitoring"""
    
    def __init__(self):
        self.start_time = time.time()
        self.operation_times = {}
        self.memory_snapshots = []
        self.cache_performance = {}
        self.optimization_impacts = {}
        self.lock = threading.RLock()
    
    def record_operation(self, operation_name: str, duration: float, memory_delta: float = 0):
        """Record performance metrics for an operation"""
        with self.lock:
            if operation_name not in self.operation_times:
                self.operation_times[operation_name] = []
            
            self.operation_times[operation_name].append({
                'duration': duration,
                'memory_delta': memory_delta,
                'timestamp': time.time()
            })
    
    def get_operation_stats(self, operation_name: str) -> Dict[str, float]:
        """Get statistics for a specific operation"""
        with self.lock:
            if operation_name not in self.operation_times:
                return {}
            
            times = [record['duration'] for record in self.operation_times[operation_name]]
            memory_deltas = [record['memory_delta'] for record in self.operation_times[operation_name]]
            
            return {
                'count': len(times),
                'avg_duration': np.mean(times),
                'std_duration': np.std(times),
                'min_duration': np.min(times),
                'max_duration': np.max(times),
                'total_duration': np.sum(times),
                'avg_memory_delta': np.mean(memory_deltas),
                'total_memory_delta': np.sum(memory_deltas)
            }
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        with self.lock:
            uptime = time.time() - self.start_time
            
            report = {
                'uptime_seconds': uptime,
                'operations': {}
            }
            
            total_operations = 0
            total_time = 0
            
            for op_name in self.operation_times:
                stats = self.get_operation_stats(op_name)
                report['operations'][op_name] = stats
                total_operations += stats.get('count', 0)
                total_time += stats.get('total_duration', 0)
            
            report['summary'] = {
                'total_operations': total_operations,
                'total_computation_time': total_time,
                'operations_per_second': total_operations / max(uptime, 1),
                'efficiency_ratio': total_time / max(uptime, 1)  # % of time spent in recorded operations
            }
            
            return report

class IntegratedPerformanceSystem:
    """Master performance optimization system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        
        # Initialize all optimization subsystems
        self.neuron_pool = NeuronPool(initial_size=self.config['neuron_pool_size'])
        self.network_builder = OptimizedNetworkBuilder()
        self.memory_optimizer = NetworkMemoryOptimizer(max_memory_mb=self.config['max_memory_mb'])
        self.parallel_processor = ParallelNeuronProcessor(max_workers=self.config['max_workers'])
        self.jit_optimizer = create_jit_optimizer(use_cuda=self.config['use_cuda'])
        self.cache_manager = create_cache_manager(max_memory_mb=self.config['cache_memory_mb'])
        
        # Performance monitoring
        self.metrics = PerformanceMetrics()
        
        # System state
        self.is_initialized = False
        self.optimization_level = self.config['optimization_level']
        
        print(f"Integrated Performance System initialized (Level {self.optimization_level})")
        self._warm_up_system()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for the performance system"""
        return {
            'neuron_pool_size': 5000,
            'max_memory_mb': 2048,
            'cache_memory_mb': 512,
            'max_workers': mp.cpu_count(),
            'use_cuda': False,  # Set to True if CUDA is available
            'optimization_level': 3,  # 1=basic, 2=advanced, 3=maximum
            'enable_profiling': True,
            'auto_optimization': True
        }
    
    def _warm_up_system(self):
        """Warm up all optimization components"""
        print("Warming up optimization system...")
        
        # Warm up JIT compiler
        test_data = np.random.randn(100, 50).astype(np.float32)
        test_weights = np.random.randn(30, 50).astype(np.float32) 
        test_biases = np.random.randn(30).astype(np.float32)
        
        # Trigger JIT compilation
        self.jit_optimizer.forward_pass(test_data, test_weights, test_biases)
        
        # Warm up caches with sample operations
        @self.cache_manager.cached_function_call
        def sample_operation(x):
            return np.dot(x, x.T)
        
        sample_operation(test_data)
        
        self.is_initialized = True
        print("System warm-up complete!")
    
    def optimized_forward_pass(self, inputs: np.ndarray, weights: np.ndarray, 
                             biases: np.ndarray, activation: str = 'tanh',
                             layer_id: str = None) -> np.ndarray:
        """Ultra-optimized forward pass with all enhancements"""
        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Try to get from neural cache first
        if layer_id:
            cached_result = self.cache_manager.neural_cache.get_forward_pass(layer_id, inputs, weights)
            if cached_result is not None:
                self.metrics.record_operation('cached_forward_pass', time.perf_counter() - start_time)
                return cached_result
        
        # Use JIT-optimized forward pass
        result = self.jit_optimizer.forward_pass(inputs, weights, biases, activation)
        
        # Cache result for future use
        if layer_id:
            self.cache_manager.neural_cache.cache_forward_pass(layer_id, inputs, weights, result)
        
        # Record performance metrics
        end_time = time.perf_counter()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        self.metrics.record_operation('optimized_forward_pass', end_time - start_time, end_memory - start_memory)
        
        return result
    
    def optimized_network_creation(self, config: 'NetworkConfig') -> 'OptimizedNetwork':
        """Create network with maximum optimization"""
        start_time = time.perf_counter()
        
        # Use optimized network builder
        network = self.network_builder.create_optimized_network(config)
        
        end_time = time.perf_counter()
        self.metrics.record_operation('optimized_network_creation', end_time - start_time)
        
        return network
    
    def bulk_optimized_neuron_creation(self, neuron_configs: List[Dict[str, Any]]) -> List['Neuron']:
        """Create neurons in bulk with maximum optimization"""
        start_time = time.perf_counter()
        
        # Use neuron pool for efficient creation
        neurons = []
        for config in neuron_configs:
            neuron_id = config.get('neuron_id', len(neurons))
            num_inputs = config.get('num_inputs', 10)
            neuron = self.neuron_pool.get_neuron(neuron_id, num_inputs)
            neurons.append(neuron)
        
        end_time = time.perf_counter()
        self.metrics.record_operation('bulk_optimized_neuron_creation', end_time - start_time)
        
        return neurons
    
    def optimized_attention_computation(self, query: np.ndarray, key: np.ndarray, 
                                      value: np.ndarray, scale_factor: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """Optimized attention computation with caching"""
        start_time = time.perf_counter()
        
        # Try cache first
        cached_result = self.cache_manager.neural_cache.get_attention(query, key, value)
        if cached_result is not None:
            self.metrics.record_operation('cached_attention', time.perf_counter() - start_time)
            return cached_result['output'], cached_result['weights']
        
        # Use JIT-optimized attention
        attention_output, attention_weights = self.jit_optimizer.attention_computation(
            query, key, value, scale_factor
        )
        
        # Cache result
        self.cache_manager.neural_cache.cache_attention(query, key, value, attention_weights, attention_output)
        
        end_time = time.perf_counter()
        self.metrics.record_operation('optimized_attention', end_time - start_time)
        
        return attention_output, attention_weights
    
    def optimized_concept_operations(self, concept_vectors: List[np.ndarray], 
                                   operation: str = 'similarity', **kwargs) -> Any:
        """Optimized concept operations with caching"""
        start_time = time.perf_counter()
        
        if operation == 'similarity':
            result = self.jit_optimizer.concept_operations(concept_vectors)
        elif operation == 'composition':
            composition_weights = kwargs.get('weights')
            if composition_weights is not None:
                result = self.jit_optimizer.concept_operations(concept_vectors, composition_weights)
            else:
                result = None
        else:
            result = None
        
        end_time = time.perf_counter()
        self.metrics.record_operation(f'optimized_concept_{operation}', end_time - start_time)
        
        return result
    
    def parallel_batch_processing(self, batch_data: List[Any], 
                                processing_func: Callable, **kwargs) -> List[Any]:
        """Process batch data in parallel with optimization"""
        start_time = time.perf_counter()
        
        # Use parallel processor
        results = self.parallel_processor.thread_pool.map(
            lambda data: processing_func(data, **kwargs), batch_data
        )
        
        end_time = time.perf_counter()
        self.metrics.record_operation('parallel_batch_processing', end_time - start_time)
        
        return list(results)
    
    def adaptive_precision_processing(self, data: np.ndarray, task_complexity: str = 'moderate',
                                    accuracy_requirement: float = 0.95) -> np.ndarray:
        """Process data with adaptive precision optimization"""
        start_time = time.perf_counter()
        
        # Determine optimal precision based on task
        if task_complexity == 'simple' and accuracy_requirement < 0.9:
            dtype = np.float16  # 16-bit precision
        elif task_complexity == 'moderate' and accuracy_requirement < 0.99:
            dtype = np.float32  # 32-bit precision
        else:
            dtype = np.float64  # 64-bit precision
        
        # Convert data to optimal precision
        optimized_data = data.astype(dtype)
        
        # Process with optimized precision
        result = optimized_data  # Simplified - would contain actual processing logic
        
        end_time = time.perf_counter()
        self.metrics.record_operation('adaptive_precision_processing', end_time - start_time)
        
        return result
    
    def get_system_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive system performance report"""
        base_report = self.metrics.get_comprehensive_report()
        
        # Add system-specific metrics
        base_report['system_stats'] = {
            'neuron_pool': self.neuron_pool.get_pool_stats(),
            'cache_stats': self.cache_manager.get_comprehensive_stats(),
            'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024,
            'optimization_level': self.optimization_level,
            'is_initialized': self.is_initialized
        }
        
        return base_report
    
    def optimize_system_performance(self):
        """Dynamically optimize system performance based on usage patterns"""
        print("Optimizing system performance...")
        
        # Optimize cache sizes
        self.cache_manager.optimize_cache_sizes()
        
        # Adjust neuron pool size based on usage
        pool_stats = self.neuron_pool.get_pool_stats()
        utilization = pool_stats['in_use_neurons'] / max(pool_stats['total_neurons'], 1)
        
        if utilization > 0.8:
            # High utilization - expand pool
            print(f"High neuron pool utilization ({utilization:.2%}). Expanding pool...")
            for i in range(1000):
                from ..core.neuron import Neuron  
                neuron = Neuron(neuron_id=pool_stats['total_neurons'] + i, num_inputs=10)
                self.neuron_pool.available_neurons.append(neuron)
                self.neuron_pool.all_neurons.append(neuron)
        
        # Clean up memory if needed
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
        if memory_usage > self.config['max_memory_mb'] * 0.8:
            print(f"High memory usage ({memory_usage:.1f} MB). Cleaning up...")
            self.cache_manager.clear_all_caches()
            self.memory_optimizer.clear_cache_if_needed()
        
        print("System performance optimization complete!")
    
    def shutdown(self):
        """Shutdown the performance system"""
        print("Shutting down integrated performance system...")
        self.parallel_processor.shutdown()
        self.cache_manager.clear_all_caches()
        print("Shutdown complete!")

def create_integrated_performance_system(config: Optional[Dict[str, Any]] = None) -> IntegratedPerformanceSystem:
    """Create integrated performance system with optional configuration"""
    return IntegratedPerformanceSystem(config)

def benchmark_integrated_system():
    """Comprehensive benchmark of the integrated performance system"""
    print("INTEGRATED PERFORMANCE SYSTEM BENCHMARK")
    print("=" * 50)
    
    # Create system
    config = {
        'optimization_level': 3,
        'neuron_pool_size': 10000,
        'max_memory_mb': 1024,
        'cache_memory_mb': 256
    }
    
    system = create_integrated_performance_system(config)
    
    # Test data
    test_inputs = np.random.randn(1000, 512).astype(np.float32)
    test_weights = np.random.randn(256, 512).astype(np.float32) 
    test_biases = np.random.randn(256).astype(np.float32)
    
    # Benchmark forward pass
    print("Testing optimized forward pass...")
    start_time = time.perf_counter()
    for i in range(100):
        result = system.optimized_forward_pass(
            test_inputs, test_weights, test_biases, 
            layer_id=f"test_layer_{i % 10}"  # Simulate caching
        )
    forward_pass_time = time.perf_counter() - start_time
    
    # Benchmark bulk neuron creation
    print("Testing bulk neuron creation...")
    neuron_configs = [{'neuron_id': i, 'num_inputs': 10} for i in range(10000)]
    start_time = time.perf_counter()
    neurons = system.bulk_optimized_neuron_creation(neuron_configs)
    neuron_creation_time = time.perf_counter() - start_time
    
    # Benchmark attention computation
    print("Testing attention computation...")
    query = np.random.randn(100, 64).astype(np.float32)
    key = np.random.randn(100, 64).astype(np.float32)
    value = np.random.randn(100, 64).astype(np.float32)
    
    start_time = time.perf_counter()
    for _ in range(50):
        attention_out, attention_weights = system.optimized_attention_computation(query, key, value)
    attention_time = time.perf_counter() - start_time
    
    # Get performance report
    report = system.get_system_performance_report()
    
    print("\nPERFORMANCE RESULTS:")
    print(f"Forward pass (100 iterations): {forward_pass_time:.6f}s")
    print(f"Bulk neuron creation (10,000): {neuron_creation_time:.6f}s") 
    print(f"Attention computation (50 iterations): {attention_time:.6f}s")
    print(f"Cache hit rate: {report['system_stats']['cache_stats']['overall_hit_rate']:.2%}")
    print(f"Memory usage: {report['system_stats']['memory_usage_mb']:.1f} MB")
    
    # Test optimization
    system.optimize_system_performance()
    
    # Shutdown
    system.shutdown()
    
    return {
        'forward_pass_time': forward_pass_time,
        'neuron_creation_time': neuron_creation_time,
        'attention_time': attention_time,
        'cache_hit_rate': report['system_stats']['cache_stats']['overall_hit_rate'],
        'memory_usage_mb': report['system_stats']['memory_usage_mb']
    }

if __name__ == "__main__":
    results = benchmark_integrated_system()