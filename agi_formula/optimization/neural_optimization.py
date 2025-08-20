"""
Neural Network Performance Optimizations

Implements vectorized operations, memory pools, and JIT compilation
to address the major bottlenecks identified in performance profiling.
"""

import numpy as np
import numba
from numba import jit, cuda
from typing import List, Dict, Any, Optional, Tuple
import psutil
import gc
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

class NeuronPool:
    """Object pool for efficient neuron creation and reuse"""
    
    def __init__(self, initial_size: int = 1000):
        self.available_neurons = []
        self.all_neurons = []
        self.initial_size = initial_size
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Pre-allocate neurons in the pool"""
        from ..core.neuron import Neuron
        
        for i in range(self.initial_size):
            neuron = Neuron(neuron_id=i, num_inputs=10)
            self.available_neurons.append(neuron)
            self.all_neurons.append(neuron)
    
    def get_neuron(self, neuron_id: int, num_inputs: int = 10, **kwargs) -> 'Neuron':
        """Get a neuron from the pool or create new one if pool is empty"""
        if self.available_neurons:
            neuron = self.available_neurons.pop()
            # Reset neuron properties
            neuron.id = neuron_id
            # Reset other properties as needed
            return neuron
        else:
            # Pool exhausted, create new neuron
            from ..core.neuron import Neuron
            neuron = Neuron(neuron_id=neuron_id, num_inputs=num_inputs, **kwargs)
            self.all_neurons.append(neuron)
            return neuron
    
    def return_neuron(self, neuron: 'Neuron'):
        """Return a neuron to the pool for reuse"""
        self.available_neurons.append(neuron)
    
    def get_pool_stats(self) -> Dict[str, int]:
        """Get statistics about pool usage"""
        return {
            'total_neurons': len(self.all_neurons),
            'available_neurons': len(self.available_neurons),
            'in_use_neurons': len(self.all_neurons) - len(self.available_neurons)
        }

class VectorizedNeuronOperations:
    """Vectorized operations for batch neuron processing"""
    
    @staticmethod
    @jit(nopython=True)
    def batch_activation(inputs: np.ndarray, weights: np.ndarray, biases: np.ndarray) -> np.ndarray:
        """Vectorized activation computation for multiple neurons"""
        return np.tanh(np.dot(inputs, weights.T) + biases)
    
    @staticmethod
    @jit(nopython=True) 
    def batch_forward_pass(layer_inputs: np.ndarray, layer_weights: np.ndarray, layer_biases: np.ndarray) -> np.ndarray:
        """Optimized forward pass for entire layer"""
        return np.tanh(np.dot(layer_inputs, layer_weights.T) + layer_biases)
    
    @staticmethod
    @jit(nopython=True)
    def batch_backward_pass(gradients: np.ndarray, weights: np.ndarray, inputs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Optimized backward pass computation"""
        # Gradient w.r.t weights
        weight_gradients = np.dot(gradients.T, inputs)
        
        # Gradient w.r.t biases  
        bias_gradients = np.sum(gradients, axis=0)
        
        # Gradient w.r.t inputs (for next layer)
        input_gradients = np.dot(gradients, weights)
        
        return weight_gradients, bias_gradients, input_gradients

class NetworkMemoryOptimizer:
    """Memory optimization for network operations"""
    
    def __init__(self, max_memory_mb: int = 1024):
        self.max_memory_mb = max_memory_mb
        self.network_cache = {}
        self.memory_pools = {}
        
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def clear_cache_if_needed(self):
        """Clear caches if memory usage is too high"""
        if self.get_memory_usage() > self.max_memory_mb:
            self.network_cache.clear()
            gc.collect()
    
    def get_cached_network(self, config_hash: str) -> Optional['Network']:
        """Get cached network configuration"""
        return self.network_cache.get(config_hash)
    
    def cache_network(self, config_hash: str, network: 'Network'):
        """Cache network configuration"""
        self.clear_cache_if_needed()
        self.network_cache[config_hash] = network
    
    def allocate_weight_matrix(self, shape: Tuple[int, int], dtype: np.dtype = np.float32) -> np.ndarray:
        """Efficiently allocate weight matrices"""
        # Use memory pool for common shapes
        shape_key = f"{shape}_{dtype}"
        
        if shape_key not in self.memory_pools:
            self.memory_pools[shape_key] = []
        
        pool = self.memory_pools[shape_key]
        if pool:
            return pool.pop()
        else:
            return np.zeros(shape, dtype=dtype)
    
    def deallocate_weight_matrix(self, matrix: np.ndarray):
        """Return weight matrix to pool"""
        shape_key = f"{matrix.shape}_{matrix.dtype}"
        
        if shape_key not in self.memory_pools:
            self.memory_pools[shape_key] = []
        
        # Reset matrix and return to pool
        matrix.fill(0)
        self.memory_pools[shape_key].append(matrix)

class ParallelNeuronProcessor:
    """Parallel processing for independent neuron operations"""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or mp.cpu_count()
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        
    def create_neurons_parallel(self, neuron_configs: List[Dict[str, Any]]) -> List['Neuron']:
        """Create neurons in parallel"""
        def create_single_neuron(config):
            from ..core.neuron import Neuron
            return Neuron(**config)
        
        futures = [self.thread_pool.submit(create_single_neuron, config) for config in neuron_configs]
        return [future.result() for future in futures]
    
    def batch_process_neurons(self, neurons: List['Neuron'], inputs: List[np.ndarray]) -> List[np.ndarray]:
        """Process neurons in parallel"""
        def process_single_neuron(args):
            neuron, input_data = args
            return neuron.process(input_data) if hasattr(neuron, 'process') else None
        
        neuron_input_pairs = list(zip(neurons, inputs))
        futures = [self.thread_pool.submit(process_single_neuron, pair) for pair in neuron_input_pairs]
        return [future.result() for future in futures if future.result() is not None]
    
    def shutdown(self):
        """Shutdown thread pool"""
        self.thread_pool.shutdown(wait=True)

class OptimizedNetworkBuilder:
    """Optimized network construction with all performance enhancements"""
    
    def __init__(self):
        self.neuron_pool = NeuronPool(initial_size=5000)
        self.memory_optimizer = NetworkMemoryOptimizer()
        self.parallel_processor = ParallelNeuronProcessor()
        self.vectorized_ops = VectorizedNeuronOperations()
        
    def create_optimized_network(self, config: 'NetworkConfig') -> 'OptimizedNetwork':
        """Create network with all optimizations enabled"""
        # Generate config hash for caching
        config_hash = self._generate_config_hash(config)
        
        # Check cache first
        cached_network = self.memory_optimizer.get_cached_network(config_hash)
        if cached_network:
            return cached_network
        
        # Create optimized network
        network = OptimizedNetwork(
            config=config,
            neuron_pool=self.neuron_pool,
            memory_optimizer=self.memory_optimizer,
            parallel_processor=self.parallel_processor,
            vectorized_ops=self.vectorized_ops
        )
        
        # Cache for future use
        self.memory_optimizer.cache_network(config_hash, network)
        
        return network
    
    def bulk_create_networks(self, configs: List['NetworkConfig']) -> List['OptimizedNetwork']:
        """Create multiple networks efficiently"""
        # Process in parallel
        def create_single_network(config):
            return self.create_optimized_network(config)
        
        futures = [self.parallel_processor.thread_pool.submit(create_single_network, config) 
                  for config in configs]
        return [future.result() for future in futures]
    
    def _generate_config_hash(self, config: 'NetworkConfig') -> str:
        """Generate hash for network configuration"""
        import hashlib
        # Simple hash based on key config parameters
        config_str = f"{getattr(config, 'input_size', 0)}_{getattr(config, 'output_size', 0)}_" + \
                    f"{getattr(config, 'num_layers', 0)}_{getattr(config, 'activation', 'tanh')}"
        return hashlib.md5(config_str.encode()).hexdigest()

class OptimizedNetwork:
    """Network with all performance optimizations applied"""
    
    def __init__(self, config: 'NetworkConfig', neuron_pool: NeuronPool, 
                 memory_optimizer: NetworkMemoryOptimizer,
                 parallel_processor: ParallelNeuronProcessor,
                 vectorized_ops: VectorizedNeuronOperations):
        self.config = config
        self.neuron_pool = neuron_pool
        self.memory_optimizer = memory_optimizer
        self.parallel_processor = parallel_processor
        self.vectorized_ops = vectorized_ops
        
        # Pre-allocated weight matrices
        self.layer_weights = []
        self.layer_biases = []
        self._initialize_optimized_structure()
    
    def _initialize_optimized_structure(self):
        """Initialize network structure with memory optimization"""
        input_size = getattr(self.config, 'input_size', 128)
        output_size = getattr(self.config, 'output_size', 64)
        num_layers = getattr(self.config, 'num_layers', 3)
        
        # Calculate layer sizes
        layer_sizes = [input_size]
        for i in range(num_layers - 1):
            # Gradually reduce size
            size = max(output_size, input_size // (2 ** (i + 1)))
            layer_sizes.append(size)
        layer_sizes.append(output_size)
        
        # Pre-allocate weight matrices
        for i in range(len(layer_sizes) - 1):
            weight_shape = (layer_sizes[i+1], layer_sizes[i])
            bias_shape = (layer_sizes[i+1],)
            
            weights = self.memory_optimizer.allocate_weight_matrix(weight_shape)
            biases = self.memory_optimizer.allocate_weight_matrix(bias_shape)
            
            # Initialize with Xavier initialization
            weights[:] = np.random.randn(*weight_shape) * np.sqrt(2.0 / layer_sizes[i])
            biases[:] = 0.1
            
            self.layer_weights.append(weights)
            self.layer_biases.append(biases)
    
    def forward_pass(self, inputs: np.ndarray) -> np.ndarray:
        """Optimized forward pass using vectorized operations"""
        current_activation = inputs
        
        for weights, biases in zip(self.layer_weights, self.layer_biases):
            current_activation = self.vectorized_ops.batch_forward_pass(
                current_activation, weights, biases
            )
        
        return current_activation
    
    def backward_pass(self, gradients: np.ndarray, inputs: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Optimized backward pass"""
        layer_gradients = []
        current_gradients = gradients
        
        # Reverse iterate through layers
        for i in reversed(range(len(self.layer_weights))):
            weights = self.layer_weights[i]
            layer_input = inputs if i == 0 else None  # Simplified for demo
            
            weight_grads, bias_grads, input_grads = self.vectorized_ops.batch_backward_pass(
                current_gradients, weights, layer_input or current_gradients
            )
            
            layer_gradients.insert(0, (weight_grads, bias_grads))
            current_gradients = input_grads
        
        return layer_gradients

# Factory function for easy integration
def create_optimized_network_builder() -> OptimizedNetworkBuilder:
    """Create optimized network builder with all performance enhancements"""
    return OptimizedNetworkBuilder()

# Performance testing function
def benchmark_optimization_improvements():
    """Benchmark the performance improvements"""
    import time
    
    print("NEURAL OPTIMIZATION BENCHMARK")
    print("=" * 40)
    
    # Test network creation speed
    def test_optimized_creation(n_networks=100):
        builder = create_optimized_network_builder()
        
        from ..core.network import NetworkConfig
        configs = [NetworkConfig() for _ in range(n_networks)]
        
        start_time = time.perf_counter()
        networks = builder.bulk_create_networks(configs)
        end_time = time.perf_counter()
        
        return end_time - start_time
    
    # Test neuron pool performance
    def test_neuron_pool_performance(n_neurons=10000):
        pool = NeuronPool(initial_size=1000)
        
        start_time = time.perf_counter()
        neurons = []
        for i in range(n_neurons):
            neuron = pool.get_neuron(i)
            neurons.append(neuron)
        
        for neuron in neurons:
            pool.return_neuron(neuron)
        
        end_time = time.perf_counter()
        return end_time - start_time
    
    print("Testing optimized network creation...")
    optimized_time = test_optimized_creation()
    print(f"Optimized creation time: {optimized_time:.6f}s")
    
    print("Testing neuron pool performance...")
    pool_time = test_neuron_pool_performance()
    print(f"Neuron pool time: {pool_time:.6f}s")
    
    print("\nOptimizations ready for integration!")
    return {
        'optimized_creation_time': optimized_time,
        'neuron_pool_time': pool_time
    }