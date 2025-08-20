"""
Intelligent Caching System

Advanced caching strategies to minimize computational overhead
by caching frequently used results and intermediate computations.
"""

import numpy as np
import hashlib
import pickle
import time
from typing import Dict, Any, Optional, Tuple, List, Callable
from collections import OrderedDict, defaultdict
import threading
from functools import wraps
import psutil
import gc

class LRUCache:
    """Least Recently Used cache with size limits"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                self.hits += 1
                return value
            else:
                self.misses += 1
                return None
    
    def put(self, key: str, value: Any):
        """Put item in cache"""
        with self.lock:
            if key in self.cache:
                # Update existing item
                self.cache.pop(key)
            elif len(self.cache) >= self.max_size:
                # Remove least recently used item
                self.cache.popitem(last=False)
            
            self.cache[key] = value
    
    def clear(self):
        """Clear cache"""
        with self.lock:
            self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / max(total_requests, 1)
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'total_requests': total_requests
        }

class TieredCache:
    """Multi-level caching with different strategies per tier"""
    
    def __init__(self, l1_size: int = 100, l2_size: int = 1000, l3_size: int = 10000):
        self.l1_cache = LRUCache(l1_size)  # Hot cache - most frequently accessed
        self.l2_cache = LRUCache(l2_size)  # Warm cache - recently accessed
        self.l3_cache = LRUCache(l3_size)  # Cold cache - historical access
        
        self.access_frequency = defaultdict(int)
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from tiered cache"""
        with self.lock:
            # Try L1 cache first (hottest)
            value = self.l1_cache.get(key)
            if value is not None:
                self.access_frequency[key] += 1
                return value
            
            # Try L2 cache
            value = self.l2_cache.get(key)
            if value is not None:
                self.access_frequency[key] += 1
                # Promote to L1 if frequently accessed
                if self.access_frequency[key] > 10:
                    self.l1_cache.put(key, value)
                return value
            
            # Try L3 cache
            value = self.l3_cache.get(key)
            if value is not None:
                self.access_frequency[key] += 1
                # Promote to L2
                self.l2_cache.put(key, value)
                return value
            
            return None
    
    def put(self, key: str, value: Any):
        """Put item in appropriate cache tier"""
        with self.lock:
            frequency = self.access_frequency[key]
            
            if frequency > 10:
                self.l1_cache.put(key, value)
            elif frequency > 3:
                self.l2_cache.put(key, value)
            else:
                self.l3_cache.put(key, value)
    
    def clear(self):
        """Clear all cache tiers"""
        with self.lock:
            self.l1_cache.clear()
            self.l2_cache.clear()
            self.l3_cache.clear()
            self.access_frequency.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all cache tiers"""
        return {
            'l1': self.l1_cache.get_stats(),
            'l2': self.l2_cache.get_stats(),
            'l3': self.l3_cache.get_stats(),
            'total_keys_tracked': len(self.access_frequency)
        }

class ComputationCache:
    """Cache for expensive computational results"""
    
    def __init__(self, max_memory_mb: int = 512):
        self.max_memory_mb = max_memory_mb
        self.cache = TieredCache(l1_size=50, l2_size=200, l3_size=1000)
        self.computation_times = {}
        self.memory_usage = {}
        self.lock = threading.RLock()
    
    def _generate_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key for function call"""
        # Create hashable representation
        key_data = {
            'function': func_name,
            'args': self._hashable_args(args),
            'kwargs': self._hashable_kwargs(kwargs)
        }
        
        key_str = str(key_data)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _hashable_args(self, args: tuple) -> tuple:
        """Convert args to hashable format"""
        hashable = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                # Use shape and a sample of values for numpy arrays
                shape = arg.shape
                if arg.size > 1000:
                    # For large arrays, use hash of flattened sample
                    sample_indices = np.linspace(0, arg.size-1, 100, dtype=int)
                    sample = arg.flat[sample_indices]
                    sample_hash = hash(sample.tobytes())
                    hashable.append(f"array_shape_{shape}_hash_{sample_hash}")
                else:
                    # For small arrays, use full hash
                    hashable.append(f"array_shape_{shape}_hash_{hash(arg.tobytes())}")
            else:
                hashable.append(arg)
        return tuple(hashable)
    
    def _hashable_kwargs(self, kwargs: dict) -> tuple:
        """Convert kwargs to hashable format"""
        hashable_items = []
        for key, value in sorted(kwargs.items()):
            if isinstance(value, np.ndarray):
                if value.size > 1000:
                    sample_indices = np.linspace(0, value.size-1, 100, dtype=int)
                    sample = value.flat[sample_indices]
                    sample_hash = hash(sample.tobytes())
                    hashable_value = f"array_shape_{value.shape}_hash_{sample_hash}"
                else:
                    hashable_value = f"array_shape_{value.shape}_hash_{hash(value.tobytes())}"
            else:
                hashable_value = value
            hashable_items.append((key, hashable_value))
        return tuple(hashable_items)
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def clear_if_needed(self):
        """Clear cache if memory usage is too high"""
        if self.get_memory_usage() > self.max_memory_mb:
            self.cache.clear()
            self.computation_times.clear()
            self.memory_usage.clear()
            gc.collect()
    
    def cached_call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with caching"""
        func_name = func.__name__
        cache_key = self._generate_key(func_name, args, kwargs)
        
        # Try to get from cache
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        # Execute function and cache result
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        computation_time = end_time - start_time
        
        # Store in cache if computation was expensive enough
        if computation_time > 0.001:  # Cache if takes more than 1ms
            with self.lock:
                self.cache.put(cache_key, result)
                self.computation_times[cache_key] = computation_time
        
        self.clear_if_needed()
        return result

class NeuralNetworkCache:
    """Specialized cache for neural network operations"""
    
    def __init__(self, max_size: int = 500):
        self.forward_pass_cache = LRUCache(max_size)
        self.weight_gradient_cache = LRUCache(max_size)
        self.activation_cache = LRUCache(max_size * 2)  # Activations are smaller
        self.attention_cache = LRUCache(max_size // 2)  # Attention computations are larger
        
    def cache_forward_pass(self, layer_id: str, inputs: np.ndarray, weights: np.ndarray, result: np.ndarray):
        """Cache forward pass result"""
        key = self._generate_layer_key("forward", layer_id, inputs, weights)
        self.forward_pass_cache.put(key, result.copy())
    
    def get_forward_pass(self, layer_id: str, inputs: np.ndarray, weights: np.ndarray) -> Optional[np.ndarray]:
        """Get cached forward pass result"""
        key = self._generate_layer_key("forward", layer_id, inputs, weights)
        return self.forward_pass_cache.get(key)
    
    def cache_activation(self, activation_type: str, inputs: np.ndarray, result: np.ndarray):
        """Cache activation function result"""
        key = self._generate_activation_key(activation_type, inputs)
        self.activation_cache.put(key, result.copy())
    
    def get_activation(self, activation_type: str, inputs: np.ndarray) -> Optional[np.ndarray]:
        """Get cached activation result"""
        key = self._generate_activation_key(activation_type, inputs)
        return self.activation_cache.get(key)
    
    def cache_attention(self, query: np.ndarray, key: np.ndarray, value: np.ndarray, 
                       attention_weights: np.ndarray, attention_output: np.ndarray):
        """Cache attention computation result"""
        cache_key = self._generate_attention_key(query, key, value)
        result = {
            'weights': attention_weights.copy(),
            'output': attention_output.copy()
        }
        self.attention_cache.put(cache_key, result)
    
    def get_attention(self, query: np.ndarray, key: np.ndarray, value: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        """Get cached attention result"""
        cache_key = self._generate_attention_key(query, key, value)
        return self.attention_cache.get(cache_key)
    
    def _generate_layer_key(self, operation: str, layer_id: str, inputs: np.ndarray, weights: np.ndarray) -> str:
        """Generate cache key for layer operation"""
        input_hash = self._array_hash(inputs)
        weight_hash = self._array_hash(weights)
        key_data = f"{operation}_{layer_id}_{input_hash}_{weight_hash}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _generate_activation_key(self, activation_type: str, inputs: np.ndarray) -> str:
        """Generate cache key for activation function"""
        input_hash = self._array_hash(inputs)
        key_data = f"activation_{activation_type}_{input_hash}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _generate_attention_key(self, query: np.ndarray, key: np.ndarray, value: np.ndarray) -> str:
        """Generate cache key for attention computation"""
        query_hash = self._array_hash(query)
        key_hash = self._array_hash(key)
        value_hash = self._array_hash(value)
        key_data = f"attention_{query_hash}_{key_hash}_{value_hash}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _array_hash(self, array: np.ndarray) -> str:
        """Generate hash for numpy array"""
        if array.size > 1000:
            # For large arrays, use sample
            sample_indices = np.linspace(0, array.size-1, 100, dtype=int)
            sample = array.flat[sample_indices]
            return f"shape_{array.shape}_sample_{hash(sample.tobytes())}"
        else:
            return f"shape_{array.shape}_full_{hash(array.tobytes())}"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'forward_pass': self.forward_pass_cache.get_stats(),
            'weight_gradient': self.weight_gradient_cache.get_stats(),
            'activation': self.activation_cache.get_stats(),
            'attention': self.attention_cache.get_stats()
        }
    
    def clear_all(self):
        """Clear all neural network caches"""
        self.forward_pass_cache.clear()
        self.weight_gradient_cache.clear()
        self.activation_cache.clear()
        self.attention_cache.clear()

class ConceptCache:
    """Specialized cache for concept operations"""
    
    def __init__(self, max_size: int = 1000):
        self.concept_similarity_cache = LRUCache(max_size)
        self.composition_cache = LRUCache(max_size // 2)
        self.concept_vectors = {}  # Store concept vectors for quick access
        
    def cache_concept_vector(self, concept_name: str, vector: np.ndarray):
        """Cache concept vector"""
        self.concept_vectors[concept_name] = vector.copy()
    
    def get_concept_vector(self, concept_name: str) -> Optional[np.ndarray]:
        """Get cached concept vector"""
        return self.concept_vectors.get(concept_name)
    
    def cache_similarity(self, concept_a: str, concept_b: str, similarity: float):
        """Cache concept similarity"""
        key = f"{min(concept_a, concept_b)}_{max(concept_a, concept_b)}"
        self.concept_similarity_cache.put(key, similarity)
    
    def get_similarity(self, concept_a: str, concept_b: str) -> Optional[float]:
        """Get cached concept similarity"""
        key = f"{min(concept_a, concept_b)}_{max(concept_a, concept_b)}"
        return self.concept_similarity_cache.get(key)
    
    def cache_composition(self, component_concepts: List[str], weights: np.ndarray, result_vector: np.ndarray):
        """Cache concept composition result"""
        key = self._generate_composition_key(component_concepts, weights)
        self.composition_cache.put(key, result_vector.copy())
    
    def get_composition(self, component_concepts: List[str], weights: np.ndarray) -> Optional[np.ndarray]:
        """Get cached composition result"""
        key = self._generate_composition_key(component_concepts, weights)
        return self.composition_cache.get(key)
    
    def _generate_composition_key(self, concepts: List[str], weights: np.ndarray) -> str:
        """Generate key for composition cache"""
        sorted_concepts = sorted(concepts)
        weight_hash = hash(weights.tobytes())
        key_data = f"composition_{'_'.join(sorted_concepts)}_weights_{weight_hash}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get concept cache statistics"""
        return {
            'similarity_cache': self.concept_similarity_cache.get_stats(),
            'composition_cache': self.composition_cache.get_stats(),
            'concept_vectors_count': len(self.concept_vectors)
        }

class IntelligentCacheManager:
    """Master cache manager that coordinates all caching systems"""
    
    def __init__(self, max_memory_mb: int = 1024):
        self.max_memory_mb = max_memory_mb
        self.computation_cache = ComputationCache(max_memory_mb // 2)
        self.neural_cache = NeuralNetworkCache(max_size=500)
        self.concept_cache = ConceptCache(max_size=1000)
        
        self.start_time = time.time()
        self.cache_hits = 0
        self.cache_misses = 0
    
    def cached_function_call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with intelligent caching"""
        return self.computation_cache.cached_call(func, *args, **kwargs)
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        uptime = time.time() - self.start_time
        total_requests = self.cache_hits + self.cache_misses
        overall_hit_rate = self.cache_hits / max(total_requests, 1)
        
        return {
            'uptime_seconds': uptime,
            'overall_hit_rate': overall_hit_rate,
            'total_cache_requests': total_requests,
            'memory_usage_mb': self.computation_cache.get_memory_usage(),
            'max_memory_mb': self.max_memory_mb,
            'computation_cache': self.computation_cache.cache.get_stats(),
            'neural_cache': self.neural_cache.get_stats(),
            'concept_cache': self.concept_cache.get_stats()
        }
    
    def clear_all_caches(self):
        """Clear all caches"""
        self.computation_cache.cache.clear()
        self.neural_cache.clear_all()
        self.concept_cache.concept_similarity_cache.clear()
        self.concept_cache.composition_cache.clear()
        gc.collect()
    
    def optimize_cache_sizes(self):
        """Dynamically optimize cache sizes based on usage patterns"""
        stats = self.get_comprehensive_stats()
        
        # If memory usage is high, reduce cache sizes
        memory_ratio = stats['memory_usage_mb'] / stats['max_memory_mb']
        if memory_ratio > 0.8:
            print("High memory usage detected. Optimizing cache sizes...")
            self.neural_cache.forward_pass_cache.max_size = max(100, int(self.neural_cache.forward_pass_cache.max_size * 0.8))
            self.concept_cache.concept_similarity_cache.max_size = max(200, int(self.concept_cache.concept_similarity_cache.max_size * 0.8))
            
        # If hit rates are low, increase cache sizes (if memory allows)
        elif stats['overall_hit_rate'] < 0.3 and memory_ratio < 0.5:
            print("Low hit rate detected. Increasing cache sizes...")
            self.neural_cache.forward_pass_cache.max_size = int(self.neural_cache.forward_pass_cache.max_size * 1.2)
            self.concept_cache.concept_similarity_cache.max_size = int(self.concept_cache.concept_similarity_cache.max_size * 1.2)

# Decorator for automatic caching
def cached(cache_manager: IntelligentCacheManager):
    """Decorator to automatically cache function results"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return cache_manager.cached_function_call(func, *args, **kwargs)
        return wrapper
    return decorator

# Global cache manager instance
global_cache_manager = IntelligentCacheManager()

# Factory function
def create_cache_manager(max_memory_mb: int = 1024) -> IntelligentCacheManager:
    """Create intelligent cache manager"""
    return IntelligentCacheManager(max_memory_mb=max_memory_mb)

# Performance benchmark
def benchmark_caching_performance():
    """Benchmark caching performance improvements"""
    import time
    
    print("CACHING OPTIMIZATION BENCHMARK")
    print("=" * 40)
    
    # Test function
    def expensive_computation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        time.sleep(0.001)  # Simulate expensive computation
        return np.dot(x, y.T)
    
    # Test data
    x = np.random.randn(100, 50)
    y = np.random.randn(100, 50)
    
    cache_manager = create_cache_manager()
    
    # Test without caching
    start_time = time.perf_counter()
    for _ in range(50):
        result_no_cache = expensive_computation(x, y)
    no_cache_time = time.perf_counter() - start_time
    
    # Test with caching
    @cached(cache_manager)
    def cached_expensive_computation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        time.sleep(0.001)  # Simulate expensive computation
        return np.dot(x, y.T)
    
    start_time = time.perf_counter()
    for _ in range(50):
        result_cached = cached_expensive_computation(x, y)
    cache_time = time.perf_counter() - start_time
    
    speedup = no_cache_time / cache_time
    stats = cache_manager.get_comprehensive_stats()
    
    print(f"Without caching: {no_cache_time:.6f}s")
    print(f"With caching: {cache_time:.6f}s")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Cache hit rate: {stats['overall_hit_rate']:.2%}")
    
    return {
        'speedup': speedup,
        'cache_time': cache_time,
        'no_cache_time': no_cache_time,
        'hit_rate': stats['overall_hit_rate']
    }