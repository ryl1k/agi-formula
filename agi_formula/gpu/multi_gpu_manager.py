"""
Multi-GPU Manager for AGI-Formula

Manages multiple GPU devices for distributed AGI computation:
- Automatic device detection and allocation
- Load balancing across available GPUs
- Memory management and synchronization
- Fault tolerance and failover
"""

import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple, Callable
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

# GPU computing imports
try:
    import cupy as cp
    GPU_AVAILABLE = True
    ArrayType = cp.ndarray
except ImportError:
    import numpy as np
    cp = None
    GPU_AVAILABLE = False
    ArrayType = np.ndarray  # Fallback to numpy for type annotations

from .advanced_kernels import AdvancedGPUKernels


class GPUDevice:
    """Represents a single GPU device with its capabilities"""
    
    def __init__(self, device_id: int):
        self.device_id = device_id
        self.kernels = None
        self.memory_info = {}
        self.load_score = 0.0
        self.available = False
        
        self._initialize_device()
    
    def _initialize_device(self):
        """Initialize the GPU device"""
        if not GPU_AVAILABLE:
            return
        
        try:
            with cp.cuda.Device(self.device_id):
                # Test device accessibility
                test_array = cp.zeros(10)
                del test_array
                
                # Initialize kernels
                self.kernels = AdvancedGPUKernels(self.device_id)
                
                # Get memory info
                self.memory_info = self._get_memory_info()
                
                self.available = True
                print(f"GPU {self.device_id} initialized successfully")
                
        except Exception as e:
            warnings.warn(f"Failed to initialize GPU {self.device_id}: {e}")
            self.available = False
    
    def _get_memory_info(self) -> Dict[str, Any]:
        """Get device memory information"""
        try:
            with cp.cuda.Device(self.device_id):
                mempool = cp.get_default_memory_pool()
                return {
                    'total_memory': cp.cuda.Device().mem_info[1],
                    'free_memory': cp.cuda.Device().mem_info[0],
                    'used_memory': mempool.used_bytes(),
                    'device_name': cp.cuda.Device().name.decode()
                }
        except:
            return {}
    
    def update_load_score(self):
        """Update device load score for load balancing"""
        if not self.available:
            self.load_score = float('inf')
            return
        
        try:
            with cp.cuda.Device(self.device_id):
                memory_info = self._get_memory_info()
                
                # Calculate load based on memory usage
                total_mem = memory_info.get('total_memory', 1)
                used_mem = memory_info.get('used_memory', 0)
                memory_load = used_mem / total_mem
                
                # Simple load score (lower is better)
                self.load_score = memory_load
                
        except Exception as e:
            self.load_score = float('inf')
            warnings.warn(f"Failed to update load score for GPU {self.device_id}: {e}")


class MultiGPUManager:
    """
    Multi-GPU manager for distributed AGI computation
    
    Features:
    - Automatic GPU detection and configuration
    - Dynamic load balancing
    - Fault tolerance with automatic failover
    - Memory optimization across devices
    - Synchronized operations
    """
    
    def __init__(self, auto_detect: bool = True, device_ids: Optional[List[int]] = None):
        self.devices: Dict[int, GPUDevice] = {}
        self.available_devices: List[int] = []
        self.load_balancer_enabled = True
        self.fault_tolerance_enabled = True
        
        # Thread pool for parallel operations
        self.executor = ThreadPoolExecutor(max_workers=8)
        
        # Initialize devices
        if auto_detect:
            self._auto_detect_devices()
        elif device_ids:
            self._initialize_specific_devices(device_ids)
        else:
            warnings.warn("No GPU devices specified or auto-detection disabled")
    
    def _auto_detect_devices(self):
        """Automatically detect and initialize available GPU devices"""
        if not GPU_AVAILABLE:
            print("CuPy not available - Multi-GPU manager will not be functional")
            return
        
        try:
            num_devices = cp.cuda.runtime.getDeviceCount()
            print(f"Detected {num_devices} GPU device(s)")
            
            for device_id in range(num_devices):
                device = GPUDevice(device_id)
                self.devices[device_id] = device
                
                if device.available:
                    self.available_devices.append(device_id)
            
            print(f"Successfully initialized {len(self.available_devices)} GPU(s): {self.available_devices}")
            
        except Exception as e:
            warnings.warn(f"Failed to detect GPU devices: {e}")
    
    def _initialize_specific_devices(self, device_ids: List[int]):
        """Initialize specific GPU devices"""
        for device_id in device_ids:
            device = GPUDevice(device_id)
            self.devices[device_id] = device
            
            if device.available:
                self.available_devices.append(device_id)
    
    def get_optimal_device(self, memory_requirement: Optional[int] = None) -> Optional[int]:
        """
        Get optimal device for next operation based on load balancing
        
        Args:
            memory_requirement: Required memory in bytes
            
        Returns:
            Device ID of optimal device, or None if none available
        """
        if not self.available_devices:
            return None
        
        if not self.load_balancer_enabled:
            return self.available_devices[0]
        
        # Update load scores
        for device_id in self.available_devices:
            self.devices[device_id].update_load_score()
        
        # Filter devices by memory requirement
        suitable_devices = []
        for device_id in self.available_devices:
            device = self.devices[device_id]
            if memory_requirement is None:
                suitable_devices.append(device_id)
            else:
                free_memory = device.memory_info.get('free_memory', 0)
                if free_memory >= memory_requirement:
                    suitable_devices.append(device_id)
        
        if not suitable_devices:
            return None
        
        # Select device with lowest load score
        best_device = min(suitable_devices, 
                         key=lambda d: self.devices[d].load_score)
        
        return best_device
    
    def distribute_computation(self, 
                             computation_func: Callable,
                             data_chunks: List[Any],
                             **kwargs) -> List[Any]:
        """
        Distribute computation across available GPUs
        
        Args:
            computation_func: Function to execute on each chunk
            data_chunks: List of data chunks to process
            **kwargs: Additional arguments for computation function
            
        Returns:
            List of results from each chunk
        """
        if not self.available_devices:
            raise RuntimeError("No GPU devices available for computation")
        
        # Distribute chunks across devices
        device_assignments = self._assign_chunks_to_devices(data_chunks)
        
        # Submit tasks to thread pool
        futures = []
        for device_id, chunks in device_assignments.items():
            future = self.executor.submit(
                self._execute_on_device,
                device_id, computation_func, chunks, **kwargs
            )
            futures.append(future)
        
        # Collect results
        all_results = []
        for future in as_completed(futures):
            try:
                results = future.result()
                all_results.extend(results)
            except Exception as e:
                if self.fault_tolerance_enabled:
                    warnings.warn(f"GPU computation failed: {e}, attempting fallback")
                    # TODO: Implement fallback mechanism
                else:
                    raise
        
        return all_results
    
    def _assign_chunks_to_devices(self, data_chunks: List[Any]) -> Dict[int, List[Any]]:
        """Assign data chunks to devices based on load balancing"""
        assignments = {device_id: [] for device_id in self.available_devices}
        
        # Simple round-robin assignment for now
        # TODO: Implement more sophisticated load-aware assignment
        for i, chunk in enumerate(data_chunks):
            device_id = self.available_devices[i % len(self.available_devices)]
            assignments[device_id].append(chunk)
        
        return assignments
    
    def _execute_on_device(self, 
                          device_id: int,
                          computation_func: Callable,
                          chunks: List[Any],
                          **kwargs) -> List[Any]:
        """Execute computation on specific device"""
        device = self.devices[device_id]
        if not device.available:
            raise RuntimeError(f"Device {device_id} not available")
        
        results = []
        
        try:
            with cp.cuda.Device(device_id):
                for chunk in chunks:
                    result = computation_func(chunk, device=device, **kwargs)
                    results.append(result)
        except Exception as e:
            if self.fault_tolerance_enabled:
                # Mark device as problematic and try fallback
                warnings.warn(f"Device {device_id} failed: {e}")
                # TODO: Implement device failover
            raise
        
        return results
    
    def parallel_matrix_operations(self,
                                 matrices_list: List[Tuple[ArrayType, ArrayType]],
                                 operation: str = 'matmul') -> List[ArrayType]:
        """
        Perform parallel matrix operations across GPUs
        
        Args:
            matrices_list: List of (matrix_a, matrix_b) tuples
            operation: Type of operation ('matmul', 'add', 'multiply')
            
        Returns:
            List of operation results
        """
        def matrix_op(matrices_pair, device, **kwargs):
            matrix_a, matrix_b = matrices_pair
            
            # Transfer to device if needed
            if hasattr(matrix_a, 'device') and matrix_a.device.id != device.device_id:
                with cp.cuda.Device(device.device_id):
                    matrix_a = cp.asarray(matrix_a)
                    matrix_b = cp.asarray(matrix_b)
            
            # Perform operation
            if operation == 'matmul':
                result = cp.matmul(matrix_a, matrix_b)
            elif operation == 'add':
                result = matrix_a + matrix_b
            elif operation == 'multiply':
                result = matrix_a * matrix_b
            else:
                raise ValueError(f"Unsupported operation: {operation}")
            
            return result
        
        return self.distribute_computation(matrix_op, matrices_list)
    
    def parallel_neuron_activation(self,
                                 input_batches: List[ArrayType],
                                 weights_batches: List[ArrayType],
                                 dependencies_batches: List[ArrayType],
                                 dep_counts_batches: List[ArrayType]) -> List[ArrayType]:
        """
        Perform parallel recursive neuron activation across GPUs
        
        Args:
            input_batches: List of input batches
            weights_batches: List of weight batches
            dependencies_batches: List of dependency batches
            dep_counts_batches: List of dependency count batches
            
        Returns:
            List of activation results
        """
        def neuron_activation(batch_data, device, **kwargs):
            inputs, weights, dependencies, dep_counts = batch_data
            
            # Use device's kernels for activation
            return device.kernels.recursive_neuron_activation(
                inputs, weights, dependencies, dep_counts
            )
        
        batch_data = list(zip(input_batches, weights_batches, 
                             dependencies_batches, dep_counts_batches))
        
        return self.distribute_computation(neuron_activation, batch_data)
    
    def parallel_attention_computation(self,
                                     queries_batches: List[ArrayType],
                                     keys_batches: List[ArrayType], 
                                     values_batches: List[ArrayType],
                                     num_heads: int = 8) -> List[Tuple[ArrayType, ArrayType]]:
        """
        Perform parallel attention computation across GPUs
        
        Args:
            queries_batches: List of query batches
            keys_batches: List of key batches
            values_batches: List of value batches
            num_heads: Number of attention heads
            
        Returns:
            List of (attention_output, attention_weights) tuples
        """
        def attention_computation(batch_data, device, **kwargs):
            queries, keys, values = batch_data
            
            # Use device's kernels for attention
            return device.kernels.multi_head_attention(
                queries, keys, values, num_heads
            )
        
        batch_data = list(zip(queries_batches, keys_batches, values_batches))
        
        return self.distribute_computation(attention_computation, batch_data)
    
    def synchronize_all_devices(self):
        """Synchronize all available GPU devices"""
        for device_id in self.available_devices:
            try:
                with cp.cuda.Device(device_id):
                    cp.cuda.Device().synchronize()
            except Exception as e:
                warnings.warn(f"Failed to synchronize device {device_id}: {e}")
    
    def get_device_status(self) -> Dict[str, Any]:
        """Get status of all devices"""
        status = {
            'total_devices': len(self.devices),
            'available_devices': len(self.available_devices),
            'device_details': {}
        }
        
        for device_id, device in self.devices.items():
            device.update_load_score()
            
            status['device_details'][device_id] = {
                'available': device.available,
                'load_score': device.load_score,
                'memory_info': device.memory_info,
                'kernels_compiled': device.kernels.kernels_compiled if device.kernels else False
            }
        
        return status
    
    def benchmark_multi_gpu_performance(self, matrix_sizes: List[int] = [500, 1000, 2000]) -> Dict[str, Any]:
        """Benchmark multi-GPU performance"""
        results = {
            'device_count': len(self.available_devices),
            'benchmarks': {}
        }
        
        for size in matrix_sizes:
            print(f"Benchmarking matrix size: {size}x{size}")
            
            # Generate test data
            num_operations = len(self.available_devices) * 4
            matrices_list = []
            
            for _ in range(num_operations):
                matrix_a = cp.random.randn(size, size).astype(cp.float32)
                matrix_b = cp.random.randn(size, size).astype(cp.float32)
                matrices_list.append((matrix_a, matrix_b))
            
            # Benchmark parallel execution
            start_time = time.time()
            results_parallel = self.parallel_matrix_operations(matrices_list, 'matmul')
            self.synchronize_all_devices()
            parallel_time = time.time() - start_time
            
            # Benchmark single-device execution for comparison
            if self.available_devices:
                device_id = self.available_devices[0]
                with cp.cuda.Device(device_id):
                    start_time = time.time()
                    for matrix_a, matrix_b in matrices_list:
                        _ = cp.matmul(matrix_a, matrix_b)
                    cp.cuda.Device().synchronize()
                    single_device_time = time.time() - start_time
                
                speedup = single_device_time / parallel_time if parallel_time > 0 else 0
            else:
                single_device_time = 0
                speedup = 0
            
            results['benchmarks'][f'size_{size}'] = {
                'operations': num_operations,
                'parallel_time_s': parallel_time,
                'single_device_time_s': single_device_time,
                'speedup': speedup,
                'operations_per_second': num_operations / parallel_time if parallel_time > 0 else 0
            }
            
            print(f"  Parallel: {parallel_time:.3f}s, Single: {single_device_time:.3f}s, Speedup: {speedup:.2f}x")
        
        return results
    
    def cleanup(self):
        """Cleanup all GPU resources"""
        print("Cleaning up multi-GPU resources...")
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Cleanup individual devices
        for device in self.devices.values():
            if device.kernels:
                device.kernels.cleanup()
        
        print("Multi-GPU cleanup completed")
    
    def __del__(self):
        """Destructor"""
        try:
            self.cleanup()
        except:
            pass