"""
Advanced GPU Kernels for AGI-Formula

Custom CUDA implementations for high-performance AGI operations:
- Recursive neuron activation
- Multi-head attention mechanisms  
- Causal reasoning computation
- Batch processing optimization
"""

import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple, Union
import warnings

# GPU computing imports
try:
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix
    GPU_AVAILABLE = True
    ArrayType = cp.ndarray
except ImportError:
    import numpy as np
    cp = None
    GPU_AVAILABLE = False
    ArrayType = np.ndarray  # Fallback to numpy for type annotations

# Try to import CUDA compilation tools
try:
    from cupy import RawKernel, RawModule
    CUDA_COMPILATION_AVAILABLE = True
except ImportError:
    CUDA_COMPILATION_AVAILABLE = False
    RawKernel = None
    RawModule = None


class AdvancedGPUKernels:
    """
    Advanced GPU kernel implementations for AGI operations
    
    Provides optimized CUDA kernels for:
    - Recursive neuron activation with dynamic programming
    - Multi-head attention with causal masking
    - Efficient causal reasoning computation
    - Batched operations with memory optimization
    """
    
    def __init__(self, device_id: int = 0, enable_compilation: bool = True):
        self.device_id = device_id
        self.enable_compilation = enable_compilation and CUDA_COMPILATION_AVAILABLE
        self.kernels_compiled = False
        self.kernel_cache = {}
        
        if not GPU_AVAILABLE:
            warnings.warn("CuPy not available. GPU kernels will not be functional.")
            return
        
        # Set device
        cp.cuda.Device(device_id).use()
        
        # Initialize memory pools
        self._setup_memory_pools()
        
        # Compile kernels if possible
        if self.enable_compilation:
            self._compile_kernels()
    
    def _setup_memory_pools(self):
        """Setup optimized memory pools"""
        try:
            # Configure memory pool for optimal performance
            mempool = cp.get_default_memory_pool()
            mempool.set_limit(size=None)  # Use all available GPU memory
            
            # Setup pinned memory pool for faster CPU-GPU transfers
            pinned_mempool = cp.get_default_pinned_memory_pool()
            pinned_mempool.set_limit(size=1024**3)  # 1GB pinned memory
            
            print(f"GPU memory pools configured on device {self.device_id}")
            
        except Exception as e:
            warnings.warn(f"Failed to setup memory pools: {e}")
    
    def _compile_kernels(self):
        """Compile custom CUDA kernels"""
        if not self.enable_compilation:
            return
        
        try:
            # Recursive neuron activation kernel
            self._compile_recursive_activation_kernel()
            
            # Multi-head attention kernel  
            self._compile_attention_kernel()
            
            # Causal reasoning kernel
            self._compile_causal_reasoning_kernel()
            
            # Batch processing kernels
            self._compile_batch_kernels()
            
            self.kernels_compiled = True
            print("Advanced GPU kernels compiled successfully")
            
        except Exception as e:
            warnings.warn(f"Failed to compile kernels: {e}")
            self.enable_compilation = False
    
    def _compile_recursive_activation_kernel(self):
        """Compile recursive neuron activation kernel"""
        kernel_code = '''
        extern "C" __global__
        void recursive_activation_kernel(
            const float* inputs,
            const float* weights,
            const int* dependencies, 
            const int* dep_counts,
            float* outputs,
            float* cache,
            bool* computed,
            int num_neurons,
            int max_deps
        ) {
            int neuron_id = blockIdx.x * blockDim.x + threadIdx.x;
            
            if (neuron_id >= num_neurons || computed[neuron_id]) {
                return;
            }
            
            float activation = 0.0f;
            
            // Add direct input contribution
            activation += inputs[neuron_id];
            
            // Add contributions from dependencies
            int dep_start = neuron_id * max_deps;
            for (int i = 0; i < dep_counts[neuron_id]; i++) {
                int dep_id = dependencies[dep_start + i];
                if (dep_id < num_neurons && computed[dep_id]) {
                    activation += cache[dep_id] * weights[dep_start + i];
                }
            }
            
            // Apply activation function (sigmoid)
            activation = 1.0f / (1.0f + expf(-activation));
            
            // Store result
            cache[neuron_id] = activation;
            outputs[neuron_id] = activation;
            computed[neuron_id] = true;
        }
        '''
        
        if RawKernel is not None:
            self.kernel_cache['recursive_activation'] = RawKernel(
                kernel_code, 'recursive_activation_kernel'
            )
    
    def _compile_attention_kernel(self):
        """Compile multi-head attention kernel"""
        kernel_code = '''
        extern "C" __global__
        void multi_head_attention_kernel(
            const float* queries,
            const float* keys, 
            const float* values,
            float* attention_weights,
            float* output,
            int batch_size,
            int seq_len,
            int num_heads,
            int head_dim,
            float scale
        ) {
            int batch_idx = blockIdx.x;
            int head_idx = blockIdx.y;
            int query_idx = threadIdx.x;
            
            if (batch_idx >= batch_size || head_idx >= num_heads || query_idx >= seq_len) {
                return;
            }
            
            // Calculate attention scores
            float max_score = -INFINITY;
            int offset = (batch_idx * num_heads + head_idx) * seq_len * head_dim;
            int query_offset = offset + query_idx * head_dim;
            
            // Find maximum for numerical stability
            for (int key_idx = 0; key_idx < seq_len; key_idx++) {
                int key_offset = offset + key_idx * head_dim;
                float score = 0.0f;
                
                for (int d = 0; d < head_dim; d++) {
                    score += queries[query_offset + d] * keys[key_offset + d];
                }
                score *= scale;
                
                if (score > max_score) {
                    max_score = score;
                }
            }
            
            // Compute softmax
            float sum_exp = 0.0f;
            int attn_offset = (batch_idx * num_heads + head_idx) * seq_len * seq_len + query_idx * seq_len;
            
            for (int key_idx = 0; key_idx < seq_len; key_idx++) {
                int key_offset = offset + key_idx * head_dim;
                float score = 0.0f;
                
                for (int d = 0; d < head_dim; d++) {
                    score += queries[query_offset + d] * keys[key_offset + d];
                }
                score = expf((score * scale) - max_score);
                attention_weights[attn_offset + key_idx] = score;
                sum_exp += score;
            }
            
            // Normalize and compute output
            for (int key_idx = 0; key_idx < seq_len; key_idx++) {
                attention_weights[attn_offset + key_idx] /= sum_exp;
            }
            
            // Compute attended values
            int output_offset = query_offset;
            for (int d = 0; d < head_dim; d++) {
                float attended_value = 0.0f;
                for (int key_idx = 0; key_idx < seq_len; key_idx++) {
                    int value_offset = offset + key_idx * head_dim;
                    attended_value += attention_weights[attn_offset + key_idx] * values[value_offset + d];
                }
                output[output_offset + d] = attended_value;
            }
        }
        '''
        
        if RawKernel is not None:
            self.kernel_cache['multi_head_attention'] = RawKernel(
                kernel_code, 'multi_head_attention_kernel'
            )
    
    def _compile_causal_reasoning_kernel(self):
        """Compile causal reasoning computation kernel"""
        kernel_code = '''
        extern "C" __global__
        void causal_reasoning_kernel(
            const float* activations,
            const float* contributions,
            const int* causal_graph,
            float* causal_strengths,
            float* mutual_info,
            int num_neurons,
            int max_connections
        ) {
            int neuron_id = blockIdx.x * blockDim.x + threadIdx.x;
            
            if (neuron_id >= num_neurons) {
                return;
            }
            
            float total_contribution = 0.0f;
            int graph_offset = neuron_id * max_connections;
            
            // Calculate causal contributions
            for (int i = 0; i < max_connections; i++) {
                int cause_id = causal_graph[graph_offset + i];
                if (cause_id >= 0 && cause_id < num_neurons) {
                    float contribution = contributions[graph_offset + i];
                    float cause_activation = activations[cause_id];
                    float effect_activation = activations[neuron_id];
                    
                    // Compute causal strength
                    float causal_strength = contribution * cause_activation * effect_activation;
                    causal_strengths[graph_offset + i] = causal_strength;
                    total_contribution += fabsf(causal_strength);
                    
                    // Compute mutual information approximation
                    if (cause_activation > 0.01f && effect_activation > 0.01f) {
                        float mi = logf(causal_strength + 1e-8f) - 
                                  logf(cause_activation + 1e-8f) - 
                                  logf(effect_activation + 1e-8f);
                        mutual_info[graph_offset + i] = fmaxf(0.0f, mi);
                    }
                }
            }
            
            // Normalize causal strengths
            if (total_contribution > 1e-8f) {
                for (int i = 0; i < max_connections; i++) {
                    causal_strengths[graph_offset + i] /= total_contribution;
                }
            }
        }
        '''
        
        if RawKernel is not None:
            self.kernel_cache['causal_reasoning'] = RawKernel(
                kernel_code, 'causal_reasoning_kernel'
            )
    
    def _compile_batch_kernels(self):
        """Compile batch processing optimization kernels"""
        # Batched matrix multiplication kernel
        matmul_kernel = '''
        extern "C" __global__
        void batched_matmul_kernel(
            const float* a,
            const float* b, 
            float* c,
            int batch_size,
            int m, int n, int k
        ) {
            int batch_idx = blockIdx.z;
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;
            
            if (batch_idx >= batch_size || row >= m || col >= n) {
                return;
            }
            
            float sum = 0.0f;
            int a_offset = batch_idx * m * k;
            int b_offset = batch_idx * k * n;
            int c_offset = batch_idx * m * n;
            
            for (int i = 0; i < k; i++) {
                sum += a[a_offset + row * k + i] * b[b_offset + i * n + col];
            }
            
            c[c_offset + row * n + col] = sum;
        }
        '''
        
        if RawKernel is not None:
            self.kernel_cache['batched_matmul'] = RawKernel(
                matmul_kernel, 'batched_matmul_kernel'
            )
        
        # Fused activation kernel
        fused_activation_kernel = '''
        extern "C" __global__
        void fused_activation_kernel(
            const float* input,
            float* output,
            int size,
            int activation_type
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            
            if (idx >= size) {
                return;
            }
            
            float x = input[idx];
            float result;
            
            switch (activation_type) {
                case 0: // Sigmoid
                    result = 1.0f / (1.0f + expf(-x));
                    break;
                case 1: // Tanh
                    result = tanhf(x);
                    break;
                case 2: // ReLU
                    result = fmaxf(0.0f, x);
                    break;
                case 3: // GELU
                    result = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
                    break;
                default:
                    result = x;
            }
            
            output[idx] = result;
        }
        '''
        
        if RawKernel is not None:
            self.kernel_cache['fused_activation'] = RawKernel(
                fused_activation_kernel, 'fused_activation_kernel'
            )
    
    def recursive_neuron_activation(self, 
                                  inputs: ArrayType,
                                  weights: ArrayType, 
                                  dependencies: ArrayType,
                                  dep_counts: ArrayType) -> ArrayType:
        """
        Optimized recursive neuron activation using custom CUDA kernel
        
        Args:
            inputs: Input activations [num_neurons]
            weights: Connection weights [num_neurons, max_deps] 
            dependencies: Dependency indices [num_neurons, max_deps]
            dep_counts: Number of dependencies per neuron [num_neurons]
            
        Returns:
            Final neuron activations [num_neurons]
        """
        if not GPU_AVAILABLE:
            raise RuntimeError("GPU not available for kernel execution")
        
        num_neurons = inputs.shape[0]
        max_deps = dependencies.shape[1]
        
        # Allocate output arrays
        outputs = cp.zeros(num_neurons, dtype=cp.float32)
        cache = cp.zeros(num_neurons, dtype=cp.float32)
        computed = cp.zeros(num_neurons, dtype=cp.bool_)
        
        if self.kernels_compiled and 'recursive_activation' in self.kernel_cache:
            # Use compiled kernel
            kernel = self.kernel_cache['recursive_activation']
            
            # Configure kernel launch
            block_size = 256
            grid_size = (num_neurons + block_size - 1) // block_size
            
            # Multiple passes for recursive dependencies
            for iteration in range(10):  # Max 10 iterations
                kernel(
                    (grid_size,), (block_size,),
                    (inputs, weights, dependencies, dep_counts,
                     outputs, cache, computed, num_neurons, max_deps)
                )
                cp.cuda.Device().synchronize()
                
                # Check convergence
                if cp.all(computed):
                    break
        else:
            # Fallback to CuPy operations
            outputs = self._recursive_activation_fallback(
                inputs, weights, dependencies, dep_counts
            )
        
        return outputs
    
    def multi_head_attention(self,
                           queries: ArrayType,
                           keys: ArrayType, 
                           values: ArrayType,
                           num_heads: int = 8) -> Tuple[ArrayType, ArrayType]:
        """
        Optimized multi-head attention using custom CUDA kernel
        
        Args:
            queries: Query tensor [batch_size, seq_len, model_dim]
            keys: Key tensor [batch_size, seq_len, model_dim]
            values: Value tensor [batch_size, seq_len, model_dim]
            num_heads: Number of attention heads
            
        Returns:
            (attention_output, attention_weights)
        """
        if not GPU_AVAILABLE:
            raise RuntimeError("GPU not available for kernel execution")
        
        batch_size, seq_len, model_dim = queries.shape
        head_dim = model_dim // num_heads
        scale = 1.0 / np.sqrt(head_dim)
        
        # Reshape for multi-head processing
        q_reshaped = queries.reshape(batch_size, seq_len, num_heads, head_dim)
        k_reshaped = keys.reshape(batch_size, seq_len, num_heads, head_dim)
        v_reshaped = values.reshape(batch_size, seq_len, num_heads, head_dim)
        
        # Allocate output arrays
        attention_weights = cp.zeros((batch_size, num_heads, seq_len, seq_len), dtype=cp.float32)
        output = cp.zeros_like(q_reshaped)
        
        if self.kernels_compiled and 'multi_head_attention' in self.kernel_cache:
            # Use compiled kernel
            kernel = self.kernel_cache['multi_head_attention']
            
            # Configure kernel launch
            block_size = min(seq_len, 256)
            grid_size = (batch_size, num_heads)
            
            kernel(
                grid_size, (block_size,),
                (q_reshaped, k_reshaped, v_reshaped,
                 attention_weights, output,
                 batch_size, seq_len, num_heads, head_dim, scale)
            )
            cp.cuda.Device().synchronize()
        else:
            # Fallback to CuPy operations
            output, attention_weights = self._attention_fallback(
                q_reshaped, k_reshaped, v_reshaped, scale
            )
        
        # Reshape output
        output = output.reshape(batch_size, seq_len, model_dim)
        
        return output, attention_weights
    
    def causal_reasoning_computation(self,
                                   activations: ArrayType,
                                   contributions: ArrayType,
                                   causal_graph: ArrayType) -> Tuple[ArrayType, ArrayType]:
        """
        Optimized causal reasoning computation using custom CUDA kernel
        
        Args:
            activations: Neuron activations [num_neurons]
            contributions: Contribution weights [num_neurons, max_connections]
            causal_graph: Causal connection graph [num_neurons, max_connections]
            
        Returns:
            (causal_strengths, mutual_information)
        """
        if not GPU_AVAILABLE:
            raise RuntimeError("GPU not available for kernel execution")
        
        num_neurons = activations.shape[0]
        max_connections = causal_graph.shape[1]
        
        # Allocate output arrays
        causal_strengths = cp.zeros_like(contributions)
        mutual_info = cp.zeros_like(contributions)
        
        if self.kernels_compiled and 'causal_reasoning' in self.kernel_cache:
            # Use compiled kernel
            kernel = self.kernel_cache['causal_reasoning']
            
            # Configure kernel launch
            block_size = 256
            grid_size = (num_neurons + block_size - 1) // block_size
            
            kernel(
                (grid_size,), (block_size,),
                (activations, contributions, causal_graph,
                 causal_strengths, mutual_info,
                 num_neurons, max_connections)
            )
            cp.cuda.Device().synchronize()
        else:
            # Fallback to CuPy operations
            causal_strengths, mutual_info = self._causal_reasoning_fallback(
                activations, contributions, causal_graph
            )
        
        return causal_strengths, mutual_info
    
    def batched_operations(self,
                          matrices_a: ArrayType,
                          matrices_b: ArrayType,
                          activation_type: int = 0) -> ArrayType:
        """
        Optimized batched matrix operations with fused activation
        
        Args:
            matrices_a: Batch of matrices A [batch_size, m, k]
            matrices_b: Batch of matrices B [batch_size, k, n]
            activation_type: 0=sigmoid, 1=tanh, 2=relu, 3=gelu
            
        Returns:
            Batch of results with activation applied [batch_size, m, n]
        """
        if not GPU_AVAILABLE:
            raise RuntimeError("GPU not available for kernel execution")
        
        batch_size, m, k = matrices_a.shape
        _, k2, n = matrices_b.shape
        
        if k != k2:
            raise ValueError(f"Matrix dimension mismatch: {k} vs {k2}")
        
        # Allocate output
        matmul_result = cp.zeros((batch_size, m, n), dtype=cp.float32)
        
        if self.kernels_compiled and 'batched_matmul' in self.kernel_cache:
            # Use compiled matmul kernel
            matmul_kernel = self.kernel_cache['batched_matmul']
            
            # Configure kernel launch
            block_size_x = min(16, n)
            block_size_y = min(16, m)
            grid_size_x = (n + block_size_x - 1) // block_size_x
            grid_size_y = (m + block_size_y - 1) // block_size_y
            
            matmul_kernel(
                (grid_size_x, grid_size_y, batch_size),
                (block_size_x, block_size_y),
                (matrices_a, matrices_b, matmul_result,
                 batch_size, m, n, k)
            )
            cp.cuda.Device().synchronize()
        else:
            # Fallback to CuPy batched matmul
            matmul_result = cp.matmul(matrices_a, matrices_b)
        
        # Apply activation function
        if self.kernels_compiled and 'fused_activation' in self.kernel_cache:
            activation_kernel = self.kernel_cache['fused_activation']
            
            total_size = matmul_result.size
            block_size = 256
            grid_size = (total_size + block_size - 1) // block_size
            
            activation_kernel(
                (grid_size,), (block_size,),
                (matmul_result, matmul_result, total_size, activation_type)
            )
            cp.cuda.Device().synchronize()
        else:
            # Fallback activation
            if activation_type == 0:  # Sigmoid
                matmul_result = 1.0 / (1.0 + cp.exp(-matmul_result))
            elif activation_type == 1:  # Tanh
                matmul_result = cp.tanh(matmul_result)
            elif activation_type == 2:  # ReLU
                matmul_result = cp.maximum(0, matmul_result)
            elif activation_type == 3:  # GELU
                matmul_result = 0.5 * matmul_result * (1 + cp.tanh(0.7978845608 * (matmul_result + 0.044715 * matmul_result**3)))
        
        return matmul_result
    
    # Fallback implementations using CuPy
    def _recursive_activation_fallback(self, inputs, weights, dependencies, dep_counts):
        """Fallback recursive activation using CuPy operations"""
        num_neurons = inputs.shape[0]
        outputs = cp.copy(inputs)
        
        # Simple iterative approach
        for _ in range(5):  # Max iterations
            new_outputs = cp.copy(inputs)
            
            for i in range(num_neurons):
                deps = dependencies[i, :dep_counts[i]]
                valid_deps = deps[deps < num_neurons]
                if len(valid_deps) > 0:
                    dep_contribution = cp.sum(outputs[valid_deps] * weights[i, :len(valid_deps)])
                    new_outputs[i] += dep_contribution
            
            # Apply sigmoid activation
            new_outputs = 1.0 / (1.0 + cp.exp(-new_outputs))
            
            # Check convergence
            if cp.allclose(outputs, new_outputs, atol=1e-6):
                break
            
            outputs = new_outputs
        
        return outputs
    
    def _attention_fallback(self, queries, keys, values, scale):
        """Fallback attention using CuPy operations"""
        # Standard multi-head attention computation
        scores = cp.matmul(queries, keys.transpose(0, 2, 1, 3)) * scale
        attention_weights = cp.softmax(scores, axis=-1)
        output = cp.matmul(attention_weights, values)
        
        return output, attention_weights
    
    def _causal_reasoning_fallback(self, activations, contributions, causal_graph):
        """Fallback causal reasoning using CuPy operations"""
        num_neurons, max_connections = causal_graph.shape
        
        causal_strengths = cp.zeros_like(contributions)
        mutual_info = cp.zeros_like(contributions)
        
        for i in range(num_neurons):
            for j in range(max_connections):
                cause_id = causal_graph[i, j]
                if 0 <= cause_id < num_neurons:
                    contribution = contributions[i, j]
                    cause_act = activations[cause_id]
                    effect_act = activations[i]
                    
                    # Causal strength
                    causal_strengths[i, j] = contribution * cause_act * effect_act
                    
                    # Mutual information approximation
                    if cause_act > 0.01 and effect_act > 0.01:
                        mi = cp.log(causal_strengths[i, j] + 1e-8) - cp.log(cause_act + 1e-8) - cp.log(effect_act + 1e-8)
                        mutual_info[i, j] = cp.maximum(0, mi)
        
        return causal_strengths, mutual_info
    
    def benchmark_kernels(self, sizes: List[int] = [100, 500, 1000]) -> Dict[str, Any]:
        """Benchmark kernel performance"""
        results = {
            'kernel_compilation': self.kernels_compiled,
            'device_id': self.device_id,
            'benchmarks': {}
        }
        
        if not GPU_AVAILABLE:
            # Return dummy results when GPU not available
            for size in sizes:
                results['benchmarks'][f'size_{size}'] = {
                    'recursive_activation_ms': 0.0,
                    'attention_ms': 0.0,
                    'batched_ops_ms': 0.0,
                    'note': 'GPU not available - dummy results'
                }
            return results
        
        for size in sizes:
            size_results = {}
            
            # Benchmark recursive activation
            inputs = cp.random.randn(size).astype(cp.float32)
            weights = cp.random.randn(size, 10).astype(cp.float32)
            dependencies = cp.random.randint(0, size, (size, 10)).astype(cp.int32)
            dep_counts = cp.random.randint(1, 11, size).astype(cp.int32)
            
            start_time = time.time()
            _ = self.recursive_neuron_activation(inputs, weights, dependencies, dep_counts)
            cp.cuda.Device().synchronize()
            size_results['recursive_activation_ms'] = (time.time() - start_time) * 1000
            
            # Benchmark attention
            batch_size = min(8, max(1, 1000 // size))
            seq_len = min(size, 128)
            model_dim = 64
            
            queries = cp.random.randn(batch_size, seq_len, model_dim).astype(cp.float32)
            keys = cp.random.randn(batch_size, seq_len, model_dim).astype(cp.float32)
            values = cp.random.randn(batch_size, seq_len, model_dim).astype(cp.float32)
            
            start_time = time.time()
            _ = self.multi_head_attention(queries, keys, values)
            cp.cuda.Device().synchronize()
            size_results['attention_ms'] = (time.time() - start_time) * 1000
            
            # Benchmark batched operations
            batch_size = min(16, max(1, 2000 // size))
            matrices_a = cp.random.randn(batch_size, size//2, size//2).astype(cp.float32)
            matrices_b = cp.random.randn(batch_size, size//2, size//2).astype(cp.float32)
            
            start_time = time.time()
            _ = self.batched_operations(matrices_a, matrices_b)
            cp.cuda.Device().synchronize()
            size_results['batched_ops_ms'] = (time.time() - start_time) * 1000
            
            results['benchmarks'][f'size_{size}'] = size_results
        
        return results
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current GPU memory usage"""
        if not GPU_AVAILABLE:
            return {
                'device_id': self.device_id,
                'gpu_available': False,
                'note': 'GPU not available'
            }
        
        try:
            mempool = cp.get_default_memory_pool()
            
            return {
                'device_id': self.device_id,
                'used_bytes': mempool.used_bytes(),
                'total_bytes': mempool.total_bytes(),
                'used_mb': mempool.used_bytes() / (1024**2),
                'total_mb': mempool.total_bytes() / (1024**2),
                'utilization': mempool.used_bytes() / max(mempool.total_bytes(), 1),
                'gpu_available': True
            }
        except Exception as e:
            return {'error': str(e)}
    
    def cleanup(self):
        """Cleanup GPU resources"""
        if GPU_AVAILABLE:
            try:
                mempool = cp.get_default_memory_pool()
                mempool.free_all_blocks()
                print(f"GPU resources cleaned up on device {self.device_id}")
            except:
                pass