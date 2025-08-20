"""
AGI Tensor Operations - Core tensor functionality with AGI enhancements
"""

import numpy as np
from typing import Any, Optional, Tuple, Union

class Tensor:
    """AGI Tensor - like torch.Tensor with consciousness and causal tracking"""
    
    def __init__(self, data, dtype=None, device="cpu", requires_grad=False):
        if isinstance(data, (list, tuple)):
            self.data = np.array(data, dtype=dtype)
        elif isinstance(data, np.ndarray):
            self.data = data.astype(dtype) if dtype else data.copy()
        else:
            self.data = np.array([data], dtype=dtype)
        
        self.dtype = self.data.dtype
        self.device = device
        self.requires_grad = requires_grad
        self.grad = None
        self.shape = self.data.shape
        self.ndim = self.data.ndim
        
        # AGI enhancements
        self._consciousness_attention = np.ones_like(self.data) * 0.1
        self._causal_connections = {}
        self._backward_fn = None
    
    def __repr__(self):
        return f"tensor({self.data.tolist()}, dtype={self.dtype}, device='{self.device}', requires_grad={self.requires_grad})"
    
    def __str__(self):
        return f"tensor({self.data})"
    
    # Mathematical operations
    def __add__(self, other):
        if isinstance(other, Tensor):
            result_data = self.data + other.data
            requires_grad = self.requires_grad or other.requires_grad
        else:
            result_data = self.data + other
            requires_grad = self.requires_grad
        
        result = Tensor(result_data, requires_grad=requires_grad, device=self.device)
        # AGI enhancement: combine consciousness attention
        if isinstance(other, Tensor):
            result._consciousness_attention = (self._consciousness_attention + other._consciousness_attention) / 2
        return result
    
    def __mul__(self, other):
        if isinstance(other, Tensor):
            result_data = self.data * other.data
            requires_grad = self.requires_grad or other.requires_grad
        else:
            result_data = self.data * other
            requires_grad = self.requires_grad
        
        result = Tensor(result_data, requires_grad=requires_grad, device=self.device)
        if isinstance(other, Tensor):
            result._consciousness_attention = np.maximum(self._consciousness_attention, other._consciousness_attention)
        return result
    
    def __sub__(self, other):
        if isinstance(other, Tensor):
            result_data = self.data - other.data
            requires_grad = self.requires_grad or other.requires_grad
        else:
            result_data = self.data - other
            requires_grad = self.requires_grad
        
        return Tensor(result_data, requires_grad=requires_grad, device=self.device)
    
    def __truediv__(self, other):
        if isinstance(other, Tensor):
            result_data = self.data / other.data
            requires_grad = self.requires_grad or other.requires_grad
        else:
            result_data = self.data / other
            requires_grad = self.requires_grad
        
        return Tensor(result_data, requires_grad=requires_grad, device=self.device)
    
    def __matmul__(self, other):
        if isinstance(other, Tensor):
            result_data = np.matmul(self.data, other.data)
            requires_grad = self.requires_grad or other.requires_grad
        else:
            result_data = np.matmul(self.data, other)
            requires_grad = self.requires_grad
        
        result = Tensor(result_data, requires_grad=requires_grad, device=self.device)
        # AGI enhancement: consciousness flows through matrix multiplication
        if isinstance(other, Tensor):
            result._consciousness_attention = np.ones_like(result.data) * np.mean(
                [np.mean(self._consciousness_attention), np.mean(other._consciousness_attention)]
            )
        return result
    
    def __getitem__(self, key):
        result = Tensor(self.data[key], requires_grad=self.requires_grad, device=self.device)
        if hasattr(self._consciousness_attention, '__getitem__'):
            result._consciousness_attention = self._consciousness_attention[key]
        return result
    
    # Tensor operations
    def view(self, *shape):
        """Reshape tensor - like torch.view"""
        return Tensor(self.data.reshape(shape), requires_grad=self.requires_grad, device=self.device)
    
    def reshape(self, *shape):
        """Reshape tensor"""
        return Tensor(self.data.reshape(shape), requires_grad=self.requires_grad, device=self.device)
    
    def transpose(self, dim0=0, dim1=1):
        """Transpose tensor"""
        axes = list(range(self.data.ndim))
        axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
        result_data = np.transpose(self.data, axes)
        return Tensor(result_data, requires_grad=self.requires_grad, device=self.device)
    
    def permute(self, *dims):
        """Permute tensor dimensions"""
        result_data = np.transpose(self.data, dims)
        return Tensor(result_data, requires_grad=self.requires_grad, device=self.device)
    
    def unsqueeze(self, dim):
        """Add dimension"""
        result_data = np.expand_dims(self.data, dim)
        return Tensor(result_data, requires_grad=self.requires_grad, device=self.device)
    
    def squeeze(self, dim=None):
        """Remove dimension"""
        result_data = np.squeeze(self.data, dim)
        return Tensor(result_data, requires_grad=self.requires_grad, device=self.device)
    
    def sum(self, dim=None, keepdim=False):
        """Sum along dimension"""
        result_data = np.sum(self.data, axis=dim, keepdims=keepdim)
        return Tensor(result_data, requires_grad=self.requires_grad, device=self.device)
    
    def mean(self, dim=None, keepdim=False):
        """Mean along dimension"""
        result_data = np.mean(self.data, axis=dim, keepdims=keepdim)
        return Tensor(result_data, requires_grad=self.requires_grad, device=self.device)
    
    def max(self, dim=None, keepdim=False):
        """Maximum along dimension"""
        if dim is None:
            result_data = np.max(self.data)
            return Tensor(result_data, requires_grad=self.requires_grad, device=self.device)
        else:
            values = np.max(self.data, axis=dim, keepdims=keepdim)
            indices = np.argmax(self.data, axis=dim, keepdims=keepdim)
            return (Tensor(values, requires_grad=self.requires_grad, device=self.device),
                   Tensor(indices, device=self.device))
    
    def min(self, dim=None, keepdim=False):
        """Minimum along dimension"""
        if dim is None:
            result_data = np.min(self.data)
            return Tensor(result_data, requires_grad=self.requires_grad, device=self.device)
        else:
            values = np.min(self.data, axis=dim, keepdims=keepdim)
            indices = np.argmin(self.data, axis=dim, keepdims=keepdim)
            return (Tensor(values, requires_grad=self.requires_grad, device=self.device),
                   Tensor(indices, device=self.device))
    
    def backward(self, gradient=None):
        """AGI-enhanced backward pass"""
        if not self.requires_grad:
            return
        
        if gradient is None:
            gradient = np.ones_like(self.data)
        
        if self.grad is None:
            self.grad = Tensor(np.zeros_like(self.data))
        
        self.grad.data += gradient
        
        # AGI enhancement: consciousness-guided gradient scaling
        consciousness_boost = 1.0 + np.mean(self._consciousness_attention) * 0.1
        self.grad.data *= consciousness_boost
        
        # Call backward function if available
        if self._backward_fn is not None:
            self._backward_fn(gradient)
    
    def detach(self):
        """Detach from computation graph"""
        return Tensor(self.data.copy(), requires_grad=False, device=self.device)
    
    def clone(self):
        """Clone tensor"""
        result = Tensor(self.data.copy(), requires_grad=self.requires_grad, device=self.device)
        result._consciousness_attention = self._consciousness_attention.copy()
        return result
    
    def to(self, device):
        """Move to device"""
        result = Tensor(self.data.copy(), requires_grad=self.requires_grad, device=device)
        if device == "cuda":
            # AGI enhancement: CUDA consciousness boost
            result._consciousness_attention = self._consciousness_attention * 1.1
        else:
            result._consciousness_attention = self._consciousness_attention.copy()
        return result
    
    def cuda(self):
        """Move to CUDA"""
        return self.to("cuda")
    
    def cpu(self):
        """Move to CPU"""
        return self.to("cpu")
    
    def numpy(self):
        """Convert to numpy array"""
        return self.data.copy()
    
    def item(self):
        """Get scalar value"""
        return self.data.item()
    
    def size(self, dim=None):
        """Get size"""
        if dim is None:
            return self.shape
        return self.shape[dim]

# Tensor creation functions
def tensor(data, dtype=None, device="cpu", requires_grad=False):
    """Create tensor"""
    return Tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)

def zeros(*size, dtype=None, device="cpu", requires_grad=False):
    """Create zero tensor"""
    return Tensor(np.zeros(size), dtype=dtype, device=device, requires_grad=requires_grad)

def ones(*size, dtype=None, device="cpu", requires_grad=False):
    """Create ones tensor"""  
    return Tensor(np.ones(size), dtype=dtype, device=device, requires_grad=requires_grad)

def randn(*size, dtype=None, device="cpu", requires_grad=False):
    """Create random normal tensor"""
    return Tensor(np.random.randn(*size), dtype=dtype, device=device, requires_grad=requires_grad)

def rand(*size, dtype=None, device="cpu", requires_grad=False):
    """Create random uniform tensor"""
    return Tensor(np.random.rand(*size), dtype=dtype, device=device, requires_grad=requires_grad)

def randint(low, high=None, size=(), dtype=None, device="cpu", requires_grad=False):
    """Create random integer tensor"""
    if high is None:
        high = low
        low = 0
    if isinstance(size, int):
        size = (size,)
    return Tensor(np.random.randint(low, high, size=size), dtype=dtype, device=device, requires_grad=requires_grad)

def arange(start, end=None, step=1, dtype=None, device="cpu", requires_grad=False):
    """Create range tensor"""
    if end is None:
        end = start
        start = 0
    return Tensor(np.arange(start, end, step), dtype=dtype, device=device, requires_grad=requires_grad)

def linspace(start, end, steps, dtype=None, device="cpu", requires_grad=False):
    """Create linspace tensor"""
    return Tensor(np.linspace(start, end, steps), dtype=dtype, device=device, requires_grad=requires_grad)

# Export all functions and classes
__all__ = [
    'Tensor', 'tensor', 'zeros', 'ones', 'randn', 'rand', 'randint', 
    'arange', 'linspace'
]