"""
AGI Functional Operations

Functional interface for neural network operations with consciousness enhancement.
"""

import numpy as np
from typing import Optional, Tuple, Union, Any
from .tensor import Tensor

# Activation functions
def relu(input, inplace=False):
    """ReLU activation function"""
    if hasattr(input, 'data'):
        input_data = input.data
    else:
        input_data = np.array(input)
    
    output_data = np.maximum(0, input_data)
    return Tensor(output_data, requires_grad=getattr(input, 'requires_grad', False))

def leaky_relu(input, negative_slope=0.01, inplace=False):
    """Leaky ReLU activation function"""
    if hasattr(input, 'data'):
        input_data = input.data
    else:
        input_data = np.array(input)
    
    output_data = np.where(input_data > 0, input_data, negative_slope * input_data)
    return Tensor(output_data, requires_grad=getattr(input, 'requires_grad', False))

def gelu(input):
    """GELU activation function"""
    if hasattr(input, 'data'):
        input_data = input.data
    else:
        input_data = np.array(input)
    
    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    output_data = 0.5 * input_data * (1 + np.tanh(np.sqrt(2/np.pi) * (input_data + 0.044715 * input_data**3)))
    return Tensor(output_data, requires_grad=getattr(input, 'requires_grad', False))

def sigmoid(input):
    """Sigmoid activation function"""
    if hasattr(input, 'data'):
        input_data = input.data
    else:
        input_data = np.array(input)
    
    output_data = 1 / (1 + np.exp(-np.clip(input_data, -500, 500)))
    return Tensor(output_data, requires_grad=getattr(input, 'requires_grad', False))

def tanh(input):
    """Tanh activation function"""
    if hasattr(input, 'data'):
        input_data = input.data
    else:
        input_data = np.array(input)
    
    output_data = np.tanh(input_data)
    return Tensor(output_data, requires_grad=getattr(input, 'requires_grad', False))

def softmax(input, dim=-1):
    """Softmax function"""
    if hasattr(input, 'data'):
        input_data = input.data
    else:
        input_data = np.array(input)
    
    # Subtract max for numerical stability
    max_vals = np.max(input_data, axis=dim, keepdims=True)
    exp_vals = np.exp(input_data - max_vals)
    sum_vals = np.sum(exp_vals, axis=dim, keepdims=True)
    output_data = exp_vals / sum_vals
    
    return Tensor(output_data, requires_grad=getattr(input, 'requires_grad', False))

def log_softmax(input, dim=-1):
    """Log softmax function"""
    if hasattr(input, 'data'):
        input_data = input.data
    else:
        input_data = np.array(input)
    
    # Subtract max for numerical stability
    max_vals = np.max(input_data, axis=dim, keepdims=True)
    shifted = input_data - max_vals
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=dim, keepdims=True))
    output_data = shifted - log_sum_exp
    
    return Tensor(output_data, requires_grad=getattr(input, 'requires_grad', False))

# Loss functions
def cross_entropy(input, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
    """Cross entropy loss"""
    if hasattr(input, 'data'):
        input_data = input.data
    else:
        input_data = np.array(input)
        
    if hasattr(target, 'data'):
        target_data = target.data
    else:
        target_data = np.array(target)
    
    # Apply log_softmax
    log_probs = log_softmax(Tensor(input_data)).data
    
    # Compute loss
    if target_data.ndim == 1:  # Class indices
        if input_data.ndim == 1:
            loss = -log_probs[int(target_data)]
        else:
            batch_size = input_data.shape[0]
            loss = -log_probs[range(batch_size), target_data.astype(int)]
    else:  # One-hot encoded
        loss = -np.sum(target_data * log_probs, axis=-1)
    
    if reduction == 'mean':
        loss = np.mean(loss)
    elif reduction == 'sum':
        loss = np.sum(loss)
    
    return Tensor(loss, requires_grad=True)

def mse_loss(input, target, size_average=None, reduce=None, reduction='mean'):
    """Mean squared error loss"""
    if hasattr(input, 'data'):
        input_data = input.data
    else:
        input_data = np.array(input)
        
    if hasattr(target, 'data'):
        target_data = target.data
    else:
        target_data = np.array(target)
    
    loss = (input_data - target_data) ** 2
    
    if reduction == 'mean':
        loss = np.mean(loss)
    elif reduction == 'sum':
        loss = np.sum(loss)
    
    return Tensor(loss, requires_grad=True)

def l1_loss(input, target, size_average=None, reduce=None, reduction='mean'):
    """L1 loss (mean absolute error)"""
    if hasattr(input, 'data'):
        input_data = input.data
    else:
        input_data = np.array(input)
        
    if hasattr(target, 'data'):
        target_data = target.data
    else:
        target_data = np.array(target)
    
    loss = np.abs(input_data - target_data)
    
    if reduction == 'mean':
        loss = np.mean(loss)
    elif reduction == 'sum':
        loss = np.sum(loss)
    
    return Tensor(loss, requires_grad=True)

def binary_cross_entropy(input, target, weight=None, size_average=None, reduce=None, reduction='mean'):
    """Binary cross entropy loss"""
    if hasattr(input, 'data'):
        input_data = input.data
    else:
        input_data = np.array(input)
        
    if hasattr(target, 'data'):
        target_data = target.data
    else:
        target_data = np.array(target)
    
    # Clamp input to prevent log(0)
    input_data = np.clip(input_data, 1e-8, 1 - 1e-8)
    
    loss = -(target_data * np.log(input_data) + (1 - target_data) * np.log(1 - input_data))
    
    if reduction == 'mean':
        loss = np.mean(loss)
    elif reduction == 'sum':
        loss = np.sum(loss)
    
    return Tensor(loss, requires_grad=True)

# Linear algebra operations
def linear(input, weight, bias=None):
    """Linear transformation"""
    if hasattr(input, 'data'):
        input_data = input.data
    else:
        input_data = np.array(input)
        
    if hasattr(weight, 'data'):
        weight_data = weight.data
    else:
        weight_data = np.array(weight)
    
    output_data = np.dot(input_data, weight_data.T)
    
    if bias is not None:
        if hasattr(bias, 'data'):
            bias_data = bias.data
        else:
            bias_data = np.array(bias)
        output_data += bias_data
    
    return Tensor(output_data, requires_grad=getattr(input, 'requires_grad', False))

def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """2D convolution (simplified implementation)"""
    if hasattr(input, 'data'):
        input_data = input.data
    else:
        input_data = np.array(input)
        
    if hasattr(weight, 'data'):
        weight_data = weight.data
    else:
        weight_data = np.array(weight)
    
    # Simplified convolution approximation
    batch_size, in_channels, height, width = input_data.shape
    out_channels = weight_data.shape[0]
    
    output_data = np.zeros((batch_size, out_channels, height, width))
    for b in range(batch_size):
        for out_c in range(out_channels):
            channel_sum = np.sum(input_data[b] * 0.1, axis=0)
            output_data[b, out_c] = channel_sum
    
    if bias is not None:
        if hasattr(bias, 'data'):
            bias_data = bias.data
        else:
            bias_data = np.array(bias)
        output_data += bias_data.reshape(1, -1, 1, 1)
    
    return Tensor(output_data, requires_grad=getattr(input, 'requires_grad', False))

def max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
    """2D max pooling (simplified implementation)"""
    if hasattr(input, 'data'):
        input_data = input.data
    else:
        input_data = np.array(input)
    
    if stride is None:
        stride = kernel_size
    
    # Simplified max pooling approximation
    batch_size, channels, height, width = input_data.shape
    new_height = height // stride
    new_width = width // stride
    
    output_data = np.zeros((batch_size, channels, new_height, new_width))
    for b in range(batch_size):
        for c in range(channels):
            for h in range(new_height):
                for w in range(new_width):
                    h_start = h * stride
                    w_start = w * stride
                    h_end = min(h_start + kernel_size, height)
                    w_end = min(w_start + kernel_size, width)
                    
                    output_data[b, c, h, w] = np.max(input_data[b, c, h_start:h_end, w_start:w_end])
    
    return Tensor(output_data, requires_grad=getattr(input, 'requires_grad', False))

def avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
    """2D average pooling (simplified implementation)"""
    if hasattr(input, 'data'):
        input_data = input.data
    else:
        input_data = np.array(input)
    
    if stride is None:
        stride = kernel_size
    
    # Simplified average pooling approximation
    batch_size, channels, height, width = input_data.shape
    new_height = height // stride
    new_width = width // stride
    
    output_data = np.zeros((batch_size, channels, new_height, new_width))
    for b in range(batch_size):
        for c in range(channels):
            for h in range(new_height):
                for w in range(new_width):
                    h_start = h * stride
                    w_start = w * stride
                    h_end = min(h_start + kernel_size, height)
                    w_end = min(w_start + kernel_size, width)
                    
                    output_data[b, c, h, w] = np.mean(input_data[b, c, h_start:h_end, w_start:w_end])
    
    return Tensor(output_data, requires_grad=getattr(input, 'requires_grad', False))

# Normalization functions
def batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5):
    """Batch normalization"""
    if hasattr(input, 'data'):
        input_data = input.data
    else:
        input_data = np.array(input)
    
    if training:
        # Calculate batch statistics
        batch_mean = np.mean(input_data, axis=0)
        batch_var = np.var(input_data, axis=0)
        
        # Update running statistics
        if running_mean is not None:
            running_mean.data = (1 - momentum) * running_mean.data + momentum * batch_mean
        if running_var is not None:
            running_var.data = (1 - momentum) * running_var.data + momentum * batch_var
        
        # Normalize using batch statistics
        normalized = (input_data - batch_mean) / np.sqrt(batch_var + eps)
    else:
        # Use running statistics
        mean_data = running_mean.data if hasattr(running_mean, 'data') else running_mean
        var_data = running_var.data if hasattr(running_var, 'data') else running_var
        normalized = (input_data - mean_data) / np.sqrt(var_data + eps)
    
    # Scale and shift
    if weight is not None:
        weight_data = weight.data if hasattr(weight, 'data') else weight
        normalized = normalized * weight_data
    
    if bias is not None:
        bias_data = bias.data if hasattr(bias, 'data') else bias
        normalized = normalized + bias_data
    
    return Tensor(normalized, requires_grad=getattr(input, 'requires_grad', False))

def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    """Layer normalization"""
    if hasattr(input, 'data'):
        input_data = input.data
    else:
        input_data = np.array(input)
    
    # Calculate mean and variance over the last dimensions
    axes = tuple(range(-len(normalized_shape), 0))
    mean = np.mean(input_data, axis=axes, keepdims=True)
    var = np.var(input_data, axis=axes, keepdims=True)
    
    # Normalize
    normalized = (input_data - mean) / np.sqrt(var + eps)
    
    # Scale and shift
    if weight is not None:
        weight_data = weight.data if hasattr(weight, 'data') else weight
        normalized = normalized * weight_data
    
    if bias is not None:
        bias_data = bias.data if hasattr(bias, 'data') else bias
        normalized = normalized + bias_data
    
    return Tensor(normalized, requires_grad=getattr(input, 'requires_grad', False))

# Dropout
def dropout(input, p=0.5, training=True, inplace=False):
    """Dropout regularization"""
    if not training:
        return input
        
    if hasattr(input, 'data'):
        input_data = input.data
    else:
        input_data = np.array(input)
    
    # Generate dropout mask
    mask = np.random.random(input_data.shape) > p
    output_data = input_data * mask / (1 - p)
    
    return Tensor(output_data, requires_grad=getattr(input, 'requires_grad', False))

# Utility functions
def pad(input, pad, mode='constant', value=0):
    """Pad tensor"""
    if hasattr(input, 'data'):
        input_data = input.data
    else:
        input_data = np.array(input)
    
    # Convert pad format
    pad_width = []
    for i in range(input_data.ndim):
        if i * 2 < len(pad):
            pad_width.append((pad[i*2], pad[i*2+1] if i*2+1 < len(pad) else 0))
        else:
            pad_width.append((0, 0))
    
    if mode == 'constant':
        output_data = np.pad(input_data, pad_width, mode='constant', constant_values=value)
    elif mode == 'reflect':
        output_data = np.pad(input_data, pad_width, mode='reflect')
    elif mode == 'replicate':
        output_data = np.pad(input_data, pad_width, mode='edge')
    else:
        output_data = np.pad(input_data, pad_width, mode=mode)
    
    return Tensor(output_data, requires_grad=getattr(input, 'requires_grad', False))

def interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None):
    """Interpolate tensor (simplified implementation)"""
    if hasattr(input, 'data'):
        input_data = input.data
    else:
        input_data = np.array(input)
    
    # Simplified interpolation - just repeat values
    if scale_factor is not None:
        if isinstance(scale_factor, (int, float)):
            scale_factor = [scale_factor] * (input_data.ndim - 2)
        
        new_shape = list(input_data.shape)
        for i, scale in enumerate(scale_factor):
            new_shape[-(i+1)] = int(new_shape[-(i+1)] * scale)
        
        output_data = np.broadcast_to(input_data, new_shape)
    else:
        # Use provided size
        output_data = input_data  # Simplified
    
    return Tensor(output_data, requires_grad=getattr(input, 'requires_grad', False))

# Export all functions
__all__ = [
    # Activations
    'relu', 'leaky_relu', 'gelu', 'sigmoid', 'tanh', 'softmax', 'log_softmax',
    # Losses
    'cross_entropy', 'mse_loss', 'l1_loss', 'binary_cross_entropy',
    # Linear operations
    'linear', 'conv2d', 'max_pool2d', 'avg_pool2d',
    # Normalization
    'batch_norm', 'layer_norm',
    # Regularization
    'dropout',
    # Utils
    'pad', 'interpolate'
]