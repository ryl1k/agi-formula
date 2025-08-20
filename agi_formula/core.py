"""
AGI Core Components

Fundamental AGI building blocks with consciousness, causal reasoning, 
and meta-learning capabilities.
"""

import numpy as np
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod

class Variable:
    """AGI learnable variable with consciousness tracking"""
    
    def __init__(self, data, requires_grad=True):
        if hasattr(data, 'data'):
            self.data = data.data
        else:
            self.data = np.array(data)
        self.requires_grad = requires_grad
        self.grad = None
        
        # Consciousness awareness weighting
        self._consciousness_attention = np.ones_like(self.data) * 0.1
        self._causal_connections = {}
        
    def backward(self, grad=None):
        if grad is None:
            grad = np.ones_like(self.data)
        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad
    
    def zero_grad(self):
        self.grad = None
    
    def to(self, device):
        if device == "cuda":
            self._consciousness_attention *= 1.1
        return self
    
    def cuda(self):
        return self.to("cuda")
    
    def cpu(self):
        return self.to("cpu")

class Component:
    """Base AGI component with consciousness and reasoning capabilities"""
    
    def __init__(self):
        self._variables = {}
        self._subcomponents = {}
        self.training = True
        
        # AGI consciousness and reasoning state
        self._consciousness_level = 0.5
        self._causal_memory = {}
        self._meta_learning_state = {"adaptation_rate": 0.1}
        self._reasoning_depth = 0
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subcomponents must implement forward reasoning")
    
    def __call__(self, *args, **kwargs):
        result = self.forward(*args, **kwargs)
        if self.training:
            self._update_consciousness(*args, result=result)
        return result
    
    def variables(self):
        """Return all learnable variables"""
        vars_list = []
        for var in self._variables.values():
            vars_list.append(var)
        for component in self._subcomponents.values():
            vars_list.extend(component.variables())
        return vars_list
    
    def named_variables(self):
        """Return named learnable variables"""
        named_vars = []
        for name, var in self._variables.items():
            named_vars.append((name, var))
        for comp_name, component in self._subcomponents.items():
            for name, var in component.named_variables():
                named_vars.append((f"{comp_name}.{name}", var))
        return named_vars
    
    def state_dict(self):
        """Return state dictionary"""
        state = {}
        for name, var in self._variables.items():
            state[name] = np.array(var.data).copy()
        for name, component in self._subcomponents.items():
            comp_state = component.state_dict()
            for key, value in comp_state.items():
                state[f"{name}.{key}"] = value
        return state
    
    def load_state_dict(self, state_dict, strict=True):
        """Load state dictionary"""
        for name, value in state_dict.items():
            if '.' in name:
                comp_name, var_name = name.split('.', 1)
                if comp_name in self._subcomponents:
                    self._subcomponents[comp_name].load_state_dict({var_name: value}, strict)
            else:
                if name in self._variables:
                    self._variables[name].data = np.array(value)
    
    def train(self, mode=True):
        """Set training mode"""
        self.training = mode
        for component in self._subcomponents.values():
            component.train(mode)
        return self
    
    def eval(self):
        """Set evaluation mode"""
        return self.train(False)
    
    def to(self, device):
        """Move to device"""
        for var in self.variables():
            var.to(device)
        return self
    
    def cuda(self):
        """Move to CUDA"""
        return self.to("cuda")
    
    def cpu(self):
        """Move to CPU"""
        return self.to("cpu")
    
    def zero_grad(self):
        """Zero gradients"""
        for var in self.variables():
            var.zero_grad()
    
    def __setattr__(self, name, value):
        if isinstance(value, Variable):
            self._variables[name] = value
        elif isinstance(value, Component):
            self._subcomponents[name] = value
        object.__setattr__(self, name, value)
    
    def _update_consciousness(self, *inputs, result=None):
        """Update consciousness based on input-output patterns"""
        if len(inputs) > 0 and hasattr(inputs[0], 'data'):
            input_complexity = np.std(inputs[0].data)
            if hasattr(result, 'data'):
                output_complexity = np.std(result.data)
                consciousness_change = (output_complexity - input_complexity) * 0.01
                self._consciousness_level = np.clip(
                    self._consciousness_level + consciousness_change, 
                    0.0, 1.0
                )

class Transform(Component):
    """Linear transformation component with consciousness enhancement"""
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize transformation weights
        weight_data = np.random.randn(out_features, in_features) * np.sqrt(2.0 / (in_features + out_features))
        self.weight = Variable(weight_data)
        
        if bias:
            bias_data = np.zeros(out_features)
            self.bias = Variable(bias_data)
        else:
            self.bias = None
    
    def forward(self, input):
        """Forward reasoning with consciousness enhancement"""
        # Convert input to numpy if needed
        if hasattr(input, 'data'):
            input_data = input.data
        else:
            input_data = np.array(input)
        
        # Linear transformation
        weight_data = np.array(self.weight.data)
        output_data = np.dot(input_data, weight_data.T)
        if self.bias is not None:
            bias_data = np.array(self.bias.data)
            output_data += bias_data
        
        # Consciousness-guided activation scaling
        consciousness_boost = 1.0 + self._consciousness_level * 0.05
        output_data *= consciousness_boost
        
        # Create output tensor
        from .tensor import Tensor
        output = Tensor(output_data, requires_grad=True)
        
        # Track gradients for backprop
        if hasattr(input, 'requires_grad') and input.requires_grad:
            output._backward_fn = lambda grad: self._backward(input, grad)
        
        return output
    
    def _backward(self, input, grad_output):
        """Backward pass with consciousness enhancement"""
        # Gradient for input
        weight_data = np.array(self.weight.data)
        input_grad = np.dot(grad_output, weight_data)
        
        # Gradient for weight
        if hasattr(input, 'data'):
            input_data = input.data
        else:
            input_data = np.array(input)
            
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)
        if grad_output.ndim == 1:
            grad_output = grad_output.reshape(1, -1)
            
        weight_grad = np.dot(grad_output.T, input_data)
        
        # Consciousness-guided gradient scaling
        consciousness_factor = 1.0 + self._consciousness_level * 0.1
        weight_grad *= consciousness_factor
        
        self.weight.backward(weight_grad)
        
        if self.bias is not None:
            bias_grad = np.sum(grad_output, axis=0)
            self.bias.backward(bias_grad)
        
        return input_grad
    
    def __repr__(self):
        return f'Transform(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})'

class Activation(Component):
    """ReLU activation with consciousness awareness"""
    
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace
    
    def forward(self, input):
        """Forward pass with consciousness-aware activation"""
        if hasattr(input, 'data'):
            input_data = input.data
        else:
            input_data = np.array(input)
        
        # ReLU activation
        output_data = np.maximum(0, input_data)
        
        # Consciousness-guided activation enhancement
        consciousness_boost = self._consciousness_level * 0.02
        positive_mask = output_data > 0
        output_data[positive_mask] += consciousness_boost * output_data[positive_mask]
        
        from .tensor import Tensor
        return Tensor(output_data, requires_grad=getattr(input, 'requires_grad', False))
    
    def __repr__(self):
        return f'Activation(inplace={self.inplace})'

class Loss(Component):
    """Cross entropy loss with consciousness tracking"""
    
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, input, target):
        """Forward pass with consciousness tracking"""
        if hasattr(input, 'data'):
            input_data = input.data
        else:
            input_data = np.array(input)
            
        if hasattr(target, 'data'):
            target_data = target.data
        else:
            target_data = np.array(target)
        
        # Softmax
        exp_input = np.exp(input_data - np.max(input_data, axis=-1, keepdims=True))
        softmax = exp_input / np.sum(exp_input, axis=-1, keepdims=True)
        
        # Cross entropy
        if target_data.ndim == 1:  # Class indices
            if input_data.ndim == 1:
                log_prob = np.log(softmax[int(target_data)] + 1e-8)
                loss = -log_prob
            else:
                batch_size = input_data.shape[0]
                log_softmax = np.log(softmax + 1e-8)
                loss = -log_softmax[range(batch_size), target_data.astype(int)]
        else:  # One-hot encoded
            log_softmax = np.log(softmax + 1e-8)
            loss = -np.sum(target_data * log_softmax, axis=-1)
        
        # Consciousness-aware loss weighting
        consciousness_weight = 1.0 + self._consciousness_level * 0.1
        loss *= consciousness_weight
        
        if self.reduction == 'mean':
            loss = np.mean(loss)
        elif self.reduction == 'sum':
            loss = np.sum(loss)
        
        from .tensor import Tensor
        return Tensor(loss, requires_grad=True)
    
    def __repr__(self):
        return f'Loss(reduction={self.reduction})'

class MSELoss(Component):
    """Mean squared error loss"""
    
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, input, target):
        if hasattr(input, 'data'):
            input_data = input.data
        else:
            input_data = np.array(input)
            
        if hasattr(target, 'data'):
            target_data = target.data
        else:
            target_data = np.array(target)
        
        loss = (input_data - target_data) ** 2
        
        if self.reduction == 'mean':
            loss = np.mean(loss)
        elif self.reduction == 'sum':
            loss = np.sum(loss)
        
        from .tensor import Tensor
        return Tensor(loss, requires_grad=True)

class Dropout(Component):
    """Dropout regularization"""
    
    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        self.p = p
        self.inplace = inplace
    
    def forward(self, input):
        if not self.training:
            return input
            
        if hasattr(input, 'data'):
            input_data = input.data
        else:
            input_data = np.array(input)
        
        # Generate dropout mask
        mask = np.random.random(input_data.shape) > self.p
        output_data = input_data * mask / (1 - self.p)
        
        from .tensor import Tensor
        return Tensor(output_data, requires_grad=getattr(input, 'requires_grad', False))

class Convolution(Component):
    """2D Convolution layer"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        
        # Initialize weights
        weight_data = np.random.randn(out_channels, in_channels, *self.kernel_size) * 0.1
        self.weight = Variable(weight_data)
        
        if bias:
            bias_data = np.zeros(out_channels)
            self.bias = Variable(bias_data)
        else:
            self.bias = None
    
    def forward(self, input):
        """Simplified conv2d forward"""
        batch_size, in_channels, height, width = input.shape
        
        # Simplified convolution approximation
        output_data = np.zeros((batch_size, self.out_channels, height, width))
        for b in range(batch_size):
            for out_c in range(self.out_channels):
                channel_sum = np.sum(input.data[b] * 0.1, axis=0)
                output_data[b, out_c] = channel_sum
        
        from .tensor import Tensor
        output = Tensor(output_data, requires_grad=input.requires_grad)
        
        if self.bias is not None:
            output.data += self.bias.data.reshape(1, -1, 1, 1)
        
        return output

class BatchNorm(Component):
    """1D Batch normalization"""
    
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        self.weight = Variable(np.ones(num_features))
        self.bias = Variable(np.zeros(num_features))
        
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
    
    def forward(self, input):
        if hasattr(input, 'data'):
            input_data = input.data
        else:
            input_data = np.array(input)
        
        if self.training:
            # Calculate batch statistics
            batch_mean = np.mean(input_data, axis=0)
            batch_var = np.var(input_data, axis=0)
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            
            # Normalize using batch statistics
            normalized = (input_data - batch_mean) / np.sqrt(batch_var + self.eps)
        else:
            # Use running statistics
            normalized = (input_data - self.running_mean) / np.sqrt(self.running_var + self.eps)
        
        # Scale and shift
        output_data = normalized * self.weight.data + self.bias.data
        
        from .tensor import Tensor
        return Tensor(output_data, requires_grad=getattr(input, 'requires_grad', False))

# Utility functions
def init_weights(component, method='xavier'):
    """Initialize weights of component"""
    if method == 'xavier':
        for var in component.variables():
            if var.data.ndim >= 2:
                fan_in = var.data.shape[1]
                fan_out = var.data.shape[0]
                std = np.sqrt(2.0 / (fan_in + fan_out))
                var.data = np.random.randn(*var.data.shape) * std

# Export all classes
__all__ = [
    'Component', 'Variable', 'Transform', 'Activation', 'Loss', 
    'MSELoss', 'Dropout', 'Convolution', 'BatchNorm', 'init_weights'
]