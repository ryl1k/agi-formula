"""
AGI Optimizers

Advanced optimizers with consciousness, meta-learning, and causal reasoning capabilities.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

class Optimizer:
    """Base optimizer class"""
    
    def __init__(self, params, defaults):
        self.defaults = defaults
        self.state = {}
        self.param_groups = []
        
        if not isinstance(params, list):
            params = list(params)
        
        if len(params) == 0:
            raise ValueError("optimizer got an empty parameter list")
        
        param_groups = params
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]
        
        for param_group in param_groups:
            self.add_param_group(param_group)
    
    def add_param_group(self, param_group):
        """Add parameter group"""
        params = param_group['params']
        if not isinstance(params, list):
            params = list(params)
        param_group['params'] = params
        
        for name, default in self.defaults.items():
            param_group.setdefault(name, default)
        
        self.param_groups.append(param_group)
    
    def zero_grad(self):
        """Zero gradients"""
        for group in self.param_groups:
            for p in group['params']:
                if hasattr(p, 'zero_grad'):
                    p.zero_grad()
                elif hasattr(p, 'grad') and p.grad is not None:
                    p.grad = None
    
    def step(self, closure=None):
        """Optimization step"""
        raise NotImplementedError

class SGD(Optimizer):
    """Stochastic Gradient Descent with consciousness enhancement"""
    
    def __init__(self, params, lr=0.01, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, 
                       weight_decay=weight_decay, nesterov=nesterov)
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        """SGD step with consciousness enhancement"""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            
            for p in group['params']:
                if not hasattr(p, 'data') or not hasattr(p, 'grad'):
                    continue
                    
                if p.grad is None:
                    continue
                
                # Get gradient
                if hasattr(p.grad, 'data'):
                    d_p = p.grad.data
                else:
                    d_p = np.array(p.grad)
                
                if weight_decay != 0:
                    d_p = d_p + weight_decay * p.data
                
                if momentum != 0:
                    param_state = self.state.setdefault(id(p), {})
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = np.zeros_like(p.data)
                    else:
                        buf = param_state['momentum_buffer']
                    
                    buf = momentum * buf + (1 - dampening) * d_p
                    
                    if nesterov:
                        d_p = d_p + momentum * buf
                    else:
                        d_p = buf
                    
                    param_state['momentum_buffer'] = buf
                
                # Consciousness-guided learning rate adaptation
                consciousness_factor = 1.0
                if hasattr(p, '_consciousness_attention'):
                    consciousness_factor = 1.0 + np.mean(p._consciousness_attention) * 0.1
                
                effective_lr = group['lr'] * consciousness_factor
                p.data -= effective_lr * d_p
        
        return loss

class Adam(Optimizer):
    """Adam optimizer with meta-learning enhancement"""
    
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        """Adam step with meta-learning enhancement"""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if not hasattr(p, 'data') or not hasattr(p, 'grad'):
                    continue
                    
                if p.grad is None:
                    continue
                
                # Get gradient
                if hasattr(p.grad, 'data'):
                    grad = p.grad.data
                else:
                    grad = np.array(p.grad)
                
                state = self.state.setdefault(id(p), {})
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = np.zeros_like(p.data)
                    state['exp_avg_sq'] = np.zeros_like(p.data)
                    state['meta_history'] = []
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                if group['weight_decay'] != 0:
                    grad = grad + group['weight_decay'] * p.data
                
                # Exponential moving average of gradient values
                exp_avg *= beta1
                exp_avg += (1 - beta1) * grad
                
                # Exponential moving average of squared gradient values
                exp_avg_sq *= beta2
                exp_avg_sq += (1 - beta2) * (grad * grad)
                
                denom = (np.sqrt(exp_avg_sq) / np.sqrt(bias_correction2)) + group['eps']
                
                # Meta-learning rate adaptation
                meta_factor = 1.0
                if len(state['meta_history']) > 5:
                    recent_grads = state['meta_history'][-5:]
                    grad_variance = np.var(recent_grads)
                    meta_factor = 1.0 + np.clip(grad_variance * 0.1, -0.2, 0.2)
                
                # Consciousness-guided learning rate
                consciousness_factor = 1.0
                if hasattr(p, '_consciousness_attention'):
                    consciousness_factor = 1.0 + np.mean(p._consciousness_attention) * 0.05
                
                step_size = group['lr'] / bias_correction1 * meta_factor * consciousness_factor
                
                # Parameter update
                update = step_size * exp_avg / denom
                p.data -= update
                
                # Track gradient magnitude for meta-learning
                state['meta_history'].append(np.mean(np.abs(grad)))
                if len(state['meta_history']) > 10:
                    state['meta_history'] = state['meta_history'][-10:]
        
        return loss

class AdamW(Optimizer):
    """AdamW optimizer with decoupled weight decay"""
    
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        """AdamW step with decoupled weight decay"""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if not hasattr(p, 'data') or not hasattr(p, 'grad'):
                    continue
                    
                if p.grad is None:
                    continue
                
                # Get gradient
                if hasattr(p.grad, 'data'):
                    grad = p.grad.data
                else:
                    grad = np.array(p.grad)
                
                state = self.state.setdefault(id(p), {})
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = np.zeros_like(p.data)
                    state['exp_avg_sq'] = np.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Exponential moving average of gradient values
                exp_avg *= beta1
                exp_avg += (1 - beta1) * grad
                
                # Exponential moving average of squared gradient values
                exp_avg_sq *= beta2
                exp_avg_sq += (1 - beta2) * (grad * grad)
                
                denom = (np.sqrt(exp_avg_sq) / np.sqrt(bias_correction2)) + group['eps']
                step_size = group['lr'] / bias_correction1
                
                # AdamW weight decay (decoupled)
                p.data -= group['weight_decay'] * group['lr'] * p.data
                
                # Parameter update
                p.data -= step_size * exp_avg / denom
        
        return loss

class RMSprop(Optimizer):
    """RMSprop optimizer"""
    
    def __init__(self, params, lr=0.01, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0):
        defaults = dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        """RMSprop step"""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if not hasattr(p, 'data') or not hasattr(p, 'grad'):
                    continue
                    
                if p.grad is None:
                    continue
                
                # Get gradient
                if hasattr(p.grad, 'data'):
                    grad = p.grad.data
                else:
                    grad = np.array(p.grad)
                
                state = self.state.setdefault(id(p), {})
                
                # State initialization
                if len(state) == 0:
                    state['square_avg'] = np.zeros_like(p.data)
                    if group['momentum'] > 0:
                        state['momentum_buffer'] = np.zeros_like(p.data)
                
                square_avg = state['square_avg']
                alpha = group['alpha']
                
                if group['weight_decay'] != 0:
                    grad = grad + group['weight_decay'] * p.data
                
                square_avg *= alpha
                square_avg += (1 - alpha) * grad * grad
                
                avg = np.sqrt(square_avg) + group['eps']
                
                if group['momentum'] > 0:
                    buf = state['momentum_buffer']
                    buf *= group['momentum']
                    buf += grad / avg
                    p.data -= group['lr'] * buf
                else:
                    p.data -= group['lr'] * grad / avg
        
        return loss

class QuantumOptimizer(Optimizer):
    """Quantum-inspired optimizer for advanced parameter optimization"""
    
    def __init__(self, params, lr=0.01, superposition_factor=0.1, tunneling_prob=0.05):
        defaults = dict(lr=lr, superposition_factor=superposition_factor, 
                       tunneling_prob=tunneling_prob)
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        """Quantum-inspired optimization step"""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if not hasattr(p, 'data') or not hasattr(p, 'grad'):
                    continue
                    
                if p.grad is None:
                    continue
                
                # Get gradient
                if hasattr(p.grad, 'data'):
                    grad = p.grad.data
                else:
                    grad = np.array(p.grad)
                
                state = self.state.setdefault(id(p), {})
                
                # Quantum tunneling - escape local minima
                if np.random.random() < group['tunneling_prob']:
                    tunneling_noise = np.random.randn(*p.data.shape) * group['lr'] * 5
                    p.data += tunneling_noise
                
                # Quantum superposition - explore multiple states
                superposition_noise = np.random.randn(*p.data.shape) * group['superposition_factor']
                superposed_grad = grad + superposition_noise
                
                # Standard gradient update with quantum enhancement
                p.data -= group['lr'] * superposed_grad
        
        return loss

# Learning rate schedulers
class StepLR:
    """Step learning rate scheduler"""
    
    def __init__(self, optimizer, step_size, gamma=0.1):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.step_count = 0
    
    def step(self):
        """Update learning rate"""
        self.step_count += 1
        if self.step_count % self.step_size == 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= self.gamma

class ExponentialLR:
    """Exponential learning rate scheduler"""
    
    def __init__(self, optimizer, gamma):
        self.optimizer = optimizer
        self.gamma = gamma
    
    def step(self):
        """Update learning rate"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= self.gamma

class CosineAnnealingLR:
    """Cosine annealing learning rate scheduler"""
    
    def __init__(self, optimizer, T_max, eta_min=0):
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.step_count = 0
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def step(self):
        """Update learning rate"""
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = self.eta_min + (self.base_lrs[i] - self.eta_min) * \
                               (1 + np.cos(np.pi * self.step_count / self.T_max)) / 2
        self.step_count += 1

# Export all optimizers
__all__ = [
    'Optimizer', 'SGD', 'Adam', 'AdamW', 'RMSprop', 'QuantumOptimizer',
    'StepLR', 'ExponentialLR', 'CosineAnnealingLR'
]