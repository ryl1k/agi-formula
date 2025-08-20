"""
AGI Data Utilities - agi.utils.data (like torch.utils.data)

Data loading and batching utilities with AGI consciousness-guided sampling.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Iterator, Union
from abc import ABC, abstractmethod

class Dataset(ABC):
    """Base dataset class - like torch.utils.data.Dataset"""
    
    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError
    
    @abstractmethod  
    def __len__(self):
        raise NotImplementedError

class TensorDataset(Dataset):
    """Tensor dataset - like torch.utils.data.TensorDataset"""
    
    def __init__(self, *tensors):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), \
            "Size mismatch between tensors"
        self.tensors = tensors
    
    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)
    
    def __len__(self):
        return self.tensors[0].size(0)

class DataLoader:
    """AGI Data loader - like torch.utils.data.DataLoader with consciousness enhancement"""
    
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0, 
                 drop_last=False, consciousness_sampling=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.drop_last = drop_last
        
        # AGI enhancement: consciousness-guided sampling
        self.consciousness_sampling = consciousness_sampling
        self._consciousness_weights = None
        
        if consciousness_sampling:
            self._initialize_consciousness_weights()
    
    def _initialize_consciousness_weights(self):
        """Initialize consciousness-based sampling weights"""
        weights = []
        for i in range(len(self.dataset)):
            try:
                sample = self.dataset[i]
                # Calculate consciousness score based on data complexity
                if isinstance(sample, tuple) and len(sample) >= 1:
                    data = sample[0]
                    if hasattr(data, 'data'):
                        complexity = np.std(data.data) + np.mean(np.abs(data.data))
                    else:
                        complexity = np.std(data) + np.mean(np.abs(data))
                    consciousness_score = min(1.0, complexity * 0.1)
                else:
                    consciousness_score = 0.5  # Default
                weights.append(consciousness_score)
            except:
                weights.append(0.5)  # Default on error
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones_like(weights) / len(weights)
        self._consciousness_weights = weights
    
    def __iter__(self):
        indices = list(range(len(self.dataset)))
        
        if self.shuffle:
            if self.consciousness_sampling and self._consciousness_weights is not None:
                # AGI enhancement: consciousness-guided shuffling
                # Slightly bias towards high-consciousness samples
                enhanced_weights = self._consciousness_weights * 0.8 + 0.2 / len(indices)
                enhanced_weights = enhanced_weights / np.sum(enhanced_weights)
                indices = np.random.choice(indices, size=len(indices), replace=False, 
                                         p=enhanced_weights).tolist()
            else:
                np.random.shuffle(indices)
        
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            
            if self.drop_last and len(batch_indices) < self.batch_size:
                break
            
            batch = [self.dataset[idx] for idx in batch_indices]
            
            # Convert to tensors and stack if needed
            if len(batch) > 0 and isinstance(batch[0], tuple):
                # Unpack tuples (input, target pairs)
                inputs = [item[0] for item in batch]
                targets = [item[1] for item in batch] if len(batch[0]) > 1 else None
                
                # Stack tensors
                stacked_inputs = self._stack_tensors(inputs)
                stacked_targets = self._stack_tensors(targets) if targets else None
                
                if stacked_targets is not None:
                    yield stacked_inputs, stacked_targets
                else:
                    yield stacked_inputs
            else:
                yield batch
    
    def _stack_tensors(self, tensors):
        """Stack list of tensors into batch"""
        if tensors is None:
            return None
        
        from ..tensor import Tensor
        
        # Handle different tensor types
        if all(hasattr(t, 'data') for t in tensors):
            # AGI tensors
            stacked_data = np.stack([t.data for t in tensors], axis=0)
            requires_grad = any(t.requires_grad for t in tensors)
            device = tensors[0].device if hasattr(tensors[0], 'device') else "cpu"
            
            result = Tensor(stacked_data, requires_grad=requires_grad, device=device)
            
            # AGI enhancement: combine consciousness attention
            if hasattr(tensors[0], '_consciousness_attention'):
                attention_stack = np.stack([t._consciousness_attention for t in tensors], axis=0)
                result._consciousness_attention = np.mean(attention_stack, axis=0)
            
            return result
        else:
            # Regular arrays
            stacked_data = np.stack([np.array(t) for t in tensors], axis=0)
            return Tensor(stacked_data)
    
    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size

class Subset(Dataset):
    """Subset of a dataset - like torch.utils.data.Subset"""
    
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
    
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]
    
    def __len__(self):
        return len(self.indices)

class ConcatDataset(Dataset):
    """Concatenated dataset - like torch.utils.data.ConcatDataset"""
    
    def __init__(self, datasets):
        self.datasets = list(datasets)
        assert len(self.datasets) > 0, 'datasets should not be an empty iterable'
        self.cumulative_sizes = self._get_cumulative_sizes()
    
    def _get_cumulative_sizes(self):
        sizes = [len(d) for d in self.datasets]
        return np.cumsum(sizes)
    
    def __len__(self):
        return self.cumulative_sizes[-1]
    
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = np.searchsorted(self.cumulative_sizes, idx, side='right')
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

# AGI-specific datasets
class ConsciousnessDataset(Dataset):
    """Dataset for consciousness training"""
    
    def __init__(self, size=1000, input_dim=100, consciousness_levels=[0.3, 0.6, 0.9]):
        self.size = size
        self.input_dim = input_dim
        self.consciousness_levels = consciousness_levels
        
        # Generate data
        self.data = []
        self.targets = []
        
        for i in range(size):
            # Choose random consciousness level
            consciousness = np.random.choice(consciousness_levels)
            
            # Generate input based on consciousness level
            if consciousness < 0.5:
                # Low consciousness - simple patterns
                input_data = np.random.randn(input_dim) * 0.5
                pattern_len = min(10, input_dim)
                input_data[:pattern_len] = np.sin(np.linspace(0, 2*np.pi, pattern_len))
            elif consciousness < 0.8:
                # Medium consciousness - complex patterns
                input_data = np.random.randn(input_dim)
                pattern_len = min(20, input_dim)
                input_data[:pattern_len] = np.sin(np.linspace(0, 4*np.pi, pattern_len))
                if input_dim > 20:
                    choice_len = min(20, input_dim - 20)
                    input_data[20:20+choice_len] = np.random.choice([0, 1], choice_len)
            else:
                # High consciousness - very complex patterns
                input_data = np.random.randn(input_dim)
                t = np.linspace(0, 4*np.pi, input_dim)
                input_data += 0.3 * np.sin(t) * np.cos(2*t)
            
            from ..tensor import Tensor
            tensor_input = Tensor(input_data, requires_grad=False)
            tensor_input._consciousness_attention = np.ones_like(input_data) * consciousness
            
            self.data.append(tensor_input)
            self.targets.append(consciousness)
    
    def __getitem__(self, index):
        from ..tensor import Tensor
        target_tensor = Tensor([self.targets[index]], requires_grad=False)
        return self.data[index], target_tensor
    
    def __len__(self):
        return self.size

class CausalDataset(Dataset):
    """Dataset for causal reasoning training"""
    
    def __init__(self, size=1000, num_variables=10):
        self.size = size
        self.num_variables = num_variables
        
        # Generate causal data
        self.data = []
        self.causal_graphs = []
        
        for i in range(size):
            # Generate random causal graph
            causal_graph = self._generate_causal_graph()
            
            # Generate data following causal structure
            variables = self._generate_causal_data(causal_graph)
            
            from ..tensor import Tensor
            input_tensor = Tensor(variables, requires_grad=False)
            target_tensor = Tensor(self._graph_to_adjacency(causal_graph), requires_grad=False)
            
            self.data.append((input_tensor, target_tensor))
            self.causal_graphs.append(causal_graph)
    
    def _generate_causal_graph(self):
        """Generate random causal graph"""
        graph = {}
        for i in range(self.num_variables):
            parents = []
            for j in range(i):  # Only earlier variables can be parents
                if np.random.random() < 0.3:  # 30% chance of causal connection
                    parents.append(j)
            graph[i] = parents
        return graph
    
    def _generate_causal_data(self, causal_graph):
        """Generate data following causal structure"""
        variables = np.random.randn(self.num_variables) * 0.5
        
        for effect, causes in causal_graph.items():
            for cause in causes:
                causal_strength = np.random.uniform(0.3, 0.8)
                variables[effect] += causal_strength * variables[cause]
        
        return variables
    
    def _graph_to_adjacency(self, causal_graph):
        """Convert causal graph to adjacency matrix"""
        adj_matrix = np.zeros((self.num_variables, self.num_variables))
        for effect, causes in causal_graph.items():
            for cause in causes:
                adj_matrix[cause, effect] = 1.0
        return adj_matrix.flatten()
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return self.size

# Utility functions
def random_split(dataset, lengths):
    """Randomly split dataset - like torch.utils.data.random_split"""
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")
    
    indices = np.random.permutation(len(dataset)).tolist()
    
    datasets = []
    offset = 0
    for length in lengths:
        subset_indices = indices[offset:offset + length]
        datasets.append(Subset(dataset, subset_indices))
        offset += length
    
    return datasets

def get_worker_info():
    """Get worker info for multiprocessing - placeholder"""
    class WorkerInfo:
        def __init__(self):
            self.id = 0
            self.num_workers = 1
    return WorkerInfo()

# Export all classes and functions
__all__ = [
    'Dataset', 'TensorDataset', 'DataLoader', 'Subset', 'ConcatDataset',
    'ConsciousnessDataset', 'CausalDataset', 'random_split', 'get_worker_info'
]