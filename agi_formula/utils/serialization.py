"""
AGI Serialization Utilities - save/load functionality like torch.save/torch.load
"""

import json
import pickle
import numpy as np
from typing import Any, Dict, Optional
import os

def save(obj, f, pickle_protocol=2):
    """Save object to file - like torch.save()"""
    
    if isinstance(f, str):
        # File path provided
        with open(f, 'w') as file:
            _save_to_file(obj, file)
    else:
        # File object provided
        _save_to_file(obj, f)

def _save_to_file(obj, file):
    """Internal function to save object to file"""
    
    if isinstance(obj, dict):
        # Save dictionary (like model state_dict)
        serializable_obj = _convert_to_serializable(obj)
        json.dump(serializable_obj, file, indent=2)
    else:
        # Handle other objects
        from ..core import Component
        from ..tensor import Tensor
        
        if isinstance(obj, Component):
            # Save module state
            state_dict = obj.state_dict()
            serializable_obj = _convert_to_serializable(state_dict)
            json.dump(serializable_obj, file, indent=2)
        elif isinstance(obj, Tensor):
            # Save tensor
            tensor_dict = {
                'data': obj.data.tolist(),
                'dtype': str(obj.dtype),
                'device': obj.device,
                'requires_grad': obj.requires_grad,
                'shape': obj.shape
            }
            if hasattr(obj, '_consciousness_attention'):
                tensor_dict['consciousness_attention'] = obj._consciousness_attention.tolist()
            json.dump(tensor_dict, file, indent=2)
        else:
            # Try to convert to serializable format
            serializable_obj = _convert_to_serializable(obj)
            json.dump(serializable_obj, file, indent=2)

def load(f, map_location=None):
    """Load object from file - like torch.load()"""
    
    if isinstance(f, str):
        # File path provided
        if not os.path.exists(f):
            raise FileNotFoundError(f"No such file or directory: '{f}'")
        
        with open(f, 'r') as file:
            return _load_from_file(file, map_location)
    else:
        # File object provided
        return _load_from_file(f, map_location)

def _load_from_file(file, map_location=None):
    """Internal function to load object from file"""
    
    try:
        data = json.load(file)
        return _convert_from_serializable(data, map_location)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to load file: {e}")

def _convert_to_serializable(obj):
    """Convert object to JSON-serializable format"""
    
    if isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            result[key] = _convert_to_serializable(value)
        return result
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return {
            '_type': 'numpy_array',
            'data': obj.tolist(),
            'dtype': str(obj.dtype),
            'shape': obj.shape
        }
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, complex):
        return {
            '_type': 'complex',
            'real': obj.real,
            'imag': obj.imag
        }
    else:
        # For other types, try to convert to basic types
        try:
            from ..tensor import Tensor
            from ..core import Variable
            
            if isinstance(obj, Tensor):
                return {
                    '_type': 'agi_tensor',
                    'data': obj.data.tolist(),
                    'dtype': str(obj.dtype),
                    'device': obj.device,
                    'requires_grad': obj.requires_grad,
                    'shape': obj.shape,
                    'consciousness_attention': getattr(obj, '_consciousness_attention', np.ones_like(obj.data) * 0.1).tolist()
                }
            elif isinstance(obj, Variable):
                return {
                    '_type': 'agi_variable',
                    'data': obj.data.tolist(),
                    'requires_grad': obj.requires_grad,
                    'consciousness_attention': getattr(obj, '_consciousness_attention', np.ones_like(obj.data) * 0.1).tolist()
                }
            else:
                return str(obj)
        except:
            return str(obj)

def _convert_from_serializable(obj, map_location=None):
    """Convert from JSON-serializable format back to objects"""
    
    if isinstance(obj, dict):
        if '_type' in obj:
            # Special object type
            if obj['_type'] == 'numpy_array':
                data = np.array(obj['data'])
                if 'dtype' in obj:
                    data = data.astype(obj['dtype'])
                return data
            elif obj['_type'] == 'complex':
                return complex(obj['real'], obj['imag'])
            elif obj['_type'] == 'agi_tensor':
                from ..tensor import Tensor
                tensor = Tensor(
                    obj['data'], 
                    dtype=obj.get('dtype'),
                    device=obj.get('device', 'cpu') if map_location is None else map_location,
                    requires_grad=obj.get('requires_grad', False)
                )
                if 'consciousness_attention' in obj:
                    tensor._consciousness_attention = np.array(obj['consciousness_attention'])
                return tensor
            elif obj['_type'] == 'agi_variable':
                from ..core import Variable
                param = Variable(obj['data'], requires_grad=obj.get('requires_grad', True))
                if 'consciousness_attention' in obj:
                    param._consciousness_attention = np.array(obj['consciousness_attention'])
                return param
            else:
                # Unknown type, return as dict
                result = {}
                for key, value in obj.items():
                    result[key] = _convert_from_serializable(value, map_location)
                return result
        else:
            # Regular dictionary
            result = {}
            for key, value in obj.items():
                result[key] = _convert_from_serializable(value, map_location)
            return result
    elif isinstance(obj, list):
        return [_convert_from_serializable(item, map_location) for item in obj]
    else:
        return obj

# Model checkpoint utilities
def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save training checkpoint"""
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if hasattr(optimizer, 'state_dict') else {},
        'loss': loss,
        'model_type': type(model).__name__
    }
    
    save(checkpoint, filepath)

def load_checkpoint(filepath, model, optimizer=None, map_location=None):
    """Load training checkpoint"""
    
    checkpoint = load(filepath, map_location)
    
    if hasattr(model, 'load_state_dict'):
        model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and hasattr(optimizer, 'load_state_dict'):
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', 0.0)
    
    return epoch, loss

# Export functions
__all__ = ['save', 'load', 'save_checkpoint', 'load_checkpoint']