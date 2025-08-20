"""
AGI Training Utilities - PyTorch-style utilities for AGI model management

Includes model saving/loading, device management, and training helpers.
"""

import os
import json
import pickle
import numpy as np
from typing import Dict, List, Any, Optional, Union
import time
from pathlib import Path

from .models import BaseAGIModel

def save_model(model: BaseAGIModel, 
               save_path: str,
               save_optimizer: bool = True,
               metadata: Optional[Dict[str, Any]] = None) -> str:
    """Save AGI model - PyTorch style interface"""
    
    # Ensure save directory exists
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Prepare save data
    save_data = {
        'model_type': model.model_type,
        'model_config': model.config.__dict__ if hasattr(model, 'config') else {},
        'model_state_dict': model.state_dict(),
        'model_parameters': model.parameters,
        'consciousness_state': model.consciousness_state,
        'causal_memory': model.causal_memory,
        'meta_learning_state': model.meta_learning_state,
        'performance_metrics': model.get_performance_metrics(),
        'save_timestamp': time.time(),
        'agi_formula_version': '1.0.0'
    }
    
    # Add metadata if provided
    if metadata:
        save_data['metadata'] = metadata
    
    # Save as JSON for compatibility (in production, would use more efficient format)
    try:
        with open(save_path, 'w') as f:
            json.dump(save_data, f, indent=2, default=_json_serializer)
        
        print(f"AGI model saved successfully: {save_path}")
        return save_path
        
    except Exception as e:
        print(f"Error saving model: {e}")
        raise

def load_model(load_path: str, 
               device: str = "cpu",
               strict: bool = True) -> BaseAGIModel:
    """Load AGI model - PyTorch style interface"""
    
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Model file not found: {load_path}")
    
    try:
        with open(load_path, 'r') as f:
            save_data = json.load(f)
        
        # Get model type and config
        model_type = save_data.get('model_type', 'CompleteAGI')
        model_config = save_data.get('model_config', {})
        
        # Create model based on type
        from .models import AGIModel, AGIModelConfig
        
        config = AGIModelConfig(**model_config)
        config.device = device
        
        # Remove model_type from config to avoid conflict
        config_dict = config.__dict__.copy()
        if 'model_type' in config_dict:
            del config_dict['model_type']
        
        model = AGIModel(model_type.lower(), **config_dict)
        
        # Load states
        if 'model_state_dict' in save_data:
            model.load_state_dict(save_data['model_state_dict'])
        
        if 'model_parameters' in save_data:
            model.parameters = save_data['model_parameters']
            
        if 'consciousness_state' in save_data:
            model.consciousness_state = save_data['consciousness_state']
            
        if 'causal_memory' in save_data:
            model.causal_memory = save_data['causal_memory']
            
        if 'meta_learning_state' in save_data:
            model.meta_learning_state = save_data['meta_learning_state']
        
        model.to(device)
        
        print(f"AGI model loaded successfully from: {load_path}")
        print(f"Model type: {model_type}")
        print(f"Device: {device}")
        
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        if strict:
            raise
        else:
            print("Creating new model with default config...")
            from .models import AGIModel
            return AGIModel("complete")

def _json_serializer(obj):
    """Custom JSON serializer for numpy arrays and other objects"""
    if isinstance(obj, np.ndarray):
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
        return {'real': obj.real, 'imag': obj.imag}
    else:
        return str(obj)

def get_device_info() -> Dict[str, Any]:
    """Get device information"""
    device_info = {
        'cpu_available': True,
        'cuda_available': False,
        'cuda_devices': 0,
        'recommended_device': 'cpu'
    }
    
    try:
        # Try to detect CUDA (simplified detection)
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            device_info['cuda_available'] = True
            device_info['cuda_devices'] = 1  # Simplified
            device_info['recommended_device'] = 'cuda'
    except:
        pass
    
    return device_info

def set_device(device: str = "auto") -> str:
    """Set optimal device for AGI training"""
    if device == "auto":
        device_info = get_device_info()
        device = device_info['recommended_device']
    
    print(f"Using device: {device}")
    return device

def create_agi_training_example():
    """Create a complete AGI training example"""
    
    print("AGI-FORMULA TRAINING EXAMPLE")
    print("=" * 40)
    print("Demonstrating PyTorch-style AGI training")
    print()
    
    # 1. Create model
    from .models import CompleteAGI
    
    model = CompleteAGI(
        num_neurons=1000,
        consciousness_level=0.8,
        causal_reasoning=True,
        self_modification=True,
        sparsity=0.01,
        device="cpu"
    )
    
    print("✓ AGI model created")
    
    # 2. Create dataset
    from .datasets import create_dataset
    
    train_dataset = create_dataset("consciousness", size=500)
    val_dataset = create_dataset("consciousness", size=100)
    
    print("✓ Consciousness dataset created")
    
    # 3. Create trainer
    from .trainers import Trainer
    
    trainer = Trainer(
        model,
        epochs=20,
        batch_size=16,
        learning_rate=0.01,
        consciousness_training=True,
        causal_discovery=True
    )
    
    print("✓ AGI trainer created")
    
    # 4. Train model
    print("\\nStarting AGI training...")
    
    results = trainer.fit(train_dataset, val_dataset, epochs=5)  # Short demo
    
    print("✓ Training completed")
    
    # 5. Save model
    save_path = "./agi_demo_model.json"
    save_model(model, save_path)
    
    print("✓ Model saved")
    
    # 6. Load model
    loaded_model = load_model(save_path)
    
    print("✓ Model loaded")
    
    print("\\nAGI TRAINING EXAMPLE COMPLETED!")
    print("=" * 40)
    
    return {
        'model': model,
        'trainer': trainer,
        'results': results,
        'save_path': save_path
    }

def validate_agi_installation() -> Dict[str, Any]:
    """Validate AGI-Formula installation and capabilities"""
    
    print("AGI-FORMULA INSTALLATION VALIDATION")
    print("=" * 45)
    
    validation_results = {
        'core_modules': {},
        'breakthrough_optimizations': {},
        'training_components': {},
        'overall_status': 'unknown'
    }
    
    # Test core modules
    core_tests = [
        ('Base AGI Model', 'from agi_formula.training.models import BaseAGIModel'),
        ('Sparse AGI', 'from agi_formula.training.models import SparseAGIModel'),
        ('Neuromorphic AGI', 'from agi_formula.training.models import NeuromorphicModel'),
        ('Conscious AGI', 'from agi_formula.training.models import ConsciousModel'),
        ('Complete AGI', 'from agi_formula.training.models import CompleteAGIModel')
    ]
    
    for test_name, import_statement in core_tests:
        try:
            exec(import_statement)
            validation_results['core_modules'][test_name] = 'PASS'
        except Exception as e:
            validation_results['core_modules'][test_name] = f'FAIL: {e}'
    
    # Test breakthrough optimizations
    optimization_tests = [
        ('Sparse Neural Networks', 'from agi_formula.optimization.sparse_neural_breakthrough import SparseAGINetwork'),
        ('Neuromorphic Processing', 'from agi_formula.optimization.neuromorphic_agi import NeuromorphicAGINetwork'),
        ('Consciousness Simulation', 'from agi_formula.cognitive.consciousness import ConsciousnessSimulator'),
        ('Causal Reasoning', 'from agi_formula.reasoning.causal_reasoning import CausalReasoningEngine')
    ]
    
    for test_name, import_statement in optimization_tests:
        try:
            exec(import_statement)
            validation_results['breakthrough_optimizations'][test_name] = 'PASS'
        except Exception as e:
            validation_results['breakthrough_optimizations'][test_name] = f'FAIL: {e}'
    
    # Test training components
    training_tests = [
        ('AGI Trainers', 'from agi_formula.training.trainers import AGITrainer'),
        ('AGI Optimizers', 'from agi_formula.training.optimizers import AdaptiveOptimizer'),
        ('AGI Datasets', 'from agi_formula.training.datasets import ConsciousnessDataset'),
        ('Training Utils', 'from agi_formula.training.utils import save_model, load_model')
    ]
    
    for test_name, import_statement in training_tests:
        try:
            exec(import_statement)
            validation_results['training_components'][test_name] = 'PASS'
        except Exception as e:
            validation_results['training_components'][test_name] = f'FAIL: {e}'
    
    # Calculate overall status
    all_tests = (list(validation_results['core_modules'].values()) + 
                list(validation_results['breakthrough_optimizations'].values()) +
                list(validation_results['training_components'].values()))
    
    passed_tests = len([test for test in all_tests if test == 'PASS'])
    total_tests = len(all_tests)
    success_rate = passed_tests / total_tests
    
    if success_rate >= 0.9:
        validation_results['overall_status'] = 'EXCELLENT'
    elif success_rate >= 0.7:
        validation_results['overall_status'] = 'GOOD'
    elif success_rate >= 0.5:
        validation_results['overall_status'] = 'PARTIAL'
    else:
        validation_results['overall_status'] = 'POOR'
    
    # Print results
    print(f"Core Modules: {len([v for v in validation_results['core_modules'].values() if v == 'PASS'])}/{len(validation_results['core_modules'])} PASS")
    print(f"Optimizations: {len([v for v in validation_results['breakthrough_optimizations'].values() if v == 'PASS'])}/{len(validation_results['breakthrough_optimizations'])} PASS")
    print(f"Training: {len([v for v in validation_results['training_components'].values() if v == 'PASS'])}/{len(validation_results['training_components'])} PASS")
    print(f"\\nOverall Status: {validation_results['overall_status']}")
    print(f"Success Rate: {success_rate:.1%}")
    
    return validation_results

# Convenience functions for quick usage
def quick_agi_train(model_type: str = "complete", 
                   dataset_type: str = "consciousness",
                   epochs: int = 50,
                   **kwargs) -> Dict[str, Any]:
    """Quick AGI training function - one-liner training"""
    
    from .models import AGIModel
    from .datasets import create_dataset
    from .trainers import train_agi_model
    
    # Create model
    model = AGIModel(model_type, **kwargs)
    
    # Create dataset
    train_data = create_dataset(dataset_type, size=1000)
    val_data = create_dataset(dataset_type, size=200)
    
    # Train
    results = train_agi_model(model, train_data, val_data, epochs=epochs)
    
    return {
        'model': model,
        'results': results,
        'performance': results.get('final_performance', 0.0)
    }

def benchmark_agi_performance(model_type: str = "complete",
                             num_trials: int = 3) -> Dict[str, Any]:
    """Benchmark AGI model performance"""
    
    print(f"BENCHMARKING {model_type.upper()} AGI MODEL")
    print("=" * 40)
    
    results = []
    
    for trial in range(num_trials):
        print(f"Trial {trial + 1}/{num_trials}...")
        
        trial_result = quick_agi_train(
            model_type=model_type,
            epochs=20,
            num_neurons=500,
            consciousness_level=0.8
        )
        
        results.append(trial_result)
    
    # Calculate statistics
    performances = [r['performance'] for r in results]
    
    benchmark_results = {
        'model_type': model_type,
        'num_trials': num_trials,
        'avg_performance': np.mean(performances),
        'std_performance': np.std(performances),
        'best_performance': min(performances),  # Lower loss is better
        'worst_performance': max(performances),
        'trials': results
    }
    
    print(f"\\nBENCHMARK RESULTS:")
    print(f"Average Performance: {benchmark_results['avg_performance']:.4f}")
    print(f"Best Performance: {benchmark_results['best_performance']:.4f}")
    print(f"Performance Std: {benchmark_results['std_performance']:.4f}")
    
    return benchmark_results