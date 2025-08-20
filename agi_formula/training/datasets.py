"""
AGI Datasets - Specialized datasets for AGI training

Includes datasets designed specifically for AGI capabilities like consciousness,
causal reasoning, and meta-learning.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Iterator
from abc import ABC, abstractmethod
import random
from dataclasses import dataclass

@dataclass
class AGIDataPoint:
    """Single data point for AGI training"""
    inputs: np.ndarray
    targets: Optional[np.ndarray] = None
    
    # AGI-specific annotations
    consciousness_level: float = 0.0
    causal_relations: List[Tuple[str, str]] = None
    meta_learning_context: Dict[str, Any] = None
    attention_focus: List[int] = None
    
    def __post_init__(self):
        if self.causal_relations is None:
            self.causal_relations = []
        if self.meta_learning_context is None:
            self.meta_learning_context = {}
        if self.attention_focus is None:
            self.attention_focus = []

class BaseAGIDataset(ABC):
    """Base class for AGI datasets"""
    
    def __init__(self, size: int = 1000, seed: int = 42):
        self.size = size
        self.seed = seed
        self.data = []
        self.current_index = 0
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        random.seed(seed)
        
        # Generate dataset
        self._generate_data()
    
    @abstractmethod
    def _generate_data(self):
        """Generate dataset - to be implemented by subclasses"""
        pass
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> AGIDataPoint:
        return self.data[idx]
    
    def __iter__(self) -> Iterator[AGIDataPoint]:
        self.current_index = 0
        return self
    
    def __next__(self) -> AGIDataPoint:
        if self.current_index >= len(self.data):
            raise StopIteration
        
        data_point = self.data[self.current_index]
        self.current_index += 1
        return data_point
    
    def batch_generator(self, batch_size: int = 32, shuffle: bool = True) -> Iterator[List[AGIDataPoint]]:
        """Generate batches of data points"""
        indices = list(range(len(self.data)))
        
        if shuffle:
            random.shuffle(indices)
        
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            batch = [self.data[idx] for idx in batch_indices]
            yield batch
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        if not self.data:
            return {}
        
        consciousness_levels = [dp.consciousness_level for dp in self.data]
        causal_counts = [len(dp.causal_relations) for dp in self.data]
        
        return {
            'size': len(self.data),
            'avg_consciousness': np.mean(consciousness_levels),
            'avg_causal_relations': np.mean(causal_counts),
            'input_shape': self.data[0].inputs.shape if self.data[0].inputs is not None else None,
            'target_shape': self.data[0].targets.shape if self.data[0].targets is not None else None
        }

class ConsciousnessDataset(BaseAGIDataset):
    """Dataset for training consciousness capabilities"""
    
    def __init__(self, size: int = 1000, input_dim: int = 100, 
                 consciousness_complexity: str = "medium"):
        self.input_dim = input_dim
        self.consciousness_complexity = consciousness_complexity
        super().__init__(size)
    
    def _generate_data(self):
        """Generate consciousness training data"""
        
        for i in range(self.size):
            # Generate input with varying complexity
            if self.consciousness_complexity == "simple":
                inputs = self._generate_simple_inputs()
                consciousness_level = 0.3
            elif self.consciousness_complexity == "complex":
                inputs = self._generate_complex_inputs()
                consciousness_level = 0.9
            else:  # medium
                inputs = self._generate_medium_inputs()
                consciousness_level = 0.6
            
            # Add noise based on consciousness requirement
            consciousness_level += np.random.normal(0, 0.1)
            consciousness_level = max(0.0, min(1.0, consciousness_level))
            
            # Generate targets (awareness indicators)
            targets = self._generate_consciousness_targets(inputs, consciousness_level)
            
            # Generate attention focus (most salient inputs)
            attention_focus = self._generate_attention_focus(inputs)
            
            data_point = AGIDataPoint(
                inputs=inputs,
                targets=targets,
                consciousness_level=consciousness_level,
                attention_focus=attention_focus,
                meta_learning_context={'complexity': self.consciousness_complexity}
            )
            
            self.data.append(data_point)
    
    def _generate_simple_inputs(self) -> np.ndarray:
        """Generate simple input patterns"""
        inputs = np.random.randn(self.input_dim) * 0.5
        
        # Add some clear patterns
        inputs[0:10] = np.sin(np.linspace(0, 2*np.pi, 10))  # Sine pattern
        
        return inputs
    
    def _generate_medium_inputs(self) -> np.ndarray:
        """Generate medium complexity inputs"""
        inputs = np.random.randn(self.input_dim)
        
        # Add multiple patterns
        inputs[0:20] = np.sin(np.linspace(0, 4*np.pi, 20))  # Complex sine
        inputs[20:40] = np.random.choice([0, 1], 20)        # Binary pattern
        inputs[40:60] = np.linspace(-1, 1, 20)             # Linear pattern
        
        return inputs
    
    def _generate_complex_inputs(self) -> np.ndarray:
        """Generate complex input patterns requiring high consciousness"""
        inputs = np.random.randn(self.input_dim)
        
        # Add multiple interacting patterns
        t = np.linspace(0, 4*np.pi, self.input_dim)
        inputs += 0.3 * np.sin(t) * np.cos(2*t)  # Modulated sine
        inputs += 0.2 * np.random.choice([-1, 0, 1], self.input_dim)  # Sparse noise
        
        # Add dependencies between different parts
        for i in range(0, self.input_dim-10, 20):
            inputs[i:i+10] = inputs[i:i+10] + 0.1 * np.sum(inputs[i+10:i+20])
        
        return inputs
    
    def _generate_consciousness_targets(self, inputs: np.ndarray, consciousness_level: float) -> np.ndarray:
        """Generate targets for consciousness training"""
        targets = np.zeros(10)  # 10-dimensional consciousness output
        
        # Target 0: Overall consciousness level
        targets[0] = consciousness_level
        
        # Target 1-3: Attention distribution
        salient_indices = np.argsort(np.abs(inputs))[-3:]
        targets[1] = salient_indices[0] / len(inputs)
        targets[2] = salient_indices[1] / len(inputs)  
        targets[3] = salient_indices[2] / len(inputs)
        
        # Target 4-6: Pattern recognition
        targets[4] = 1.0 if np.std(inputs[:20]) > 0.5 else 0.0  # Variability detection
        targets[5] = 1.0 if np.corrcoef(inputs[:50], inputs[50:100])[0,1] > 0.3 else 0.0  # Correlation detection
        targets[6] = 1.0 if len(np.where(np.abs(inputs) > 2)[0]) > 5 else 0.0  # Outlier detection
        
        # Target 7-9: Higher-order awareness
        targets[7] = consciousness_level * np.mean(np.abs(inputs))  # Intensity awareness
        targets[8] = consciousness_level * (1.0 if np.sum(inputs) > 0 else 0.0)  # Valence awareness
        targets[9] = consciousness_level * min(1.0, np.std(inputs))  # Complexity awareness
        
        return targets
    
    def _generate_attention_focus(self, inputs: np.ndarray) -> List[int]:
        """Generate attention focus indices"""
        # Focus on most salient inputs
        salience = np.abs(inputs)
        top_indices = np.argsort(salience)[-5:]  # Top 5 most salient
        return top_indices.tolist()

class CausalDataset(BaseAGIDataset):
    """Dataset for training causal reasoning capabilities"""
    
    def __init__(self, size: int = 1000, num_variables: int = 10, 
                 max_causal_depth: int = 3):
        self.num_variables = num_variables
        self.max_causal_depth = max_causal_depth
        super().__init__(size)
    
    def _generate_data(self):
        """Generate causal reasoning data"""
        
        for i in range(self.size):
            # Generate causal structure
            causal_graph = self._generate_causal_graph()
            
            # Generate data following causal structure
            inputs, causal_relations = self._generate_causal_data(causal_graph)
            
            # Generate targets (causal predictions)
            targets = self._generate_causal_targets(inputs, causal_graph)
            
            data_point = AGIDataPoint(
                inputs=inputs,
                targets=targets,
                causal_relations=causal_relations,
                meta_learning_context={'causal_graph': causal_graph}
            )
            
            self.data.append(data_point)
    
    def _generate_causal_graph(self) -> Dict[int, List[int]]:
        """Generate random causal graph structure"""
        graph = {i: [] for i in range(self.num_variables)}
        
        # Add causal connections
        for i in range(self.num_variables):
            # Each variable can have 0-3 parents
            num_parents = np.random.poisson(1)
            num_parents = min(num_parents, 3)
            
            # Choose parents (must be earlier in topological order)
            possible_parents = list(range(i))
            if possible_parents:
                parents = np.random.choice(
                    possible_parents, 
                    size=min(num_parents, len(possible_parents)), 
                    replace=False
                )
                for parent in parents:
                    graph[parent].append(i)
        
        return graph
    
    def _generate_causal_data(self, causal_graph: Dict[int, List[int]]) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
        """Generate data following causal structure"""
        values = np.random.randn(self.num_variables) * 0.5
        causal_relations = []
        
        # Generate values following causal dependencies
        for cause in range(self.num_variables):
            for effect in causal_graph[cause]:
                # Effect depends on cause
                causal_strength = np.random.uniform(0.3, 0.8)
                noise = np.random.randn() * 0.2
                
                values[effect] += causal_strength * values[cause] + noise
                causal_relations.append((f"var_{cause}", f"var_{effect}"))
        
        return values, causal_relations
    
    def _generate_causal_targets(self, inputs: np.ndarray, causal_graph: Dict[int, List[int]]) -> np.ndarray:
        """Generate targets for causal learning"""
        # Target is adjacency matrix of causal graph
        targets = np.zeros((self.num_variables, self.num_variables))
        
        for cause, effects in causal_graph.items():
            for effect in effects:
                targets[cause, effect] = 1.0
        
        return targets.flatten()

class MetaLearningDataset(BaseAGIDataset):
    """Dataset for training meta-learning capabilities"""
    
    def __init__(self, size: int = 1000, num_tasks: int = 50, 
                 task_complexity: str = "varied"):
        self.num_tasks = num_tasks
        self.task_complexity = task_complexity
        self.task_types = ['classification', 'regression', 'pattern_completion', 'sequence_prediction']
        super().__init__(size)
    
    def _generate_data(self):
        """Generate meta-learning data"""
        
        # Generate task contexts
        task_contexts = self._generate_task_contexts()
        
        for i in range(self.size):
            # Choose random task context
            task_context = random.choice(task_contexts)
            
            # Generate data for this task
            inputs, targets = self._generate_task_data(task_context)
            
            # Generate meta-learning context
            meta_context = {
                'task_type': task_context['type'],
                'difficulty': task_context['difficulty'],
                'similar_tasks': task_context['similar_tasks'],
                'optimal_strategy': task_context['optimal_strategy']
            }
            
            data_point = AGIDataPoint(
                inputs=inputs,
                targets=targets,
                meta_learning_context=meta_context
            )
            
            self.data.append(data_point)
    
    def _generate_task_contexts(self) -> List[Dict[str, Any]]:
        """Generate different task contexts for meta-learning"""
        contexts = []
        
        for i in range(self.num_tasks):
            task_type = random.choice(self.task_types)
            difficulty = random.uniform(0.2, 1.0)
            
            # Generate similar tasks
            similar_tasks = [j for j in range(self.num_tasks) if abs(i - j) <= 3 and j != i]
            
            # Determine optimal strategy based on task type and difficulty
            if task_type == 'classification' and difficulty > 0.7:
                optimal_strategy = 'high_attention_focused'
            elif task_type == 'regression' and difficulty < 0.4:
                optimal_strategy = 'low_learning_rate_smooth'
            else:
                optimal_strategy = 'balanced_exploration'
            
            context = {
                'id': i,
                'type': task_type,
                'difficulty': difficulty,
                'similar_tasks': similar_tasks,
                'optimal_strategy': optimal_strategy
            }
            
            contexts.append(context)
        
        return contexts
    
    def _generate_task_data(self, task_context: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Generate data for specific task"""
        task_type = task_context['type']
        difficulty = task_context['difficulty']
        
        if task_type == 'classification':
            inputs, targets = self._generate_classification_task(difficulty)
        elif task_type == 'regression':
            inputs, targets = self._generate_regression_task(difficulty)
        elif task_type == 'pattern_completion':
            inputs, targets = self._generate_pattern_task(difficulty)
        else:  # sequence_prediction
            inputs, targets = self._generate_sequence_task(difficulty)
        
        return inputs, targets
    
    def _generate_classification_task(self, difficulty: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generate classification task"""
        dim = 20
        inputs = np.random.randn(dim)
        
        # Create classification boundary based on difficulty
        if difficulty < 0.3:  # Easy
            targets = np.array([1.0 if inputs[0] > 0 else 0.0])
        elif difficulty < 0.7:  # Medium
            targets = np.array([1.0 if np.sum(inputs[:5]) > 0 else 0.0])
        else:  # Hard
            targets = np.array([1.0 if np.dot(inputs[:10], inputs[10:20]) > 0 else 0.0])
        
        return inputs, targets
    
    def _generate_regression_task(self, difficulty: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generate regression task"""
        dim = 15
        inputs = np.random.randn(dim)
        
        if difficulty < 0.3:  # Easy
            targets = np.array([inputs[0] * 2 + 1])
        elif difficulty < 0.7:  # Medium
            targets = np.array([np.sum(inputs[:5]) * 0.5])
        else:  # Hard
            targets = np.array([np.sin(np.sum(inputs[:10])) * np.cos(np.sum(inputs[10:]))])
        
        return inputs, targets
    
    def _generate_pattern_task(self, difficulty: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generate pattern completion task"""
        length = 20
        
        if difficulty < 0.3:  # Easy - simple repetition
            pattern = np.tile([1, 0, -1], length // 3 + 1)[:length]
        elif difficulty < 0.7:  # Medium - arithmetic progression
            pattern = np.linspace(-1, 1, length)
        else:  # Hard - complex pattern
            t = np.linspace(0, 4*np.pi, length)
            pattern = np.sin(t) * np.cos(2*t)
        
        # Mask some elements for completion
        mask_size = int(length * 0.3)
        mask_indices = np.random.choice(length, mask_size, replace=False)
        
        inputs = pattern.copy()
        inputs[mask_indices] = 0  # Mask elements
        
        targets = pattern[mask_indices]  # Target is masked elements
        
        return inputs, targets
    
    def _generate_sequence_task(self, difficulty: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generate sequence prediction task"""
        length = 10
        
        if difficulty < 0.3:  # Easy - arithmetic sequence
            sequence = np.arange(length) * 0.5
        elif difficulty < 0.7:  # Medium - geometric-like
            sequence = np.array([i**1.5 for i in range(length)]) * 0.1
        else:  # Hard - Fibonacci-like
            sequence = np.zeros(length)
            sequence[0], sequence[1] = 1, 1
            for i in range(2, length):
                sequence[i] = sequence[i-1] + sequence[i-2] * 0.6
        
        inputs = sequence[:-1]  # All but last
        targets = np.array([sequence[-1]])  # Predict last element
        
        return inputs, targets

class MultimodalDataset(BaseAGIDataset):
    """Dataset combining multiple modalities for AGI training"""
    
    def __init__(self, size: int = 1000, vision_dim: int = 64, 
                 audio_dim: int = 32, text_dim: int = 50):
        self.vision_dim = vision_dim
        self.audio_dim = audio_dim 
        self.text_dim = text_dim
        super().__init__(size)
    
    def _generate_data(self):
        """Generate multimodal data"""
        
        for i in range(self.size):
            # Generate vision data (simplified image)
            vision_data = self._generate_vision_data()
            
            # Generate audio data (simplified audio features)
            audio_data = self._generate_audio_data()
            
            # Generate text data (simplified text features)
            text_data = self._generate_text_data()
            
            # Combine modalities
            inputs = np.concatenate([vision_data, audio_data, text_data])
            
            # Generate cross-modal targets
            targets = self._generate_multimodal_targets(vision_data, audio_data, text_data)
            
            # Generate attention focus across modalities
            attention_focus = self._generate_multimodal_attention(inputs)
            
            data_point = AGIDataPoint(
                inputs=inputs,
                targets=targets,
                attention_focus=attention_focus,
                meta_learning_context={
                    'modalities': ['vision', 'audio', 'text'],
                    'vision_dim': self.vision_dim,
                    'audio_dim': self.audio_dim,
                    'text_dim': self.text_dim
                }
            )
            
            self.data.append(data_point)
    
    def _generate_vision_data(self) -> np.ndarray:
        """Generate simplified vision data"""
        # Create simple 2D patterns flattened
        size = int(np.sqrt(self.vision_dim))
        image = np.random.randn(size, size)
        
        # Add some structure
        center = size // 2
        for i in range(size):
            for j in range(size):
                distance = np.sqrt((i - center)**2 + (j - center)**2)
                image[i, j] += 0.5 * np.exp(-distance / 3)  # Add center blob
        
        return image.flatten()
    
    def _generate_audio_data(self) -> np.ndarray:
        """Generate simplified audio features"""
        # Simulate audio features like MFCCs
        audio_features = np.random.randn(self.audio_dim)
        
        # Add some temporal structure
        for i in range(1, self.audio_dim):
            audio_features[i] += 0.3 * audio_features[i-1]  # Temporal correlation
        
        return audio_features
    
    def _generate_text_data(self) -> np.ndarray:
        """Generate simplified text features"""
        # Simulate text embeddings
        text_features = np.random.randn(self.text_dim)
        
        # Add some semantic structure
        # Group features into "words"
        word_size = 5
        for i in range(0, self.text_dim - word_size, word_size):
            word_embedding = np.random.randn(word_size) * 0.8
            text_features[i:i+word_size] = word_embedding
        
        return text_features
    
    def _generate_multimodal_targets(self, vision: np.ndarray, audio: np.ndarray, text: np.ndarray) -> np.ndarray:
        """Generate targets requiring multimodal understanding"""
        targets = np.zeros(5)
        
        # Cross-modal similarity (ensure compatible dimensions)
        min_len = min(len(vision), len(audio))
        targets[0] = np.corrcoef(vision[:min_len], audio[:min_len])[0, 1] if min_len > 1 else 0.0
        
        min_len = min(len(audio), len(text))
        targets[1] = np.corrcoef(audio[:min_len], text[:min_len])[0, 1] if min_len > 1 else 0.0
        
        # Multimodal intensity
        targets[2] = (np.mean(np.abs(vision)) + np.mean(np.abs(audio)) + np.mean(np.abs(text))) / 3
        
        # Cross-modal attention
        targets[3] = 1.0 if np.max(vision) > np.max(audio) else 0.0
        targets[4] = 1.0 if np.max(text) > np.max(vision) else 0.0
        
        return targets
    
    def _generate_multimodal_attention(self, inputs: np.ndarray) -> List[int]:
        """Generate attention focus across modalities"""
        # Find most salient features across all modalities
        salience = np.abs(inputs)
        
        # Get top features from each modality
        vision_end = self.vision_dim
        audio_end = vision_end + self.audio_dim
        text_end = audio_end + self.text_dim
        
        vision_top = np.argmax(salience[:vision_end])
        audio_top = np.argmax(salience[vision_end:audio_end]) + vision_end
        text_top = np.argmax(salience[audio_end:text_end]) + audio_end
        
        return [vision_top, audio_top, text_top]

# Factory functions for easy dataset creation
def create_dataset(dataset_type: str, size: int = 1000, **kwargs):
    """Create dataset - PyTorch style factory function"""
    
    if dataset_type.lower() == "consciousness":
        return ConsciousnessDataset(size, **kwargs)
    elif dataset_type.lower() == "causal":
        return CausalDataset(size, **kwargs)
    elif dataset_type.lower() == "meta":
        return MetaLearningDataset(size, **kwargs)
    elif dataset_type.lower() == "multimodal":
        return MultimodalDataset(size, **kwargs)
    else:
        return ConsciousnessDataset(size, **kwargs)  # Default

# Convenience aliases
ConsciousDataset = ConsciousnessDataset
MetaDataset = MetaLearningDataset
MultiModalDataset = MultimodalDataset

# Add missing AGIDataset alias
AGIDataset = BaseAGIDataset