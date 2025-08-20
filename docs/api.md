# AGI-Formula API Reference

This document provides comprehensive API documentation for all AGI-Formula modules and components.

## Core Components

### `agi_formula.core`

#### `Component`
Base class for all AGI components with consciousness integration.

```python
class Component:
    def __init__(self):
        self._consciousness_level = 0.5
        
    def forward(self, x):
        """Forward pass through the component"""
        raise NotImplementedError
        
    def variables(self):
        """Return all trainable variables"""
        return []
```

#### `Transform`
Consciousness-enhanced linear transformation.

```python
class Transform(Component):
    def __init__(self, input_dim: int, output_dim: int, bias: bool = True):
        """
        Args:
            input_dim: Input dimension size
            output_dim: Output dimension size  
            bias: Whether to include bias term
        """
```

#### `Activation`
Adaptive activation function with consciousness modulation.

```python
class Activation(Component):
    def __init__(self, activation_type: str = "relu"):
        """
        Args:
            activation_type: Type of activation ('relu', 'tanh', 'sigmoid', 'gelu')
        """
```

#### `MSELoss`
Mean squared error loss with consciousness weighting.

```python
class MSELoss:
    def __call__(self, prediction, target):
        """
        Args:
            prediction: Model predictions
            target: Target values
        Returns:
            Tensor: Computed loss
        """
```

### `agi_formula.tensor`

#### `tensor()`
Create AGI tensor with consciousness tracking.

```python
def tensor(data, dtype=np.float32, requires_grad=False):
    """
    Args:
        data: Input data (list, array, or scalar)
        dtype: Data type (default: np.float32)
        requires_grad: Whether tensor requires gradients
    Returns:
        Tensor: AGI tensor object
    """
```

#### `Tensor`
Core tensor class with consciousness integration.

```python
class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data)
        self.requires_grad = requires_grad
        self._consciousness_attention = np.ones_like(self.data) * 0.1
        
    def backward(self):
        """Compute gradients through computational graph"""
        
    def item(self):
        """Extract scalar value from tensor"""
```

## Consciousness System

### `agi_formula.consciousness`

#### `ConsciousState`
Represents the state of consciousness in AGI systems.

```python
class ConsciousState:
    def __init__(self, awareness_level=0.5):
        """
        Args:
            awareness_level: Initial consciousness level (0.0-1.0)
        """
        self.awareness_level = awareness_level
        self.attention_focus = {}
        self.experience_history = []
        
    def attend_to(self, stimulus, intensity=1.0):
        """Focus attention on stimulus"""
        
    def integrate_experience(self, experience):
        """Integrate new experience into consciousness"""
        
    def reflect(self):
        """Perform meta-cognitive reflection"""
```

#### `ConsciousAgent`
Autonomous agent with consciousness capabilities.

```python
class ConsciousAgent:
    def __init__(self, consciousness_level=0.5):
        """
        Args:
            consciousness_level: Initial consciousness level
        """
        
    def perceive(self, stimulus):
        """Perceive and process environmental stimulus"""
        
    def reason(self, problem, context=None):
        """Apply reasoning to problem with conscious awareness"""
        
    def learn(self, experience):
        """Learn from experience with conscious integration"""
        
    def meta_learn(self, learning_outcome):
        """Meta-learning that improves learning process"""
```

## Reasoning Engine

### `agi_formula.reasoning`

#### `ReasoningEngine`
Integrated multi-modal reasoning system.

```python
class ReasoningEngine:
    def __init__(self):
        self.logical_reasoner = LogicalReasoner()
        self.causal_reasoner = CausalReasoner()
        self.temporal_reasoner = TemporalReasoner()
        self.abstract_reasoner = AbstractReasoner()
        
    def reason(self, query, context=None, reasoning_types=None):
        """
        Perform integrated reasoning across all modalities
        
        Args:
            query: Question or problem to reason about
            context: Additional context information
            reasoning_types: List of reasoning types to use
            
        Returns:
            dict: Reasoning results with confidence scores
        """
```

#### `LogicalReasoner`
Handles logical inference and rule-based reasoning.

```python
class LogicalReasoner:
    def __init__(self):
        self.facts = set()
        self.rules = []
        
    def add_fact(self, fact):
        """Add fact to knowledge base"""
        
    def add_rule(self, premise, conclusion, confidence=1.0):
        """Add logical rule"""
        
    def infer(self, query):
        """Perform logical inference"""
```

#### `CausalReasoner`
Handles causal discovery and reasoning.

```python
class CausalReasoner:
    def __init__(self):
        self.causal_graph = {}
        self.interventions = []
        
    def add_causal_link(self, cause, effect, strength=1.0):
        """Add causal relationship"""
        
    def discover_causes(self, effect, observations):
        """Discover potential causes for effect"""
        
    def predict_effect(self, cause, intervention_value):
        """Predict effect of intervention"""
```

#### `TemporalReasoner`
Handles temporal and sequential reasoning.

```python
class TemporalReasoner:
    def __init__(self):
        self.temporal_sequences = []
        self.pattern_memory = {}
        
    def add_temporal_sequence(self, events, timestamps=None):
        """Add temporal sequence to memory"""
        
    def predict_next(self, current_sequence):
        """Predict next event in sequence"""
```

#### `AbstractReasoner`
Handles abstract reasoning and pattern recognition.

```python
class AbstractReasoner:
    def __init__(self):
        self.abstractions = {}
        self.analogy_mappings = []
        
    def create_abstraction(self, instances, abstraction_name):
        """Create abstract concept from instances"""
        
    def find_analogies(self, source_domain, target_domain):
        """Find analogical mappings between domains"""
```

## Intelligence System

### `agi_formula.intelligence`

#### `Intelligence`
Core intelligence system integrating consciousness and reasoning.

```python
class Intelligence:
    def __init__(self, consciousness_level=0.7):
        """
        Args:
            consciousness_level: Initial consciousness level
        """
        self.conscious_agent = ConsciousAgent(consciousness_level)
        self.reasoning_engine = ReasoningEngine()
        self.goal_system = GoalSystem()
        
    def perceive(self, stimulus):
        """Perceive and understand input stimulus"""
        
    def think(self, problem, context=None):
        """Think about problem using integrated reasoning"""
        
    def create(self, goal, constraints=None):
        """Generate creative solutions for goal"""
        
    def learn(self, experience):
        """Learn from experience with consciousness"""
        
    def adapt(self, environment_change):
        """Adapt behavior to environmental changes"""
```

#### `GoalSystem`
Hierarchical goal management system.

```python
class GoalSystem:
    def __init__(self):
        self.goals = []
        self.goal_hierarchy = {}
        
    def set_goal(self, goal_description, priority=0.5):
        """Set new goal with priority"""
        
    def achieve_goal(self, goal_id, strategy=None):
        """Work toward achieving specific goal"""
        
    def evaluate_progress(self, goal_id):
        """Evaluate progress toward goal"""
```

## Optimization

### `agi_formula.optim`

#### `Adam`
Adam optimizer with consciousness-aware updates.

```python
class Adam:
    def __init__(self, variables, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        """
        Args:
            variables: List of variables to optimize
            lr: Learning rate
            betas: Coefficients for computing running averages
            eps: Small constant for numerical stability
        """
        
    def zero_grad(self):
        """Reset all gradients to zero"""
        
    def step(self):
        """Perform optimization step"""
```

#### `SGD`
Stochastic gradient descent with consciousness modulation.

```python
class SGD:
    def __init__(self, variables, lr=0.01, momentum=0.0):
        """
        Args:
            variables: List of variables to optimize
            lr: Learning rate
            momentum: Momentum factor
        """
```

#### `AdamW`
AdamW optimizer with decoupled weight decay.

```python
class AdamW:
    def __init__(self, variables, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        """
        Args:
            variables: List of variables to optimize
            lr: Learning rate
            betas: Coefficients for computing running averages  
            eps: Small constant for numerical stability
            weight_decay: Weight decay coefficient
        """
```

#### `RMSprop`
RMSprop optimizer with adaptive learning rates.

```python
class RMSprop:
    def __init__(self, variables, lr=0.01, alpha=0.99, eps=1e-8):
        """
        Args:
            variables: List of variables to optimize
            lr: Learning rate
            alpha: Smoothing constant
            eps: Small constant for numerical stability
        """
```

#### `QuantumOptimizer`
Quantum-inspired optimization algorithm.

```python
class QuantumOptimizer:
    def __init__(self, variables, lr=0.01, quantum_strength=0.1):
        """
        Args:
            variables: List of variables to optimize
            lr: Learning rate
            quantum_strength: Strength of quantum fluctuations
        """
```

## Utility Functions

### `agi_formula.functional`

#### Activation Functions
```python
def relu(x):
    """ReLU activation function"""
    
def tanh(x):  
    """Hyperbolic tangent activation"""
    
def sigmoid(x):
    """Sigmoid activation function"""
    
def gelu(x):
    """Gaussian Error Linear Unit"""
```

#### Loss Functions
```python
def mse_loss(prediction, target):
    """Mean squared error loss"""
    
def cross_entropy_loss(prediction, target):
    """Cross entropy loss for classification"""
    
def huber_loss(prediction, target, delta=1.0):
    """Huber loss for robust regression"""
```

#### Tensor Operations
```python
def matmul(a, b):
    """Matrix multiplication"""
    
def softmax(x, dim=-1):
    """Softmax function"""
    
def layer_norm(x, eps=1e-5):
    """Layer normalization"""
```

### `agi_formula`

#### Utility Functions
```python
def randn(*shape):
    """Generate random tensor with normal distribution"""
    
def zeros(*shape):
    """Create tensor filled with zeros"""
    
def ones(*shape):
    """Create tensor filled with ones"""
    
def eye(n):
    """Create identity matrix"""
```

## Configuration

### Environment Variables
- `AGI_CONSCIOUSNESS_LEVEL`: Default consciousness level (0.0-1.0)
- `AGI_REASONING_DEPTH`: Default reasoning depth for complex queries
- `AGI_CREATIVITY_THRESHOLD`: Threshold for creative solution generation
- `AGI_META_LEARNING_RATE`: Rate of meta-learning adaptation

### Configuration File
Create `agi_config.yaml` to customize behavior:

```yaml
consciousness:
  default_level: 0.7
  evolution_rate: 0.1
  
reasoning:
  max_depth: 10
  confidence_threshold: 0.5
  
intelligence:
  creativity_factor: 0.3
  goal_priority_decay: 0.95
  
optimization:
  default_lr: 0.001
  consciousness_modulation: true
```

## Examples

### Basic Usage
```python
import agi_formula as agi

# Create conscious model
model = agi.core.Transform(10, 5)
optimizer = agi.optim.Adam(model.variables())

# Training with consciousness evolution
for epoch in range(100):
    x = agi.randn(32, 10)
    y = agi.randn(32, 5)
    
    pred = model(x)
    loss = agi.core.MSELoss()(pred, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss={loss.item():.4f}, "
              f"Consciousness={model._consciousness_level:.3f}")
```

### ARC-AGI Pattern Recognition
```python
# Create AGI intelligence for pattern recognition
intelligence = agi.Intelligence(consciousness_level=0.8)

# Process ARC-AGI style pattern
pattern = agi.tensor([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
perceived, concepts = intelligence.perceive(pattern)

print(f"Extracted {len(concepts)} pattern concepts")

# Generate creative completions
creative_solutions = intelligence.create(
    "Complete pattern maintaining symmetry",
    constraints={"grid_size": 3, "binary_values": True}
)
```

### Multi-Modal Reasoning
```python
# Initialize reasoning engine
reasoning_engine = agi.ReasoningEngine()

# Add facts and rules
reasoning_engine.logical_reasoner.add_fact("sky_is_blue")
reasoning_engine.logical_reasoner.add_rule("sky_is_blue", "weather_is_clear", 0.8)

# Perform causal reasoning
observations = [
    {"variables": {"temperature": 25, "humidity": 60, "rain": False}},
    {"variables": {"temperature": 15, "humidity": 80, "rain": True}}
]
causes = reasoning_engine.causal_reasoner.discover_causes("rain", observations)

# Temporal prediction
sequence = ["sunny", "cloudy", "rainy"]
predictions = reasoning_engine.temporal_reasoner.predict_next(sequence)
```

This API reference covers the main components of AGI-Formula. For more detailed examples and advanced usage patterns, see the [examples directory](../examples/) and [comprehensive benchmarks](../temp_testing/).