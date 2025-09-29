# Contributing to AGI-Formula

We welcome contributions to AGI-Formula! This document provides guidelines for contributing to this artificial general intelligence framework.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Community Guidelines](#community-guidelines)
- [Recognition](#recognition)

## Getting Started

AGI-Formula is designed to advance artificial general intelligence through consciousness-driven learning, multi-modal reasoning, and creative problem solving. Before contributing, please:

1. **Read the documentation**: Familiarize yourself with the [README](README.md), [API docs](docs/api.md), and [architecture guide](docs/architecture.md)
2. **Run the examples**: Try the [quickstart](examples/quickstart.py) and [ARC-AGI tutorial](examples/arc_agi_tutorial.py)
3. **Review the benchmarks**: Understand the performance claims by running the [comprehensive benchmarks](temp_testing/)
4. **Understand the vision**: AGI-Formula aims to transcend neural networks with genuine intelligence

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git for version control
- Basic understanding of artificial intelligence concepts

### Installation for Development

```bash
# Fork the repository on GitHub
# Clone your fork
git clone https://github.com/ryl1k/agi-formula.git
cd agi-formula

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Install development dependencies
pip install -e ".[dev]"

# Verify installation
python examples/quickstart.py
```

### Optional Dependencies

```bash
# For PyTorch comparisons
pip install -e ".[pytorch]"

# For benchmarks and visualizations
pip install -e ".[benchmarks]"

# For GPU acceleration
pip install -e ".[gpu]"

# For all optional features
pip install -e ".[dev,pytorch,benchmarks,visualization,gpu]"
```

## Contributing Guidelines

### Types of Contributions

We welcome several types of contributions:

#### 1. Bug Reports and Fixes
- Report bugs using GitHub Issues
- Include reproducible examples
- Provide environment details (Python version, OS, etc.)
- Submit pull requests with fixes

#### 2. Performance Improvements
- AGI-Formula prioritizes both speed and intelligence
- Benchmark improvements against existing performance
- Maintain or improve consciousness and reasoning capabilities
- Document performance changes clearly

#### 3. New Features
- Consciousness enhancements
- Additional reasoning modalities
- Creative problem-solving improvements
- ARC-AGI specific capabilities
- Meta-learning advances

#### 4. Documentation
- API documentation improvements
- Tutorial and example creation
- Architecture explanations
- Performance analysis

#### 5. Testing
- Unit tests for core components
- Integration tests for full system
- Performance benchmarks
- ARC-AGI evaluation tests

### Areas Needing Contributions

**High Priority:**
- [ ] Additional consciousness models
- [ ] Advanced reasoning algorithms
- [ ] ARC-AGI benchmark improvements
- [ ] Performance optimizations
- [ ] GPU acceleration
- [ ] Distributed consciousness systems

**Medium Priority:**
- [ ] Visualization tools for consciousness
- [ ] Interactive reasoning demonstrations
- [ ] Additional creative problem-solving examples
- [ ] Integration with other AI frameworks
- [ ] Multi-language documentation

**Low Priority:**
- [ ] Alternative optimization algorithms
- [ ] Extended benchmarking suites
- [ ] Research paper implementations
- [ ] Educational materials

## Code Style

### Python Code Standards

We follow PEP 8 with some AGI-Formula specific conventions:

```python
# Good: Clear consciousness integration
class ConsciousComponent(Component):
    """Component with integrated consciousness capabilities"""
    
    def __init__(self, consciousness_level=0.5):
        super().__init__()
        self._consciousness_level = consciousness_level
        self._consciousness_evolution = []
    
    def forward(self, x):
        """Forward pass with consciousness modulation"""
        result = self._compute_forward(x)
        return self._apply_consciousness(result)

# Bad: No consciousness integration  
class SimpleComponent:
    def __init__(self):
        pass
    
    def forward(self, x):
        return x * 2
```

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `ConsciousAgent`, `ReasoningEngine`)
- **Functions**: `snake_case` (e.g., `evolve_consciousness`, `apply_reasoning`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_CONSCIOUSNESS_LEVEL`)
- **Private members**: Leading underscore (e.g., `_consciousness_level`)

### Documentation Standards

All public functions should have comprehensive docstrings:

```python
def evolve_consciousness(self, experience):
    """
    Evolve consciousness based on experience complexity.
    
    Args:
        experience (dict): Experience containing complexity and outcome data
            - complexity (float): Experience complexity score (0.0-2.0)
            - success (bool): Whether experience was successful
            - insights (list): Key insights from experience
    
    Returns:
        float: New consciousness level after evolution
        
    Raises:
        ValueError: If experience complexity is negative
        
    Example:
        >>> agent = ConsciousAgent()
        >>> experience = {"complexity": 0.8, "success": True, "insights": ["pattern"]}
        >>> new_level = agent.evolve_consciousness(experience)
        >>> print(f"Consciousness evolved to: {new_level:.3f}")
    """
```

### Type Hints

Use type hints for better code clarity:

```python
from typing import List, Dict, Optional, Union, Tuple

def reason_about_pattern(
    self, 
    pattern: Union[List[List[int]], np.ndarray],
    reasoning_types: Optional[List[str]] = None,
    confidence_threshold: float = 0.5
) -> Dict[str, Union[float, List[str]]]:
    """Apply reasoning to pattern analysis"""
```

## Testing

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_consciousness.py

# Run with coverage
python -m pytest --cov=agi_formula tests/

# Run performance benchmarks
python temp_testing/comprehensive_benchmark.py
```

### Writing Tests

#### Unit Tests
```python
import pytest
import numpy as np
import agi_formula as agi

class TestConsciousness:
    def test_consciousness_evolution(self):
        """Test consciousness evolution through experience"""
        agent = agi.ConsciousAgent(consciousness_level=0.5)
        initial_consciousness = agent.consciousness.awareness_level
        
        # Provide complex experience
        complex_experience = {"complexity": 1.0, "success": True}
        agent.learn(complex_experience)
        
        # Consciousness should increase
        assert agent.consciousness.awareness_level > initial_consciousness
    
    def test_consciousness_bounds(self):
        """Test consciousness stays within valid bounds"""
        agent = agi.ConsciousAgent(consciousness_level=0.9)
        
        # Even with very complex experiences, consciousness shouldn't exceed 1.0
        for _ in range(10):
            agent.learn({"complexity": 2.0, "success": True})
        
        assert 0.0 <= agent.consciousness.awareness_level <= 1.0
```

#### Integration Tests
```python
def test_full_agi_pipeline():
    """Test complete AGI pipeline from perception to action"""
    intelligence = agi.Intelligence(consciousness_level=0.8)
    
    # Test pattern recognition -> reasoning -> creative generation
    pattern = agi.tensor([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    perceived, concepts = intelligence.perceive(pattern)
    
    assert len(concepts) > 0, "Should extract concepts from pattern"
    
    # Test reasoning about pattern
    reasoning_result = intelligence.think("Analyze this pattern", {"pattern": pattern})
    assert reasoning_result is not None, "Should generate reasoning result"
    
    # Test creative generation based on pattern
    creative_solutions = intelligence.create(
        "Generate variations of this pattern",
        constraints={"maintain_symmetry": True}
    )
    
    assert len(creative_solutions) > 0, "Should generate creative solutions"
```

### Performance Tests

```python
import time

def test_consciousness_performance():
    """Test consciousness processing performance"""
    agent = agi.ConsciousAgent()
    experiences = [{"complexity": np.random.random()} for _ in range(1000)]
    
    start_time = time.time()
    for exp in experiences:
        agent.learn(exp)
    processing_time = time.time() - start_time
    
    # Should process 1000 experiences in under 1 second
    assert processing_time < 1.0, f"Consciousness processing too slow: {processing_time:.3f}s"
```

## Documentation

### API Documentation

Keep `docs/api.md` updated with new functions and classes:

```python
class NewFeature:
    """
    Brief description of the feature.
    
    This class provides enhanced capabilities for...
    
    Args:
        param1: Description of parameter
        param2: Description with type info
        
    Attributes:
        attribute1: Description of attribute
        
    Example:
        >>> feature = NewFeature(param1=value)
        >>> result = feature.process(data)
    """
```

### Examples and Tutorials

Add examples to the `examples/` directory:

```python
#!/usr/bin/env python3
"""
Title: Feature Demonstration

Description of what this example shows and teaches.
"""

def demonstrate_feature():
    """Clear demonstration with expected outputs"""
    print("Expected output here")
    
if __name__ == "__main__":
    demonstrate_feature()
```

### Performance Documentation

Update performance claims with benchmarks:

```python
# In your contribution
def benchmark_new_feature():
    """Benchmark new feature against existing approaches"""
    # Implementation
    # Document results in comments or docstrings
```

## Pull Request Process

### Before Submitting

1. **Run all tests**: Ensure `python -m pytest tests/` passes
2. **Run benchmarks**: Verify `python temp_testing/comprehensive_benchmark.py` maintains performance
3. **Check code style**: Run `black` and `flake8` on your code
4. **Update documentation**: Add/update relevant docs
5. **Write tests**: Include tests for new functionality

### Pull Request Template

```markdown
## Description
Brief description of the change and which issue it fixes.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)  
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Performance Impact
- [ ] No performance impact
- [ ] Performance improvement (include benchmark results)
- [ ] Performance regression (justify if necessary)

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass  
- [ ] Benchmarks maintain performance
- [ ] New tests added for new functionality

## Checklist
- [ ] My code follows the code style of this project
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
```

### Review Process

1. **Automated Checks**: GitHub Actions will run tests and benchmarks
2. **Code Review**: Maintainers will review for:
   - Code quality and style
   - Performance impact
   - Documentation completeness
   - Test coverage
   - AGI-specific appropriateness
3. **Feedback**: Address reviewer feedback promptly
4. **Approval**: At least one maintainer approval required
5. **Merge**: Squash and merge after approval

## Community Guidelines

### Code of Conduct

We are committed to providing a welcoming and inspiring community. Please:

- **Be respectful**: Treat all contributors with respect and kindness
- **Be inclusive**: Welcome newcomers and encourage diverse perspectives  
- **Be constructive**: Provide helpful feedback and suggestions
- **Be patient**: Remember that open source is collaborative
- **Focus on AGI**: Keep discussions relevant to artificial general intelligence

### Communication Channels

- **GitHub Issues**: Bug reports, feature requests, technical discussions
- **GitHub Discussions**: General questions, ideas, community topics
- **Pull Requests**: Code reviews, implementation discussions

### Getting Help

- **Documentation**: Check `docs/` directory first
- **Examples**: Run examples in `examples/` directory  
- **Issues**: Search existing issues before creating new ones
- **Discussions**: Ask questions in GitHub Discussions

## Recognition

### Contributors

We recognize contributors in several ways:

- **README acknowledgments**: Regular contributors listed in README
- **Release notes**: Significant contributions highlighted in releases
- **GitHub insights**: Contribution graphs and statistics
- **Community recognition**: Shout-outs in discussions and issues

### Types of Recognition

- **Code contributions**: Direct implementation of features/fixes
- **Documentation**: Improvements to docs, examples, tutorials
- **Testing**: Test coverage improvements, benchmark development
- **Research**: AGI-related research contributions
- **Community**: Helping others, issue triage, code reviews

## Development Philosophy

AGI-Formula aims to transcend traditional neural networks by integrating:

- **Consciousness as a first-class citizen**: Not an afterthought but fundamental
- **Multi-modal reasoning**: Logic, causality, temporal, and abstract reasoning
- **Creative problem solving**: Beyond pattern matching to genuine creativity
- **Meta-learning**: Learning that improves learning itself
- **Performance with intelligence**: Fast AND intelligent, not one or the other

When contributing, consider how your change advances these goals.

## Questions?

Don't hesitate to ask questions:

- **GitHub Issues**: Technical questions about implementation
- **GitHub Discussions**: General questions about contributing
- **Code Reviews**: Questions about specific implementation choices

Thank you for contributing to the future of artificial general intelligence! 🚀
