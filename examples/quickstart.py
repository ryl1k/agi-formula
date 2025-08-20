#!/usr/bin/env python3
"""
AGI-Formula Quick Start Example

This example demonstrates basic AGI-Formula usage including:
- Creating conscious AGI models
- Training with consciousness evolution
- Multi-modal reasoning
- Creative problem solving
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agi_formula as agi
import numpy as np

def basic_agi_example():
    """Basic AGI-Formula usage demonstration"""
    print("=== Basic AGI-Formula Example ===\n")
    
    # Create a simple conscious AGI model
    class SimpleAGI(agi.core.Component):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super().__init__()
            self.fc1 = agi.core.Transform(input_dim, hidden_dim)
            self.activation = agi.core.Activation()
            self.fc2 = agi.core.Transform(hidden_dim, output_dim)
            
        def forward(self, x):
            x = self.fc1(x)
            x = self.activation(x)
            x = self.fc2(x)
            return x
    
    # Initialize model
    model = SimpleAGI(input_dim=10, hidden_dim=20, output_dim=5)
    optimizer = agi.optim.Adam(model.variables(), lr=0.001)
    loss_fn = agi.core.MSELoss()
    
    print(f"Created AGI model with initial consciousness: {model._consciousness_level:.3f}")
    
    # Generate sample data
    X = agi.tensor(np.random.randn(32, 10))
    y = agi.tensor(np.random.randn(32, 5))
    
    # Training loop with consciousness tracking
    print("\nTraining with consciousness evolution:")
    print("Epoch | Loss    | Consciousness")
    print("------|---------|-------------")
    
    for epoch in range(20):
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(X)
        loss = loss_fn(predictions, y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track consciousness evolution
        if epoch % 5 == 0:
            print(f"{epoch:5d} | {loss.item():7.4f} | {model._consciousness_level:11.3f}")
    
    print(f"\nFinal consciousness level: {model._consciousness_level:.3f}")
    print(f"Consciousness growth: {model._consciousness_level - 0.5:.3f}")

def reasoning_example():
    """Demonstrate multi-modal reasoning capabilities"""
    print("\n=== Multi-Modal Reasoning Example ===\n")
    
    # Create reasoning engine
    reasoning_engine = agi.ReasoningEngine()
    
    # Logical reasoning
    print("1. Logical Reasoning:")
    reasoning_engine.logical_reasoner.add_fact("sky_is_blue")
    reasoning_engine.logical_reasoner.add_fact("weather_is_sunny")
    reasoning_engine.logical_reasoner.add_rule("sky_is_blue", "good_weather", confidence=0.8)
    reasoning_engine.logical_reasoner.add_rule("weather_is_sunny", "go_outside", confidence=0.9)
    
    logical_inferences = reasoning_engine.logical_reasoner.infer("weather_decision")
    print(f"   Generated {len(logical_inferences)} logical inferences")
    for inference in logical_inferences[:3]:
        print(f"   - {inference['conclusion']} (confidence: {inference['confidence']:.2f})")
    
    # Causal reasoning
    print("\n2. Causal Reasoning:")
    observations = [
        {'variables': {'temperature': 25, 'ice_cream_sales': 100}},
        {'variables': {'temperature': 30, 'ice_cream_sales': 150}},
        {'variables': {'temperature': 20, 'ice_cream_sales': 80}},
    ]
    
    causes = reasoning_engine.causal_reasoner.discover_causes('ice_cream_sales', observations)
    print(f"   Discovered {len(causes)} causal relationships")
    for cause in causes:
        print(f"   - {cause['cause']} → ice_cream_sales (strength: {cause['strength']:.2f})")
    
    # Temporal reasoning
    print("\n3. Temporal Reasoning:")
    sequence = ["morning", "work", "lunch", "work", "evening"]
    reasoning_engine.temporal_reasoner.add_temporal_sequence(sequence)
    
    test_sequence = ["morning", "work", "lunch"]
    predictions = reasoning_engine.temporal_reasoner.predict_next(test_sequence)
    print(f"   Next event predictions: {dict(predictions)}")

def consciousness_example():
    """Demonstrate consciousness system capabilities"""
    print("\n=== Consciousness System Example ===\n")
    
    # Create conscious agent
    agent = agi.ConsciousAgent(consciousness_level=0.4)
    
    print(f"Initial consciousness: {agent.consciousness.awareness_level:.3f}")
    
    # Simulate learning experiences
    experiences = [
        {"type": "simple_pattern", "complexity": 0.3},
        {"type": "medium_pattern", "complexity": 0.6}, 
        {"type": "complex_pattern", "complexity": 0.9},
        {"type": "very_complex", "complexity": 1.2},
    ]
    
    print("\nLearning from experiences:")
    for i, experience in enumerate(experiences):
        # Agent processes experience
        stimulus = agi.tensor(np.random.randn(10) * experience["complexity"])
        perceived = agent.perceive(stimulus)
        
        # Agent learns from experience
        agent.learn(experience)
        
        print(f"Experience {i+1}: {experience['type']} -> "
              f"Consciousness: {agent.consciousness.awareness_level:.3f}")
    
    print(f"\nFinal consciousness: {agent.consciousness.awareness_level:.3f}")
    print(f"Consciousness evolution: {agent.consciousness.awareness_level - 0.4:.3f}")

def intelligence_example():
    """Demonstrate creative intelligence capabilities"""
    print("\n=== Creative Intelligence Example ===\n")
    
    # Create intelligence system
    intelligence = agi.Intelligence(consciousness_level=0.8)
    
    # Pattern recognition
    print("1. Pattern Recognition:")
    pattern = agi.tensor([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    perceived, concepts = intelligence.perceive(pattern)
    print(f"   Extracted {len(concepts)} concepts from cross pattern")
    
    # Problem solving
    print("\n2. Problem Solving:")
    problem_context = {
        "problem_type": "optimization",
        "constraints": ["minimize_cost", "maximize_efficiency"],
        "domain": "logistics"
    }
    
    solution = intelligence.think("How to optimize delivery routes?", problem_context)
    if solution:
        print(f"   Solution method: {solution['method']}")
        print(f"   Confidence: {solution['confidence']:.3f}")
    else:
        print("   No solution found")
    
    # Creative generation
    print("\n3. Creative Generation:")
    creative_goal = "Design a new pattern based on symmetry principles"
    constraints = {"maintain_balance": True, "use_binary_values": True}
    
    creative_solutions = intelligence.create(creative_goal, constraints)
    print(f"   Generated {len(creative_solutions)} creative solutions")
    
    for i, solution in enumerate(creative_solutions[:2]):
        print(f"   Solution {i+1}: creativity={solution['creativity_score']:.2f}, "
              f"feasibility={solution['feasibility']:.2f}")

def arc_agi_example():
    """Demonstrate ARC-AGI style reasoning"""
    print("\n=== ARC-AGI Style Reasoning Example ===\n")
    
    intelligence = agi.Intelligence(consciousness_level=0.9)
    
    # Pattern completion task
    print("1. Pattern Completion Task:")
    incomplete_pattern = [
        [1, None, 1],
        [None, 1, None],
        [1, None, 1]
    ]
    
    print(f"   Incomplete pattern: {incomplete_pattern}")
    
    # Analyze pattern
    context = {
        "pattern_type": "completion",
        "missing_positions": [(0, 1), (1, 0), (1, 2), (2, 1)],
        "theme": "cross_pattern"
    }
    
    completion_solution = intelligence.think("Complete this symmetric pattern", context)
    if completion_solution:
        print(f"   Completion method: {completion_solution['method']}")
        print(f"   Confidence: {completion_solution['confidence']:.3f}")
    
    # Rule induction task
    print("\n2. Rule Induction Task:")
    examples = [
        {"input": [[1, 0]], "output": [[0, 1]]},
        {"input": [[1, 1, 0]], "output": [[0, 1, 1]]},
    ]
    
    print("   Learning from examples:")
    for i, example in enumerate(examples):
        print(f"   Example {i+1}: {example['input']} -> {example['output']}")
    
    rule_context = {
        "examples": examples,
        "transformation_type": "spatial",
        "pattern_type": "horizontal_flip"
    }
    
    rule_solution = intelligence.think("What rule transforms input to output?", rule_context)
    if rule_solution:
        print(f"   Discovered rule: {rule_solution['method']}")
        print(f"   Rule confidence: {rule_solution['confidence']:.3f}")

def main():
    """Run all examples"""
    print("AGI-Formula Quick Start Examples")
    print("=" * 50)
    
    try:
        basic_agi_example()
        reasoning_example()
        consciousness_example()
        intelligence_example()
        arc_agi_example()
        
        print("\n" + "=" * 50)
        print("✅ All examples completed successfully!")
        print("\nNext steps:")
        print("1. Explore the comprehensive benchmarks: python temp_testing/comprehensive_benchmark.py")
        print("2. Try ARC-AGI specific tests: python temp_testing/arc_agi_specific_test.py")
        print("3. Check out more examples in the examples/ directory")
        print("4. Read the documentation in docs/")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("Make sure you have installed AGI-Formula: pip install -e .")

if __name__ == "__main__":
    main()