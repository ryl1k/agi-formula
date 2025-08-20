"""
Advanced Training Examples for AGI-Formula

Complex, challenging training scenarios that test the full capabilities
of the AGI system including reasoning, composition, and adaptation.
"""

import numpy as np
import time
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import json
import itertools

# Import AGI-Formula components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agi_formula import Network, NetworkConfig, MaskedTrainer
from agi_formula.benchmarks.agi_benchmarks import AGIBenchmarks
from agi_formula.visualize.interactive_viz import AGIVisualizationSuite


@dataclass
class TrainingScenario:
    """Advanced training scenario definition"""
    name: str
    description: str
    data_generator: callable
    difficulty: str
    expected_capabilities: List[str]
    success_criteria: Dict[str, float]


class AdvancedDataGenerators:
    """
    Advanced data generators for challenging AGI training scenarios
    """
    
    @staticmethod
    def logical_reasoning_sequences(num_examples: int = 500, sequence_length: int = 10) -> List[np.ndarray]:
        """
        Generate logical reasoning sequences with complex patterns
        
        Examples: if-then rules, logical operators, nested conditions
        """
        data = []
        
        for _ in range(num_examples):
            # Create logical sequence with if-then patterns
            sequence = np.zeros(sequence_length)
            
            # Rule 1: If position 0 > 0.5, then position 2 = position 0 * 0.8
            if np.random.rand() > 0.5:
                sequence[0] = np.random.uniform(0.6, 1.0)
                sequence[2] = sequence[0] * 0.8 + np.random.normal(0, 0.05)
            else:
                sequence[0] = np.random.uniform(0.0, 0.4)
                sequence[2] = np.random.uniform(0.0, 0.3)
            
            # Rule 2: XOR pattern between positions 1 and 3
            sequence[1] = np.random.choice([0.2, 0.8])
            sequence[3] = 0.8 if sequence[1] < 0.5 else 0.2
            sequence[3] += np.random.normal(0, 0.05)
            
            # Rule 3: Sum constraint for positions 4-6
            target_sum = np.random.uniform(1.0, 2.0)
            weights = np.random.dirichlet([1, 1, 1])
            sequence[4:7] = weights * target_sum
            
            # Complex dependency: position 7 depends on multiple previous
            sequence[7] = (sequence[0] + sequence[3] - sequence[5]) / 2
            sequence[7] = np.clip(sequence[7], 0, 1)
            
            # Noise in remaining positions
            sequence[8:] = np.random.uniform(0, 1, sequence_length - 8)
            
            data.append(sequence)
        
        return data
    
    @staticmethod
    def causal_chain_problems(num_examples: int = 300, chain_length: int = 5) -> List[np.ndarray]:
        """
        Generate problems requiring deep causal reasoning
        
        Each example has a clear causal chain: A → B → C → D → E
        """
        data = []
        
        for _ in range(num_examples):
            sequence = np.zeros(chain_length * 2)  # Double length for context
            
            # Create causal chain
            cause_strength = np.random.uniform(0.7, 0.9)
            noise_level = np.random.uniform(0.05, 0.15)
            
            # Initial cause
            sequence[0] = np.random.uniform(0.2, 0.8)
            
            # Propagate through causal chain
            for i in range(1, chain_length):
                # Each effect depends on previous cause + some noise
                sequence[i] = sequence[i-1] * cause_strength + np.random.normal(0, noise_level)
                sequence[i] = np.clip(sequence[i], 0, 1)
            
            # Add confounding variables (correlated but not causal)
            for i in range(chain_length, len(sequence)):
                # Correlated with random chain element but not causal
                correlation_target = np.random.randint(0, chain_length)
                correlation_strength = np.random.uniform(0.3, 0.6)
                sequence[i] = (sequence[correlation_target] * correlation_strength + 
                              np.random.uniform(0, 1) * (1 - correlation_strength))
            
            data.append(sequence)
        
        return data
    
    @staticmethod
    def compositional_concept_learning(num_examples: int = 400) -> List[Dict[str, Any]]:
        """
        Generate compositional concept learning tasks
        
        Concepts: COLOR, SHAPE, SIZE, TEXTURE that can be combined
        """
        concepts = {
            'color': ['red', 'blue', 'green', 'yellow'],
            'shape': ['circle', 'square', 'triangle', 'diamond'],
            'size': ['small', 'medium', 'large'],
            'texture': ['smooth', 'rough', 'striped', 'dotted']
        }
        
        # Create concept encodings
        concept_encodings = {}
        encoding_dim = 3
        
        for category, items in concepts.items():
            concept_encodings[category] = {}
            for i, item in enumerate(items):
                # Use one-hot-like encoding with some overlap for similarity
                encoding = np.zeros(encoding_dim)
                encoding[i % encoding_dim] = 1.0
                if i > 0:
                    encoding[(i-1) % encoding_dim] = 0.3  # Similarity encoding
                concept_encodings[category][item] = encoding
        
        data = []
        
        for _ in range(num_examples):
            # Select random concepts
            selected_concepts = {}
            for category in concepts:
                selected_concepts[category] = np.random.choice(concepts[category])
            
            # Encode individual concepts
            individual_encodings = []
            for category in ['color', 'shape', 'size', 'texture']:
                encoding = concept_encodings[category][selected_concepts[category]]
                individual_encodings.extend(encoding)
            
            # Create composite encoding (more complex than sum)
            composite_encoding = np.array(individual_encodings)
            
            # Add interaction terms (concept combinations)
            color_shape_interaction = (concept_encodings['color'][selected_concepts['color']] * 
                                     concept_encodings['shape'][selected_concepts['shape']])
            size_texture_interaction = (concept_encodings['size'][selected_concepts['size']] * 
                                      concept_encodings['texture'][selected_concepts['texture']])
            
            # Combine all features
            full_encoding = np.concatenate([
                composite_encoding,
                color_shape_interaction,
                size_texture_interaction,
                [1.0 if selected_concepts['color'] == 'red' and selected_concepts['shape'] == 'circle' else 0.0],  # Special combination
                [1.0 if selected_concepts['size'] == 'large' and selected_concepts['texture'] == 'rough' else 0.0]   # Another special combination
            ])
            
            data.append({
                'encoding': full_encoding,
                'concepts': selected_concepts,
                'special_combinations': [
                    selected_concepts['color'] == 'red' and selected_concepts['shape'] == 'circle',
                    selected_concepts['size'] == 'large' and selected_concepts['texture'] == 'rough'
                ]
            })
        
        return data
    
    @staticmethod
    def meta_learning_tasks(num_tasks: int = 50, examples_per_task: int = 20) -> List[Dict[str, Any]]:
        """
        Generate meta-learning tasks for adaptation testing
        
        Each task is a different function the network needs to learn quickly
        """
        tasks = []
        
        task_types = [
            'linear_function',
            'quadratic_function',
            'sine_wave',
            'step_function',
            'exponential_decay',
            'polynomial'
        ]
        
        for task_id in range(num_tasks):
            task_type = np.random.choice(task_types)
            
            # Generate task-specific parameters
            if task_type == 'linear_function':
                slope = np.random.uniform(-2, 2)
                intercept = np.random.uniform(-1, 1)
                func = lambda x: slope * x + intercept
                
            elif task_type == 'quadratic_function':
                a = np.random.uniform(-1, 1)
                b = np.random.uniform(-1, 1)
                c = np.random.uniform(-1, 1)
                func = lambda x: a * x**2 + b * x + c
                
            elif task_type == 'sine_wave':
                amplitude = np.random.uniform(0.5, 2.0)
                frequency = np.random.uniform(0.5, 3.0)
                phase = np.random.uniform(0, 2*np.pi)
                func = lambda x: amplitude * np.sin(frequency * x + phase)
                
            elif task_type == 'step_function':
                threshold = np.random.uniform(-1, 1)
                low_value = np.random.uniform(-1, 0)
                high_value = np.random.uniform(0, 1)
                func = lambda x: high_value if x > threshold else low_value
                
            elif task_type == 'exponential_decay':
                scale = np.random.uniform(0.5, 2.0)
                decay_rate = np.random.uniform(0.5, 2.0)
                func = lambda x: scale * np.exp(-decay_rate * abs(x))
                
            else:  # polynomial
                coeffs = np.random.uniform(-0.5, 0.5, 4)
                func = lambda x: sum(c * x**i for i, c in enumerate(coeffs))
            
            # Generate examples for this task
            examples = []
            x_values = np.random.uniform(-2, 2, examples_per_task)
            
            for x in x_values:
                try:
                    y = func(x)
                    # Normalize y to reasonable range
                    y = np.clip(y, -3, 3)
                    examples.append({'input': x, 'output': y})
                except:
                    # Handle any mathematical errors
                    examples.append({'input': x, 'output': 0.0})
            
            tasks.append({
                'task_id': task_id,
                'task_type': task_type,
                'examples': examples,
                'function': func
            })
        
        return tasks
    
    @staticmethod
    def temporal_pattern_recognition(num_sequences: int = 200, sequence_length: int = 15) -> List[np.ndarray]:
        """
        Generate temporal sequences with complex patterns over time
        """
        sequences = []
        
        pattern_types = ['fibonacci_like', 'arithmetic_progression', 'geometric_progression', 
                        'alternating_pattern', 'complex_recursive']
        
        for _ in range(num_sequences):
            pattern_type = np.random.choice(pattern_types)
            sequence = np.zeros(sequence_length)
            
            if pattern_type == 'fibonacci_like':
                # Fibonacci-like sequence with variations
                sequence[0] = np.random.uniform(0.1, 0.3)
                sequence[1] = np.random.uniform(0.1, 0.3)
                
                for i in range(2, sequence_length):
                    sequence[i] = (sequence[i-1] + sequence[i-2]) * np.random.uniform(0.4, 0.6)
                    sequence[i] = min(sequence[i], 1.0)  # Prevent explosion
            
            elif pattern_type == 'arithmetic_progression':
                start = np.random.uniform(0.1, 0.5)
                diff = np.random.uniform(-0.05, 0.05)
                
                for i in range(sequence_length):
                    sequence[i] = start + i * diff
                    sequence[i] = np.clip(sequence[i], 0, 1)
            
            elif pattern_type == 'geometric_progression':
                start = np.random.uniform(0.2, 0.8)
                ratio = np.random.uniform(0.8, 1.2)
                
                for i in range(sequence_length):
                    sequence[i] = start * (ratio ** i)
                    sequence[i] = np.clip(sequence[i], 0, 1)
            
            elif pattern_type == 'alternating_pattern':
                values = [np.random.uniform(0.2, 0.4), np.random.uniform(0.6, 0.8)]
                for i in range(sequence_length):
                    sequence[i] = values[i % 2] + np.random.normal(0, 0.05)
                    sequence[i] = np.clip(sequence[i], 0, 1)
            
            else:  # complex_recursive
                # Pattern depends on multiple previous values
                sequence[0] = np.random.uniform(0.3, 0.7)
                sequence[1] = np.random.uniform(0.3, 0.7)
                sequence[2] = np.random.uniform(0.3, 0.7)
                
                for i in range(3, sequence_length):
                    sequence[i] = (0.4 * sequence[i-1] + 0.3 * sequence[i-2] + 0.2 * sequence[i-3] + 
                                  np.random.normal(0, 0.05))
                    sequence[i] = np.clip(sequence[i], 0, 1)
            
            sequences.append(sequence)
        
        return sequences


class AdvancedTrainingRunner:
    """
    Runner for advanced training scenarios with comprehensive evaluation
    """
    
    def __init__(self, network_config: Optional[NetworkConfig] = None):
        self.network_config = network_config or NetworkConfig(
            num_neurons=75,
            input_size=15,
            output_size=10,
            concepts=['logic', 'causality', 'composition', 'temporal', 'meta'],
            enable_self_modification=True,
            memory_depth=100
        )
        
        self.scenarios = self._define_scenarios()
        self.results = {}
    
    def _define_scenarios(self) -> List[TrainingScenario]:
        """Define all advanced training scenarios"""
        return [
            TrainingScenario(
                name="Logical Reasoning Mastery",
                description="Learn complex logical patterns and if-then relationships",
                data_generator=lambda: AdvancedDataGenerators.logical_reasoning_sequences(500, 10),
                difficulty="Hard",
                expected_capabilities=["logical_reasoning", "pattern_recognition", "rule_learning"],
                success_criteria={"prediction_accuracy": 0.75, "causal_reasoning": 0.70}
            ),
            
            TrainingScenario(
                name="Deep Causal Understanding",
                description="Master long causal chains and confounding variables",
                data_generator=lambda: AdvancedDataGenerators.causal_chain_problems(300, 6),
                difficulty="Expert",
                expected_capabilities=["causal_reasoning", "dependency_tracking", "noise_filtering"],
                success_criteria={"causal_reasoning": 0.80, "reasoning_depth": 0.70}
            ),
            
            TrainingScenario(
                name="Compositional Concept Mastery",
                description="Learn to combine and compose complex concepts",
                data_generator=lambda: [item['encoding'] for item in AdvancedDataGenerators.compositional_concept_learning(400)],
                difficulty="Hard",
                expected_capabilities=["concept_composition", "feature_interaction", "hierarchical_learning"],
                success_criteria={"concept_coherence": 0.85, "compositional_reasoning": 0.75}
            ),
            
            TrainingScenario(
                name="Meta-Learning Adaptation",
                description="Quickly adapt to new tasks with minimal examples",
                data_generator=lambda: self._prepare_meta_learning_data(),
                difficulty="Expert",
                expected_capabilities=["meta_learning", "rapid_adaptation", "transfer_learning"],
                success_criteria={"adaptation_ability": 0.70, "few_shot_learning": 0.65}
            ),
            
            TrainingScenario(
                name="Temporal Pattern Mastery",
                description="Understand complex temporal dependencies and sequences",
                data_generator=lambda: AdvancedDataGenerators.temporal_pattern_recognition(200, 15),
                difficulty="Hard",
                expected_capabilities=["temporal_reasoning", "sequence_learning", "memory_utilization"],
                success_criteria={"prediction_accuracy": 0.70, "reasoning_depth": 0.65}
            )
        ]
    
    def _prepare_meta_learning_data(self) -> List[np.ndarray]:
        """Prepare meta-learning data in appropriate format"""
        tasks = AdvancedDataGenerators.meta_learning_tasks(20, 15)
        
        # Convert to network input format
        data = []
        for task in tasks:
            for example in task['examples']:
                # Create input vector: [task_context, input_value, padding...]
                input_vector = np.zeros(self.network_config.input_size)
                input_vector[0] = task['task_id'] / 50.0  # Normalized task ID
                input_vector[1] = (example['input'] + 2) / 4.0  # Normalized input
                input_vector[2] = (example['output'] + 3) / 6.0  # Normalized output
                
                # Add some task type encoding
                task_type_encoding = {'linear_function': 0.2, 'quadratic_function': 0.4, 
                                    'sine_wave': 0.6, 'step_function': 0.8, 
                                    'exponential_decay': 1.0, 'polynomial': 0.5}
                input_vector[3] = task_type_encoding.get(task['task_type'], 0.5)
                
                # Random padding for remaining dimensions
                input_vector[4:] = np.random.uniform(0, 0.3, len(input_vector) - 4)
                
                data.append(input_vector)
        
        return data
    
    def run_scenario(self, scenario: TrainingScenario, epochs: int = 50, verbose: bool = True) -> Dict[str, Any]:
        """Run a single advanced training scenario"""
        if verbose:
            print(f"\n🎯 Starting scenario: {scenario.name}")
            print(f"   Description: {scenario.description}")
            print(f"   Difficulty: {scenario.difficulty}")
            print(f"   Expected capabilities: {', '.join(scenario.expected_capabilities)}")
        
        # Create fresh network for this scenario
        network = Network(self.network_config)
        trainer = MaskedTrainer(network)
        
        # Generate training data
        start_time = time.time()
        training_data = scenario.data_generator()
        data_gen_time = time.time() - start_time
        
        if verbose:
            print(f"   📊 Generated {len(training_data)} training examples in {data_gen_time:.2f}s")
        
        # Pre-training evaluation
        benchmarks = AGIBenchmarks(network, save_results=False)
        pre_capabilities = benchmarks.run_full_suite(include_expensive=False)
        
        # Train the network
        if verbose:
            print(f"   🚀 Training for {epochs} epochs...")
        
        training_start = time.time()
        history = trainer.train(training_data, epochs=epochs, verbose=verbose)
        training_time = time.time() - training_start
        
        # Post-training evaluation
        post_capabilities = benchmarks.run_full_suite(include_expensive=False)
        
        # Analyze results
        results = self._analyze_scenario_results(
            scenario, pre_capabilities, post_capabilities, history, training_time
        )
        
        if verbose:
            self._print_scenario_results(scenario, results)
        
        return results
    
    def run_all_scenarios(self, epochs_per_scenario: int = 30) -> Dict[str, Any]:
        """Run all advanced training scenarios"""
        print("🚀 Starting Advanced AGI Training Challenge")
        print("=" * 60)
        
        all_results = {}
        
        for i, scenario in enumerate(self.scenarios, 1):
            print(f"\n📋 Scenario {i}/{len(self.scenarios)}")
            
            try:
                results = self.run_scenario(scenario, epochs=epochs_per_scenario)
                all_results[scenario.name] = results
                
                # Brief pause between scenarios
                time.sleep(1)
                
            except Exception as e:
                print(f"❌ Error in scenario {scenario.name}: {e}")
                all_results[scenario.name] = {"error": str(e)}
        
        # Generate comprehensive report
        overall_report = self._generate_overall_report(all_results)
        
        print("\n" + "=" * 60)
        print("🎊 ADVANCED TRAINING CHALLENGE COMPLETE!")
        print("=" * 60)
        self._print_overall_report(overall_report)
        
        return {
            'individual_results': all_results,
            'overall_report': overall_report
        }
    
    def _analyze_scenario_results(self, scenario: TrainingScenario, 
                                 pre_capabilities, post_capabilities, 
                                 history, training_time) -> Dict[str, Any]:
        """Analyze results of a training scenario"""
        
        # Calculate improvements
        improvements = {}
        for metric in scenario.success_criteria:
            pre_score = pre_capabilities.individual_scores.get(metric, 0)
            post_score = post_capabilities.individual_scores.get(metric, 0)
            improvements[metric] = post_score - pre_score
        
        # Check success criteria
        success_flags = {}
        for metric, threshold in scenario.success_criteria.items():
            achieved_score = post_capabilities.individual_scores.get(metric, 0)
            success_flags[metric] = achieved_score >= threshold
        
        overall_success = all(success_flags.values())
        
        # Training stability analysis
        if hasattr(history, 'losses') and history.losses:
            loss_stability = 1.0 - (np.std(history.losses[-10:]) / np.mean(history.losses[-10:]))
            convergence_rate = abs(history.losses[0] - history.losses[-1]) / history.losses[0] if history.losses[0] > 0 else 0
        else:
            loss_stability = 0.5
            convergence_rate = 0.5
        
        return {
            'scenario_name': scenario.name,
            'success': overall_success,
            'success_flags': success_flags,
            'pre_capabilities': pre_capabilities.individual_scores,
            'post_capabilities': post_capabilities.individual_scores,
            'improvements': improvements,
            'training_time': training_time,
            'loss_stability': loss_stability,
            'convergence_rate': convergence_rate,
            'final_agi_score': post_capabilities.overall_score
        }
    
    def _print_scenario_results(self, scenario: TrainingScenario, results: Dict[str, Any]):
        """Print detailed results for a scenario"""
        print(f"\n   📊 RESULTS:")
        print(f"   {'='*50}")
        
        # Success status
        status_emoji = "✅" if results['success'] else "❌"
        print(f"   {status_emoji} Overall Success: {results['success']}")
        
        # Individual metrics
        print(f"   📈 Capability Improvements:")
        for metric, improvement in results['improvements'].items():
            threshold = scenario.success_criteria[metric]
            achieved = results['post_capabilities'][metric]
            success_emoji = "✅" if results['success_flags'][metric] else "❌"
            
            print(f"      {success_emoji} {metric}: {achieved:.3f} (target: {threshold:.3f}, improvement: {improvement:+.3f})")
        
        # Training metrics
        print(f"   ⏱️  Training Time: {results['training_time']:.2f}s")
        print(f"   📉 Loss Stability: {results['loss_stability']:.3f}")
        print(f"   🎯 Final AGI Score: {results['final_agi_score']:.3f}")
    
    def _generate_overall_report(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall performance report"""
        successful_scenarios = sum(1 for r in all_results.values() 
                                 if isinstance(r, dict) and r.get('success', False))
        
        total_scenarios = len([r for r in all_results.values() if isinstance(r, dict) and 'success' in r])
        
        success_rate = successful_scenarios / total_scenarios if total_scenarios > 0 else 0
        
        # Calculate average improvements
        all_improvements = {}
        for result in all_results.values():
            if isinstance(result, dict) and 'improvements' in result:
                for metric, improvement in result['improvements'].items():
                    if metric not in all_improvements:
                        all_improvements[metric] = []
                    all_improvements[metric].append(improvement)
        
        avg_improvements = {metric: np.mean(values) 
                           for metric, values in all_improvements.items()}
        
        # Find best and worst scenarios
        scenario_scores = {}
        for name, result in all_results.items():
            if isinstance(result, dict) and 'final_agi_score' in result:
                scenario_scores[name] = result['final_agi_score']
        
        best_scenario = max(scenario_scores.items(), key=lambda x: x[1]) if scenario_scores else ("None", 0)
        worst_scenario = min(scenario_scores.items(), key=lambda x: x[1]) if scenario_scores else ("None", 0)
        
        return {
            'success_rate': success_rate,
            'successful_scenarios': successful_scenarios,
            'total_scenarios': total_scenarios,
            'average_improvements': avg_improvements,
            'best_scenario': best_scenario,
            'worst_scenario': worst_scenario,
            'overall_assessment': self._assess_overall_performance(success_rate, avg_improvements)
        }
    
    def _print_overall_report(self, report: Dict[str, Any]):
        """Print comprehensive overall report"""
        print(f"📊 OVERALL PERFORMANCE SUMMARY:")
        print(f"   Success Rate: {report['successful_scenarios']}/{report['total_scenarios']} ({report['success_rate']*100:.1f}%)")
        print(f"   Best Scenario: {report['best_scenario'][0]} (AGI Score: {report['best_scenario'][1]:.3f})")
        print(f"   Worst Scenario: {report['worst_scenario'][0]} (AGI Score: {report['worst_scenario'][1]:.3f})")
        
        print(f"\n📈 AVERAGE CAPABILITY IMPROVEMENTS:")
        for metric, improvement in report['average_improvements'].items():
            trend_emoji = "📈" if improvement > 0 else "📉" if improvement < 0 else "➡️"
            print(f"   {trend_emoji} {metric}: {improvement:+.3f}")
        
        print(f"\n🎯 ASSESSMENT: {report['overall_assessment']}")
    
    def _assess_overall_performance(self, success_rate: float, avg_improvements: Dict[str, float]) -> str:
        """Assess overall performance and provide recommendations"""
        if success_rate >= 0.8:
            return "EXCELLENT - Network demonstrates strong AGI capabilities across scenarios"
        elif success_rate >= 0.6:
            return "GOOD - Network shows promising AGI development with room for improvement"
        elif success_rate >= 0.4:
            return "FAIR - Network has basic AGI foundations but needs significant development"
        else:
            return "NEEDS WORK - Network requires fundamental improvements in AGI capabilities"


def main():
    """Run advanced training examples"""
    print("🧠 AGI-Formula Advanced Training Examples")
    print("=" * 50)
    
    # Create advanced training runner
    runner = AdvancedTrainingRunner()
    
    # Run all scenarios
    results = runner.run_all_scenarios(epochs_per_scenario=25)
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = f"advanced_training_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                serializable_results[key] = {k: v for k, v in value.items() 
                                           if not isinstance(v, np.ndarray)}
            else:
                serializable_results[key] = value
        
        json.dump(serializable_results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {results_file}")
    print("🎉 Advanced training examples completed!")


if __name__ == "__main__":
    main()