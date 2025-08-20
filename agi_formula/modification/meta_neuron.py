"""Meta-learning neuron for self-modification in AGI-Formula."""

from typing import Dict, List, Optional, Tuple, Set, Any
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict, deque
import random
import math
import time


@dataclass
class ModificationProposal:
    """Represents a proposed modification to the network."""
    modification_id: str
    modification_type: str  # "add_neuron", "remove_neuron", "modify_weights", "change_topology"
    target_neurons: List[int]
    parameters: Dict[str, Any]
    expected_improvement: float
    confidence: float
    safety_score: float
    resource_cost: float
    reversible: bool = True


@dataclass
class ModificationHistory:
    """Track history of modifications and their outcomes."""
    proposal: ModificationProposal
    applied_timestamp: float
    performance_before: Dict[str, float]
    performance_after: Dict[str, float]
    actual_improvement: float
    success: bool
    reverted: bool = False
    revert_timestamp: Optional[float] = None


class MetaNeuron:
    """
    Meta-learning neuron that analyzes network performance and suggests improvements.
    
    The MetaNeuron observes the network's behavior and learning patterns to identify
    opportunities for self-modification, such as:
    - Adding new neurons in underperforming areas
    - Removing redundant neurons
    - Adjusting connection strengths
    - Modifying network topology
    - Optimizing attention patterns
    """
    
    def __init__(self, network: 'Network', meta_learning_rate: float = 0.01):
        """Initialize meta-learning neuron."""
        self.network = network
        self.meta_learning_rate = meta_learning_rate
        
        # Performance monitoring
        self.performance_history: deque = deque(maxlen=100)
        self.performance_metrics: Dict[str, List[float]] = defaultdict(list)
        
        # Modification tracking
        self.modification_history: List[ModificationHistory] = []
        self.pending_proposals: List[ModificationProposal] = []
        self.modification_count = 0
        
        # Analysis state
        self.neuron_performance: Dict[int, Dict[str, float]] = defaultdict(dict)
        self.connection_strengths: Dict[Tuple[int, int], float] = {}
        self.learning_patterns: Dict[str, List[float]] = defaultdict(list)
        
        # Meta-learning parameters
        self.exploration_rate = 0.1
        self.modification_threshold = 0.05  # Minimum improvement to justify modification
        self.safety_threshold = 0.7  # Minimum safety score required
        
        # Resource management
        self.max_neurons = self.network.config.num_neurons * 2  # Don't exceed 2x original size
        self.modification_budget = 10  # Max modifications per session
        
        # Success tracking
        self.successful_modifications = 0
        self.failed_modifications = 0
        self.reverted_modifications = 0
        
        # Performance baseline
        self.baseline_performance = None
        
    def analyze_network_performance(self, current_metrics: Dict[str, float]) -> None:
        """
        Analyze current network performance to identify improvement opportunities.
        
        Args:
            current_metrics: Current performance metrics from training/validation
        """
        self.performance_history.append(current_metrics)
        
        # Update performance metrics history
        for metric_name, value in current_metrics.items():
            self.performance_metrics[metric_name].append(value)
            
            # Keep only recent history
            if len(self.performance_metrics[metric_name]) > 50:
                self.performance_metrics[metric_name].pop(0)
        
        # Set baseline if not established
        if self.baseline_performance is None:
            self.baseline_performance = current_metrics.copy()
        
        # Analyze individual neuron performance
        self._analyze_neuron_performance()
        
        # Analyze connection patterns
        self._analyze_connection_patterns()
        
        # Identify learning patterns
        self._analyze_learning_patterns()
        
        # Detect performance issues
        issues = self._detect_performance_issues()
        
        # Generate improvement proposals
        if issues:
            proposals = self._generate_improvement_proposals(issues)
            self.pending_proposals.extend(proposals)
    
    def _analyze_neuron_performance(self) -> None:
        """Analyze performance of individual neurons."""
        for neuron_id, neuron in self.network.neurons.items():
            # Activation statistics
            activation = neuron.state.activation
            self.neuron_performance[neuron_id]['activation'] = activation
            self.neuron_performance[neuron_id]['activation_variance'] = self._get_activation_variance(neuron_id)
            
            # Causal contribution
            if neuron_id in self.network.causal_cache.entries:
                entry = self.network.causal_cache.entries[neuron_id]
                self.neuron_performance[neuron_id]['causal_contribution'] = abs(entry.contribution)
                self.neuron_performance[neuron_id]['causal_stability'] = getattr(entry, 'causal_stability', 0.5)
            
            # Attention involvement
            attention_stats = self._get_neuron_attention_stats(neuron_id)
            self.neuron_performance[neuron_id].update(attention_stats)
            
            # Concept relevance
            if hasattr(neuron, 'concept_type') and neuron.concept_type:
                self.neuron_performance[neuron_id]['has_concept'] = 1.0
            else:
                self.neuron_performance[neuron_id]['has_concept'] = 0.0
    
    def _analyze_connection_patterns(self) -> None:
        """Analyze connection patterns and strengths."""
        # Analyze current network topology
        for neuron_id, neuron in self.network.neurons.items():
            # Get attention patterns for this neuron
            if hasattr(self.network, 'attention_module'):
                attention_patterns = self.network.attention_module.get_attention_patterns_for_neuron(neuron_id)
                
                for target_id, strength in attention_patterns.items():
                    connection_key = (neuron_id, target_id)
                    self.connection_strengths[connection_key] = strength
    
    def _analyze_learning_patterns(self) -> None:
        """Analyze learning patterns and trends."""
        if len(self.performance_history) < 5:
            return
        
        recent_performance = list(self.performance_history)[-10:]
        
        for metric_name in recent_performance[0].keys():
            values = [p[metric_name] for p in recent_performance]
            
            # Calculate trend
            trend = np.corrcoef(range(len(values)), values)[0, 1] if len(values) > 1 else 0.0
            self.learning_patterns[metric_name].append(trend)
            
            # Keep recent patterns
            if len(self.learning_patterns[metric_name]) > 20:
                self.learning_patterns[metric_name].pop(0)
    
    def _detect_performance_issues(self) -> List[Dict[str, Any]]:
        """Detect performance issues that might benefit from modification."""
        issues = []
        
        if len(self.performance_history) < 3:
            return issues
        
        recent_performance = list(self.performance_history)[-5:]
        
        # Issue 1: Stagnant learning
        for metric_name, values in self.performance_metrics.items():
            if len(values) >= 10:
                recent_trend = np.corrcoef(range(len(values[-10:])), values[-10:])[0, 1]
                if abs(recent_trend) < 0.01:  # Very flat trend
                    issues.append({
                        'type': 'stagnant_learning',
                        'metric': metric_name,
                        'severity': 0.8,
                        'description': f'Learning has stagnated for {metric_name}'
                    })
        
        # Issue 2: Underperforming neurons
        underperforming_neurons = []
        for neuron_id, performance in self.neuron_performance.items():
            causal_contribution = performance.get('causal_contribution', 0.0)
            activation_variance = performance.get('activation_variance', 0.0)
            
            if causal_contribution < 0.1 and activation_variance < 0.05:
                underperforming_neurons.append(neuron_id)
        
        if underperforming_neurons:
            issues.append({
                'type': 'underperforming_neurons',
                'neurons': underperforming_neurons,
                'severity': 0.6,
                'description': f'{len(underperforming_neurons)} neurons are underperforming'
            })
        
        # Issue 3: Bottlenecks in attention
        high_attention_neurons = []
        for neuron_id, performance in self.neuron_performance.items():
            attention_frequency = performance.get('attention_frequency', 0.0)
            if attention_frequency > 0.8:  # Very frequently attended to
                high_attention_neurons.append(neuron_id)
        
        if high_attention_neurons:
            issues.append({
                'type': 'attention_bottleneck',
                'neurons': high_attention_neurons,
                'severity': 0.7,
                'description': f'Attention bottleneck detected in {len(high_attention_neurons)} neurons'
            })
        
        # Issue 4: Poor concept coverage
        neurons_with_concepts = sum(1 for p in self.neuron_performance.values() 
                                  if p.get('has_concept', 0.0) > 0.5)
        total_neurons = len(self.network.neurons)
        concept_coverage = neurons_with_concepts / total_neurons
        
        if concept_coverage < 0.3:
            issues.append({
                'type': 'poor_concept_coverage',
                'coverage': concept_coverage,
                'severity': 0.5,
                'description': f'Only {concept_coverage:.1%} of neurons have concepts'
            })
        
        return issues
    
    def _generate_improvement_proposals(self, issues: List[Dict[str, Any]]) -> List[ModificationProposal]:
        """Generate concrete improvement proposals based on detected issues."""
        proposals = []
        
        for issue in issues:
            if issue['type'] == 'stagnant_learning':
                # Propose adding exploration neurons
                proposal = ModificationProposal(
                    modification_id=f"explore_{self.modification_count}",
                    modification_type="add_exploration_neurons",
                    target_neurons=[],
                    parameters={'num_neurons': 3, 'exploration_rate': 0.3},
                    expected_improvement=0.1,
                    confidence=0.6,
                    safety_score=0.9,
                    resource_cost=0.2
                )
                proposals.append(proposal)
            
            elif issue['type'] == 'underperforming_neurons':
                # Propose removing or modifying underperforming neurons
                underperforming = issue['neurons'][:3]  # Limit to top 3
                proposal = ModificationProposal(
                    modification_id=f"optimize_{self.modification_count}",
                    modification_type="optimize_neurons",
                    target_neurons=underperforming,
                    parameters={'optimization_type': 'reinitialize'},
                    expected_improvement=0.05,
                    confidence=0.7,
                    safety_score=0.8,
                    resource_cost=0.1
                )
                proposals.append(proposal)
            
            elif issue['type'] == 'attention_bottleneck':
                # Propose adding assistant neurons to distribute load
                bottleneck_neurons = issue['neurons']
                proposal = ModificationProposal(
                    modification_id=f"distribute_{self.modification_count}",
                    modification_type="add_assistant_neurons",
                    target_neurons=bottleneck_neurons,
                    parameters={'assistants_per_neuron': 2},
                    expected_improvement=0.08,
                    confidence=0.8,
                    safety_score=0.85,
                    resource_cost=0.3
                )
                proposals.append(proposal)
            
            elif issue['type'] == 'poor_concept_coverage':
                # Propose adding concept-specific neurons
                proposal = ModificationProposal(
                    modification_id=f"concepts_{self.modification_count}",
                    modification_type="add_concept_neurons",
                    target_neurons=[],
                    parameters={'concept_types': ['semantic', 'relational', 'temporal']},
                    expected_improvement=0.12,
                    confidence=0.75,
                    safety_score=0.9,
                    resource_cost=0.25
                )
                proposals.append(proposal)
            
            self.modification_count += 1
        
        return proposals
    
    def select_best_proposals(self, max_proposals: int = 3) -> List[ModificationProposal]:
        """Select the best modification proposals based on expected benefit and safety."""
        if not self.pending_proposals:
            return []
        
        # Score each proposal
        scored_proposals = []
        for proposal in self.pending_proposals:
            # Benefit score
            benefit_score = proposal.expected_improvement * proposal.confidence
            
            # Safety score
            safety_penalty = max(0.0, self.safety_threshold - proposal.safety_score)
            
            # Resource penalty
            resource_penalty = proposal.resource_cost * 0.5
            
            # Experience bonus (favor modification types that worked before)
            experience_bonus = self._get_experience_bonus(proposal.modification_type)
            
            total_score = benefit_score - safety_penalty - resource_penalty + experience_bonus
            
            scored_proposals.append((total_score, proposal))
        
        # Sort by score and select top proposals
        scored_proposals.sort(key=lambda x: x[0], reverse=True)
        
        selected = []
        total_cost = 0.0
        
        for score, proposal in scored_proposals:
            if len(selected) >= max_proposals:
                break
            
            # Check resource constraints
            if total_cost + proposal.resource_cost <= 1.0:  # Budget constraint
                if self._passes_safety_checks(proposal):
                    selected.append(proposal)
                    total_cost += proposal.resource_cost
        
        return selected
    
    def apply_modification(self, proposal: ModificationProposal) -> bool:
        """
        Apply a modification proposal to the network.
        
        Args:
            proposal: The modification to apply
            
        Returns:
            True if modification was successfully applied
        """
        # Record performance before modification
        current_performance = self._get_current_performance()
        
        success = False
        
        try:
            if proposal.modification_type == "add_exploration_neurons":
                success = self._add_exploration_neurons(proposal)
            elif proposal.modification_type == "optimize_neurons":
                success = self._optimize_neurons(proposal)
            elif proposal.modification_type == "add_assistant_neurons":
                success = self._add_assistant_neurons(proposal)
            elif proposal.modification_type == "add_concept_neurons":
                success = self._add_concept_neurons(proposal)
            else:
                print(f"Unknown modification type: {proposal.modification_type}")
                return False
            
            if success:
                # Record the modification
                modification_record = ModificationHistory(
                    proposal=proposal,
                    applied_timestamp=time.time(),
                    performance_before=current_performance,
                    performance_after={},  # Will be filled later
                    actual_improvement=0.0,  # Will be calculated later
                    success=True
                )
                self.modification_history.append(modification_record)
                self.successful_modifications += 1
                
                # Remove from pending
                if proposal in self.pending_proposals:
                    self.pending_proposals.remove(proposal)
                
                print(f"Successfully applied modification: {proposal.modification_id}")
                return True
            else:
                self.failed_modifications += 1
                return False
                
        except Exception as e:
            print(f"Error applying modification {proposal.modification_id}: {e}")
            self.failed_modifications += 1
            return False
    
    def _add_exploration_neurons(self, proposal: ModificationProposal) -> bool:
        """Add neurons focused on exploration."""
        from ..core.neuron import Neuron, NeuronConfig
        
        num_neurons = proposal.parameters.get('num_neurons', 3)
        exploration_rate = proposal.parameters.get('exploration_rate', 0.3)
        
        # Find good positions to add neurons
        current_neuron_count = len(self.network.neurons)
        if current_neuron_count >= self.max_neurons:
            return False
        
        for i in range(num_neurons):
            if len(self.network.neurons) >= self.max_neurons:
                break
            
            # Create exploration neuron
            neuron_id = max(self.network.neurons.keys()) + 1
            config = NeuronConfig(
                neuron_id=neuron_id,
                input_size=self.network.config.input_size,
                concept_type="exploration",
                activation_threshold=0.1,
                learning_rate=self.meta_learning_rate * 2
            )
            
            neuron = Neuron(config)
            
            # Set up connections with high randomness
            neuron.weights = np.random.normal(0, 0.3, config.input_size)
            
            self.network.neurons[neuron_id] = neuron
            
            # Add to appropriate neuron lists
            if hasattr(self.network, 'hidden_neurons'):
                self.network.hidden_neurons.append(neuron_id)
        
        return True
    
    def _optimize_neurons(self, proposal: ModificationProposal) -> bool:
        """Optimize underperforming neurons."""
        optimization_type = proposal.parameters.get('optimization_type', 'reinitialize')
        
        for neuron_id in proposal.target_neurons:
            if neuron_id not in self.network.neurons:
                continue
            
            neuron = self.network.neurons[neuron_id]
            
            if optimization_type == 'reinitialize':
                # Reinitialize weights with better initialization
                input_size = len(neuron.weights) if hasattr(neuron, 'weights') else self.network.config.input_size
                neuron.weights = np.random.normal(0, 0.1, input_size)
                
                # Reset activation threshold
                neuron.config.activation_threshold = random.uniform(0.1, 0.3)
                
            elif optimization_type == 'adjust_learning':
                # Increase learning rate for faster adaptation
                neuron.config.learning_rate *= 1.5
        
        return True
    
    def _add_assistant_neurons(self, proposal: ModificationProposal) -> bool:
        """Add assistant neurons to help with bottlenecks."""
        from ..core.neuron import Neuron, NeuronConfig
        
        assistants_per_neuron = proposal.parameters.get('assistants_per_neuron', 2)
        
        for bottleneck_neuron_id in proposal.target_neurons:
            if bottleneck_neuron_id not in self.network.neurons:
                continue
            
            bottleneck_neuron = self.network.neurons[bottleneck_neuron_id]
            
            for i in range(assistants_per_neuron):
                if len(self.network.neurons) >= self.max_neurons:
                    break
                
                # Create assistant neuron
                neuron_id = max(self.network.neurons.keys()) + 1
                config = NeuronConfig(
                    neuron_id=neuron_id,
                    input_size=self.network.config.input_size,
                    concept_type=f"assistant_{bottleneck_neuron_id}",
                    activation_threshold=bottleneck_neuron.config.activation_threshold * 0.8,
                    learning_rate=bottleneck_neuron.config.learning_rate
                )
                
                assistant = Neuron(config)
                
                # Initialize with similar weights to bottleneck neuron
                if hasattr(bottleneck_neuron, 'weights'):
                    assistant.weights = bottleneck_neuron.weights.copy() + np.random.normal(0, 0.05, len(bottleneck_neuron.weights))
                
                self.network.neurons[neuron_id] = assistant
                
                # Add to hidden neurons
                if hasattr(self.network, 'hidden_neurons'):
                    self.network.hidden_neurons.append(neuron_id)
        
        return True
    
    def _add_concept_neurons(self, proposal: ModificationProposal) -> bool:
        """Add neurons with specific concept types."""
        from ..core.neuron import Neuron, NeuronConfig
        
        concept_types = proposal.parameters.get('concept_types', ['semantic'])
        
        for concept_type in concept_types:
            if len(self.network.neurons) >= self.max_neurons:
                break
            
            # Create concept neuron
            neuron_id = max(self.network.neurons.keys()) + 1
            config = NeuronConfig(
                neuron_id=neuron_id,
                input_size=self.network.config.input_size,
                concept_type=concept_type,
                activation_threshold=0.2,
                learning_rate=self.meta_learning_rate
            )
            
            neuron = Neuron(config)
            
            # Initialize weights appropriate for concept type
            if concept_type == 'semantic':
                neuron.weights = np.random.normal(0, 0.15, config.input_size)
            elif concept_type == 'relational':
                neuron.weights = np.random.uniform(-0.1, 0.1, config.input_size)
            elif concept_type == 'temporal':
                neuron.weights = np.random.normal(0, 0.2, config.input_size)
            
            self.network.neurons[neuron_id] = neuron
            
            # Add to appropriate neuron lists
            if hasattr(self.network, 'hidden_neurons'):
                self.network.hidden_neurons.append(neuron_id)
        
        return True
    
    # Helper methods
    def _get_activation_variance(self, neuron_id: int) -> float:
        """Get activation variance for a neuron."""
        # This would typically look at historical activations
        # For now, return a placeholder
        return random.uniform(0.0, 0.3)
    
    def _get_neuron_attention_stats(self, neuron_id: int) -> Dict[str, float]:
        """Get attention statistics for a neuron."""
        # Placeholder implementation
        return {
            'attention_frequency': random.uniform(0.0, 1.0),
            'attention_strength': random.uniform(0.0, 1.0)
        }
    
    def _get_experience_bonus(self, modification_type: str) -> float:
        """Get experience bonus for a modification type."""
        successful_count = 0
        total_count = 0
        
        for record in self.modification_history:
            if record.proposal.modification_type == modification_type:
                total_count += 1
                if record.success and not record.reverted:
                    successful_count += 1
        
        if total_count == 0:
            return 0.0
        
        success_rate = successful_count / total_count
        return success_rate * 0.1  # Small bonus for proven modification types
    
    def _passes_safety_checks(self, proposal: ModificationProposal) -> bool:
        """Check if a proposal passes safety requirements."""
        # Safety score threshold
        if proposal.safety_score < self.safety_threshold:
            return False
        
        # Resource constraints
        current_neurons = len(self.network.neurons)
        if proposal.modification_type.startswith('add_') and current_neurons >= self.max_neurons:
            return False
        
        # Don't modify critical neurons (inputs/outputs)
        for neuron_id in proposal.target_neurons:
            if neuron_id in self.network.input_neurons or neuron_id in self.network.output_neurons:
                return False
        
        return True
    
    def _get_current_performance(self) -> Dict[str, float]:
        """Get current network performance metrics."""
        if self.performance_history:
            return self.performance_history[-1].copy()
        else:
            return {}
    
    def evaluate_modification_success(self, new_performance: Dict[str, float]) -> None:
        """Evaluate if recent modifications were successful."""
        if not self.modification_history:
            return
        
        # Find recent modifications that haven't been evaluated
        for record in self.modification_history:
            if not record.performance_after:  # Not yet evaluated
                record.performance_after = new_performance.copy()
                
                # Calculate actual improvement
                baseline_metric = 'prediction_accuracy'  # Main metric
                before_value = record.performance_before.get(baseline_metric, 0.0)
                after_value = new_performance.get(baseline_metric, 0.0)
                record.actual_improvement = after_value - before_value
                
                # Check if modification was actually beneficial
                if record.actual_improvement < self.modification_threshold:
                    print(f"Modification {record.proposal.modification_id} did not meet improvement threshold")
    
    def get_meta_statistics(self) -> Dict[str, Any]:
        """Get comprehensive meta-learning statistics."""
        return {
            'total_modifications': len(self.modification_history),
            'successful_modifications': self.successful_modifications,
            'failed_modifications': self.failed_modifications,
            'reverted_modifications': self.reverted_modifications,
            'success_rate': self.successful_modifications / max(1, len(self.modification_history)),
            
            'pending_proposals': len(self.pending_proposals),
            'network_size': len(self.network.neurons),
            'max_neurons': self.max_neurons,
            
            'avg_improvement': np.mean([r.actual_improvement for r in self.modification_history if r.actual_improvement != 0.0]) if self.modification_history else 0.0,
            
            'modification_types': {
                mod_type: len([r for r in self.modification_history if r.proposal.modification_type == mod_type])
                for mod_type in set(r.proposal.modification_type for r in self.modification_history)
            } if self.modification_history else {},
            
            'performance_trend': len(self.performance_history),
            'baseline_performance': self.baseline_performance
        }