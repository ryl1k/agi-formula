"""
AGI Reasoning Engine

Advanced reasoning capabilities including logical, causal, temporal,
and abstract reasoning for artificial general intelligence.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from abc import ABC, abstractmethod
import itertools

class Concept:
    """Represents a concept that can be reasoned about"""
    
    def __init__(self, name, properties=None, relations=None):
        self.name = name
        self.properties = properties or {}
        self.relations = relations or {}
        self.activation_level = 0.0
        self.creation_time = 0
        
    def activate(self, strength=0.1):
        """Activate this concept"""
        self.activation_level = min(1.0, self.activation_level + strength)
        
    def decay(self, rate=0.01):
        """Natural decay of activation"""
        self.activation_level = max(0.0, self.activation_level - rate)
        
    def relate_to(self, other_concept, relationship_type, strength=1.0):
        """Create relationship to another concept"""
        if relationship_type not in self.relations:
            self.relations[relationship_type] = []
        
        self.relations[relationship_type].append({
            'concept': other_concept,
            'strength': strength,
            'bidirectional': False
        })
        
    def __repr__(self):
        return f"Concept('{self.name}', activation={self.activation_level:.3f})"

class LogicalReasoner:
    """Handles logical reasoning operations"""
    
    def __init__(self):
        self.facts = set()
        self.rules = []
        self.inference_history = []
        
    def add_fact(self, fact):
        """Add a fact to the knowledge base"""
        self.facts.add(fact)
        
    def add_rule(self, premise, conclusion, confidence=1.0):
        """Add a logical rule"""
        self.rules.append({
            'premise': premise,
            'conclusion': conclusion,
            'confidence': confidence,
            'usage_count': 0
        })
        
    def infer(self, query):
        """Perform logical inference"""
        # Simple forward chaining
        inferences = []
        
        for rule in self.rules:
            if self._matches_premise(rule['premise'], self.facts):
                conclusion = rule['conclusion']
                confidence = rule['confidence']
                
                if conclusion not in self.facts:
                    inferences.append({
                        'conclusion': conclusion,
                        'confidence': confidence,
                        'rule_used': rule
                    })
                    
                    # Add to facts if confidence high enough
                    if confidence > 0.7:
                        self.facts.add(conclusion)
                        
                rule['usage_count'] += 1
                
        self.inference_history.append({
            'query': query,
            'inferences': inferences,
            'facts_at_time': len(self.facts)
        })
        
        return inferences
        
    def _matches_premise(self, premise, facts):
        """Check if premise matches current facts"""
        if isinstance(premise, str):
            return premise in facts
        elif isinstance(premise, (list, tuple)):
            return all(p in facts for p in premise)
        return False

class CausalReasoner:
    """Handles causal reasoning and discovery"""
    
    def __init__(self):
        self.causal_graph = {}
        self.interventions = []
        self.causal_strength_threshold = 0.3
        
    def add_causal_link(self, cause, effect, strength=1.0, evidence=None):
        """Add a causal relationship"""
        if cause not in self.causal_graph:
            self.causal_graph[cause] = {}
            
        self.causal_graph[cause][effect] = {
            'strength': strength,
            'evidence': evidence or [],
            'discovered_at': len(self.interventions)
        }
        
    def discover_causes(self, effect, observations):
        """Discover potential causes for an effect"""
        potential_causes = []
        
        # Simple correlation-based causal discovery
        for obs in observations:
            if 'variables' in obs and effect in obs['variables']:
                effect_value = obs['variables'][effect]
                
                for var_name, var_value in obs['variables'].items():
                    if var_name != effect:
                        # Calculate correlation (simplified)
                        correlation = self._calculate_correlation(
                            var_value, effect_value
                        )
                        
                        if correlation > self.causal_strength_threshold:
                            potential_causes.append({
                                'cause': var_name,
                                'strength': correlation,
                                'observation': obs
                            })
                            
        return potential_causes
    
    def _calculate_correlation(self, x, y):
        """Simple correlation calculation"""
        # Placeholder for more sophisticated causal discovery
        try:
            if np.isscalar(x) and np.isscalar(y):
                # For scalar values, use simple difference-based correlation
                return 1.0 - abs(x - y) / (abs(x) + abs(y) + 1.0)
            else:
                corr_matrix = np.corrcoef([x, y])
                return abs(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0
        except:
            return 0.5  # Default correlation
    
    def intervene(self, variable, value):
        """Perform intervention for causal testing"""
        intervention = {
            'variable': variable,
            'value': value,
            'timestamp': len(self.interventions)
        }
        self.interventions.append(intervention)
        return intervention
    
    def predict_effect(self, cause, intervention_value):
        """Predict effect of intervention"""
        predictions = {}
        
        if cause in self.causal_graph:
            for effect, relationship in self.causal_graph[cause].items():
                predicted_change = intervention_value * relationship['strength']
                predictions[effect] = predicted_change
                
        return predictions

class TemporalReasoner:
    """Handles temporal and sequential reasoning"""
    
    def __init__(self):
        self.temporal_sequences = []
        self.pattern_memory = {}
        self.time_horizon = 10
        
    def add_temporal_sequence(self, events, timestamps=None):
        """Add a sequence of temporal events"""
        if timestamps is None:
            timestamps = list(range(len(events)))
            
        sequence = {
            'events': events,
            'timestamps': timestamps,
            'patterns': self._extract_patterns(events)
        }
        
        self.temporal_sequences.append(sequence)
        self._update_pattern_memory(sequence['patterns'])
        
    def _extract_patterns(self, events):
        """Extract temporal patterns from events"""
        patterns = []
        
        # Extract n-grams
        for n in range(2, min(4, len(events) + 1)):
            for i in range(len(events) - n + 1):
                pattern = tuple(events[i:i+n])
                patterns.append({
                    'pattern': pattern,
                    'length': n,
                    'position': i
                })
                
        return patterns
    
    def _update_pattern_memory(self, patterns):
        """Update memory of temporal patterns"""
        for pattern_info in patterns:
            pattern = pattern_info['pattern']
            
            if pattern not in self.pattern_memory:
                self.pattern_memory[pattern] = {
                    'count': 0,
                    'confidence': 0.0,
                    'contexts': []
                }
                
            self.pattern_memory[pattern]['count'] += 1
            self.pattern_memory[pattern]['confidence'] = min(1.0,
                self.pattern_memory[pattern]['count'] / 10.0)
                
    def predict_next(self, current_sequence):
        """Predict next event in sequence"""
        predictions = {}
        
        # Find matching patterns
        for length in range(min(len(current_sequence), 3), 0, -1):
            recent_pattern = tuple(current_sequence[-length:])
            
            if recent_pattern in self.pattern_memory:
                pattern_info = self.pattern_memory[recent_pattern]
                
                # Look for patterns that extend this one
                for stored_pattern in self.pattern_memory:
                    if (len(stored_pattern) == length + 1 and 
                        stored_pattern[:-1] == recent_pattern):
                        
                        next_event = stored_pattern[-1]
                        confidence = self.pattern_memory[stored_pattern]['confidence']
                        
                        if next_event not in predictions:
                            predictions[next_event] = 0
                        predictions[next_event] += confidence
                        
        return predictions

class AbstractReasoner:
    """Handles abstract reasoning and pattern recognition"""
    
    def __init__(self):
        self.abstractions = {}
        self.analogy_mappings = []
        self.abstraction_hierarchy = {}
        
    def create_abstraction(self, instances, abstraction_name):
        """Create abstract concept from instances"""
        if not instances:
            return None
            
        # Extract common properties
        common_properties = self._find_common_properties(instances)
        
        abstraction = {
            'name': abstraction_name,
            'instances': instances,
            'properties': common_properties,
            'generalization_level': len(instances),
            'created_from': [str(inst) for inst in instances]
        }
        
        self.abstractions[abstraction_name] = abstraction
        return abstraction
    
    def _find_common_properties(self, instances):
        """Find properties common across instances"""
        if not instances:
            return {}
            
        common_props = {}
        
        # Simple property extraction (placeholder)
        for instance in instances:
            if hasattr(instance, 'properties'):
                for prop_name, prop_value in instance.properties.items():
                    if prop_name not in common_props:
                        common_props[prop_name] = []
                    common_props[prop_name].append(prop_value)
                    
        # Keep properties that appear in most instances
        filtered_props = {}
        for prop_name, values in common_props.items():
            if len(set(values)) == 1:  # Same value across all instances
                filtered_props[prop_name] = values[0]
            elif len(set(values)) / len(values) < 0.5:  # Most instances share value
                # Take most common value
                from collections import Counter
                most_common = Counter(values).most_common(1)[0][0]
                filtered_props[prop_name] = most_common
                
        return filtered_props
    
    def find_analogies(self, source_domain, target_domain):
        """Find analogical mappings between domains"""
        analogies = []
        
        # Simple structural alignment
        if isinstance(source_domain, dict) and isinstance(target_domain, dict):
            for source_key, source_value in source_domain.items():
                for target_key, target_value in target_domain.items():
                    similarity = self._calculate_similarity(source_value, target_value)
                    
                    if similarity > 0.5:
                        analogies.append({
                            'source': (source_key, source_value),
                            'target': (target_key, target_value),
                            'similarity': similarity,
                            'mapping_type': 'structural'
                        })
                        
        self.analogy_mappings.extend(analogies)
        return analogies
    
    def _calculate_similarity(self, obj1, obj2):
        """Calculate similarity between two objects"""
        if type(obj1) != type(obj2):
            return 0.0
            
        if isinstance(obj1, (int, float)) and isinstance(obj2, (int, float)):
            return 1.0 - abs(obj1 - obj2) / max(abs(obj1), abs(obj2), 1.0)
        elif isinstance(obj1, str) and isinstance(obj2, str):
            # Simple string similarity
            common_chars = set(obj1) & set(obj2)
            total_chars = set(obj1) | set(obj2)
            return len(common_chars) / len(total_chars) if total_chars else 1.0
        else:
            return 1.0 if obj1 == obj2 else 0.0

class ReasoningEngine:
    """Integrated reasoning engine combining all reasoning types"""
    
    def __init__(self):
        self.logical_reasoner = LogicalReasoner()
        self.causal_reasoner = CausalReasoner()
        self.temporal_reasoner = TemporalReasoner()
        self.abstract_reasoner = AbstractReasoner()
        
        self.reasoning_history = []
        self.active_concepts = {}
        
    def reason(self, query, context=None, reasoning_types=None):
        """Perform integrated reasoning"""
        if reasoning_types is None:
            reasoning_types = ['logical', 'causal', 'temporal', 'abstract']
            
        results = {}
        
        if 'logical' in reasoning_types:
            results['logical'] = self.logical_reasoner.infer(query)
            
        if 'causal' in reasoning_types and context:
            if 'observations' in context:
                results['causal'] = self.causal_reasoner.discover_causes(
                    query, context['observations']
                )
                
        if 'temporal' in reasoning_types and context:
            if 'sequence' in context:
                results['temporal'] = self.temporal_reasoner.predict_next(
                    context['sequence']
                )
                
        if 'abstract' in reasoning_types and context:
            if 'source_domain' in context and 'target_domain' in context:
                results['abstract'] = self.abstract_reasoner.find_analogies(
                    context['source_domain'], context['target_domain']
                )
                
        reasoning_result = {
            'query': query,
            'context': context,
            'results': results,
            'timestamp': len(self.reasoning_history),
            'confidence': self._calculate_overall_confidence(results)
        }
        
        self.reasoning_history.append(reasoning_result)
        return reasoning_result
    
    def _calculate_overall_confidence(self, results):
        """Calculate overall confidence in reasoning results"""
        confidences = []
        
        for reasoning_type, result in results.items():
            if reasoning_type == 'logical' and result:
                avg_conf = np.mean([r['confidence'] for r in result])
                confidences.append(avg_conf)
            elif reasoning_type == 'causal' and result:
                avg_conf = np.mean([r['strength'] for r in result])
                confidences.append(avg_conf)
            elif reasoning_type == 'temporal' and result:
                if result:
                    max_conf = max(result.values())
                    confidences.append(max_conf)
            elif reasoning_type == 'abstract' and result:
                avg_conf = np.mean([r['similarity'] for r in result])
                confidences.append(avg_conf)
                
        return np.mean(confidences) if confidences else 0.0

__all__ = [
    'Concept', 'LogicalReasoner', 'CausalReasoner', 
    'TemporalReasoner', 'AbstractReasoner', 'ReasoningEngine'
]