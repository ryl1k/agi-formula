"""Dynamic topology management for AGI-Formula attention system."""

from typing import List, Dict, Tuple, Optional, Set
import numpy as np
from dataclasses import dataclass


@dataclass
class ConnectionCandidate:
    """Represents a potential connection between neurons."""
    from_neuron: int
    to_neuron: int
    strength: float
    confidence: float
    justification: str


class DynamicTopology:
    """
    Manages dynamic network connections and topology changes.
    
    This class handles:
    - Adding new connections based on attention patterns
    - Pruning weak or unused connections
    - Suggesting beneficial new connections
    - Maintaining network connectivity constraints
    """
    
    def __init__(
        self, 
        connection_threshold: float = 0.3,
        pruning_threshold: float = 0.05,
        max_connections_per_neuron: int = 20
    ):
        """
        Initialize dynamic topology manager.
        
        Args:
            connection_threshold: Minimum strength to add new connections
            pruning_threshold: Minimum strength to keep existing connections
            max_connections_per_neuron: Maximum connections per neuron
        """
        self.connection_threshold = connection_threshold
        self.pruning_threshold = pruning_threshold
        self.max_connections_per_neuron = max_connections_per_neuron
        
        # Track connection usage and performance
        self.connection_usage: Dict[Tuple[int, int], int] = {}
        self.connection_success: Dict[Tuple[int, int], List[float]] = {}
        self.suggested_connections: List[ConnectionCandidate] = []
        
        # Statistics
        self.connections_added = 0
        self.connections_removed = 0
        self.topology_changes = 0
    
    def add_connection(
        self, 
        from_neuron: int, 
        to_neuron: int, 
        weight: float,
        neurons_dict: Dict[int, 'Neuron']
    ) -> bool:
        """
        Add a new connection between neurons.
        
        Args:
            from_neuron: Source neuron ID
            to_neuron: Target neuron ID
            weight: Connection weight
            neurons_dict: Dictionary of all neurons
            
        Returns:
            True if connection was added successfully
        """
        # Validate neurons exist
        if from_neuron not in neurons_dict or to_neuron not in neurons_dict:
            return False
        
        # Prevent self-connections
        if from_neuron == to_neuron:
            return False
        
        target_neuron = neurons_dict[to_neuron]
        
        # Check if connection already exists
        if from_neuron in target_neuron.neighbors:
            return False
        
        # Check connection limits
        if len(target_neuron.neighbors) >= self.max_connections_per_neuron:
            # Remove weakest connection first
            self._remove_weakest_connection(target_neuron)
        
        # Add the connection
        target_neuron.add_neighbor(from_neuron, weight)
        
        # Track the addition
        key = (from_neuron, to_neuron)
        self.connection_usage[key] = 0
        self.connection_success[key] = []
        
        self.connections_added += 1
        self.topology_changes += 1
        
        return True
    
    def remove_connection(
        self, 
        from_neuron: int, 
        to_neuron: int,
        neurons_dict: Dict[int, 'Neuron']
    ) -> bool:
        """
        Remove a connection between neurons.
        
        Args:
            from_neuron: Source neuron ID
            to_neuron: Target neuron ID
            neurons_dict: Dictionary of all neurons
            
        Returns:
            True if connection was removed successfully
        """
        if to_neuron not in neurons_dict:
            return False
        
        target_neuron = neurons_dict[to_neuron]
        success = target_neuron.remove_neighbor(from_neuron)
        
        if success:
            # Clean up tracking data
            key = (from_neuron, to_neuron)
            self.connection_usage.pop(key, None)
            self.connection_success.pop(key, None)
            
            self.connections_removed += 1
            self.topology_changes += 1
        
        return success
    
    def prune_connections(
        self, 
        neurons_dict: Dict[int, 'Neuron'],
        min_usage: int = 5
    ) -> int:
        """
        Prune weak or unused connections.
        
        Args:
            neurons_dict: Dictionary of all neurons
            min_usage: Minimum usage count to keep connection
            
        Returns:
            Number of connections pruned
        """
        connections_to_remove = []
        
        # Find connections to prune
        for (from_id, to_id), usage_count in self.connection_usage.items():
            # Skip if not enough usage data
            if usage_count < min_usage:
                continue
            
            # Get success rate
            success_rates = self.connection_success.get((from_id, to_id), [])
            if not success_rates:
                continue
            
            avg_success = np.mean(success_rates)
            
            # Mark for removal if below threshold
            if avg_success < self.pruning_threshold:
                connections_to_remove.append((from_id, to_id))
        
        # Remove marked connections
        pruned_count = 0
        for from_id, to_id in connections_to_remove:
            if self.remove_connection(from_id, to_id, neurons_dict):
                pruned_count += 1
        
        return pruned_count
    
    def suggest_new_connections(
        self, 
        neuron_id: int,
        neurons_dict: Dict[int, 'Neuron'],
        causal_cache: 'CausalCache',
        max_suggestions: int = 5
    ) -> List[ConnectionCandidate]:
        """
        Suggest beneficial new connections for a neuron.
        
        Args:
            neuron_id: Target neuron to suggest connections for
            neurons_dict: Dictionary of all neurons
            causal_cache: Causal cache for relationship analysis
            max_suggestions: Maximum number of suggestions
            
        Returns:
            List of connection candidates
        """
        if neuron_id not in neurons_dict:
            return []
        
        target_neuron = neurons_dict[neuron_id]
        current_neighbors = set(target_neuron.neighbors)
        candidates = []
        
        # Get influential neurons from causal cache
        influential_neurons = causal_cache.get_most_influential_neurons(top_k=20)
        
        for source_id, contribution in influential_neurons:
            # Skip if already connected or same neuron
            if source_id == neuron_id or source_id in current_neighbors:
                continue
            
            # Skip if source neuron doesn't exist
            if source_id not in neurons_dict:
                continue
            
            # Calculate connection strength based on various factors
            strength = self._calculate_connection_strength(
                source_id, neuron_id, neurons_dict, causal_cache
            )
            
            if strength >= self.connection_threshold:
                # Get confidence based on causal information
                confidence = self._calculate_connection_confidence(
                    source_id, neuron_id, causal_cache
                )
                
                justification = self._generate_connection_justification(
                    source_id, neuron_id, neurons_dict, contribution, strength
                )
                
                candidate = ConnectionCandidate(
                    from_neuron=source_id,
                    to_neuron=neuron_id,
                    strength=strength,
                    confidence=confidence,
                    justification=justification
                )
                
                candidates.append(candidate)
        
        # Sort by strength and return top candidates
        candidates.sort(key=lambda c: c.strength, reverse=True)
        return candidates[:max_suggestions]
    
    def update_connection_usage(
        self, 
        from_neuron: int, 
        to_neuron: int, 
        success_score: float
    ) -> None:
        """
        Update usage statistics for a connection.
        
        Args:
            from_neuron: Source neuron ID
            to_neuron: Target neuron ID
            success_score: Success score (0.0 to 1.0) for this usage
        """
        key = (from_neuron, to_neuron)
        
        # Update usage count
        if key not in self.connection_usage:
            self.connection_usage[key] = 0
        self.connection_usage[key] += 1
        
        # Update success history
        if key not in self.connection_success:
            self.connection_success[key] = []
        
        self.connection_success[key].append(success_score)
        
        # Keep only recent history
        if len(self.connection_success[key]) > 100:
            self.connection_success[key].pop(0)
    
    def _calculate_connection_strength(
        self,
        source_id: int,
        target_id: int,
        neurons_dict: Dict[int, 'Neuron'],
        causal_cache: 'CausalCache'
    ) -> float:
        """Calculate the strength of a potential connection."""
        source_neuron = neurons_dict[source_id]
        target_neuron = neurons_dict[target_id]
        
        strength = 0.0
        
        # Factor 1: Causal contribution
        causal_entry = causal_cache.entries.get(source_id)
        if causal_entry:
            strength += abs(causal_entry.contribution) * 0.4
        
        # Factor 2: Concept compatibility
        if source_neuron.concept_type and target_neuron.concept_type:
            compatibility = self._get_concept_compatibility(
                source_neuron.concept_type, 
                target_neuron.concept_type
            )
            strength += compatibility * 0.3
        
        # Factor 3: Activation correlation
        correlation = self._get_activation_correlation(source_id, target_id, causal_cache)
        strength += correlation * 0.2
        
        # Factor 4: Network position (avoid creating too dense clusters)
        position_score = self._get_position_score(source_id, target_id, neurons_dict)
        strength += position_score * 0.1
        
        return min(1.0, strength)
    
    def _calculate_connection_confidence(
        self,
        source_id: int,
        target_id: int,
        causal_cache: 'CausalCache'
    ) -> float:
        """Calculate confidence in a potential connection."""
        # Base confidence
        confidence = 0.5
        
        # Increase confidence based on causal evidence
        causal_entry = causal_cache.entries.get(source_id)
        if causal_entry:
            confidence = max(confidence, causal_entry.confidence)
        
        # Increase confidence based on mutual information
        if causal_entry and causal_entry.mutual_information > 0:
            confidence += causal_entry.mutual_information * 0.2
        
        return min(1.0, confidence)
    
    def _get_concept_compatibility(self, concept_a: str, concept_b: str) -> float:
        """Get compatibility score between two concepts."""
        # Simple heuristic compatibility matrix
        compatibility_matrix = {
            ('color', 'shape'): 0.8,
            ('color', 'object'): 0.7,
            ('shape', 'object'): 0.9,
            ('object', 'action'): 0.6,
            ('color', 'action'): 0.3,
            ('shape', 'action'): 0.4,
            ('meta', 'composite'): 0.8,
            ('input', 'color'): 0.7,
            ('input', 'shape'): 0.7,
        }
        
        # Check both directions
        score = compatibility_matrix.get((concept_a, concept_b), 0.5)
        score = max(score, compatibility_matrix.get((concept_b, concept_a), 0.5))
        
        return score
    
    def _get_activation_correlation(
        self, 
        source_id: int, 
        target_id: int, 
        causal_cache: 'CausalCache'
    ) -> float:
        """Get activation correlation between two neurons."""
        # Use correlation matrix from causal cache
        correlation = causal_cache.correlation_matrix.get((source_id, target_id), 0.0)
        correlation = max(correlation, causal_cache.correlation_matrix.get((target_id, source_id), 0.0))
        
        return correlation
    
    def _get_position_score(
        self, 
        source_id: int, 
        target_id: int, 
        neurons_dict: Dict[int, 'Neuron']
    ) -> float:
        """Get position score to avoid overly dense connections."""
        target_neuron = neurons_dict[target_id]
        
        # Penalize if target already has many connections
        current_connections = len(target_neuron.neighbors)
        max_connections = self.max_connections_per_neuron
        
        if current_connections >= max_connections * 0.8:
            return 0.2  # Low score for dense nodes
        else:
            return 0.8  # Good score for sparse nodes
    
    def _generate_connection_justification(
        self,
        source_id: int,
        target_id: int,
        neurons_dict: Dict[int, 'Neuron'],
        contribution: float,
        strength: float
    ) -> str:
        """Generate human-readable justification for connection."""
        source_neuron = neurons_dict[source_id]
        target_neuron = neurons_dict[target_id]
        
        source_type = source_neuron.concept_type or "unknown"
        target_type = target_neuron.concept_type or "unknown"
        
        return (f"Connect {source_type} neuron {source_id} to {target_type} neuron {target_id}: "
                f"contribution={contribution:.3f}, strength={strength:.3f}")
    
    def _remove_weakest_connection(self, neuron: 'Neuron') -> bool:
        """Remove the weakest connection from a neuron."""
        if not neuron.neighbors:
            return False
        
        # Find weakest connection based on success rates
        weakest_neighbor = None
        weakest_score = float('inf')
        
        for neighbor_id in neuron.neighbors:
            key = (neighbor_id, neuron.id)
            success_rates = self.connection_success.get(key, [0.5])
            avg_success = np.mean(success_rates)
            
            if avg_success < weakest_score:
                weakest_score = avg_success
                weakest_neighbor = neighbor_id
        
        if weakest_neighbor is not None:
            return neuron.remove_neighbor(weakest_neighbor)
        
        return False
    
    def get_topology_statistics(self) -> Dict:
        """Get statistics about topology changes and connections."""
        total_connections = len(self.connection_usage)
        
        if total_connections > 0:
            usage_counts = list(self.connection_usage.values())
            success_rates = [
                np.mean(rates) for rates in self.connection_success.values() if rates
            ]
            
            avg_usage = np.mean(usage_counts)
            avg_success = np.mean(success_rates) if success_rates else 0.0
        else:
            avg_usage = 0.0
            avg_success = 0.0
        
        return {
            'total_connections_tracked': total_connections,
            'connections_added': self.connections_added,
            'connections_removed': self.connections_removed,
            'topology_changes': self.topology_changes,
            'avg_connection_usage': avg_usage,
            'avg_connection_success': avg_success,
            'suggested_connections': len(self.suggested_connections)
        }
    
    def export_topology_graph(self, neurons_dict: Dict[int, 'Neuron']) -> Dict:
        """Export current topology as a graph structure."""
        edges = []
        nodes = set()
        
        for neuron_id, neuron in neurons_dict.items():
            nodes.add(neuron_id)
            
            for neighbor_id in neuron.neighbors:
                nodes.add(neighbor_id)
                
                # Get connection statistics
                key = (neighbor_id, neuron_id)
                usage = self.connection_usage.get(key, 0)
                success_rates = self.connection_success.get(key, [])
                avg_success = np.mean(success_rates) if success_rates else 0.5
                
                edges.append({
                    'from': neighbor_id,
                    'to': neuron_id,
                    'usage_count': usage,
                    'success_rate': avg_success,
                    'type': 'dynamic_connection'
                })
        
        return {
            'nodes': list(nodes),
            'edges': edges,
            'metadata': {
                'total_nodes': len(nodes),
                'total_edges': len(edges),
                'topology_changes': self.topology_changes
            }
        }