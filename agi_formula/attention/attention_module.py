"""Attention mechanism implementation for AGI-Formula."""

from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class AttentionConfig:
    """Configuration for attention mechanism."""
    top_k: int = 5
    temperature: float = 1.0
    min_score_threshold: float = 0.1
    score_decay: float = 0.9
    

class AttentionModule:
    """
    Computes attention scores and selects neurons for recursive activation.
    
    This implements the attention mechanism A(n,N) from the AGI formula:
    - Scores neurons based on causal contribution, confidence, and mutual information
    - Selects top-k neurons for recursive computation
    - Adapts based on success feedback
    """
    
    def __init__(self, config: AttentionConfig = None):
        """
        Initialize attention module.
        
        Args:
            config: Configuration for attention mechanism
        """
        self.config = config or AttentionConfig()
        
        # Attention weight history for adaptation
        self.attention_weights: Dict[Tuple[int, int], float] = {}
        self.success_feedback: Dict[Tuple[int, int], List[float]] = {}
        
        # Performance tracking
        self.selection_history: List[Dict] = []
        
    def compute_scores(
        self, 
        neuron: 'Neuron', 
        candidate_neighbors: List[int],
        causal_cache: 'CausalCache'
    ) -> Dict[int, float]:
        """
        Compute attention scores for candidate neighbors.
        
        Score formula:
        score = (contribution / (uncertainty + ε)) * confidence + MI_boost + historical_success
        
        Args:
            neuron: The neuron requesting attention
            candidate_neighbors: List of potential neighbors to attend to
            causal_cache: Cache containing causal information
            
        Returns:
            Dictionary mapping neighbor_id to attention score
        """
        scores = {}
        epsilon = 1e-6
        
        for neighbor_id in candidate_neighbors:
            # Get causal information
            contribution = causal_cache.get_contribution(neighbor_id)
            
            if contribution is None:
                # No causal information available - use small default
                scores[neighbor_id] = self.config.min_score_threshold
                continue
            
            # Get causal entry for detailed information
            entry = causal_cache.entries.get(neighbor_id)
            if not entry:
                scores[neighbor_id] = self.config.min_score_threshold
                continue
            
            # Base score: contribution normalized by uncertainty, weighted by confidence
            base_score = (abs(contribution) / (1.0 + epsilon)) * entry.confidence
            
            # Mutual information boost
            mi_boost = entry.mutual_information * 0.3
            
            # Historical success rate between this neuron and neighbor
            historical_success = self._get_historical_success(neuron.id, neighbor_id)
            
            # Adaptive attention weight
            adaptive_weight = self._get_adaptive_weight(neuron.id, neighbor_id)
            
            # Combined score
            total_score = (base_score + mi_boost + historical_success) * adaptive_weight
            
            # Apply temperature scaling
            total_score = total_score / self.config.temperature
            
            scores[neighbor_id] = max(total_score, self.config.min_score_threshold)
        
        # Normalize scores to sum to 1
        total = sum(scores.values())
        if total > 0:
            scores = {nid: score / total for nid, score in scores.items()}
        
        return scores
    
    def select_top_k(
        self, 
        scores: Dict[int, float], 
        k: Optional[int] = None
    ) -> List[int]:
        """
        Select top-k neurons based on attention scores.
        
        Args:
            scores: Dictionary of neuron_id -> attention_score
            k: Number of neurons to select (uses config.top_k if None)
            
        Returns:
            List of selected neuron IDs, sorted by score (highest first)
        """
        if not scores:
            return []
        
        k = k or self.config.top_k
        
        # Filter by minimum threshold
        filtered_scores = {
            nid: score for nid, score in scores.items() 
            if score >= self.config.min_score_threshold
        }
        
        if not filtered_scores:
            return []
        
        # Sort by score and take top-k
        sorted_neurons = sorted(
            filtered_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        selected = [nid for nid, score in sorted_neurons[:k]]
        
        # Store selection for analysis
        self.selection_history.append({
            'selected': selected.copy(),
            'scores': filtered_scores.copy(),
            'total_candidates': len(scores),
            'k_used': k
        })
        
        # Keep only recent history
        if len(self.selection_history) > 1000:
            self.selection_history.pop(0)
        
        return selected
    
    def update_attention_weights(
        self, 
        neuron_id: int, 
        neighbor_id: int, 
        success_score: float
    ) -> None:
        """
        Update attention weights based on success feedback.
        
        Args:
            neuron_id: ID of the neuron that used attention
            neighbor_id: ID of the neighbor that was attended to
            success_score: Success score (0.0 to 1.0) for this attention choice
        """
        key = (neuron_id, neighbor_id)
        
        # Initialize if not exists
        if key not in self.attention_weights:
            self.attention_weights[key] = 1.0
        if key not in self.success_feedback:
            self.success_feedback[key] = []
        
        # Store success feedback
        self.success_feedback[key].append(success_score)
        if len(self.success_feedback[key]) > 50:  # Keep recent feedback
            self.success_feedback[key].pop(0)
        
        # Update attention weight using exponential moving average
        alpha = 0.1
        current_weight = self.attention_weights[key]
        self.attention_weights[key] = alpha * success_score + (1 - alpha) * current_weight
        
        # Apply decay to prevent weight explosion
        self.attention_weights[key] *= self.config.score_decay
        
        # Ensure weights stay in reasonable range
        self.attention_weights[key] = np.clip(self.attention_weights[key], 0.1, 2.0)
    
    def _get_historical_success(self, neuron_id: int, neighbor_id: int) -> float:
        """Get historical success rate for neuron pair."""
        key = (neuron_id, neighbor_id)
        feedback = self.success_feedback.get(key, [])
        
        if not feedback:
            return 0.5  # Neutral prior
        
        # Return average success rate
        return np.mean(feedback)
    
    def _get_adaptive_weight(self, neuron_id: int, neighbor_id: int) -> float:
        """Get adaptive attention weight for neuron pair."""
        key = (neuron_id, neighbor_id)
        return self.attention_weights.get(key, 1.0)
    
    def get_attention_statistics(self) -> Dict:
        """Get statistics about attention mechanism performance."""
        if not self.selection_history:
            return {}
        
        recent_selections = self.selection_history[-100:]  # Last 100 selections
        
        # Average number of candidates and selections
        avg_candidates = np.mean([s['total_candidates'] for s in recent_selections])
        avg_k_used = np.mean([s['k_used'] for s in recent_selections])
        
        # Score distribution statistics
        all_scores = []
        for selection in recent_selections:
            all_scores.extend(selection['scores'].values())
        
        score_stats = {
            'mean_score': np.mean(all_scores) if all_scores else 0.0,
            'std_score': np.std(all_scores) if all_scores else 0.0,
            'min_score': np.min(all_scores) if all_scores else 0.0,
            'max_score': np.max(all_scores) if all_scores else 0.0
        }
        
        # Attention weight statistics
        if self.attention_weights:
            weight_values = list(self.attention_weights.values())
            weight_stats = {
                'mean_weight': np.mean(weight_values),
                'std_weight': np.std(weight_values),
                'num_learned_weights': len(self.attention_weights)
            }
        else:
            weight_stats = {
                'mean_weight': 1.0,
                'std_weight': 0.0,
                'num_learned_weights': 0
            }
        
        return {
            'selection_stats': {
                'avg_candidates': avg_candidates,
                'avg_k_used': avg_k_used,
                'total_selections': len(self.selection_history)
            },
            'score_stats': score_stats,
            'weight_stats': weight_stats,
            'success_feedback_pairs': len(self.success_feedback)
        }
    
    def reset_adaptation(self) -> None:
        """Reset all adaptive weights and feedback history."""
        self.attention_weights.clear()
        self.success_feedback.clear()
        self.selection_history.clear()
    
    def get_most_attended_pairs(self, top_n: int = 10) -> List[Tuple[Tuple[int, int], float]]:
        """Get the most frequently attended neuron pairs."""
        if not self.attention_weights:
            return []
        
        sorted_pairs = sorted(
            self.attention_weights.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_pairs[:top_n]
    
    def export_attention_graph(self) -> Dict:
        """Export attention relationships as a graph structure."""
        edges = []
        nodes = set()
        
        for (from_id, to_id), weight in self.attention_weights.items():
            nodes.add(from_id)
            nodes.add(to_id)
            
            success_rate = self._get_historical_success(from_id, to_id)
            
            edges.append({
                'from': from_id,
                'to': to_id,
                'weight': weight,
                'success_rate': success_rate,
                'type': 'attention'
            })
        
        return {
            'nodes': list(nodes),
            'edges': edges,
            'metadata': {
                'total_attention_pairs': len(self.attention_weights),
                'config': {
                    'top_k': self.config.top_k,
                    'temperature': self.config.temperature,
                    'min_threshold': self.config.min_score_threshold
                }
            }
        }