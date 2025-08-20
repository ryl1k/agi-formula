"""Adaptive attention mechanism with advanced learning capabilities for AGI-Formula."""

from typing import List, Dict, Tuple, Optional, Set
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict, deque
import random
import math


@dataclass
class AttentionPattern:
    """Represents a learned attention pattern."""
    source_neuron: int
    target_neurons: List[int]
    context_signature: np.ndarray
    success_rate: float
    usage_count: int
    last_used: float
    adaptation_rate: float = 0.1


@dataclass
class AdaptiveAttentionConfig:
    """Configuration for adaptive attention mechanism."""
    # Basic attention
    top_k: int = 5
    temperature: float = 1.0
    min_score_threshold: float = 0.1
    
    # Adaptive learning
    learning_rate: float = 0.05
    adaptation_rate: float = 0.1
    pattern_memory_size: int = 1000
    context_window_size: int = 10
    
    # Meta-learning
    meta_learning_enabled: bool = True
    meta_learning_rate: float = 0.01
    exploration_rate: float = 0.1
    exploitation_bonus: float = 0.2
    
    # Pattern recognition
    pattern_similarity_threshold: float = 0.8
    pattern_decay_rate: float = 0.95
    context_importance: float = 0.3
    
    # Advanced features
    multi_head_attention: bool = True
    num_attention_heads: int = 4
    temporal_attention: bool = True
    causal_attention_bias: float = 0.2


class AdaptiveAttentionModule:
    """
    Advanced attention mechanism with adaptive learning and meta-cognition.
    
    Features:
    - Learns successful attention patterns over time
    - Adapts to different contexts and situations
    - Uses meta-learning to improve learning itself
    - Supports multi-head attention for complex scenarios
    - Temporal attention for sequence understanding
    - Causal attention bias for better reasoning
    """
    
    def __init__(self, config: AdaptiveAttentionConfig = None):
        """Initialize adaptive attention module."""
        self.config = config or AdaptiveAttentionConfig()
        
        # Learned attention patterns
        self.attention_patterns: List[AttentionPattern] = []
        self.pattern_index: Dict[int, List[int]] = defaultdict(list)  # neuron_id -> pattern indices
        
        # Context tracking
        self.context_history: deque = deque(maxlen=self.config.context_window_size)
        self.global_context: np.ndarray = np.zeros(20)  # Global context vector
        
        # Multi-head attention
        self.attention_heads: List[Dict] = []
        if self.config.multi_head_attention:
            for i in range(self.config.num_attention_heads):
                self.attention_heads.append({
                    'weights': {},
                    'biases': {},
                    'specialization': f'head_{i}'
                })
        
        # Meta-learning components
        self.meta_parameters: Dict[str, float] = {
            'learning_rate': self.config.learning_rate,
            'exploration_rate': self.config.exploration_rate,
            'temperature': self.config.temperature
        }
        self.meta_performance_history: deque = deque(maxlen=100)
        
        # Temporal attention
        self.temporal_weights: Dict[Tuple[int, int], List[float]] = defaultdict(list)
        self.sequence_memory: deque = deque(maxlen=50)
        
        # Performance tracking
        self.adaptation_events = 0
        self.pattern_matches = 0
        self.exploration_decisions = 0
        self.exploitation_decisions = 0
        
        # Advanced metrics
        self.attention_efficiency_history: List[float] = []
        self.context_prediction_accuracy: List[float] = []
        self.pattern_generalization_scores: List[float] = []
    
    def compute_adaptive_scores(
        self,
        neuron: 'Neuron',
        candidate_neighbors: List[int],
        causal_cache: 'CausalCache',
        current_context: np.ndarray,
        timestep: int
    ) -> Dict[int, float]:
        """
        Compute attention scores using adaptive learning and patterns.
        
        Args:
            neuron: The neuron requesting attention
            candidate_neighbors: Potential neurons to attend to
            causal_cache: Causal information cache
            current_context: Current context vector
            timestep: Current time step
            
        Returns:
            Dictionary of attention scores
        """
        # Update global context
        self._update_global_context(current_context, timestep)
        
        # Base scores from traditional attention
        base_scores = self._compute_base_attention_scores(
            neuron, candidate_neighbors, causal_cache
        )
        
        # Adaptive pattern matching
        pattern_scores = self._compute_pattern_based_scores(
            neuron.id, candidate_neighbors, current_context
        )
        
        # Multi-head attention scores
        multihead_scores = {}
        if self.config.multi_head_attention:
            multihead_scores = self._compute_multihead_scores(
                neuron, candidate_neighbors, current_context
            )
        
        # Temporal attention scores
        temporal_scores = {}
        if self.config.temporal_attention:
            temporal_scores = self._compute_temporal_scores(
                neuron.id, candidate_neighbors, timestep
            )
        
        # Causal attention bias
        causal_scores = self._compute_causal_attention_bias(
            neuron.id, candidate_neighbors, causal_cache
        )
        
        # Meta-learning adjustments
        meta_adjustments = self._apply_meta_learning_adjustments(
            neuron.id, candidate_neighbors
        )
        
        # Combine all scores
        final_scores = {}
        for neighbor_id in candidate_neighbors:
            score = base_scores.get(neighbor_id, 0.0)
            
            # Add pattern-based boost
            score += pattern_scores.get(neighbor_id, 0.0) * 0.3
            
            # Add multi-head contributions
            if multihead_scores:
                score += np.mean([head_scores.get(neighbor_id, 0.0) 
                                for head_scores in multihead_scores.values()]) * 0.2
            
            # Add temporal bias
            score += temporal_scores.get(neighbor_id, 0.0) * 0.15
            
            # Add causal bias
            score += causal_scores.get(neighbor_id, 0.0) * self.config.causal_attention_bias
            
            # Apply meta-learning adjustments
            score *= meta_adjustments.get(neighbor_id, 1.0)
            
            final_scores[neighbor_id] = score
        
        # Apply exploration vs exploitation
        final_scores = self._apply_exploration_strategy(final_scores, current_context)
        
        # Normalize scores
        total_score = sum(final_scores.values())
        if total_score > 0:
            final_scores = {nid: score / total_score for nid, score in final_scores.items()}
        
        return final_scores
    
    def adaptive_select_top_k(
        self,
        scores: Dict[int, float],
        current_context: np.ndarray,
        k: Optional[int] = None
    ) -> List[int]:
        """
        Adaptively select top-k neurons with dynamic k adjustment.
        
        Args:
            scores: Attention scores
            current_context: Current context for adaptation
            k: Number of neurons to select (adaptive if None)
            
        Returns:
            Selected neuron IDs
        """
        if not scores:
            return []
        
        # Adaptive k selection
        if k is None:
            k = self._compute_adaptive_k(scores, current_context)
        
        k = min(k, len(scores))
        
        # Filter by minimum threshold
        filtered_scores = {
            nid: score for nid, score in scores.items()
            if score >= self.config.min_score_threshold
        }
        
        if not filtered_scores:
            return []
        
        # Select top-k with some randomness for exploration
        sorted_items = sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Apply exploration: sometimes select slightly suboptimal choices
        if random.random() < self.meta_parameters['exploration_rate']:
            self.exploration_decisions += 1
            # Add some randomness to selection
            selected = []
            available = list(sorted_items)
            
            for _ in range(min(k, len(available))):
                if available:
                    # Probability weighted by score
                    weights = [item[1] for item in available]
                    total_weight = sum(weights)
                    if total_weight > 0:
                        probs = [w / total_weight for w in weights]
                        idx = np.random.choice(len(available), p=probs)
                        selected.append(available[idx][0])
                        available.pop(idx)
        else:
            self.exploitation_decisions += 1
            # Standard top-k selection
            selected = [nid for nid, score in sorted_items[:k]]
        
        return selected
    
    def update_with_feedback(
        self,
        neuron_id: int,
        selected_neighbors: List[int],
        success_scores: Dict[int, float],
        current_context: np.ndarray,
        timestep: int
    ) -> None:
        """
        Update attention mechanism based on feedback.
        
        Args:
            neuron_id: Source neuron
            selected_neighbors: Neurons that were attended to
            success_scores: Success score for each attended neuron
            current_context: Context during attention
            timestep: Current timestep
        """
        # Update or create attention patterns
        self._update_attention_patterns(
            neuron_id, selected_neighbors, success_scores, current_context, timestep
        )
        
        # Update multi-head attention weights
        if self.config.multi_head_attention:
            self._update_multihead_weights(neuron_id, selected_neighbors, success_scores)
        
        # Update temporal weights
        if self.config.temporal_attention:
            self._update_temporal_weights(neuron_id, selected_neighbors, success_scores, timestep)
        
        # Meta-learning updates
        if self.config.meta_learning_enabled:
            self._update_meta_parameters(success_scores)
        
        # Track performance metrics
        self._update_performance_metrics(success_scores)
        
        self.adaptation_events += 1
    
    def _compute_base_attention_scores(
        self,
        neuron: 'Neuron',
        candidates: List[int],
        causal_cache: 'CausalCache'
    ) -> Dict[int, float]:
        """Compute base attention scores using traditional methods."""
        scores = {}
        epsilon = 1e-6
        
        for neighbor_id in candidates:
            contribution = causal_cache.get_contribution(neighbor_id)
            
            if contribution is None:
                scores[neighbor_id] = self.config.min_score_threshold
                continue
            
            entry = causal_cache.entries.get(neighbor_id)
            if not entry:
                scores[neighbor_id] = self.config.min_score_threshold
                continue
            
            # Base score calculation
            base_score = (abs(contribution) / (1.0 + epsilon)) * entry.confidence
            
            # Add mutual information boost
            mi_boost = getattr(entry, 'mutual_information', 0.0) * 0.3
            
            scores[neighbor_id] = base_score + mi_boost
        
        return scores
    
    def _compute_pattern_based_scores(
        self,
        neuron_id: int,
        candidates: List[int],
        current_context: np.ndarray
    ) -> Dict[int, float]:
        """Compute scores based on learned attention patterns."""
        pattern_scores = {nid: 0.0 for nid in candidates}
        
        # Find matching patterns for this neuron
        if neuron_id not in self.pattern_index:
            return pattern_scores
        
        for pattern_idx in self.pattern_index[neuron_id]:
            if pattern_idx >= len(self.attention_patterns):
                continue
                
            pattern = self.attention_patterns[pattern_idx]
            
            # Check context similarity
            context_similarity = self._compute_context_similarity(
                current_context, pattern.context_signature
            )
            
            if context_similarity >= self.config.pattern_similarity_threshold:
                self.pattern_matches += 1
                
                # Boost scores for neurons in this pattern
                pattern_weight = pattern.success_rate * context_similarity
                
                for target_neuron in pattern.target_neurons:
                    if target_neuron in candidates:
                        pattern_scores[target_neuron] += pattern_weight
        
        return pattern_scores
    
    def _compute_multihead_scores(
        self,
        neuron: 'Neuron',
        candidates: List[int],
        context: np.ndarray
    ) -> Dict[str, Dict[int, float]]:
        """Compute multi-head attention scores."""
        head_scores = {}
        
        for i, head in enumerate(self.attention_heads):
            head_name = head['specialization']
            head_scores[head_name] = {}
            
            for neighbor_id in candidates:
                # Each head has its own learned weights and biases
                weight_key = (neuron.id, neighbor_id)
                weight = head['weights'].get(weight_key, random.uniform(0.8, 1.2))
                bias = head['biases'].get(neighbor_id, 0.0)
                
                # Context-dependent attention
                if len(context) > i:
                    context_factor = context[i % len(context)]
                else:
                    context_factor = 0.5
                
                head_scores[head_name][neighbor_id] = weight * context_factor + bias
        
        return head_scores
    
    def _compute_temporal_scores(
        self,
        neuron_id: int,
        candidates: List[int],
        timestep: int
    ) -> Dict[int, float]:
        """Compute temporal attention scores based on sequence patterns."""
        temporal_scores = {nid: 0.0 for nid in candidates}
        
        # Look for temporal patterns in recent sequence
        if len(self.sequence_memory) < 3:
            return temporal_scores
        
        recent_sequence = list(self.sequence_memory)[-5:]  # Last 5 steps
        
        for neighbor_id in candidates:
            # Check if this neighbor was useful in similar temporal contexts
            temporal_key = (neuron_id, neighbor_id)
            if temporal_key in self.temporal_weights:
                weights = self.temporal_weights[temporal_key]
                if weights:
                    # Weight by recency (more recent = higher weight)
                    weighted_score = 0.0
                    total_weight = 0.0
                    
                    for i, weight in enumerate(weights[-5:]):  # Last 5 temporal weights
                        recency_factor = (i + 1) / len(weights[-5:])
                        weighted_score += weight * recency_factor
                        total_weight += recency_factor
                    
                    if total_weight > 0:
                        temporal_scores[neighbor_id] = weighted_score / total_weight
        
        # Store current step in sequence memory
        self.sequence_memory.append((timestep, neuron_id, candidates.copy()))
        
        return temporal_scores
    
    def _compute_causal_attention_bias(
        self,
        neuron_id: int,
        candidates: List[int],
        causal_cache: 'CausalCache'
    ) -> Dict[int, float]:
        """Compute attention bias based on causal relationships."""
        causal_scores = {nid: 0.0 for nid in candidates}
        
        # Bias towards neurons that have strong causal relationships
        for neighbor_id in candidates:
            # Check if this neighbor is a known cause
            if hasattr(causal_cache, 'entries') and neuron_id in causal_cache.entries:
                entry = causal_cache.entries[neuron_id]
                
                if hasattr(entry, 'direct_causes') and neighbor_id in entry.direct_causes:
                    causal_scores[neighbor_id] += 0.3  # Strong bias for direct causes
                elif hasattr(entry, 'indirect_causes') and neighbor_id in entry.indirect_causes:
                    causal_scores[neighbor_id] += 0.1  # Weaker bias for indirect causes
            
            # Check reverse causation (this neuron causes the neighbor)
            if hasattr(causal_cache, 'entries') and neighbor_id in causal_cache.entries:
                neighbor_entry = causal_cache.entries[neighbor_id]
                
                if hasattr(neighbor_entry, 'direct_causes') and neuron_id in neighbor_entry.direct_causes:
                    causal_scores[neighbor_id] += 0.2  # Moderate bias for reverse causation
        
        return causal_scores
    
    def _apply_meta_learning_adjustments(
        self,
        neuron_id: int,
        candidates: List[int]
    ) -> Dict[int, float]:
        """Apply meta-learning adjustments to attention scores."""
        adjustments = {nid: 1.0 for nid in candidates}
        
        if not self.config.meta_learning_enabled:
            return adjustments
        
        # Adjust based on recent performance
        if len(self.meta_performance_history) > 10:
            recent_performance = np.mean(list(self.meta_performance_history)[-10:])
            
            # If performance is declining, increase exploration
            if recent_performance < 0.5:
                exploration_boost = 1.2
                for nid in candidates:
                    adjustments[nid] *= exploration_boost
            
            # If performance is good, slightly favor exploitation
            elif recent_performance > 0.8:
                exploitation_boost = 1.1
                # Boost highest-scoring candidates more
                for nid in candidates:
                    adjustments[nid] *= exploitation_boost
        
        return adjustments
    
    def _apply_exploration_strategy(
        self,
        scores: Dict[int, float],
        context: np.ndarray
    ) -> Dict[int, float]:
        """Apply exploration vs exploitation strategy."""
        if not scores:
            return scores
        
        # Context-dependent exploration rate
        context_novelty = self._assess_context_novelty(context)
        dynamic_exploration_rate = self.meta_parameters['exploration_rate'] * (1.0 + context_novelty)
        
        # Apply exploration noise
        if random.random() < dynamic_exploration_rate:
            # Add small random noise to encourage exploration
            noise_strength = 0.1 * dynamic_exploration_rate
            
            for nid in scores:
                noise = random.uniform(-noise_strength, noise_strength)
                scores[nid] += noise
                scores[nid] = max(0.0, scores[nid])  # Keep non-negative
        
        return scores
    
    def _compute_adaptive_k(self, scores: Dict[int, float], context: np.ndarray) -> int:
        """Adaptively compute the value of k based on context and scores."""
        base_k = self.config.top_k
        
        # Adjust k based on score distribution
        if scores:
            score_values = list(scores.values())
            score_std = np.std(score_values)
            score_mean = np.mean(score_values)
            
            # If scores are very concentrated, use fewer neurons
            if score_std < 0.1 * score_mean:
                k = max(1, base_k - 1)
            # If scores are spread out, use more neurons
            elif score_std > 0.5 * score_mean:
                k = min(len(scores), base_k + 2)
            else:
                k = base_k
        else:
            k = base_k
        
        # Context-based adjustment
        context_complexity = np.std(context) if len(context) > 1 else 0.5
        if context_complexity > 0.8:
            k += 1  # More complex context might need more attention
        
        return max(1, min(k, len(scores)))
    
    def _update_attention_patterns(
        self,
        neuron_id: int,
        selected_neighbors: List[int],
        success_scores: Dict[int, float],
        context: np.ndarray,
        timestep: int
    ) -> None:
        """Update learned attention patterns based on feedback."""
        if not selected_neighbors:
            return
        
        avg_success = np.mean(list(success_scores.values())) if success_scores else 0.5
        
        # Find existing pattern or create new one
        matching_pattern = None
        for pattern_idx in self.pattern_index.get(neuron_id, []):
            if pattern_idx < len(self.attention_patterns):
                pattern = self.attention_patterns[pattern_idx]
                context_similarity = self._compute_context_similarity(context, pattern.context_signature)
                
                if context_similarity >= self.config.pattern_similarity_threshold:
                    matching_pattern = pattern
                    break
        
        if matching_pattern:
            # Update existing pattern
            old_rate = matching_pattern.success_rate
            matching_pattern.success_rate = (
                (1 - matching_pattern.adaptation_rate) * old_rate +
                matching_pattern.adaptation_rate * avg_success
            )
            matching_pattern.usage_count += 1
            matching_pattern.last_used = timestep
        else:
            # Create new pattern
            new_pattern = AttentionPattern(
                source_neuron=neuron_id,
                target_neurons=selected_neighbors.copy(),
                context_signature=context.copy(),
                success_rate=avg_success,
                usage_count=1,
                last_used=timestep,
                adaptation_rate=self.config.adaptation_rate
            )
            
            self.attention_patterns.append(new_pattern)
            pattern_idx = len(self.attention_patterns) - 1
            self.pattern_index[neuron_id].append(pattern_idx)
        
        # Cleanup old patterns
        self._cleanup_attention_patterns()
    
    def _update_multihead_weights(
        self,
        neuron_id: int,
        selected_neighbors: List[int],
        success_scores: Dict[int, float]
    ) -> None:
        """Update multi-head attention weights based on feedback."""
        for i, head in enumerate(self.attention_heads):
            for neighbor_id in selected_neighbors:
                weight_key = (neuron_id, neighbor_id)
                success = success_scores.get(neighbor_id, 0.5)
                
                # Update weight using gradient-like update
                current_weight = head['weights'].get(weight_key, 1.0)
                head['weights'][weight_key] = current_weight + self.config.learning_rate * (success - 0.5)
                
                # Update bias
                current_bias = head['biases'].get(neighbor_id, 0.0)
                head['biases'][neighbor_id] = current_bias + self.config.learning_rate * (success - 0.5) * 0.1
    
    def _update_temporal_weights(
        self,
        neuron_id: int,
        selected_neighbors: List[int],
        success_scores: Dict[int, float],
        timestep: int
    ) -> None:
        """Update temporal attention weights."""
        for neighbor_id in selected_neighbors:
            temporal_key = (neuron_id, neighbor_id)
            success = success_scores.get(neighbor_id, 0.5)
            
            self.temporal_weights[temporal_key].append(success)
            
            # Keep only recent temporal weights
            if len(self.temporal_weights[temporal_key]) > 20:
                self.temporal_weights[temporal_key].pop(0)
    
    def _update_meta_parameters(self, success_scores: Dict[int, float]) -> None:
        """Update meta-learning parameters based on performance."""
        if not success_scores:
            return
        
        avg_success = np.mean(list(success_scores.values()))
        self.meta_performance_history.append(avg_success)
        
        # Adapt meta-parameters based on recent performance
        if len(self.meta_performance_history) >= 10:
            recent_trend = np.mean(list(self.meta_performance_history)[-5:]) - np.mean(list(self.meta_performance_history)[-10:-5])
            
            # If performance is improving, continue current strategy
            if recent_trend > 0.05:
                pass  # Keep current parameters
            
            # If performance is declining, adjust parameters
            elif recent_trend < -0.05:
                # Increase exploration
                self.meta_parameters['exploration_rate'] = min(0.3, self.meta_parameters['exploration_rate'] * 1.1)
                # Decrease temperature for more focused attention
                self.meta_parameters['temperature'] = max(0.5, self.meta_parameters['temperature'] * 0.95)
            
            # If performance is stable but low, try different strategies
            elif avg_success < 0.4:
                # Increase learning rate
                self.meta_parameters['learning_rate'] = min(0.1, self.meta_parameters['learning_rate'] * 1.05)
    
    def _update_performance_metrics(self, success_scores: Dict[int, float]) -> None:
        """Update various performance metrics."""
        if success_scores:
            avg_success = np.mean(list(success_scores.values()))
            self.attention_efficiency_history.append(avg_success)
            
            # Keep only recent history
            if len(self.attention_efficiency_history) > 100:
                self.attention_efficiency_history.pop(0)
    
    def _compute_context_similarity(self, context1: np.ndarray, context2: np.ndarray) -> float:
        """Compute similarity between two context vectors."""
        if len(context1) != len(context2):
            return 0.0
        
        # Cosine similarity
        norm1 = np.linalg.norm(context1)
        norm2 = np.linalg.norm(context2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(context1, context2) / (norm1 * norm2)
    
    def _assess_context_novelty(self, context: np.ndarray) -> float:
        """Assess how novel the current context is."""
        if len(self.context_history) < 5:
            return 1.0  # High novelty if we have little history
        
        similarities = []
        for past_context in self.context_history:
            if len(past_context) == len(context):
                similarity = self._compute_context_similarity(context, past_context)
                similarities.append(similarity)
        
        if similarities:
            max_similarity = max(similarities)
            return 1.0 - max_similarity  # High novelty = low similarity to past
        else:
            return 1.0
    
    def _update_global_context(self, current_context: np.ndarray, timestep: int) -> None:
        """Update global context vector."""
        # Add current context to history
        self.context_history.append(current_context.copy())
        
        # Update global context as exponential moving average
        alpha = 0.1
        if len(current_context) == len(self.global_context):
            self.global_context = alpha * current_context + (1 - alpha) * self.global_context
        else:
            # Resize global context if needed
            self.global_context = current_context.copy()
    
    def _cleanup_attention_patterns(self) -> None:
        """Clean up old or unused attention patterns."""
        if len(self.attention_patterns) <= self.config.pattern_memory_size:
            return
        
        # Sort patterns by usage and success
        pattern_scores = []
        for i, pattern in enumerate(self.attention_patterns):
            score = pattern.success_rate * math.log(1 + pattern.usage_count)
            pattern_scores.append((score, i))
        
        # Keep top patterns
        pattern_scores.sort(reverse=True)
        indices_to_keep = [idx for _, idx in pattern_scores[:self.config.pattern_memory_size]]
        
        new_patterns = [self.attention_patterns[i] for i in indices_to_keep]
        self.attention_patterns = new_patterns
        
        # Rebuild pattern index
        self.pattern_index.clear()
        for i, pattern in enumerate(self.attention_patterns):
            self.pattern_index[pattern.source_neuron].append(i)
    
    def get_adaptive_statistics(self) -> Dict:
        """Get comprehensive statistics about adaptive attention."""
        stats = {
            'adaptation_events': self.adaptation_events,
            'pattern_matches': self.pattern_matches,
            'exploration_decisions': self.exploration_decisions,
            'exploitation_decisions': self.exploitation_decisions,
            
            'learned_patterns': len(self.attention_patterns),
            'neurons_with_patterns': len(self.pattern_index),
            'avg_pattern_success': np.mean([p.success_rate for p in self.attention_patterns]) if self.attention_patterns else 0.0,
            
            'meta_parameters': self.meta_parameters.copy(),
            'recent_performance': np.mean(list(self.meta_performance_history)[-10:]) if len(self.meta_performance_history) >= 10 else 0.5,
            
            'efficiency_trend': np.mean(self.attention_efficiency_history[-20:]) if len(self.attention_efficiency_history) >= 20 else 0.5,
            
            'temporal_patterns': len(self.temporal_weights),
            'context_diversity': len(self.context_history),
        }
        
        if self.config.multi_head_attention:
            stats['multihead_weights'] = sum(len(head['weights']) for head in self.attention_heads)
            stats['multihead_biases'] = sum(len(head['biases']) for head in self.attention_heads)
        
        return stats