"""
Cross-Modal Attention Mechanisms for AGI-Formula

Advanced attention mechanisms for multi-modal information processing:
- Cross-modal attention between different sensory modalities
- Self-attention within modalities
- Hierarchical attention across multiple scales
- Temporal attention for sequential data
- Adaptive attention with dynamic focus
- Multi-head attention with learned representations
"""

import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

from .data_pipeline import ModalityType, MultiModalFrame, ModalityData


class AttentionMechanism(Enum):
    """Types of attention mechanisms"""
    CROSS_MODAL = "cross_modal"
    SELF_ATTENTION = "self_attention"
    HIERARCHICAL = "hierarchical"
    TEMPORAL = "temporal"
    ADAPTIVE = "adaptive"
    MULTI_HEAD = "multi_head"
    SPARSE = "sparse"
    GRAPH = "graph"


class AttentionType(Enum):
    """Attention computation types"""
    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"
    SCALED_DOT_PRODUCT = "scaled_dot_product"
    BILINEAR = "bilinear"
    CONTENT_BASED = "content_based"


@dataclass
class AttentionWeights:
    """Attention weight matrices and metadata"""
    weights: np.ndarray
    mechanism: AttentionMechanism
    source_modality: Optional[ModalityType]
    target_modality: Optional[ModalityType]
    attention_scores: Optional[np.ndarray] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AttentionOutput:
    """Output of attention mechanism"""
    attended_features: np.ndarray
    attention_weights: AttentionWeights
    attention_distribution: np.ndarray
    entropy: float
    processing_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class CrossModalAttention:
    """
    Advanced cross-modal attention system for multi-modal AGI
    
    Features:
    - Multi-head cross-modal attention
    - Adaptive attention mechanisms
    - Hierarchical attention processing
    - Temporal attention for sequences
    - Graph-based attention for structured data
    - Sparse attention for efficiency
    - Learned attention patterns
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        
        # Attention mechanisms
        self.attention_mechanisms = {
            AttentionMechanism.CROSS_MODAL: self._cross_modal_attention,
            AttentionMechanism.SELF_ATTENTION: self._self_attention,
            AttentionMechanism.HIERARCHICAL: self._hierarchical_attention,
            AttentionMechanism.TEMPORAL: self._temporal_attention,
            AttentionMechanism.ADAPTIVE: self._adaptive_attention,
            AttentionMechanism.MULTI_HEAD: self._multi_head_attention,
            AttentionMechanism.SPARSE: self._sparse_attention,
            AttentionMechanism.GRAPH: self._graph_attention
        }
        
        # Attention components
        self.multi_head_processor = MultiHeadAttentionProcessor(self.config.get('multi_head', {}))
        self.temporal_processor = TemporalAttentionProcessor(self.config.get('temporal', {}))
        self.adaptive_controller = AdaptiveAttentionController(self.config.get('adaptive', {}))
        
        # Learned parameters
        self.attention_matrices = {}
        self.bias_vectors = {}
        self.scaling_factors = {}
        
        # State and history
        self.attention_history = []
        self.learned_patterns = {}
        
        # Performance monitoring
        self.stats = {
            'attention_computations': 0,
            'processing_times_ms': [],
            'attention_entropies': [],
            'mechanism_usage': {mech: 0 for mech in AttentionMechanism}
        }
        
        # Initialize attention system
        self._initialize_attention()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for cross-modal attention"""
        return {
            'feature_dim': 512,
            'attention_dim': 64,
            'num_heads': 8,
            'temperature': 1.0,
            'dropout_rate': 0.1,
            'max_sequence_length': 100,
            'attention_types': [
                AttentionType.SCALED_DOT_PRODUCT,
                AttentionType.ADDITIVE,
                AttentionType.BILINEAR
            ],
            'modality_combinations': [
                (ModalityType.VISION, ModalityType.LANGUAGE),
                (ModalityType.VISION, ModalityType.AUDIO),
                (ModalityType.LANGUAGE, ModalityType.AUDIO),
                (ModalityType.TACTILE, ModalityType.PROPRIOCEPTION)
            ],
            'hierarchical_levels': 3,
            'sparse_attention': {
                'enabled': True,
                'sparsity_ratio': 0.1,
                'top_k': 10
            },
            'adaptive_control': {
                'learning_rate': 0.01,
                'adaptation_window': 10,
                'focus_threshold': 0.5
            }
        }
    
    def _initialize_attention(self):
        """Initialize attention mechanisms and parameters"""
        feature_dim = self.config['feature_dim']
        attention_dim = self.config['attention_dim']
        
        # Initialize attention matrices for each modality pair
        for source_mod in ModalityType:
            for target_mod in ModalityType:
                key = (source_mod, target_mod)
                
                # Query, Key, Value projection matrices
                self.attention_matrices[f"W_q_{key}"] = np.random.normal(
                    0, 0.1, (feature_dim, attention_dim)
                ).astype(np.float32)
                
                self.attention_matrices[f"W_k_{key}"] = np.random.normal(
                    0, 0.1, (feature_dim, attention_dim)
                ).astype(np.float32)
                
                self.attention_matrices[f"W_v_{key}"] = np.random.normal(
                    0, 0.1, (feature_dim, feature_dim)
                ).astype(np.float32)
                
                # Bias vectors
                self.bias_vectors[f"b_q_{key}"] = np.zeros(attention_dim).astype(np.float32)
                self.bias_vectors[f"b_k_{key}"] = np.zeros(attention_dim).astype(np.float32)
                self.bias_vectors[f"b_v_{key}"] = np.zeros(feature_dim).astype(np.float32)
        
        # Initialize scaling factors
        self.scaling_factors['temperature'] = self.config['temperature']
        self.scaling_factors['attention_scale'] = 1.0 / np.sqrt(attention_dim)
        
        print(f"Cross-modal attention initialized with {len(self.attention_mechanisms)} mechanisms")
    
    def compute_attention(self, frame: MultiModalFrame, 
                         mechanism: AttentionMechanism = AttentionMechanism.CROSS_MODAL,
                         source_modality: Optional[ModalityType] = None,
                         target_modality: Optional[ModalityType] = None) -> AttentionOutput:
        """Compute attention between modalities in a frame"""
        start_time = time.time()
        
        # Validate inputs
        if mechanism not in self.attention_mechanisms:
            raise ValueError(f"Unknown attention mechanism: {mechanism}")
        
        # Apply attention mechanism
        attention_result = self.attention_mechanisms[mechanism](
            frame, source_modality, target_modality
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        # Update statistics
        self._update_stats(mechanism, processing_time, attention_result)
        
        return attention_result
    
    def _cross_modal_attention(self, frame: MultiModalFrame, 
                              source_modality: Optional[ModalityType],
                              target_modality: Optional[ModalityType]) -> AttentionOutput:
        """Compute cross-modal attention between different modalities"""
        if source_modality is None or target_modality is None:
            # Compute attention between all modality pairs
            return self._compute_all_pairs_attention(frame)
        
        # Get modality data
        if source_modality not in frame.modalities or target_modality not in frame.modalities:
            return self._create_empty_attention_output()
        
        source_data = frame.modalities[source_modality]
        target_data = frame.modalities[target_modality]
        
        if source_data.features is None or target_data.features is None:
            return self._create_empty_attention_output()
        
        # Compute cross-modal attention
        attended_features, attention_weights, attention_scores = self._compute_scaled_dot_product_attention(
            source_data.features, target_data.features, target_data.features,
            source_modality, target_modality
        )
        
        # Calculate attention entropy
        entropy = self._compute_attention_entropy(attention_weights.weights)
        
        # Create attention weights object
        attn_weights = AttentionWeights(
            weights=attention_weights.weights,
            mechanism=AttentionMechanism.CROSS_MODAL,
            source_modality=source_modality,
            target_modality=target_modality,
            attention_scores=attention_scores,
            confidence=min(source_data.confidence, target_data.confidence)
        )
        
        return AttentionOutput(
            attended_features=attended_features,
            attention_weights=attn_weights,
            attention_distribution=attention_weights.weights,
            entropy=entropy,
            processing_time_ms=0,  # Will be set by caller
            metadata={
                'source_modality': source_modality.value,
                'target_modality': target_modality.value,
                'attention_type': 'cross_modal'
            }
        )
    
    def _compute_all_pairs_attention(self, frame: MultiModalFrame) -> AttentionOutput:
        """Compute attention between all modality pairs"""
        modalities = list(frame.modalities.keys())
        
        if len(modalities) < 2:
            return self._create_empty_attention_output()
        
        all_attended_features = []
        all_attention_weights = []
        total_entropy = 0.0
        
        # Compute attention for each pair
        for i, source_mod in enumerate(modalities):
            for j, target_mod in enumerate(modalities):
                if i != j:  # Skip self-attention for now
                    pair_result = self._cross_modal_attention(frame, source_mod, target_mod)
                    all_attended_features.append(pair_result.attended_features)
                    all_attention_weights.append(pair_result.attention_weights.weights)
                    total_entropy += pair_result.entropy
        
        if not all_attended_features:
            return self._create_empty_attention_output()
        
        # Combine all attended features
        combined_features = np.mean(all_attended_features, axis=0)
        combined_weights = np.mean(all_attention_weights, axis=0)
        avg_entropy = total_entropy / len(all_attended_features)
        
        attn_weights = AttentionWeights(
            weights=combined_weights,
            mechanism=AttentionMechanism.CROSS_MODAL,
            source_modality=None,
            target_modality=None,
            confidence=np.mean([data.confidence for data in frame.modalities.values()])
        )
        
        return AttentionOutput(
            attended_features=combined_features,
            attention_weights=attn_weights,
            attention_distribution=combined_weights,
            entropy=avg_entropy,
            processing_time_ms=0,
            metadata={'attention_type': 'all_pairs_cross_modal'}
        )
    
    def _self_attention(self, frame: MultiModalFrame, 
                       source_modality: Optional[ModalityType],
                       target_modality: Optional[ModalityType]) -> AttentionOutput:
        """Compute self-attention within a modality"""
        if source_modality is None:
            # Apply self-attention to all modalities
            return self._compute_all_modalities_self_attention(frame)
        
        if source_modality not in frame.modalities:
            return self._create_empty_attention_output()
        
        modality_data = frame.modalities[source_modality]
        if modality_data.features is None:
            return self._create_empty_attention_output()
        
        # Self-attention uses the same features for Q, K, V
        features = modality_data.features
        
        # Reshape features for sequence-like self-attention
        # For simplicity, treat feature dimensions as sequence elements
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        attended_features, attention_weights, attention_scores = self._compute_scaled_dot_product_attention(
            features, features, features, source_modality, source_modality
        )
        
        entropy = self._compute_attention_entropy(attention_weights.weights)
        
        attn_weights = AttentionWeights(
            weights=attention_weights.weights,
            mechanism=AttentionMechanism.SELF_ATTENTION,
            source_modality=source_modality,
            target_modality=source_modality,
            attention_scores=attention_scores,
            confidence=modality_data.confidence
        )
        
        return AttentionOutput(
            attended_features=attended_features.flatten(),
            attention_weights=attn_weights,
            attention_distribution=attention_weights.weights,
            entropy=entropy,
            processing_time_ms=0,
            metadata={
                'modality': source_modality.value,
                'attention_type': 'self_attention'
            }
        )
    
    def _hierarchical_attention(self, frame: MultiModalFrame, 
                              source_modality: Optional[ModalityType],
                              target_modality: Optional[ModalityType]) -> AttentionOutput:
        """Compute hierarchical attention across multiple levels"""
        levels = self.config['hierarchical_levels']
        
        # Level 1: Within-modality attention
        level1_results = []
        for modality in frame.modalities:
            result = self._self_attention(frame, modality, None)
            level1_results.append(result)
        
        # Level 2: Cross-modality attention
        level2_results = []
        modalities = list(frame.modalities.keys())
        for i, source_mod in enumerate(modalities):
            for j, target_mod in enumerate(modalities):
                if i != j:
                    result = self._cross_modal_attention(frame, source_mod, target_mod)
                    level2_results.append(result)
        
        # Level 3: Global attention integration
        all_features = []
        all_weights = []
        
        for result in level1_results + level2_results:
            all_features.append(result.attended_features)
            all_weights.append(result.attention_weights.weights)
        
        if not all_features:
            return self._create_empty_attention_output()
        
        # Hierarchical combination
        combined_features = self._hierarchical_combination(all_features, levels)
        combined_weights = np.mean(all_weights, axis=0)
        entropy = np.mean([r.entropy for r in level1_results + level2_results])
        
        attn_weights = AttentionWeights(
            weights=combined_weights,
            mechanism=AttentionMechanism.HIERARCHICAL,
            source_modality=None,
            target_modality=None,
            confidence=np.mean([data.confidence for data in frame.modalities.values()])
        )
        
        return AttentionOutput(
            attended_features=combined_features,
            attention_weights=attn_weights,
            attention_distribution=combined_weights,
            entropy=entropy,
            processing_time_ms=0,
            metadata={'attention_type': 'hierarchical', 'levels': levels}
        )
    
    def _temporal_attention(self, frame: MultiModalFrame, 
                          source_modality: Optional[ModalityType],
                          target_modality: Optional[ModalityType]) -> AttentionOutput:
        """Compute temporal attention across frame history"""
        return self.temporal_processor.compute_temporal_attention(
            frame, self.attention_history, source_modality, target_modality
        )
    
    def _adaptive_attention(self, frame: MultiModalFrame, 
                          source_modality: Optional[ModalityType],
                          target_modality: Optional[ModalityType]) -> AttentionOutput:
        """Compute adaptive attention with dynamic focus"""
        return self.adaptive_controller.compute_adaptive_attention(
            frame, self.learned_patterns, source_modality, target_modality
        )
    
    def _multi_head_attention(self, frame: MultiModalFrame, 
                            source_modality: Optional[ModalityType],
                            target_modality: Optional[ModalityType]) -> AttentionOutput:
        """Compute multi-head attention"""
        return self.multi_head_processor.compute_multi_head_attention(
            frame, source_modality, target_modality
        )
    
    def _sparse_attention(self, frame: MultiModalFrame, 
                         source_modality: Optional[ModalityType],
                         target_modality: Optional[ModalityType]) -> AttentionOutput:
        """Compute sparse attention for efficiency"""
        # First compute regular cross-modal attention
        full_attention = self._cross_modal_attention(frame, source_modality, target_modality)
        
        # Apply sparsity
        sparse_weights = self._apply_sparsity(
            full_attention.attention_weights.weights,
            self.config['sparse_attention']['sparsity_ratio']
        )
        
        # Recompute attended features with sparse weights
        if source_modality and target_modality:
            if source_modality in frame.modalities and target_modality in frame.modalities:
                target_features = frame.modalities[target_modality].features
                if target_features is not None:
                    attended_features = np.dot(sparse_weights, target_features.reshape(-1, 1)).flatten()
                else:
                    attended_features = full_attention.attended_features
            else:
                attended_features = full_attention.attended_features
        else:
            attended_features = full_attention.attended_features
        
        attn_weights = AttentionWeights(
            weights=sparse_weights,
            mechanism=AttentionMechanism.SPARSE,
            source_modality=source_modality,
            target_modality=target_modality,
            confidence=full_attention.attention_weights.confidence
        )
        
        return AttentionOutput(
            attended_features=attended_features,
            attention_weights=attn_weights,
            attention_distribution=sparse_weights,
            entropy=self._compute_attention_entropy(sparse_weights),
            processing_time_ms=0,
            metadata={'attention_type': 'sparse', 'sparsity_applied': True}
        )
    
    def _graph_attention(self, frame: MultiModalFrame, 
                        source_modality: Optional[ModalityType],
                        target_modality: Optional[ModalityType]) -> AttentionOutput:
        """Compute graph-based attention using modality relationships"""
        # Create modality graph
        modality_graph = self._create_modality_graph(frame)
        
        # Apply graph attention
        attended_features, attention_weights = self._apply_graph_attention(
            frame, modality_graph, source_modality, target_modality
        )
        
        entropy = self._compute_attention_entropy(attention_weights)
        
        attn_weights = AttentionWeights(
            weights=attention_weights,
            mechanism=AttentionMechanism.GRAPH,
            source_modality=source_modality,
            target_modality=target_modality,
            confidence=np.mean([data.confidence for data in frame.modalities.values()])
        )
        
        return AttentionOutput(
            attended_features=attended_features,
            attention_weights=attn_weights,
            attention_distribution=attention_weights,
            entropy=entropy,
            processing_time_ms=0,
            metadata={'attention_type': 'graph'}
        )
    
    def _compute_scaled_dot_product_attention(self, query_features: np.ndarray,
                                            key_features: np.ndarray,
                                            value_features: np.ndarray,
                                            source_modality: ModalityType,
                                            target_modality: ModalityType) -> Tuple[np.ndarray, AttentionWeights, np.ndarray]:
        """Compute scaled dot-product attention"""
        # Get projection matrices
        key = (source_modality, target_modality)
        
        W_q = self.attention_matrices.get(f"W_q_{key}", np.eye(len(query_features), self.config['attention_dim']))
        W_k = self.attention_matrices.get(f"W_k_{key}", np.eye(len(key_features), self.config['attention_dim']))
        W_v = self.attention_matrices.get(f"W_v_{key}", np.eye(len(value_features)))
        
        # Ensure proper dimensions
        if len(query_features.shape) == 1:
            query_features = query_features.reshape(1, -1)
        if len(key_features.shape) == 1:
            key_features = key_features.reshape(1, -1)
        if len(value_features.shape) == 1:
            value_features = value_features.reshape(1, -1)
        
        # Project features
        try:
            # Adjust matrix dimensions if needed
            if W_q.shape[0] != query_features.shape[1]:
                W_q = np.random.normal(0, 0.1, (query_features.shape[1], self.config['attention_dim'])).astype(np.float32)
            if W_k.shape[0] != key_features.shape[1]:
                W_k = np.random.normal(0, 0.1, (key_features.shape[1], self.config['attention_dim'])).astype(np.float32)
            if W_v.shape[0] != value_features.shape[1]:
                W_v = np.random.normal(0, 0.1, (value_features.shape[1], value_features.shape[1])).astype(np.float32)
            
            Q = np.dot(query_features, W_q)
            K = np.dot(key_features, W_k)
            V = np.dot(value_features, W_v)
        except Exception as e:
            logging.warning(f"Error in attention projection: {e}")
            # Fallback to identity projection
            Q = query_features[:, :self.config['attention_dim']] if query_features.shape[1] >= self.config['attention_dim'] else query_features
            K = key_features[:, :self.config['attention_dim']] if key_features.shape[1] >= self.config['attention_dim'] else key_features
            V = value_features
        
        # Compute attention scores
        attention_scores = np.dot(Q, K.T) * self.scaling_factors['attention_scale']
        
        # Apply temperature scaling
        attention_scores = attention_scores / self.scaling_factors['temperature']
        
        # Apply softmax
        attention_weights = self._softmax(attention_scores)
        
        # Apply attention to values
        attended_features = np.dot(attention_weights, V)
        
        # Flatten if needed
        if attended_features.ndim > 1:
            attended_features = attended_features.flatten()
        
        attn_weights = AttentionWeights(
            weights=attention_weights,
            mechanism=AttentionMechanism.CROSS_MODAL,
            source_modality=source_modality,
            target_modality=target_modality
        )
        
        return attended_features, attn_weights, attention_scores
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax with numerical stability"""
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        
        # Numerical stability
        x_max = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)
        sum_exp_x = np.sum(exp_x, axis=-1, keepdims=True)
        
        return exp_x / (sum_exp_x + 1e-8)
    
    def _compute_attention_entropy(self, attention_weights: np.ndarray) -> float:
        """Compute entropy of attention distribution"""
        # Flatten weights if needed
        weights = attention_weights.flatten()
        
        # Add small epsilon to avoid log(0)
        weights = weights + 1e-8
        
        # Normalize to ensure it's a probability distribution
        weights = weights / np.sum(weights)
        
        # Compute entropy
        entropy = -np.sum(weights * np.log(weights))
        
        return float(entropy)
    
    def _hierarchical_combination(self, features_list: List[np.ndarray], levels: int) -> np.ndarray:
        """Combine features hierarchically across levels"""
        if not features_list:
            return np.zeros(self.config['feature_dim'])
        
        # Ensure all features have the same length
        max_len = max(len(f) for f in features_list)
        normalized_features = []
        
        for features in features_list:
            if len(features) < max_len:
                padded = np.pad(features, (0, max_len - len(features)))
            else:
                padded = features[:max_len]
            normalized_features.append(padded)
        
        # Hierarchical combination with weighted averaging
        level_weights = np.exp(-0.5 * np.arange(len(normalized_features)))
        level_weights = level_weights / np.sum(level_weights)
        
        combined = np.average(normalized_features, axis=0, weights=level_weights)
        
        return combined.astype(np.float32)
    
    def _apply_sparsity(self, attention_weights: np.ndarray, sparsity_ratio: float) -> np.ndarray:
        """Apply sparsity to attention weights"""
        flat_weights = attention_weights.flatten()
        
        # Calculate threshold for sparsity
        threshold = np.percentile(flat_weights, sparsity_ratio * 100)
        
        # Apply threshold
        sparse_weights = np.where(flat_weights >= threshold, flat_weights, 0)
        
        # Renormalize
        if np.sum(sparse_weights) > 0:
            sparse_weights = sparse_weights / np.sum(sparse_weights)
        
        return sparse_weights.reshape(attention_weights.shape)
    
    def _create_modality_graph(self, frame: MultiModalFrame) -> Dict[str, Any]:
        """Create graph representation of modality relationships"""
        modalities = list(frame.modalities.keys())
        n_modalities = len(modalities)
        
        # Create adjacency matrix based on modality compatibility
        adjacency = np.zeros((n_modalities, n_modalities))
        
        for i, mod1 in enumerate(modalities):
            for j, mod2 in enumerate(modalities):
                if i != j:
                    # Compute compatibility score
                    compatibility = self._compute_modality_compatibility(mod1, mod2)
                    adjacency[i, j] = compatibility
        
        return {
            'modalities': modalities,
            'adjacency': adjacency,
            'node_features': {
                mod: frame.modalities[mod].features 
                for mod in modalities 
                if frame.modalities[mod].features is not None
            }
        }
    
    def _compute_modality_compatibility(self, mod1: ModalityType, mod2: ModalityType) -> float:
        """Compute compatibility score between two modalities"""
        # Predefined compatibility matrix
        compatibility_matrix = {
            (ModalityType.VISION, ModalityType.LANGUAGE): 0.8,
            (ModalityType.VISION, ModalityType.AUDIO): 0.7,
            (ModalityType.LANGUAGE, ModalityType.AUDIO): 0.9,
            (ModalityType.TACTILE, ModalityType.PROPRIOCEPTION): 0.6,
            (ModalityType.VISION, ModalityType.TACTILE): 0.5,
            (ModalityType.AUDIO, ModalityType.TACTILE): 0.4,
        }
        
        # Check both directions
        key1 = (mod1, mod2)
        key2 = (mod2, mod1)
        
        if key1 in compatibility_matrix:
            return compatibility_matrix[key1]
        elif key2 in compatibility_matrix:
            return compatibility_matrix[key2]
        else:
            return 0.3  # Default low compatibility
    
    def _apply_graph_attention(self, frame: MultiModalFrame, 
                             modality_graph: Dict[str, Any],
                             source_modality: Optional[ModalityType],
                             target_modality: Optional[ModalityType]) -> Tuple[np.ndarray, np.ndarray]:
        """Apply graph attention mechanism"""
        modalities = modality_graph['modalities']
        adjacency = modality_graph['adjacency']
        node_features = modality_graph['node_features']
        
        if not node_features:
            return np.zeros(self.config['feature_dim']), np.zeros((len(modalities), len(modalities)))
        
        # Graph attention computation (simplified)
        n_nodes = len(modalities)
        attention_matrix = np.zeros((n_nodes, n_nodes))
        
        for i in range(n_nodes):
            for j in range(n_nodes):
                if adjacency[i, j] > 0:
                    # Compute attention based on feature similarity and graph structure
                    mod_i = modalities[i]
                    mod_j = modalities[j]
                    
                    if mod_i in node_features and mod_j in node_features:
                        feat_i = node_features[mod_i]
                        feat_j = node_features[mod_j]
                        
                        # Feature similarity
                        similarity = self._compute_feature_similarity(feat_i, feat_j)
                        
                        # Graph structure weight
                        structure_weight = adjacency[i, j]
                        
                        # Combined attention score
                        attention_matrix[i, j] = similarity * structure_weight
        
        # Normalize attention matrix
        row_sums = np.sum(attention_matrix, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        attention_matrix = attention_matrix / row_sums
        
        # Apply attention to aggregate features
        aggregated_features = []
        for i, mod in enumerate(modalities):
            if mod in node_features:
                # Weighted combination of neighbor features
                neighbor_features = []
                neighbor_weights = []
                
                for j, neighbor_mod in enumerate(modalities):
                    if neighbor_mod in node_features and attention_matrix[i, j] > 0:
                        neighbor_features.append(node_features[neighbor_mod])
                        neighbor_weights.append(attention_matrix[i, j])
                
                if neighbor_features:
                    # Ensure same length
                    max_len = max(len(f) for f in neighbor_features)
                    padded_features = []
                    for features in neighbor_features:
                        if len(features) < max_len:
                            padded = np.pad(features, (0, max_len - len(features)))
                        else:
                            padded = features[:max_len]
                        padded_features.append(padded)
                    
                    weights = np.array(neighbor_weights)
                    weights = weights / np.sum(weights)
                    
                    aggregated = np.average(padded_features, axis=0, weights=weights)
                    aggregated_features.append(aggregated)
        
        if aggregated_features:
            final_features = np.mean(aggregated_features, axis=0)
        else:
            final_features = np.zeros(self.config['feature_dim'])
        
        return final_features.astype(np.float32), attention_matrix
    
    def _compute_feature_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Compute similarity between two feature vectors"""
        # Ensure same length
        min_len = min(len(feat1), len(feat2))
        f1 = feat1[:min_len]
        f2 = feat2[:min_len]
        
        # Cosine similarity
        dot_product = np.dot(f1, f2)
        norm1 = np.linalg.norm(f1)
        norm2 = np.linalg.norm(f2)
        
        if norm1 > 0 and norm2 > 0:
            similarity = dot_product / (norm1 * norm2)
        else:
            similarity = 0.0
        
        return float(np.clip(similarity, 0, 1))
    
    def _compute_all_modalities_self_attention(self, frame: MultiModalFrame) -> AttentionOutput:
        """Compute self-attention for all modalities"""
        all_results = []
        
        for modality in frame.modalities:
            result = self._self_attention(frame, modality, None)
            all_results.append(result)
        
        if not all_results:
            return self._create_empty_attention_output()
        
        # Combine results
        combined_features = np.mean([r.attended_features for r in all_results], axis=0)
        combined_weights = np.mean([r.attention_weights.weights for r in all_results], axis=0)
        avg_entropy = np.mean([r.entropy for r in all_results])
        
        attn_weights = AttentionWeights(
            weights=combined_weights,
            mechanism=AttentionMechanism.SELF_ATTENTION,
            source_modality=None,
            target_modality=None,
            confidence=np.mean([data.confidence for data in frame.modalities.values()])
        )
        
        return AttentionOutput(
            attended_features=combined_features,
            attention_weights=attn_weights,
            attention_distribution=combined_weights,
            entropy=avg_entropy,
            processing_time_ms=0,
            metadata={'attention_type': 'all_modalities_self_attention'}
        )
    
    def _create_empty_attention_output(self) -> AttentionOutput:
        """Create empty attention output for invalid cases"""
        empty_weights = AttentionWeights(
            weights=np.zeros((1, 1)),
            mechanism=AttentionMechanism.CROSS_MODAL,
            source_modality=None,
            target_modality=None,
            confidence=0.0
        )
        
        return AttentionOutput(
            attended_features=np.zeros(self.config['feature_dim']),
            attention_weights=empty_weights,
            attention_distribution=np.zeros(1),
            entropy=0.0,
            processing_time_ms=0.0,
            metadata={'status': 'empty'}
        )
    
    def _update_stats(self, mechanism: AttentionMechanism, 
                     processing_time: float, result: AttentionOutput):
        """Update attention statistics"""
        self.stats['attention_computations'] += 1
        self.stats['processing_times_ms'].append(processing_time)
        self.stats['attention_entropies'].append(result.entropy)
        self.stats['mechanism_usage'][mechanism] += 1
        
        # Add to history
        self.attention_history.append(result)
        
        # Maintain history length
        max_history = 100
        if len(self.attention_history) > max_history:
            self.attention_history.pop(0)
    
    def get_attention_stats(self) -> Dict[str, Any]:
        """Get attention processing statistics"""
        stats = self.stats.copy()
        
        if stats['processing_times_ms']:
            stats['avg_processing_time_ms'] = np.mean(stats['processing_times_ms'][-100:])
        
        if stats['attention_entropies']:
            stats['avg_entropy'] = np.mean(stats['attention_entropies'][-100:])
        
        # Mechanism usage rates
        total_computations = max(1, stats['attention_computations'])
        stats['mechanism_usage_rates'] = {
            mech.value: count / total_computations 
            for mech, count in stats['mechanism_usage'].items()
        }
        
        return stats


class MultiHeadAttentionProcessor:
    """Multi-head attention processor"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.num_heads = config.get('num_heads', 8)
    
    def compute_multi_head_attention(self, frame: MultiModalFrame, 
                                   source_modality: Optional[ModalityType],
                                   target_modality: Optional[ModalityType]) -> AttentionOutput:
        """Compute multi-head attention"""
        # Placeholder implementation
        # In practice, this would implement proper multi-head attention
        return AttentionOutput(
            attended_features=np.zeros(512),
            attention_weights=AttentionWeights(
                weights=np.zeros((1, 1)),
                mechanism=AttentionMechanism.MULTI_HEAD,
                source_modality=source_modality,
                target_modality=target_modality
            ),
            attention_distribution=np.zeros(1),
            entropy=0.0,
            processing_time_ms=0.0,
            metadata={'attention_type': 'multi_head'}
        )


class TemporalAttentionProcessor:
    """Temporal attention processor"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def compute_temporal_attention(self, frame: MultiModalFrame, 
                                 history: List[AttentionOutput],
                                 source_modality: Optional[ModalityType],
                                 target_modality: Optional[ModalityType]) -> AttentionOutput:
        """Compute temporal attention"""
        # Placeholder implementation
        return AttentionOutput(
            attended_features=np.zeros(512),
            attention_weights=AttentionWeights(
                weights=np.zeros((1, 1)),
                mechanism=AttentionMechanism.TEMPORAL,
                source_modality=source_modality,
                target_modality=target_modality
            ),
            attention_distribution=np.zeros(1),
            entropy=0.0,
            processing_time_ms=0.0,
            metadata={'attention_type': 'temporal'}
        )


class AdaptiveAttentionController:
    """Adaptive attention controller"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def compute_adaptive_attention(self, frame: MultiModalFrame,
                                 learned_patterns: Dict[str, Any],
                                 source_modality: Optional[ModalityType],
                                 target_modality: Optional[ModalityType]) -> AttentionOutput:
        """Compute adaptive attention"""
        # Placeholder implementation
        return AttentionOutput(
            attended_features=np.zeros(512),
            attention_weights=AttentionWeights(
                weights=np.zeros((1, 1)),
                mechanism=AttentionMechanism.ADAPTIVE,
                source_modality=source_modality,
                target_modality=target_modality
            ),
            attention_distribution=np.zeros(1),
            entropy=0.0,
            processing_time_ms=0.0,
            metadata={'attention_type': 'adaptive'}
        )