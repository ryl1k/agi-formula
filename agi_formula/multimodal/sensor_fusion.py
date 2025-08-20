"""
Sensor Fusion System for AGI-Formula Multi-Modal Processing

Advanced sensor fusion for integrating multiple sensory modalities:
- Multi-modal data alignment and synchronization
- Feature-level and decision-level fusion strategies
- Adaptive fusion weights based on modality confidence
- Temporal fusion with attention mechanisms
- Uncertainty quantification and reliability assessment
- Real-time fusion processing
"""

import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging

from .data_pipeline import ModalityType, MultiModalFrame, ModalityData


class FusionStrategy(Enum):
    """Types of fusion strategies"""
    EARLY_FUSION = "early"           # Feature-level fusion
    LATE_FUSION = "late"             # Decision-level fusion
    HYBRID_FUSION = "hybrid"         # Combined early and late fusion
    ATTENTION_FUSION = "attention"   # Attention-based fusion
    ADAPTIVE_FUSION = "adaptive"     # Adaptive weight fusion
    HIERARCHICAL_FUSION = "hierarchical"  # Multi-level fusion


class FusionLevel(Enum):
    """Levels of fusion processing"""
    FEATURE_LEVEL = "feature"
    REPRESENTATION_LEVEL = "representation"
    DECISION_LEVEL = "decision"
    SEMANTIC_LEVEL = "semantic"


@dataclass
class FusionWeights:
    """Weights for different modalities in fusion"""
    modality_weights: Dict[ModalityType, float]
    temporal_weights: np.ndarray
    confidence_threshold: float = 0.3
    normalization_method: str = "softmax"


@dataclass
class FusionResult:
    """Result of sensor fusion process"""
    fused_features: np.ndarray
    fusion_weights: FusionWeights
    confidence_score: float
    modality_contributions: Dict[ModalityType, float]
    fusion_strategy: FusionStrategy
    processing_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class SensorFusion:
    """
    Advanced sensor fusion system for multi-modal AGI
    
    Features:
    - Multiple fusion strategies (early, late, hybrid, attention-based)
    - Adaptive weight computation based on modality reliability
    - Temporal fusion with memory and attention
    - Uncertainty quantification and confidence estimation
    - Real-time processing with optimized algorithms
    - Extensible architecture for custom fusion methods
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        
        # Fusion strategies
        self.fusion_strategies = {
            FusionStrategy.EARLY_FUSION: self._early_fusion,
            FusionStrategy.LATE_FUSION: self._late_fusion,
            FusionStrategy.HYBRID_FUSION: self._hybrid_fusion,
            FusionStrategy.ATTENTION_FUSION: self._attention_fusion,
            FusionStrategy.ADAPTIVE_FUSION: self._adaptive_fusion,
            FusionStrategy.HIERARCHICAL_FUSION: self._hierarchical_fusion
        }
        
        # Fusion components
        self.attention_mechanism = AttentionFusion(self.config.get('attention', {}))
        self.uncertainty_estimator = UncertaintyEstimator()
        self.temporal_integrator = TemporalIntegrator(self.config.get('temporal', {}))
        
        # State management
        self.fusion_history = []
        self.modality_reliability = {modality: 1.0 for modality in ModalityType}
        self.adaptive_weights = {}
        
        # Performance monitoring
        self.stats = {
            'fusions_performed': 0,
            'processing_times_ms': [],
            'fusion_confidences': [],
            'modality_usage': {modality: 0 for modality in ModalityType}
        }
        
        # Initialize fusion system
        self._initialize_fusion()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for sensor fusion"""
        return {
            'default_strategy': FusionStrategy.ADAPTIVE_FUSION,
            'target_feature_dim': 512,
            'confidence_threshold': 0.3,
            'max_history_length': 50,
            'temporal_window_size': 5,
            'weight_adaptation': {
                'learning_rate': 0.01,
                'momentum': 0.9,
                'adaptation_interval': 10
            },
            'fusion_levels': [
                FusionLevel.FEATURE_LEVEL,
                FusionLevel.REPRESENTATION_LEVEL,
                FusionLevel.DECISION_LEVEL
            ],
            'modality_preferences': {
                ModalityType.VISION: 0.3,
                ModalityType.LANGUAGE: 0.25,
                ModalityType.AUDIO: 0.2,
                ModalityType.TACTILE: 0.15,
                ModalityType.PROPRIOCEPTION: 0.1
            },
            'quality_control': {
                'min_modalities': 1,
                'max_missing_ratio': 0.7,
                'confidence_weighting': True
            }
        }
    
    def _initialize_fusion(self):
        """Initialize the sensor fusion system"""
        # Initialize adaptive weights
        for modality in ModalityType:
            self.adaptive_weights[modality] = self.config['modality_preferences'][modality]
        
        print(f"Sensor fusion system initialized with {len(self.fusion_strategies)} strategies")
    
    def fuse_multimodal_frame(self, frame: MultiModalFrame, 
                             strategy: Optional[FusionStrategy] = None) -> FusionResult:
        """Fuse a multi-modal frame using specified strategy"""
        start_time = time.time()
        
        if strategy is None:
            strategy = self.config['default_strategy']
        
        # Validate frame
        if not self._validate_frame(frame):
            return self._create_empty_result(strategy, start_time)
        
        # Update modality reliability
        self._update_modality_reliability(frame)
        
        # Compute fusion weights
        fusion_weights = self._compute_fusion_weights(frame)
        
        # Apply fusion strategy
        if strategy in self.fusion_strategies:
            fused_features = self.fusion_strategies[strategy](frame, fusion_weights)
        else:
            logging.warning(f"Unknown fusion strategy: {strategy}, using default")
            fused_features = self.fusion_strategies[self.config['default_strategy']](frame, fusion_weights)
        
        # Compute confidence score
        confidence_score = self._compute_confidence_score(frame, fused_features)
        
        # Compute modality contributions
        modality_contributions = self._compute_modality_contributions(frame, fusion_weights)
        
        # Create fusion result
        processing_time = (time.time() - start_time) * 1000
        
        result = FusionResult(
            fused_features=fused_features,
            fusion_weights=fusion_weights,
            confidence_score=confidence_score,
            modality_contributions=modality_contributions,
            fusion_strategy=strategy,
            processing_time_ms=processing_time,
            metadata={
                'frame_id': frame.frame_id,
                'num_modalities': len(frame.modalities),
                'synchronization_quality': frame.synchronized
            }
        )
        
        # Update statistics and history
        self._update_stats(result)
        self._add_to_history(result)
        
        return result
    
    def _validate_frame(self, frame: MultiModalFrame) -> bool:
        """Validate multi-modal frame for fusion"""
        # Check minimum number of modalities
        if len(frame.modalities) < self.config['quality_control']['min_modalities']:
            return False
        
        # Check if too many modalities are missing
        total_modalities = len(ModalityType)
        missing_ratio = (total_modalities - len(frame.modalities)) / total_modalities
        if missing_ratio > self.config['quality_control']['max_missing_ratio']:
            return False
        
        # Check modality data quality
        for modality_data in frame.modalities.values():
            if modality_data.confidence < self.config['confidence_threshold']:
                continue  # Skip low-confidence modalities
            
            if modality_data.features is None:
                return False
        
        return True
    
    def _update_modality_reliability(self, frame: MultiModalFrame):
        """Update reliability estimates for each modality"""
        learning_rate = self.config['weight_adaptation']['learning_rate']
        
        for modality, modality_data in frame.modalities.items():
            # Update reliability based on confidence
            current_reliability = self.modality_reliability[modality]
            observed_reliability = modality_data.confidence
            
            # Exponential moving average
            updated_reliability = (
                (1 - learning_rate) * current_reliability + 
                learning_rate * observed_reliability
            )
            
            self.modality_reliability[modality] = updated_reliability
        
        # Decay reliability for missing modalities
        for modality in ModalityType:
            if modality not in frame.modalities:
                self.modality_reliability[modality] *= 0.95  # Gradual decay
    
    def _compute_fusion_weights(self, frame: MultiModalFrame) -> FusionWeights:
        """Compute fusion weights for the frame"""
        modality_weights = {}
        
        for modality in ModalityType:
            if modality in frame.modalities:
                modality_data = frame.modalities[modality]
                
                # Base weight from configuration
                base_weight = self.config['modality_preferences'][modality]
                
                # Reliability weight
                reliability_weight = self.modality_reliability[modality]
                
                # Confidence weight
                confidence_weight = modality_data.confidence
                
                # Adaptive weight
                adaptive_weight = self.adaptive_weights[modality]
                
                # Combined weight
                combined_weight = (
                    base_weight * 0.3 +
                    reliability_weight * 0.3 +
                    confidence_weight * 0.2 +
                    adaptive_weight * 0.2
                )
                
                modality_weights[modality] = combined_weight
            else:
                modality_weights[modality] = 0.0
        
        # Normalize weights
        total_weight = sum(modality_weights.values())
        if total_weight > 0:
            modality_weights = {m: w / total_weight for m, w in modality_weights.items()}
        
        # Temporal weights (for temporal fusion)
        temporal_weights = self._compute_temporal_weights(frame)
        
        return FusionWeights(
            modality_weights=modality_weights,
            temporal_weights=temporal_weights,
            confidence_threshold=self.config['confidence_threshold'],
            normalization_method="softmax"
        )
    
    def _compute_temporal_weights(self, frame: MultiModalFrame) -> np.ndarray:
        """Compute temporal weights for the fusion"""
        window_size = self.config['temporal_window_size']
        
        # Simple temporal weighting (more recent = higher weight)
        weights = np.exp(-0.1 * np.arange(window_size))
        weights = weights / np.sum(weights)
        
        return weights
    
    def _early_fusion(self, frame: MultiModalFrame, fusion_weights: FusionWeights) -> np.ndarray:
        """Early fusion: concatenate and combine features"""
        features_list = []
        weights_list = []
        
        for modality, modality_data in frame.modalities.items():
            if (modality_data.features is not None and 
                modality_data.confidence >= fusion_weights.confidence_threshold):
                
                features_list.append(modality_data.features)
                weights_list.append(fusion_weights.modality_weights[modality])
        
        if not features_list:
            return np.zeros(self.config['target_feature_dim'])
        
        # Normalize feature dimensions
        normalized_features = []
        for features in features_list:
            if len(features) > 0:
                # Normalize to unit length
                norm = np.linalg.norm(features)
                if norm > 0:
                    normalized_features.append(features / norm)
                else:
                    normalized_features.append(features)
            else:
                normalized_features.append(np.zeros(1))
        
        # Concatenate features
        concatenated = np.concatenate(normalized_features)
        
        # Project to target dimension
        target_dim = self.config['target_feature_dim']
        if len(concatenated) > target_dim:
            # Downsample
            indices = np.linspace(0, len(concatenated)-1, target_dim, dtype=int)
            fused_features = concatenated[indices]
        elif len(concatenated) < target_dim:
            # Pad
            fused_features = np.pad(concatenated, (0, target_dim - len(concatenated)))
        else:
            fused_features = concatenated
        
        return fused_features.astype(np.float32)
    
    def _late_fusion(self, frame: MultiModalFrame, fusion_weights: FusionWeights) -> np.ndarray:
        """Late fusion: weighted combination of modality decisions"""
        decisions = []
        weights = []
        
        for modality, modality_data in frame.modalities.items():
            if (modality_data.features is not None and 
                modality_data.confidence >= fusion_weights.confidence_threshold):
                
                # Convert features to decisions (simplified)
                decision = self._features_to_decision(modality_data.features)
                decisions.append(decision)
                weights.append(fusion_weights.modality_weights[modality])
        
        if not decisions:
            return np.zeros(self.config['target_feature_dim'])
        
        # Weighted combination of decisions
        weights = np.array(weights)
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        
        # Combine decisions
        combined_decision = np.average(decisions, axis=0, weights=weights)
        
        # Convert back to features
        fused_features = self._decision_to_features(combined_decision)
        
        return fused_features
    
    def _hybrid_fusion(self, frame: MultiModalFrame, fusion_weights: FusionWeights) -> np.ndarray:
        """Hybrid fusion: combination of early and late fusion"""
        # Perform early fusion
        early_features = self._early_fusion(frame, fusion_weights)
        
        # Perform late fusion
        late_features = self._late_fusion(frame, fusion_weights)
        
        # Combine early and late features
        hybrid_weight = 0.6  # Weight for early fusion
        fused_features = (
            hybrid_weight * early_features + 
            (1 - hybrid_weight) * late_features
        )
        
        return fused_features
    
    def _attention_fusion(self, frame: MultiModalFrame, fusion_weights: FusionWeights) -> np.ndarray:
        """Attention-based fusion using cross-modal attention"""
        return self.attention_mechanism.fuse_with_attention(frame, fusion_weights)
    
    def _adaptive_fusion(self, frame: MultiModalFrame, fusion_weights: FusionWeights) -> np.ndarray:
        """Adaptive fusion with dynamic weight adjustment"""
        # Start with early fusion
        base_features = self._early_fusion(frame, fusion_weights)
        
        # Adaptive refinement based on confidence and reliability
        for modality, modality_data in frame.modalities.items():
            if (modality_data.features is not None and 
                modality_data.confidence >= fusion_weights.confidence_threshold):
                
                # Compute adaptation factor
                reliability = self.modality_reliability[modality]
                confidence = modality_data.confidence
                adaptation_factor = reliability * confidence
                
                # Update adaptive weights
                learning_rate = self.config['weight_adaptation']['learning_rate']
                momentum = self.config['weight_adaptation']['momentum']
                
                current_weight = self.adaptive_weights[modality]
                new_weight = (
                    momentum * current_weight + 
                    learning_rate * adaptation_factor
                )
                self.adaptive_weights[modality] = new_weight
        
        # Recompute fusion with updated weights
        updated_weights = self._compute_fusion_weights(frame)
        adaptive_features = self._early_fusion(frame, updated_weights)
        
        return adaptive_features
    
    def _hierarchical_fusion(self, frame: MultiModalFrame, fusion_weights: FusionWeights) -> np.ndarray:
        """Hierarchical fusion with multiple levels"""
        # Level 1: Feature-level fusion within modality groups
        visual_features = self._fuse_visual_modalities(frame, fusion_weights)
        language_features = self._fuse_language_modalities(frame, fusion_weights)
        audio_features = self._fuse_audio_modalities(frame, fusion_weights)
        
        # Level 2: Cross-modal fusion
        cross_modal_features = self._fuse_cross_modal(
            visual_features, language_features, audio_features, fusion_weights
        )
        
        # Level 3: Final integration
        fused_features = self._final_integration(cross_modal_features)
        
        return fused_features
    
    def _fuse_visual_modalities(self, frame: MultiModalFrame, fusion_weights: FusionWeights) -> np.ndarray:
        """Fuse visual-related modalities"""
        visual_modalities = [ModalityType.VISION]
        return self._fuse_modality_group(frame, visual_modalities, fusion_weights)
    
    def _fuse_language_modalities(self, frame: MultiModalFrame, fusion_weights: FusionWeights) -> np.ndarray:
        """Fuse language-related modalities"""
        language_modalities = [ModalityType.LANGUAGE]
        return self._fuse_modality_group(frame, language_modalities, fusion_weights)
    
    def _fuse_audio_modalities(self, frame: MultiModalFrame, fusion_weights: FusionWeights) -> np.ndarray:
        """Fuse audio-related modalities"""
        audio_modalities = [ModalityType.AUDIO]
        return self._fuse_modality_group(frame, audio_modalities, fusion_weights)
    
    def _fuse_modality_group(self, frame: MultiModalFrame, 
                           modalities: List[ModalityType], 
                           fusion_weights: FusionWeights) -> np.ndarray:
        """Fuse a group of related modalities"""
        features_list = []
        weights_list = []
        
        for modality in modalities:
            if modality in frame.modalities:
                modality_data = frame.modalities[modality]
                if (modality_data.features is not None and 
                    modality_data.confidence >= fusion_weights.confidence_threshold):
                    
                    features_list.append(modality_data.features)
                    weights_list.append(fusion_weights.modality_weights[modality])
        
        if not features_list:
            return np.zeros(self.config['target_feature_dim'] // 3)
        
        # Weighted average of features
        weights = np.array(weights_list)
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
            
            # Ensure all features have same length for averaging
            max_len = max(len(f) for f in features_list)
            padded_features = []
            for features in features_list:
                if len(features) < max_len:
                    padded = np.pad(features, (0, max_len - len(features)))
                else:
                    padded = features[:max_len]
                padded_features.append(padded)
            
            fused = np.average(padded_features, axis=0, weights=weights)
        else:
            fused = np.concatenate(features_list)
        
        # Project to target dimension
        target_dim = self.config['target_feature_dim'] // 3
        if len(fused) > target_dim:
            indices = np.linspace(0, len(fused)-1, target_dim, dtype=int)
            fused = fused[indices]
        elif len(fused) < target_dim:
            fused = np.pad(fused, (0, target_dim - len(fused)))
        
        return fused.astype(np.float32)
    
    def _fuse_cross_modal(self, visual: np.ndarray, language: np.ndarray, 
                         audio: np.ndarray, fusion_weights: FusionWeights) -> np.ndarray:
        """Fuse across different modality types"""
        # Concatenate modality group features
        cross_modal = np.concatenate([visual, language, audio])
        
        # Apply cross-modal attention (simplified)
        attention_weights = self._compute_cross_modal_attention(visual, language, audio)
        
        # Weight the features
        weighted_features = cross_modal * attention_weights
        
        return weighted_features
    
    def _compute_cross_modal_attention(self, visual: np.ndarray, 
                                     language: np.ndarray, 
                                     audio: np.ndarray) -> np.ndarray:
        """Compute cross-modal attention weights"""
        # Simple attention based on feature magnitude
        all_features = np.concatenate([visual, language, audio])
        
        # Compute attention scores
        attention_scores = np.abs(all_features)
        
        # Apply softmax
        exp_scores = np.exp(attention_scores - np.max(attention_scores))
        attention_weights = exp_scores / np.sum(exp_scores)
        
        return attention_weights
    
    def _final_integration(self, cross_modal_features: np.ndarray) -> np.ndarray:
        """Final integration of cross-modal features"""
        # Project to target dimension
        target_dim = self.config['target_feature_dim']
        
        if len(cross_modal_features) > target_dim:
            # Use learned projection (simplified as linear projection)
            indices = np.linspace(0, len(cross_modal_features)-1, target_dim, dtype=int)
            integrated = cross_modal_features[indices]
        elif len(cross_modal_features) < target_dim:
            integrated = np.pad(cross_modal_features, (0, target_dim - len(cross_modal_features)))
        else:
            integrated = cross_modal_features
        
        return integrated.astype(np.float32)
    
    def _features_to_decision(self, features: np.ndarray) -> np.ndarray:
        """Convert features to decision representation"""
        # Simple decision conversion (e.g., through classification)
        # In practice, this would use task-specific models
        
        # Normalize features
        if np.linalg.norm(features) > 0:
            normalized = features / np.linalg.norm(features)
        else:
            normalized = features
        
        # Convert to decision scores (simplified)
        decision_dim = min(10, len(features))
        if len(normalized) >= decision_dim:
            indices = np.linspace(0, len(normalized)-1, decision_dim, dtype=int)
            decision = normalized[indices]
        else:
            decision = np.pad(normalized, (0, decision_dim - len(normalized)))
        
        # Apply sigmoid to get probabilities
        decision = 1 / (1 + np.exp(-decision))
        
        return decision
    
    def _decision_to_features(self, decision: np.ndarray) -> np.ndarray:
        """Convert decision representation back to features"""
        # Expand decision to feature space
        target_dim = self.config['target_feature_dim']
        
        if len(decision) < target_dim:
            # Upsample using interpolation
            indices = np.linspace(0, len(decision)-1, target_dim)
            features = np.interp(indices, np.arange(len(decision)), decision)
        else:
            features = decision[:target_dim]
        
        return features.astype(np.float32)
    
    def _compute_confidence_score(self, frame: MultiModalFrame, 
                                fused_features: np.ndarray) -> float:
        """Compute confidence score for the fusion result"""
        return self.uncertainty_estimator.estimate_confidence(frame, fused_features)
    
    def _compute_modality_contributions(self, frame: MultiModalFrame, 
                                      fusion_weights: FusionWeights) -> Dict[ModalityType, float]:
        """Compute contribution of each modality to the fusion"""
        contributions = {}
        
        total_weight = sum(fusion_weights.modality_weights.values())
        
        for modality in ModalityType:
            if modality in frame.modalities:
                # Normalized weight as contribution
                weight = fusion_weights.modality_weights[modality]
                contribution = weight / total_weight if total_weight > 0 else 0.0
                contributions[modality] = contribution
            else:
                contributions[modality] = 0.0
        
        return contributions
    
    def _create_empty_result(self, strategy: FusionStrategy, start_time: float) -> FusionResult:
        """Create empty fusion result for invalid frames"""
        processing_time = (time.time() - start_time) * 1000
        
        return FusionResult(
            fused_features=np.zeros(self.config['target_feature_dim']),
            fusion_weights=FusionWeights(
                modality_weights={m: 0.0 for m in ModalityType},
                temporal_weights=np.zeros(self.config['temporal_window_size'])
            ),
            confidence_score=0.0,
            modality_contributions={m: 0.0 for m in ModalityType},
            fusion_strategy=strategy,
            processing_time_ms=processing_time,
            metadata={'status': 'invalid_frame'}
        )
    
    def _update_stats(self, result: FusionResult):
        """Update fusion statistics"""
        self.stats['fusions_performed'] += 1
        self.stats['processing_times_ms'].append(result.processing_time_ms)
        self.stats['fusion_confidences'].append(result.confidence_score)
        
        # Update modality usage
        for modality, contribution in result.modality_contributions.items():
            if contribution > 0:
                self.stats['modality_usage'][modality] += 1
    
    def _add_to_history(self, result: FusionResult):
        """Add fusion result to history"""
        self.fusion_history.append(result)
        
        # Maintain history length
        max_length = self.config['max_history_length']
        if len(self.fusion_history) > max_length:
            self.fusion_history.pop(0)
    
    def get_fusion_stats(self) -> Dict[str, Any]:
        """Get fusion performance statistics"""
        stats = self.stats.copy()
        
        if stats['processing_times_ms']:
            stats['avg_processing_time_ms'] = np.mean(stats['processing_times_ms'][-100:])
            stats['max_processing_time_ms'] = np.max(stats['processing_times_ms'][-100:])
        
        if stats['fusion_confidences']:
            stats['avg_confidence'] = np.mean(stats['fusion_confidences'][-100:])
            stats['min_confidence'] = np.min(stats['fusion_confidences'][-100:])
        
        # Modality usage statistics
        total_fusions = max(1, stats['fusions_performed'])
        stats['modality_usage_rates'] = {
            modality.value: count / total_fusions 
            for modality, count in stats['modality_usage'].items()
        }
        
        # Current reliability estimates
        stats['modality_reliability'] = self.modality_reliability.copy()
        stats['adaptive_weights'] = self.adaptive_weights.copy()
        
        return stats


class AttentionFusion:
    """Cross-modal attention mechanism for fusion"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {
            'attention_heads': 4,
            'attention_dim': 64,
            'temperature': 1.0
        }
    
    def fuse_with_attention(self, frame: MultiModalFrame, 
                          fusion_weights: FusionWeights) -> np.ndarray:
        """Perform attention-based fusion"""
        features_dict = {}
        
        # Collect features from available modalities
        for modality, modality_data in frame.modalities.items():
            if (modality_data.features is not None and 
                modality_data.confidence >= fusion_weights.confidence_threshold):
                features_dict[modality] = modality_data.features
        
        if not features_dict:
            return np.zeros(512)  # Default feature dimension
        
        # Convert to query-key-value format
        queries, keys, values = self._prepare_qkv(features_dict)
        
        # Compute multi-head attention
        attended_features = self._multi_head_attention(queries, keys, values)
        
        return attended_features
    
    def _prepare_qkv(self, features_dict: Dict[ModalityType, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare query, key, value matrices for attention"""
        # For simplicity, use the same features for Q, K, V
        all_features = []
        
        for features in features_dict.values():
            # Normalize features
            if np.linalg.norm(features) > 0:
                normalized = features / np.linalg.norm(features)
            else:
                normalized = features
            all_features.append(normalized)
        
        # Pad features to same length
        if all_features:
            max_len = max(len(f) for f in all_features)
            padded_features = []
            
            for features in all_features:
                if len(features) < max_len:
                    padded = np.pad(features, (0, max_len - len(features)))
                else:
                    padded = features[:max_len]
                padded_features.append(padded)
            
            feature_matrix = np.array(padded_features)
            
            # Q, K, V are the same for self-attention
            return feature_matrix, feature_matrix, feature_matrix
        
        return np.array([]), np.array([]), np.array([])
    
    def _multi_head_attention(self, queries: np.ndarray, 
                            keys: np.ndarray, 
                            values: np.ndarray) -> np.ndarray:
        """Compute multi-head attention"""
        if len(queries) == 0:
            return np.zeros(512)
        
        num_heads = self.config['attention_heads']
        temp = self.config['temperature']
        
        # Simple single-head attention for demonstration
        # Compute attention scores
        scores = np.dot(queries, keys.T) / temp
        
        # Apply softmax
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        # Apply attention to values
        attended = np.dot(attention_weights, values)
        
        # Aggregate across modalities
        fused_features = np.mean(attended, axis=0)
        
        # Project to target dimension
        target_dim = 512
        if len(fused_features) > target_dim:
            indices = np.linspace(0, len(fused_features)-1, target_dim, dtype=int)
            fused_features = fused_features[indices]
        elif len(fused_features) < target_dim:
            fused_features = np.pad(fused_features, (0, target_dim - len(fused_features)))
        
        return fused_features.astype(np.float32)


class UncertaintyEstimator:
    """Estimate uncertainty and confidence in fusion results"""
    
    def __init__(self):
        pass
    
    def estimate_confidence(self, frame: MultiModalFrame, 
                          fused_features: np.ndarray) -> float:
        """Estimate confidence in fusion result"""
        confidence_factors = []
        
        # Factor 1: Number of available modalities
        modality_factor = len(frame.modalities) / len(ModalityType)
        confidence_factors.append(modality_factor)
        
        # Factor 2: Average modality confidence
        if frame.modalities:
            avg_modality_confidence = np.mean([
                data.confidence for data in frame.modalities.values()
            ])
            confidence_factors.append(avg_modality_confidence)
        
        # Factor 3: Feature consistency (variance measure)
        if len(fused_features) > 0:
            feature_variance = np.var(fused_features)
            consistency_factor = np.exp(-feature_variance)  # Lower variance = higher confidence
            confidence_factors.append(consistency_factor)
        
        # Factor 4: Synchronization quality
        sync_factor = 1.0 if frame.synchronized else 0.7
        confidence_factors.append(sync_factor)
        
        # Combine factors
        overall_confidence = np.mean(confidence_factors) if confidence_factors else 0.0
        
        return float(np.clip(overall_confidence, 0.0, 1.0))


class TemporalIntegrator:
    """Integrate information across temporal frames"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {
            'window_size': 5,
            'decay_factor': 0.9,
            'integration_method': 'weighted_average'
        }
        
        self.temporal_buffer = []
    
    def integrate_temporal(self, current_result: FusionResult) -> np.ndarray:
        """Integrate current result with temporal history"""
        # Add to temporal buffer
        self.temporal_buffer.append(current_result)
        
        # Maintain window size
        window_size = self.config['window_size']
        if len(self.temporal_buffer) > window_size:
            self.temporal_buffer.pop(0)
        
        # Temporal integration
        if self.config['integration_method'] == 'weighted_average':
            return self._weighted_average_integration()
        else:
            return current_result.fused_features
    
    def _weighted_average_integration(self) -> np.ndarray:
        """Weighted average temporal integration"""
        if not self.temporal_buffer:
            return np.zeros(512)
        
        weights = []
        features_list = []
        
        decay_factor = self.config['decay_factor']
        
        for i, result in enumerate(self.temporal_buffer):
            # More recent frames have higher weights
            age = len(self.temporal_buffer) - 1 - i
            weight = (decay_factor ** age) * result.confidence_score
            
            weights.append(weight)
            features_list.append(result.fused_features)
        
        # Weighted average
        weights = np.array(weights)
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
            integrated_features = np.average(features_list, axis=0, weights=weights)
        else:
            integrated_features = features_list[-1] if features_list else np.zeros(512)
        
        return integrated_features.astype(np.float32)