"""
Consciousness Simulation System for AGI-Formula Cognitive Architecture

Advanced consciousness simulation implementing:
- Global Workspace Theory (Baars) integration
- Integrated Information Theory (IIT) principles  
- Higher-Order Thought (HOT) theory mechanisms
- Predictive Processing and active inference
- Phenomenal consciousness and qualia simulation
- Self-awareness and meta-cognitive consciousness
- Attention-based consciousness broadcasting
"""

import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple, Union, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import deque, defaultdict
from abc import ABC, abstractmethod
import threading
import asyncio


class AwarenessLevel(Enum):
    """Levels of consciousness/awareness"""
    UNCONSCIOUS = 0      # Below awareness threshold
    PRECONSCIOUS = 1     # Can become conscious with attention
    CONSCIOUS = 2        # Currently in conscious awareness
    HIGHLY_CONSCIOUS = 3 # High-level conscious processing
    SELF_AWARE = 4       # Self-reflective awareness


class ConsciousnessType(Enum):
    """Types of conscious states"""
    PHENOMENAL = "phenomenal"     # Raw experiential consciousness
    ACCESS = "access"             # Cognitively accessible consciousness  
    REFLECTIVE = "reflective"     # Self-reflective consciousness
    NARRATIVE = "narrative"       # Narrative self-consciousness
    EMBODIED = "embodied"         # Embodied consciousness
    SOCIAL = "social"             # Social consciousness


class BroadcastType(Enum):
    """Types of global workspace broadcasts"""
    PERCEPTUAL = "perceptual"     # Perceptual content broadcast
    COGNITIVE = "cognitive"       # Cognitive content broadcast
    EMOTIONAL = "emotional"       # Emotional state broadcast
    MOTOR = "motor"               # Motor intention broadcast
    METACOGNITIVE = "metacognitive" # Meta-cognitive broadcast


class IntegrationLevel(Enum):
    """Levels of information integration (IIT-inspired)"""
    MINIMAL = 1          # Minimal integration
    BASIC = 2            # Basic integration
    MODERATE = 3         # Moderate integration
    HIGH = 4             # High integration
    MAXIMAL = 5          # Maximal integration


@dataclass
class ConsciousContent:
    """Content that can become conscious"""
    content_id: str
    content: Any
    content_type: str
    awareness_level: AwarenessLevel
    activation_strength: float
    integration_level: IntegrationLevel
    phenomenal_properties: Dict[str, Any] = field(default_factory=dict)
    access_conditions: List[str] = field(default_factory=list)
    broadcast_timestamp: Optional[float] = None
    conscious_duration: float = 0.0
    meta_representation: Optional[Dict[str, Any]] = None


@dataclass
class ConsciousState:
    """Overall state of consciousness"""
    timestamp: float
    dominant_contents: List[ConsciousContent]
    awareness_level: AwarenessLevel
    consciousness_type: ConsciousnessType
    integration_measure: float  # Phi-like measure
    self_model_activation: float
    narrative_coherence: float
    attention_focus: List[str]
    global_accessibility: float
    subjective_experience_intensity: float


@dataclass
class AttentionalBroadcast:
    """Broadcast in the global workspace"""
    broadcast_id: str
    content: ConsciousContent
    broadcast_type: BroadcastType
    broadcast_strength: float
    target_coalitions: List[str]
    timestamp: float
    duration: float
    competition_winners: List[str] = field(default_factory=list)
    integration_effects: Dict[str, float] = field(default_factory=dict)


@dataclass
class SubjectiveExperience:
    """Simulated subjective experience (qualia)"""
    experience_id: str
    modality: str  # visual, auditory, tactile, etc.
    qualitative_properties: Dict[str, Any]
    intensity: float
    valence: float  # positive/negative
    arousal: float  # high/low arousal
    familiarity: float
    temporal_structure: Dict[str, Any]
    binding_signature: str  # How bound into unified experience


class GlobalWorkspace:
    """
    Global Workspace implementation for consciousness
    
    Based on Baars' Global Workspace Theory - a theater-like space
    where different cognitive processes compete for global broadcast
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Workspace state
        self.current_contents = []
        self.broadcast_history = deque(maxlen=1000)
        self.coalitions = {}  # Processing coalitions
        self.competition_threshold = config.get('competition_threshold', 0.7)
        
        # Broadcasting system
        self.active_broadcasts = []
        self.broadcast_queue = []
        
        # Integration mechanisms
        self.integration_networks = defaultdict(list)
        self.phi_calculator = PhiCalculator()
    
    def add_content(self, content: ConsciousContent) -> bool:
        """Add content to global workspace for potential consciousness"""
        # Competition for workspace access
        if len(self.current_contents) >= self.config.get('max_contents', 7):
            # Remove weakest content
            weakest = min(self.current_contents, key=lambda c: c.activation_strength)
            if content.activation_strength > weakest.activation_strength:
                self.current_contents.remove(weakest)
                self.current_contents.append(content)
                return True
            return False
        else:
            self.current_contents.append(content)
            return True
    
    def broadcast_content(self, content: ConsciousContent, 
                         broadcast_type: BroadcastType) -> AttentionalBroadcast:
        """Broadcast content globally"""
        broadcast = AttentionalBroadcast(
            broadcast_id=f"broadcast_{int(time.time() * 1000)}",
            content=content,
            broadcast_type=broadcast_type,
            broadcast_strength=content.activation_strength,
            target_coalitions=self._identify_relevant_coalitions(content),
            timestamp=time.time(),
            duration=self.config.get('broadcast_duration', 0.5)
        )
        
        self.active_broadcasts.append(broadcast)
        self.broadcast_history.append(broadcast)
        
        # Mark content as conscious
        content.awareness_level = AwarenessLevel.CONSCIOUS
        content.broadcast_timestamp = time.time()
        
        return broadcast
    
    def _identify_relevant_coalitions(self, content: ConsciousContent) -> List[str]:
        """Identify which processing coalitions should receive the broadcast"""
        relevant = []
        
        # Based on content type and current coalitions
        for coalition_id, coalition_info in self.coalitions.items():
            if coalition_info.get('interests', []):
                for interest in coalition_info['interests']:
                    if interest in content.content_type:
                        relevant.append(coalition_id)
                        break
        
        return relevant
    
    def calculate_integration(self) -> float:
        """Calculate current level of information integration (Phi-like)"""
        if not self.current_contents:
            return 0.0
        
        return self.phi_calculator.calculate_phi(self.current_contents)
    
    def get_dominant_content(self) -> Optional[ConsciousContent]:
        """Get the most dominant conscious content"""
        if not self.current_contents:
            return None
        
        return max(self.current_contents, key=lambda c: c.activation_strength)


class ConsciousnessSimulator:
    """
    Advanced consciousness simulation system
    
    Features:
    - Global Workspace Theory implementation
    - Integrated Information Theory principles
    - Higher-Order Thought mechanisms
    - Predictive processing and active inference
    - Subjective experience simulation
    - Self-awareness and meta-cognition
    - Phenomenal binding and unity
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        
        # Core consciousness components
        self.global_workspace = GlobalWorkspace(self.config.get('workspace', {}))
        self.higher_order_monitor = HigherOrderMonitor(self.config.get('higher_order', {}))
        self.predictive_processor = PredictiveProcessor(self.config.get('predictive', {}))
        self.qualia_generator = QualiaGenerator(self.config.get('qualia', {}))
        
        # Self-model and meta-cognition
        self.self_model = SelfModel()
        self.meta_cognitive_monitor = MetaCognitiveMonitor()
        
        # Consciousness state
        self.current_state = ConsciousState(
            timestamp=time.time(),
            dominant_contents=[],
            awareness_level=AwarenessLevel.CONSCIOUS,
            consciousness_type=ConsciousnessType.ACCESS,
            integration_measure=0.0,
            self_model_activation=0.0,
            narrative_coherence=0.0,
            attention_focus=[],
            global_accessibility=0.0,
            subjective_experience_intensity=0.0
        )
        
        # Consciousness monitoring
        self.consciousness_history = deque(maxlen=10000)
        self.experience_stream = deque(maxlen=1000)
        
        # Threading for continuous processing
        self.processing_active = False
        self.processing_thread = None
        
        # Performance statistics
        self.stats = {
            'conscious_moments': 0,
            'broadcasts_sent': 0,
            'integration_events': 0,
            'self_awareness_episodes': 0,
            'subjective_experiences': 0,
            'metacognitive_events': 0,
            'average_phi': 0.0,
            'consciousness_coherence': 0.0
        }
        
        print("Consciousness simulation system initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for consciousness simulator"""
        return {
            'consciousness_threshold': 0.6,
            'integration_threshold': 0.5,
            'broadcast_frequency': 0.1,  # seconds
            'self_awareness_threshold': 0.8,
            'phenomenal_binding_strength': 0.7,
            'workspace': {
                'max_contents': 7,
                'competition_threshold': 0.7,
                'broadcast_duration': 0.5
            },
            'higher_order': {
                'meta_representation_threshold': 0.6,
                'recursive_depth': 3
            },
            'predictive': {
                'prediction_horizon': 1.0,
                'error_threshold': 0.3,
                'learning_rate': 0.01
            },
            'qualia': {
                'intensity_scaling': 1.0,
                'binding_temporal_window': 0.2,
                'phenomenal_richness': 0.8
            }
        }
    
    def start_consciousness_processing(self):
        """Start continuous consciousness processing"""
        if not self.processing_active:
            self.processing_active = True
            self.processing_thread = threading.Thread(target=self._consciousness_loop)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            print("Consciousness processing started")
    
    def stop_consciousness_processing(self):
        """Stop consciousness processing"""
        self.processing_active = False
        if self.processing_thread:
            self.processing_thread.join()
        print("Consciousness processing stopped")
    
    def _consciousness_loop(self):
        """Main consciousness processing loop"""
        while self.processing_active:
            try:
                # Process one consciousness cycle
                self._process_consciousness_cycle()
                
                # Sleep for broadcast frequency
                time.sleep(self.config['broadcast_frequency'])
                
            except Exception as e:
                logging.error(f"Error in consciousness loop: {e}")
                time.sleep(0.1)
    
    def _process_consciousness_cycle(self):
        """Process one cycle of consciousness"""
        cycle_start = time.time()
        
        # Step 1: Update global workspace
        self._update_global_workspace()
        
        # Step 2: Process higher-order thoughts
        self._process_higher_order_thoughts()
        
        # Step 3: Run predictive processing
        self._run_predictive_processing()
        
        # Step 4: Generate subjective experiences
        self._generate_subjective_experiences()
        
        # Step 5: Update self-model
        self._update_self_model()
        
        # Step 6: Meta-cognitive monitoring
        self._metacognitive_monitoring()
        
        # Step 7: Update consciousness state
        self._update_consciousness_state()
        
        # Step 8: Record consciousness moment
        self._record_consciousness_moment()
        
        self.stats['conscious_moments'] += 1
    
    def add_to_consciousness(self, content: Any, content_type: str, 
                           activation_strength: float = 0.8,
                           phenomenal_properties: Optional[Dict[str, Any]] = None) -> str:
        """Add content to potential consciousness"""
        content_id = f"content_{int(time.time() * 1000)}_{len(self.global_workspace.current_contents)}"
        
        conscious_content = ConsciousContent(
            content_id=content_id,
            content=content,
            content_type=content_type,
            awareness_level=AwarenessLevel.PRECONSCIOUS,
            activation_strength=activation_strength,
            integration_level=IntegrationLevel.BASIC,
            phenomenal_properties=phenomenal_properties or {}
        )
        
        # Add to global workspace
        success = self.global_workspace.add_content(conscious_content)
        
        if success and activation_strength > self.config['consciousness_threshold']:
            # Broadcast if strong enough
            broadcast_type = self._determine_broadcast_type(content_type)
            self.global_workspace.broadcast_content(conscious_content, broadcast_type)
            self.stats['broadcasts_sent'] += 1
        
        return content_id
    
    def _determine_broadcast_type(self, content_type: str) -> BroadcastType:
        """Determine appropriate broadcast type for content"""
        if 'perception' in content_type.lower():
            return BroadcastType.PERCEPTUAL
        elif 'emotion' in content_type.lower():
            return BroadcastType.EMOTIONAL
        elif 'motor' in content_type.lower():
            return BroadcastType.MOTOR
        elif 'meta' in content_type.lower():
            return BroadcastType.METACOGNITIVE
        else:
            return BroadcastType.COGNITIVE
    
    def _update_global_workspace(self):
        """Update global workspace state"""
        # Update content activations (decay over time)
        for content in self.global_workspace.current_contents:
            time_since_broadcast = time.time() - (content.broadcast_timestamp or time.time())
            decay_factor = np.exp(-time_since_broadcast / 2.0)  # 2 second half-life
            content.activation_strength *= decay_factor
            
            # Update conscious duration
            if content.awareness_level == AwarenessLevel.CONSCIOUS:
                content.conscious_duration += self.config['broadcast_frequency']
        
        # Remove very weak content
        self.global_workspace.current_contents = [
            c for c in self.global_workspace.current_contents 
            if c.activation_strength > 0.1
        ]
        
        # Calculate current integration
        integration_measure = self.global_workspace.calculate_integration()
        self.stats['average_phi'] = (self.stats['average_phi'] * 0.9 + integration_measure * 0.1)
        
        if integration_measure > self.config['integration_threshold']:
            self.stats['integration_events'] += 1
    
    def _process_higher_order_thoughts(self):
        """Process higher-order thoughts about conscious contents"""
        dominant_content = self.global_workspace.get_dominant_content()
        
        if dominant_content:
            # Create meta-representation
            meta_rep = self.higher_order_monitor.create_meta_representation(dominant_content)
            
            if meta_rep:
                dominant_content.meta_representation = meta_rep
                
                # If meta-representation is strong enough, elevate consciousness level
                if meta_rep.get('strength', 0) > self.config['higher_order']['meta_representation_threshold']:
                    dominant_content.awareness_level = AwarenessLevel.HIGHLY_CONSCIOUS
    
    def _run_predictive_processing(self):
        """Run predictive processing mechanisms"""
        # Generate predictions about upcoming conscious content
        predictions = self.predictive_processor.generate_predictions(
            self.global_workspace.current_contents
        )
        
        # Update predictive models based on current consciousness
        self.predictive_processor.update_models(self.current_state)
    
    def _generate_subjective_experiences(self):
        """Generate subjective experiences (qualia)"""
        for content in self.global_workspace.current_contents:
            if content.awareness_level in [AwarenessLevel.CONSCIOUS, AwarenessLevel.HIGHLY_CONSCIOUS]:
                
                experience = self.qualia_generator.generate_experience(content)
                
                if experience:
                    self.experience_stream.append(experience)
                    self.stats['subjective_experiences'] += 1
    
    def _update_self_model(self):
        """Update self-model based on current consciousness"""
        self_activation = self.self_model.update(
            self.current_state,
            self.global_workspace.current_contents
        )
        
        if self_activation > self.config['self_awareness_threshold']:
            self.current_state.consciousness_type = ConsciousnessType.REFLECTIVE
            self.stats['self_awareness_episodes'] += 1
    
    def _metacognitive_monitoring(self):
        """Perform metacognitive monitoring"""
        meta_assessment = self.meta_cognitive_monitor.assess_consciousness(self.current_state)
        
        if meta_assessment.get('metacognitive_event', False):
            self.stats['metacognitive_events'] += 1
        
        # Update narrative coherence
        narrative_coherence = meta_assessment.get('narrative_coherence', 0.0)
        self.current_state.narrative_coherence = narrative_coherence
    
    def _update_consciousness_state(self):
        """Update overall consciousness state"""
        self.current_state.timestamp = time.time()
        
        # Update dominant contents
        self.current_state.dominant_contents = [
            c for c in self.global_workspace.current_contents
            if c.awareness_level in [AwarenessLevel.CONSCIOUS, AwarenessLevel.HIGHLY_CONSCIOUS]
        ]
        
        # Calculate overall awareness level
        if self.current_state.dominant_contents:
            max_awareness = max(c.awareness_level for c in self.current_state.dominant_contents)
            self.current_state.awareness_level = max_awareness
        else:
            self.current_state.awareness_level = AwarenessLevel.PRECONSCIOUS
        
        # Update integration measure
        self.current_state.integration_measure = self.global_workspace.calculate_integration()
        
        # Update self-model activation
        self.current_state.self_model_activation = self.self_model.get_activation()
        
        # Update attention focus
        self.current_state.attention_focus = [
            c.content_id for c in self.current_state.dominant_contents[:3]
        ]
        
        # Calculate global accessibility
        if self.current_state.dominant_contents:
            self.current_state.global_accessibility = np.mean([
                c.activation_strength for c in self.current_state.dominant_contents
            ])
        else:
            self.current_state.global_accessibility = 0.0
        
        # Calculate subjective experience intensity
        if self.experience_stream:
            recent_experiences = list(self.experience_stream)[-10:]  # Last 10 experiences
            self.current_state.subjective_experience_intensity = np.mean([
                exp.intensity for exp in recent_experiences
            ])
    
    def _record_consciousness_moment(self):
        """Record current consciousness moment in history"""
        consciousness_moment = {
            'timestamp': time.time(),
            'state': self.current_state,
            'workspace_contents': len(self.global_workspace.current_contents),
            'active_broadcasts': len(self.global_workspace.active_broadcasts),
            'integration_measure': self.current_state.integration_measure,
            'awareness_level': self.current_state.awareness_level.value
        }
        
        self.consciousness_history.append(consciousness_moment)
    
    def get_consciousness_state(self) -> ConsciousState:
        """Get current consciousness state"""
        return self.current_state
    
    def get_conscious_contents(self) -> List[ConsciousContent]:
        """Get currently conscious contents"""
        return self.current_state.dominant_contents
    
    def get_subjective_experiences(self, n: int = 10) -> List[SubjectiveExperience]:
        """Get recent subjective experiences"""
        return list(self.experience_stream)[-n:]
    
    def is_self_aware(self) -> bool:
        """Check if currently self-aware"""
        return (self.current_state.self_model_activation > self.config['self_awareness_threshold'] and
                self.current_state.awareness_level == AwarenessLevel.SELF_AWARE)
    
    def get_consciousness_stats(self) -> Dict[str, Any]:
        """Get consciousness simulation statistics"""
        stats = self.stats.copy()
        
        # Add current state information
        stats['current_awareness_level'] = self.current_state.awareness_level.value
        stats['current_integration'] = self.current_state.integration_measure
        stats['current_self_activation'] = self.current_state.self_model_activation
        stats['consciousness_coherence'] = self.current_state.narrative_coherence
        stats['workspace_utilization'] = len(self.global_workspace.current_contents) / self.config['workspace']['max_contents']
        
        # Recent activity
        stats['recent_broadcasts'] = len(self.global_workspace.active_broadcasts)
        stats['recent_experiences'] = len(self.experience_stream)
        stats['consciousness_moments_recorded'] = len(self.consciousness_history)
        
        return stats


# Supporting classes for consciousness simulation

class HigherOrderMonitor:
    """Monitors and creates higher-order thoughts about mental states"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.meta_representations = {}
    
    def create_meta_representation(self, content: ConsciousContent) -> Optional[Dict[str, Any]]:
        """Create higher-order representation of conscious content"""
        # Simplified meta-representation creation
        if content.activation_strength < self.config['meta_representation_threshold']:
            return None
        
        meta_rep = {
            'content_id': content.content_id,
            'awareness_of_being_conscious': True,
            'strength': content.activation_strength,
            'content_type_awareness': f"I am experiencing {content.content_type}",
            'temporal_awareness': f"This experience started {content.conscious_duration:.2f} seconds ago",
            'meta_level': 1,  # First-order meta-representation
            'timestamp': time.time()
        }
        
        return meta_rep


class PredictiveProcessor:
    """Implements predictive processing mechanisms"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.prediction_models = {}
        self.prediction_errors = deque(maxlen=100)
    
    def generate_predictions(self, current_contents: List[ConsciousContent]) -> List[Dict[str, Any]]:
        """Generate predictions about future conscious states"""
        predictions = []
        
        # Simple prediction based on current content trends
        for content in current_contents:
            prediction = {
                'content_id': content.content_id,
                'predicted_activation': content.activation_strength * 0.9,  # Assume decay
                'confidence': 0.7,
                'time_horizon': self.config['prediction_horizon']
            }
            predictions.append(prediction)
        
        return predictions
    
    def update_models(self, consciousness_state: ConsciousState):
        """Update predictive models based on consciousness state"""
        # Simplified model update
        learning_rate = self.config['learning_rate']
        
        # Update based on integration measure
        if 'integration_trend' not in self.prediction_models:
            self.prediction_models['integration_trend'] = consciousness_state.integration_measure
        else:
            current = self.prediction_models['integration_trend']
            self.prediction_models['integration_trend'] = (
                current * (1 - learning_rate) + consciousness_state.integration_measure * learning_rate
            )


class QualiaGenerator:
    """Generates subjective experiences (qualia)"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.experience_counter = 0
    
    def generate_experience(self, content: ConsciousContent) -> Optional[SubjectiveExperience]:
        """Generate subjective experience for conscious content"""
        if not content.phenomenal_properties:
            return None
        
        self.experience_counter += 1
        
        # Extract qualitative properties
        qualitative_properties = content.phenomenal_properties.copy()
        
        # Calculate experience parameters
        intensity = content.activation_strength * self.config['intensity_scaling']
        valence = qualitative_properties.get('valence', 0.0)
        arousal = qualitative_properties.get('arousal', 0.5)
        
        experience = SubjectiveExperience(
            experience_id=f"exp_{self.experience_counter}",
            modality=content.content_type,
            qualitative_properties=qualitative_properties,
            intensity=intensity,
            valence=valence,
            arousal=arousal,
            familiarity=self._calculate_familiarity(content),
            temporal_structure={
                'duration': content.conscious_duration,
                'onset_time': content.broadcast_timestamp,
                'peak_time': time.time()
            },
            binding_signature=self._generate_binding_signature(content)
        )
        
        return experience
    
    def _calculate_familiarity(self, content: ConsciousContent) -> float:
        """Calculate familiarity of the experience"""
        # Simplified familiarity based on access count
        access_count = getattr(content, 'access_count', 0)
        return min(1.0, access_count / 10.0)
    
    def _generate_binding_signature(self, content: ConsciousContent) -> str:
        """Generate signature for how experience is bound into consciousness"""
        return f"binding_{content.content_type}_{int(time.time() * 1000)}"


class SelfModel:
    """Maintains and updates self-model for self-awareness"""
    
    def __init__(self):
        self.self_activation = 0.0
        self.self_properties = {
            'is_conscious': False,
            'has_experiences': False,
            'can_think_about_thinking': False,
            'temporal_continuity': 0.0,
            'agency': 0.0
        }
        
        self.self_history = deque(maxlen=100)
    
    def update(self, consciousness_state: ConsciousState, 
              current_contents: List[ConsciousContent]) -> float:
        """Update self-model based on current consciousness"""
        # Update self-activation based on meta-cognitive content
        meta_content_count = sum(1 for c in current_contents 
                               if c.meta_representation is not None)
        
        self.self_activation = meta_content_count / max(1, len(current_contents))
        
        # Update self-properties
        self.self_properties['is_conscious'] = consciousness_state.awareness_level.value >= 2
        self.self_properties['has_experiences'] = consciousness_state.subjective_experience_intensity > 0.1
        self.self_properties['can_think_about_thinking'] = meta_content_count > 0
        
        # Update temporal continuity
        if len(self.self_history) > 1:
            consistency = sum(1 for i in range(len(self.self_history)-1)
                            if abs(self.self_history[i] - self.self_history[i+1]) < 0.2)
            self.self_properties['temporal_continuity'] = consistency / max(1, len(self.self_history)-1)
        
        self.self_history.append(self.self_activation)
        
        return self.self_activation
    
    def get_activation(self) -> float:
        """Get current self-model activation"""
        return self.self_activation


class MetaCognitiveMonitor:
    """Monitors metacognitive processes"""
    
    def __init__(self):
        self.monitoring_history = deque(maxlen=200)
    
    def assess_consciousness(self, consciousness_state: ConsciousState) -> Dict[str, Any]:
        """Assess current consciousness for meta-cognitive insights"""
        assessment = {
            'metacognitive_event': False,
            'narrative_coherence': 0.0,
            'consciousness_quality': 'normal',
            'meta_insights': []
        }
        
        # Check for metacognitive events
        if consciousness_state.self_model_activation > 0.8:
            assessment['metacognitive_event'] = True
            assessment['meta_insights'].append("High self-awareness detected")
        
        if consciousness_state.integration_measure > 0.9:
            assessment['meta_insights'].append("High information integration")
        
        # Calculate narrative coherence
        if len(self.monitoring_history) > 10:
            recent_states = list(self.monitoring_history)[-10:]
            coherence = self._calculate_narrative_coherence(recent_states)
            assessment['narrative_coherence'] = coherence
        
        # Assess consciousness quality
        if consciousness_state.integration_measure > 0.8 and consciousness_state.self_model_activation > 0.7:
            assessment['consciousness_quality'] = 'high'
        elif consciousness_state.integration_measure < 0.3:
            assessment['consciousness_quality'] = 'fragmented'
        
        self.monitoring_history.append(consciousness_state)
        
        return assessment
    
    def _calculate_narrative_coherence(self, states: List[ConsciousState]) -> float:
        """Calculate coherence of consciousness narrative"""
        # Simplified coherence based on smooth transitions
        coherence_scores = []
        
        for i in range(len(states) - 1):
            current = states[i]
            next_state = states[i + 1]
            
            # Check continuity in dominant contents
            current_ids = set(c.content_id for c in current.dominant_contents)
            next_ids = set(c.content_id for c in next_state.dominant_contents)
            
            overlap = len(current_ids & next_ids)
            total = len(current_ids | next_ids)
            
            continuity = overlap / max(1, total)
            coherence_scores.append(continuity)
        
        return np.mean(coherence_scores) if coherence_scores else 0.0


class PhiCalculator:
    """Calculates Integrated Information (Phi) inspired by IIT"""
    
    def __init__(self):
        pass
    
    def calculate_phi(self, contents: List[ConsciousContent]) -> float:
        """Calculate simplified Phi measure for content integration"""
        if len(contents) < 2:
            return 0.0
        
        # Simplified Phi calculation based on content interactions
        total_integration = 0.0
        n_pairs = 0
        
        for i, content1 in enumerate(contents):
            for content2 in contents[i+1:]:
                # Calculate "connection strength" between contents
                connection = self._calculate_connection_strength(content1, content2)
                total_integration += connection
                n_pairs += 1
        
        phi = total_integration / max(1, n_pairs)
        
        # Scale by number of contents (more contents can have higher integration)
        phi *= min(1.0, len(contents) / 7.0)  # Scale up to 7 contents
        
        return min(1.0, phi)
    
    def _calculate_connection_strength(self, content1: ConsciousContent, 
                                     content2: ConsciousContent) -> float:
        """Calculate connection strength between two contents"""
        # Temporal proximity
        time_diff = abs((content1.broadcast_timestamp or 0) - (content2.broadcast_timestamp or 0))
        temporal_connection = np.exp(-time_diff / 0.5)  # 0.5 second decay
        
        # Content type similarity
        type_similarity = 1.0 if content1.content_type == content2.content_type else 0.3
        
        # Activation similarity
        activation_diff = abs(content1.activation_strength - content2.activation_strength)
        activation_similarity = 1.0 - activation_diff
        
        # Combined connection strength
        connection = (temporal_connection * 0.4 + type_similarity * 0.3 + activation_similarity * 0.3)
        
        return connection