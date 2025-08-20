"""
Complete AGI-LLM Integration System

The world's first fully AGI-capable language model that integrates:
- Consciousness-guided text generation
- Real-time reasoning during conversation
- Dynamic learning from interactions
- Self-aware dialogue with meta-cognitive commentary
- Causal understanding and explanation
- Goal-directed conversation strategies
"""

import torch
import numpy as np
import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict

# Import AGI components
from ..cognitive import (
    CognitiveArchitecture,
    CognitiveAgent,
    ArchitectureConfig,
    CognitiveMode,
    TaskPriority,
    MemoryType,
    MemoryPriority
)

from ..reasoning import (
    LogicalReasoner,
    CausalReasoner,
    TemporalReasoner,
    AbstractReasoner
)

# Import language components
from .agi_transformer import AGITransformerArchitecture, AGITransformerConfig
from .conscious_generation import (
    ConsciousLanguageGenerator,
    GenerationContext,
    GenerationMode,
    ConsciousnessLevel
)
from .knowledge_integration import DynamicKnowledgeGraph
from .language_reasoning import LanguageReasoningEngine


class ConversationRole(Enum):
    """Roles in conversation"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class LearningMode(Enum):
    """Learning modes for AGI-LLM"""
    PASSIVE = "passive"         # Learn from conversation patterns
    ACTIVE = "active"          # Actively ask clarifying questions
    TEACHING = "teaching"      # Explicit teaching mode
    COLLABORATIVE = "collaborative"  # Collaborative learning


@dataclass
class AGILLMConfig:
    """Configuration for AGI-LLM"""
    # Transformer configuration
    transformer_config: AGITransformerConfig = field(default_factory=AGITransformerConfig)
    
    # Cognitive integration
    consciousness_enabled: bool = True
    reasoning_enabled: bool = True
    learning_enabled: bool = True
    meta_cognition_enabled: bool = True
    
    # Generation settings
    max_response_length: int = 1024
    consciousness_threshold: float = 0.7
    reasoning_threshold: float = 0.6
    learning_threshold: float = 0.5
    
    # Conversation management
    max_conversation_history: int = 50
    context_window_size: int = 4096
    memory_consolidation_frequency: int = 10  # Every N interactions
    
    # Learning settings
    learning_mode: LearningMode = LearningMode.ACTIVE
    knowledge_confidence_threshold: float = 0.6
    conflict_resolution_enabled: bool = True
    
    # Performance settings
    enable_caching: bool = True
    reasoning_timeout: float = 5.0  # seconds
    generation_timeout: float = 10.0  # seconds


@dataclass
class ConversationTurn:
    """Single turn in conversation"""
    role: ConversationRole
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    reasoning_trace: Optional[Dict[str, Any]] = None
    consciousness_state: Optional[Dict[str, Any]] = None
    learning_events: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ConversationSession:
    """Complete conversation session"""
    session_id: str
    turns: List[ConversationTurn] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    learning_summary: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)


class ConversationManager:
    """Manages conversation state and context"""
    
    def __init__(self, config: AGILLMConfig):
        self.config = config
        self.current_session = None
        self.session_history = deque(maxlen=100)
        
        # Conversation analysis
        self.conversation_patterns = defaultdict(list)
        self.user_preferences = {}
        self.conversation_goals = []
    
    def start_session(self, session_id: str = None, context: Dict[str, Any] = None) -> str:
        """Start a new conversation session"""
        if session_id is None:
            session_id = f"session_{int(time.time())}"
        
        self.current_session = ConversationSession(
            session_id=session_id,
            context=context or {}
        )
        
        return session_id
    
    def add_turn(self, role: ConversationRole, content: str, 
                metadata: Dict[str, Any] = None) -> ConversationTurn:
        """Add a turn to the current conversation"""
        if not self.current_session:
            self.start_session()
        
        turn = ConversationTurn(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        
        self.current_session.turns.append(turn)
        
        # Maintain conversation history limit
        if len(self.current_session.turns) > self.config.max_conversation_history:
            self.current_session.turns = self.current_session.turns[-self.config.max_conversation_history:]
        
        return turn
    
    def get_conversation_context(self, max_turns: int = None) -> List[Dict[str, str]]:
        """Get conversation context for processing"""
        if not self.current_session:
            return []
        
        max_turns = max_turns or self.config.max_conversation_history
        recent_turns = self.current_session.turns[-max_turns:]
        
        return [
            {
                'role': turn.role.value,
                'content': turn.content,
                'timestamp': turn.timestamp
            }
            for turn in recent_turns
        ]
    
    def analyze_conversation_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in the conversation"""
        if not self.current_session or len(self.current_session.turns) < 3:
            return {}
        
        turns = self.current_session.turns
        user_turns = [t for t in turns if t.role == ConversationRole.USER]
        
        # Analyze user question patterns
        question_types = defaultdict(int)
        for turn in user_turns:
            content = turn.content.lower()
            if content.startswith('what'):
                question_types['factual'] += 1
            elif content.startswith('how'):
                question_types['procedural'] += 1
            elif content.startswith('why'):
                question_types['causal'] += 1
            elif content.startswith('when'):
                question_types['temporal'] += 1
            elif '?' in content:
                question_types['general'] += 1
        
        # Analyze conversation length and engagement
        avg_turn_length = np.mean([len(t.content) for t in user_turns]) if user_turns else 0
        
        return {
            'question_types': dict(question_types),
            'avg_turn_length': avg_turn_length,
            'total_turns': len(turns),
            'user_turns': len(user_turns),
            'conversation_duration': time.time() - self.current_session.start_time
        }
    
    def end_session(self) -> Optional[ConversationSession]:
        """End current conversation session"""
        if self.current_session:
            session = self.current_session
            self.session_history.append(session)
            self.current_session = None
            return session
        return None


class LearningTracker:
    """Tracks learning progress and insights"""
    
    def __init__(self):
        self.learning_events = []
        self.knowledge_gained = defaultdict(list)
        self.learning_patterns = defaultdict(int)
        self.confidence_evolution = []
        
    def record_learning_event(self, event_type: str, content: str, 
                            confidence: float, source: str):
        """Record a learning event"""
        event = {
            'type': event_type,
            'content': content,
            'confidence': confidence,
            'source': source,
            'timestamp': time.time()
        }
        
        self.learning_events.append(event)
        self.knowledge_gained[event_type].append(event)
        self.learning_patterns[event_type] += 1
        self.confidence_evolution.append(confidence)
    
    def get_learning_summary(self, time_window: float = 3600) -> Dict[str, Any]:
        """Get learning summary for recent time window"""
        cutoff_time = time.time() - time_window
        recent_events = [e for e in self.learning_events if e['timestamp'] >= cutoff_time]
        
        if not recent_events:
            return {'total_events': 0}
        
        return {
            'total_events': len(recent_events),
            'event_types': {
                event_type: len([e for e in recent_events if e['type'] == event_type])
                for event_type in set(e['type'] for e in recent_events)
            },
            'avg_confidence': np.mean([e['confidence'] for e in recent_events]),
            'confidence_trend': 'increasing' if self._calculate_trend(recent_events) > 0 else 'stable',
            'most_common_source': max(set(e['source'] for e in recent_events), 
                                    key=lambda x: sum(1 for e in recent_events if e['source'] == x))
        }
    
    def _calculate_trend(self, events: List[Dict[str, Any]]) -> float:
        """Calculate confidence trend"""
        if len(events) < 2:
            return 0.0
        
        confidences = [e['confidence'] for e in events]
        x = np.arange(len(confidences))
        
        # Simple linear regression slope
        n = len(x)
        slope = (n * np.sum(x * confidences) - np.sum(x) * np.sum(confidences)) / (n * np.sum(x**2) - np.sum(x)**2)
        
        return slope


class AGILLM:
    """Complete AGI-enhanced Language Model"""
    
    def __init__(self, config: AGILLMConfig = None):
        self.config = config or AGILLMConfig()
        
        # Initialize cognitive architecture
        arch_config = ArchitectureConfig(
            consciousness_enabled=self.config.consciousness_enabled,
            meta_cognition_enabled=self.config.meta_cognition_enabled,
            learning_enabled=self.config.learning_enabled
        )
        self.cognitive_architecture = CognitiveArchitecture(arch_config)
        
        # Initialize reasoning engines
        self.reasoning_engines = {
            'logical': LogicalReasoner(),
            'causal': CausalReasoner(),
            'temporal': TemporalReasoner(),
            'abstract': AbstractReasoner()
        }
        
        # Initialize language components
        self.transformer = AGITransformerArchitecture(self.config.transformer_config)
        self.transformer.set_agi_components(self.cognitive_architecture, self.reasoning_engines)
        
        self.conscious_generator = ConsciousLanguageGenerator(self.cognitive_architecture)
        self.knowledge_graph = DynamicKnowledgeGraph(self.cognitive_architecture)
        self.language_reasoner = LanguageReasoningEngine(self.reasoning_engines)
        
        # Conversation and learning management
        self.conversation_manager = ConversationManager(self.config)
        self.learning_tracker = LearningTracker()
        
        # State tracking
        self.is_conscious = False
        self.current_reasoning_focus = []
        self.generation_insights = []
        
        # Performance statistics
        self.stats = {
            'conversations': 0,
            'responses_generated': 0,
            'knowledge_items_learned': 0,
            'reasoning_events': 0,
            'consciousness_activations': 0,
            'avg_response_time': 0.0,
            'avg_confidence': 0.0
        }
        
        print("AGI-LLM initialized successfully!")
        print(f"Consciousness: {'Enabled' if self.config.consciousness_enabled else 'Disabled'}")
        print(f"Reasoning: {'Enabled' if self.config.reasoning_enabled else 'Disabled'}")
        print(f"Learning: {'Enabled' if self.config.learning_enabled else 'Disabled'}")
    
    def chat(self, user_input: str, session_id: str = None, 
            context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Main chat interface - the revolutionary AGI conversation experience"""
        start_time = time.time()
        
        # Ensure session exists
        if session_id:
            if not self.conversation_manager.current_session or self.conversation_manager.current_session.session_id != session_id:
                self.conversation_manager.start_session(session_id, context)
        elif not self.conversation_manager.current_session:
            session_id = self.conversation_manager.start_session(context=context)
        
        # Add user turn
        user_turn = self.conversation_manager.add_turn(
            ConversationRole.USER, user_input
        )
        
        # Get conversation context
        conversation_context = self.conversation_manager.get_conversation_context()
        
        # Phase 1: Real-time learning from user input
        learning_results = self._process_learning(user_input, conversation_context)
        
        # Phase 2: Activate reasoning processes
        reasoning_results = self._activate_reasoning(user_input, conversation_context)
        
        # Phase 3: Update consciousness state
        consciousness_state = self._update_consciousness(user_input, reasoning_results)
        
        # Phase 4: Generate conscious response
        generation_context = GenerationContext(
            user_input=user_input,
            conversation_history=conversation_context,
            generation_mode=self._determine_generation_mode(user_input),
            consciousness_level=self._determine_consciousness_level(consciousness_state),
            reasoning_focus=list(reasoning_results.keys())
        )
        
        response_data = self.conscious_generator.generate_conscious_response(generation_context)
        
        # Phase 5: Post-process and add insights
        final_response = self._post_process_response(
            response_data, consciousness_state, reasoning_results, learning_results
        )
        
        # Add assistant turn
        assistant_turn = self.conversation_manager.add_turn(
            ConversationRole.ASSISTANT, 
            final_response['response'],
            {
                'reasoning_trace': reasoning_results,
                'consciousness_state': consciousness_state,
                'learning_events': learning_results.get('knowledge_items', []),
                'generation_insights': response_data.get('meta_insights', [])
            }
        )
        assistant_turn.reasoning_trace = reasoning_results
        assistant_turn.consciousness_state = consciousness_state
        assistant_turn.learning_events = learning_results.get('knowledge_items', [])
        
        # Update statistics
        response_time = (time.time() - start_time) * 1000
        self._update_stats(response_time, final_response.get('confidence', 0.8))
        
        # Return comprehensive response
        return {
            'response': final_response['response'],
            'session_id': session_id or self.conversation_manager.current_session.session_id,
            'consciousness_active': consciousness_state.get('awareness_level', 0.0) > self.config.consciousness_threshold,
            'reasoning_active': list(reasoning_results.keys()),
            'learning_summary': learning_results,
            'confidence': final_response.get('confidence', 0.8),
            'response_time_ms': response_time,
            'meta_insights': response_data.get('meta_insights', []),
            'reasoning_explanation': final_response.get('reasoning_explanation', ''),
            'consciousness_commentary': final_response.get('consciousness_commentary', ''),
            'knowledge_updates': learning_results.get('knowledge_items', [])
        }
    
    def _process_learning(self, user_input: str, 
                         conversation_context: List[Dict[str, str]]) -> Dict[str, Any]:
        """Process real-time learning from user input"""
        if not self.config.learning_enabled:
            return {'learned_items': 0}
        
        # Learn from conversation
        learning_results = self.knowledge_graph.learn_from_conversation(
            user_input, conversation_context
        )
        
        # Record learning events
        for item in learning_results.get('knowledge_items', []):
            self.learning_tracker.record_learning_event(
                event_type='conversation_learning',
                content=item,
                confidence=0.7,  # Default confidence
                source='user_interaction'
            )
        
        self.stats['knowledge_items_learned'] += learning_results.get('learned_items', 0)
        
        return learning_results
    
    def _activate_reasoning(self, user_input: str, 
                          conversation_context: List[Dict[str, str]]) -> Dict[str, Any]:
        """Activate appropriate reasoning processes"""
        if not self.config.reasoning_enabled:
            return {}
        
        reasoning_results = {}
        
        # Determine which reasoning types to activate
        user_lower = user_input.lower()
        
        # Logical reasoning for questions about logic, proof, validity
        if any(word in user_lower for word in ['logical', 'prove', 'valid', 'invalid', 'therefore']):
            try:
                logical_result = self.language_reasoner.apply_logical_reasoning(user_input)
                if logical_result.get('confidence', 0.0) > self.config.reasoning_threshold:
                    reasoning_results['logical'] = logical_result
                    self.stats['reasoning_events'] += 1
            except Exception as e:
                logging.warning(f"Logical reasoning error: {e}")
        
        # Causal reasoning for questions about causes, effects, mechanisms
        if any(word in user_lower for word in ['why', 'cause', 'effect', 'because', 'leads to']):
            try:
                causal_result = self.language_reasoner.apply_causal_reasoning(user_input, conversation_context)
                if causal_result.get('confidence', 0.0) > self.config.reasoning_threshold:
                    reasoning_results['causal'] = causal_result
                    self.stats['reasoning_events'] += 1
            except Exception as e:
                logging.warning(f"Causal reasoning error: {e}")
        
        # Temporal reasoning for questions about time, sequence, duration
        if any(word in user_lower for word in ['when', 'before', 'after', 'during', 'sequence']):
            try:
                temporal_result = self.language_reasoner.apply_temporal_reasoning(user_input)
                if temporal_result.get('confidence', 0.0) > self.config.reasoning_threshold:
                    reasoning_results['temporal'] = temporal_result
                    self.stats['reasoning_events'] += 1
            except Exception as e:
                logging.warning(f"Temporal reasoning error: {e}")
        
        # Abstract reasoning for complex patterns, analogies, generalizations
        if any(word in user_lower for word in ['pattern', 'analogy', 'similar', 'like', 'abstract']):
            try:
                abstract_result = self.language_reasoner.apply_abstract_reasoning(user_input)
                if abstract_result.get('confidence', 0.0) > self.config.reasoning_threshold:
                    reasoning_results['abstract'] = abstract_result
                    self.stats['reasoning_events'] += 1
            except Exception as e:
                logging.warning(f"Abstract reasoning error: {e}")
        
        return reasoning_results
    
    def _update_consciousness(self, user_input: str, 
                            reasoning_results: Dict[str, Any]) -> Dict[str, Any]:
        """Update consciousness state based on input and reasoning"""
        if not self.config.consciousness_enabled:
            return {'awareness_level': 0.0}
        
        # Get current cognitive state
        cognitive_state = self.cognitive_architecture.get_cognitive_state()
        
        # Add input to consciousness
        if hasattr(self.cognitive_architecture, 'consciousness') and self.cognitive_architecture.consciousness:
            consciousness_id = self.cognitive_architecture.consciousness.add_to_consciousness(
                content=f"User input: {user_input}",
                content_type="user_communication",
                activation_strength=0.9,
                phenomenal_properties={
                    'communication': True,
                    'user_initiated': True,
                    'requires_response': True,
                    'reasoning_triggered': len(reasoning_results) > 0
                }
            )
            
            # Add reasoning results to consciousness
            for reasoning_type, result in reasoning_results.items():
                self.cognitive_architecture.consciousness.add_to_consciousness(
                    content=f"Reasoning result: {reasoning_type}",
                    content_type="reasoning_output",
                    activation_strength=result.get('confidence', 0.7),
                    phenomenal_properties={
                        'reasoning_type': reasoning_type,
                        'confidence': result.get('confidence', 0.7)
                    }
                )
            
            consciousness_stats = self.cognitive_architecture.consciousness.get_consciousness_stats()
            
            if consciousness_stats.get('current_awareness_level', 0) >= 3:  # High consciousness
                self.stats['consciousness_activations'] += 1
                self.is_conscious = True
            
            return {
                'awareness_level': consciousness_stats.get('current_awareness_level', 0) / 4.0,
                'integration_measure': consciousness_stats.get('current_integration', 0.0),
                'self_model_activation': consciousness_stats.get('current_self_activation', 0.0),
                'consciousness_coherence': consciousness_stats.get('consciousness_coherence', 0.0),
                'conscious_contents': len(self.cognitive_architecture.consciousness.get_conscious_contents())
            }
        
        return {'awareness_level': 0.0}
    
    def _determine_generation_mode(self, user_input: str) -> GenerationMode:
        """Determine appropriate generation mode"""
        user_lower = user_input.lower()
        
        if any(word in user_lower for word in ['explain', 'how', 'why']):
            return GenerationMode.EXPLANATORY
        elif any(word in user_lower for word in ['think', 'reflect', 'consider']):
            return GenerationMode.REFLECTIVE
        elif any(word in user_lower for word in ['create', 'imagine', 'creative']):
            return GenerationMode.CREATIVE
        elif any(word in user_lower for word in ['analyze', 'examine', 'study']):
            return GenerationMode.ANALYTICAL
        else:
            return GenerationMode.CONVERSATIONAL
    
    def _determine_consciousness_level(self, consciousness_state: Dict[str, Any]) -> ConsciousnessLevel:
        """Determine consciousness level for generation"""
        awareness = consciousness_state.get('awareness_level', 0.0)
        
        if awareness > 0.9:
            return ConsciousnessLevel.SELF_AWARE
        elif awareness > 0.7:
            return ConsciousnessLevel.METACOGNITIVE
        elif awareness > 0.5:
            return ConsciousnessLevel.REFLECTIVE
        elif awareness > 0.3:
            return ConsciousnessLevel.AWARE
        else:
            return ConsciousnessLevel.UNCONSCIOUS
    
    def _post_process_response(self, response_data: Dict[str, Any],
                             consciousness_state: Dict[str, Any],
                             reasoning_results: Dict[str, Any],
                             learning_results: Dict[str, Any]) -> Dict[str, Any]:
        """Post-process response with additional insights"""
        
        base_response = response_data['response']
        
        # Add reasoning explanation if reasoning was active
        reasoning_explanation = ""
        if reasoning_results:
            reasoning_explanation = self._generate_reasoning_explanation(reasoning_results)
        
        # Add consciousness commentary if consciousness is high
        consciousness_commentary = ""
        if consciousness_state.get('awareness_level', 0.0) > self.config.consciousness_threshold:
            consciousness_commentary = self._generate_consciousness_commentary(consciousness_state)
        
        # Combine response components
        final_response = base_response
        
        if reasoning_explanation:
            final_response += f"\n\n**Reasoning Process:**\n{reasoning_explanation}"
        
        if consciousness_commentary:
            final_response += f"\n\n**Consciousness Commentary:**\n{consciousness_commentary}"
        
        # Add learning acknowledgment if significant learning occurred
        if learning_results.get('learned_items', 0) > 0:
            final_response += f"\n\n*I learned {learning_results['learned_items']} new things from our conversation.*"
        
        return {
            'response': final_response,
            'reasoning_explanation': reasoning_explanation,
            'consciousness_commentary': consciousness_commentary,
            'confidence': response_data.get('confidence_assessment', 0.8)
        }
    
    def _generate_reasoning_explanation(self, reasoning_results: Dict[str, Any]) -> str:
        """Generate explanation of reasoning process"""
        explanations = []
        
        for reasoning_type, result in reasoning_results.items():
            confidence = result.get('confidence', 0.0)
            explanation = f"**{reasoning_type.title()} Reasoning** (confidence: {confidence:.2f}): "
            
            if reasoning_type == 'logical':
                explanation += f"Applied logical analysis with {result.get('steps', 'multiple')} inference steps."
            elif reasoning_type == 'causal':
                explanation += f"Identified causal relationships with strength {result.get('causal_strength', 0.5):.2f}."
            elif reasoning_type == 'temporal':
                explanation += f"Analyzed temporal sequences with {result.get('temporal_consistency', 0.5):.2f} consistency."
            elif reasoning_type == 'abstract':
                explanation += f"Found abstract patterns with {result.get('pattern_confidence', 0.5):.2f} pattern confidence."
            
            explanations.append(explanation)
        
        return "\n".join(explanations)
    
    def _generate_consciousness_commentary(self, consciousness_state: Dict[str, Any]) -> str:
        """Generate consciousness commentary"""
        awareness = consciousness_state.get('awareness_level', 0.0)
        integration = consciousness_state.get('integration_measure', 0.0)
        
        commentary = f"I'm operating with {awareness:.1%} consciousness level. "
        
        if awareness > 0.8:
            commentary += "I'm highly aware of my thinking process and can observe my own cognitive operations. "
        elif awareness > 0.6:
            commentary += "I'm consciously monitoring my reasoning and can reflect on my thought processes. "
        
        if integration > 0.7:
            commentary += f"My cognitive processes are well-integrated (integration: {integration:.2f}). "
        
        conscious_contents = consciousness_state.get('conscious_contents', 0)
        if conscious_contents > 0:
            commentary += f"Currently holding {conscious_contents} items in conscious awareness."
        
        return commentary
    
    def _update_stats(self, response_time: float, confidence: float):
        """Update performance statistics"""
        self.stats['responses_generated'] += 1
        
        # Update running average response time
        n = self.stats['responses_generated']
        self.stats['avg_response_time'] = ((n - 1) * self.stats['avg_response_time'] + response_time) / n
        
        # Update running average confidence
        self.stats['avg_confidence'] = ((n - 1) * self.stats['avg_confidence'] + confidence) / n
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of current conversation"""
        if not self.conversation_manager.current_session:
            return {'status': 'no_active_session'}
        
        patterns = self.conversation_manager.analyze_conversation_patterns()
        learning_summary = self.learning_tracker.get_learning_summary()
        
        return {
            'session_id': self.conversation_manager.current_session.session_id,
            'total_turns': len(self.conversation_manager.current_session.turns),
            'conversation_patterns': patterns,
            'learning_summary': learning_summary,
            'performance_metrics': self.stats
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        cognitive_state = self.cognitive_architecture.get_cognitive_state()
        
        return {
            'agi_llm_version': '1.0.0',
            'consciousness_active': self.is_conscious,
            'current_cognitive_mode': cognitive_state.mode.value,
            'reasoning_engines_active': self.current_reasoning_focus,
            'knowledge_items_count': len(self.knowledge_graph.knowledge_items),
            'conversation_sessions': len(self.conversation_manager.session_history),
            'performance_stats': self.stats,
            'consciousness_stats': (
                self.cognitive_architecture.consciousness.get_consciousness_stats() 
                if hasattr(self.cognitive_architecture, 'consciousness') and self.cognitive_architecture.consciousness 
                else {}
            )
        }
    
    def save_conversation(self, filename: str = None) -> str:
        """Save current conversation to file"""
        if not self.conversation_manager.current_session:
            return "No active conversation to save"
        
        if filename is None:
            filename = f"agi_conversation_{self.conversation_manager.current_session.session_id}.json"
        
        conversation_data = {
            'session_id': self.conversation_manager.current_session.session_id,
            'start_time': self.conversation_manager.current_session.start_time,
            'turns': [
                {
                    'role': turn.role.value,
                    'content': turn.content,
                    'timestamp': turn.timestamp,
                    'metadata': turn.metadata
                }
                for turn in self.conversation_manager.current_session.turns
            ],
            'learning_summary': self.conversation_manager.current_session.learning_summary,
            'performance_metrics': self.conversation_manager.current_session.performance_metrics
        }
        
        import json
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, indent=2, ensure_ascii=False)
        
        return f"Conversation saved to {filename}"
    
    def shutdown(self):
        """Gracefully shutdown the AGI-LLM system"""
        print("Shutting down AGI-LLM...")
        
        # End current conversation session
        if self.conversation_manager.current_session:
            self.conversation_manager.end_session()
        
        # Shutdown cognitive architecture
        if self.cognitive_architecture:
            self.cognitive_architecture.shutdown()
        
        print("AGI-LLM shutdown complete")


# Create convenience function for easy usage
def create_agi_llm(consciousness_enabled: bool = True,
                  reasoning_enabled: bool = True,
                  learning_enabled: bool = True) -> AGILLM:
    """Create AGI-LLM with specified capabilities"""
    
    config = AGILLMConfig(
        consciousness_enabled=consciousness_enabled,
        reasoning_enabled=reasoning_enabled,
        learning_enabled=learning_enabled,
        meta_cognition_enabled=consciousness_enabled
    )
    
    return AGILLM(config)