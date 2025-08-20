"""
Unified Cognitive Architecture for AGI-Formula

This module integrates all cognitive components into a unified architecture:
- Executive control and attention management
- Working memory and long-term integration  
- Consciousness simulation and awareness
- Reasoning system integration
- Multi-modal processing coordination
- Meta-cognitive monitoring and control
"""

import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
import threading
import asyncio
from collections import deque, defaultdict

# Import cognitive components
from .executive_control import ExecutiveController, CognitiveTask, TaskPriority, AttentionType
from .working_memory import WorkingMemoryManager, MemoryType, MemoryPriority, MemoryChunk
from .consciousness import ConsciousnessSimulator, AwarenessLevel, ConsciousnessType


class CognitiveMode(Enum):
    """Different modes of cognitive operation"""
    REACTIVE = "reactive"         # Reactive processing mode
    DELIBERATIVE = "deliberative" # Deliberative reasoning mode
    CREATIVE = "creative"         # Creative problem-solving mode
    REFLECTIVE = "reflective"     # Self-reflective mode
    LEARNING = "learning"         # Learning and adaptation mode
    SOCIAL = "social"             # Social interaction mode


class ProcessingPriority(Enum):
    """Processing priorities across the architecture"""
    SURVIVAL = 5      # Critical survival needs
    GOAL_CRITICAL = 4 # Critical for current goals
    ATTENTION = 3     # Attention-demanding tasks
    ROUTINE = 2       # Routine processing
    BACKGROUND = 1    # Background processing


@dataclass
class CognitiveProcess:
    """Representation of a cognitive process"""
    process_id: str
    name: str
    process_type: str
    priority: ProcessingPriority
    resource_requirements: Dict[str, float]
    input_requirements: List[str]
    output_products: List[str]
    active: bool = False
    performance_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class CognitiveState:
    """Overall state of the cognitive architecture"""
    timestamp: float
    mode: CognitiveMode
    consciousness_level: AwarenessLevel
    attention_focus: List[str]
    working_memory_load: float
    executive_control_active: bool
    reasoning_active: Dict[str, bool]
    learning_rate: float
    adaptation_level: float
    performance_indicators: Dict[str, float]
    meta_cognitive_insights: List[str]


@dataclass
class ArchitectureConfig:
    """Configuration for cognitive architecture"""
    max_concurrent_processes: int = 10
    consciousness_enabled: bool = True
    meta_cognition_enabled: bool = True
    learning_enabled: bool = True
    adaptation_rate: float = 0.01
    performance_monitoring: bool = True
    integration_frequency: float = 0.1  # seconds


class CognitiveArchitecture:
    """
    Unified cognitive architecture integrating all AGI components
    
    This is the top-level orchestrator that coordinates:
    - Executive control and resource allocation
    - Working memory management 
    - Consciousness simulation
    - Reasoning system integration
    - Multi-modal processing
    - Learning and adaptation
    - Meta-cognitive monitoring
    """
    
    def __init__(self, config: Optional[ArchitectureConfig] = None):
        self.config = config or ArchitectureConfig()
        
        # Initialize core cognitive systems
        self.executive_controller = ExecutiveController()
        self.working_memory = WorkingMemoryManager()
        self.consciousness = ConsciousnessSimulator() if self.config.consciousness_enabled else None
        
        # Cognitive state
        self.current_state = CognitiveState(
            timestamp=time.time(),
            mode=CognitiveMode.REACTIVE,
            consciousness_level=AwarenessLevel.CONSCIOUS,
            attention_focus=[],
            working_memory_load=0.0,
            executive_control_active=True,
            reasoning_active={},
            learning_rate=0.01,
            adaptation_level=0.5,
            performance_indicators={},
            meta_cognitive_insights=[]
        )
        
        # Process management
        self.active_processes = {}
        self.process_registry = {}
        self.integration_thread = None
        self.integration_active = False
        
        # Performance monitoring
        self.performance_history = deque(maxlen=1000)
        self.adaptation_history = deque(maxlen=100)
        
        # Statistics
        self.stats = {
            'cognitive_cycles': 0,
            'mode_changes': 0,
            'processes_executed': 0,
            'consciousness_events': 0,
            'learning_events': 0,
            'adaptation_events': 0,
            'performance_improvements': 0
        }
        
        # Initialize architecture
        self._initialize_architecture()
        
        print("Unified cognitive architecture initialized")
    
    def _initialize_architecture(self):
        """Initialize the cognitive architecture"""
        # Register core processes
        self._register_core_processes()
        
        # Start consciousness if enabled
        if self.consciousness:
            self.consciousness.start_consciousness_processing()
        
        # Initialize integration
        self._start_integration_cycle()
        
        print("Cognitive architecture fully operational")
    
    def _register_core_processes(self):
        """Register core cognitive processes"""
        core_processes = [
            CognitiveProcess(
                process_id="perception",
                name="Perceptual Processing",
                process_type="perception",
                priority=ProcessingPriority.ATTENTION,
                resource_requirements={"perception": 0.7, "attention": 0.5},
                input_requirements=["sensory_input"],
                output_products=["perceptual_content"]
            ),
            CognitiveProcess(
                process_id="reasoning",
                name="Reasoning and Inference",
                process_type="reasoning",
                priority=ProcessingPriority.GOAL_CRITICAL,
                resource_requirements={"reasoning": 0.8, "working_memory": 0.6},
                input_requirements=["facts", "goals"],
                output_products=["inferences", "decisions"]
            ),
            CognitiveProcess(
                process_id="learning",
                name="Learning and Adaptation",
                process_type="learning",
                priority=ProcessingPriority.ROUTINE,
                resource_requirements={"processing_power": 0.6, "working_memory": 0.4},
                input_requirements=["experience", "feedback"],
                output_products=["knowledge", "adaptations"]
            ),
            CognitiveProcess(
                process_id="planning",
                name="Goal Planning",
                process_type="planning",
                priority=ProcessingPriority.GOAL_CRITICAL,
                resource_requirements={"reasoning": 0.7, "working_memory": 0.8},
                input_requirements=["goals", "world_model"],
                output_products=["plans", "actions"]
            )
        ]
        
        for process in core_processes:
            self.process_registry[process.process_id] = process
    
    def _start_integration_cycle(self):
        """Start the cognitive integration cycle"""
        if not self.integration_active:
            self.integration_active = True
            self.integration_thread = threading.Thread(target=self._integration_loop)
            self.integration_thread.daemon = True
            self.integration_thread.start()
    
    def _integration_loop(self):
        """Main integration loop coordinating all cognitive systems"""
        while self.integration_active:
            try:
                # Process one cognitive cycle
                self._process_cognitive_cycle()
                
                # Sleep for integration frequency
                time.sleep(self.config.integration_frequency)
                
            except Exception as e:
                logging.error(f"Error in cognitive integration: {e}")
                time.sleep(0.1)
    
    def _process_cognitive_cycle(self):
        """Process one complete cognitive cycle"""
        cycle_start = time.time()
        
        # Step 1: Update cognitive state
        self._update_cognitive_state()
        
        # Step 2: Executive control cycle
        executive_summary = self.executive_controller.execute_cognitive_cycle()
        
        # Step 3: Working memory maintenance
        self.working_memory.update_working_memory()
        
        # Step 4: Process active cognitive processes
        self._process_active_cognitive_processes()
        
        # Step 5: Mode management
        self._manage_cognitive_mode()
        
        # Step 6: Meta-cognitive monitoring
        if self.config.meta_cognition_enabled:
            self._meta_cognitive_monitoring()
        
        # Step 7: Learning and adaptation
        if self.config.learning_enabled:
            self._learning_and_adaptation()
        
        # Step 8: Performance monitoring
        if self.config.performance_monitoring:
            self._performance_monitoring(executive_summary)
        
        # Step 9: Integration with consciousness
        if self.consciousness:
            self._integrate_with_consciousness()
        
        cycle_time = (time.time() - cycle_start) * 1000
        
        # Record cycle
        self.stats['cognitive_cycles'] += 1
        
        # Store performance
        self.performance_history.append({
            'timestamp': time.time(),
            'cycle_time': cycle_time,
            'state': self.current_state,
            'executive_summary': executive_summary
        })
    
    def set_goal(self, goal_description: str, priority: TaskPriority = TaskPriority.NORMAL, 
                context: Optional[Dict[str, Any]] = None) -> str:
        """Set a high-level cognitive goal"""
        # Create goal structure
        goal = {
            'description': goal_description,
            'type': self._infer_goal_type(goal_description),
            'priority': priority.value,
            'context': context or {},
            'estimated_duration': self._estimate_goal_duration(goal_description),
            'deadline': None
        }
        
        # Pass to executive controller
        goal_id = self.executive_controller.set_goal(goal)
        
        # Store in working memory
        self.working_memory.store_information(
            content=goal,
            memory_type=MemoryType.EPISODIC,
            priority=MemoryPriority.HIGH,
            context={'type': 'goal', 'goal_id': goal_id},
            retrieval_cues={goal_description, 'goal', 'current_task'}
        )
        
        # Add to consciousness if available
        if self.consciousness:
            self.consciousness.add_to_consciousness(
                content=f"New goal: {goal_description}",
                content_type="goal_setting",
                activation_strength=0.8,
                phenomenal_properties={
                    'valence': 0.3,  # Slightly positive
                    'arousal': 0.7,  # High arousal for new goals
                    'goal_related': True
                }
            )
        
        return goal_id
    
    def process_input(self, input_data: Any, input_type: str, 
                     context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process external input through the cognitive architecture"""
        processing_start = time.time()
        
        # Store input in working memory
        input_chunk_id = self.working_memory.store_information(
            content=input_data,
            memory_type=self._determine_memory_type(input_type),
            priority=MemoryPriority.HIGH,
            context=context or {'type': 'input', 'input_type': input_type},
            retrieval_cues={input_type, 'recent_input', 'current'}
        )
        
        # Add to consciousness
        if self.consciousness:
            consciousness_id = self.consciousness.add_to_consciousness(
                content=input_data,
                content_type=input_type,
                activation_strength=0.7,
                phenomenal_properties=self._generate_phenomenal_properties(input_data, input_type)
            )
        
        # Update attention focus
        self.current_state.attention_focus = [input_chunk_id]
        
        # Process through relevant cognitive processes
        processing_results = self._route_to_cognitive_processes(input_data, input_type)
        
        processing_time = (time.time() - processing_start) * 1000
        
        return {
            'input_chunk_id': input_chunk_id,
            'consciousness_id': consciousness_id if self.consciousness else None,
            'processing_results': processing_results,
            'processing_time_ms': processing_time,
            'cognitive_state': self.current_state
        }
    
    def get_response(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate response to a query using the full cognitive architecture"""
        # Process query as input
        query_result = self.process_input(query, "language_query", context)
        
        # Retrieve relevant memories
        relevant_memories = self.working_memory.retrieve_information(
            retrieval_cues={query, "relevant", "context"},
            context=context
        )
        
        # Engage reasoning if needed
        if self._requires_reasoning(query):
            self.current_state.mode = CognitiveMode.DELIBERATIVE
            
            # Create reasoning task
            reasoning_task = CognitiveTask(
                task_id=f"reasoning_{int(time.time())}",
                name=f"Reason about: {query[:50]}",
                priority=TaskPriority.HIGH,
                resource_requirements={
                    "reasoning": 0.8,
                    "working_memory": 0.6,
                    "attention": 0.7
                },
                estimated_duration=2.0,
                metadata={'query': query, 'context': context}
            )
            
            self.executive_controller.task_scheduler.add_task(reasoning_task)
        
        # Generate response based on cognitive processing
        response = self._generate_response(query, relevant_memories, query_result)
        
        # Store response in memory
        self.working_memory.store_information(
            content={'query': query, 'response': response},
            memory_type=MemoryType.EPISODIC,
            priority=MemoryPriority.NORMAL,
            context={'type': 'interaction', 'timestamp': time.time()},
            retrieval_cues={'conversation', 'response', query}
        )
        
        return response
    
    def _update_cognitive_state(self):
        """Update overall cognitive state"""
        self.current_state.timestamp = time.time()
        
        # Update consciousness level
        if self.consciousness:
            consciousness_state = self.consciousness.get_consciousness_state()
            self.current_state.consciousness_level = consciousness_state.awareness_level
        
        # Update attention focus from executive controller
        executive_stats = self.executive_controller.get_executive_stats()
        self.current_state.attention_focus = executive_stats.get('attention_state', {}).get('focus_targets', [])
        
        # Update working memory load
        memory_state = self.working_memory.get_memory_state()
        self.current_state.working_memory_load = np.mean([
            buffer_state['utilization'] for buffer_state in memory_state['buffer_states'].values()
        ])
        
        # Update executive control status
        self.current_state.executive_control_active = len(executive_stats.get('active_tasks', 0)) > 0
        
        # Update reasoning activity
        self.current_state.reasoning_active = {
            'logical': True,  # Would check actual reasoning systems
            'causal': True,
            'temporal': True,
            'abstract': True
        }
    
    def _process_active_cognitive_processes(self):
        """Process all active cognitive processes"""
        for process_id, process in self.active_processes.items():
            if process.active:
                try:
                    # Simulate process execution
                    self._execute_cognitive_process(process)
                    process.performance_metrics['executions'] = process.performance_metrics.get('executions', 0) + 1
                    
                except Exception as e:
                    logging.error(f"Error executing process {process_id}: {e}")
                    process.performance_metrics['errors'] = process.performance_metrics.get('errors', 0) + 1
        
        self.stats['processes_executed'] += len(self.active_processes)
    
    def _execute_cognitive_process(self, process: CognitiveProcess):
        """Execute a specific cognitive process"""
        # This would interface with actual processing systems
        # For now, we simulate successful execution
        
        if process.process_type == "perception":
            # Would interface with multimodal pipeline
            pass
        elif process.process_type == "reasoning":
            # Would interface with reasoning systems
            pass
        elif process.process_type == "learning":
            # Would interface with learning systems
            pass
        elif process.process_type == "planning":
            # Would interface with planning systems
            pass
    
    def _manage_cognitive_mode(self):
        """Manage transitions between cognitive modes"""
        current_mode = self.current_state.mode
        
        # Determine if mode change is needed
        new_mode = self._determine_optimal_mode()
        
        if new_mode != current_mode:
            self.current_state.mode = new_mode
            self.stats['mode_changes'] += 1
            
            # Adjust processing based on new mode
            self._adjust_for_mode(new_mode)
    
    def _determine_optimal_mode(self) -> CognitiveMode:
        """Determine optimal cognitive mode based on current state"""
        # Simple heuristic-based mode selection
        
        if self.current_state.consciousness_level == AwarenessLevel.SELF_AWARE:
            return CognitiveMode.REFLECTIVE
        
        if self.current_state.working_memory_load > 0.8:
            return CognitiveMode.REACTIVE  # Reduce load
        
        executive_stats = self.executive_controller.get_executive_stats()
        cognitive_load = executive_stats.get('current_cognitive_load', 0)
        
        if cognitive_load > 0.7:
            return CognitiveMode.DELIBERATIVE
        elif cognitive_load < 0.3:
            return CognitiveMode.CREATIVE
        else:
            return CognitiveMode.REACTIVE
    
    def _adjust_for_mode(self, mode: CognitiveMode):
        """Adjust cognitive processing for new mode"""
        if mode == CognitiveMode.DELIBERATIVE:
            # Increase reasoning resource allocation
            self.executive_controller.resource_manager.reallocate_resources({
                "reasoning": 0.8,
                "working_memory": 0.7,
                "attention": 0.6
            })
        elif mode == CognitiveMode.CREATIVE:
            # Increase exploration and reduce constraints
            pass
        elif mode == CognitiveMode.REFLECTIVE:
            # Focus on self-monitoring and meta-cognition
            if self.consciousness:
                self.consciousness.add_to_consciousness(
                    "Entering reflective mode",
                    "meta_cognitive",
                    0.9
                )
    
    def _meta_cognitive_monitoring(self):
        """Perform meta-cognitive monitoring and control"""
        # Assess current cognitive performance
        performance_assessment = self._assess_cognitive_performance()
        
        # Generate meta-cognitive insights
        insights = []
        
        if performance_assessment['efficiency'] < 0.6:
            insights.append("Cognitive efficiency below optimal - consider strategy adjustment")
        
        if performance_assessment['resource_utilization'] > 0.9:
            insights.append("High resource utilization - may need load balancing")
        
        if performance_assessment['goal_progress'] < 0.3:
            insights.append("Slow goal progress - may need goal reassessment")
        
        self.current_state.meta_cognitive_insights = insights
        
        # Consciousness integration
        if self.consciousness and insights:
            for insight in insights:
                self.consciousness.add_to_consciousness(
                    insight,
                    "meta_cognitive_insight",
                    0.7,
                    {'valence': -0.2, 'arousal': 0.6}  # Slightly negative, moderate arousal
                )
    
    def _learning_and_adaptation(self):
        """Perform learning and adaptation"""
        # Simple adaptation based on recent performance
        if len(self.performance_history) >= 10:
            recent_performance = list(self.performance_history)[-10:]
            avg_cycle_time = np.mean([p['cycle_time'] for p in recent_performance])
            
            # Adapt integration frequency based on performance
            target_cycle_time = 50  # ms
            
            if avg_cycle_time > target_cycle_time * 1.2:
                # Too slow, reduce frequency
                self.config.integration_frequency *= 1.1
                self.stats['adaptation_events'] += 1
            elif avg_cycle_time < target_cycle_time * 0.8:
                # Too fast, can increase frequency
                self.config.integration_frequency *= 0.95
                self.stats['adaptation_events'] += 1
            
            # Update learning rate
            self.current_state.learning_rate = 0.01 * (target_cycle_time / max(avg_cycle_time, 1))
            
            self.stats['learning_events'] += 1
    
    def _performance_monitoring(self, executive_summary: Dict[str, Any]):
        """Monitor cognitive architecture performance"""
        performance_indicators = {}
        
        # Executive control performance
        performance_indicators['executive_efficiency'] = executive_summary.get('resource_utilization', {}).get('efficiency', 0.5)
        
        # Working memory performance
        memory_stats = self.working_memory.get_working_memory_stats()
        performance_indicators['memory_efficiency'] = memory_stats.get('memory_efficiency', 0.5)
        
        # Consciousness performance
        if self.consciousness:
            consciousness_stats = self.consciousness.get_consciousness_stats()
            performance_indicators['consciousness_coherence'] = consciousness_stats.get('consciousness_coherence', 0.5)
        
        # Overall system performance
        performance_indicators['overall_efficiency'] = np.mean(list(performance_indicators.values()))
        
        self.current_state.performance_indicators = performance_indicators
        
        # Check for improvements
        if (len(self.performance_history) > 1 and
            performance_indicators['overall_efficiency'] > 
            self.performance_history[-1].get('executive_summary', {}).get('overall_efficiency', 0)):
            self.stats['performance_improvements'] += 1
    
    def _integrate_with_consciousness(self):
        """Integrate cognitive processes with consciousness"""
        if not self.consciousness:
            return
        
        consciousness_stats = self.consciousness.get_consciousness_stats()
        
        # Add consciousness events to statistics
        self.stats['consciousness_events'] += consciousness_stats.get('recent_broadcasts', 0)
        
        # Integrate consciousness state with cognitive state
        consciousness_state = self.consciousness.get_consciousness_state()
        
        if consciousness_state.consciousness_type == ConsciousnessType.REFLECTIVE:
            self.current_state.mode = CognitiveMode.REFLECTIVE
        
        # Share working memory contents with consciousness
        memory_state = self.working_memory.get_memory_state()
        
        for buffer_type, buffer_state in memory_state['buffer_states'].items():
            if buffer_state['active_chunks'] > 0:
                self.consciousness.add_to_consciousness(
                    f"Active {buffer_type} memory content",
                    f"working_memory_{buffer_type}",
                    buffer_state['average_activation']
                )
    
    # Utility methods
    
    def _infer_goal_type(self, description: str) -> str:
        """Infer goal type from description"""
        description_lower = description.lower()
        
        if any(word in description_lower for word in ['learn', 'understand', 'study']):
            return 'learning'
        elif any(word in description_lower for word in ['solve', 'reason', 'figure']):
            return 'problem_solving'
        elif any(word in description_lower for word in ['remember', 'recall', 'find']):
            return 'memory_retrieval'
        elif any(word in description_lower for word in ['create', 'generate', 'make']):
            return 'creative'
        else:
            return 'general'
    
    def _estimate_goal_duration(self, description: str) -> float:
        """Estimate goal completion duration"""
        # Simple heuristic based on goal complexity
        word_count = len(description.split())
        
        if word_count < 5:
            return 2.0  # Simple goals
        elif word_count < 15:
            return 5.0  # Moderate goals
        else:
            return 10.0  # Complex goals
    
    def _determine_memory_type(self, input_type: str) -> MemoryType:
        """Determine appropriate memory type for input"""
        if 'visual' in input_type or 'spatial' in input_type:
            return MemoryType.VISUOSPATIAL_SKETCHPAD
        elif 'audio' in input_type or 'verbal' in input_type or 'language' in input_type:
            return MemoryType.PHONOLOGICAL_LOOP
        elif 'episodic' in input_type or 'event' in input_type:
            return MemoryType.EPISODIC_BUFFER
        elif 'semantic' in input_type or 'knowledge' in input_type:
            return MemoryType.SEMANTIC_BUFFER
        else:
            return MemoryType.CENTRAL_EXECUTIVE
    
    def _generate_phenomenal_properties(self, data: Any, input_type: str) -> Dict[str, Any]:
        """Generate phenomenal properties for consciousness"""
        properties = {
            'input_type': input_type,
            'valence': 0.0,  # Neutral by default
            'arousal': 0.5,  # Medium arousal
            'intensity': 0.7,
            'familiarity': 0.5
        }
        
        # Adjust based on input type
        if 'positive' in str(data).lower():
            properties['valence'] = 0.7
        elif 'negative' in str(data).lower():
            properties['valence'] = -0.7
        
        if 'urgent' in str(data).lower() or 'important' in str(data).lower():
            properties['arousal'] = 0.9
        
        return properties
    
    def _route_to_cognitive_processes(self, data: Any, input_type: str) -> Dict[str, Any]:
        """Route input to appropriate cognitive processes"""
        results = {}
        
        # Activate relevant processes
        if input_type in ['visual', 'auditory', 'sensory']:
            if 'perception' in self.process_registry:
                process = self.process_registry['perception']
                process.active = True
                self.active_processes['perception'] = process
                results['perception'] = 'activated'
        
        if input_type in ['question', 'problem', 'reasoning']:
            if 'reasoning' in self.process_registry:
                process = self.process_registry['reasoning']
                process.active = True
                self.active_processes['reasoning'] = process
                results['reasoning'] = 'activated'
        
        return results
    
    def _requires_reasoning(self, query: str) -> bool:
        """Determine if query requires reasoning"""
        reasoning_keywords = ['why', 'how', 'what if', 'because', 'explain', 'analyze', 'compare']
        return any(keyword in query.lower() for keyword in reasoning_keywords)
    
    def _generate_response(self, query: str, memories: List[MemoryChunk], 
                          processing_result: Dict[str, Any]) -> str:
        """Generate response based on cognitive processing"""
        # Simple response generation
        if memories:
            memory_content = [str(m.content) for m in memories[:3]]  # Top 3 memories
            response = f"Based on my understanding: {' '.join(memory_content[:100])}"
        else:
            response = "I understand your query and am processing it with my cognitive systems."
        
        # Add consciousness insights if available
        if self.consciousness:
            consciousness_state = self.consciousness.get_consciousness_state()
            if consciousness_state.awareness_level == AwarenessLevel.HIGHLY_CONSCIOUS:
                response += " (I'm highly conscious of this processing)"
        
        return response[:500]  # Limit response length
    
    def _assess_cognitive_performance(self) -> Dict[str, Any]:
        """Assess current cognitive performance"""
        performance = {}
        
        # Executive performance
        executive_stats = self.executive_controller.get_executive_stats()
        performance['efficiency'] = executive_stats.get('current_cognitive_load', 0.5)
        
        # Resource utilization
        resource_utils = list(executive_stats.get('resource_states', {}).values())
        if resource_utils:
            avg_utilization = np.mean([r.get('utilization', 0.5) for r in resource_utils])
            performance['resource_utilization'] = avg_utilization
        else:
            performance['resource_utilization'] = 0.5
        
        # Goal progress
        performance['goal_progress'] = 0.7  # Placeholder
        
        return performance
    
    def get_cognitive_state(self) -> CognitiveState:
        """Get current cognitive state"""
        return self.current_state
    
    def get_architecture_stats(self) -> Dict[str, Any]:
        """Get comprehensive architecture statistics"""
        stats = self.stats.copy()
        
        # Add component statistics
        stats['executive_stats'] = self.executive_controller.get_executive_stats()
        stats['memory_stats'] = self.working_memory.get_working_memory_stats()
        
        if self.consciousness:
            stats['consciousness_stats'] = self.consciousness.get_consciousness_stats()
        
        # Add current state
        stats['current_state'] = {
            'mode': self.current_state.mode.value,
            'consciousness_level': self.current_state.consciousness_level.value,
            'working_memory_load': self.current_state.working_memory_load,
            'performance_indicators': self.current_state.performance_indicators
        }
        
        # Add performance history summary
        if self.performance_history:
            recent_performance = list(self.performance_history)[-10:]
            stats['recent_performance'] = {
                'avg_cycle_time': np.mean([p['cycle_time'] for p in recent_performance]),
                'performance_trend': 'stable'  # Would calculate actual trend
            }
        
        return stats
    
    def shutdown(self):
        """Gracefully shutdown the cognitive architecture"""
        print("Shutting down cognitive architecture...")
        
        # Stop integration loop
        self.integration_active = False
        if self.integration_thread:
            self.integration_thread.join()
        
        # Stop consciousness processing
        if self.consciousness:
            self.consciousness.stop_consciousness_processing()
        
        print("Cognitive architecture shutdown complete")


class CognitiveAgent:
    """
    High-level cognitive agent interface
    
    Provides a simplified interface to the full cognitive architecture
    for easy interaction and testing
    """
    
    def __init__(self, config: Optional[ArchitectureConfig] = None):
        self.architecture = CognitiveArchitecture(config)
        self.interaction_history = deque(maxlen=1000)
    
    def think(self, thought: str) -> Dict[str, Any]:
        """Process a thought through the cognitive architecture"""
        result = self.architecture.process_input(thought, "internal_thought")
        
        self.interaction_history.append({
            'type': 'thought',
            'input': thought,
            'result': result,
            'timestamp': time.time()
        })
        
        return result
    
    def perceive(self, sensory_data: Any, modality: str) -> Dict[str, Any]:
        """Process sensory input"""
        result = self.architecture.process_input(sensory_data, f"sensory_{modality}")
        
        self.interaction_history.append({
            'type': 'perception',
            'modality': modality,
            'result': result,
            'timestamp': time.time()
        })
        
        return result
    
    def respond_to(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate response to query"""
        response = self.architecture.get_response(query, context)
        
        self.interaction_history.append({
            'type': 'response',
            'query': query,
            'response': response,
            'context': context,
            'timestamp': time.time()
        })
        
        return response
    
    def set_goal(self, goal: str, priority: str = "normal") -> str:
        """Set a goal for the agent"""
        priority_map = {
            'low': TaskPriority.LOW,
            'normal': TaskPriority.NORMAL,
            'high': TaskPriority.HIGH,
            'critical': TaskPriority.CRITICAL
        }
        
        goal_id = self.architecture.set_goal(goal, priority_map.get(priority, TaskPriority.NORMAL))
        
        self.interaction_history.append({
            'type': 'goal_setting',
            'goal': goal,
            'priority': priority,
            'goal_id': goal_id,
            'timestamp': time.time()
        })
        
        return goal_id
    
    def get_state(self) -> Dict[str, Any]:
        """Get current agent state"""
        return {
            'cognitive_state': self.architecture.get_cognitive_state(),
            'architecture_stats': self.architecture.get_architecture_stats(),
            'recent_interactions': list(self.interaction_history)[-5:]
        }
    
    def shutdown(self):
        """Shutdown the cognitive agent"""
        self.architecture.shutdown()