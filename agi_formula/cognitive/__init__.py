"""
Cognitive Architecture Module for AGI-Formula

This module implements a unified cognitive architecture that integrates:
- Executive control and attention management
- Working memory and long-term memory systems
- Consciousness simulation and awareness
- Goal-directed behavior and planning
- Meta-cognitive monitoring and control
- Cognitive load management and resource allocation
"""

from .executive_control import (
    ExecutiveController,
    AttentionManager,
    ResourceManager,
    TaskScheduler,
    CognitiveTask,
    AttentionState,
    ExecutionPlan,
    TaskPriority,
    AttentionType
)

from .working_memory import (
    WorkingMemoryManager,
    MemoryBuffer,
    MemoryChunk,
    MemoryType,
    MemoryPriority,
    ConsolidationState,
    BindingType
)

from .consciousness import (
    ConsciousnessSimulator,
    AwarenessLevel,
    ConsciousState,
    GlobalWorkspace,
    AttentionalBroadcast,
    SubjectiveExperience
)

from .cognitive_architecture import (
    CognitiveArchitecture,
    CognitiveAgent,
    CognitiveProcess,
    CognitiveState,
    ArchitectureConfig
)

__all__ = [
    # Executive control
    'ExecutiveController',
    'AttentionManager', 
    'ResourceManager',
    'TaskScheduler',
    'CognitiveTask',
    'AttentionState',
    'ExecutionPlan',
    'TaskPriority',
    'AttentionType',
    
    # Working memory
    'WorkingMemoryManager',
    'MemoryBuffer',
    'MemoryChunk',
    'MemoryType',
    'MemoryPriority',
    'ConsolidationState',
    'BindingType',
    
    # Consciousness
    'ConsciousnessSimulator',
    'AwarenessLevel',
    'ConsciousState',
    'GlobalWorkspace',
    'AttentionalBroadcast',
    'SubjectiveExperience',
    
    # Main architecture
    'CognitiveArchitecture',
    'CognitiveAgent',
    'CognitiveProcess',
    'CognitiveState',
    'ArchitectureConfig'
]