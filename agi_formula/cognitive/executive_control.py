"""
Executive Control System for AGI-Formula Cognitive Architecture

Advanced executive control implementing:
- Attention management and selective focus
- Resource allocation and cognitive load management  
- Task scheduling and priority management
- Goal-directed behavior and planning
- Conflict monitoring and resolution
- Meta-cognitive control and monitoring
"""

import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple, Union, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import deque, defaultdict
import heapq
from abc import ABC, abstractmethod


class AttentionType(Enum):
    """Types of attentional processes"""
    FOCUSED = "focused"           # Focused attention on specific target
    SELECTIVE = "selective"       # Selective attention filtering
    DIVIDED = "divided"           # Divided attention across targets
    SUSTAINED = "sustained"       # Sustained attention over time
    EXECUTIVE = "executive"       # Executive attention control


class TaskPriority(Enum):
    """Priority levels for cognitive tasks"""
    CRITICAL = 4    # Critical/urgent tasks
    HIGH = 3        # High priority tasks
    NORMAL = 2      # Normal priority tasks
    LOW = 1         # Low priority tasks
    BACKGROUND = 0  # Background processing


class ResourceType(Enum):
    """Types of cognitive resources"""
    ATTENTION = "attention"
    WORKING_MEMORY = "working_memory"
    PROCESSING_POWER = "processing_power"
    REASONING = "reasoning"
    PERCEPTION = "perception"
    MOTOR = "motor"


class ControlSignal(Enum):
    """Executive control signals"""
    INHIBIT = "inhibit"
    ACTIVATE = "activate"
    MODULATE = "modulate"
    SWITCH = "switch"
    MAINTAIN = "maintain"
    UPDATE = "update"


@dataclass
class AttentionState:
    """Current state of attentional system"""
    focus_targets: List[str]
    attention_type: AttentionType
    intensity: float  # 0.0 to 1.0
    selectivity: float  # How selective the attention is
    sustainability: float  # How long attention can be maintained
    interference_resistance: float  # Resistance to distraction
    timestamp: float = field(default_factory=time.time)


@dataclass
class CognitiveTask:
    """Representation of a cognitive task"""
    task_id: str
    name: str
    priority: TaskPriority
    resource_requirements: Dict[ResourceType, float]
    estimated_duration: float
    deadline: Optional[float] = None
    dependencies: List[str] = field(default_factory=list)
    subtasks: List['CognitiveTask'] = field(default_factory=list)
    status: str = "pending"  # pending, running, completed, failed
    progress: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionPlan:
    """Plan for task execution"""
    plan_id: str
    tasks: List[CognitiveTask]
    resource_allocation: Dict[ResourceType, float]
    execution_order: List[str]
    estimated_completion_time: float
    contingency_plans: List['ExecutionPlan'] = field(default_factory=list)
    success_probability: float = 1.0


@dataclass
class ResourceState:
    """Current state of cognitive resources"""
    resource_type: ResourceType
    available_capacity: float
    allocated_capacity: float
    utilization_rate: float
    efficiency: float
    fatigue_level: float = 0.0
    recovery_rate: float = 0.1


class ExecutiveController:
    """
    Central executive control system for cognitive architecture
    
    Features:
    - Attention management and selective focus control
    - Resource allocation and load balancing
    - Task scheduling and priority management
    - Goal-directed behavior coordination
    - Conflict monitoring and resolution
    - Meta-cognitive monitoring and control
    - Adaptive control based on performance feedback
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        
        # Core components
        self.attention_manager = AttentionManager(self.config.get('attention', {}))
        self.resource_manager = ResourceManager(self.config.get('resources', {}))
        self.task_scheduler = TaskScheduler(self.config.get('scheduling', {}))
        
        # Executive state
        self.active_goals = []
        self.task_queue = []
        self.execution_history = deque(maxlen=self.config['max_history'])
        self.conflict_monitor = ConflictMonitor()
        
        # Meta-cognitive monitoring
        self.performance_monitor = PerformanceMonitor()
        self.cognitive_load = 0.0
        self.control_signals = deque(maxlen=100)
        
        # Performance statistics
        self.stats = {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'average_completion_time': 0.0,
            'resource_efficiency': {},
            'attention_switches': 0,
            'conflict_resolutions': 0,
            'control_interventions': 0
        }
        
        print("Executive control system initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for executive controller"""
        return {
            'max_concurrent_tasks': 5,
            'max_history': 1000,
            'attention_threshold': 0.3,
            'resource_reallocation_threshold': 0.8,
            'conflict_detection_sensitivity': 0.7,
            'meta_cognitive_monitoring': True,
            'adaptive_control': True,
            'performance_feedback_weight': 0.3,
            'default_time_horizon': 10.0,
            'attention': {
                'default_focus_duration': 5.0,
                'attention_switching_cost': 0.1,
                'sustained_attention_decay': 0.05
            },
            'resources': {
                'total_capacity': {
                    ResourceType.ATTENTION: 1.0,
                    ResourceType.WORKING_MEMORY: 1.0,
                    ResourceType.PROCESSING_POWER: 1.0,
                    ResourceType.REASONING: 1.0,
                    ResourceType.PERCEPTION: 1.0,
                    ResourceType.MOTOR: 1.0
                }
            },
            'scheduling': {
                'scheduling_algorithm': 'priority_based',
                'preemption_allowed': True,
                'load_balancing': True
            }
        }
    
    def set_goal(self, goal: Dict[str, Any]) -> str:
        """Set a high-level cognitive goal"""
        goal_id = f"goal_{len(self.active_goals)}"
        goal['id'] = goal_id
        goal['timestamp'] = time.time()
        goal['status'] = 'active'
        
        self.active_goals.append(goal)
        
        # Generate tasks to achieve the goal
        tasks = self._decompose_goal_to_tasks(goal)
        
        # Add tasks to scheduler
        for task in tasks:
            self.task_scheduler.add_task(task)
        
        return goal_id
    
    def execute_cognitive_cycle(self) -> Dict[str, Any]:
        """Execute one cycle of cognitive processing"""
        cycle_start = time.time()
        
        # Step 1: Monitor current state
        current_state = self._assess_cognitive_state()
        
        # Step 2: Detect conflicts and issues
        conflicts = self.conflict_monitor.detect_conflicts(current_state)
        
        # Step 3: Update attention based on current priorities
        self._update_attention_state()
        
        # Step 4: Reallocate resources if needed
        if self._should_reallocate_resources(current_state):
            self._reallocate_resources()
        
        # Step 5: Schedule and execute tasks
        execution_results = self._execute_scheduled_tasks()
        
        # Step 6: Apply executive control
        control_actions = self._apply_executive_control(conflicts, current_state)
        
        # Step 7: Update performance monitoring
        self.performance_monitor.update(execution_results)
        
        # Step 8: Meta-cognitive assessment
        meta_assessment = self._meta_cognitive_assessment(current_state, execution_results)
        
        cycle_time = (time.time() - cycle_start) * 1000
        
        cycle_summary = {
            'cycle_duration_ms': cycle_time,
            'cognitive_state': current_state,
            'conflicts_detected': len(conflicts),
            'tasks_executed': len(execution_results.get('completed_tasks', [])),
            'control_actions': control_actions,
            'meta_assessment': meta_assessment,
            'resource_utilization': self.resource_manager.get_utilization_summary(),
            'attention_state': self.attention_manager.get_current_state()
        }
        
        return cycle_summary
    
    def _decompose_goal_to_tasks(self, goal: Dict[str, Any]) -> List[CognitiveTask]:
        """Decompose high-level goal into executable tasks"""
        tasks = []
        
        goal_type = goal.get('type', 'general')
        
        if goal_type == 'reasoning':
            tasks.extend(self._create_reasoning_tasks(goal))
        elif goal_type == 'learning':
            tasks.extend(self._create_learning_tasks(goal))
        elif goal_type == 'perception':
            tasks.extend(self._create_perception_tasks(goal))
        elif goal_type == 'problem_solving':
            tasks.extend(self._create_problem_solving_tasks(goal))
        else:
            # Generic task decomposition
            task = CognitiveTask(
                task_id=f"task_{goal['id']}",
                name=f"Execute goal: {goal.get('description', 'Unknown')}",
                priority=TaskPriority.NORMAL,
                resource_requirements={
                    ResourceType.ATTENTION: 0.5,
                    ResourceType.WORKING_MEMORY: 0.3,
                    ResourceType.PROCESSING_POWER: 0.4
                },
                estimated_duration=goal.get('estimated_duration', 5.0),
                deadline=goal.get('deadline')
            )
            tasks.append(task)
        
        return tasks
    
    def _create_reasoning_tasks(self, goal: Dict[str, Any]) -> List[CognitiveTask]:
        """Create tasks for reasoning goals"""
        tasks = []
        
        # Analysis task
        analysis_task = CognitiveTask(
            task_id=f"analyze_{goal['id']}",
            name="Analyze reasoning problem",
            priority=TaskPriority.HIGH,
            resource_requirements={
                ResourceType.REASONING: 0.8,
                ResourceType.WORKING_MEMORY: 0.6,
                ResourceType.ATTENTION: 0.7
            },
            estimated_duration=2.0
        )
        tasks.append(analysis_task)
        
        # Inference task
        inference_task = CognitiveTask(
            task_id=f"infer_{goal['id']}",
            name="Perform logical inference",
            priority=TaskPriority.HIGH,
            resource_requirements={
                ResourceType.REASONING: 0.9,
                ResourceType.WORKING_MEMORY: 0.5,
                ResourceType.PROCESSING_POWER: 0.6
            },
            estimated_duration=3.0,
            dependencies=[analysis_task.task_id]
        )
        tasks.append(inference_task)
        
        # Validation task
        validation_task = CognitiveTask(
            task_id=f"validate_{goal['id']}",
            name="Validate reasoning results",
            priority=TaskPriority.NORMAL,
            resource_requirements={
                ResourceType.REASONING: 0.6,
                ResourceType.ATTENTION: 0.5
            },
            estimated_duration=1.0,
            dependencies=[inference_task.task_id]
        )
        tasks.append(validation_task)
        
        return tasks
    
    def _create_learning_tasks(self, goal: Dict[str, Any]) -> List[CognitiveTask]:
        """Create tasks for learning goals"""
        tasks = []
        
        # Information gathering
        gather_task = CognitiveTask(
            task_id=f"gather_{goal['id']}",
            name="Gather learning material",
            priority=TaskPriority.NORMAL,
            resource_requirements={
                ResourceType.PERCEPTION: 0.6,
                ResourceType.ATTENTION: 0.5,
                ResourceType.WORKING_MEMORY: 0.4
            },
            estimated_duration=2.0
        )
        tasks.append(gather_task)
        
        # Processing and integration
        process_task = CognitiveTask(
            task_id=f"process_{goal['id']}",
            name="Process and integrate information",
            priority=TaskPriority.HIGH,
            resource_requirements={
                ResourceType.PROCESSING_POWER: 0.8,
                ResourceType.WORKING_MEMORY: 0.7,
                ResourceType.REASONING: 0.6
            },
            estimated_duration=4.0,
            dependencies=[gather_task.task_id]
        )
        tasks.append(process_task)
        
        return tasks
    
    def _create_perception_tasks(self, goal: Dict[str, Any]) -> List[CognitiveTask]:
        """Create tasks for perception goals"""
        tasks = []
        
        # Sensory processing
        sense_task = CognitiveTask(
            task_id=f"sense_{goal['id']}",
            name="Process sensory input",
            priority=TaskPriority.HIGH,
            resource_requirements={
                ResourceType.PERCEPTION: 0.9,
                ResourceType.ATTENTION: 0.6,
                ResourceType.PROCESSING_POWER: 0.5
            },
            estimated_duration=1.0
        )
        tasks.append(sense_task)
        
        # Pattern recognition
        recognize_task = CognitiveTask(
            task_id=f"recognize_{goal['id']}",
            name="Recognize patterns",
            priority=TaskPriority.NORMAL,
            resource_requirements={
                ResourceType.PROCESSING_POWER: 0.7,
                ResourceType.WORKING_MEMORY: 0.5
            },
            estimated_duration=2.0,
            dependencies=[sense_task.task_id]
        )
        tasks.append(recognize_task)
        
        return tasks
    
    def _create_problem_solving_tasks(self, goal: Dict[str, Any]) -> List[CognitiveTask]:
        """Create tasks for problem-solving goals"""
        tasks = []
        
        # Problem analysis
        analyze_task = CognitiveTask(
            task_id=f"analyze_problem_{goal['id']}",
            name="Analyze problem structure",
            priority=TaskPriority.HIGH,
            resource_requirements={
                ResourceType.REASONING: 0.8,
                ResourceType.WORKING_MEMORY: 0.7,
                ResourceType.ATTENTION: 0.6
            },
            estimated_duration=2.0
        )
        tasks.append(analyze_task)
        
        # Solution generation
        generate_task = CognitiveTask(
            task_id=f"generate_solutions_{goal['id']}",
            name="Generate potential solutions",
            priority=TaskPriority.HIGH,
            resource_requirements={
                ResourceType.REASONING: 0.9,
                ResourceType.PROCESSING_POWER: 0.7,
                ResourceType.WORKING_MEMORY: 0.8
            },
            estimated_duration=3.0,
            dependencies=[analyze_task.task_id]
        )
        tasks.append(generate_task)
        
        # Solution evaluation
        evaluate_task = CognitiveTask(
            task_id=f"evaluate_solutions_{goal['id']}",
            name="Evaluate solution candidates",
            priority=TaskPriority.NORMAL,
            resource_requirements={
                ResourceType.REASONING: 0.7,
                ResourceType.WORKING_MEMORY: 0.5,
                ResourceType.ATTENTION: 0.6
            },
            estimated_duration=2.0,
            dependencies=[generate_task.task_id]
        )
        tasks.append(evaluate_task)
        
        return tasks
    
    def _assess_cognitive_state(self) -> Dict[str, Any]:
        """Assess current cognitive state"""
        state = {
            'timestamp': time.time(),
            'cognitive_load': self._calculate_cognitive_load(),
            'attention_state': self.attention_manager.get_current_state(),
            'resource_utilization': self.resource_manager.get_utilization_summary(),
            'active_tasks': len(self.task_scheduler.get_active_tasks()),
            'goal_progress': self._assess_goal_progress(),
            'fatigue_level': self._assess_fatigue_level(),
            'performance_trend': self.performance_monitor.get_trend()
        }
        
        return state
    
    def _calculate_cognitive_load(self) -> float:
        """Calculate current cognitive load"""
        resource_load = sum(
            state.utilization_rate 
            for state in self.resource_manager.resource_states.values()
        ) / len(self.resource_manager.resource_states)
        
        task_load = min(1.0, len(self.task_scheduler.get_active_tasks()) / self.config['max_concurrent_tasks'])
        
        attention_load = self.attention_manager.get_current_state().intensity
        
        self.cognitive_load = (resource_load * 0.4 + task_load * 0.3 + attention_load * 0.3)
        
        return self.cognitive_load
    
    def _should_reallocate_resources(self, state: Dict[str, Any]) -> bool:
        """Determine if resource reallocation is needed"""
        # Check if any resource is over-utilized
        for resource_type, utilization in state['resource_utilization'].items():
            if utilization > self.config['resource_reallocation_threshold']:
                return True
        
        # Check if cognitive load is too high
        if state['cognitive_load'] > 0.9:
            return True
        
        # Check if performance is declining
        if state['performance_trend'] < -0.1:
            return True
        
        return False
    
    def _update_attention_state(self):
        """Update attentional focus based on current priorities"""
        # Get highest priority active tasks
        active_tasks = self.task_scheduler.get_active_tasks()
        
        if active_tasks:
            # Focus on highest priority task
            highest_priority_task = max(active_tasks, key=lambda t: t.priority.value)
            
            self.attention_manager.set_focus([highest_priority_task.task_id])
            self.stats['attention_switches'] += 1
        
        # Update sustained attention
        self.attention_manager.update_sustained_attention()
    
    def _reallocate_resources(self):
        """Reallocate cognitive resources based on current demands"""
        # Get current resource demands
        active_tasks = self.task_scheduler.get_active_tasks()
        
        total_demands = defaultdict(float)
        for task in active_tasks:
            for resource_type, demand in task.resource_requirements.items():
                total_demands[resource_type] += demand
        
        # Reallocate based on demand
        self.resource_manager.reallocate_resources(dict(total_demands))
    
    def _execute_scheduled_tasks(self) -> Dict[str, Any]:
        """Execute currently scheduled tasks"""
        return self.task_scheduler.execute_tasks()
    
    def _apply_executive_control(self, conflicts: List[Dict[str, Any]], 
                               state: Dict[str, Any]) -> List[str]:
        """Apply executive control based on conflicts and state"""
        control_actions = []
        
        # Resolve conflicts
        for conflict in conflicts:
            action = self._resolve_conflict(conflict)
            if action:
                control_actions.append(action)
                self.control_signals.append({
                    'signal': ControlSignal.MODULATE,
                    'target': conflict['source'],
                    'timestamp': time.time()
                })
        
        # Apply load balancing if needed
        if state['cognitive_load'] > 0.8:
            control_actions.append(self._apply_load_balancing())
        
        # Maintain attention if focus is degrading
        if state['attention_state'].intensity < self.config['attention_threshold']:
            control_actions.append(self._enhance_attention())
        
        self.stats['control_interventions'] += len(control_actions)
        
        return control_actions
    
    def _resolve_conflict(self, conflict: Dict[str, Any]) -> Optional[str]:
        """Resolve a specific cognitive conflict"""
        conflict_type = conflict.get('type', 'unknown')
        
        if conflict_type == 'resource_competition':
            # Prioritize higher priority task
            return "prioritize_high_priority_task"
        elif conflict_type == 'attention_split':
            # Focus attention on most important target
            return "focus_attention"
        elif conflict_type == 'goal_interference':
            # Temporarily suspend lower priority goal
            return "suspend_interfering_goal"
        else:
            return "apply_general_conflict_resolution"
    
    def _apply_load_balancing(self) -> str:
        """Apply load balancing to reduce cognitive load"""
        # Defer low priority tasks
        low_priority_tasks = [
            task for task in self.task_scheduler.get_active_tasks()
            if task.priority == TaskPriority.LOW
        ]
        
        for task in low_priority_tasks:
            self.task_scheduler.defer_task(task.task_id)
        
        return "load_balancing_applied"
    
    def _enhance_attention(self) -> str:
        """Enhance attentional focus"""
        self.attention_manager.increase_intensity(0.2)
        return "attention_enhanced"
    
    def _meta_cognitive_assessment(self, state: Dict[str, Any], 
                                 execution_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform meta-cognitive assessment of performance"""
        assessment = {
            'cognitive_efficiency': self._calculate_cognitive_efficiency(state),
            'goal_achievement_rate': self._calculate_goal_achievement_rate(),
            'resource_optimization': self._assess_resource_optimization(state),
            'adaptation_needed': self._assess_adaptation_needs(state, execution_results),
            'learning_opportunities': self._identify_learning_opportunities(execution_results)
        }
        
        return assessment
    
    def _calculate_cognitive_efficiency(self, state: Dict[str, Any]) -> float:
        """Calculate overall cognitive efficiency"""
        task_completion_rate = self.stats['tasks_completed'] / max(1, self.stats['tasks_completed'] + self.stats['tasks_failed'])
        resource_efficiency = np.mean(list(state['resource_utilization'].values()))
        load_balance = 1.0 - abs(state['cognitive_load'] - 0.7)  # Optimal load around 0.7
        
        efficiency = (task_completion_rate * 0.4 + resource_efficiency * 0.3 + load_balance * 0.3)
        
        return efficiency
    
    def _calculate_goal_achievement_rate(self) -> float:
        """Calculate rate of goal achievement"""
        if not self.active_goals:
            return 1.0
        
        completed_goals = len([g for g in self.active_goals if g['status'] == 'completed'])
        return completed_goals / len(self.active_goals)
    
    def _assess_resource_optimization(self, state: Dict[str, Any]) -> float:
        """Assess how well resources are being utilized"""
        utilizations = list(state['resource_utilization'].values())
        
        # Optimal utilization is around 0.7-0.8
        optimal_range = (0.7, 0.8)
        optimization_score = 0.0
        
        for util in utilizations:
            if optimal_range[0] <= util <= optimal_range[1]:
                optimization_score += 1.0
            else:
                # Penalty for being outside optimal range
                distance = min(abs(util - optimal_range[0]), abs(util - optimal_range[1]))
                optimization_score += max(0.0, 1.0 - distance)
        
        return optimization_score / len(utilizations)
    
    def _assess_adaptation_needs(self, state: Dict[str, Any], 
                               execution_results: Dict[str, Any]) -> List[str]:
        """Assess what adaptations might be needed"""
        adaptations = []
        
        if state['cognitive_load'] > 0.9:
            adaptations.append("reduce_task_complexity")
        
        if state['performance_trend'] < -0.2:
            adaptations.append("improve_strategy")
        
        if state['fatigue_level'] > 0.8:
            adaptations.append("schedule_rest")
        
        failed_tasks = execution_results.get('failed_tasks', [])
        if len(failed_tasks) > 2:
            adaptations.append("revise_task_allocation")
        
        return adaptations
    
    def _identify_learning_opportunities(self, execution_results: Dict[str, Any]) -> List[str]:
        """Identify opportunities for learning and improvement"""
        opportunities = []
        
        # Analyze task failures for learning opportunities
        failed_tasks = execution_results.get('failed_tasks', [])
        if failed_tasks:
            opportunities.append("learn_from_task_failures")
        
        # Check for repeated inefficiencies
        if self.performance_monitor.detect_repeated_inefficiencies():
            opportunities.append("optimize_recurring_processes")
        
        # Look for resource allocation improvements
        if self._detect_resource_allocation_issues():
            opportunities.append("improve_resource_allocation_strategy")
        
        return opportunities
    
    def _assess_goal_progress(self) -> float:
        """Assess progress toward active goals"""
        if not self.active_goals:
            return 1.0
        
        total_progress = sum(goal.get('progress', 0.0) for goal in self.active_goals)
        return total_progress / len(self.active_goals)
    
    def _assess_fatigue_level(self) -> float:
        """Assess current cognitive fatigue level"""
        # Combine resource fatigue levels
        resource_fatigue = np.mean([
            state.fatigue_level for state in self.resource_manager.resource_states.values()
        ])
        
        # Factor in sustained high load
        load_fatigue = max(0.0, self.cognitive_load - 0.8) * 0.5
        
        # Factor in time since last rest
        # (This would be implemented with actual timing in a real system)
        time_fatigue = 0.1
        
        total_fatigue = resource_fatigue * 0.5 + load_fatigue * 0.3 + time_fatigue * 0.2
        
        return min(1.0, total_fatigue)
    
    def _detect_resource_allocation_issues(self) -> bool:
        """Detect if there are resource allocation inefficiencies"""
        # Check for consistent over/under-utilization
        utilizations = [
            state.utilization_rate for state in self.resource_manager.resource_states.values()
        ]
        
        # If there's high variance in utilization, there might be allocation issues
        variance = np.var(utilizations)
        return variance > 0.2
    
    def get_executive_stats(self) -> Dict[str, Any]:
        """Get executive control statistics"""
        stats = self.stats.copy()
        
        # Add current state information
        stats['current_cognitive_load'] = self.cognitive_load
        stats['active_goals'] = len(self.active_goals)
        stats['active_tasks'] = len(self.task_scheduler.get_active_tasks())
        stats['attention_state'] = self.attention_manager.get_current_state()
        stats['resource_states'] = {
            rt.value: {
                'utilization': state.utilization_rate,
                'efficiency': state.efficiency,
                'fatigue': state.fatigue_level
            }
            for rt, state in self.resource_manager.resource_states.items()
        }
        
        return stats


class AttentionManager:
    """Manages attentional processes and selective focus"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.current_state = AttentionState(
            focus_targets=[],
            attention_type=AttentionType.SELECTIVE,
            intensity=0.5,
            selectivity=0.7,
            sustainability=1.0,
            interference_resistance=0.6
        )
        
        self.focus_history = deque(maxlen=100)
    
    def set_focus(self, targets: List[str], attention_type: AttentionType = AttentionType.FOCUSED):
        """Set attentional focus on specific targets"""
        self.current_state.focus_targets = targets
        self.current_state.attention_type = attention_type
        self.current_state.timestamp = time.time()
        
        self.focus_history.append({
            'targets': targets.copy(),
            'timestamp': time.time(),
            'type': attention_type
        })
    
    def increase_intensity(self, amount: float):
        """Increase attention intensity"""
        self.current_state.intensity = min(1.0, self.current_state.intensity + amount)
    
    def update_sustained_attention(self):
        """Update sustained attention based on time and fatigue"""
        decay_rate = self.config.get('sustained_attention_decay', 0.05)
        self.current_state.sustainability = max(0.1, self.current_state.sustainability - decay_rate)
    
    def get_current_state(self) -> AttentionState:
        """Get current attention state"""
        return self.current_state


class ResourceManager:
    """Manages cognitive resource allocation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.resource_states = {}
        
        # Initialize resource states
        total_capacity = config.get('total_capacity', {})
        for resource_type in ResourceType:
            capacity = total_capacity.get(resource_type, 1.0)
            self.resource_states[resource_type] = ResourceState(
                resource_type=resource_type,
                available_capacity=capacity,
                allocated_capacity=0.0,
                utilization_rate=0.0,
                efficiency=1.0
            )
    
    def allocate_resources(self, requirements: Dict[ResourceType, float]) -> bool:
        """Allocate resources for a task"""
        # Check if resources are available
        for resource_type, amount in requirements.items():
            if resource_type in self.resource_states:
                state = self.resource_states[resource_type]
                if state.available_capacity < amount:
                    return False
        
        # Allocate resources
        for resource_type, amount in requirements.items():
            if resource_type in self.resource_states:
                state = self.resource_states[resource_type]
                state.allocated_capacity += amount
                state.available_capacity -= amount
                state.utilization_rate = state.allocated_capacity / (state.allocated_capacity + state.available_capacity)
        
        return True
    
    def deallocate_resources(self, requirements: Dict[ResourceType, float]):
        """Deallocate resources after task completion"""
        for resource_type, amount in requirements.items():
            if resource_type in self.resource_states:
                state = self.resource_states[resource_type]
                state.allocated_capacity = max(0.0, state.allocated_capacity - amount)
                state.available_capacity += amount
                total_capacity = state.allocated_capacity + state.available_capacity
                state.utilization_rate = state.allocated_capacity / total_capacity if total_capacity > 0 else 0.0
    
    def reallocate_resources(self, new_demands: Dict[ResourceType, float]):
        """Reallocate resources based on new demands"""
        # Simplified reallocation - in practice would be more sophisticated
        for resource_type, demand in new_demands.items():
            if resource_type in self.resource_states:
                state = self.resource_states[resource_type]
                total = state.allocated_capacity + state.available_capacity
                
                # Try to meet demand
                target_allocation = min(demand, total * 0.9)  # Leave 10% buffer
                
                state.allocated_capacity = target_allocation
                state.available_capacity = total - target_allocation
                state.utilization_rate = target_allocation / total if total > 0 else 0.0
    
    def get_utilization_summary(self) -> Dict[str, float]:
        """Get summary of resource utilization"""
        return {
            resource_type.value: state.utilization_rate 
            for resource_type, state in self.resource_states.items()
        }


class TaskScheduler:
    """Schedules and manages cognitive task execution"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.task_queue = []
        self.active_tasks = []
        self.completed_tasks = []
        self.failed_tasks = []
    
    def add_task(self, task: CognitiveTask):
        """Add a task to the scheduler"""
        heapq.heappush(self.task_queue, (-task.priority.value, time.time(), task))
    
    def get_active_tasks(self) -> List[CognitiveTask]:
        """Get currently active tasks"""
        return self.active_tasks.copy()
    
    def execute_tasks(self) -> Dict[str, Any]:
        """Execute scheduled tasks"""
        completed = []
        failed = []
        
        # Move tasks from queue to active if there's capacity
        max_concurrent = 3  # Simplified
        
        while (len(self.active_tasks) < max_concurrent and 
               self.task_queue and 
               self._can_start_task(self.task_queue[0][2])):
            
            _, _, task = heapq.heappop(self.task_queue)
            task.status = "running"
            self.active_tasks.append(task)
        
        # Simulate task execution
        for task in self.active_tasks[:]:
            task.progress += 0.3  # Simplified progress
            
            if task.progress >= 1.0:
                task.status = "completed"
                task.progress = 1.0
                completed.append(task)
                self.active_tasks.remove(task)
                self.completed_tasks.append(task)
            elif np.random.random() < 0.05:  # 5% failure rate
                task.status = "failed"
                failed.append(task)
                self.active_tasks.remove(task)
                self.failed_tasks.append(task)
        
        return {
            'completed_tasks': completed,
            'failed_tasks': failed,
            'active_tasks': self.active_tasks.copy()
        }
    
    def _can_start_task(self, task: CognitiveTask) -> bool:
        """Check if a task can be started"""
        # Check dependencies
        for dep_id in task.dependencies:
            if not any(t.task_id == dep_id and t.status == "completed" 
                      for t in self.completed_tasks):
                return False
        
        return True
    
    def defer_task(self, task_id: str):
        """Defer a task to reduce load"""
        for task in self.active_tasks:
            if task.task_id == task_id:
                task.status = "pending"
                self.active_tasks.remove(task)
                self.add_task(task)
                break


class ConflictMonitor:
    """Monitors for cognitive conflicts and interference"""
    
    def __init__(self):
        self.conflict_history = deque(maxlen=100)
    
    def detect_conflicts(self, cognitive_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect cognitive conflicts in current state"""
        conflicts = []
        
        # Resource competition conflicts
        if cognitive_state['cognitive_load'] > 0.9:
            conflicts.append({
                'type': 'resource_competition',
                'severity': cognitive_state['cognitive_load'],
                'source': 'resource_manager',
                'description': 'High cognitive load causing resource competition'
            })
        
        # Attention splitting conflicts
        attention_targets = len(cognitive_state['attention_state'].focus_targets)
        if attention_targets > 3:
            conflicts.append({
                'type': 'attention_split',
                'severity': attention_targets / 5.0,
                'source': 'attention_manager',
                'description': 'Attention split across too many targets'
            })
        
        # Store conflicts in history
        for conflict in conflicts:
            conflict['timestamp'] = time.time()
            self.conflict_history.append(conflict)
        
        return conflicts


class PerformanceMonitor:
    """Monitors cognitive performance and trends"""
    
    def __init__(self):
        self.performance_history = deque(maxlen=200)
        self.efficiency_trend = deque(maxlen=50)
    
    def update(self, execution_results: Dict[str, Any]):
        """Update performance metrics"""
        completed = len(execution_results.get('completed_tasks', []))
        failed = len(execution_results.get('failed_tasks', []))
        total = completed + failed
        
        if total > 0:
            success_rate = completed / total
            self.performance_history.append({
                'timestamp': time.time(),
                'success_rate': success_rate,
                'completed_tasks': completed,
                'failed_tasks': failed
            })
        
        # Update efficiency trend
        if len(self.performance_history) >= 2:
            recent_performance = np.mean([
                p['success_rate'] for p in list(self.performance_history)[-10:]
            ])
            self.efficiency_trend.append(recent_performance)
    
    def get_trend(self) -> float:
        """Get performance trend (-1 to 1)"""
        if len(self.efficiency_trend) < 5:
            return 0.0
        
        recent = np.mean(list(self.efficiency_trend)[-5:])
        older = np.mean(list(self.efficiency_trend)[:-5]) if len(self.efficiency_trend) > 5 else recent
        
        return (recent - older) / max(older, 0.1)
    
    def detect_repeated_inefficiencies(self) -> bool:
        """Detect if there are repeated inefficiencies"""
        if len(self.performance_history) < 10:
            return False
        
        recent_failures = sum(
            p['failed_tasks'] for p in list(self.performance_history)[-10:]
        )
        
        return recent_failures > 5