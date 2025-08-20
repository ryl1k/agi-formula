"""
Working Memory Management System for AGI-Formula Cognitive Architecture

Advanced working memory implementation featuring:
- Multi-component working memory model (Baddeley & Hitch)
- Dynamic capacity allocation and load management
- Memory consolidation and long-term storage integration
- Interference management and decay processes
- Attention-based memory refreshing and maintenance
- Context-dependent memory retrieval and binding
"""

import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import deque, defaultdict
import heapq
from abc import ABC, abstractmethod


class MemoryType(Enum):
    """Types of working memory components"""
    PHONOLOGICAL_LOOP = "phonological"      # Verbal/auditory information
    VISUOSPATIAL_SKETCHPAD = "visuospatial" # Visual/spatial information
    EPISODIC_BUFFER = "episodic"             # Integrated episodes
    CENTRAL_EXECUTIVE = "executive"          # Control and coordination
    SEMANTIC_BUFFER = "semantic"             # Semantic knowledge access


class MemoryPriority(Enum):
    """Priority levels for memory chunks"""
    CRITICAL = 4    # Critical information (current goals, etc.)
    HIGH = 3        # High priority (active tasks)
    NORMAL = 2      # Normal priority (context information)
    LOW = 1         # Low priority (background information)
    MINIMAL = 0     # Minimal priority (cached data)


class ConsolidationState(Enum):
    """States of memory consolidation"""
    ACTIVE = "active"           # Currently active in working memory
    CONSOLIDATING = "consolidating"  # Being consolidated
    CONSOLIDATED = "consolidated"    # Moved to long-term memory
    FORGOTTEN = "forgotten"     # Decayed and lost


class BindingType(Enum):
    """Types of memory binding"""
    TEMPORAL = "temporal"       # Temporal binding
    SPATIAL = "spatial"         # Spatial binding
    FEATURE = "feature"         # Feature binding
    CONCEPTUAL = "conceptual"   # Conceptual binding
    EPISODIC = "episodic"       # Episodic binding


@dataclass
class MemoryChunk:
    """Individual chunk of information in working memory"""
    chunk_id: str
    content: Any
    memory_type: MemoryType
    priority: MemoryPriority
    activation_level: float  # 0.0 to 1.0
    creation_time: float
    last_accessed: float
    access_count: int = 0
    decay_rate: float = 0.05
    consolidation_state: ConsolidationState = ConsolidationState.ACTIVE
    bindings: Dict[BindingType, List[str]] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    retrieval_cues: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        if not hasattr(self, 'last_accessed'):
            self.last_accessed = self.creation_time
    
    def get_current_activation(self) -> float:
        """Get current activation level considering decay"""
        time_since_access = time.time() - self.last_accessed
        decayed_activation = self.activation_level * np.exp(-self.decay_rate * time_since_access)
        return max(0.0, decayed_activation)
    
    def refresh(self, boost: float = 0.2):
        """Refresh memory chunk activation"""
        self.activation_level = min(1.0, self.activation_level + boost)
        self.last_accessed = time.time()
        self.access_count += 1


@dataclass
class MemoryBuffer:
    """Buffer for a specific type of working memory"""
    buffer_type: MemoryType
    capacity: int
    chunks: List[MemoryChunk] = field(default_factory=list)
    interference_threshold: float = 0.7
    rehearsal_rate: float = 0.1
    
    def add_chunk(self, chunk: MemoryChunk) -> bool:
        """Add chunk to buffer, handling capacity constraints"""
        if len(self.chunks) >= self.capacity:
            # Remove least activated chunk
            self._evict_chunk()
        
        self.chunks.append(chunk)
        return True
    
    def _evict_chunk(self):
        """Evict least activated chunk from buffer"""
        if not self.chunks:
            return
        
        # Find chunk with lowest current activation
        min_activation_chunk = min(self.chunks, key=lambda c: c.get_current_activation())
        
        # Mark for consolidation if above threshold
        if min_activation_chunk.get_current_activation() > 0.3:
            min_activation_chunk.consolidation_state = ConsolidationState.CONSOLIDATING
        else:
            min_activation_chunk.consolidation_state = ConsolidationState.FORGOTTEN
        
        self.chunks.remove(min_activation_chunk)
    
    def get_active_chunks(self, threshold: float = 0.1) -> List[MemoryChunk]:
        """Get chunks above activation threshold"""
        return [chunk for chunk in self.chunks if chunk.get_current_activation() > threshold]
    
    def rehearse(self):
        """Perform rehearsal to maintain activation"""
        for chunk in self.chunks:
            if chunk.get_current_activation() > 0.2:  # Only rehearse active chunks
                chunk.refresh(self.rehearsal_rate)


class WorkingMemoryManager:
    """
    Comprehensive working memory management system
    
    Features:
    - Multi-component working memory architecture
    - Dynamic capacity management and interference control
    - Memory consolidation and long-term integration
    - Attention-based maintenance and rehearsal
    - Context-dependent retrieval and binding
    - Cognitive load monitoring and optimization
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        
        # Initialize memory buffers
        self.buffers = {}
        for memory_type in MemoryType:
            capacity = self.config['buffer_capacities'].get(memory_type.value, 7)
            self.buffers[memory_type] = MemoryBuffer(
                buffer_type=memory_type,
                capacity=capacity,
                interference_threshold=self.config.get('interference_threshold', 0.7),
                rehearsal_rate=self.config.get('rehearsal_rate', 0.1)
            )
        
        # Memory management components
        self.consolidation_manager = ConsolidationManager(self.config.get('consolidation', {}))
        self.retrieval_manager = RetrievalManager(self.config.get('retrieval', {}))
        self.binding_manager = BindingManager()
        
        # Memory state tracking
        self.total_chunks = 0
        self.retrieval_history = deque(maxlen=1000)
        self.consolidation_queue = []
        
        # Performance monitoring
        self.stats = {
            'chunks_created': 0,
            'chunks_retrieved': 0,
            'chunks_consolidated': 0,
            'chunks_forgotten': 0,
            'retrieval_accuracy': [],
            'consolidation_success_rate': [],
            'interference_events': 0,
            'rehearsal_cycles': 0
        }
        
        print("Working memory management system initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for working memory"""
        return {
            'buffer_capacities': {
                'phonological': 7,      # Miller's 7±2 for phonological loop
                'visuospatial': 4,      # Typical visuospatial capacity
                'episodic': 5,          # Episodic buffer capacity
                'executive': 3,         # Central executive capacity
                'semantic': 10          # Semantic buffer capacity
            },
            'default_decay_rate': 0.05,
            'interference_threshold': 0.7,
            'consolidation_threshold': 0.6,
            'retrieval_threshold': 0.3,
            'rehearsal_rate': 0.1,
            'attention_boost': 0.3,
            'binding_strength': 0.8,
            'consolidation': {
                'consolidation_delay': 2.0,
                'success_threshold': 0.7,
                'batch_size': 5
            },
            'retrieval': {
                'cue_matching_threshold': 0.5,
                'context_weight': 0.3,
                'recency_weight': 0.2,
                'frequency_weight': 0.5
            }
        }
    
    def store_information(self, content: Any, memory_type: MemoryType, 
                         priority: MemoryPriority = MemoryPriority.NORMAL,
                         context: Optional[Dict[str, Any]] = None,
                         retrieval_cues: Optional[Set[str]] = None) -> str:
        """Store new information in working memory"""
        chunk_id = f"chunk_{self.total_chunks}_{int(time.time() * 1000)}"
        
        chunk = MemoryChunk(
            chunk_id=chunk_id,
            content=content,
            memory_type=memory_type,
            priority=priority,
            activation_level=1.0,  # Start fully activated
            creation_time=time.time(),
            last_accessed=time.time(),
            decay_rate=self.config['default_decay_rate'],
            context=context or {},
            retrieval_cues=retrieval_cues or set()
        )
        
        # Store in appropriate buffer
        buffer = self.buffers[memory_type]
        success = buffer.add_chunk(chunk)
        
        if success:
            self.total_chunks += 1
            self.stats['chunks_created'] += 1
            
            # Create automatic bindings
            self._create_automatic_bindings(chunk)
            
            return chunk_id
        
        return ""
    
    def retrieve_information(self, retrieval_cues: Set[str], 
                           memory_type: Optional[MemoryType] = None,
                           context: Optional[Dict[str, Any]] = None) -> List[MemoryChunk]:
        """Retrieve information from working memory"""
        start_time = time.time()
        
        retrieved_chunks = self.retrieval_manager.retrieve(
            retrieval_cues, self.buffers, memory_type, context
        )
        
        # Boost activation of retrieved chunks
        for chunk in retrieved_chunks:
            chunk.refresh(self.config['attention_boost'])
        
        # Record retrieval
        self.retrieval_history.append({
            'cues': retrieval_cues,
            'memory_type': memory_type,
            'context': context,
            'results': len(retrieved_chunks),
            'timestamp': time.time(),
            'retrieval_time': (time.time() - start_time) * 1000
        })
        
        self.stats['chunks_retrieved'] += len(retrieved_chunks)
        
        return retrieved_chunks
    
    def update_working_memory(self):
        """Perform working memory maintenance cycle"""
        # Rehearsal for all buffers
        for buffer in self.buffers.values():
            buffer.rehearse()
        
        self.stats['rehearsal_cycles'] += 1
        
        # Check for consolidation candidates
        consolidation_candidates = self._identify_consolidation_candidates()
        
        for chunk in consolidation_candidates:
            self.consolidation_queue.append(chunk)
        
        # Process consolidation queue
        self._process_consolidation_queue()
        
        # Detect and handle interference
        self._detect_and_handle_interference()
        
        # Garbage collection for forgotten chunks
        self._garbage_collect_forgotten_chunks()
    
    def bind_chunks(self, chunk_ids: List[str], binding_type: BindingType, 
                   binding_strength: float = None) -> bool:
        """Create binding between memory chunks"""
        if binding_strength is None:
            binding_strength = self.config['binding_strength']
        
        return self.binding_manager.create_binding(
            chunk_ids, binding_type, binding_strength, self.buffers
        )
    
    def get_memory_state(self) -> Dict[str, Any]:
        """Get comprehensive working memory state"""
        state = {
            'timestamp': time.time(),
            'total_chunks': sum(len(buffer.chunks) for buffer in self.buffers.values()),
            'buffer_states': {},
            'consolidation_queue_size': len(self.consolidation_queue),
            'recent_retrievals': len(self.retrieval_history)
        }
        
        for memory_type, buffer in self.buffers.items():
            active_chunks = buffer.get_active_chunks()
            state['buffer_states'][memory_type.value] = {
                'capacity': buffer.capacity,
                'current_chunks': len(buffer.chunks),
                'active_chunks': len(active_chunks),
                'utilization': len(buffer.chunks) / buffer.capacity,
                'average_activation': np.mean([c.get_current_activation() for c in active_chunks]) if active_chunks else 0.0
            }
        
        return state
    
    def _create_automatic_bindings(self, chunk: MemoryChunk):
        """Create automatic bindings for new chunks"""
        # Temporal binding with recent chunks
        recent_chunks = self._get_recent_chunks(time_window=2.0)
        if recent_chunks:
            temporal_chunk_ids = [c.chunk_id for c in recent_chunks[-3:]]  # Bind to 3 most recent
            temporal_chunk_ids.append(chunk.chunk_id)
            self.binding_manager.create_binding(
                temporal_chunk_ids, BindingType.TEMPORAL, 0.6, self.buffers
            )
        
        # Context-based binding
        if chunk.context:
            context_chunks = self._find_chunks_with_similar_context(chunk.context)
            if context_chunks:
                context_chunk_ids = [c.chunk_id for c in context_chunks[:2]]
                context_chunk_ids.append(chunk.chunk_id)
                self.binding_manager.create_binding(
                    context_chunk_ids, BindingType.CONCEPTUAL, 0.7, self.buffers
                )
    
    def _identify_consolidation_candidates(self) -> List[MemoryChunk]:
        """Identify chunks ready for consolidation"""
        candidates = []
        consolidation_threshold = self.config['consolidation_threshold']
        
        for buffer in self.buffers.values():
            for chunk in buffer.chunks:
                # Criteria for consolidation
                age = time.time() - chunk.creation_time
                activation = chunk.get_current_activation()
                access_frequency = chunk.access_count / max(1, age / 60)  # Access per minute
                
                consolidation_score = (
                    min(age / 60, 1.0) * 0.3 +  # Age factor (up to 1 minute)
                    activation * 0.4 +           # Current activation
                    min(access_frequency, 1.0) * 0.3  # Access frequency
                )
                
                if (consolidation_score > consolidation_threshold and 
                    chunk.consolidation_state == ConsolidationState.ACTIVE):
                    candidates.append(chunk)
        
        return candidates
    
    def _process_consolidation_queue(self):
        """Process chunks in consolidation queue"""
        batch_size = self.config['consolidation']['batch_size']
        
        while self.consolidation_queue and len(self.consolidation_queue) >= batch_size:
            batch = self.consolidation_queue[:batch_size]
            self.consolidation_queue = self.consolidation_queue[batch_size:]
            
            success = self.consolidation_manager.consolidate_batch(batch)
            
            if success:
                for chunk in batch:
                    chunk.consolidation_state = ConsolidationState.CONSOLIDATED
                self.stats['chunks_consolidated'] += len(batch)
                self.stats['consolidation_success_rate'].append(1.0)
            else:
                self.stats['consolidation_success_rate'].append(0.0)
    
    def _detect_and_handle_interference(self):
        """Detect and handle interference between memory chunks"""
        interference_threshold = self.config['interference_threshold']
        
        for buffer in self.buffers.values():
            chunks = buffer.get_active_chunks()
            
            # Check for content similarity (simplified)
            for i, chunk1 in enumerate(chunks):
                for chunk2 in chunks[i+1:]:
                    similarity = self._calculate_content_similarity(chunk1, chunk2)
                    
                    if similarity > interference_threshold:
                        self._handle_interference(chunk1, chunk2)
                        self.stats['interference_events'] += 1
    
    def _handle_interference(self, chunk1: MemoryChunk, chunk2: MemoryChunk):
        """Handle interference between two chunks"""
        # Keep chunk with higher priority and activation
        chunk1_score = chunk1.priority.value * chunk1.get_current_activation()
        chunk2_score = chunk2.priority.value * chunk2.get_current_activation()
        
        if chunk1_score < chunk2_score:
            # Reduce activation of chunk1
            chunk1.activation_level *= 0.7
        else:
            # Reduce activation of chunk2
            chunk2.activation_level *= 0.7
    
    def _calculate_content_similarity(self, chunk1: MemoryChunk, chunk2: MemoryChunk) -> float:
        """Calculate similarity between chunk contents (simplified)"""
        # This would be much more sophisticated in a real implementation
        if type(chunk1.content) != type(chunk2.content):
            return 0.0
        
        if isinstance(chunk1.content, str) and isinstance(chunk2.content, str):
            # Simple string similarity
            common_words = set(chunk1.content.lower().split()) & set(chunk2.content.lower().split())
            total_words = set(chunk1.content.lower().split()) | set(chunk2.content.lower().split())
            return len(common_words) / max(1, len(total_words))
        
        return 0.0  # Default for other types
    
    def _garbage_collect_forgotten_chunks(self):
        """Remove chunks marked as forgotten"""
        chunks_removed = 0
        
        for buffer in self.buffers.values():
            buffer.chunks = [
                chunk for chunk in buffer.chunks 
                if chunk.consolidation_state != ConsolidationState.FORGOTTEN
            ]
            
        self.stats['chunks_forgotten'] += chunks_removed
    
    def _get_recent_chunks(self, time_window: float) -> List[MemoryChunk]:
        """Get chunks created within time window"""
        cutoff_time = time.time() - time_window
        recent_chunks = []
        
        for buffer in self.buffers.values():
            for chunk in buffer.chunks:
                if chunk.creation_time >= cutoff_time:
                    recent_chunks.append(chunk)
        
        return sorted(recent_chunks, key=lambda c: c.creation_time)
    
    def _find_chunks_with_similar_context(self, context: Dict[str, Any]) -> List[MemoryChunk]:
        """Find chunks with similar context"""
        similar_chunks = []
        
        for buffer in self.buffers.values():
            for chunk in buffer.chunks:
                if chunk.context:
                    # Simple context similarity
                    common_keys = set(context.keys()) & set(chunk.context.keys())
                    if common_keys:
                        similarity = len(common_keys) / len(set(context.keys()) | set(chunk.context.keys()))
                        if similarity > 0.5:
                            similar_chunks.append(chunk)
        
        return similar_chunks
    
    def get_working_memory_stats(self) -> Dict[str, Any]:
        """Get working memory performance statistics"""
        stats = self.stats.copy()
        
        # Add derived statistics
        if self.stats['consolidation_success_rate']:
            stats['avg_consolidation_success'] = np.mean(self.stats['consolidation_success_rate'])
        
        if self.stats['retrieval_accuracy']:
            stats['avg_retrieval_accuracy'] = np.mean(self.stats['retrieval_accuracy'])
        
        # Current state statistics
        memory_state = self.get_memory_state()
        stats['current_memory_state'] = memory_state
        
        # Efficiency metrics
        total_capacity = sum(buffer.capacity for buffer in self.buffers.values())
        total_chunks = memory_state['total_chunks']
        stats['memory_efficiency'] = total_chunks / total_capacity if total_capacity > 0 else 0.0
        
        return stats


class RetrievalManager:
    """Manages memory retrieval processes"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def retrieve(self, retrieval_cues: Set[str], buffers: Dict[MemoryType, MemoryBuffer],
                memory_type: Optional[MemoryType] = None, 
                context: Optional[Dict[str, Any]] = None) -> List[MemoryChunk]:
        """Retrieve chunks matching retrieval cues"""
        candidates = []
        
        # Search in specified buffer or all buffers
        search_buffers = [buffers[memory_type]] if memory_type else list(buffers.values())
        
        for buffer in search_buffers:
            for chunk in buffer.get_active_chunks(self.config['retrieval_threshold']):
                match_score = self._calculate_match_score(chunk, retrieval_cues, context)
                if match_score > self.config['cue_matching_threshold']:
                    candidates.append((chunk, match_score))
        
        # Sort by match score and return top candidates
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return [chunk for chunk, score in candidates[:10]]  # Top 10 matches
    
    def _calculate_match_score(self, chunk: MemoryChunk, retrieval_cues: Set[str], 
                              context: Optional[Dict[str, Any]]) -> float:
        """Calculate how well a chunk matches retrieval cues"""
        score = 0.0
        
        # Cue matching
        chunk_cues = chunk.retrieval_cues
        if chunk_cues:
            cue_overlap = len(retrieval_cues & chunk_cues)
            cue_score = cue_overlap / max(1, len(retrieval_cues | chunk_cues))
            score += cue_score * 0.5
        
        # Context matching
        if context and chunk.context:
            context_similarity = self._calculate_context_similarity(chunk.context, context)
            score += context_similarity * self.config['context_weight']
        
        # Recency and frequency
        age = time.time() - chunk.creation_time
        recency_score = np.exp(-age / 60)  # Decay over minutes
        frequency_score = min(1.0, chunk.access_count / 10)
        
        score += recency_score * self.config['recency_weight']
        score += frequency_score * self.config['frequency_weight']
        
        return score
    
    def _calculate_context_similarity(self, context1: Dict[str, Any], 
                                    context2: Dict[str, Any]) -> float:
        """Calculate similarity between contexts"""
        common_keys = set(context1.keys()) & set(context2.keys())
        if not common_keys:
            return 0.0
        
        similarity = 0.0
        for key in common_keys:
            if context1[key] == context2[key]:
                similarity += 1.0
            elif isinstance(context1[key], (int, float)) and isinstance(context2[key], (int, float)):
                # Numeric similarity
                max_val = max(abs(context1[key]), abs(context2[key]))
                if max_val > 0:
                    similarity += 1.0 - abs(context1[key] - context2[key]) / max_val
        
        return similarity / len(common_keys)


class ConsolidationManager:
    """Manages memory consolidation to long-term storage"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.consolidation_history = deque(maxlen=1000)
    
    def consolidate_batch(self, chunks: List[MemoryChunk]) -> bool:
        """Consolidate a batch of memory chunks"""
        # Simplified consolidation process
        success_threshold = self.config['success_threshold']
        
        consolidation_scores = []
        for chunk in chunks:
            # Calculate consolidation success probability
            score = self._calculate_consolidation_score(chunk)
            consolidation_scores.append(score)
        
        avg_score = np.mean(consolidation_scores)
        success = avg_score > success_threshold
        
        # Record consolidation attempt
        self.consolidation_history.append({
            'chunks': [c.chunk_id for c in chunks],
            'success': success,
            'avg_score': avg_score,
            'timestamp': time.time()
        })
        
        return success
    
    def _calculate_consolidation_score(self, chunk: MemoryChunk) -> float:
        """Calculate consolidation success score for a chunk"""
        # Factors affecting consolidation success
        age_factor = min(1.0, (time.time() - chunk.creation_time) / 300)  # 5 minutes max
        activation_factor = chunk.get_current_activation()
        frequency_factor = min(1.0, chunk.access_count / 5)
        priority_factor = chunk.priority.value / 4.0
        
        score = (age_factor * 0.2 + activation_factor * 0.4 + 
                frequency_factor * 0.3 + priority_factor * 0.1)
        
        return score


class BindingManager:
    """Manages bindings between memory chunks"""
    
    def __init__(self):
        self.bindings = defaultdict(list)  # binding_type -> list of bindings
    
    def create_binding(self, chunk_ids: List[str], binding_type: BindingType, 
                      strength: float, buffers: Dict[MemoryType, MemoryBuffer]) -> bool:
        """Create binding between chunks"""
        if len(chunk_ids) < 2:
            return False
        
        # Find chunks in buffers
        chunks = []
        for buffer in buffers.values():
            for chunk in buffer.chunks:
                if chunk.chunk_id in chunk_ids:
                    chunks.append(chunk)
        
        if len(chunks) != len(chunk_ids):
            return False  # Not all chunks found
        
        # Create bidirectional bindings
        for chunk in chunks:
            other_chunk_ids = [c.chunk_id for c in chunks if c.chunk_id != chunk.chunk_id]
            if binding_type not in chunk.bindings:
                chunk.bindings[binding_type] = []
            chunk.bindings[binding_type].extend(other_chunk_ids)
        
        # Record binding
        binding_record = {
            'chunk_ids': chunk_ids,
            'binding_type': binding_type,
            'strength': strength,
            'timestamp': time.time()
        }
        self.bindings[binding_type].append(binding_record)
        
        return True
    
    def get_bound_chunks(self, chunk_id: str, binding_type: BindingType, 
                        buffers: Dict[MemoryType, MemoryBuffer]) -> List[MemoryChunk]:
        """Get chunks bound to a specific chunk"""
        # Find the chunk
        target_chunk = None
        for buffer in buffers.values():
            for chunk in buffer.chunks:
                if chunk.chunk_id == chunk_id:
                    target_chunk = chunk
                    break
            if target_chunk:
                break
        
        if not target_chunk or binding_type not in target_chunk.bindings:
            return []
        
        # Find bound chunks
        bound_chunk_ids = target_chunk.bindings[binding_type]
        bound_chunks = []
        
        for buffer in buffers.values():
            for chunk in buffer.chunks:
                if chunk.chunk_id in bound_chunk_ids:
                    bound_chunks.append(chunk)
        
        return bound_chunks