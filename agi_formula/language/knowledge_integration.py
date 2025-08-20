"""
Dynamic Knowledge Integration System

Real-time knowledge learning and integration from conversations:
- Dynamic knowledge graph construction
- Causal knowledge modeling
- Conflict resolution and consistency maintenance
- Conversational learning from user interactions
- Knowledge validation and verification
"""

import time
import uuid
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import logging
import json
import networkx as nx

from ..cognitive import (
    CognitiveArchitecture,
    MemoryType,
    MemoryPriority,
    BindingType
)
from ..reasoning import CausalReasoner


class KnowledgeType(Enum):
    """Types of knowledge"""
    FACTUAL = "factual"           # Factual information
    PROCEDURAL = "procedural"     # How-to knowledge
    CONCEPTUAL = "conceptual"     # Conceptual understanding
    CAUSAL = "causal"            # Causal relationships
    TEMPORAL = "temporal"        # Temporal knowledge
    CONDITIONAL = "conditional"   # If-then knowledge
    EXPERIENTIAL = "experiential" # Experience-based knowledge


class ConfidenceLevel(Enum):
    """Confidence levels for knowledge"""
    CERTAIN = 1.0
    HIGH = 0.8
    MEDIUM = 0.6
    LOW = 0.4
    UNCERTAIN = 0.2


class ConflictType(Enum):
    """Types of knowledge conflicts"""
    DIRECT_CONTRADICTION = "direct_contradiction"
    INCONSISTENT_IMPLICATIONS = "inconsistent_implications"
    TEMPORAL_INCONSISTENCY = "temporal_inconsistency"
    CAUSAL_CONTRADICTION = "causal_contradiction"
    CONFIDENCE_MISMATCH = "confidence_mismatch"


@dataclass
class KnowledgeItem:
    """Individual knowledge item"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    knowledge_type: KnowledgeType = KnowledgeType.FACTUAL
    confidence: float = 0.8
    source: str = "conversation"
    timestamp: float = field(default_factory=time.time)
    related_concepts: Set[str] = field(default_factory=set)
    causal_relations: List[Dict[str, Any]] = field(default_factory=list)
    temporal_context: Optional[Dict[str, Any]] = None
    evidence: List[str] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)
    validation_status: str = "pending"  # pending, validated, rejected
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConceptNode:
    """Node in the knowledge graph representing a concept"""
    id: str
    name: str
    definition: str
    concept_type: str = "general"
    properties: Dict[str, Any] = field(default_factory=dict)
    creation_time: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    confidence: float = 0.8
    related_knowledge: List[str] = field(default_factory=list)


@dataclass
class KnowledgeRelation:
    """Relationship between knowledge items or concepts"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""
    target_id: str = ""
    relation_type: str = "related_to"
    strength: float = 0.7
    confidence: float = 0.8
    evidence: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConversationalLearning:
    """Learns from user conversations in real-time"""
    
    def __init__(self, cognitive_architecture: Optional[CognitiveArchitecture] = None):
        self.cognitive_architecture = cognitive_architecture
        self.learning_patterns = {
            'teaching': [
                r"(.+) is (.+)",
                r"(.+) means (.+)",
                r"(.+) refers to (.+)",
                r"(.+) can be defined as (.+)"
            ],
            'causal': [
                r"(.+) causes (.+)",
                r"(.+) leads to (.+)",
                r"(.+) results in (.+)",
                r"because (.+), (.+)"
            ],
            'procedural': [
                r"to (.+), you (.+)",
                r"the way to (.+) is (.+)",
                r"(.+) by (.+)",
                r"first (.+), then (.+)"
            ],
            'temporal': [
                r"before (.+), (.+)",
                r"after (.+), (.+)",
                r"(.+) happens when (.+)",
                r"during (.+), (.+)"
            ]
        }
        
        self.learned_knowledge = []
        self.learning_confidence = defaultdict(float)
    
    def extract_knowledge_from_conversation(self, user_input: str, 
                                          conversation_context: List[Dict[str, str]]) -> List[KnowledgeItem]:
        """Extract new knowledge from user input"""
        extracted_knowledge = []
        
        # Check for explicit teaching patterns
        for knowledge_type, patterns in self.learning_patterns.items():
            for pattern in patterns:
                import re
                matches = re.findall(pattern, user_input.lower())
                
                for match in matches:
                    if isinstance(match, tuple) and len(match) >= 2:
                        knowledge_item = self._create_knowledge_item(
                            match, knowledge_type, user_input, conversation_context
                        )
                        extracted_knowledge.append(knowledge_item)
        
        # Extract conceptual knowledge from context
        conceptual_knowledge = self._extract_conceptual_knowledge(
            user_input, conversation_context
        )
        extracted_knowledge.extend(conceptual_knowledge)
        
        return extracted_knowledge
    
    def _create_knowledge_item(self, match: Tuple[str, ...], 
                             knowledge_type: str,
                             full_input: str,
                             context: List[Dict[str, str]]) -> KnowledgeItem:
        """Create knowledge item from pattern match"""
        
        if knowledge_type == 'teaching':
            subject, definition = match[0].strip(), match[1].strip()
            content = f"{subject} is {definition}"
            k_type = KnowledgeType.FACTUAL
            related_concepts = {subject, definition}
            
        elif knowledge_type == 'causal':
            cause, effect = match[0].strip(), match[1].strip()
            content = f"{cause} causes {effect}"
            k_type = KnowledgeType.CAUSAL
            related_concepts = {cause, effect}
            
        elif knowledge_type == 'procedural':
            goal, method = match[0].strip(), match[1].strip()
            content = f"To {goal}, {method}"
            k_type = KnowledgeType.PROCEDURAL
            related_concepts = {goal, method}
            
        elif knowledge_type == 'temporal':
            first, second = match[0].strip(), match[1].strip()
            content = f"Temporal relationship: {first} and {second}"
            k_type = KnowledgeType.TEMPORAL
            related_concepts = {first, second}
        
        else:
            content = " ".join(match)
            k_type = KnowledgeType.CONCEPTUAL
            related_concepts = set(match)
        
        # Assess confidence based on context and phrasing
        confidence = self._assess_learning_confidence(full_input, context, content)
        
        return KnowledgeItem(
            content=content,
            knowledge_type=k_type,
            confidence=confidence,
            source="user_teaching",
            related_concepts=related_concepts,
            evidence=[full_input],
            metadata={
                'conversation_context': len(context),
                'pattern_type': knowledge_type,
                'extracted_from': full_input[:100]
            }
        )
    
    def _extract_conceptual_knowledge(self, user_input: str,
                                    context: List[Dict[str, str]]) -> List[KnowledgeItem]:
        """Extract conceptual knowledge from input"""
        knowledge_items = []
        
        # Look for new concept introductions
        import re
        
        # Pattern: "A [concept] is a type of [category]"
        type_patterns = re.findall(r"([a-zA-Z\s]+) is a (?:type|kind) of ([a-zA-Z\s]+)", user_input)
        for concept, category in type_patterns:
            knowledge_item = KnowledgeItem(
                content=f"{concept.strip()} is a type of {category.strip()}",
                knowledge_type=KnowledgeType.CONCEPTUAL,
                confidence=0.9,
                source="user_teaching",
                related_concepts={concept.strip(), category.strip()},
                evidence=[user_input]
            )
            knowledge_items.append(knowledge_item)
        
        # Pattern: "X has property Y"
        property_patterns = re.findall(r"([a-zA-Z\s]+) has ([a-zA-Z\s]+)", user_input)
        for entity, property in property_patterns:
            knowledge_item = KnowledgeItem(
                content=f"{entity.strip()} has property: {property.strip()}",
                knowledge_type=KnowledgeType.FACTUAL,
                confidence=0.8,
                source="user_teaching",
                related_concepts={entity.strip(), property.strip()},
                evidence=[user_input]
            )
            knowledge_items.append(knowledge_item)
        
        return knowledge_items
    
    def _assess_learning_confidence(self, input_text: str, 
                                  context: List[Dict[str, str]],
                                  extracted_content: str) -> float:
        """Assess confidence in learned knowledge"""
        base_confidence = 0.7
        
        # Increase confidence for explicit teaching language
        teaching_indicators = ["is defined as", "means", "refers to", "can be understood as"]
        if any(indicator in input_text.lower() for indicator in teaching_indicators):
            base_confidence += 0.15
        
        # Increase confidence for authoritative language
        authority_indicators = ["the fact is", "it's established that", "research shows"]
        if any(indicator in input_text.lower() for indicator in authority_indicators):
            base_confidence += 0.1
        
        # Decrease confidence for uncertain language
        uncertainty_indicators = ["maybe", "perhaps", "might be", "could be", "i think"]
        if any(indicator in input_text.lower() for indicator in uncertainty_indicators):
            base_confidence -= 0.2
        
        # Adjust based on context length (more context = higher confidence)
        context_bonus = min(len(context) * 0.02, 0.1)
        base_confidence += context_bonus
        
        return max(0.1, min(1.0, base_confidence))
    
    def integrate_with_working_memory(self, knowledge_items: List[KnowledgeItem]):
        """Integrate learned knowledge with working memory"""
        if not self.cognitive_architecture:
            return
        
        for item in knowledge_items:
            # Store in working memory
            chunk_id = self.cognitive_architecture.working_memory.store_information(
                content=item.content,
                memory_type=MemoryType.SEMANTIC_BUFFER,
                priority=MemoryPriority.HIGH if item.confidence > 0.8 else MemoryPriority.NORMAL,
                context={
                    'knowledge_type': item.knowledge_type.value,
                    'confidence': item.confidence,
                    'source': item.source,
                    'learning_session': True
                },
                retrieval_cues=item.related_concepts.union({'learned_knowledge'})
            )
            
            # Add to consciousness if high confidence
            if (item.confidence > 0.8 and 
                hasattr(self.cognitive_architecture, 'consciousness') and 
                self.cognitive_architecture.consciousness):
                
                self.cognitive_architecture.consciousness.add_to_consciousness(
                    content=f"Learned new knowledge: {item.content}",
                    content_type="learning_event",
                    activation_strength=item.confidence,
                    phenomenal_properties={
                        'learning': True,
                        'confidence': item.confidence,
                        'valence': 0.6,  # Positive learning experience
                        'arousal': 0.7   # Engaging
                    }
                )


class KnowledgeConflictResolver:
    """Resolves conflicts between knowledge items"""
    
    def __init__(self):
        self.resolution_strategies = {
            ConflictType.DIRECT_CONTRADICTION: self._resolve_contradiction,
            ConflictType.INCONSISTENT_IMPLICATIONS: self._resolve_implications,
            ConflictType.TEMPORAL_INCONSISTENCY: self._resolve_temporal,
            ConflictType.CAUSAL_CONTRADICTION: self._resolve_causal,
            ConflictType.CONFIDENCE_MISMATCH: self._resolve_confidence
        }
        
        self.resolution_history = []
    
    def detect_conflicts(self, knowledge_items: List[KnowledgeItem]) -> List[Dict[str, Any]]:
        """Detect conflicts between knowledge items"""
        conflicts = []
        
        for i, item1 in enumerate(knowledge_items):
            for j, item2 in enumerate(knowledge_items[i+1:], i+1):
                conflict = self._check_conflict_pair(item1, item2)
                if conflict:
                    conflicts.append({
                        'conflict_id': str(uuid.uuid4()),
                        'item1_id': item1.id,
                        'item2_id': item2.id,
                        'conflict_type': conflict['type'],
                        'severity': conflict['severity'],
                        'description': conflict['description']
                    })
        
        return conflicts
    
    def _check_conflict_pair(self, item1: KnowledgeItem, 
                           item2: KnowledgeItem) -> Optional[Dict[str, Any]]:
        """Check if two knowledge items conflict"""
        
        # Check for direct contradiction
        if self._is_direct_contradiction(item1, item2):
            return {
                'type': ConflictType.DIRECT_CONTRADICTION,
                'severity': 'high',
                'description': f"Direct contradiction between '{item1.content}' and '{item2.content}'"
            }
        
        # Check for causal contradictions
        if item1.knowledge_type == KnowledgeType.CAUSAL and item2.knowledge_type == KnowledgeType.CAUSAL:
            if self._is_causal_contradiction(item1, item2):
                return {
                    'type': ConflictType.CAUSAL_CONTRADICTION,
                    'severity': 'medium',
                    'description': f"Causal contradiction between '{item1.content}' and '{item2.content}'"
                }
        
        # Check confidence mismatches
        if (item1.related_concepts & item2.related_concepts and 
            abs(item1.confidence - item2.confidence) > 0.4):
            return {
                'type': ConflictType.CONFIDENCE_MISMATCH,
                'severity': 'low',
                'description': f"Confidence mismatch: {item1.confidence:.2f} vs {item2.confidence:.2f}"
            }
        
        return None
    
    def _is_direct_contradiction(self, item1: KnowledgeItem, item2: KnowledgeItem) -> bool:
        """Check if items directly contradict each other"""
        # Simple contradiction detection
        content1_lower = item1.content.lower()
        content2_lower = item2.content.lower()
        
        # Look for negation patterns
        negation_pairs = [
            ("is", "is not"),
            ("can", "cannot"),
            ("will", "will not"),
            ("does", "does not")
        ]
        
        for pos, neg in negation_pairs:
            if pos in content1_lower and neg in content2_lower:
                return True
            if neg in content1_lower and pos in content2_lower:
                return True
        
        return False
    
    def _is_causal_contradiction(self, item1: KnowledgeItem, item2: KnowledgeItem) -> bool:
        """Check if causal items contradict each other"""
        # Simplified causal contradiction detection
        # Would need more sophisticated causal reasoning
        
        # Check if they claim opposite causal relationships
        if item1.related_concepts & item2.related_concepts:
            # Same concepts involved but different causal claims
            return "causes" in item1.content and "prevents" in item2.content
        
        return False
    
    def resolve_conflict(self, conflict: Dict[str, Any], 
                        knowledge_items: List[KnowledgeItem]) -> Dict[str, Any]:
        """Resolve a specific conflict"""
        conflict_type = conflict['conflict_type']
        
        if conflict_type in self.resolution_strategies:
            resolution = self.resolution_strategies[conflict_type](conflict, knowledge_items)
        else:
            resolution = self._default_resolution(conflict, knowledge_items)
        
        # Record resolution
        self.resolution_history.append({
            'conflict_id': conflict['conflict_id'],
            'resolution': resolution,
            'timestamp': time.time()
        })
        
        return resolution
    
    def _resolve_contradiction(self, conflict: Dict[str, Any], 
                             knowledge_items: List[KnowledgeItem]) -> Dict[str, Any]:
        """Resolve direct contradiction"""
        item1 = next(item for item in knowledge_items if item.id == conflict['item1_id'])
        item2 = next(item for item in knowledge_items if item.id == conflict['item2_id'])
        
        # Prefer higher confidence item
        if item1.confidence > item2.confidence:
            return {
                'resolution_type': 'prefer_higher_confidence',
                'preferred_item': item1.id,
                'rejected_item': item2.id,
                'reasoning': f"Preferred item with confidence {item1.confidence} over {item2.confidence}"
            }
        elif item2.confidence > item1.confidence:
            return {
                'resolution_type': 'prefer_higher_confidence',
                'preferred_item': item2.id,
                'rejected_item': item1.id,
                'reasoning': f"Preferred item with confidence {item2.confidence} over {item1.confidence}"
            }
        else:
            # Equal confidence - prefer more recent
            if item1.timestamp > item2.timestamp:
                return {
                    'resolution_type': 'prefer_more_recent',
                    'preferred_item': item1.id,
                    'rejected_item': item2.id,
                    'reasoning': "Preferred more recent information"
                }
            else:
                return {
                    'resolution_type': 'prefer_more_recent',
                    'preferred_item': item2.id,
                    'rejected_item': item1.id,
                    'reasoning': "Preferred more recent information"
                }
    
    def _resolve_implications(self, conflict: Dict[str, Any], 
                            knowledge_items: List[KnowledgeItem]) -> Dict[str, Any]:
        """Resolve inconsistent implications"""
        return {
            'resolution_type': 'mark_for_review',
            'reasoning': "Inconsistent implications require deeper analysis"
        }
    
    def _resolve_temporal(self, conflict: Dict[str, Any], 
                        knowledge_items: List[KnowledgeItem]) -> Dict[str, Any]:
        """Resolve temporal inconsistency"""
        return {
            'resolution_type': 'temporal_ordering',
            'reasoning': "Applied temporal reasoning to resolve sequence"
        }
    
    def _resolve_causal(self, conflict: Dict[str, Any], 
                      knowledge_items: List[KnowledgeItem]) -> Dict[str, Any]:
        """Resolve causal contradiction"""
        return {
            'resolution_type': 'causal_analysis',
            'reasoning': "Applied causal reasoning to resolve contradiction"
        }
    
    def _resolve_confidence(self, conflict: Dict[str, Any], 
                          knowledge_items: List[KnowledgeItem]) -> Dict[str, Any]:
        """Resolve confidence mismatch"""
        return {
            'resolution_type': 'confidence_adjustment',
            'reasoning': "Adjusted confidence levels based on evidence"
        }
    
    def _default_resolution(self, conflict: Dict[str, Any], 
                          knowledge_items: List[KnowledgeItem]) -> Dict[str, Any]:
        """Default resolution strategy"""
        return {
            'resolution_type': 'needs_manual_review',
            'reasoning': "Conflict requires manual review and resolution"
        }


class CausalKnowledgeUpdater:
    """Updates causal knowledge models based on new information"""
    
    def __init__(self, causal_reasoner: Optional[CausalReasoner] = None):
        self.causal_reasoner = causal_reasoner
        self.causal_graph = nx.DiGraph()
        self.causal_knowledge = []
        
    def add_causal_knowledge(self, knowledge_item: KnowledgeItem):
        """Add causal knowledge to the model"""
        if knowledge_item.knowledge_type != KnowledgeType.CAUSAL:
            return
        
        # Extract causal relationships from content
        causal_relations = self._extract_causal_relations(knowledge_item)
        
        for relation in causal_relations:
            self._add_causal_edge(relation['cause'], relation['effect'], knowledge_item)
        
        self.causal_knowledge.append(knowledge_item)
    
    def _extract_causal_relations(self, knowledge_item: KnowledgeItem) -> List[Dict[str, str]]:
        """Extract causal relations from knowledge content"""
        content = knowledge_item.content.lower()
        relations = []
        
        # Simple pattern matching for causal relations
        import re
        
        # Pattern: "X causes Y"
        causes_pattern = re.findall(r"(.+?)\s+causes?\s+(.+)", content)
        for cause, effect in causes_pattern:
            relations.append({
                'cause': cause.strip(),
                'effect': effect.strip(),
                'strength': knowledge_item.confidence
            })
        
        # Pattern: "X leads to Y"
        leads_pattern = re.findall(r"(.+?)\s+leads?\s+to\s+(.+)", content)
        for cause, effect in leads_pattern:
            relations.append({
                'cause': cause.strip(),
                'effect': effect.strip(),
                'strength': knowledge_item.confidence
            })
        
        return relations
    
    def _add_causal_edge(self, cause: str, effect: str, knowledge_item: KnowledgeItem):
        """Add causal edge to the graph"""
        self.causal_graph.add_edge(cause, effect, 
                                 strength=knowledge_item.confidence,
                                 source=knowledge_item.id,
                                 timestamp=knowledge_item.timestamp)
    
    def update_causal_strength(self, cause: str, effect: str, new_evidence: float):
        """Update the strength of a causal relationship"""
        if self.causal_graph.has_edge(cause, effect):
            current_strength = self.causal_graph[cause][effect]['strength']
            # Average with new evidence
            updated_strength = (current_strength + new_evidence) / 2
            self.causal_graph[cause][effect]['strength'] = updated_strength
    
    def get_causal_chain(self, start: str, end: str) -> List[str]:
        """Get causal chain between two concepts"""
        try:
            path = nx.shortest_path(self.causal_graph, start, end)
            return path
        except nx.NetworkXNoPath:
            return []
    
    def get_causal_effects(self, cause: str) -> List[Tuple[str, float]]:
        """Get all effects of a given cause"""
        if cause not in self.causal_graph:
            return []
        
        effects = []
        for successor in self.causal_graph.successors(cause):
            strength = self.causal_graph[cause][successor]['strength']
            effects.append((successor, strength))
        
        return sorted(effects, key=lambda x: x[1], reverse=True)
    
    def get_causal_causes(self, effect: str) -> List[Tuple[str, float]]:
        """Get all causes of a given effect"""
        if effect not in self.causal_graph:
            return []
        
        causes = []
        for predecessor in self.causal_graph.predecessors(effect):
            strength = self.causal_graph[predecessor][effect]['strength']
            causes.append((predecessor, strength))
        
        return sorted(causes, key=lambda x: x[1], reverse=True)


class DynamicKnowledgeGraph:
    """Main dynamic knowledge graph system"""
    
    def __init__(self, cognitive_architecture: Optional[CognitiveArchitecture] = None):
        self.cognitive_architecture = cognitive_architecture
        
        # Knowledge storage
        self.knowledge_items = {}
        self.concept_nodes = {}
        self.relations = {}
        
        # Component systems
        self.conversational_learner = ConversationalLearning(cognitive_architecture)
        self.conflict_resolver = KnowledgeConflictResolver()
        self.causal_updater = CausalKnowledgeUpdater()
        
        # Knowledge graph
        self.knowledge_graph = nx.MultiDiGraph()
        
        # Statistics
        self.stats = {
            'knowledge_items_count': 0,
            'concepts_count': 0,
            'relations_count': 0,
            'conflicts_resolved': 0,
            'learning_sessions': 0
        }
    
    def learn_from_conversation(self, user_input: str, 
                               conversation_context: List[Dict[str, str]]) -> Dict[str, Any]:
        """Learn new knowledge from user conversation"""
        learning_start = time.time()
        
        # Extract knowledge from conversation
        new_knowledge = self.conversational_learner.extract_knowledge_from_conversation(
            user_input, conversation_context
        )
        
        if not new_knowledge:
            return {
                'learned_items': 0,
                'new_concepts': 0,
                'conflicts_detected': 0,
                'learning_time': (time.time() - learning_start) * 1000
            }
        
        # Detect conflicts with existing knowledge
        all_knowledge = list(self.knowledge_items.values()) + new_knowledge
        conflicts = self.conflict_resolver.detect_conflicts(all_knowledge)
        
        # Resolve conflicts
        for conflict in conflicts:
            resolution = self.conflict_resolver.resolve_conflict(conflict, all_knowledge)
            self._apply_conflict_resolution(resolution, all_knowledge)
        
        # Integrate new knowledge
        new_concepts = 0
        for item in new_knowledge:
            self._add_knowledge_item(item)
            
            # Update causal knowledge if applicable
            if item.knowledge_type == KnowledgeType.CAUSAL:
                self.causal_updater.add_causal_knowledge(item)
            
            # Create concept nodes for new concepts
            for concept in item.related_concepts:
                if concept not in self.concept_nodes:
                    self._create_concept_node(concept, item)
                    new_concepts += 1
        
        # Integrate with cognitive architecture
        if self.cognitive_architecture:
            self.conversational_learner.integrate_with_working_memory(new_knowledge)
        
        # Update statistics
        self.stats['learning_sessions'] += 1
        self.stats['conflicts_resolved'] += len(conflicts)
        
        learning_time = (time.time() - learning_start) * 1000
        
        return {
            'learned_items': len(new_knowledge),
            'new_concepts': new_concepts,
            'conflicts_detected': len(conflicts),
            'conflicts_resolved': len(conflicts),
            'learning_time': learning_time,
            'knowledge_items': [item.content for item in new_knowledge]
        }
    
    def _add_knowledge_item(self, item: KnowledgeItem):
        """Add knowledge item to the graph"""
        self.knowledge_items[item.id] = item
        
        # Add to knowledge graph
        self.knowledge_graph.add_node(item.id, 
                                    content=item.content,
                                    knowledge_type=item.knowledge_type.value,
                                    confidence=item.confidence,
                                    timestamp=item.timestamp)
        
        self.stats['knowledge_items_count'] += 1
    
    def _create_concept_node(self, concept_name: str, source_item: KnowledgeItem):
        """Create a new concept node"""
        concept_id = f"concept_{len(self.concept_nodes)}"
        
        concept_node = ConceptNode(
            id=concept_id,
            name=concept_name,
            definition=f"Concept extracted from: {source_item.content}",
            confidence=source_item.confidence
        )
        
        self.concept_nodes[concept_name] = concept_node
        
        # Add to knowledge graph
        self.knowledge_graph.add_node(concept_id,
                                    name=concept_name,
                                    node_type='concept',
                                    confidence=concept_node.confidence)
        
        # Link to source knowledge item
        self.knowledge_graph.add_edge(source_item.id, concept_id, 
                                    relation_type='mentions_concept')
        
        self.stats['concepts_count'] += 1
    
    def _apply_conflict_resolution(self, resolution: Dict[str, Any], 
                                 knowledge_items: List[KnowledgeItem]):
        """Apply conflict resolution"""
        if resolution['resolution_type'] == 'prefer_higher_confidence':
            # Mark rejected item for removal or lower confidence
            rejected_id = resolution['rejected_item']
            for item in knowledge_items:
                if item.id == rejected_id:
                    item.confidence *= 0.5  # Reduce confidence
                    item.validation_status = 'disputed'
        
        elif resolution['resolution_type'] == 'prefer_more_recent':
            # Similar handling for recency preference
            rejected_id = resolution['rejected_item']
            for item in knowledge_items:
                if item.id == rejected_id:
                    item.confidence *= 0.7
                    item.validation_status = 'superseded'
    
    def query_knowledge(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Query the knowledge graph"""
        results = []
        query_lower = query.lower()
        
        # Search through knowledge items
        for item in self.knowledge_items.values():
            if query_lower in item.content.lower():
                relevance_score = self._calculate_relevance(query, item)
                results.append({
                    'content': item.content,
                    'type': item.knowledge_type.value,
                    'confidence': item.confidence,
                    'relevance': relevance_score,
                    'source': item.source,
                    'timestamp': item.timestamp
                })
        
        # Sort by relevance and confidence
        results.sort(key=lambda x: (x['relevance'], x['confidence']), reverse=True)
        
        return results[:max_results]
    
    def _calculate_relevance(self, query: str, item: KnowledgeItem) -> float:
        """Calculate relevance score for query-item pair"""
        query_words = set(query.lower().split())
        content_words = set(item.content.lower().split())
        
        # Jaccard similarity
        intersection = len(query_words & content_words)
        union = len(query_words | content_words)
        
        if union == 0:
            return 0.0
        
        jaccard_sim = intersection / union
        
        # Boost score based on confidence and recency
        confidence_boost = item.confidence * 0.2
        recency_boost = min((time.time() - item.timestamp) / (24 * 3600), 0.2)  # Recent items get boost
        
        return jaccard_sim + confidence_boost + recency_boost
    
    def get_related_concepts(self, concept: str, max_depth: int = 2) -> List[str]:
        """Get concepts related to the given concept"""
        if concept not in self.concept_nodes:
            return []
        
        concept_id = self.concept_nodes[concept].id
        
        # Find connected concepts in the graph
        related = []
        
        try:
            # Get neighbors within max_depth
            for node in nx.single_source_shortest_path_length(
                self.knowledge_graph, concept_id, cutoff=max_depth
            ):
                if node != concept_id and 'name' in self.knowledge_graph.nodes[node]:
                    related.append(self.knowledge_graph.nodes[node]['name'])
        except:
            pass
        
        return related
    
    def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get knowledge graph statistics"""
        return {
            **self.stats,
            'graph_density': nx.density(self.knowledge_graph),
            'average_confidence': np.mean([item.confidence for item in self.knowledge_items.values()]) if self.knowledge_items else 0.0,
            'knowledge_types': {
                kt.value: sum(1 for item in self.knowledge_items.values() if item.knowledge_type == kt)
                for kt in KnowledgeType
            }
        }