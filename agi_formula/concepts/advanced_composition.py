"""Advanced concept composition system for AGI-Formula."""

from typing import Dict, List, Set, Tuple, Optional, Union
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import math
import itertools


class ConceptType(Enum):
    """Types of concepts in the composition system."""
    PRIMITIVE = "primitive"
    COMPOSITE = "composite"
    ABSTRACT = "abstract"
    RELATIONAL = "relational"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    CAUSAL = "causal"


@dataclass
class ConceptProperties:
    """Properties and metadata for a concept."""
    concept_id: str
    concept_type: ConceptType
    embedding: np.ndarray
    semantic_features: Dict[str, float] = field(default_factory=dict)
    
    # Composition properties
    composability_score: float = 1.0
    abstract_level: int = 0  # 0 = most concrete, higher = more abstract
    temporal_stability: float = 1.0
    spatial_relevance: float = 0.5
    
    # Usage statistics
    usage_count: int = 0
    success_rate: float = 0.5
    last_used: float = 0.0
    
    # Relationships
    parent_concepts: List[str] = field(default_factory=list)
    child_concepts: List[str] = field(default_factory=list)
    related_concepts: Dict[str, float] = field(default_factory=dict)
    
    # Composition constraints
    incompatible_concepts: Set[str] = field(default_factory=set)
    required_concepts: Set[str] = field(default_factory=set)
    max_composition_size: int = 10


@dataclass
class CompositionRule:
    """Rule for how concepts can be composed."""
    rule_id: str
    source_concepts: List[str]
    target_concept: str
    rule_type: str  # "combination", "abstraction", "specialization", etc.
    confidence: float
    conditions: Dict[str, any] = field(default_factory=dict)
    success_history: List[float] = field(default_factory=list)


@dataclass
class CompositionResult:
    """Result of a concept composition operation."""
    composed_concept: str
    component_concepts: List[str]
    composition_strength: float
    semantic_coherence: float
    novelty_score: float
    confidence: float
    reasoning_path: List[str] = field(default_factory=list)
    
    # Detailed analysis
    compatibility_matrix: np.ndarray = None
    semantic_distance: float = 0.0
    compositional_complexity: float = 0.0


class AdvancedConceptComposition:
    """
    Advanced concept composition system with hierarchical concepts,
    semantic reasoning, and dynamic composition rules.
    
    Features:
    - Hierarchical concept organization
    - Semantic compatibility analysis
    - Dynamic composition rule learning
    - Multi-modal concept understanding
    - Temporal and spatial concept reasoning
    - Emergent concept discovery
    """
    
    def __init__(self, embedding_dim: int = 64):
        """Initialize the advanced concept composition system."""
        self.embedding_dim = embedding_dim
        
        # Core concept storage
        self.concepts: Dict[str, ConceptProperties] = {}
        self.concept_embeddings: Dict[str, np.ndarray] = {}
        
        # Composition rules and patterns
        self.composition_rules: List[CompositionRule] = []
        self.learned_patterns: Dict[str, List[str]] = defaultdict(list)
        
        # Semantic relationships
        self.semantic_graph: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.compatibility_cache: Dict[Tuple[str, str], float] = {}
        
        # Hierarchical organization
        self.concept_hierarchy: Dict[str, Set[str]] = defaultdict(set)  # parent -> children
        self.reverse_hierarchy: Dict[str, Set[str]] = defaultdict(set)  # child -> parents
        
        # Composition history and learning
        self.composition_history: deque = deque(maxlen=1000)
        self.successful_compositions: Dict[str, List[CompositionResult]] = defaultdict(list)
        self.failed_compositions: Dict[str, int] = defaultdict(int)
        
        # Advanced features
        self.emergent_concepts: Set[str] = set()
        self.temporal_concepts: Dict[str, List[float]] = defaultdict(list)
        self.spatial_concepts: Dict[str, np.ndarray] = {}
        
        # Performance metrics
        self.composition_attempts = 0
        self.successful_compositions_count = 0
        self.novel_concepts_discovered = 0
        
        # Initialize with basic concepts
        self._initialize_primitive_concepts()
    
    def register_concept(
        self,
        concept_id: str,
        concept_type: ConceptType,
        semantic_features: Dict[str, float],
        embedding: Optional[np.ndarray] = None
    ) -> ConceptProperties:
        """
        Register a new concept in the composition system.
        
        Args:
            concept_id: Unique identifier for the concept
            concept_type: Type of the concept
            semantic_features: Feature dictionary
            embedding: Optional pre-computed embedding
            
        Returns:
            ConceptProperties object for the registered concept
        """
        if embedding is None:
            embedding = self._generate_embedding_from_features(semantic_features)
        
        concept = ConceptProperties(
            concept_id=concept_id,
            concept_type=concept_type,
            embedding=embedding,
            semantic_features=semantic_features
        )
        
        self.concepts[concept_id] = concept
        self.concept_embeddings[concept_id] = embedding
        
        # Update semantic relationships
        self._update_semantic_relationships(concept_id)
        
        return concept
    
    def compose_concepts(
        self,
        component_concepts: List[str],
        composition_type: str = "automatic",
        target_concept: Optional[str] = None
    ) -> CompositionResult:
        """
        Compose multiple concepts into a new composite concept.
        
        Args:
            component_concepts: List of concept IDs to compose
            composition_type: Type of composition ("automatic", "guided", "hierarchical")
            target_concept: Optional target concept ID
            
        Returns:
            CompositionResult with details about the composition
        """
        self.composition_attempts += 1
        
        # Validate input concepts
        valid_concepts = [cid for cid in component_concepts if cid in self.concepts]
        if len(valid_concepts) < 2:
            return self._create_failed_composition_result(component_concepts, "insufficient_valid_concepts")
        
        # Check compatibility
        compatibility_score = self._compute_compatibility_matrix(valid_concepts)
        if compatibility_score < 0.3:
            return self._create_failed_composition_result(valid_concepts, "incompatible_concepts")
        
        # Generate composed concept
        if target_concept is None:
            target_concept = self._generate_composed_concept_id(valid_concepts)
        
        # Compute composition metrics
        composition_strength = self._compute_composition_strength(valid_concepts)
        semantic_coherence = self._compute_semantic_coherence(valid_concepts)
        novelty_score = self._compute_novelty_score(valid_concepts, target_concept)
        
        # Create composed embedding
        composed_embedding = self._compose_embeddings(valid_concepts)
        
        # Apply composition rules
        rule_adjustments = self._apply_composition_rules(valid_concepts, target_concept)
        
        # Create composition result
        result = CompositionResult(
            composed_concept=target_concept,
            component_concepts=valid_concepts,
            composition_strength=composition_strength * rule_adjustments,
            semantic_coherence=semantic_coherence,
            novelty_score=novelty_score,
            confidence=min(compatibility_score, composition_strength),
            compatibility_matrix=self._get_compatibility_matrix(valid_concepts)
        )
        
        # Register the new composite concept if successful
        if result.confidence > 0.5:
            self._register_composite_concept(result, composed_embedding)
            self.successful_compositions_count += 1
            self.successful_compositions[target_concept].append(result)
        else:
            self.failed_compositions[target_concept] += 1
        
        # Learn from this composition
        self._learn_from_composition(result)
        
        # Store in history
        self.composition_history.append(result)
        
        return result
    
    def decompose_concept(self, concept_id: str) -> List[str]:
        """
        Decompose a composite concept into its component parts.
        
        Args:
            concept_id: The concept to decompose
            
        Returns:
            List of component concept IDs
        """
        if concept_id not in self.concepts:
            return []
        
        concept = self.concepts[concept_id]
        
        # If it's a composite concept, return known components
        if concept.concept_type == ConceptType.COMPOSITE and concept.parent_concepts:
            return concept.parent_concepts.copy()
        
        # For other concepts, try to infer components
        components = self._infer_concept_components(concept_id)
        return components
    
    def find_concept_analogies(
        self,
        source_concept: str,
        target_domain_concepts: List[str],
        analogy_strength_threshold: float = 0.6
    ) -> List[Tuple[str, float]]:
        """
        Find analogical relationships between concepts across domains.
        
        Args:
            source_concept: The source concept for analogy
            target_domain_concepts: Concepts in the target domain
            analogy_strength_threshold: Minimum analogy strength
            
        Returns:
            List of (concept_id, analogy_strength) tuples
        """
        if source_concept not in self.concepts:
            return []
        
        source_embedding = self.concept_embeddings[source_concept]
        source_features = self.concepts[source_concept].semantic_features
        
        analogies = []
        
        for target_concept in target_domain_concepts:
            if target_concept not in self.concepts:
                continue
            
            # Compute structural similarity
            target_embedding = self.concept_embeddings[target_concept]
            embedding_similarity = self._cosine_similarity(source_embedding, target_embedding)
            
            # Compute functional similarity
            target_features = self.concepts[target_concept].semantic_features
            functional_similarity = self._compute_functional_similarity(source_features, target_features)
            
            # Compute relational similarity
            relational_similarity = self._compute_relational_similarity(source_concept, target_concept)
            
            # Combined analogy strength
            analogy_strength = (
                0.4 * embedding_similarity +
                0.3 * functional_similarity +
                0.3 * relational_similarity
            )
            
            if analogy_strength >= analogy_strength_threshold:
                analogies.append((target_concept, analogy_strength))
        
        # Sort by analogy strength
        analogies.sort(key=lambda x: x[1], reverse=True)
        return analogies
    
    def discover_emergent_concepts(
        self,
        activation_patterns: Dict[str, List[float]],
        emergence_threshold: float = 0.8
    ) -> List[str]:
        """
        Discover emergent concepts from activation patterns.
        
        Args:
            activation_patterns: Neuron activation patterns over time
            emergence_threshold: Threshold for concept emergence
            
        Returns:
            List of discovered emergent concept IDs
        """
        emergent_concepts = []
        
        # Analyze activation patterns for clusters
        pattern_clusters = self._cluster_activation_patterns(activation_patterns)
        
        for cluster_id, cluster_patterns in pattern_clusters.items():
            # Check if this represents a novel concept
            novelty_score = self._assess_pattern_novelty(cluster_patterns)
            
            if novelty_score >= emergence_threshold:
                # Create emergent concept
                emergent_concept_id = f"emergent_{cluster_id}_{len(self.emergent_concepts)}"
                
                # Generate embedding for emergent concept
                emergent_embedding = self._generate_emergent_embedding(cluster_patterns)
                
                # Infer semantic features
                semantic_features = self._infer_semantic_features_from_patterns(cluster_patterns)
                
                # Register emergent concept
                self.register_concept(
                    emergent_concept_id,
                    ConceptType.ABSTRACT,
                    semantic_features,
                    emergent_embedding
                )
                
                self.emergent_concepts.add(emergent_concept_id)
                emergent_concepts.append(emergent_concept_id)
                self.novel_concepts_discovered += 1
        
        return emergent_concepts
    
    def get_concept_compatibility(self, concept1: str, concept2: str) -> float:
        """Get compatibility score between two concepts."""
        cache_key = tuple(sorted([concept1, concept2]))
        
        if cache_key in self.compatibility_cache:
            return self.compatibility_cache[cache_key]
        
        compatibility = self._compute_pairwise_compatibility(concept1, concept2)
        self.compatibility_cache[cache_key] = compatibility
        
        return compatibility
    
    def suggest_compositions(
        self,
        base_concepts: List[str],
        max_suggestions: int = 5
    ) -> List[Tuple[List[str], float]]:
        """
        Suggest possible concept compositions.
        
        Args:
            base_concepts: Available concepts for composition
            max_suggestions: Maximum number of suggestions
            
        Returns:
            List of (concept_combination, predicted_success) tuples
        """
        suggestions = []
        
        # Generate all possible combinations of 2-4 concepts
        for r in range(2, min(5, len(base_concepts) + 1)):
            for combination in itertools.combinations(base_concepts, r):
                # Predict composition success
                success_score = self._predict_composition_success(list(combination))
                
                if success_score > 0.4:  # Only suggest promising compositions
                    suggestions.append((list(combination), success_score))
        
        # Sort by predicted success and return top suggestions
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions[:max_suggestions]
    
    def _initialize_primitive_concepts(self) -> None:
        """Initialize the system with basic primitive concepts."""
        primitive_concepts = {
            "color": {"visual": 1.0, "categorical": 1.0, "observable": 1.0},
            "shape": {"visual": 1.0, "geometric": 1.0, "structural": 1.0},
            "size": {"quantitative": 1.0, "measurable": 1.0, "comparative": 1.0},
            "texture": {"tactile": 1.0, "surface": 1.0, "material": 1.0},
            "movement": {"temporal": 1.0, "dynamic": 1.0, "spatial": 1.0},
            "sound": {"auditory": 1.0, "wave": 1.0, "temporal": 0.8},
            "temperature": {"sensory": 1.0, "thermal": 1.0, "measurable": 1.0},
        }
        
        for concept_id, features in primitive_concepts.items():
            self.register_concept(concept_id, ConceptType.PRIMITIVE, features)
    
    def _generate_embedding_from_features(self, features: Dict[str, float]) -> np.ndarray:
        """Generate embedding vector from semantic features."""
        # Create a feature-based embedding
        embedding = np.random.normal(0, 0.1, self.embedding_dim)
        
        # Encode features into embedding
        for i, (feature, value) in enumerate(features.items()):
            if i < self.embedding_dim:
                embedding[i] += value
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def _update_semantic_relationships(self, concept_id: str) -> None:
        """Update semantic relationships for a concept."""
        if concept_id not in self.concept_embeddings:
            return
        
        current_embedding = self.concept_embeddings[concept_id]
        
        # Compute relationships with all other concepts
        for other_id, other_embedding in self.concept_embeddings.items():
            if other_id != concept_id:
                similarity = self._cosine_similarity(current_embedding, other_embedding)
                
                # Store bidirectional relationships
                self.semantic_graph[concept_id][other_id] = similarity
                self.semantic_graph[other_id][concept_id] = similarity
    
    def _compute_compatibility_matrix(self, concepts: List[str]) -> float:
        """Compute overall compatibility score for a list of concepts."""
        if len(concepts) < 2:
            return 1.0
        
        total_compatibility = 0.0
        pair_count = 0
        
        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                compatibility = self.get_concept_compatibility(concepts[i], concepts[j])
                total_compatibility += compatibility
                pair_count += 1
        
        return total_compatibility / pair_count if pair_count > 0 else 0.0
    
    def _compute_pairwise_compatibility(self, concept1: str, concept2: str) -> float:
        """Compute compatibility between two concepts."""
        if concept1 not in self.concepts or concept2 not in self.concepts:
            return 0.0
        
        c1 = self.concepts[concept1]
        c2 = self.concepts[concept2]
        
        # Check explicit incompatibilities
        if concept2 in c1.incompatible_concepts or concept1 in c2.incompatible_concepts:
            return 0.0
        
        # Semantic similarity
        semantic_sim = self.semantic_graph.get(concept1, {}).get(concept2, 0.0)
        
        # Feature similarity
        feature_sim = self._compute_feature_similarity(c1.semantic_features, c2.semantic_features)
        
        # Type compatibility
        type_compat = self._compute_type_compatibility(c1.concept_type, c2.concept_type)
        
        # Abstract level compatibility (closer levels are more compatible)
        level_diff = abs(c1.abstract_level - c2.abstract_level)
        level_compat = 1.0 / (1.0 + level_diff * 0.3)
        
        # Combined compatibility
        compatibility = (
            0.3 * semantic_sim +
            0.3 * feature_sim +
            0.2 * type_compat +
            0.2 * level_compat
        )
        
        return compatibility
    
    def _compute_composition_strength(self, concepts: List[str]) -> float:
        """Compute the strength of composition for given concepts."""
        if len(concepts) < 2:
            return 0.0
        
        # Base strength from individual concept properties
        base_strength = np.mean([self.concepts[cid].composability_score for cid in concepts])
        
        # Synergy bonus for well-matching concepts
        avg_compatibility = self._compute_compatibility_matrix(concepts)
        synergy_bonus = avg_compatibility * 0.3
        
        # Complexity penalty for too many concepts
        complexity_penalty = max(0.0, (len(concepts) - 3) * 0.1)
        
        strength = base_strength + synergy_bonus - complexity_penalty
        return max(0.0, min(1.0, strength))
    
    def _compute_semantic_coherence(self, concepts: List[str]) -> float:
        """Compute semantic coherence of concept combination."""
        if len(concepts) < 2:
            return 1.0
        
        # Get embeddings
        embeddings = [self.concept_embeddings[cid] for cid in concepts if cid in self.concept_embeddings]
        
        if len(embeddings) < 2:
            return 0.0
        
        # Compute centroid
        centroid = np.mean(embeddings, axis=0)
        
        # Compute average distance to centroid (lower = more coherent)
        distances = [np.linalg.norm(emb - centroid) for emb in embeddings]
        avg_distance = np.mean(distances)
        
        # Convert to coherence score (higher = more coherent)
        coherence = 1.0 / (1.0 + avg_distance)
        return coherence
    
    def _compute_novelty_score(self, concepts: List[str], target_concept: str) -> float:
        """Compute novelty score for a composition."""
        # Check if this exact composition has been tried before
        composition_signature = tuple(sorted(concepts))
        
        novelty = 1.0
        
        # Reduce novelty if composition exists
        if target_concept in self.successful_compositions:
            for past_result in self.successful_compositions[target_concept]:
                past_signature = tuple(sorted(past_result.component_concepts))
                if past_signature == composition_signature:
                    novelty *= 0.1  # Very low novelty for exact match
                else:
                    # Partial overlap reduces novelty
                    overlap = len(set(concepts) & set(past_result.component_concepts))
                    overlap_ratio = overlap / len(concepts)
                    novelty *= (1.0 - overlap_ratio * 0.5)
        
        return max(0.0, min(1.0, novelty))
    
    def _compose_embeddings(self, concepts: List[str]) -> np.ndarray:
        """Compose embeddings of multiple concepts."""
        embeddings = [self.concept_embeddings[cid] for cid in concepts if cid in self.concept_embeddings]
        
        if not embeddings:
            return np.random.normal(0, 0.1, self.embedding_dim)
        
        # Weighted average based on concept importance
        weights = [self.concepts[cid].usage_count + 1 for cid in concepts if cid in self.concepts]
        weights = np.array(weights) / sum(weights)
        
        composed = np.average(embeddings, axis=0, weights=weights)
        
        # Add small random component for uniqueness
        composed += np.random.normal(0, 0.05, self.embedding_dim)
        
        # Normalize
        norm = np.linalg.norm(composed)
        if norm > 0:
            composed = composed / norm
        
        return composed
    
    def _apply_composition_rules(self, concepts: List[str], target_concept: str) -> float:
        """Apply learned composition rules and return adjustment factor."""
        adjustment = 1.0
        
        for rule in self.composition_rules:
            if self._rule_matches(rule, concepts, target_concept):
                # Apply rule adjustment based on rule confidence
                rule_strength = rule.confidence * np.mean(rule.success_history) if rule.success_history else rule.confidence
                adjustment *= (1.0 + rule_strength * 0.2)
        
        return adjustment
    
    def _rule_matches(self, rule: CompositionRule, concepts: List[str], target: str) -> bool:
        """Check if a composition rule matches the current composition."""
        # Check if all source concepts are present
        source_set = set(rule.source_concepts)
        concept_set = set(concepts)
        
        if rule.rule_type == "exact_match":
            return source_set == concept_set
        elif rule.rule_type == "subset":
            return source_set.issubset(concept_set)
        elif rule.rule_type == "superset":
            return concept_set.issubset(source_set)
        elif rule.rule_type == "overlap":
            overlap = len(source_set & concept_set)
            return overlap >= rule.conditions.get("min_overlap", 1)
        
        return False
    
    def _generate_composed_concept_id(self, concepts: List[str]) -> str:
        """Generate ID for a composed concept."""
        # Sort concepts for consistency
        sorted_concepts = sorted(concepts)
        
        # Create readable name
        if len(sorted_concepts) <= 3:
            composed_id = "_".join(sorted_concepts)
        else:
            # Use first few concepts + count
            composed_id = "_".join(sorted_concepts[:3]) + f"_plus{len(sorted_concepts)-3}"
        
        # Ensure uniqueness
        counter = 1
        base_id = composed_id
        while composed_id in self.concepts:
            composed_id = f"{base_id}_v{counter}"
            counter += 1
        
        return composed_id
    
    def _register_composite_concept(self, result: CompositionResult, embedding: np.ndarray) -> None:
        """Register a successfully composed concept."""
        # Infer semantic features from components
        component_features = {}
        for concept_id in result.component_concepts:
            if concept_id in self.concepts:
                for feature, value in self.concepts[concept_id].semantic_features.items():
                    if feature not in component_features:
                        component_features[feature] = []
                    component_features[feature].append(value)
        
        # Average features
        combined_features = {
            feature: np.mean(values) for feature, values in component_features.items()
        }
        
        # Add composition-specific features
        combined_features["composite"] = 1.0
        combined_features["composition_strength"] = result.composition_strength
        combined_features["semantic_coherence"] = result.semantic_coherence
        
        # Register the concept
        concept = self.register_concept(
            result.composed_concept,
            ConceptType.COMPOSITE,
            combined_features,
            embedding
        )
        
        # Set parent relationships
        concept.parent_concepts = result.component_concepts.copy()
        
        # Update hierarchy
        for parent_id in result.component_concepts:
            self.concept_hierarchy[parent_id].add(result.composed_concept)
            self.reverse_hierarchy[result.composed_concept].add(parent_id)
    
    def _learn_from_composition(self, result: CompositionResult) -> None:
        """Learn composition rules from successful/failed compositions."""
        if result.confidence > 0.7:
            # Learn successful pattern
            pattern_key = tuple(sorted(result.component_concepts))
            self.learned_patterns[result.composed_concept].append(pattern_key)
            
            # Create or update composition rule
            rule_id = f"rule_{len(self.composition_rules)}"
            rule = CompositionRule(
                rule_id=rule_id,
                source_concepts=result.component_concepts.copy(),
                target_concept=result.composed_concept,
                rule_type="subset",
                confidence=result.confidence
            )
            rule.success_history.append(result.confidence)
            self.composition_rules.append(rule)
    
    def _create_failed_composition_result(self, concepts: List[str], reason: str) -> CompositionResult:
        """Create a result object for failed composition."""
        return CompositionResult(
            composed_concept=f"failed_{hash(tuple(concepts))}",
            component_concepts=concepts,
            composition_strength=0.0,
            semantic_coherence=0.0,
            novelty_score=0.0,
            confidence=0.0,
            reasoning_path=[f"Failed: {reason}"]
        )
    
    # Helper methods for advanced features
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return np.dot(a, b) / (norm_a * norm_b)
    
    def _compute_feature_similarity(self, features1: Dict[str, float], features2: Dict[str, float]) -> float:
        """Compute similarity between feature dictionaries."""
        all_features = set(features1.keys()) | set(features2.keys())
        
        if not all_features:
            return 0.0
        
        similarities = []
        for feature in all_features:
            val1 = features1.get(feature, 0.0)
            val2 = features2.get(feature, 0.0)
            similarities.append(1.0 - abs(val1 - val2))
        
        return np.mean(similarities)
    
    def _compute_type_compatibility(self, type1: ConceptType, type2: ConceptType) -> float:
        """Compute compatibility between concept types."""
        # Define type compatibility matrix
        compatibility_matrix = {
            (ConceptType.PRIMITIVE, ConceptType.PRIMITIVE): 0.8,
            (ConceptType.PRIMITIVE, ConceptType.COMPOSITE): 0.9,
            (ConceptType.COMPOSITE, ConceptType.COMPOSITE): 0.7,
            (ConceptType.ABSTRACT, ConceptType.PRIMITIVE): 0.6,
            (ConceptType.ABSTRACT, ConceptType.COMPOSITE): 0.8,
            (ConceptType.RELATIONAL, ConceptType.PRIMITIVE): 0.5,
            (ConceptType.RELATIONAL, ConceptType.COMPOSITE): 0.7,
        }
        
        # Check both directions
        key1 = (type1, type2)
        key2 = (type2, type1)
        
        return compatibility_matrix.get(key1, compatibility_matrix.get(key2, 0.5))
    
    def get_composition_statistics(self) -> Dict:
        """Get comprehensive statistics about concept composition."""
        return {
            "total_concepts": len(self.concepts),
            "primitive_concepts": len([c for c in self.concepts.values() if c.concept_type == ConceptType.PRIMITIVE]),
            "composite_concepts": len([c for c in self.concepts.values() if c.concept_type == ConceptType.COMPOSITE]),
            "emergent_concepts": len(self.emergent_concepts),
            
            "composition_attempts": self.composition_attempts,
            "successful_compositions": self.successful_compositions_count,
            "success_rate": self.successful_compositions_count / max(1, self.composition_attempts),
            
            "learned_rules": len(self.composition_rules),
            "learned_patterns": sum(len(patterns) for patterns in self.learned_patterns.values()),
            
            "semantic_relationships": sum(len(rels) for rels in self.semantic_graph.values()),
            "cached_compatibilities": len(self.compatibility_cache),
            
            "novel_concepts_discovered": self.novel_concepts_discovered,
            "hierarchical_levels": max([c.abstract_level for c in self.concepts.values()]) if self.concepts else 0,
        }
    
    # Placeholder implementations for complex methods that would need more space
    def _infer_concept_components(self, concept_id: str) -> List[str]:
        """Infer components of a concept through analysis."""
        # Simplified implementation - in practice this would be more sophisticated
        return []
    
    def _compute_functional_similarity(self, features1: Dict, features2: Dict) -> float:
        """Compute functional similarity between concepts."""
        return self._compute_feature_similarity(features1, features2)
    
    def _compute_relational_similarity(self, concept1: str, concept2: str) -> float:
        """Compute relational similarity between concepts."""
        # Simplified - would analyze relationship patterns
        return 0.5
    
    def _cluster_activation_patterns(self, patterns: Dict) -> Dict:
        """Cluster activation patterns to find emergent concepts."""
        # Placeholder - would use clustering algorithms
        return {"cluster_1": patterns}
    
    def _assess_pattern_novelty(self, patterns: Dict) -> float:
        """Assess novelty of activation patterns."""
        # Placeholder - would compare with known patterns
        return 0.5
    
    def _generate_emergent_embedding(self, patterns: Dict) -> np.ndarray:
        """Generate embedding for emergent concept."""
        return np.random.normal(0, 0.1, self.embedding_dim)
    
    def _infer_semantic_features_from_patterns(self, patterns: Dict) -> Dict[str, float]:
        """Infer semantic features from activation patterns."""
        return {"emergent": 1.0, "pattern_based": 1.0}
    
    def _predict_composition_success(self, concepts: List[str]) -> float:
        """Predict success probability for concept composition."""
        # Simplified prediction based on compatibility
        return self._compute_compatibility_matrix(concepts)
    
    def _get_compatibility_matrix(self, concepts: List[str]) -> np.ndarray:
        """Get full compatibility matrix for concepts."""
        n = len(concepts)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    matrix[i, j] = self.get_concept_compatibility(concepts[i], concepts[j])
                else:
                    matrix[i, j] = 1.0
        
        return matrix