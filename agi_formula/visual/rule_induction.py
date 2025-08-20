"""
Visual Rule Induction Engine for AGI-LLM

Advanced system for learning and applying visual transformation rules:
- Pattern-based rule induction from examples
- Hierarchical rule learning and composition
- Context-aware rule application
- Meta-learning for rule generalization
- Integration with consciousness and reasoning systems

This goes beyond ARC-AGI to enable true visual intelligence in our AGI-LLM.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, Counter
import itertools
import copy

from .grid_processor import GridObject, BoundingBox, GridRepresentation, ObjectShape
from .pattern_detector import VisualPattern, PatternType, PatternMatch
from .spatial_transformer import Transformation, TransformationType, TransformationParameters
from .feature_extractor import FeatureVector, VisualFeatureExtractor


class RuleType(Enum):
    """Types of visual rules"""
    # Basic transformation rules
    GEOMETRIC_TRANSFORM = "geometric_transform"
    COLOR_CHANGE = "color_change"
    SIZE_SCALING = "size_scaling"
    
    # Pattern-based rules
    PATTERN_COMPLETION = "pattern_completion"
    PATTERN_EXTENSION = "pattern_extension"
    PATTERN_REPLICATION = "pattern_replication"
    
    # Conditional rules
    CONDITIONAL_TRANSFORM = "conditional_transform"
    CONTEXT_DEPENDENT = "context_dependent"
    MULTI_STEP = "multi_step"
    
    # Abstract rules
    ANALOGY_BASED = "analogy_based"
    CONCEPTUAL = "conceptual"
    EMERGENT = "emergent"
    
    # Meta-rules
    COMPOSITION = "composition"
    HIERARCHICAL = "hierarchical"
    ADAPTIVE = "adaptive"


class RuleConfidence(Enum):
    """Confidence levels for rules"""
    CERTAIN = "certain"      # 0.9-1.0
    HIGH = "high"           # 0.7-0.9
    MODERATE = "moderate"   # 0.5-0.7
    LOW = "low"            # 0.3-0.5
    UNCERTAIN = "uncertain" # 0.0-0.3


@dataclass
class RuleCondition:
    """Condition that triggers a rule application"""
    condition_type: str
    parameters: Dict[str, Any]
    confidence: float
    description: str
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate if condition is met"""
        # This would contain specific condition evaluation logic
        return True  # Simplified for now


@dataclass 
class RuleAction:
    """Action to be performed when rule is applied"""
    action_type: str
    parameters: Dict[str, Any]
    expected_outcome: Optional[str] = None
    side_effects: List[str] = field(default_factory=list)
    
    def execute(self, target: Any, context: Dict[str, Any]) -> Any:
        """Execute the rule action"""
        # This would contain specific action execution logic
        return target  # Simplified for now


@dataclass
class VisualRule:
    """Complete visual transformation rule"""
    rule_id: str
    rule_type: RuleType
    conditions: List[RuleCondition]
    actions: List[RuleAction]
    
    # Rule metadata
    confidence: float
    generality: float  # How broadly applicable the rule is
    specificity: float # How specific the rule conditions are
    complexity: int    # Number of steps/conditions
    
    # Learning history
    training_examples: List[Tuple[Any, Any]] = field(default_factory=list)
    success_rate: float = 1.0
    application_count: int = 0
    last_updated: float = 0.0
    
    # Rule relationships
    parent_rules: List[str] = field(default_factory=list)
    child_rules: List[str] = field(default_factory=list)
    related_rules: List[str] = field(default_factory=list)
    
    # Context information
    applicable_contexts: List[str] = field(default_factory=list)
    failure_contexts: List[str] = field(default_factory=list)
    
    def applies_to(self, context: Dict[str, Any]) -> float:
        """Determine applicability score for given context"""
        if not self.conditions:
            return self.confidence
        
        condition_scores = []
        for condition in self.conditions:
            if condition.evaluate(context):
                condition_scores.append(condition.confidence)
            else:
                condition_scores.append(0.0)
        
        # Rule applies if all conditions are met
        if all(score > 0.5 for score in condition_scores):
            return min(self.confidence, np.mean(condition_scores))
        else:
            return 0.0
    
    def apply(self, target: Any, context: Dict[str, Any]) -> Any:
        """Apply rule to target"""
        result = target
        
        for action in self.actions:
            result = action.execute(result, context)
        
        self.application_count += 1
        return result


@dataclass
class RuleHypothesis:
    """Hypothesis about a potential rule"""
    hypothesis_id: str
    rule_type: RuleType
    evidence: List[Tuple[Any, Any]]  # (input, output) pairs
    confidence: float
    description: str
    
    # Hypothesis testing
    supporting_evidence: int = 0
    contradicting_evidence: int = 0
    test_results: List[bool] = field(default_factory=list)
    
    def add_evidence(self, input_example: Any, output_example: Any, supports: bool):
        """Add evidence for or against hypothesis"""
        self.evidence.append((input_example, output_example))
        self.test_results.append(supports)
        
        if supports:
            self.supporting_evidence += 1
        else:
            self.contradicting_evidence += 1
        
        # Update confidence based on evidence ratio
        total_evidence = self.supporting_evidence + self.contradicting_evidence
        if total_evidence > 0:
            self.confidence = self.supporting_evidence / total_evidence


class VisualRuleInductionEngine:
    """Advanced system for learning visual transformation rules"""
    
    def __init__(self, cognitive_architecture=None, reasoning_engines=None):
        self.logger = logging.getLogger(__name__)
        
        # Integration with AGI components
        self.cognitive_architecture = cognitive_architecture
        self.reasoning_engines = reasoning_engines or {}
        
        # Rule storage and management
        self.learned_rules: Dict[str, VisualRule] = {}
        self.rule_hierarchy: Dict[str, Set[str]] = defaultdict(set)
        self.active_hypotheses: Dict[str, RuleHypothesis] = {}
        
        # Learning parameters
        self.min_examples_for_rule = 3
        self.min_confidence_for_rule = 0.6
        self.max_hypotheses = 100
        
        # Meta-learning state
        self.rule_success_patterns: Dict[str, float] = {}
        self.context_preferences: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Feature extractor for rule analysis
        self.feature_extractor = VisualFeatureExtractor()
    
    def learn_from_examples(self, examples: List[Tuple[np.ndarray, np.ndarray]], 
                          context: Dict[str, Any] = None) -> List[VisualRule]:
        """Learn visual transformation rules from input-output examples"""
        try:
            self.logger.info(f"Learning from {len(examples)} examples")
            
            # Phase 1: Generate hypotheses
            hypotheses = self._generate_hypotheses(examples, context or {})
            
            # Phase 2: Test and validate hypotheses
            validated_hypotheses = self._validate_hypotheses(hypotheses, examples)
            
            # Phase 3: Convert best hypotheses to rules
            new_rules = self._hypotheses_to_rules(validated_hypotheses, examples)
            
            # Phase 4: Integrate with existing knowledge
            integrated_rules = self._integrate_rules(new_rules)
            
            # Phase 5: Meta-learning updates
            self._update_meta_knowledge(integrated_rules, examples, context or {})
            
            return integrated_rules
            
        except Exception as e:
            self.logger.error(f"Error learning from examples: {e}")
            return []
    
    def apply_rules(self, input_grid: np.ndarray, 
                   context: Dict[str, Any] = None) -> List[Tuple[np.ndarray, VisualRule, float]]:
        """Apply learned rules to generate possible outputs"""
        try:
            context = context or {}
            results = []
            
            # Find applicable rules
            applicable_rules = []
            for rule in self.learned_rules.values():
                applicability = rule.applies_to(context)
                if applicability > 0.3:  # Threshold for consideration
                    applicable_rules.append((rule, applicability))
            
            # Sort by applicability
            applicable_rules.sort(key=lambda x: x[1], reverse=True)
            
            # Apply top rules
            for rule, applicability in applicable_rules[:5]:  # Top 5 rules
                try:
                    output = self._apply_visual_rule(input_grid, rule, context)
                    if output is not None:
                        results.append((output, rule, applicability))
                except Exception as e:
                    self.logger.warning(f"Failed to apply rule {rule.rule_id}: {e}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error applying rules: {e}")
            return []
    
    def explain_rule_application(self, input_grid: np.ndarray, output_grid: np.ndarray,
                               rule: VisualRule) -> Dict[str, Any]:
        """Generate explanation for how a rule was applied"""
        try:
            explanation = {
                'rule_id': rule.rule_id,
                'rule_type': rule.rule_type.value,
                'confidence': rule.confidence,
                'description': f"Applied {rule.rule_type.value} rule with {len(rule.conditions)} conditions",
                'steps': [],
                'reasoning': []
            }
            
            # Analyze what changed
            differences = self._analyze_grid_differences(input_grid, output_grid)
            explanation['changes_detected'] = differences
            
            # Explain each condition
            for i, condition in enumerate(rule.conditions):
                explanation['steps'].append(f"Step {i+1}: {condition.description}")
            
            # Explain each action
            for i, action in enumerate(rule.actions):
                explanation['steps'].append(f"Action {i+1}: {action.action_type}")
            
            # Add reasoning if available
            if self.reasoning_engines.get('abstract'):
                reasoning_result = self.reasoning_engines['abstract'].reason_about_transformation(
                    input_grid, output_grid
                )
                explanation['reasoning'] = reasoning_result
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"Error explaining rule application: {e}")
            return {'error': str(e)}
    
    def get_rule_insights(self) -> Dict[str, Any]:
        """Get insights about learned rules for consciousness integration"""
        try:
            insights = {
                'total_rules': len(self.learned_rules),
                'rule_types': Counter(rule.rule_type.value for rule in self.learned_rules.values()),
                'confidence_distribution': {},
                'complexity_analysis': {},
                'success_patterns': self.rule_success_patterns.copy(),
                'most_applicable_rules': [],
                'learning_trends': {}
            }
            
            if not self.learned_rules:
                return insights
            
            # Confidence distribution
            confidences = [rule.confidence for rule in self.learned_rules.values()]
            insights['confidence_distribution'] = {
                'mean': float(np.mean(confidences)),
                'std': float(np.std(confidences)),
                'min': float(np.min(confidences)),
                'max': float(np.max(confidences))
            }
            
            # Complexity analysis
            complexities = [rule.complexity for rule in self.learned_rules.values()]
            insights['complexity_analysis'] = {
                'mean': float(np.mean(complexities)),
                'simple_rules': sum(1 for c in complexities if c <= 2),
                'complex_rules': sum(1 for c in complexities if c >= 5)
            }
            
            # Most applicable rules
            rule_scores = []
            for rule in self.learned_rules.values():
                score = rule.confidence * rule.generality * (rule.success_rate ** 2)
                rule_scores.append((rule.rule_id, rule.rule_type.value, score))
            
            rule_scores.sort(key=lambda x: x[2], reverse=True)
            insights['most_applicable_rules'] = rule_scores[:5]
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error generating rule insights: {e}")
            return {'error': str(e)}
    
    def _generate_hypotheses(self, examples: List[Tuple[np.ndarray, np.ndarray]], 
                           context: Dict[str, Any]) -> List[RuleHypothesis]:
        """Generate rule hypotheses from examples"""
        hypotheses = []
        
        try:
            # Hypothesis 1: Direct transformation patterns
            geometric_hypotheses = self._generate_geometric_hypotheses(examples)
            hypotheses.extend(geometric_hypotheses)
            
            # Hypothesis 2: Color transformation patterns
            color_hypotheses = self._generate_color_hypotheses(examples)
            hypotheses.extend(color_hypotheses)
            
            # Hypothesis 3: Pattern completion hypotheses
            pattern_hypotheses = self._generate_pattern_hypotheses(examples)
            hypotheses.extend(pattern_hypotheses)
            
            # Hypothesis 4: Conditional transformation hypotheses
            conditional_hypotheses = self._generate_conditional_hypotheses(examples)
            hypotheses.extend(conditional_hypotheses)
            
            # Hypothesis 5: Abstract reasoning hypotheses
            if self.reasoning_engines.get('abstract'):
                abstract_hypotheses = self._generate_abstract_hypotheses(examples)
                hypotheses.extend(abstract_hypotheses)
            
            # Limit number of hypotheses
            if len(hypotheses) > self.max_hypotheses:
                hypotheses.sort(key=lambda h: h.confidence, reverse=True)
                hypotheses = hypotheses[:self.max_hypotheses]
            
            self.logger.info(f"Generated {len(hypotheses)} hypotheses")
            return hypotheses
            
        except Exception as e:
            self.logger.error(f"Error generating hypotheses: {e}")
            return []
    
    def _generate_geometric_hypotheses(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> List[RuleHypothesis]:
        """Generate hypotheses about geometric transformations"""
        hypotheses = []
        
        # Test common geometric transformations
        transformations = [
            (TransformationType.ROTATION_90, "90-degree rotation"),
            (TransformationType.ROTATION_180, "180-degree rotation"),
            (TransformationType.ROTATION_270, "270-degree rotation"),
            (TransformationType.REFLECTION_HORIZONTAL, "horizontal reflection"),
            (TransformationType.REFLECTION_VERTICAL, "vertical reflection"),
        ]
        
        for transform_type, description in transformations:
            supporting_count = 0
            
            for input_grid, output_grid in examples:
                # Test if transformation explains input->output
                if self._test_transformation(input_grid, output_grid, transform_type):
                    supporting_count += 1
            
            if supporting_count > 0:
                confidence = supporting_count / len(examples)
                
                hypothesis = RuleHypothesis(
                    hypothesis_id=f"geo_{transform_type.value}_{len(hypotheses)}",
                    rule_type=RuleType.GEOMETRIC_TRANSFORM,
                    evidence=examples,
                    confidence=confidence,
                    description=f"Geometric transformation: {description}",
                    supporting_evidence=supporting_count,
                    contradicting_evidence=len(examples) - supporting_count
                )
                hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _generate_color_hypotheses(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> List[RuleHypothesis]:
        """Generate hypotheses about color transformations"""
        hypotheses = []
        
        # Analyze color changes across examples
        color_mappings = defaultdict(list)
        
        for input_grid, output_grid in examples:
            input_colors = set(input_grid.flatten())
            output_colors = set(output_grid.flatten())
            
            # Simple color mapping detection
            if input_grid.shape == output_grid.shape:
                for i in range(input_grid.shape[0]):
                    for j in range(input_grid.shape[1]):
                        input_color = input_grid[i, j]
                        output_color = output_grid[i, j]
                        color_mappings[input_color].append(output_color)
        
        # Generate hypotheses from consistent mappings
        for input_color, output_colors in color_mappings.items():
            if len(output_colors) >= 2:  # Need multiple examples
                most_common_output = Counter(output_colors).most_common(1)[0]
                if most_common_output[1] >= len(examples) // 2:  # Majority support
                    
                    confidence = most_common_output[1] / len(output_colors)
                    
                    hypothesis = RuleHypothesis(
                        hypothesis_id=f"color_{input_color}_to_{most_common_output[0]}_{len(hypotheses)}",
                        rule_type=RuleType.COLOR_CHANGE,
                        evidence=examples,
                        confidence=confidence,
                        description=f"Color change: {input_color} -> {most_common_output[0]}",
                        supporting_evidence=most_common_output[1],
                        contradicting_evidence=len(output_colors) - most_common_output[1]
                    )
                    hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _generate_pattern_hypotheses(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> List[RuleHypothesis]:
        """Generate hypotheses about pattern transformations"""
        hypotheses = []
        
        # Pattern completion hypothesis
        completion_evidence = 0
        
        for input_grid, output_grid in examples:
            # Simple pattern completion check
            if self._suggests_pattern_completion(input_grid, output_grid):
                completion_evidence += 1
        
        if completion_evidence > 0:
            confidence = completion_evidence / len(examples)
            
            hypothesis = RuleHypothesis(
                hypothesis_id=f"pattern_completion_{len(hypotheses)}",
                rule_type=RuleType.PATTERN_COMPLETION,
                evidence=examples,
                confidence=confidence,
                description="Pattern completion transformation",
                supporting_evidence=completion_evidence,
                contradicting_evidence=len(examples) - completion_evidence
            )
            hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _generate_conditional_hypotheses(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> List[RuleHypothesis]:
        """Generate hypotheses about conditional transformations"""
        hypotheses = []
        
        # Context-dependent transformation hypothesis
        context_evidence = 0
        
        for input_grid, output_grid in examples:
            # Check if transformation depends on context (simplified)
            if self._suggests_conditional_transform(input_grid, output_grid):
                context_evidence += 1
        
        if context_evidence > 0:
            confidence = context_evidence / len(examples)
            
            hypothesis = RuleHypothesis(
                hypothesis_id=f"conditional_{len(hypotheses)}",
                rule_type=RuleType.CONDITIONAL_TRANSFORM,
                evidence=examples,
                confidence=confidence,
                description="Context-dependent transformation",
                supporting_evidence=context_evidence,
                contradicting_evidence=len(examples) - context_evidence
            )
            hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _generate_abstract_hypotheses(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> List[RuleHypothesis]:
        """Generate abstract reasoning hypotheses using AGI reasoning engine"""
        hypotheses = []
        
        try:
            # Use abstract reasoning engine to identify patterns
            abstract_reasoner = self.reasoning_engines['abstract']
            
            for i, (input_grid, output_grid) in enumerate(examples[:3]):  # Limit for performance
                reasoning_result = abstract_reasoner.analyze_transformation(input_grid, output_grid)
                
                if reasoning_result.get('confidence', 0) > 0.5:
                    hypothesis = RuleHypothesis(
                        hypothesis_id=f"abstract_{i}_{len(hypotheses)}",
                        rule_type=RuleType.ANALOGY_BASED,
                        evidence=[examples[i]],
                        confidence=reasoning_result.get('confidence', 0.5),
                        description=reasoning_result.get('description', 'Abstract pattern'),
                        supporting_evidence=1,
                        contradicting_evidence=0
                    )
                    hypotheses.append(hypothesis)
        
        except Exception as e:
            self.logger.warning(f"Abstract hypothesis generation failed: {e}")
        
        return hypotheses
    
    def _validate_hypotheses(self, hypotheses: List[RuleHypothesis], 
                           examples: List[Tuple[np.ndarray, np.ndarray]]) -> List[RuleHypothesis]:
        """Validate hypotheses against examples"""
        validated = []
        
        for hypothesis in hypotheses:
            if hypothesis.confidence >= self.min_confidence_for_rule:
                # Additional validation could be added here
                validated.append(hypothesis)
        
        return validated
    
    def _hypotheses_to_rules(self, hypotheses: List[RuleHypothesis], 
                           examples: List[Tuple[np.ndarray, np.ndarray]]) -> List[VisualRule]:
        """Convert validated hypotheses to visual rules"""
        rules = []
        
        for hypothesis in hypotheses:
            rule = self._create_rule_from_hypothesis(hypothesis, examples)
            if rule:
                rules.append(rule)
        
        return rules
    
    def _create_rule_from_hypothesis(self, hypothesis: RuleHypothesis,
                                   examples: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[VisualRule]:
        """Create a visual rule from a validated hypothesis"""
        try:
            # Create conditions based on hypothesis type
            conditions = self._create_conditions_for_hypothesis(hypothesis)
            
            # Create actions based on hypothesis type
            actions = self._create_actions_for_hypothesis(hypothesis)
            
            # Calculate rule properties
            generality = min(1.0, hypothesis.supporting_evidence / len(examples))
            specificity = 1.0 - generality
            complexity = len(conditions) + len(actions)
            
            rule = VisualRule(
                rule_id=f"rule_{hypothesis.hypothesis_id}",
                rule_type=hypothesis.rule_type,
                conditions=conditions,
                actions=actions,
                confidence=hypothesis.confidence,
                generality=generality,
                specificity=specificity,
                complexity=complexity,
                training_examples=examples.copy()
            )
            
            return rule
            
        except Exception as e:
            self.logger.error(f"Error creating rule from hypothesis: {e}")
            return None
    
    def _create_conditions_for_hypothesis(self, hypothesis: RuleHypothesis) -> List[RuleCondition]:
        """Create conditions for a hypothesis"""
        conditions = []
        
        if hypothesis.rule_type == RuleType.GEOMETRIC_TRANSFORM:
            condition = RuleCondition(
                condition_type="has_transformable_structure",
                parameters={"min_objects": 1},
                confidence=0.8,
                description="Grid contains transformable structures"
            )
            conditions.append(condition)
        
        elif hypothesis.rule_type == RuleType.COLOR_CHANGE:
            condition = RuleCondition(
                condition_type="contains_target_color",
                parameters={"target_colors": []},  # Would be filled from hypothesis
                confidence=0.9,
                description="Grid contains colors that can be changed"
            )
            conditions.append(condition)
        
        return conditions
    
    def _create_actions_for_hypothesis(self, hypothesis: RuleHypothesis) -> List[RuleAction]:
        """Create actions for a hypothesis"""
        actions = []
        
        if hypothesis.rule_type == RuleType.GEOMETRIC_TRANSFORM:
            action = RuleAction(
                action_type="apply_geometric_transform",
                parameters={"transform_type": "rotation_90"},  # Would be determined from hypothesis
                expected_outcome="transformed_grid"
            )
            actions.append(action)
        
        elif hypothesis.rule_type == RuleType.COLOR_CHANGE:
            action = RuleAction(
                action_type="apply_color_mapping",
                parameters={"color_map": {}},  # Would be filled from hypothesis
                expected_outcome="recolored_grid"
            )
            actions.append(action)
        
        return actions
    
    def _integrate_rules(self, new_rules: List[VisualRule]) -> List[VisualRule]:
        """Integrate new rules with existing knowledge"""
        integrated_rules = []
        
        for rule in new_rules:
            # Check for conflicts with existing rules
            conflicts = self._find_rule_conflicts(rule)
            
            if not conflicts:
                # Add to learned rules
                self.learned_rules[rule.rule_id] = rule
                integrated_rules.append(rule)
            else:
                # Handle conflicts (merge, update, or reject)
                resolved_rule = self._resolve_rule_conflicts(rule, conflicts)
                if resolved_rule:
                    self.learned_rules[resolved_rule.rule_id] = resolved_rule
                    integrated_rules.append(resolved_rule)
        
        # Update rule hierarchy
        self._update_rule_hierarchy(integrated_rules)
        
        return integrated_rules
    
    def _update_meta_knowledge(self, rules: List[VisualRule], 
                             examples: List[Tuple[np.ndarray, np.ndarray]],
                             context: Dict[str, Any]):
        """Update meta-learning knowledge"""
        for rule in rules:
            # Track success patterns
            self.rule_success_patterns[rule.rule_type.value] = rule.confidence
            
            # Update context preferences
            for ctx_key, ctx_value in context.items():
                if isinstance(ctx_value, (str, int, float)):
                    self.context_preferences[rule.rule_type.value][str(ctx_value)] = rule.confidence
    
    # Helper methods for specific tests
    
    def _test_transformation(self, input_grid: np.ndarray, output_grid: np.ndarray,
                           transform_type: TransformationType) -> bool:
        """Test if a transformation explains the input->output mapping"""
        try:
            if transform_type == TransformationType.ROTATION_90:
                expected = np.rot90(input_grid, 1)
            elif transform_type == TransformationType.ROTATION_180:
                expected = np.rot90(input_grid, 2)
            elif transform_type == TransformationType.ROTATION_270:
                expected = np.rot90(input_grid, 3)
            elif transform_type == TransformationType.REFLECTION_HORIZONTAL:
                expected = np.flipud(input_grid)
            elif transform_type == TransformationType.REFLECTION_VERTICAL:
                expected = np.fliplr(input_grid)
            else:
                return False
            
            return np.array_equal(expected, output_grid)
            
        except Exception:
            return False
    
    def _suggests_pattern_completion(self, input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
        """Check if the transformation suggests pattern completion"""
        # Simplified: check if output has more non-zero elements than input
        input_nonzero = np.count_nonzero(input_grid)
        output_nonzero = np.count_nonzero(output_grid)
        
        return output_nonzero > input_nonzero
    
    def _suggests_conditional_transform(self, input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
        """Check if transformation appears to be context-dependent"""
        # Simplified: check if transformation preserves some elements but changes others
        if input_grid.shape != output_grid.shape:
            return False
        
        preserved = np.sum(input_grid == output_grid)
        total = input_grid.size
        
        # Context-dependent if some but not all elements are preserved
        return 0.1 < (preserved / total) < 0.9
    
    def _apply_visual_rule(self, input_grid: np.ndarray, rule: VisualRule, 
                          context: Dict[str, Any]) -> Optional[np.ndarray]:
        """Apply a visual rule to generate output"""
        try:
            # This is a simplified implementation
            # In practice, would execute the specific actions defined in the rule
            
            if rule.rule_type == RuleType.GEOMETRIC_TRANSFORM:
                # Apply geometric transformation
                for action in rule.actions:
                    if action.action_type == "apply_geometric_transform":
                        transform_type = action.parameters.get("transform_type", "rotation_90")
                        
                        if transform_type == "rotation_90":
                            return np.rot90(input_grid, 1)
                        elif transform_type == "rotation_180":
                            return np.rot90(input_grid, 2)
                        elif transform_type == "rotation_270":
                            return np.rot90(input_grid, 3)
                        elif transform_type == "reflection_horizontal":
                            return np.flipud(input_grid)
                        elif transform_type == "reflection_vertical":
                            return np.fliplr(input_grid)
            
            elif rule.rule_type == RuleType.COLOR_CHANGE:
                # Apply color changes
                result = input_grid.copy()
                for action in rule.actions:
                    if action.action_type == "apply_color_mapping":
                        color_map = action.parameters.get("color_map", {})
                        for old_color, new_color in color_map.items():
                            result[result == old_color] = new_color
                return result
            
            return input_grid  # Fallback
            
        except Exception as e:
            self.logger.error(f"Error applying visual rule: {e}")
            return None
    
    def _analyze_grid_differences(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Dict[str, Any]:
        """Analyze differences between input and output grids"""
        differences = {
            'shape_changed': input_grid.shape != output_grid.shape,
            'pixels_changed': 0,
            'colors_added': [],
            'colors_removed': [],
            'transformations_detected': []
        }
        
        if input_grid.shape == output_grid.shape:
            differences['pixels_changed'] = np.sum(input_grid != output_grid)
            
            input_colors = set(input_grid.flatten())
            output_colors = set(output_grid.flatten())
            
            differences['colors_added'] = list(output_colors - input_colors)
            differences['colors_removed'] = list(input_colors - output_colors)
        
        return differences
    
    def _find_rule_conflicts(self, rule: VisualRule) -> List[VisualRule]:
        """Find conflicts with existing rules"""
        conflicts = []
        
        for existing_rule in self.learned_rules.values():
            if (existing_rule.rule_type == rule.rule_type and
                self._rules_conflict(existing_rule, rule)):
                conflicts.append(existing_rule)
        
        return conflicts
    
    def _rules_conflict(self, rule1: VisualRule, rule2: VisualRule) -> bool:
        """Check if two rules conflict"""
        # Simplified conflict detection
        return (rule1.rule_type == rule2.rule_type and 
                abs(rule1.confidence - rule2.confidence) > 0.5)
    
    def _resolve_rule_conflicts(self, new_rule: VisualRule, 
                              conflicts: List[VisualRule]) -> Optional[VisualRule]:
        """Resolve conflicts between rules"""
        # Simple resolution: choose the rule with higher confidence
        best_rule = new_rule
        
        for conflict_rule in conflicts:
            if conflict_rule.confidence > best_rule.confidence:
                best_rule = conflict_rule
        
        return best_rule
    
    def _update_rule_hierarchy(self, rules: List[VisualRule]):
        """Update rule hierarchy relationships"""
        for rule in rules:
            # Simple hierarchy: more specific rules are children of more general ones
            for existing_rule in self.learned_rules.values():
                if (existing_rule.rule_id != rule.rule_id and
                    existing_rule.generality > rule.generality and
                    existing_rule.rule_type == rule.rule_type):
                    
                    self.rule_hierarchy[existing_rule.rule_id].add(rule.rule_id)
                    rule.parent_rules.append(existing_rule.rule_id)
                    existing_rule.child_rules.append(rule.rule_id)