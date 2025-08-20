"""
Meta-Learning for Visual Rules in AGI-LLM

Advanced meta-learning system that learns to learn visual patterns:
- Learns optimal strategies for different types of visual problems
- Adapts learning approaches based on context and domain
- Transfers knowledge across different visual tasks
- Integrates with consciousness for strategic learning decisions
- Builds meta-cognitive awareness of learning processes

This enables our AGI-LLM to become increasingly effective at visual reasoning.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, deque
import time
import copy

from .rule_induction import VisualRule, RuleType, VisualRuleInductionEngine, RuleHypothesis
from .grid_processor import GridRepresentation
from .pattern_detector import VisualPattern, PatternType


class LearningStrategy(Enum):
    """Different learning strategies for visual rules"""
    CONSERVATIVE = "conservative"    # Focus on high-confidence patterns
    EXPLORATORY = "exploratory"     # Try many diverse hypotheses
    BALANCED = "balanced"           # Mix of conservative and exploratory
    ADAPTIVE = "adaptive"           # Change strategy based on context
    HIERARCHICAL = "hierarchical"   # Build from simple to complex
    ANALOGICAL = "analogical"       # Use analogies and transfer learning
    COMPOSITIONAL = "compositional" # Combine simple rules into complex ones


class LearningContext(Enum):
    """Different contexts that affect learning"""
    GEOMETRIC_TRANSFORMS = "geometric_transforms"
    PATTERN_COMPLETION = "pattern_completion"
    COLOR_TRANSFORMATIONS = "color_transformations"
    COMPLEX_REASONING = "complex_reasoning"
    NOVEL_DOMAIN = "novel_domain"
    FAMILIAR_DOMAIN = "familiar_domain"
    TIME_CONSTRAINED = "time_constrained"
    HIGH_ACCURACY_REQUIRED = "high_accuracy_required"


class MetaCognitiveFeedback(Enum):
    """Types of meta-cognitive feedback"""
    CONFIDENCE_TOO_LOW = "confidence_too_low"
    OVERFITTING = "overfitting"
    UNDERFITTING = "underfitting"
    STRATEGY_WORKING = "strategy_working"
    NEED_MORE_EXAMPLES = "need_more_examples"
    TRANSFER_OPPORTUNITY = "transfer_opportunity"
    COMPLEXITY_MISMATCH = "complexity_mismatch"


@dataclass
class LearningEpisode:
    """Record of a learning episode"""
    episode_id: str
    context: LearningContext
    strategy_used: LearningStrategy
    examples_count: int
    rules_learned: int
    avg_confidence: float
    learning_time: float
    success_metrics: Dict[str, float]
    
    # Feedback and outcomes
    final_performance: float
    meta_feedback: List[MetaCognitiveFeedback]
    consciousness_insights: List[str]
    
    # Strategy effectiveness
    strategy_effectiveness: float = 0.0
    would_use_again: bool = True


@dataclass
class StrategyPerformanceTracker:
    """Tracks performance of different strategies"""
    strategy: LearningStrategy
    context_performance: Dict[LearningContext, List[float]] = field(default_factory=lambda: defaultdict(list))
    total_uses: int = 0
    total_success: float = 0.0
    avg_performance: float = 0.0
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    
    def update_performance(self, context: LearningContext, performance: float):
        """Update performance tracking"""
        self.context_performance[context].append(performance)
        self.total_uses += 1
        self.total_success += performance
        self.avg_performance = self.total_success / self.total_uses
    
    def get_context_performance(self, context: LearningContext) -> float:
        """Get average performance for specific context"""
        performances = self.context_performance[context]
        return np.mean(performances) if performances else 0.0


@dataclass
class MetaKnowledge:
    """Meta-knowledge about learning"""
    # Strategy preferences
    strategy_preferences: Dict[LearningContext, LearningStrategy] = field(default_factory=dict)
    
    # Performance patterns
    context_difficulty: Dict[LearningContext, float] = field(default_factory=dict)
    transfer_opportunities: Dict[Tuple[LearningContext, LearningContext], float] = field(default_factory=dict)
    
    # Learning patterns
    optimal_example_counts: Dict[LearningContext, int] = field(default_factory=dict)
    complexity_progression: Dict[LearningContext, List[RuleType]] = field(default_factory=dict)
    
    # Meta-cognitive insights
    learning_biases: List[str] = field(default_factory=list)
    success_indicators: List[str] = field(default_factory=list)
    failure_patterns: List[str] = field(default_factory=list)


class VisualMetaLearner:
    """Meta-learning system for visual rule learning"""
    
    def __init__(self, rule_induction_engine: VisualRuleInductionEngine, 
                 cognitive_architecture=None):
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.rule_engine = rule_induction_engine
        self.cognitive_architecture = cognitive_architecture
        
        # Meta-learning state
        self.meta_knowledge = MetaKnowledge()
        self.strategy_trackers = {
            strategy: StrategyPerformanceTracker(strategy) 
            for strategy in LearningStrategy
        }
        
        # Learning history
        self.learning_episodes: List[LearningEpisode] = []
        self.recent_episodes = deque(maxlen=50)  # Keep recent episodes for analysis
        
        # Current learning state
        self.current_strategy = LearningStrategy.BALANCED
        self.adaptation_threshold = 0.1  # When to adapt strategy
        self.min_episodes_for_adaptation = 5
        
        # Meta-cognitive monitoring
        self.meta_cognitive_active = False
        self.consciousness_integration_enabled = True
        
        # Transfer learning
        self.domain_knowledge: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.cross_domain_mappings: Dict[Tuple[str, str], float] = {}
    
    def meta_learn_from_session(self, examples: List[Tuple[np.ndarray, np.ndarray]], 
                               context: LearningContext,
                               strategy: Optional[LearningStrategy] = None) -> Dict[str, Any]:
        """Perform meta-learning from a visual learning session"""
        try:
            session_start = time.time()
            
            # Phase 1: Strategy selection
            selected_strategy = strategy or self._select_optimal_strategy(context)
            
            # Phase 2: Conscious preparation if enabled
            preparation_insights = self._prepare_conscious_learning(context, selected_strategy)
            
            # Phase 3: Execute learning with selected strategy
            learning_results = self._execute_strategy(examples, context, selected_strategy)
            
            # Phase 4: Meta-cognitive evaluation
            episode_feedback = self._evaluate_learning_episode(
                learning_results, context, selected_strategy, session_start
            )
            
            # Phase 5: Update meta-knowledge
            self._update_meta_knowledge(episode_feedback)
            
            # Phase 6: Consider strategy adaptation
            self._consider_strategy_adaptation(context)
            
            # Phase 7: Generate consciousness insights
            consciousness_insights = self._generate_consciousness_insights(episode_feedback)
            
            return {
                'strategy_used': selected_strategy.value,
                'rules_learned': learning_results.get('rules_count', 0),
                'learning_effectiveness': episode_feedback.strategy_effectiveness,
                'meta_feedback': [f.value for f in episode_feedback.meta_feedback],
                'consciousness_insights': consciousness_insights,
                'preparation_insights': preparation_insights,
                'transfer_opportunities': self._identify_transfer_opportunities(context),
                'recommended_next_strategy': self._recommend_next_strategy(context)
            }
            
        except Exception as e:
            self.logger.error(f"Meta-learning session failed: {e}")
            return {'error': str(e)}
    
    def _select_optimal_strategy(self, context: LearningContext) -> LearningStrategy:
        """Select optimal learning strategy for context"""
        try:
            # Check if we have a learned preference
            if context in self.meta_knowledge.strategy_preferences:
                preferred = self.meta_knowledge.strategy_preferences[context]
                
                # Verify the preference is still valid
                tracker = self.strategy_trackers[preferred]
                if tracker.get_context_performance(context) > 0.6:
                    return preferred
            
            # Find best performing strategy for this context
            best_strategy = LearningStrategy.BALANCED
            best_performance = 0.0
            
            for strategy, tracker in self.strategy_trackers.items():
                performance = tracker.get_context_performance(context)
                if performance > best_performance:
                    best_performance = performance
                    best_strategy = strategy
            
            # If no strong evidence, use context-appropriate defaults
            if best_performance < 0.3:
                best_strategy = self._get_default_strategy_for_context(context)
            
            return best_strategy
            
        except Exception as e:
            self.logger.error(f"Strategy selection failed: {e}")
            return LearningStrategy.BALANCED
    
    def _get_default_strategy_for_context(self, context: LearningContext) -> LearningStrategy:
        """Get default strategy for context"""
        defaults = {
            LearningContext.GEOMETRIC_TRANSFORMS: LearningStrategy.CONSERVATIVE,
            LearningContext.PATTERN_COMPLETION: LearningStrategy.HIERARCHICAL,
            LearningContext.COLOR_TRANSFORMATIONS: LearningStrategy.BALANCED,
            LearningContext.COMPLEX_REASONING: LearningStrategy.ANALOGICAL,
            LearningContext.NOVEL_DOMAIN: LearningStrategy.EXPLORATORY,
            LearningContext.FAMILIAR_DOMAIN: LearningStrategy.CONSERVATIVE,
            LearningContext.TIME_CONSTRAINED: LearningStrategy.CONSERVATIVE,
            LearningContext.HIGH_ACCURACY_REQUIRED: LearningStrategy.CONSERVATIVE
        }
        
        return defaults.get(context, LearningStrategy.BALANCED)
    
    def _prepare_conscious_learning(self, context: LearningContext, 
                                  strategy: LearningStrategy) -> List[str]:
        """Prepare consciousness for learning session"""
        insights = []
        
        if not (self.consciousness_integration_enabled and self.cognitive_architecture):
            return insights
        
        try:
            # Add learning context to consciousness
            if hasattr(self.cognitive_architecture, 'consciousness'):
                consciousness = self.cognitive_architecture.consciousness
                
                consciousness.add_to_consciousness(
                    content=f"Preparing to learn visual rules in {context.value} context",
                    content_type="learning_intention",
                    activation_strength=0.8,
                    phenomenal_properties={
                        'context': context.value,
                        'strategy': strategy.value,
                        'meta_learning': True
                    }
                )
                
                insights.append(f"Consciousness primed for {context.value} learning")
                insights.append(f"Strategy selected: {strategy.value}")
                
                # Add relevant prior knowledge to consciousness
                relevant_knowledge = self._retrieve_relevant_meta_knowledge(context)
                if relevant_knowledge:
                    consciousness.add_to_consciousness(
                        content=f"Relevant prior knowledge: {len(relevant_knowledge)} insights",
                        content_type="prior_knowledge",
                        activation_strength=0.6
                    )
                    insights.extend(relevant_knowledge[:3])  # Top 3 insights
        
        except Exception as e:
            self.logger.warning(f"Consciousness preparation failed: {e}")
        
        return insights
    
    def _execute_strategy(self, examples: List[Tuple[np.ndarray, np.ndarray]], 
                         context: LearningContext, 
                         strategy: LearningStrategy) -> Dict[str, Any]:
        """Execute learning with specified strategy"""
        try:
            # Configure rule engine based on strategy
            original_config = self._configure_rule_engine_for_strategy(strategy, context)
            
            # Execute learning
            if strategy == LearningStrategy.CONSERVATIVE:
                results = self._execute_conservative_learning(examples, context)
            elif strategy == LearningStrategy.EXPLORATORY:
                results = self._execute_exploratory_learning(examples, context)
            elif strategy == LearningStrategy.HIERARCHICAL:
                results = self._execute_hierarchical_learning(examples, context)
            elif strategy == LearningStrategy.ANALOGICAL:
                results = self._execute_analogical_learning(examples, context)
            elif strategy == LearningStrategy.COMPOSITIONAL:
                results = self._execute_compositional_learning(examples, context)
            elif strategy == LearningStrategy.ADAPTIVE:
                results = self._execute_adaptive_learning(examples, context)
            else:  # BALANCED
                results = self._execute_balanced_learning(examples, context)
            
            # Restore original configuration
            self._restore_rule_engine_config(original_config)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Strategy execution failed: {e}")
            return {'rules_count': 0, 'confidence': 0.0}
    
    def _execute_conservative_learning(self, examples: List[Tuple[np.ndarray, np.ndarray]], 
                                     context: LearningContext) -> Dict[str, Any]:
        """Execute conservative learning strategy"""
        # Conservative: High confidence threshold, fewer hypotheses
        self.rule_engine.min_confidence_for_rule = 0.8
        self.rule_engine.max_hypotheses = 20
        
        learned_rules = self.rule_engine.learn_from_examples(examples, {'context': context.value})
        
        return {
            'rules_count': len(learned_rules),
            'confidence': np.mean([r.confidence for r in learned_rules]) if learned_rules else 0.0,
            'rules': learned_rules
        }
    
    def _execute_exploratory_learning(self, examples: List[Tuple[np.ndarray, np.ndarray]], 
                                    context: LearningContext) -> Dict[str, Any]:
        """Execute exploratory learning strategy"""
        # Exploratory: Lower confidence threshold, more hypotheses
        self.rule_engine.min_confidence_for_rule = 0.4
        self.rule_engine.max_hypotheses = 100
        
        learned_rules = self.rule_engine.learn_from_examples(examples, {'context': context.value})
        
        return {
            'rules_count': len(learned_rules),
            'confidence': np.mean([r.confidence for r in learned_rules]) if learned_rules else 0.0,
            'rules': learned_rules,
            'exploration_breadth': len(learned_rules)
        }
    
    def _execute_hierarchical_learning(self, examples: List[Tuple[np.ndarray, np.ndarray]], 
                                     context: LearningContext) -> Dict[str, Any]:
        """Execute hierarchical learning strategy"""
        # Hierarchical: Learn simple rules first, then build complexity
        all_rules = []
        
        # Phase 1: Simple rules
        self.rule_engine.min_confidence_for_rule = 0.7
        self.rule_engine.max_hypotheses = 30
        
        simple_rules = self.rule_engine.learn_from_examples(examples, {'context': context.value, 'phase': 'simple'})
        all_rules.extend(simple_rules)
        
        # Phase 2: Complex rules building on simple ones
        if simple_rules:
            self.rule_engine.min_confidence_for_rule = 0.6
            self.rule_engine.max_hypotheses = 50
            
            complex_rules = self.rule_engine.learn_from_examples(examples, {'context': context.value, 'phase': 'complex'})
            all_rules.extend(complex_rules)
        
        return {
            'rules_count': len(all_rules),
            'confidence': np.mean([r.confidence for r in all_rules]) if all_rules else 0.0,
            'rules': all_rules,
            'hierarchy_levels': 2
        }
    
    def _execute_analogical_learning(self, examples: List[Tuple[np.ndarray, np.ndarray]], 
                                   context: LearningContext) -> Dict[str, Any]:
        """Execute analogical learning strategy"""
        # Try to use analogies from similar contexts
        analogous_contexts = self._find_analogous_contexts(context)
        
        learned_rules = []
        
        # First try learning with analogies
        for analog_context in analogous_contexts[:2]:  # Top 2 analogies
            analog_rules = self._retrieve_rules_for_context(analog_context)
            if analog_rules:
                # Try to adapt analog rules to current examples
                adapted_rules = self._adapt_rules_to_examples(analog_rules, examples, context)
                learned_rules.extend(adapted_rules)
        
        # Then learn new rules normally
        new_rules = self.rule_engine.learn_from_examples(examples, {'context': context.value})
        learned_rules.extend(new_rules)
        
        return {
            'rules_count': len(learned_rules),
            'confidence': np.mean([r.confidence for r in learned_rules]) if learned_rules else 0.0,
            'rules': learned_rules,
            'analogies_used': len(analogous_contexts)
        }
    
    def _execute_compositional_learning(self, examples: List[Tuple[np.ndarray, np.ndarray]], 
                                      context: LearningContext) -> Dict[str, Any]:
        """Execute compositional learning strategy"""
        # Learn simple components and compose them
        component_rules = []
        
        # Learn component rules with high precision
        self.rule_engine.min_confidence_for_rule = 0.8
        
        basic_rules = self.rule_engine.learn_from_examples(examples, {'context': context.value})
        component_rules.extend(basic_rules)
        
        # Try to compose rules
        composite_rules = self._compose_rules(basic_rules, examples)
        component_rules.extend(composite_rules)
        
        return {
            'rules_count': len(component_rules),
            'confidence': np.mean([r.confidence for r in component_rules]) if component_rules else 0.0,
            'rules': component_rules,
            'compositions_created': len(composite_rules)
        }
    
    def _execute_adaptive_learning(self, examples: List[Tuple[np.ndarray, np.ndarray]], 
                                 context: LearningContext) -> Dict[str, Any]:
        """Execute adaptive learning strategy"""
        # Start with one strategy and adapt based on intermediate results
        current_strategy = LearningStrategy.BALANCED
        
        # Try initial learning
        results = self._execute_balanced_learning(examples, context)
        
        # Adapt based on results
        if results.get('confidence', 0.0) < 0.5:
            # Low confidence - try exploratory
            self.logger.info("Adapting to exploratory strategy due to low confidence")
            results = self._execute_exploratory_learning(examples, context)
        elif len(results.get('rules', [])) > 10:
            # Too many rules - try conservative
            self.logger.info("Adapting to conservative strategy due to too many rules")
            results = self._execute_conservative_learning(examples, context)
        
        return results
    
    def _execute_balanced_learning(self, examples: List[Tuple[np.ndarray, np.ndarray]], 
                                 context: LearningContext) -> Dict[str, Any]:
        """Execute balanced learning strategy"""
        # Balanced: Moderate settings
        self.rule_engine.min_confidence_for_rule = 0.6
        self.rule_engine.max_hypotheses = 50
        
        learned_rules = self.rule_engine.learn_from_examples(examples, {'context': context.value})
        
        return {
            'rules_count': len(learned_rules),
            'confidence': np.mean([r.confidence for r in learned_rules]) if learned_rules else 0.0,
            'rules': learned_rules
        }
    
    def _evaluate_learning_episode(self, learning_results: Dict[str, Any], 
                                 context: LearningContext, 
                                 strategy: LearningStrategy,
                                 session_start: float) -> LearningEpisode:
        """Evaluate and record learning episode"""
        try:
            learning_time = time.time() - session_start
            rules_learned = learning_results.get('rules_count', 0)
            avg_confidence = learning_results.get('confidence', 0.0)
            
            # Calculate effectiveness metrics
            effectiveness = self._calculate_strategy_effectiveness(learning_results, context)
            
            # Generate meta-cognitive feedback
            meta_feedback = self._generate_meta_feedback(learning_results, context, strategy)
            
            # Create episode record
            episode = LearningEpisode(
                episode_id=f"episode_{len(self.learning_episodes)}_{int(time.time())}",
                context=context,
                strategy_used=strategy,
                examples_count=learning_results.get('examples_count', 0),
                rules_learned=rules_learned,
                avg_confidence=avg_confidence,
                learning_time=learning_time,
                success_metrics={
                    'rules_per_second': rules_learned / max(learning_time, 0.001),
                    'confidence_weighted_rules': rules_learned * avg_confidence
                },
                final_performance=effectiveness,
                meta_feedback=meta_feedback,
                consciousness_insights=[],
                strategy_effectiveness=effectiveness
            )
            
            # Add to history
            self.learning_episodes.append(episode)
            self.recent_episodes.append(episode)
            
            return episode
            
        except Exception as e:
            self.logger.error(f"Episode evaluation failed: {e}")
            # Return minimal episode
            return LearningEpisode(
                episode_id="error_episode",
                context=context,
                strategy_used=strategy,
                examples_count=0,
                rules_learned=0,
                avg_confidence=0.0,
                learning_time=0.0,
                success_metrics={},
                final_performance=0.0,
                meta_feedback=[],
                consciousness_insights=[]
            )
    
    def _calculate_strategy_effectiveness(self, learning_results: Dict[str, Any], 
                                        context: LearningContext) -> float:
        """Calculate how effective the strategy was"""
        rules_count = learning_results.get('rules_count', 0)
        confidence = learning_results.get('confidence', 0.0)
        
        # Base effectiveness from rules learned and confidence
        base_effectiveness = min(1.0, (rules_count / 10.0) * confidence)
        
        # Adjust based on context expectations
        context_expectations = {
            LearningContext.GEOMETRIC_TRANSFORMS: {'min_rules': 3, 'min_confidence': 0.8},
            LearningContext.PATTERN_COMPLETION: {'min_rules': 5, 'min_confidence': 0.7},
            LearningContext.COLOR_TRANSFORMATIONS: {'min_rules': 2, 'min_confidence': 0.9},
            LearningContext.COMPLEX_REASONING: {'min_rules': 8, 'min_confidence': 0.6}
        }
        
        expectations = context_expectations.get(context, {'min_rules': 3, 'min_confidence': 0.7})
        
        rules_bonus = min(1.0, rules_count / expectations['min_rules'])
        confidence_bonus = min(1.0, confidence / expectations['min_confidence'])
        
        effectiveness = 0.6 * base_effectiveness + 0.2 * rules_bonus + 0.2 * confidence_bonus
        
        return min(1.0, effectiveness)
    
    def _generate_meta_feedback(self, learning_results: Dict[str, Any], 
                               context: LearningContext, 
                               strategy: LearningStrategy) -> List[MetaCognitiveFeedback]:
        """Generate meta-cognitive feedback about learning"""
        feedback = []
        
        confidence = learning_results.get('confidence', 0.0)
        rules_count = learning_results.get('rules_count', 0)
        
        # Confidence feedback
        if confidence < 0.4:
            feedback.append(MetaCognitiveFeedback.CONFIDENCE_TOO_LOW)
        elif confidence > 0.95:
            feedback.append(MetaCognitiveFeedback.OVERFITTING)
        
        # Rules count feedback
        if rules_count == 0:
            feedback.append(MetaCognitiveFeedback.NEED_MORE_EXAMPLES)
        elif rules_count > 20:
            feedback.append(MetaCognitiveFeedback.OVERFITTING)
        elif 3 <= rules_count <= 8 and confidence > 0.6:
            feedback.append(MetaCognitiveFeedback.STRATEGY_WORKING)
        
        # Context-specific feedback
        if context in [LearningContext.NOVEL_DOMAIN] and rules_count > 0:
            feedback.append(MetaCognitiveFeedback.TRANSFER_OPPORTUNITY)
        
        return feedback
    
    def _update_meta_knowledge(self, episode: LearningEpisode):
        """Update meta-knowledge based on learning episode"""
        try:
            # Update strategy performance
            tracker = self.strategy_trackers[episode.strategy_used]
            tracker.update_performance(episode.context, episode.strategy_effectiveness)
            
            # Update strategy preferences if this was significantly better
            current_best = self.meta_knowledge.strategy_preferences.get(episode.context)
            if (not current_best or 
                episode.strategy_effectiveness > tracker.get_context_performance(episode.context) + 0.1):
                self.meta_knowledge.strategy_preferences[episode.context] = episode.strategy_used
            
            # Update context difficulty
            if episode.context not in self.meta_knowledge.context_difficulty:
                self.meta_knowledge.context_difficulty[episode.context] = episode.strategy_effectiveness
            else:
                # Moving average
                current = self.meta_knowledge.context_difficulty[episode.context]
                self.meta_knowledge.context_difficulty[episode.context] = (
                    0.8 * current + 0.2 * episode.strategy_effectiveness
                )
            
            # Update optimal example counts
            if episode.strategy_effectiveness > 0.7:
                self.meta_knowledge.optimal_example_counts[episode.context] = episode.examples_count
            
            # Learn from feedback
            self._process_meta_feedback(episode)
            
        except Exception as e:
            self.logger.error(f"Meta-knowledge update failed: {e}")
    
    def _process_meta_feedback(self, episode: LearningEpisode):
        """Process meta-cognitive feedback to improve learning"""
        for feedback in episode.meta_feedback:
            if feedback == MetaCognitiveFeedback.CONFIDENCE_TOO_LOW:
                self.meta_knowledge.learning_biases.append(
                    f"Strategy {episode.strategy_used.value} produces low confidence in {episode.context.value}"
                )
            elif feedback == MetaCognitiveFeedback.STRATEGY_WORKING:
                self.meta_knowledge.success_indicators.append(
                    f"Strategy {episode.strategy_used.value} works well for {episode.context.value}"
                )
            elif feedback == MetaCognitiveFeedback.OVERFITTING:
                self.meta_knowledge.failure_patterns.append(
                    f"Overfitting risk with {episode.strategy_used.value} in {episode.context.value}"
                )
    
    def _consider_strategy_adaptation(self, context: LearningContext):
        """Consider if strategy adaptation is needed"""
        if len(self.recent_episodes) < self.min_episodes_for_adaptation:
            return
        
        # Check recent performance in this context
        context_episodes = [ep for ep in self.recent_episodes if ep.context == context]
        
        if len(context_episodes) >= 3:
            recent_performance = np.mean([ep.strategy_effectiveness for ep in context_episodes[-3:]])
            
            # If performance is declining, consider adaptation
            if recent_performance < 0.5:
                self.logger.info(f"Considering strategy adaptation for {context.value} due to declining performance")
                
                # Find best performing strategy for this context
                best_strategy = max(
                    self.strategy_trackers.keys(),
                    key=lambda s: self.strategy_trackers[s].get_context_performance(context)
                )
                
                if best_strategy != self.current_strategy:
                    self.logger.info(f"Adapting from {self.current_strategy.value} to {best_strategy.value}")
                    self.current_strategy = best_strategy
    
    def _generate_consciousness_insights(self, episode: LearningEpisode) -> List[str]:
        """Generate insights for consciousness integration"""
        insights = []
        
        try:
            if self.consciousness_integration_enabled and self.cognitive_architecture:
                # Add learning outcome to consciousness
                if hasattr(self.cognitive_architecture, 'consciousness'):
                    consciousness = self.cognitive_architecture.consciousness
                    
                    consciousness.add_to_consciousness(
                        content=f"Completed learning episode: {episode.rules_learned} rules learned",
                        content_type="learning_outcome",
                        activation_strength=min(1.0, episode.strategy_effectiveness),
                        phenomenal_properties={
                            'context': episode.context.value,
                            'strategy': episode.strategy_used.value,
                            'effectiveness': episode.strategy_effectiveness,
                            'meta_learning': True
                        }
                    )
                    
                    insights.append(f"Learned {episode.rules_learned} visual rules")
                    insights.append(f"Strategy effectiveness: {episode.strategy_effectiveness:.2f}")
                    
                    # Add meta-cognitive insights
                    if episode.meta_feedback:
                        top_feedback = episode.meta_feedback[0].value.replace('_', ' ')
                        insights.append(f"Key insight: {top_feedback}")
                        
                        consciousness.add_to_consciousness(
                            content=f"Meta-cognitive insight: {top_feedback}",
                            content_type="meta_cognition",
                            activation_strength=0.7
                        )
            
        except Exception as e:
            self.logger.warning(f"Consciousness insight generation failed: {e}")
        
        return insights
    
    # Helper methods for strategy execution
    
    def _configure_rule_engine_for_strategy(self, strategy: LearningStrategy, 
                                          context: LearningContext) -> Dict[str, Any]:
        """Configure rule engine for specific strategy"""
        original_config = {
            'min_confidence_for_rule': self.rule_engine.min_confidence_for_rule,
            'max_hypotheses': self.rule_engine.max_hypotheses,
            'min_examples_for_rule': self.rule_engine.min_examples_for_rule
        }
        
        return original_config
    
    def _restore_rule_engine_config(self, original_config: Dict[str, Any]):
        """Restore original rule engine configuration"""
        self.rule_engine.min_confidence_for_rule = original_config['min_confidence_for_rule']
        self.rule_engine.max_hypotheses = original_config['max_hypotheses']
        self.rule_engine.min_examples_for_rule = original_config['min_examples_for_rule']
    
    def _retrieve_relevant_meta_knowledge(self, context: LearningContext) -> List[str]:
        """Retrieve relevant meta-knowledge for context"""
        relevant_knowledge = []
        
        # Strategy preferences
        if context in self.meta_knowledge.strategy_preferences:
            preferred = self.meta_knowledge.strategy_preferences[context]
            relevant_knowledge.append(f"Preferred strategy for {context.value}: {preferred.value}")
        
        # Success indicators
        context_indicators = [
            indicator for indicator in self.meta_knowledge.success_indicators
            if context.value in indicator
        ]
        relevant_knowledge.extend(context_indicators[:2])
        
        return relevant_knowledge
    
    def _find_analogous_contexts(self, context: LearningContext) -> List[LearningContext]:
        """Find contexts analogous to the given context"""
        # Simple analogy based on context similarity
        context_similarities = {
            LearningContext.GEOMETRIC_TRANSFORMS: [LearningContext.PATTERN_COMPLETION],
            LearningContext.PATTERN_COMPLETION: [LearningContext.GEOMETRIC_TRANSFORMS, LearningContext.COLOR_TRANSFORMATIONS],
            LearningContext.COLOR_TRANSFORMATIONS: [LearningContext.PATTERN_COMPLETION],
            LearningContext.COMPLEX_REASONING: [LearningContext.NOVEL_DOMAIN],
        }
        
        return context_similarities.get(context, [])
    
    def _retrieve_rules_for_context(self, context: LearningContext) -> List[VisualRule]:
        """Retrieve learned rules for a specific context"""
        # This would retrieve rules learned in similar contexts
        # Simplified implementation
        return []
    
    def _adapt_rules_to_examples(self, rules: List[VisualRule], 
                               examples: List[Tuple[np.ndarray, np.ndarray]], 
                               context: LearningContext) -> List[VisualRule]:
        """Adapt existing rules to new examples"""
        # Simplified rule adaptation
        adapted_rules = []
        
        for rule in rules[:3]:  # Limit to top 3 rules
            # Create adapted copy
            adapted_rule = copy.deepcopy(rule)
            adapted_rule.rule_id = f"adapted_{adapted_rule.rule_id}"
            adapted_rule.confidence *= 0.8  # Reduce confidence for adapted rule
            adapted_rules.append(adapted_rule)
        
        return adapted_rules
    
    def _compose_rules(self, basic_rules: List[VisualRule], 
                      examples: List[Tuple[np.ndarray, np.ndarray]]) -> List[VisualRule]:
        """Compose basic rules into more complex ones"""
        # Simplified rule composition
        composite_rules = []
        
        # Try composing pairs of rules
        for i, rule1 in enumerate(basic_rules):
            for rule2 in basic_rules[i+1:]:
                if self._rules_can_compose(rule1, rule2):
                    composite = self._create_composite_rule(rule1, rule2)
                    if composite:
                        composite_rules.append(composite)
        
        return composite_rules[:5]  # Limit number of compositions
    
    def _rules_can_compose(self, rule1: VisualRule, rule2: VisualRule) -> bool:
        """Check if two rules can be composed"""
        # Simple composition check
        return (rule1.rule_type != rule2.rule_type and 
                rule1.confidence > 0.6 and rule2.confidence > 0.6)
    
    def _create_composite_rule(self, rule1: VisualRule, rule2: VisualRule) -> Optional[VisualRule]:
        """Create a composite rule from two rules"""
        try:
            composite = VisualRule(
                rule_id=f"composite_{rule1.rule_id}_{rule2.rule_id}",
                rule_type=RuleType.COMPOSITION,
                conditions=rule1.conditions + rule2.conditions,
                actions=rule1.actions + rule2.actions,
                confidence=min(rule1.confidence, rule2.confidence) * 0.9,
                generality=(rule1.generality + rule2.generality) / 2,
                specificity=(rule1.specificity + rule2.specificity) / 2,
                complexity=rule1.complexity + rule2.complexity
            )
            
            return composite
            
        except Exception as e:
            self.logger.error(f"Composite rule creation failed: {e}")
            return None
    
    def _identify_transfer_opportunities(self, context: LearningContext) -> List[str]:
        """Identify opportunities for transfer learning"""
        opportunities = []
        
        # Find contexts with good performance
        for other_context, difficulty in self.meta_knowledge.context_difficulty.items():
            if other_context != context and difficulty > 0.7:
                opportunities.append(f"Transfer from {other_context.value} (performance: {difficulty:.2f})")
        
        return opportunities[:3]
    
    def _recommend_next_strategy(self, context: LearningContext) -> str:
        """Recommend strategy for next learning session"""
        current_performance = self.strategy_trackers[self.current_strategy].get_context_performance(context)
        
        if current_performance > 0.8:
            return f"Continue with {self.current_strategy.value} (working well)"
        elif current_performance < 0.4:
            # Try different strategy
            alternatives = [s for s in LearningStrategy if s != self.current_strategy]
            best_alternative = max(alternatives, key=lambda s: self.strategy_trackers[s].get_context_performance(context))
            return f"Switch to {best_alternative.value} (current strategy underperforming)"
        else:
            return f"Monitor {self.current_strategy.value} performance"
    
    def get_meta_learning_status(self) -> Dict[str, Any]:
        """Get comprehensive meta-learning status"""
        return {
            'total_episodes': len(self.learning_episodes),
            'current_strategy': self.current_strategy.value,
            'strategy_performances': {
                strategy.value: {
                    'avg_performance': tracker.avg_performance,
                    'total_uses': tracker.total_uses
                }
                for strategy, tracker in self.strategy_trackers.items()
            },
            'context_difficulties': {
                context.value: difficulty
                for context, difficulty in self.meta_knowledge.context_difficulty.items()
            },
            'preferred_strategies': {
                context.value: strategy.value
                for context, strategy in self.meta_knowledge.strategy_preferences.items()
            },
            'recent_performance': np.mean([ep.strategy_effectiveness for ep in self.recent_episodes]) if self.recent_episodes else 0.0,
            'learning_insights': len(self.meta_knowledge.success_indicators)
        }