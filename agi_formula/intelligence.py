"""
AGI Intelligence Architecture

Core intelligence system that integrates consciousness, reasoning,
learning, and adaptation for artificial general intelligence.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from .consciousness import ConsciousAgent
from .reasoning import ReasoningEngine, Concept

class Intelligence:
    """Core AGI intelligence system"""
    
    def __init__(self, consciousness_level=0.5, learning_rate=0.01):
        self.conscious_agent = ConsciousAgent(consciousness_level)
        self.reasoning_engine = ReasoningEngine()
        self.learning_rate = learning_rate
        
        # Core intelligence components
        self.knowledge_graph = {}
        self.skill_repertoire = {}
        self.goal_system = GoalSystem()
        self.adaptation_engine = AdaptationEngine()
        
        # Intelligence metrics
        self.intelligence_quotient = 1.0
        self.learning_efficiency = 1.0
        self.problem_solving_ability = 1.0
        self.creativity_index = 0.5
        
    def perceive(self, stimulus):
        """Intelligent perception with consciousness"""
        # Conscious perception
        perceived = self.conscious_agent.perceive(stimulus)
        
        # Extract concepts from perception
        concepts = self._extract_concepts(perceived)
        
        # Update knowledge graph
        for concept in concepts:
            self._integrate_knowledge(concept)
            
        return perceived, concepts
    
    def think(self, problem, context=None):
        """Think about a problem using integrated intelligence"""
        # Conscious reasoning
        processed_input, reasoning_strength = self.conscious_agent.reason(problem)
        
        # Apply reasoning engine
        reasoning_result = self.reasoning_engine.reason(
            problem, context, reasoning_types=['logical', 'causal', 'temporal', 'abstract']
        )
        
        # Generate solution hypotheses
        hypotheses = self._generate_hypotheses(problem, reasoning_result, reasoning_strength)
        
        # Evaluate and select best hypothesis
        best_solution = self._evaluate_hypotheses(hypotheses)
        
        # Learn from thinking process
        self._learn_from_thinking(problem, best_solution, reasoning_result)
        
        return best_solution
    
    def learn(self, experience, feedback=None):
        """Learn from experience with conscious reflection"""
        # Conscious learning
        self.conscious_agent.meta_learn(experience)
        
        # Extract patterns
        patterns = self._extract_patterns(experience)
        
        # Update skills
        new_skills = self._develop_skills(patterns, feedback)
        
        # Adapt behavior
        adaptations = self.adaptation_engine.adapt(experience, feedback)
        
        # Update intelligence metrics
        self._update_intelligence_metrics(experience, feedback, new_skills, adaptations)
        
        return {
            'patterns_learned': patterns,
            'skills_developed': new_skills,
            'adaptations': adaptations,
            'intelligence_growth': self.intelligence_quotient
        }
    
    def create(self, goal, constraints=None):
        """Creative problem solving and generation"""
        # Set creative goal
        self.goal_system.set_goal(goal, 'creative')
        
        # Generate creative ideas
        ideas = self._generate_creative_ideas(goal, constraints)
        
        # Evaluate creativity
        evaluated_ideas = self._evaluate_creativity(ideas)
        
        # Refine best ideas
        refined_solutions = self._refine_solutions(evaluated_ideas[:3])  # Top 3
        
        # Update creativity index
        self.creativity_index = min(1.0, self.creativity_index + 0.01)
        
        return refined_solutions
    
    def adapt(self, environment_changes):
        """Adapt to environmental changes"""
        # Analyze changes
        change_analysis = self._analyze_changes(environment_changes)
        
        # Generate adaptation strategies
        strategies = self.adaptation_engine.generate_strategies(change_analysis)
        
        # Select and implement best strategy
        best_strategy = self._select_adaptation_strategy(strategies)
        
        # Update behavior patterns
        self._update_behavior_patterns(best_strategy)
        
        return best_strategy
    
    def _extract_concepts(self, perceived_data):
        """Extract concepts from perceived data"""
        concepts = []
        
        if hasattr(perceived_data, 'shape') and len(perceived_data.shape) > 0:
            # Analyze data patterns
            mean_val = np.mean(perceived_data)
            std_val = np.std(perceived_data)
            
            # Create concepts based on statistical properties
            if std_val > 0.5:
                concepts.append(Concept('high_variance', {'variance': std_val}))
            elif std_val < 0.1:
                concepts.append(Concept('low_variance', {'variance': std_val}))
                
            if mean_val > 0.5:
                concepts.append(Concept('positive_bias', {'mean': mean_val}))
            elif mean_val < -0.5:
                concepts.append(Concept('negative_bias', {'mean': mean_val}))
                
        return concepts
    
    def _integrate_knowledge(self, concept):
        """Integrate new concept into knowledge graph"""
        concept_name = concept.name
        
        if concept_name not in self.knowledge_graph:
            self.knowledge_graph[concept_name] = {
                'concept': concept,
                'connections': {},
                'activation_history': [],
                'importance': 1.0
            }
        else:
            # Strengthen existing concept
            self.knowledge_graph[concept_name]['importance'] += 0.1
            
        # Create connections to related concepts
        for existing_name, existing_data in self.knowledge_graph.items():
            if existing_name != concept_name:
                similarity = self._calculate_concept_similarity(
                    concept, existing_data['concept']
                )
                if similarity > 0.3:
                    self.knowledge_graph[concept_name]['connections'][existing_name] = similarity
                    self.knowledge_graph[existing_name]['connections'][concept_name] = similarity
    
    def _calculate_concept_similarity(self, concept1, concept2):
        """Calculate similarity between concepts"""
        if not concept1.properties or not concept2.properties:
            return 0.0
            
        common_props = set(concept1.properties.keys()) & set(concept2.properties.keys())
        total_props = set(concept1.properties.keys()) | set(concept2.properties.keys())
        
        if not total_props:
            return 0.0
            
        return len(common_props) / len(total_props)
    
    def _generate_hypotheses(self, problem, reasoning_result, reasoning_strength):
        """Generate solution hypotheses"""
        hypotheses = []
        
        # Hypothesis from logical reasoning
        if 'logical' in reasoning_result['results']:
            for inference in reasoning_result['results']['logical']:
                hypotheses.append({
                    'solution': inference['conclusion'],
                    'confidence': inference['confidence'] * reasoning_strength,
                    'method': 'logical',
                    'evidence': inference
                })
        
        # Hypothesis from causal reasoning
        if 'causal' in reasoning_result['results']:
            for cause in reasoning_result['results']['causal']:
                hypotheses.append({
                    'solution': f"address_cause_{cause['cause']}",
                    'confidence': cause['strength'] * reasoning_strength,
                    'method': 'causal',
                    'evidence': cause
                })
        
        # Creative hypothesis
        creative_solution = self._generate_creative_solution(problem)
        if creative_solution:
            hypotheses.append({
                'solution': creative_solution,
                'confidence': self.creativity_index * reasoning_strength,
                'method': 'creative',
                'evidence': {'creativity_score': self.creativity_index}
            })
            
        return hypotheses
    
    def _generate_creative_solution(self, problem):
        """Generate creative solution to problem"""
        # Combine random concepts from knowledge graph
        concepts = list(self.knowledge_graph.keys())
        if len(concepts) >= 2:
            selected_concepts = np.random.choice(concepts, size=2, replace=False)
            return f"combine_{selected_concepts[0]}_with_{selected_concepts[1]}"
        return None
    
    def _evaluate_hypotheses(self, hypotheses):
        """Evaluate and select best hypothesis"""
        if not hypotheses:
            return None
            
        # Sort by confidence
        sorted_hypotheses = sorted(hypotheses, key=lambda h: h['confidence'], reverse=True)
        
        best_hypothesis = sorted_hypotheses[0]
        
        # Add evaluation metadata
        best_hypothesis['evaluation'] = {
            'total_candidates': len(hypotheses),
            'selection_confidence': best_hypothesis['confidence'],
            'alternative_solutions': len(sorted_hypotheses) - 1
        }
        
        return best_hypothesis
    
    def _learn_from_thinking(self, problem, solution, reasoning_result):
        """Learn from the thinking process"""
        if solution:
            # Record successful reasoning pattern
            pattern = {
                'problem_type': type(problem).__name__,
                'solution_method': solution['method'],
                'success_confidence': solution['confidence'],
                'reasoning_types_used': list(reasoning_result['results'].keys())
            }
            
            # Store in knowledge graph
            pattern_concept = Concept(
                f"reasoning_pattern_{len(self.knowledge_graph)}", 
                properties=pattern
            )
            self._integrate_knowledge(pattern_concept)
    
    def _extract_patterns(self, experience):
        """Extract learning patterns from experience"""
        patterns = []
        
        if hasattr(experience, '__dict__'):
            for attr_name, attr_value in experience.__dict__.items():
                if isinstance(attr_value, (int, float, str)):
                    patterns.append({
                        'type': 'attribute_pattern',
                        'attribute': attr_name,
                        'value': attr_value,
                        'pattern_strength': 1.0
                    })
                    
        return patterns
    
    def _develop_skills(self, patterns, feedback):
        """Develop new skills from patterns"""
        new_skills = []
        
        for pattern in patterns:
            skill_name = f"skill_{pattern['type']}_{len(self.skill_repertoire)}"
            
            skill = {
                'name': skill_name,
                'pattern': pattern,
                'proficiency': 0.1,
                'usage_count': 0,
                'success_rate': 0.5
            }
            
            # Improve skill based on feedback
            if feedback and feedback.get('success', False):
                skill['proficiency'] += 0.1
                skill['success_rate'] += 0.05
                
            self.skill_repertoire[skill_name] = skill
            new_skills.append(skill)
            
        return new_skills
    
    def _update_intelligence_metrics(self, experience, feedback, new_skills, adaptations):
        """Update intelligence metrics based on learning"""
        # Learning efficiency
        if new_skills:
            self.learning_efficiency = min(2.0, self.learning_efficiency + 0.01 * len(new_skills))
            
        # Problem solving ability
        if feedback and feedback.get('success', False):
            self.problem_solving_ability = min(2.0, self.problem_solving_ability + 0.005)
        elif feedback and not feedback.get('success', True):
            self.problem_solving_ability = max(0.1, self.problem_solving_ability - 0.001)
            
        # Overall intelligence quotient
        self.intelligence_quotient = (
            self.learning_efficiency * 0.4 + 
            self.problem_solving_ability * 0.4 + 
            self.creativity_index * 0.2
        )
    
    def _generate_creative_ideas(self, goal, constraints):
        """Generate creative ideas for goal achievement"""
        ideas = []
        
        # Combine existing concepts creatively
        concepts = list(self.knowledge_graph.keys())
        
        for i in range(min(10, len(concepts))):  # Generate up to 10 ideas
            if len(concepts) >= 2:
                selected = np.random.choice(concepts, size=2, replace=False)
                idea = {
                    'concept_combination': selected,
                    'novelty_score': np.random.random(),
                    'feasibility': np.random.random(),
                    'goal_alignment': np.random.random()
                }
                ideas.append(idea)
                
        return ideas
    
    def _evaluate_creativity(self, ideas):
        """Evaluate creativity of generated ideas"""
        for idea in ideas:
            # Creativity score combines novelty, feasibility, and goal alignment
            creativity_score = (
                idea['novelty_score'] * 0.4 +
                idea['feasibility'] * 0.3 +
                idea['goal_alignment'] * 0.3
            )
            idea['creativity_score'] = creativity_score
            
        return sorted(ideas, key=lambda x: x['creativity_score'], reverse=True)
    
    def _refine_solutions(self, top_ideas):
        """Refine top creative solutions"""
        refined = []
        
        for idea in top_ideas:
            refined_idea = idea.copy()
            
            # Refine based on known successful patterns
            for skill_name, skill in self.skill_repertoire.items():
                if skill['success_rate'] > 0.7:  # High success rate
                    refined_idea[f'enhanced_with_{skill_name}'] = True
                    refined_idea['creativity_score'] += 0.1
                    
            refined.append(refined_idea)
            
        return refined
    
    def _analyze_changes(self, environment_changes):
        """Analyze environmental changes"""
        return {
            'change_magnitude': np.random.random(),  # Placeholder
            'adaptation_required': True,
            'affected_skills': list(self.skill_repertoire.keys())[:3]
        }
    
    def _select_adaptation_strategy(self, strategies):
        """Select best adaptation strategy"""
        if strategies:
            return max(strategies, key=lambda s: s.get('effectiveness', 0))
        return None
    
    def _update_behavior_patterns(self, strategy):
        """Update behavior based on adaptation strategy"""
        if strategy:
            # Update learning rate based on strategy
            self.learning_rate *= strategy.get('learning_adjustment', 1.0)
            self.learning_rate = np.clip(self.learning_rate, 0.001, 0.1)

class GoalSystem:
    """Goal setting and achievement system"""
    
    def __init__(self):
        self.goals = {}
        self.goal_hierarchy = {}
        self.achievement_history = []
        
    def set_goal(self, goal_description, goal_type='general', priority=1.0):
        """Set a new goal"""
        goal_id = f"goal_{len(self.goals)}"
        
        goal = {
            'id': goal_id,
            'description': goal_description,
            'type': goal_type,
            'priority': priority,
            'status': 'active',
            'created_at': len(self.achievement_history),
            'sub_goals': []
        }
        
        self.goals[goal_id] = goal
        return goal_id
    
    def achieve_goal(self, goal_id, success_level=1.0):
        """Mark goal as achieved"""
        if goal_id in self.goals:
            self.goals[goal_id]['status'] = 'completed'
            self.goals[goal_id]['success_level'] = success_level
            
            self.achievement_history.append({
                'goal_id': goal_id,
                'success_level': success_level,
                'completed_at': len(self.achievement_history)
            })

class AdaptationEngine:
    """Handles adaptation to changing environments"""
    
    def __init__(self):
        self.adaptation_strategies = {}
        self.environment_model = {}
        self.adaptation_history = []
        
    def adapt(self, experience, feedback):
        """Generate adaptation based on experience"""
        adaptation = {
            'trigger': experience,
            'feedback_quality': feedback.get('quality', 0.5) if feedback else 0.5,
            'adaptation_type': 'learning_rate_adjustment',
            'effectiveness': np.random.random()  # Placeholder
        }
        
        self.adaptation_history.append(adaptation)
        return adaptation
    
    def generate_strategies(self, change_analysis):
        """Generate adaptation strategies"""
        strategies = []
        
        if change_analysis.get('adaptation_required', False):
            strategies.append({
                'name': 'increase_learning_rate',
                'learning_adjustment': 1.1,
                'effectiveness': 0.8
            })
            
            strategies.append({
                'name': 'diversify_exploration',
                'learning_adjustment': 1.05,
                'effectiveness': 0.6
            })
            
        return strategies

__all__ = ['Intelligence', 'GoalSystem', 'AdaptationEngine']