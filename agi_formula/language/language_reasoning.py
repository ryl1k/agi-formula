"""
Language-Specific Reasoning Engine

Specialized reasoning for natural language understanding:
- Semantic causal reasoning
- Pragmatic inference
- Discourse analysis
- Language-grounded logical reasoning
"""

import re
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

from ..reasoning import (
    LogicalReasoner,
    CausalReasoner, 
    TemporalReasoner,
    AbstractReasoner
)


class SemanticRelationType(Enum):
    """Types of semantic relations"""
    SYNONYMY = "synonymy"
    ANTONYMY = "antonymy"
    HYPONYMY = "hyponymy"
    MERONYMY = "meronymy"
    CAUSATION = "causation"
    TEMPORAL = "temporal"
    SIMILARITY = "similarity"


class PragmaticIntent(Enum):
    """Types of pragmatic intentions"""
    REQUEST = "request"
    COMMAND = "command"
    QUESTION = "question"
    ASSERTION = "assertion"
    PROMISE = "promise"
    THREAT = "threat"
    SUGGESTION = "suggestion"
    COMPLIMENT = "compliment"
    CRITICISM = "criticism"


@dataclass
class SemanticFrame:
    """Semantic frame for understanding"""
    frame_type: str
    roles: Dict[str, str]
    confidence: float = 0.8
    evidence: List[str] = field(default_factory=list)


@dataclass
class PragmaticContext:
    """Pragmatic context for interpretation"""
    speaker_intent: Optional[PragmaticIntent] = None
    implicatures: List[str] = field(default_factory=list)
    presuppositions: List[str] = field(default_factory=list)
    social_context: Dict[str, Any] = field(default_factory=dict)


class SemanticCausalReasoner:
    """Semantic-aware causal reasoning"""
    
    def __init__(self, causal_reasoner: CausalReasoner):
        self.causal_reasoner = causal_reasoner
        self.causal_indicators = {
            'explicit': [
                'because', 'since', 'due to', 'caused by', 'results in',
                'leads to', 'brings about', 'produces', 'creates',
                'triggers', 'induces', 'generates', 'provokes'
            ],
            'implicit': [
                'therefore', 'thus', 'hence', 'consequently', 'as a result',
                'so', 'accordingly', 'for this reason'
            ],
            'negative': [
                'prevents', 'stops', 'blocks', 'inhibits', 'reduces',
                'decreases', 'eliminates', 'avoids'
            ]
        }
        
        self.causal_patterns = [
            # Pattern: X causes Y
            r"(.+?)\s+(?:causes?|leads? to|results? in|brings? about|produces?|creates?|triggers?|induces?|generates?|provokes?)\s+(.+)",
            # Pattern: Y is caused by X  
            r"(.+?)\s+(?:is|are|was|were)\s+(?:caused by|due to|because of|the result of)\s+(.+)",
            # Pattern: Because X, Y
            r"because\s+(.+?),\s*(.+)",
            # Pattern: X, therefore Y
            r"(.+?),\s*(?:therefore|thus|hence|consequently|as a result|so)\s+(.+)"
        ]
    
    def extract_causal_relations(self, text: str) -> List[Dict[str, Any]]:
        """Extract causal relations from text"""
        relations = []
        text = text.lower().strip()
        
        for pattern in self.causal_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            
            for match in matches:
                if len(match) == 2:
                    cause, effect = match[0].strip(), match[1].strip()
                    
                    # Determine causal strength based on language
                    strength = self._assess_causal_strength(text, cause, effect)
                    
                    relations.append({
                        'cause': cause,
                        'effect': effect,
                        'strength': strength,
                        'type': 'explicit' if any(ind in text for ind in self.causal_indicators['explicit']) else 'implicit',
                        'evidence': text,
                        'confidence': 0.8 if 'explicit' else 0.6
                    })
        
        return relations
    
    def _assess_causal_strength(self, text: str, cause: str, effect: str) -> float:
        """Assess strength of causal relation based on language"""
        strength_indicators = {
            'strong': ['always', 'inevitably', 'necessarily', 'certainly', 'definitely'],
            'moderate': ['usually', 'typically', 'generally', 'often', 'frequently'],
            'weak': ['sometimes', 'occasionally', 'might', 'could', 'may', 'possibly']
        }
        
        text_lower = text.lower()
        
        for strength, indicators in strength_indicators.items():
            if any(indicator in text_lower for indicator in indicators):
                if strength == 'strong':
                    return 0.9
                elif strength == 'moderate':
                    return 0.7
                else:  # weak
                    return 0.4
        
        return 0.6  # Default moderate strength
    
    def reason_about_causation(self, query: str, context: List[Dict[str, str]]) -> Dict[str, Any]:
        """Reason about causal relationships in query"""
        # Extract causal relations from query
        query_relations = self.extract_causal_relations(query)
        
        # Extract causal relations from context
        context_relations = []
        for turn in context:
            if turn.get('content'):
                relations = self.extract_causal_relations(turn['content'])
                context_relations.extend(relations)
        
        # Build causal model
        causal_evidence = query_relations + context_relations
        
        if not causal_evidence:
            return {
                'causal_relations': [],
                'causal_strength': 0.0,
                'confidence': 0.2,
                'reasoning': "No clear causal relations identified in text"
            }
        
        # Analyze causal consistency
        consistency_score = self._analyze_causal_consistency(causal_evidence)
        
        # Generate causal explanation
        explanation = self._generate_causal_explanation(causal_evidence)
        
        return {
            'causal_relations': causal_evidence,
            'causal_strength': np.mean([r['strength'] for r in causal_evidence]),
            'consistency_score': consistency_score,
            'confidence': min(0.9, np.mean([r['confidence'] for r in causal_evidence]) + 0.1),
            'reasoning': explanation,
            'intervention_suggestions': self._suggest_interventions(causal_evidence)
        }
    
    def _analyze_causal_consistency(self, relations: List[Dict[str, Any]]) -> float:
        """Analyze consistency of causal relations"""
        if len(relations) < 2:
            return 1.0
        
        # Check for contradictory causal claims
        contradictions = 0
        total_pairs = 0
        
        for i, rel1 in enumerate(relations):
            for rel2 in relations[i+1:]:
                total_pairs += 1
                
                # Check if same cause-effect pair with different strengths
                if (rel1['cause'] == rel2['cause'] and rel1['effect'] == rel2['effect']):
                    strength_diff = abs(rel1['strength'] - rel2['strength'])
                    if strength_diff > 0.4:  # Significant difference
                        contradictions += 1
                
                # Check for contradictory relations (cause prevents vs causes same effect)
                if (rel1['cause'] == rel2['cause'] and rel1['effect'] == rel2['effect']):
                    if ('prevent' in rel1['evidence'] and 'cause' in rel2['evidence']) or \
                       ('cause' in rel1['evidence'] and 'prevent' in rel2['evidence']):
                        contradictions += 1
        
        consistency = 1.0 - (contradictions / max(1, total_pairs))
        return consistency
    
    def _generate_causal_explanation(self, relations: List[Dict[str, Any]]) -> str:
        """Generate natural language explanation of causal relations"""
        if not relations:
            return "No causal relationships identified."
        
        explanations = []
        for rel in relations:
            strength_desc = self._strength_to_description(rel['strength'])
            explanations.append(f"{rel['cause']} {strength_desc} causes {rel['effect']}")
        
        if len(explanations) == 1:
            return f"The causal relationship is: {explanations[0]}."
        else:
            return f"Multiple causal relationships identified: {'; '.join(explanations)}."
    
    def _strength_to_description(self, strength: float) -> str:
        """Convert strength score to description"""
        if strength > 0.8:
            return "strongly"
        elif strength > 0.6:
            return "moderately"
        elif strength > 0.4:
            return "weakly"
        else:
            return "possibly"
    
    def _suggest_interventions(self, relations: List[Dict[str, Any]]) -> List[str]:
        """Suggest potential interventions based on causal relations"""
        suggestions = []
        
        for rel in relations:
            if rel['strength'] > 0.6:  # Only for strong relations
                suggestions.append(f"To increase {rel['effect']}, consider enhancing {rel['cause']}")
                suggestions.append(f"To decrease {rel['effect']}, consider reducing {rel['cause']}")
        
        return suggestions[:3]  # Limit to top 3 suggestions


class PragmaticInferenceEngine:
    """Engine for pragmatic inference and implicature detection"""
    
    def __init__(self):
        self.intent_patterns = {
            PragmaticIntent.REQUEST: [
                r"could you (?:please )?(.+)",
                r"would you (?:mind )?(.+)",
                r"can you (.+)",
                r"please (.+)"
            ],
            PragmaticIntent.COMMAND: [
                r"(.+)!$",
                r"you (?:must|should|need to) (.+)",
                r"(?:go|do|make|get) (.+)"
            ],
            PragmaticIntent.QUESTION: [
                r"(?:what|when|where|who|why|how|which) (.+)\?",
                r"is (.+)\?",
                r"are (.+)\?",
                r"do (?:you|they|we) (.+)\?"
            ],
            PragmaticIntent.ASSERTION: [
                r"(.+) is (.+)",
                r"(.+) are (.+)",
                r"i (?:think|believe|know) (?:that )?(.+)"
            ]
        }
        
        self.implicature_indicators = [
            'actually', 'in fact', 'really', 'quite', 'rather',
            'sort of', 'kind of', 'somewhat', 'pretty much'
        ]
    
    def analyze_pragmatic_intent(self, text: str, context: List[Dict[str, str]]) -> PragmaticContext:
        """Analyze pragmatic intent and context"""
        text_lower = text.lower().strip()
        
        # Detect intent
        intent = self._detect_intent(text_lower)
        
        # Detect implicatures
        implicatures = self._detect_implicatures(text_lower, context)
        
        # Detect presuppositions
        presuppositions = self._detect_presuppositions(text_lower)
        
        # Analyze social context
        social_context = self._analyze_social_context(text_lower, context)
        
        return PragmaticContext(
            speaker_intent=intent,
            implicatures=implicatures,
            presuppositions=presuppositions,
            social_context=social_context
        )
    
    def _detect_intent(self, text: str) -> Optional[PragmaticIntent]:
        """Detect speaker intent from text"""
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    return intent
        
        # Default based on punctuation
        if text.endswith('?'):
            return PragmaticIntent.QUESTION
        elif text.endswith('!'):
            return PragmaticIntent.COMMAND
        else:
            return PragmaticIntent.ASSERTION
    
    def _detect_implicatures(self, text: str, context: List[Dict[str, str]]) -> List[str]:
        """Detect conversational implicatures"""
        implicatures = []
        
        # Quantity implicatures (saying less than you know)
        if any(indicator in text for indicator in ['some', 'a few', 'certain']):
            implicatures.append("Speaker may know more than they're saying")
        
        # Quality implicatures (hedging, uncertainty)
        if any(indicator in text for indicator in self.implicature_indicators):
            implicatures.append("Speaker is hedging or expressing uncertainty")
        
        # Manner implicatures (unusual phrasing)
        if any(phrase in text for phrase in ['if you know what i mean', 'so to speak', 'as it were']):
            implicatures.append("Speaker is using non-literal language")
        
        # Relation implicatures (topic changes)
        if context and len(context) > 1:
            last_content = context[-1].get('content', '').lower()
            if self._topic_changed(last_content, text):
                implicatures.append("Speaker may be avoiding the topic")
        
        return implicatures
    
    def _detect_presuppositions(self, text: str) -> List[str]:
        """Detect presuppositions in text"""
        presuppositions = []
        
        # Definite descriptions presuppose existence
        definite_pattern = r"the (.+?) (?:is|was|are|were|has|have)"
        matches = re.findall(definite_pattern, text)
        for match in matches:
            presuppositions.append(f"Existence of {match}")
        
        # Possessive constructions
        possessive_pattern = r"(\w+)'s (.+)"
        matches = re.findall(possessive_pattern, text)
        for owner, possessed in matches:
            presuppositions.append(f"{owner} has {possessed}")
        
        # Temporal presuppositions
        if 'still' in text:
            presuppositions.append("Previous state continues")
        if 'again' in text:
            presuppositions.append("Previous occurrence of the event")
        
        return presuppositions
    
    def _analyze_social_context(self, text: str, context: List[Dict[str, str]]) -> Dict[str, Any]:
        """Analyze social context"""
        social_context = {
            'politeness_level': self._assess_politeness(text),
            'formality_level': self._assess_formality(text),
            'emotional_tone': self._assess_emotional_tone(text)
        }
        
        return social_context
    
    def _assess_politeness(self, text: str) -> str:
        """Assess politeness level"""
        polite_markers = ['please', 'thank you', 'excuse me', 'sorry', 'would you mind']
        if any(marker in text for marker in polite_markers):
            return 'high'
        elif any(word in text for word in ['can you', 'could you']):
            return 'medium'
        else:
            return 'low'
    
    def _assess_formality(self, text: str) -> str:
        """Assess formality level"""
        formal_markers = ['furthermore', 'moreover', 'nevertheless', 'consequently']
        informal_markers = ['yeah', 'ok', 'kinda', 'gonna', 'wanna']
        
        if any(marker in text for marker in formal_markers):
            return 'high'
        elif any(marker in text for marker in informal_markers):
            return 'low'
        else:
            return 'medium'
    
    def _assess_emotional_tone(self, text: str) -> str:
        """Assess emotional tone"""
        positive_words = ['good', 'great', 'excellent', 'wonderful', 'amazing']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disappointing']
        
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'
    
    def _topic_changed(self, previous_text: str, current_text: str) -> bool:
        """Check if topic changed between texts"""
        # Simple topic change detection based on word overlap
        prev_words = set(previous_text.split())
        curr_words = set(current_text.split())
        
        # Remove stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        prev_words -= stop_words
        curr_words -= stop_words
        
        if not prev_words or not curr_words:
            return False
        
        overlap = len(prev_words & curr_words) / len(prev_words | curr_words)
        return overlap < 0.3  # Low overlap suggests topic change


class DiscourseAnalyzer:
    """Analyzes discourse structure and coherence"""
    
    def __init__(self):
        self.discourse_markers = {
            'addition': ['also', 'furthermore', 'moreover', 'in addition', 'besides'],
            'contrast': ['but', 'however', 'nevertheless', 'on the other hand', 'although'],
            'cause_effect': ['because', 'since', 'therefore', 'consequently', 'as a result'],
            'temporal': ['then', 'next', 'after', 'before', 'meanwhile', 'subsequently'],
            'example': ['for example', 'for instance', 'such as', 'like', 'including']
        }
    
    def analyze_discourse_structure(self, conversation: List[Dict[str, str]]) -> Dict[str, Any]:
        """Analyze overall discourse structure"""
        if len(conversation) < 2:
            return {'coherence_score': 1.0, 'structure_type': 'minimal'}
        
        # Analyze coherence between turns
        coherence_scores = []
        relations = []
        
        for i in range(len(conversation) - 1):
            current_turn = conversation[i].get('content', '')
            next_turn = conversation[i + 1].get('content', '')
            
            coherence = self._calculate_turn_coherence(current_turn, next_turn)
            coherence_scores.append(coherence)
            
            relation = self._identify_discourse_relation(current_turn, next_turn)
            relations.append(relation)
        
        overall_coherence = np.mean(coherence_scores) if coherence_scores else 1.0
        
        return {
            'coherence_score': overall_coherence,
            'turn_coherences': coherence_scores,
            'discourse_relations': relations,
            'structure_type': self._classify_discourse_structure(relations),
            'discourse_markers_used': self._count_discourse_markers(conversation)
        }
    
    def _calculate_turn_coherence(self, turn1: str, turn2: str) -> float:
        """Calculate coherence between two turns"""
        # Lexical cohesion (word overlap)
        words1 = set(turn1.lower().split())
        words2 = set(turn2.lower().split())
        
        # Remove stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'i', 'you', 'he', 'she', 'it', 'we', 'they'}
        words1 -= stop_words
        words2 -= stop_words
        
        if not words1 or not words2:
            return 0.5  # Neutral coherence for empty content
        
        # Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        lexical_coherence = intersection / union if union > 0 else 0
        
        # Semantic coherence (simplified)
        semantic_coherence = self._assess_semantic_coherence(turn1, turn2)
        
        # Combined coherence
        return (lexical_coherence + semantic_coherence) / 2
    
    def _assess_semantic_coherence(self, turn1: str, turn2: str) -> float:
        """Assess semantic coherence between turns"""
        # Simple semantic coherence based on question-answer patterns
        turn1_lower = turn1.lower()
        turn2_lower = turn2.lower()
        
        # Question-answer coherence
        if turn1_lower.endswith('?'):
            # Turn1 is question, check if turn2 is relevant answer
            question_words = ['what', 'when', 'where', 'who', 'why', 'how']
            for qword in question_words:
                if qword in turn1_lower:
                    # Check if answer contains relevant content
                    if qword == 'what' and any(word in turn2_lower for word in ['is', 'are', 'means']):
                        return 0.8
                    elif qword == 'when' and any(word in turn2_lower for word in ['time', 'date', 'year', 'day']):
                        return 0.8
                    elif qword == 'where' and any(word in turn2_lower for word in ['place', 'location', 'here', 'there']):
                        return 0.8
                    elif qword == 'why' and any(word in turn2_lower for word in ['because', 'since', 'due to']):
                        return 0.8
                    elif qword == 'how' and any(word in turn2_lower for word in ['by', 'through', 'method', 'way']):
                        return 0.8
        
        # Statement continuation
        if not turn1_lower.endswith('?') and not turn2_lower.endswith('?'):
            # Both statements - check for logical continuation
            if any(marker in turn2_lower for marker in ['also', 'furthermore', 'moreover']):
                return 0.7
            elif any(marker in turn2_lower for marker in ['but', 'however', 'although']):
                return 0.6
        
        return 0.5  # Default moderate coherence
    
    def _identify_discourse_relation(self, turn1: str, turn2: str) -> str:
        """Identify discourse relation between turns"""
        turn2_lower = turn2.lower()
        
        for relation_type, markers in self.discourse_markers.items():
            if any(marker in turn2_lower for marker in markers):
                return relation_type
        
        # Default relations based on content
        if turn1.endswith('?') and not turn2.endswith('?'):
            return 'question_answer'
        elif 'yes' in turn2_lower or 'no' in turn2_lower:
            return 'confirmation'
        else:
            return 'continuation'
    
    def _classify_discourse_structure(self, relations: List[str]) -> str:
        """Classify overall discourse structure"""
        if not relations:
            return 'minimal'
        
        relation_counts = {}
        for relation in relations:
            relation_counts[relation] = relation_counts.get(relation, 0) + 1
        
        dominant_relation = max(relation_counts, key=relation_counts.get)
        
        if dominant_relation == 'question_answer':
            return 'interview'
        elif dominant_relation in ['cause_effect', 'temporal']:
            return 'narrative'
        elif dominant_relation in ['addition', 'example']:
            return 'explanatory'
        elif dominant_relation in ['contrast', 'confirmation']:
            return 'argumentative'
        else:
            return 'conversational'
    
    def _count_discourse_markers(self, conversation: List[Dict[str, str]]) -> Dict[str, int]:
        """Count discourse markers used in conversation"""
        marker_counts = {relation_type: 0 for relation_type in self.discourse_markers}
        
        for turn in conversation:
            content = turn.get('content', '').lower()
            for relation_type, markers in self.discourse_markers.items():
                for marker in markers:
                    if marker in content:
                        marker_counts[relation_type] += 1
        
        return marker_counts


class LanguageReasoningEngine:
    """Main language reasoning engine"""
    
    def __init__(self, reasoning_engines: Dict[str, Any]):
        self.logical_reasoner = reasoning_engines.get('logical')
        self.causal_reasoner = reasoning_engines.get('causal')
        self.temporal_reasoner = reasoning_engines.get('temporal')
        self.abstract_reasoner = reasoning_engines.get('abstract')
        
        # Language-specific reasoners
        self.semantic_causal_reasoner = SemanticCausalReasoner(self.causal_reasoner)
        self.pragmatic_engine = PragmaticInferenceEngine()
        self.discourse_analyzer = DiscourseAnalyzer()
    
    def apply_logical_reasoning(self, text: str) -> Dict[str, Any]:
        """Apply logical reasoning to text"""
        try:
            # Extract logical statements
            statements = self._extract_logical_statements(text)
            
            if not statements:
                return {'confidence': 0.2, 'reasoning': 'No clear logical statements found'}
            
            # Simple logical analysis
            logical_structure = self._analyze_logical_structure(statements)
            
            return {
                'confidence': 0.8,
                'logical_statements': statements,
                'logical_structure': logical_structure,
                'reasoning': f"Identified {len(statements)} logical statements with {logical_structure['type']} structure"
            }
        except Exception as e:
            logging.warning(f"Logical reasoning error: {e}")
            return {'confidence': 0.1, 'reasoning': f'Error in logical analysis: {str(e)}'}
    
    def apply_causal_reasoning(self, text: str, context: List[Dict[str, str]]) -> Dict[str, Any]:
        """Apply causal reasoning to text with context"""
        try:
            return self.semantic_causal_reasoner.reason_about_causation(text, context)
        except Exception as e:
            logging.warning(f"Causal reasoning error: {e}")
            return {'confidence': 0.1, 'reasoning': f'Error in causal analysis: {str(e)}'}
    
    def apply_temporal_reasoning(self, text: str) -> Dict[str, Any]:
        """Apply temporal reasoning to text"""
        try:
            # Extract temporal expressions
            temporal_expressions = self._extract_temporal_expressions(text)
            
            if not temporal_expressions:
                return {'confidence': 0.3, 'reasoning': 'No clear temporal expressions found'}
            
            # Analyze temporal structure
            temporal_structure = self._analyze_temporal_structure(temporal_expressions)
            
            return {
                'confidence': 0.7,
                'temporal_expressions': temporal_expressions,
                'temporal_structure': temporal_structure,
                'reasoning': f"Identified {len(temporal_expressions)} temporal expressions"
            }
        except Exception as e:
            logging.warning(f"Temporal reasoning error: {e}")
            return {'confidence': 0.1, 'reasoning': f'Error in temporal analysis: {str(e)}'}
    
    def apply_abstract_reasoning(self, text: str) -> Dict[str, Any]:
        """Apply abstract reasoning to text"""
        try:
            # Look for abstract patterns
            patterns = self._identify_abstract_patterns(text)
            
            if not patterns:
                return {'confidence': 0.3, 'reasoning': 'No clear abstract patterns found'}
            
            return {
                'confidence': 0.6,
                'abstract_patterns': patterns,
                'reasoning': f"Identified {len(patterns)} abstract patterns or analogies"
            }
        except Exception as e:
            logging.warning(f"Abstract reasoning error: {e}")
            return {'confidence': 0.1, 'reasoning': f'Error in abstract analysis: {str(e)}'}
    
    def apply_pragmatic_reasoning(self, text: str, context: List[Dict[str, str]]) -> Dict[str, Any]:
        """Apply pragmatic reasoning to text"""
        try:
            pragmatic_context = self.pragmatic_engine.analyze_pragmatic_intent(text, context)
            
            return {
                'confidence': 0.8,
                'speaker_intent': pragmatic_context.speaker_intent.value if pragmatic_context.speaker_intent else None,
                'implicatures': pragmatic_context.implicatures,
                'presuppositions': pragmatic_context.presuppositions,
                'social_context': pragmatic_context.social_context,
                'reasoning': f"Identified intent: {pragmatic_context.speaker_intent.value if pragmatic_context.speaker_intent else 'unclear'}"
            }
        except Exception as e:
            logging.warning(f"Pragmatic reasoning error: {e}")
            return {'confidence': 0.1, 'reasoning': f'Error in pragmatic analysis: {str(e)}'}
    
    def analyze_discourse(self, conversation: List[Dict[str, str]]) -> Dict[str, Any]:
        """Analyze discourse structure of conversation"""
        try:
            return self.discourse_analyzer.analyze_discourse_structure(conversation)
        except Exception as e:
            logging.warning(f"Discourse analysis error: {e}")
            return {'coherence_score': 0.5, 'reasoning': f'Error in discourse analysis: {str(e)}'}
    
    def _extract_logical_statements(self, text: str) -> List[str]:
        """Extract logical statements from text"""
        # Simple logical statement patterns
        logical_patterns = [
            r"if (.+) then (.+)",
            r"(.+) implies (.+)",
            r"all (.+) are (.+)",
            r"some (.+) are (.+)",
            r"no (.+) are (.+)",
            r"(.+) is (?:true|false|valid|invalid)"
        ]
        
        statements = []
        for pattern in logical_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    statements.append(" -> ".join(match))
                else:
                    statements.append(match)
        
        return statements
    
    def _analyze_logical_structure(self, statements: List[str]) -> Dict[str, Any]:
        """Analyze logical structure of statements"""
        if not statements:
            return {'type': 'none'}
        
        # Check for conditional statements
        if any('->' in stmt or 'if' in stmt.lower() for stmt in statements):
            return {'type': 'conditional', 'complexity': 'moderate'}
        
        # Check for quantified statements
        if any(word in ' '.join(statements).lower() for word in ['all', 'some', 'no', 'every']):
            return {'type': 'quantified', 'complexity': 'high'}
        
        return {'type': 'simple', 'complexity': 'low'}
    
    def _extract_temporal_expressions(self, text: str) -> List[Dict[str, str]]:
        """Extract temporal expressions from text"""
        temporal_patterns = [
            (r"before (.+)", "before"),
            (r"after (.+)", "after"), 
            (r"during (.+)", "during"),
            (r"when (.+)", "when"),
            (r"(\d+) (?:years?|months?|days?|hours?|minutes?) (?:ago|later)", "duration"),
            (r"(?:yesterday|today|tomorrow)", "relative_time"),
            (r"(?:first|then|next|finally)", "sequence")
        ]
        
        expressions = []
        for pattern, temp_type in temporal_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                expressions.append({
                    'expression': match if isinstance(match, str) else match[0],
                    'type': temp_type
                })
        
        return expressions
    
    def _analyze_temporal_structure(self, expressions: List[Dict[str, str]]) -> Dict[str, Any]:
        """Analyze temporal structure"""
        if not expressions:
            return {'type': 'atemporal'}
        
        types = [expr['type'] for expr in expressions]
        
        if 'sequence' in types:
            return {'type': 'sequential', 'complexity': 'moderate'}
        elif any(t in types for t in ['before', 'after', 'during']):
            return {'type': 'relational', 'complexity': 'high'}
        elif 'duration' in types:
            return {'type': 'durational', 'complexity': 'low'}
        else:
            return {'type': 'simple_temporal', 'complexity': 'low'}
    
    def _identify_abstract_patterns(self, text: str) -> List[Dict[str, str]]:
        """Identify abstract patterns in text"""
        patterns = []
        
        # Analogy patterns
        analogy_patterns = [
            r"(.+) is like (.+)",
            r"(.+) is similar to (.+)",
            r"(.+) resembles (.+)",
            r"think of (.+) as (.+)"
        ]
        
        for pattern in analogy_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                patterns.append({
                    'type': 'analogy',
                    'source': match[0].strip(),
                    'target': match[1].strip()
                })
        
        # Pattern/generalization indicators
        pattern_indicators = ['pattern', 'trend', 'generally', 'typically', 'usually', 'in general']
        if any(indicator in text.lower() for indicator in pattern_indicators):
            patterns.append({
                'type': 'generalization',
                'content': text[:100]  # First 100 chars
            })
        
        return patterns