"""
AGI-LLM Language Module

Revolutionary language model that integrates:
- Neural transformer architecture with consciousness
- Real-time learning from conversations
- Causal and logical reasoning in text generation
- Meta-cognitive awareness during dialogue
- Dynamic knowledge integration
"""

from .agi_transformer import (
    AGITransformerArchitecture,
    ConsciousAttentionLayer,
    ReasoningIntegrationLayer,
    CognitiveEmbedding
)

from .conscious_generation import (
    ConsciousLanguageGenerator,
    MetaCognitiveNarrator,
    ReasoningExplainer,
    GoalDirectedDialogue
)

from .knowledge_integration import (
    DynamicKnowledgeGraph,
    ConversationalLearning,
    KnowledgeConflictResolver,
    CausalKnowledgeUpdater
)

from .language_reasoning import (
    LanguageReasoningEngine,
    SemanticCausalReasoner,
    PragmaticInferenceEngine,
    DiscourseAnalyzer
)

from .agi_llm import (
    AGILLM,
    AGILLMConfig,
    ConversationManager,
    LearningTracker
)

__all__ = [
    # Core architecture
    'AGITransformerArchitecture',
    'ConsciousAttentionLayer', 
    'ReasoningIntegrationLayer',
    'CognitiveEmbedding',
    
    # Conscious generation
    'ConsciousLanguageGenerator',
    'MetaCognitiveNarrator',
    'ReasoningExplainer', 
    'GoalDirectedDialogue',
    
    # Knowledge integration
    'DynamicKnowledgeGraph',
    'ConversationalLearning',
    'KnowledgeConflictResolver',
    'CausalKnowledgeUpdater',
    
    # Language reasoning
    'LanguageReasoningEngine',
    'SemanticCausalReasoner',
    'PragmaticInferenceEngine',
    'DiscourseAnalyzer',
    
    # Main AGI-LLM
    'AGILLM',
    'AGILLMConfig',
    'ConversationManager',
    'LearningTracker'
]