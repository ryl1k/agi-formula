"""
Language Processing Module for AGI-Formula

Advanced natural language processing capabilities for multi-modal AGI:
- Text preprocessing and tokenization
- Semantic analysis and understanding
- Language generation and synthesis
- Multi-lingual support
- Contextual embeddings and representations
- Syntactic and semantic parsing
"""

import numpy as np
import time
import re
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging


class LanguageTask(Enum):
    """Types of language processing tasks"""
    TOKENIZATION = "tokenization"
    EMBEDDING = "embedding"
    SENTIMENT_ANALYSIS = "sentiment"
    NAMED_ENTITY_RECOGNITION = "ner"
    PART_OF_SPEECH = "pos"
    DEPENDENCY_PARSING = "dependency"
    SEMANTIC_PARSING = "semantic"
    TEXT_GENERATION = "generation"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"


@dataclass
class TextToken:
    """Individual text token with metadata"""
    text: str
    position: int
    token_id: int
    pos_tag: Optional[str] = None
    ner_tag: Optional[str] = None
    embedding: Optional[np.ndarray] = None
    confidence: float = 1.0


@dataclass
class LanguageFeatures:
    """Container for extracted language features"""
    text: str
    tokens: List[TextToken]
    embeddings: np.ndarray
    semantic_features: np.ndarray
    syntactic_features: np.ndarray
    metadata: Dict[str, Any]
    confidence: float


@dataclass
class LanguageAnalysis:
    """Comprehensive language analysis results"""
    original_text: str
    features: LanguageFeatures
    sentiment: Dict[str, float]
    entities: List[Dict[str, Any]]
    syntax_tree: Optional[Dict[str, Any]]
    semantic_roles: List[Dict[str, Any]]
    complexity_metrics: Dict[str, float]


class TextEncoder:
    """
    Advanced text encoding and embedding system
    
    Features:
    - Multiple encoding strategies (word-level, character-level, subword)
    - Contextual embeddings with attention
    - Multi-lingual support
    - Adaptive vocabulary management
    - Efficient batch processing
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        
        # Vocabulary and encoding
        self.vocabulary = {}
        self.reverse_vocabulary = {}
        self.vocab_size = 0
        self.embedding_matrix = None
        
        # Special tokens
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1, 
            '<START>': 2,
            '<END>': 3,
            '<MASK>': 4
        }
        
        # Initialize encoder
        self._initialize_encoder()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for text encoder"""
        return {
            'max_vocab_size': 50000,
            'embedding_dim': 256,
            'max_sequence_length': 512,
            'min_token_frequency': 2,
            'encoding_strategy': 'word',  # word, char, subword
            'case_sensitive': False,
            'remove_punctuation': False,
            'language': 'en',
            'pretrained_embeddings': None
        }
    
    def _initialize_encoder(self):
        """Initialize the text encoder"""
        # Initialize vocabulary with special tokens
        self.vocabulary = self.special_tokens.copy()
        self.reverse_vocabulary = {v: k for k, v in self.special_tokens.items()}
        self.vocab_size = len(self.special_tokens)
        
        # Initialize embedding matrix
        self._initialize_embeddings()
        
        print(f"Text encoder initialized with {self.vocab_size} tokens")
    
    def _initialize_embeddings(self):
        """Initialize embedding matrix"""
        embedding_dim = self.config['embedding_dim']
        
        # Initialize with random embeddings
        self.embedding_matrix = np.random.normal(
            0, 0.1, (self.config['max_vocab_size'], embedding_dim)
        ).astype(np.float32)
        
        # Special token embeddings
        self.embedding_matrix[self.special_tokens['<PAD>']] = np.zeros(embedding_dim)
    
    def build_vocabulary(self, texts: List[str]):
        """Build vocabulary from a corpus of texts"""
        token_counts = {}
        
        for text in texts:
            tokens = self._tokenize_text(text)
            for token in tokens:
                token_counts[token] = token_counts.get(token, 0) + 1
        
        # Add tokens that meet frequency threshold
        for token, count in token_counts.items():
            if (count >= self.config['min_token_frequency'] and 
                token not in self.vocabulary and
                self.vocab_size < self.config['max_vocab_size']):
                
                self.vocabulary[token] = self.vocab_size
                self.reverse_vocabulary[self.vocab_size] = token
                self.vocab_size += 1
        
        # Update embedding matrix size
        if self.vocab_size > self.embedding_matrix.shape[0]:
            old_embeddings = self.embedding_matrix
            self.embedding_matrix = np.random.normal(
                0, 0.1, (self.vocab_size, self.config['embedding_dim'])
            ).astype(np.float32)
            self.embedding_matrix[:old_embeddings.shape[0]] = old_embeddings
        
        print(f"Vocabulary built with {self.vocab_size} tokens")
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text based on encoding strategy"""
        if not self.config['case_sensitive']:
            text = text.lower()
        
        if self.config['encoding_strategy'] == 'word':
            return self._word_tokenize(text)
        elif self.config['encoding_strategy'] == 'char':
            return self._char_tokenize(text)
        elif self.config['encoding_strategy'] == 'subword':
            return self._subword_tokenize(text)
        else:
            return self._word_tokenize(text)
    
    def _word_tokenize(self, text: str) -> List[str]:
        """Word-level tokenization"""
        # Simple word tokenization
        if self.config['remove_punctuation']:
            text = re.sub(r'[^\w\s]', '', text)
        
        tokens = text.split()
        
        # Further split on punctuation if not removed
        if not self.config['remove_punctuation']:
            refined_tokens = []
            for token in tokens:
                # Split on punctuation
                parts = re.findall(r'\w+|[^\w\s]', token)
                refined_tokens.extend(parts)
            tokens = refined_tokens
        
        return tokens
    
    def _char_tokenize(self, text: str) -> List[str]:
        """Character-level tokenization"""
        return list(text)
    
    def _subword_tokenize(self, text: str) -> List[str]:
        """Subword tokenization (simplified BPE-like)"""
        # Simplified subword tokenization
        words = self._word_tokenize(text)
        subwords = []
        
        for word in words:
            if len(word) <= 3:
                subwords.append(word)
            else:
                # Split into subwords
                for i in range(0, len(word), 3):
                    subword = word[i:i+3]
                    if i + 3 < len(word):
                        subword += '##'  # Continuation marker
                    subwords.append(subword)
        
        return subwords
    
    def encode_text(self, text: str) -> Tuple[List[int], List[TextToken]]:
        """Encode text to token IDs and create token objects"""
        tokens = self._tokenize_text(text)
        token_ids = []
        text_tokens = []
        
        for i, token in enumerate(tokens):
            token_id = self.vocabulary.get(token, self.special_tokens['<UNK>'])
            token_ids.append(token_id)
            
            # Get embedding for token
            embedding = self.embedding_matrix[token_id] if token_id < len(self.embedding_matrix) else None
            
            text_token = TextToken(
                text=token,
                position=i,
                token_id=token_id,
                embedding=embedding,
                confidence=1.0 if token in self.vocabulary else 0.5
            )
            text_tokens.append(text_token)
        
        return token_ids, text_tokens
    
    def decode_tokens(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text"""
        tokens = []
        for token_id in token_ids:
            if token_id in self.reverse_vocabulary:
                token = self.reverse_vocabulary[token_id]
                if token not in ['<PAD>', '<START>', '<END>', '<MASK>']:
                    tokens.append(token)
        
        # Join tokens back to text
        if self.config['encoding_strategy'] == 'char':
            return ''.join(tokens)
        elif self.config['encoding_strategy'] == 'subword':
            # Handle subword joining
            text = ''
            for token in tokens:
                if token.endswith('##'):
                    text += token[:-2]
                else:
                    text += token + ' '
            return text.strip()
        else:
            return ' '.join(tokens)
    
    def get_embeddings(self, text: str) -> Tuple[np.ndarray, List[TextToken]]:
        """Get embeddings for text"""
        token_ids, text_tokens = self.encode_text(text)
        
        # Get embeddings
        embeddings = []
        for token_id in token_ids:
            if token_id < len(self.embedding_matrix):
                embeddings.append(self.embedding_matrix[token_id])
            else:
                embeddings.append(np.zeros(self.config['embedding_dim']))
        
        return np.array(embeddings), text_tokens
    
    def update_embeddings(self, token_id: int, embedding: np.ndarray):
        """Update embedding for a specific token"""
        if 0 <= token_id < len(self.embedding_matrix):
            self.embedding_matrix[token_id] = embedding


class LanguageProcessor:
    """
    Comprehensive language processing system for AGI
    
    Features:
    - Multi-task language understanding
    - Semantic and syntactic analysis
    - Contextual processing with attention
    - Language generation capabilities
    - Integration with multi-modal pipeline
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        
        # Initialize components
        self.text_encoder = TextEncoder(self.config.get('encoding', {}))
        
        # Analysis components
        self.sentiment_analyzer = SentimentAnalyzer()
        self.ner_processor = NamedEntityRecognizer()
        self.pos_tagger = POSTagger()
        self.dependency_parser = DependencyParser()
        
        # Performance monitoring
        self.stats = {
            'texts_processed': 0,
            'processing_time_ms': [],
            'encoding_time_ms': [],
            'analysis_accuracy': {}
        }
        
        print("Language processor initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for language processor"""
        return {
            'max_text_length': 1000,
            'enable_sentiment_analysis': True,
            'enable_ner': True,
            'enable_pos_tagging': True,
            'enable_dependency_parsing': True,
            'enable_semantic_analysis': True,
            'language_model_size': 'base',  # tiny, base, large
            'batch_size': 32
        }
    
    def process_text(self, text: str, 
                    tasks: Optional[List[LanguageTask]] = None) -> LanguageAnalysis:
        """Process text with comprehensive language analysis"""
        start_time = time.time()
        
        if tasks is None:
            tasks = [
                LanguageTask.TOKENIZATION,
                LanguageTask.EMBEDDING,
                LanguageTask.SENTIMENT_ANALYSIS,
                LanguageTask.NAMED_ENTITY_RECOGNITION,
                LanguageTask.PART_OF_SPEECH
            ]
        
        # Preprocess text
        processed_text = self._preprocess_text(text)
        
        # Extract language features
        features = self._extract_language_features(processed_text, tasks)
        
        # Perform analysis tasks
        analysis_results = {}
        
        if LanguageTask.SENTIMENT_ANALYSIS in tasks and self.config['enable_sentiment_analysis']:
            analysis_results['sentiment'] = self.sentiment_analyzer.analyze(processed_text)
        
        if LanguageTask.NAMED_ENTITY_RECOGNITION in tasks and self.config['enable_ner']:
            analysis_results['entities'] = self.ner_processor.extract_entities(features.tokens)
        
        if LanguageTask.PART_OF_SPEECH in tasks and self.config['enable_pos_tagging']:
            features.tokens = self.pos_tagger.tag_tokens(features.tokens)
        
        if LanguageTask.DEPENDENCY_PARSING in tasks and self.config['enable_dependency_parsing']:
            analysis_results['syntax_tree'] = self.dependency_parser.parse(features.tokens)
        
        # Calculate complexity metrics
        complexity_metrics = self._calculate_complexity_metrics(processed_text, features)
        
        # Create comprehensive analysis
        analysis = LanguageAnalysis(
            original_text=text,
            features=features,
            sentiment=analysis_results.get('sentiment', {}),
            entities=analysis_results.get('entities', []),
            syntax_tree=analysis_results.get('syntax_tree'),
            semantic_roles=[],  # Placeholder for semantic role labeling
            complexity_metrics=complexity_metrics
        )
        
        processing_time = (time.time() - start_time) * 1000
        self.stats['texts_processed'] += 1
        self.stats['processing_time_ms'].append(processing_time)
        
        return analysis
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for analysis"""
        # Truncate if too long
        if len(text) > self.config['max_text_length']:
            text = text[:self.config['max_text_length']]
        
        # Basic cleaning
        text = text.strip()
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def _extract_language_features(self, text: str, tasks: List[LanguageTask]) -> LanguageFeatures:
        """Extract comprehensive language features"""
        start_time = time.time()
        
        # Tokenization and encoding
        embeddings, tokens = self.text_encoder.get_embeddings(text)
        
        # Extract semantic features
        semantic_features = self._extract_semantic_features(embeddings, tokens)
        
        # Extract syntactic features
        syntactic_features = self._extract_syntactic_features(tokens)
        
        encoding_time = (time.time() - start_time) * 1000
        self.stats['encoding_time_ms'].append(encoding_time)
        
        features = LanguageFeatures(
            text=text,
            tokens=tokens,
            embeddings=embeddings,
            semantic_features=semantic_features,
            syntactic_features=syntactic_features,
            metadata={
                'num_tokens': len(tokens),
                'avg_token_length': np.mean([len(t.text) for t in tokens]),
                'encoding_time_ms': encoding_time
            },
            confidence=np.mean([t.confidence for t in tokens]) if tokens else 0.0
        )
        
        return features
    
    def _extract_semantic_features(self, embeddings: np.ndarray, tokens: List[TextToken]) -> np.ndarray:
        """Extract semantic features from embeddings"""
        if len(embeddings) == 0:
            return np.zeros(self.text_encoder.config['embedding_dim'])
        
        # Global semantic features
        mean_embedding = np.mean(embeddings, axis=0)
        max_embedding = np.max(embeddings, axis=0)
        min_embedding = np.min(embeddings, axis=0)
        
        # Attention-weighted features (simplified)
        attention_weights = self._compute_attention_weights(embeddings)
        weighted_embedding = np.sum(embeddings * attention_weights[:, np.newaxis], axis=0)
        
        # Combine features
        semantic_features = np.concatenate([
            mean_embedding,
            max_embedding,
            min_embedding,
            weighted_embedding
        ])
        
        return semantic_features
    
    def _compute_attention_weights(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute attention weights for embeddings"""
        if len(embeddings) == 0:
            return np.array([])
        
        # Simple attention mechanism
        # Compute similarity to mean embedding
        mean_emb = np.mean(embeddings, axis=0)
        similarities = np.dot(embeddings, mean_emb) / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(mean_emb) + 1e-8
        )
        
        # Softmax
        exp_similarities = np.exp(similarities - np.max(similarities))
        attention_weights = exp_similarities / np.sum(exp_similarities)
        
        return attention_weights
    
    def _extract_syntactic_features(self, tokens: List[TextToken]) -> np.ndarray:
        """Extract syntactic features from tokens"""
        if not tokens:
            return np.zeros(64)  # Default syntactic feature size
        
        features = []
        
        # Token-level features
        features.append(len(tokens))  # Sequence length
        features.append(np.mean([len(t.text) for t in tokens]))  # Average token length
        features.append(len(set(t.text.lower() for t in tokens)))  # Unique tokens
        
        # Character-level features
        text = ' '.join(t.text for t in tokens)
        features.append(len(text))  # Total characters
        features.append(text.count(' '))  # Number of spaces
        features.append(sum(c.isupper() for c in text))  # Uppercase letters
        features.append(sum(c.islower() for c in text))  # Lowercase letters
        features.append(sum(c.isdigit() for c in text))  # Digits
        features.append(sum(not c.isalnum() and c != ' ' for c in text))  # Punctuation
        
        # Linguistic patterns
        features.append(sum(1 for t in tokens if t.text.lower() in ['the', 'a', 'an']))  # Articles
        features.append(sum(1 for t in tokens if t.text.lower() in ['and', 'or', 'but']))  # Conjunctions
        features.append(sum(1 for t in tokens if t.text.lower() in ['i', 'you', 'he', 'she', 'it', 'we', 'they']))  # Pronouns
        
        # Pad to fixed size
        while len(features) < 64:
            features.append(0.0)
        
        return np.array(features[:64], dtype=np.float32)
    
    def _calculate_complexity_metrics(self, text: str, features: LanguageFeatures) -> Dict[str, float]:
        """Calculate text complexity metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['char_count'] = len(text)
        metrics['word_count'] = len(features.tokens)
        metrics['sentence_count'] = text.count('.') + text.count('!') + text.count('?')
        
        # Lexical diversity
        if features.tokens:
            unique_tokens = len(set(t.text.lower() for t in features.tokens))
            metrics['lexical_diversity'] = unique_tokens / len(features.tokens)
            metrics['avg_word_length'] = np.mean([len(t.text) for t in features.tokens])
        else:
            metrics['lexical_diversity'] = 0.0
            metrics['avg_word_length'] = 0.0
        
        # Readability approximation (simplified Flesch score)
        if metrics['sentence_count'] > 0 and metrics['word_count'] > 0:
            avg_sentence_length = metrics['word_count'] / metrics['sentence_count']
            avg_syllables = metrics['avg_word_length'] * 0.5  # Rough approximation
            
            flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables)
            metrics['readability_score'] = max(0, min(100, flesch_score))
        else:
            metrics['readability_score'] = 0.0
        
        # Semantic complexity (based on embedding variance)
        if len(features.embeddings) > 0:
            embedding_variance = np.mean(np.var(features.embeddings, axis=0))
            metrics['semantic_complexity'] = float(embedding_variance)
        else:
            metrics['semantic_complexity'] = 0.0
        
        return metrics
    
    def generate_text(self, prompt: str, max_length: int = 100) -> str:
        """Generate text based on a prompt"""
        # Simplified text generation
        # In practice, this would use a language model
        
        # Encode prompt
        token_ids, tokens = self.text_encoder.encode_text(prompt)
        
        # Generate tokens (simplified random generation)
        generated_ids = token_ids.copy()
        
        for _ in range(min(max_length, 50)):  # Limit generation
            # Simple next token prediction (random for demo)
            next_token_id = np.random.choice(
                list(range(5, min(self.text_encoder.vocab_size, 1000)))
            )
            generated_ids.append(next_token_id)
            
            # Stop at end token
            if next_token_id == self.text_encoder.special_tokens['<END>']:
                break
        
        # Decode generated text
        generated_text = self.text_encoder.decode_tokens(generated_ids)
        
        return generated_text
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get language processing statistics"""
        stats = self.stats.copy()
        
        if stats['processing_time_ms']:
            stats['avg_processing_time_ms'] = np.mean(stats['processing_time_ms'][-100:])
            stats['max_processing_time_ms'] = np.max(stats['processing_time_ms'][-100:])
        
        if stats['encoding_time_ms']:
            stats['avg_encoding_time_ms'] = np.mean(stats['encoding_time_ms'][-100:])
        
        stats['vocabulary_size'] = self.text_encoder.vocab_size
        
        return stats


class SentimentAnalyzer:
    """Simple sentiment analysis component"""
    
    def __init__(self):
        # Simple lexicon-based sentiment analysis
        self.positive_words = set([
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'love', 'like', 'enjoy', 'happy', 'pleased', 'satisfied'
        ])
        
        self.negative_words = set([
            'bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike',
            'sad', 'angry', 'disappointed', 'frustrated', 'annoyed'
        ])
    
    def analyze(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text"""
        words = text.lower().split()
        
        positive_score = sum(1 for word in words if word in self.positive_words)
        negative_score = sum(1 for word in words if word in self.negative_words)
        
        total_words = len(words)
        
        if total_words == 0:
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0, 'compound': 0.0}
        
        positive = positive_score / total_words
        negative = negative_score / total_words
        neutral = 1.0 - positive - negative
        compound = positive - negative
        
        return {
            'positive': positive,
            'negative': negative,
            'neutral': neutral,
            'compound': compound
        }


class NamedEntityRecognizer:
    """Simple named entity recognition component"""
    
    def __init__(self):
        # Simple pattern-based NER
        self.entity_patterns = {
            'PERSON': [r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'],
            'ORGANIZATION': [r'\b[A-Z][A-Za-z]* (Inc|Corp|LLC|Ltd)\b'],
            'LOCATION': [r'\b[A-Z][a-z]+ (City|State|Country)\b'],
            'DATE': [r'\b\d{1,2}/\d{1,2}/\d{4}\b', r'\b\d{4}-\d{2}-\d{2}\b'],
            'NUMBER': [r'\b\d+\b']
        }
    
    def extract_entities(self, tokens: List[TextToken]) -> List[Dict[str, Any]]:
        """Extract named entities from tokens"""
        text = ' '.join(t.text for t in tokens)
        entities = []
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    entities.append({
                        'text': match.group(),
                        'label': entity_type,
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': 0.8  # Static confidence for demo
                    })
        
        return entities


class POSTagger:
    """Simple part-of-speech tagger"""
    
    def __init__(self):
        # Simple rule-based POS tagging
        self.pos_rules = {
            'DT': ['the', 'a', 'an', 'this', 'that', 'these', 'those'],
            'NN': ['cat', 'dog', 'house', 'car', 'book', 'person'],
            'VB': ['run', 'walk', 'jump', 'eat', 'sleep', 'work'],
            'JJ': ['good', 'bad', 'big', 'small', 'red', 'blue'],
            'PRP': ['i', 'you', 'he', 'she', 'it', 'we', 'they'],
            'IN': ['in', 'on', 'at', 'by', 'for', 'with', 'to']
        }
    
    def tag_tokens(self, tokens: List[TextToken]) -> List[TextToken]:
        """Add POS tags to tokens"""
        for token in tokens:
            token.pos_tag = self._get_pos_tag(token.text.lower())
        
        return tokens
    
    def _get_pos_tag(self, word: str) -> str:
        """Get POS tag for a word"""
        for pos_tag, words in self.pos_rules.items():
            if word in words:
                return pos_tag
        
        # Simple heuristics
        if word.endswith('ing'):
            return 'VBG'
        elif word.endswith('ed'):
            return 'VBD'
        elif word.endswith('ly'):
            return 'RB'
        elif word.endswith('s') and len(word) > 2:
            return 'NNS'
        else:
            return 'NN'  # Default to noun


class DependencyParser:
    """Simple dependency parser"""
    
    def __init__(self):
        pass
    
    def parse(self, tokens: List[TextToken]) -> Dict[str, Any]:
        """Parse dependency relationships between tokens"""
        # Simplified dependency parsing
        # In practice, this would use a trained dependency parser
        
        dependencies = []
        
        for i, token in enumerate(tokens):
            if i == 0:
                # Root token
                dependencies.append({
                    'id': i,
                    'text': token.text,
                    'head': -1,
                    'relation': 'ROOT'
                })
            else:
                # Simple heuristic: attach to previous token
                dependencies.append({
                    'id': i,
                    'text': token.text,
                    'head': i - 1,
                    'relation': 'dep'
                })
        
        return {
            'tokens': [t.text for t in tokens],
            'dependencies': dependencies
        }