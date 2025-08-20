"""
AGI-Formula Multi-Modal Processing Module

Advanced multi-modal data processing and fusion for AGI systems:
- Vision processing and computer vision capabilities
- Natural language processing and understanding
- Audio processing and speech recognition
- Sensor fusion and cross-modal attention
- Multi-modal learning and reasoning
"""

from .data_pipeline import MultiModalPipeline, ModalityType
from .vision_processor import VisionProcessor, VisualFeatureExtractor
from .language_processor import LanguageProcessor, TextEncoder
from .audio_processor import AudioProcessor, AudioFeatureExtractor
from .sensor_fusion import SensorFusion, FusionStrategy
from .cross_modal_attention import CrossModalAttention, AttentionMechanism

__all__ = [
    'MultiModalPipeline',
    'ModalityType',
    'VisionProcessor',
    'VisualFeatureExtractor',
    'LanguageProcessor', 
    'TextEncoder',
    'AudioProcessor',
    'AudioFeatureExtractor',
    'SensorFusion',
    'FusionStrategy',
    'CrossModalAttention',
    'AttentionMechanism'
]