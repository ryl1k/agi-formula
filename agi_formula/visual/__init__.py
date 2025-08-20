"""
Visual Processing Module for AGI-Formula

Specialized visual processing capabilities for ARC-AGI tasks:
- Grid-based visual pattern recognition
- Spatial transformation analysis
- Visual rule induction
- Object tracking and identification
"""

from .grid_processor import (
    VisualGridProcessor,
    GridRepresentation,
    GridPattern,
    GridObject,
    SpatialRelation
)

from .pattern_detector import (
    PatternDetector,
    PatternType,
    VisualPattern,
    PatternMatch
)

from .spatial_transformer import (
    SpatialTransformationEngine,
    TransformationType,
    Transformation,
    TransformationRule,
    TransformationParameters
)

from .feature_extractor import (
    VisualFeatureExtractor,
    FeatureVector,
    ShapeFeatures,
    ColorFeatures,
    SpatialFeatures
)

__all__ = [
    'VisualGridProcessor',
    'GridRepresentation', 
    'GridPattern',
    'GridObject',
    'SpatialRelation',
    'PatternDetector',
    'PatternType',
    'VisualPattern',
    'PatternMatch',
    'SpatialTransformationEngine',
    'TransformationType',
    'Transformation',
    'TransformationRule',
    'TransformationParameters',
    'VisualFeatureExtractor',
    'FeatureVector',
    'ShapeFeatures',
    'ColorFeatures',
    'SpatialFeatures'
]