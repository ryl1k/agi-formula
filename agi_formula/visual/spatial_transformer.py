"""
Spatial Transformation Engine for ARC-AGI Visual Processing

Advanced spatial transformation capabilities for visual grid analysis:
- Rotation, reflection, translation operations
- Transformation detection and classification  
- Transformation rule synthesis
- Geometric invariance analysis
- Spatial reasoning and constraint satisfaction
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict
import math

from .grid_processor import GridObject, BoundingBox, GridRepresentation, ObjectShape


class TransformationType(Enum):
    """Types of spatial transformations"""
    # Basic transformations
    ROTATION_90 = "rotation_90"
    ROTATION_180 = "rotation_180"
    ROTATION_270 = "rotation_270"
    
    # Reflections
    REFLECTION_HORIZONTAL = "reflection_horizontal"
    REFLECTION_VERTICAL = "reflection_vertical"
    REFLECTION_DIAGONAL_MAIN = "reflection_diagonal_main"
    REFLECTION_DIAGONAL_ANTI = "reflection_diagonal_anti"
    
    # Translations
    TRANSLATION = "translation"
    
    # Scaling
    SCALING = "scaling"
    
    # Complex transformations
    SHEAR = "shear"
    AFFINE = "affine"
    
    # Composite transformations
    ROTATION_THEN_TRANSLATION = "rotation_then_translation"
    REFLECTION_THEN_TRANSLATION = "reflection_then_translation"
    SCALING_THEN_TRANSLATION = "scaling_then_translation"
    
    # Identity
    IDENTITY = "identity"


class CoordinateSystem(Enum):
    """Coordinate system types"""
    GRID_BASED = "grid_based"  # Integer grid coordinates
    CONTINUOUS = "continuous"   # Continuous coordinates
    NORMALIZED = "normalized"   # Normalized [0,1] coordinates


@dataclass
class TransformationParameters:
    """Parameters for a specific transformation"""
    transformation_type: TransformationType
    parameters: Dict[str, Any]
    coordinate_system: CoordinateSystem = CoordinateSystem.GRID_BASED
    
    def __post_init__(self):
        # Validate parameters based on transformation type
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate transformation parameters"""
        if self.transformation_type == TransformationType.TRANSLATION:
            if 'dx' not in self.parameters or 'dy' not in self.parameters:
                raise ValueError("Translation requires 'dx' and 'dy' parameters")
        elif self.transformation_type == TransformationType.SCALING:
            if 'scale_x' not in self.parameters or 'scale_y' not in self.parameters:
                raise ValueError("Scaling requires 'scale_x' and 'scale_y' parameters")


@dataclass 
class Transformation:
    """Represents a complete spatial transformation"""
    transformation_id: str
    transformation_type: TransformationType
    parameters: TransformationParameters
    source_region: Optional[BoundingBox] = None
    target_region: Optional[BoundingBox] = None
    confidence: float = 0.8
    
    # Transformation matrix (3x3 homogeneous coordinates)
    matrix: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if self.matrix is None:
            self.matrix = self._compute_transformation_matrix()
    
    def _compute_transformation_matrix(self) -> np.ndarray:
        """Compute 3x3 transformation matrix"""
        if self.transformation_type == TransformationType.IDENTITY:
            return np.eye(3)
        
        elif self.transformation_type == TransformationType.TRANSLATION:
            dx = self.parameters.parameters['dx']
            dy = self.parameters.parameters['dy']
            return np.array([
                [1, 0, dx],
                [0, 1, dy], 
                [0, 0, 1]
            ])
        
        elif self.transformation_type == TransformationType.ROTATION_90:
            return np.array([
                [0, -1, 0],
                [1, 0, 0],
                [0, 0, 1]
            ])
        
        elif self.transformation_type == TransformationType.ROTATION_180:
            return np.array([
                [-1, 0, 0],
                [0, -1, 0],
                [0, 0, 1]
            ])
        
        elif self.transformation_type == TransformationType.ROTATION_270:
            return np.array([
                [0, 1, 0],
                [-1, 0, 0],
                [0, 0, 1]
            ])
        
        elif self.transformation_type == TransformationType.REFLECTION_HORIZONTAL:
            return np.array([
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, 1]
            ])
        
        elif self.transformation_type == TransformationType.REFLECTION_VERTICAL:
            return np.array([
                [-1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ])
        
        elif self.transformation_type == TransformationType.REFLECTION_DIAGONAL_MAIN:
            return np.array([
                [0, 1, 0],
                [1, 0, 0],
                [0, 0, 1]
            ])
        
        elif self.transformation_type == TransformationType.REFLECTION_DIAGONAL_ANTI:
            return np.array([
                [0, -1, 0],
                [-1, 0, 0],
                [0, 0, 1]
            ])
        
        elif self.transformation_type == TransformationType.SCALING:
            sx = self.parameters.parameters['scale_x']
            sy = self.parameters.parameters['scale_y']
            return np.array([
                [sx, 0, 0],
                [0, sy, 0],
                [0, 0, 1]
            ])
        
        else:
            return np.eye(3)  # Default to identity


@dataclass
class TransformationRule:
    """Rule for applying transformations"""
    rule_id: str
    condition: str  # Description of when this rule applies
    transformation: Transformation
    confidence: float
    application_context: Dict[str, Any]
    examples: List[Tuple[Any, Any]] = field(default_factory=list)  # (input, output) pairs


class SpatialTransformationEngine:
    """Advanced spatial transformation engine"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Transformation detection settings
        self.similarity_threshold = 0.8
        self.transformation_confidence_threshold = 0.6
        
        # Pre-defined transformation templates
        self._init_transformation_templates()
        
        # Transformation rule database
        self.learned_rules: List[TransformationRule] = []
        
    def _init_transformation_templates(self):
        """Initialize common transformation templates"""
        self.transformation_templates = {
            TransformationType.ROTATION_90: self._create_rotation_template(90),
            TransformationType.ROTATION_180: self._create_rotation_template(180),
            TransformationType.ROTATION_270: self._create_rotation_template(270),
            TransformationType.REFLECTION_HORIZONTAL: self._create_reflection_template('horizontal'),
            TransformationType.REFLECTION_VERTICAL: self._create_reflection_template('vertical'),
            TransformationType.TRANSLATION: self._create_translation_template()
        }
    
    def apply_transformation(self, grid: np.ndarray, transformation: Transformation) -> np.ndarray:
        """Apply transformation to a grid"""
        try:
            if transformation.transformation_type == TransformationType.IDENTITY:
                return grid.copy()
            
            elif transformation.transformation_type == TransformationType.ROTATION_90:
                return np.rot90(grid, 1)
            
            elif transformation.transformation_type == TransformationType.ROTATION_180:
                return np.rot90(grid, 2)
            
            elif transformation.transformation_type == TransformationType.ROTATION_270:
                return np.rot90(grid, 3)
            
            elif transformation.transformation_type == TransformationType.REFLECTION_HORIZONTAL:
                return np.flipud(grid)
            
            elif transformation.transformation_type == TransformationType.REFLECTION_VERTICAL:
                return np.fliplr(grid)
            
            elif transformation.transformation_type == TransformationType.REFLECTION_DIAGONAL_MAIN:
                return grid.T
            
            elif transformation.transformation_type == TransformationType.REFLECTION_DIAGONAL_ANTI:
                return np.flipud(np.fliplr(grid.T))
            
            elif transformation.transformation_type == TransformationType.TRANSLATION:
                return self._apply_translation(grid, transformation.parameters.parameters)
            
            elif transformation.transformation_type == TransformationType.SCALING:
                return self._apply_scaling(grid, transformation.parameters.parameters)
            
            else:
                self.logger.warning(f"Unsupported transformation type: {transformation.transformation_type}")
                return grid.copy()
                
        except Exception as e:
            self.logger.error(f"Error applying transformation: {e}")
            return grid.copy()
    
    def detect_transformation(self, source_grid: np.ndarray, target_grid: np.ndarray) -> Optional[Transformation]:
        """Detect transformation that converts source to target grid"""
        try:
            # Try each transformation type
            transformations_to_test = [
                TransformationType.IDENTITY,
                TransformationType.ROTATION_90,
                TransformationType.ROTATION_180,
                TransformationType.ROTATION_270,
                TransformationType.REFLECTION_HORIZONTAL,
                TransformationType.REFLECTION_VERTICAL,
                TransformationType.REFLECTION_DIAGONAL_MAIN,
                TransformationType.REFLECTION_DIAGONAL_ANTI
            ]
            
            for trans_type in transformations_to_test:
                transformation = self._create_basic_transformation(trans_type)
                transformed = self.apply_transformation(source_grid, transformation)
                
                similarity = self._calculate_grid_similarity(transformed, target_grid)
                
                if similarity >= self.similarity_threshold:
                    transformation.confidence = similarity
                    return transformation
            
            # Try translation transformations
            translation_transform = self._detect_translation(source_grid, target_grid)
            if translation_transform:
                return translation_transform
            
            # Try scaling transformations
            scaling_transform = self._detect_scaling(source_grid, target_grid)
            if scaling_transform:
                return scaling_transform
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error detecting transformation: {e}")
            return None
    
    def detect_object_transformation(self, source_obj: GridObject, target_obj: GridObject) -> Optional[Transformation]:
        """Detect transformation between two objects"""
        try:
            # Check if objects have same shape and color
            if source_obj.shape != target_obj.shape or source_obj.color != target_obj.color:
                return None
            
            # Calculate center displacement
            source_center = source_obj.bounding_box.center
            target_center = target_obj.bounding_box.center
            
            dx = target_center[1] - source_center[1]  # Column difference
            dy = target_center[0] - source_center[0]  # Row difference
            
            # Check for pure translation
            if abs(dx) > 0.1 or abs(dy) > 0.1:
                params = TransformationParameters(
                    TransformationType.TRANSLATION,
                    {'dx': dx, 'dy': dy}
                )
                
                transformation = Transformation(
                    transformation_id=f"translate_{source_obj.object_id}_to_{target_obj.object_id}",
                    transformation_type=TransformationType.TRANSLATION,
                    parameters=params,
                    source_region=source_obj.bounding_box,
                    target_region=target_obj.bounding_box,
                    confidence=0.9
                )
                
                return transformation
            
            # Check for scaling
            size_ratio = target_obj.size / source_obj.size if source_obj.size > 0 else 1.0
            if abs(size_ratio - 1.0) > 0.1:
                # Estimate scaling factors
                width_ratio = target_obj.bounding_box.width / source_obj.bounding_box.width
                height_ratio = target_obj.bounding_box.height / source_obj.bounding_box.height
                
                params = TransformationParameters(
                    TransformationType.SCALING,
                    {'scale_x': width_ratio, 'scale_y': height_ratio}
                )
                
                transformation = Transformation(
                    transformation_id=f"scale_{source_obj.object_id}_to_{target_obj.object_id}",
                    transformation_type=TransformationType.SCALING,
                    parameters=params,
                    source_region=source_obj.bounding_box,
                    target_region=target_obj.bounding_box,
                    confidence=0.8
                )
                
                return transformation
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error detecting object transformation: {e}")
            return None
    
    def analyze_transformation_sequence(self, grids: List[np.ndarray]) -> List[Transformation]:
        """Analyze sequence of grids to detect transformation pattern"""
        transformations = []
        
        if len(grids) < 2:
            return transformations
        
        for i in range(len(grids) - 1):
            transformation = self.detect_transformation(grids[i], grids[i + 1])
            if transformation:
                transformations.append(transformation)
            else:
                # If no transformation detected, add identity
                identity_transform = self._create_basic_transformation(TransformationType.IDENTITY)
                transformations.append(identity_transform)
        
        return transformations
    
    def synthesize_transformation_rule(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[TransformationRule]:
        """Synthesize transformation rule from examples"""
        try:
            if not examples:
                return None
            
            # Detect transformation for each example
            detected_transformations = []
            for source, target in examples:
                transformation = self.detect_transformation(source, target)
                if transformation:
                    detected_transformations.append(transformation)
            
            if not detected_transformations:
                return None
            
            # Find most common transformation type
            transformation_types = [t.transformation_type for t in detected_transformations]
            most_common_type = max(set(transformation_types), key=transformation_types.count)
            
            # Filter to most common type
            common_transformations = [t for t in detected_transformations if t.transformation_type == most_common_type]
            
            # Check consistency
            consistency_ratio = len(common_transformations) / len(examples)
            
            if consistency_ratio >= 0.7:  # At least 70% consistency
                # Create rule
                avg_confidence = np.mean([t.confidence for t in common_transformations])
                
                rule = TransformationRule(
                    rule_id=f"rule_{most_common_type.value}_{len(self.learned_rules)}",
                    condition=f"Apply {most_common_type.value} transformation",
                    transformation=common_transformations[0],  # Use first as template
                    confidence=avg_confidence * consistency_ratio,
                    application_context={
                        'transformation_type': most_common_type,
                        'consistency_ratio': consistency_ratio,
                        'examples_count': len(examples)
                    },
                    examples=examples
                )
                
                self.learned_rules.append(rule)
                return rule
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error synthesizing transformation rule: {e}")
            return None
    
    def predict_transformation(self, source_grid: np.ndarray, context: Dict[str, Any] = None) -> List[Tuple[Transformation, float]]:
        """Predict likely transformations for a source grid"""
        predictions = []
        
        # Use learned rules to make predictions
        for rule in self.learned_rules:
            # Simple context matching (can be enhanced)
            context_match_score = self._evaluate_context_match(rule, context or {})
            
            if context_match_score > 0.5:
                prediction_confidence = rule.confidence * context_match_score
                predictions.append((rule.transformation, prediction_confidence))
        
        # Add default transformations if no learned rules apply
        if not predictions:
            default_transformations = [
                TransformationType.ROTATION_90,
                TransformationType.ROTATION_180,
                TransformationType.REFLECTION_HORIZONTAL,
                TransformationType.REFLECTION_VERTICAL
            ]
            
            for trans_type in default_transformations:
                transformation = self._create_basic_transformation(trans_type)
                predictions.append((transformation, 0.5))  # Default confidence
        
        # Sort by confidence
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions[:5]  # Return top 5 predictions
    
    def apply_transformation_to_objects(self, objects: List[GridObject], transformation: Transformation) -> List[GridObject]:
        """Apply transformation to a list of objects"""
        transformed_objects = []
        
        for obj in objects:
            transformed_obj = self._transform_object(obj, transformation)
            if transformed_obj:
                transformed_objects.append(transformed_obj)
        
        return transformed_objects
    
    def _transform_object(self, obj: GridObject, transformation: Transformation) -> Optional[GridObject]:
        """Transform a single object"""
        try:
            # Transform object cells
            transformed_cells = set()
            
            for row, col in obj.cells:
                # Convert to homogeneous coordinates
                point = np.array([col, row, 1])  # Note: x=col, y=row
                
                # Apply transformation matrix
                transformed_point = transformation.matrix @ point
                
                # Convert back to grid coordinates
                new_col = int(round(transformed_point[0]))
                new_row = int(round(transformed_point[1]))
                
                transformed_cells.add((new_row, new_col))
            
            # Create new bounding box
            if transformed_cells:
                rows = [r for r, c in transformed_cells]
                cols = [c for r, c in transformed_cells]
                
                new_bbox = BoundingBox(
                    min_row=min(rows),
                    max_row=max(rows),
                    min_col=min(cols),
                    max_col=max(cols)
                )
                
                # Create transformed object
                transformed_obj = GridObject(
                    object_id=obj.object_id,
                    color=obj.color,
                    cells=transformed_cells,
                    bounding_box=new_bbox,
                    shape=obj.shape,  # Shape might change - could be enhanced
                    size=len(transformed_cells),
                    confidence=obj.confidence * transformation.confidence
                )
                
                return transformed_obj
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error transforming object: {e}")
            return None
    
    # Helper methods
    
    def _create_basic_transformation(self, transformation_type: TransformationType) -> Transformation:
        """Create basic transformation with default parameters"""
        params = TransformationParameters(transformation_type, {})
        
        return Transformation(
            transformation_id=f"basic_{transformation_type.value}",
            transformation_type=transformation_type,
            parameters=params
        )
    
    def _create_rotation_template(self, angle: int) -> Callable:
        """Create rotation transformation template"""
        def template(grid: np.ndarray) -> np.ndarray:
            if angle == 90:
                return np.rot90(grid, 1)
            elif angle == 180:
                return np.rot90(grid, 2)
            elif angle == 270:
                return np.rot90(grid, 3)
            else:
                return grid
        return template
    
    def _create_reflection_template(self, axis: str) -> Callable:
        """Create reflection transformation template"""
        def template(grid: np.ndarray) -> np.ndarray:
            if axis == 'horizontal':
                return np.flipud(grid)
            elif axis == 'vertical':
                return np.fliplr(grid)
            else:
                return grid
        return template
    
    def _create_translation_template(self) -> Callable:
        """Create translation transformation template"""
        def template(grid: np.ndarray, dx: int = 0, dy: int = 0) -> np.ndarray:
            return self._apply_translation(grid, {'dx': dx, 'dy': dy})
        return template
    
    def _apply_translation(self, grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Apply translation transformation"""
        dx = int(params.get('dx', 0))
        dy = int(params.get('dy', 0))
        
        height, width = grid.shape
        result = np.zeros_like(grid)
        
        for row in range(height):
            for col in range(width):
                new_row = row + dy
                new_col = col + dx
                
                if 0 <= new_row < height and 0 <= new_col < width:
                    result[new_row, new_col] = grid[row, col]
        
        return result
    
    def _apply_scaling(self, grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Apply scaling transformation"""
        scale_x = params.get('scale_x', 1.0)
        scale_y = params.get('scale_y', 1.0)
        
        height, width = grid.shape
        new_height = int(height * scale_y)
        new_width = int(width * scale_x)
        
        # Simple nearest neighbor scaling
        result = np.zeros((new_height, new_width), dtype=grid.dtype)
        
        for new_row in range(new_height):
            for new_col in range(new_width):
                orig_row = int(new_row / scale_y)
                orig_col = int(new_col / scale_x)
                
                if 0 <= orig_row < height and 0 <= orig_col < width:
                    result[new_row, new_col] = grid[orig_row, orig_col]
        
        return result
    
    def _calculate_grid_similarity(self, grid1: np.ndarray, grid2: np.ndarray) -> float:
        """Calculate similarity between two grids"""
        if grid1.shape != grid2.shape:
            return 0.0
        
        total_cells = grid1.size
        matching_cells = np.sum(grid1 == grid2)
        
        return matching_cells / total_cells
    
    def _detect_translation(self, source: np.ndarray, target: np.ndarray) -> Optional[Transformation]:
        """Detect translation transformation"""
        if source.shape != target.shape:
            return None
        
        height, width = source.shape
        
        # Try different translation offsets
        for dy in range(-height, height + 1):
            for dx in range(-width, width + 1):
                params = TransformationParameters(
                    TransformationType.TRANSLATION,
                    {'dx': dx, 'dy': dy}
                )
                
                transformation = Transformation(
                    transformation_id=f"translate_{dx}_{dy}",
                    transformation_type=TransformationType.TRANSLATION,
                    parameters=params
                )
                
                transformed = self.apply_transformation(source, transformation)
                similarity = self._calculate_grid_similarity(transformed, target)
                
                if similarity >= self.similarity_threshold:
                    transformation.confidence = similarity
                    return transformation
        
        return None
    
    def _detect_scaling(self, source: np.ndarray, target: np.ndarray) -> Optional[Transformation]:
        """Detect scaling transformation"""
        source_height, source_width = source.shape
        target_height, target_width = target.shape
        
        # Calculate scale factors
        scale_y = target_height / source_height
        scale_x = target_width / source_width
        
        # Only consider reasonable scale factors
        if 0.1 <= scale_x <= 10 and 0.1 <= scale_y <= 10:
            params = TransformationParameters(
                TransformationType.SCALING,
                {'scale_x': scale_x, 'scale_y': scale_y}
            )
            
            transformation = Transformation(
                transformation_id=f"scale_{scale_x:.2f}_{scale_y:.2f}",
                transformation_type=TransformationType.SCALING,
                parameters=params
            )
            
            transformed = self.apply_transformation(source, transformation)
            similarity = self._calculate_grid_similarity(transformed, target)
            
            if similarity >= self.similarity_threshold:
                transformation.confidence = similarity
                return transformation
        
        return None
    
    def _evaluate_context_match(self, rule: TransformationRule, context: Dict[str, Any]) -> float:
        """Evaluate how well a rule matches the given context"""
        # Simple context matching - can be enhanced
        match_score = 0.5  # Base score
        
        # Check transformation type preference
        if 'preferred_transformation' in context:
            if rule.transformation.transformation_type == context['preferred_transformation']:
                match_score += 0.3
        
        # Check grid size compatibility
        if 'grid_size' in context and 'grid_size' in rule.application_context:
            if context['grid_size'] == rule.application_context['grid_size']:
                match_score += 0.2
        
        return min(1.0, match_score)
    
    def get_transformation_statistics(self) -> Dict[str, Any]:
        """Get statistics about learned transformations"""
        if not self.learned_rules:
            return {'total_rules': 0}
        
        # Count rules by transformation type
        type_counts = defaultdict(int)
        for rule in self.learned_rules:
            type_counts[rule.transformation.transformation_type.value] += 1
        
        # Calculate average confidence
        avg_confidence = np.mean([rule.confidence for rule in self.learned_rules])
        
        return {
            'total_rules': len(self.learned_rules),
            'transformation_type_distribution': dict(type_counts),
            'average_confidence': avg_confidence,
            'most_common_transformation': max(type_counts, key=type_counts.get) if type_counts else None
        }