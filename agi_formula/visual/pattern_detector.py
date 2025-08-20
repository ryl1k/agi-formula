"""
Pattern Detection Engine for ARC-AGI Visual Analysis

Advanced pattern detection algorithms for identifying complex visual patterns:
- Geometric pattern recognition
- Sequential pattern analysis
- Symmetry detection
- Repetition and tiling patterns
- Transformation pattern identification
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, Counter
import itertools

from .grid_processor import GridObject, BoundingBox, GridRepresentation, ObjectShape


class PatternType(Enum):
    """Types of visual patterns"""
    # Geometric patterns
    SYMMETRY = "symmetry"
    ROTATION = "rotation"
    REFLECTION = "reflection"
    
    # Sequential patterns
    SEQUENCE = "sequence"
    PROGRESSION = "progression"
    ALTERNATION = "alternation"
    
    # Spatial patterns
    GRID_ARRANGEMENT = "grid_arrangement"
    LINEAR_ARRANGEMENT = "linear_arrangement"
    CIRCULAR_ARRANGEMENT = "circular_arrangement"
    
    # Repetition patterns
    TILING = "tiling"
    FRACTAL = "fractal"
    PERIODIC = "periodic"
    
    # Transformation patterns
    SCALING = "scaling"
    MORPHING = "morphing"
    COLOR_CHANGE = "color_change"
    
    # Complex patterns
    NESTED = "nested"
    COMPOSITE = "composite"
    CONDITIONAL = "conditional"


class PatternComplexity(Enum):
    """Pattern complexity levels"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    HIGHLY_COMPLEX = "highly_complex"


@dataclass
class PatternMatch:
    """Represents a detected pattern match"""
    pattern_type: PatternType
    confidence: float
    elements: List[Any]
    parameters: Dict[str, Any]
    description: str
    complexity: PatternComplexity
    spatial_extent: Optional[BoundingBox] = None
    transformation_rule: Optional[str] = None


@dataclass
class VisualPattern:
    """Complete visual pattern with all matches"""
    pattern_id: str
    pattern_type: PatternType
    matches: List[PatternMatch]
    global_confidence: float
    pattern_strength: float
    description: str
    potential_rules: List[str]


class PatternDetector:
    """Advanced pattern detection engine"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Pattern detection thresholds
        self.min_confidence = 0.5
        self.min_pattern_strength = 0.6
        self.max_patterns_per_type = 10
        
        # Pattern complexity weights
        self.complexity_weights = {
            PatternComplexity.SIMPLE: 1.0,
            PatternComplexity.MODERATE: 0.8,
            PatternComplexity.COMPLEX: 0.6,
            PatternComplexity.HIGHLY_COMPLEX: 0.4
        }
    
    def detect_all_patterns(self, grid_repr: GridRepresentation) -> List[VisualPattern]:
        """Detect all patterns in a grid representation"""
        try:
            all_patterns = []
            
            # Geometric patterns
            all_patterns.extend(self._detect_symmetry_patterns(grid_repr))
            # Note: rotation and reflection patterns are covered by symmetry patterns
            
            # Sequential patterns
            all_patterns.extend(self._detect_sequence_patterns(grid_repr))
            all_patterns.extend(self._detect_progression_patterns(grid_repr))
            all_patterns.extend(self._detect_alternation_patterns(grid_repr))
            
            # Spatial patterns
            all_patterns.extend(self._detect_spatial_arrangements(grid_repr))
            
            # Repetition patterns
            all_patterns.extend(self._detect_repetition_patterns(grid_repr))
            
            # Transformation patterns
            all_patterns.extend(self._detect_transformation_patterns(grid_repr))
            
            # Complex patterns
            all_patterns.extend(self._detect_complex_patterns(grid_repr))
            
            # Filter and rank patterns
            filtered_patterns = self._filter_and_rank_patterns(all_patterns)
            
            return filtered_patterns
            
        except Exception as e:
            self.logger.error(f"Error in pattern detection: {e}")
            return []
    
    def _detect_symmetry_patterns(self, grid_repr: GridRepresentation) -> List[VisualPattern]:
        """Detect symmetry patterns"""
        patterns = []
        grid = grid_repr.original_grid
        
        # Check various symmetries
        symmetries_found = []
        
        # Vertical symmetry (reflection about vertical axis)
        if self._check_vertical_symmetry(grid):
            symmetries_found.append({
                'type': 'vertical',
                'axis': 'vertical_center',
                'confidence': 0.95
            })
        
        # Horizontal symmetry (reflection about horizontal axis)
        if self._check_horizontal_symmetry(grid):
            symmetries_found.append({
                'type': 'horizontal',
                'axis': 'horizontal_center',
                'confidence': 0.95
            })
        
        # Point symmetry (180-degree rotation)
        if self._check_point_symmetry(grid):
            symmetries_found.append({
                'type': 'point',
                'center': (grid.shape[0]//2, grid.shape[1]//2),
                'confidence': 0.9
            })
        
        # Diagonal symmetries
        if grid.shape[0] == grid.shape[1]:
            if self._check_main_diagonal_symmetry(grid):
                symmetries_found.append({
                    'type': 'main_diagonal',
                    'axis': 'main_diagonal',
                    'confidence': 0.9
                })
            
            if self._check_anti_diagonal_symmetry(grid):
                symmetries_found.append({
                    'type': 'anti_diagonal',
                    'axis': 'anti_diagonal',
                    'confidence': 0.9
                })
        
        # Create pattern matches
        matches = []
        for sym in symmetries_found:
            match = PatternMatch(
                pattern_type=PatternType.SYMMETRY,
                confidence=sym['confidence'],
                elements=[sym],
                parameters={'symmetry_type': sym['type']},
                description=f"{sym['type']} symmetry detected",
                complexity=PatternComplexity.SIMPLE
            )
            matches.append(match)
        
        if matches:
            pattern = VisualPattern(
                pattern_id="symmetry_global",
                pattern_type=PatternType.SYMMETRY,
                matches=matches,
                global_confidence=np.mean([m.confidence for m in matches]),
                pattern_strength=len(matches) / 4.0,  # Max 4 basic symmetries
                description=f"Grid exhibits {len(matches)} symmetry types",
                potential_rules=[f"Grid has {sym['type']} symmetry" for sym in symmetries_found]
            )
            patterns.append(pattern)
        
        return patterns
    
    def _detect_sequence_patterns(self, grid_repr: GridRepresentation) -> List[VisualPattern]:
        """Detect sequential patterns in object arrangements"""
        patterns = []
        objects = grid_repr.objects
        
        if len(objects) < 3:
            return patterns
        
        # Sort objects by different criteria to find sequences
        sort_criteria = [
            ('position_left_to_right', lambda obj: obj.bounding_box.center[1]),
            ('position_top_to_bottom', lambda obj: obj.bounding_box.center[0]),
            ('size', lambda obj: obj.size),
            ('color', lambda obj: obj.color)
        ]
        
        for criterion_name, sort_key in sort_criteria:
            sorted_objects = sorted(objects, key=sort_key)
            
            # Check for various sequence types
            sequences = self._analyze_object_sequence(sorted_objects, criterion_name)
            
            for seq in sequences:
                match = PatternMatch(
                    pattern_type=PatternType.SEQUENCE,
                    confidence=seq['confidence'],
                    elements=seq['objects'],
                    parameters={
                        'sequence_type': seq['type'],
                        'sort_criterion': criterion_name
                    },
                    description=seq['description'],
                    complexity=self._assess_sequence_complexity(seq)
                )
                
                if len(patterns) == 0 or not any(p.pattern_type == PatternType.SEQUENCE for p in patterns):
                    pattern = VisualPattern(
                        pattern_id=f"sequence_{criterion_name}",
                        pattern_type=PatternType.SEQUENCE,
                        matches=[match],
                        global_confidence=seq['confidence'],
                        pattern_strength=seq['strength'],
                        description=f"Sequential pattern in {criterion_name}",
                        potential_rules=[seq['rule']]
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _detect_progression_patterns(self, grid_repr: GridRepresentation) -> List[VisualPattern]:
        """Detect progression patterns (arithmetic, geometric)"""
        patterns = []
        objects = grid_repr.objects
        
        if len(objects) < 3:
            return patterns
        
        # Check size progressions
        size_progression = self._check_size_progression(objects)
        if size_progression:
            match = PatternMatch(
                pattern_type=PatternType.PROGRESSION,
                confidence=size_progression['confidence'],
                elements=size_progression['objects'],
                parameters={
                    'progression_type': size_progression['type'],
                    'attribute': 'size'
                },
                description=size_progression['description'],
                complexity=PatternComplexity.MODERATE
            )
            
            pattern = VisualPattern(
                pattern_id="size_progression",
                pattern_type=PatternType.PROGRESSION,
                matches=[match],
                global_confidence=size_progression['confidence'],
                pattern_strength=size_progression['strength'],
                description="Size progression pattern",
                potential_rules=[size_progression['rule']]
            )
            patterns.append(pattern)
        
        # Check color progressions (if objects have orderable colors)
        color_progression = self._check_color_progression(objects)
        if color_progression:
            match = PatternMatch(
                pattern_type=PatternType.PROGRESSION,
                confidence=color_progression['confidence'],
                elements=color_progression['objects'],
                parameters={
                    'progression_type': color_progression['type'],
                    'attribute': 'color'
                },
                description=color_progression['description'],
                complexity=PatternComplexity.MODERATE
            )
            
            pattern = VisualPattern(
                pattern_id="color_progression",
                pattern_type=PatternType.PROGRESSION,
                matches=[match],
                global_confidence=color_progression['confidence'],
                pattern_strength=color_progression['strength'],
                description="Color progression pattern",
                potential_rules=[color_progression['rule']]
            )
            patterns.append(pattern)
        
        return patterns
    
    def _detect_alternation_patterns(self, grid_repr: GridRepresentation) -> List[VisualPattern]:
        """Detect alternating patterns"""
        patterns = []
        objects = grid_repr.objects
        
        if len(objects) < 4:
            return patterns
        
        # Sort objects by position (left to right, top to bottom)
        sorted_objects = sorted(objects, key=lambda obj: (obj.bounding_box.center[0], obj.bounding_box.center[1]))
        
        # Check color alternation
        color_alt = self._check_color_alternation(sorted_objects)
        if color_alt:
            match = PatternMatch(
                pattern_type=PatternType.ALTERNATION,
                confidence=color_alt['confidence'],
                elements=color_alt['objects'],
                parameters={'alternation_attribute': 'color'},
                description=color_alt['description'],
                complexity=PatternComplexity.SIMPLE
            )
            
            pattern = VisualPattern(
                pattern_id="color_alternation",
                pattern_type=PatternType.ALTERNATION,
                matches=[match],
                global_confidence=color_alt['confidence'],
                pattern_strength=color_alt['strength'],
                description="Color alternation pattern",
                potential_rules=[color_alt['rule']]
            )
            patterns.append(pattern)
        
        # Check shape alternation
        shape_alt = self._check_shape_alternation(sorted_objects)
        if shape_alt:
            match = PatternMatch(
                pattern_type=PatternType.ALTERNATION,
                confidence=shape_alt['confidence'],
                elements=shape_alt['objects'],
                parameters={'alternation_attribute': 'shape'},
                description=shape_alt['description'],
                complexity=PatternComplexity.SIMPLE
            )
            
            pattern = VisualPattern(
                pattern_id="shape_alternation",
                pattern_type=PatternType.ALTERNATION,
                matches=[match],
                global_confidence=shape_alt['confidence'],
                pattern_strength=shape_alt['strength'],
                description="Shape alternation pattern",
                potential_rules=[shape_alt['rule']]
            )
            patterns.append(pattern)
        
        return patterns
    
    def _detect_spatial_arrangements(self, grid_repr: GridRepresentation) -> List[VisualPattern]:
        """Detect spatial arrangement patterns"""
        patterns = []
        objects = grid_repr.objects
        
        if len(objects) < 3:
            return patterns
        
        # Grid arrangement
        grid_pattern = self._check_grid_arrangement(objects)
        if grid_pattern:
            match = PatternMatch(
                pattern_type=PatternType.GRID_ARRANGEMENT,
                confidence=grid_pattern['confidence'],
                elements=grid_pattern['objects'],
                parameters={
                    'rows': grid_pattern['rows'],
                    'cols': grid_pattern['cols']
                },
                description=grid_pattern['description'],
                complexity=PatternComplexity.MODERATE
            )
            
            pattern = VisualPattern(
                pattern_id="grid_arrangement",
                pattern_type=PatternType.GRID_ARRANGEMENT,
                matches=[match],
                global_confidence=grid_pattern['confidence'],
                pattern_strength=grid_pattern['strength'],
                description="Objects arranged in grid pattern",
                potential_rules=[grid_pattern['rule']]
            )
            patterns.append(pattern)
        
        # Linear arrangement
        linear_pattern = self._check_linear_arrangement(objects)
        if linear_pattern:
            match = PatternMatch(
                pattern_type=PatternType.LINEAR_ARRANGEMENT,
                confidence=linear_pattern['confidence'],
                elements=linear_pattern['objects'],
                parameters={'direction': linear_pattern['direction']},
                description=linear_pattern['description'],
                complexity=PatternComplexity.SIMPLE
            )
            
            pattern = VisualPattern(
                pattern_id="linear_arrangement",
                pattern_type=PatternType.LINEAR_ARRANGEMENT,
                matches=[match],
                global_confidence=linear_pattern['confidence'],
                pattern_strength=linear_pattern['strength'],
                description="Objects arranged linearly",
                potential_rules=[linear_pattern['rule']]
            )
            patterns.append(pattern)
        
        return patterns
    
    def _detect_repetition_patterns(self, grid_repr: GridRepresentation) -> List[VisualPattern]:
        """Detect repetition and tiling patterns"""
        patterns = []
        
        # This would involve more complex analysis of repeating subgrids
        # For now, we'll implement basic repetition detection
        
        return patterns
    
    def _detect_transformation_patterns(self, grid_repr: GridRepresentation) -> List[VisualPattern]:
        """Detect transformation patterns"""
        patterns = []
        objects = grid_repr.objects
        
        # Group objects by similar properties to find transformation series
        color_groups = defaultdict(list)
        for obj in objects:
            color_groups[obj.color].append(obj)
        
        # Check for scaling transformations within color groups
        for color, color_objects in color_groups.items():
            if len(color_objects) >= 3:
                scaling_pattern = self._check_scaling_pattern(color_objects)
                if scaling_pattern:
                    match = PatternMatch(
                        pattern_type=PatternType.SCALING,
                        confidence=scaling_pattern['confidence'],
                        elements=scaling_pattern['objects'],
                        parameters={'scale_factor': scaling_pattern['scale_factor']},
                        description=scaling_pattern['description'],
                        complexity=PatternComplexity.MODERATE
                    )
                    
                    pattern = VisualPattern(
                        pattern_id=f"scaling_color_{color}",
                        pattern_type=PatternType.SCALING,
                        matches=[match],
                        global_confidence=scaling_pattern['confidence'],
                        pattern_strength=scaling_pattern['strength'],
                        description=f"Scaling pattern in color {color} objects",
                        potential_rules=[scaling_pattern['rule']]
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _detect_complex_patterns(self, grid_repr: GridRepresentation) -> List[VisualPattern]:
        """Detect complex patterns (nested, composite, conditional)"""
        patterns = []
        
        # Nested patterns - objects inside other objects
        nested_pattern = self._check_nested_patterns(grid_repr.objects)
        if nested_pattern:
            match = PatternMatch(
                pattern_type=PatternType.NESTED,
                confidence=nested_pattern['confidence'],
                elements=nested_pattern['pairs'],
                parameters={'nesting_depth': nested_pattern['depth']},
                description=nested_pattern['description'],
                complexity=PatternComplexity.COMPLEX
            )
            
            pattern = VisualPattern(
                pattern_id="nested_objects",
                pattern_type=PatternType.NESTED,
                matches=[match],
                global_confidence=nested_pattern['confidence'],
                pattern_strength=nested_pattern['strength'],
                description="Objects nested within other objects",
                potential_rules=[nested_pattern['rule']]
            )
            patterns.append(pattern)
        
        return patterns
    
    # Helper methods for specific pattern checks
    
    def _check_vertical_symmetry(self, grid: np.ndarray) -> bool:
        """Check if grid has vertical symmetry"""
        return np.array_equal(grid, np.fliplr(grid))
    
    def _check_horizontal_symmetry(self, grid: np.ndarray) -> bool:
        """Check if grid has horizontal symmetry"""
        return np.array_equal(grid, np.flipud(grid))
    
    def _check_point_symmetry(self, grid: np.ndarray) -> bool:
        """Check if grid has point symmetry (180-degree rotation)"""
        return np.array_equal(grid, np.rot90(grid, 2))
    
    def _check_main_diagonal_symmetry(self, grid: np.ndarray) -> bool:
        """Check if grid has main diagonal symmetry"""
        return np.array_equal(grid, grid.T)
    
    def _check_anti_diagonal_symmetry(self, grid: np.ndarray) -> bool:
        """Check if grid has anti-diagonal symmetry"""
        return np.array_equal(grid, np.fliplr(grid.T))
    
    def _analyze_object_sequence(self, sorted_objects: List[GridObject], criterion: str) -> List[Dict[str, Any]]:
        """Analyze sequence patterns in sorted objects"""
        sequences = []
        
        if len(sorted_objects) < 3:
            return sequences
        
        # Check for simple sequences based on different attributes
        if criterion in ['position_left_to_right', 'position_top_to_bottom']:
            # Check for evenly spaced objects
            positions = [obj.bounding_box.center[1 if 'left_to_right' in criterion else 0] for obj in sorted_objects]
            if self._is_arithmetic_sequence(positions, tolerance=1.5):
                sequences.append({
                    'type': 'evenly_spaced',
                    'objects': sorted_objects,
                    'confidence': 0.8,
                    'strength': 0.7,
                    'description': f'Objects are evenly spaced {criterion}',
                    'rule': f'Objects maintain equal spacing in {criterion} direction'
                })
        
        elif criterion == 'size':
            sizes = [obj.size for obj in sorted_objects]
            if self._is_arithmetic_sequence(sizes):
                sequences.append({
                    'type': 'arithmetic_size',
                    'objects': sorted_objects,
                    'confidence': 0.9,
                    'strength': 0.8,
                    'description': 'Object sizes form arithmetic sequence',
                    'rule': 'Object sizes increase by constant amount'
                })
            elif self._is_geometric_sequence(sizes):
                sequences.append({
                    'type': 'geometric_size',
                    'objects': sorted_objects,
                    'confidence': 0.85,
                    'strength': 0.7,
                    'description': 'Object sizes form geometric sequence',
                    'rule': 'Object sizes increase by constant ratio'
                })
        
        return sequences
    
    def _check_size_progression(self, objects: List[GridObject]) -> Optional[Dict[str, Any]]:
        """Check for size progression patterns"""
        if len(objects) < 3:
            return None
        
        # Sort by position (left to right, then top to bottom)
        sorted_objects = sorted(objects, key=lambda obj: (obj.bounding_box.center[0], obj.bounding_box.center[1]))
        sizes = [obj.size for obj in sorted_objects]
        
        if self._is_arithmetic_sequence(sizes):
            return {
                'type': 'arithmetic',
                'objects': sorted_objects,
                'confidence': 0.9,
                'strength': 0.8,
                'description': 'Object sizes increase arithmetically',
                'rule': f'Size increases by {sizes[1] - sizes[0]} each step'
            }
        elif self._is_geometric_sequence(sizes):
            ratio = sizes[1] / sizes[0] if sizes[0] > 0 else 1
            return {
                'type': 'geometric',
                'objects': sorted_objects,
                'confidence': 0.85,
                'strength': 0.7,
                'description': 'Object sizes increase geometrically',
                'rule': f'Size multiplies by {ratio:.2f} each step'
            }
        
        return None
    
    def _check_color_progression(self, objects: List[GridObject]) -> Optional[Dict[str, Any]]:
        """Check for color progression patterns"""
        if len(objects) < 3:
            return None
        
        # Sort by position
        sorted_objects = sorted(objects, key=lambda obj: (obj.bounding_box.center[0], obj.bounding_box.center[1]))
        colors = [obj.color for obj in sorted_objects]
        
        # Check if colors form an arithmetic sequence (assuming orderable colors)
        if self._is_arithmetic_sequence(colors):
            return {
                'type': 'arithmetic',
                'objects': sorted_objects,
                'confidence': 0.8,
                'strength': 0.7,
                'description': 'Object colors increase in sequence',
                'rule': f'Color increases by {colors[1] - colors[0]} each step'
            }
        
        return None
    
    def _check_color_alternation(self, sorted_objects: List[GridObject]) -> Optional[Dict[str, Any]]:
        """Check for color alternation pattern"""
        if len(sorted_objects) < 4:
            return None
        
        colors = [obj.color for obj in sorted_objects]
        
        # Check for simple alternation (ABAB pattern)
        if len(set(colors)) == 2:
            alternates = True
            for i in range(2, len(colors)):
                if colors[i] != colors[i-2]:
                    alternates = False
                    break
            
            if alternates:
                unique_colors = list(set(colors))
                return {
                    'objects': sorted_objects,
                    'confidence': 0.9,
                    'strength': 0.8,
                    'description': f'Colors alternate between {unique_colors[0]} and {unique_colors[1]}',
                    'rule': f'Colors alternate between {unique_colors[0]} and {unique_colors[1]}'
                }
        
        return None
    
    def _check_shape_alternation(self, sorted_objects: List[GridObject]) -> Optional[Dict[str, Any]]:
        """Check for shape alternation pattern"""
        if len(sorted_objects) < 4:
            return None
        
        shapes = [obj.shape for obj in sorted_objects]
        
        # Check for simple alternation
        if len(set(shapes)) == 2:
            alternates = True
            for i in range(2, len(shapes)):
                if shapes[i] != shapes[i-2]:
                    alternates = False
                    break
            
            if alternates:
                unique_shapes = list(set(shapes))
                return {
                    'objects': sorted_objects,
                    'confidence': 0.85,
                    'strength': 0.7,
                    'description': f'Shapes alternate between {unique_shapes[0].value} and {unique_shapes[1].value}',
                    'rule': f'Shapes alternate between {unique_shapes[0].value} and {unique_shapes[1].value}'
                }
        
        return None
    
    def _check_grid_arrangement(self, objects: List[GridObject]) -> Optional[Dict[str, Any]]:
        """Check if objects are arranged in a grid pattern"""
        if len(objects) < 6:  # Need at least 2x3 or 3x2 grid
            return None
        
        # Get object centers
        centers = [obj.bounding_box.center for obj in objects]
        
        # Try to find grid dimensions
        for rows in range(2, int(len(objects)**0.5) + 2):
            if len(objects) % rows == 0:
                cols = len(objects) // rows
                
                if self._objects_form_grid(centers, rows, cols):
                    return {
                        'objects': objects,
                        'rows': rows,
                        'cols': cols,
                        'confidence': 0.8,
                        'strength': 0.7,
                        'description': f'Objects arranged in {rows}x{cols} grid',
                        'rule': f'Objects form {rows}x{cols} grid arrangement'
                    }
        
        return None
    
    def _check_linear_arrangement(self, objects: List[GridObject]) -> Optional[Dict[str, Any]]:
        """Check if objects are arranged linearly"""
        if len(objects) < 3:
            return None
        
        centers = [obj.bounding_box.center for obj in objects]
        
        # Check horizontal alignment
        rows = [center[0] for center in centers]
        if max(rows) - min(rows) < 2:  # Allow some tolerance
            return {
                'objects': objects,
                'direction': 'horizontal',
                'confidence': 0.8,
                'strength': 0.7,
                'description': 'Objects arranged horizontally',
                'rule': 'Objects form horizontal line'
            }
        
        # Check vertical alignment
        cols = [center[1] for center in centers]
        if max(cols) - min(cols) < 2:  # Allow some tolerance
            return {
                'objects': objects,
                'direction': 'vertical',
                'confidence': 0.8,
                'strength': 0.7,
                'description': 'Objects arranged vertically',
                'rule': 'Objects form vertical line'
            }
        
        # Check diagonal alignment
        if self._is_diagonal_line(centers):
            return {
                'objects': objects,
                'direction': 'diagonal',
                'confidence': 0.7,
                'strength': 0.6,
                'description': 'Objects arranged diagonally',
                'rule': 'Objects form diagonal line'
            }
        
        return None
    
    def _check_scaling_pattern(self, objects: List[GridObject]) -> Optional[Dict[str, Any]]:
        """Check for scaling transformation pattern"""
        if len(objects) < 3:
            return None
        
        # Sort by position
        sorted_objects = sorted(objects, key=lambda obj: (obj.bounding_box.center[0], obj.bounding_box.center[1]))
        sizes = [obj.size for obj in sorted_objects]
        
        # Check if sizes form geometric sequence
        if self._is_geometric_sequence(sizes):
            ratio = sizes[1] / sizes[0] if sizes[0] > 0 else 1
            return {
                'objects': sorted_objects,
                'scale_factor': ratio,
                'confidence': 0.8,
                'strength': 0.7,
                'description': f'Objects scale by factor {ratio:.2f}',
                'rule': f'Each object is {ratio:.2f}x larger than previous'
            }
        
        return None
    
    def _check_nested_patterns(self, objects: List[GridObject]) -> Optional[Dict[str, Any]]:
        """Check for nested object patterns"""
        if len(objects) < 2:
            return None
        
        nested_pairs = []
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects):
                if i != j and self._is_nested(obj1, obj2):
                    nested_pairs.append((obj1, obj2))
        
        if len(nested_pairs) >= 2:
            return {
                'pairs': nested_pairs,
                'depth': 1,  # Could be enhanced to detect deeper nesting
                'confidence': 0.7,
                'strength': len(nested_pairs) / len(objects),
                'description': f'{len(nested_pairs)} nested object pairs found',
                'rule': 'Some objects are nested within others'
            }
        
        return None
    
    def _is_nested(self, obj1: GridObject, obj2: GridObject) -> bool:
        """Check if obj1 is nested inside obj2"""
        bbox1, bbox2 = obj1.bounding_box, obj2.bounding_box
        return (bbox1.min_row >= bbox2.min_row and bbox1.max_row <= bbox2.max_row and
                bbox1.min_col >= bbox2.min_col and bbox1.max_col <= bbox2.max_col and
                obj1.size < obj2.size)
    
    def _is_arithmetic_sequence(self, values: List[float], tolerance: float = 0.1) -> bool:
        """Check if values form an arithmetic sequence"""
        if len(values) < 3:
            return False
        
        differences = [values[i+1] - values[i] for i in range(len(values)-1)]
        avg_diff = np.mean(differences)
        
        return all(abs(diff - avg_diff) <= tolerance for diff in differences)
    
    def _is_geometric_sequence(self, values: List[float], tolerance: float = 0.1) -> bool:
        """Check if values form a geometric sequence"""
        if len(values) < 3 or any(v <= 0 for v in values):
            return False
        
        ratios = [values[i+1] / values[i] for i in range(len(values)-1)]
        avg_ratio = np.mean(ratios)
        
        return all(abs(ratio - avg_ratio) <= tolerance for ratio in ratios)
    
    def _objects_form_grid(self, centers: List[Tuple[float, float]], rows: int, cols: int) -> bool:
        """Check if object centers form a grid pattern"""
        if len(centers) != rows * cols:
            return False
        
        # Sort centers by row, then by column
        sorted_centers = sorted(centers, key=lambda c: (c[0], c[1]))
        
        # Check if sorted centers form regular grid
        tolerance = 2.0
        
        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                if idx >= len(sorted_centers):
                    return False
                
                expected_row = sorted_centers[0][0] + r * (sorted_centers[cols][0] - sorted_centers[0][0]) if rows > 1 and cols < len(sorted_centers) else sorted_centers[0][0]
                expected_col = sorted_centers[0][1] + c * (sorted_centers[1][1] - sorted_centers[0][1]) if cols > 1 else sorted_centers[0][1]
                
                actual = sorted_centers[idx]
                if abs(actual[0] - expected_row) > tolerance or abs(actual[1] - expected_col) > tolerance:
                    return False
        
        return True
    
    def _is_diagonal_line(self, centers: List[Tuple[float, float]]) -> bool:
        """Check if centers form a diagonal line"""
        if len(centers) < 3:
            return False
        
        # Calculate slope between first two points
        p1, p2 = centers[0], centers[1]
        if abs(p2[1] - p1[1]) < 1e-6:  # Vertical line
            return False
        
        slope = (p2[0] - p1[0]) / (p2[1] - p1[1])
        
        # Check if all points lie on the same line
        tolerance = 1.0
        for i in range(2, len(centers)):
            p = centers[i]
            expected_row = p1[0] + slope * (p[1] - p1[1])
            if abs(p[0] - expected_row) > tolerance:
                return False
        
        return True
    
    def _assess_sequence_complexity(self, sequence: Dict[str, Any]) -> PatternComplexity:
        """Assess complexity of a sequence pattern"""
        if sequence['type'] in ['evenly_spaced', 'arithmetic_size']:
            return PatternComplexity.SIMPLE
        elif sequence['type'] in ['geometric_size']:
            return PatternComplexity.MODERATE
        else:
            return PatternComplexity.COMPLEX
    
    def _filter_and_rank_patterns(self, patterns: List[VisualPattern]) -> List[VisualPattern]:
        """Filter and rank patterns by confidence and strength"""
        # Filter by minimum confidence
        filtered = [p for p in patterns if p.global_confidence >= self.min_confidence and p.pattern_strength >= self.min_pattern_strength]
        
        # Sort by combined score (confidence * strength * complexity_weight)
        def pattern_score(pattern):
            complexity_weight = self.complexity_weights.get(pattern.matches[0].complexity, 0.5)
            return pattern.global_confidence * pattern.pattern_strength * complexity_weight
        
        filtered.sort(key=pattern_score, reverse=True)
        
        # Limit number of patterns per type
        type_counts = defaultdict(int)
        final_patterns = []
        
        for pattern in filtered:
            if type_counts[pattern.pattern_type] < self.max_patterns_per_type:
                final_patterns.append(pattern)
                type_counts[pattern.pattern_type] += 1
        
        return final_patterns