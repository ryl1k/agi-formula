"""
Visual Grid Processor for ARC-AGI Tasks

Specialized processor for analyzing visual grids in ARC-AGI challenges:
- Grid parsing and structured representation
- Shape and object detection
- Color pattern analysis
- Spatial relationship extraction
- Connectivity and topology analysis
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, deque


class ObjectShape(Enum):
    """Types of shapes that can be detected"""
    RECTANGLE = "rectangle"
    SQUARE = "square"
    LINE_HORIZONTAL = "line_horizontal"
    LINE_VERTICAL = "line_vertical"
    LINE_DIAGONAL = "line_diagonal"
    L_SHAPE = "l_shape"
    T_SHAPE = "t_shape"
    CROSS = "cross"
    CIRCLE_LIKE = "circle_like"
    IRREGULAR = "irregular"
    SINGLE_CELL = "single_cell"


class SpatialRelation(Enum):
    """Spatial relationships between objects"""
    ABOVE = "above"
    BELOW = "below"
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    INSIDE = "inside"
    OUTSIDE = "outside"
    ADJACENT = "adjacent"
    DIAGONAL = "diagonal"
    OVERLAPPING = "overlapping"
    SEPARATE = "separate"
    TOUCHING = "touching"


@dataclass
class BoundingBox:
    """Bounding box for grid objects"""
    min_row: int
    max_row: int
    min_col: int
    max_col: int
    
    @property
    def width(self) -> int:
        return self.max_col - self.min_col + 1
    
    @property
    def height(self) -> int:
        return self.max_row - self.min_row + 1
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.min_row + self.max_row) / 2, (self.min_col + self.max_col) / 2)


@dataclass
class GridObject:
    """Represents a detected object in the grid"""
    object_id: int
    color: int
    cells: Set[Tuple[int, int]]
    bounding_box: BoundingBox
    shape: ObjectShape
    size: int
    confidence: float = 0.8
    
    def __post_init__(self):
        self.size = len(self.cells)
    
    def get_relative_positions(self) -> Set[Tuple[int, int]]:
        """Get positions relative to bounding box top-left"""
        min_row, min_col = self.bounding_box.min_row, self.bounding_box.min_col
        return {(r - min_row, c - min_col) for r, c in self.cells}
    
    def overlaps_with(self, other: 'GridObject') -> bool:
        """Check if this object overlaps with another"""
        return bool(self.cells & other.cells)
    
    def is_adjacent_to(self, other: 'GridObject') -> bool:
        """Check if this object is adjacent to another"""
        for r1, c1 in self.cells:
            for r2, c2 in other.cells:
                if abs(r1 - r2) + abs(c1 - c2) == 1:
                    return True
        return False


@dataclass
class GridPattern:
    """Represents a pattern detected in the grid"""
    pattern_type: str
    elements: List[Any]
    confidence: float
    description: str
    spatial_extent: Optional[BoundingBox] = None


@dataclass
class GridRepresentation:
    """Structured representation of a grid"""
    original_grid: np.ndarray
    height: int
    width: int
    colors: Set[int]
    background_color: int
    
    # Detected objects
    objects: List[GridObject]
    object_by_id: Dict[int, GridObject]
    objects_by_color: Dict[int, List[GridObject]]
    
    # Spatial relationships
    spatial_relations: Dict[Tuple[int, int], List[SpatialRelation]]
    
    # Patterns
    detected_patterns: List[GridPattern]
    
    # Grid properties
    symmetries: List[str]
    connectivity_graph: Dict[int, Set[int]]
    color_distribution: Dict[int, int]


class VisualGridProcessor:
    """Main processor for visual grid analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Shape detection templates
        self._init_shape_templates()
        
        # Color analysis settings
        self.min_object_size = 1
        self.max_object_size = 100
        self.connectivity_type = 4  # 4-connected or 8-connected
        
    def _init_shape_templates(self):
        """Initialize shape detection templates"""
        self.shape_templates = {
            ObjectShape.RECTANGLE: self._is_rectangle,
            ObjectShape.SQUARE: self._is_square,
            ObjectShape.LINE_HORIZONTAL: self._is_horizontal_line,
            ObjectShape.LINE_VERTICAL: self._is_vertical_line,
            ObjectShape.LINE_DIAGONAL: self._is_diagonal_line,
            ObjectShape.L_SHAPE: self._is_l_shape,
            ObjectShape.T_SHAPE: self._is_t_shape,
            ObjectShape.CROSS: self._is_cross,
            ObjectShape.SINGLE_CELL: self._is_single_cell
        }
    
    def process_grid(self, grid: Union[List[List[int]], np.ndarray]) -> GridRepresentation:
        """Main processing function for a grid"""
        try:
            # Convert to numpy array if needed
            if isinstance(grid, list):
                grid = np.array(grid)
            
            height, width = grid.shape
            
            # Basic grid analysis
            colors = set(grid.flatten())
            background_color = self._detect_background_color(grid)
            color_distribution = {color: int(np.sum(grid == color)) for color in colors}
            
            # Object detection
            objects = self._detect_objects(grid, background_color)
            
            # Build object mappings
            object_by_id = {obj.object_id: obj for obj in objects}
            objects_by_color = defaultdict(list)
            for obj in objects:
                objects_by_color[obj.color].append(obj)
            
            # Spatial relationship analysis
            spatial_relations = self._analyze_spatial_relations(objects)
            
            # Pattern detection
            patterns = self._detect_patterns(grid, objects)
            
            # Symmetry detection
            symmetries = self._detect_symmetries(grid)
            
            # Build connectivity graph
            connectivity_graph = self._build_connectivity_graph(objects)
            
            return GridRepresentation(
                original_grid=grid,
                height=height,
                width=width,
                colors=colors,
                background_color=background_color,
                objects=objects,
                object_by_id=object_by_id,
                objects_by_color=dict(objects_by_color),
                spatial_relations=spatial_relations,
                detected_patterns=patterns,
                symmetries=symmetries,
                connectivity_graph=connectivity_graph,
                color_distribution=color_distribution
            )
            
        except Exception as e:
            self.logger.error(f"Error processing grid: {e}")
            raise
    
    def _detect_background_color(self, grid: np.ndarray) -> int:
        """Detect the most likely background color"""
        # Background is usually the most common color
        colors, counts = np.unique(grid, return_counts=True)
        background_idx = np.argmax(counts)
        return int(colors[background_idx])
    
    def _detect_objects(self, grid: np.ndarray, background_color: int) -> List[GridObject]:
        """Detect connected components as objects"""
        height, width = grid.shape
        visited = np.zeros_like(grid, dtype=bool)
        objects = []
        object_id = 0
        
        for row in range(height):
            for col in range(width):
                if not visited[row, col] and grid[row, col] != background_color:
                    # Found new object - flood fill to get all connected cells
                    color = grid[row, col]
                    cells = self._flood_fill(grid, row, col, color, visited)
                    
                    if self.min_object_size <= len(cells) <= self.max_object_size:
                        # Create bounding box
                        rows = [r for r, c in cells]
                        cols = [c for r, c in cells]
                        bbox = BoundingBox(
                            min_row=min(rows),
                            max_row=max(rows),
                            min_col=min(cols),
                            max_col=max(cols)
                        )
                        
                        # Detect shape
                        shape = self._classify_shape(cells, bbox)
                        
                        # Create object
                        obj = GridObject(
                            object_id=object_id,
                            color=int(color),
                            cells=cells,
                            bounding_box=bbox,
                            shape=shape,
                            size=len(cells)
                        )
                        
                        objects.append(obj)
                        object_id += 1
        
        return objects
    
    def _flood_fill(self, grid: np.ndarray, start_row: int, start_col: int, 
                   target_color: int, visited: np.ndarray) -> Set[Tuple[int, int]]:
        """Flood fill algorithm to find connected component"""
        height, width = grid.shape
        cells = set()
        queue = deque([(start_row, start_col)])
        
        while queue:
            row, col = queue.popleft()
            
            if (row < 0 or row >= height or col < 0 or col >= width or
                visited[row, col] or grid[row, col] != target_color):
                continue
            
            visited[row, col] = True
            cells.add((row, col))
            
            # Add neighbors (4-connected)
            if self.connectivity_type == 4:
                neighbors = [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]
            else:  # 8-connected
                neighbors = [(row+dr, col+dc) for dr in [-1, 0, 1] for dc in [-1, 0, 1] if dr != 0 or dc != 0]
            
            for nr, nc in neighbors:
                if 0 <= nr < height and 0 <= nc < width:
                    queue.append((nr, nc))
        
        return cells
    
    def _classify_shape(self, cells: Set[Tuple[int, int]], bbox: BoundingBox) -> ObjectShape:
        """Classify the shape of an object"""
        # Try each shape template
        for shape_type, classifier in self.shape_templates.items():
            if classifier(cells, bbox):
                return shape_type
        
        return ObjectShape.IRREGULAR
    
    def _is_single_cell(self, cells: Set[Tuple[int, int]], bbox: BoundingBox) -> bool:
        """Check if object is a single cell"""
        return len(cells) == 1
    
    def _is_rectangle(self, cells: Set[Tuple[int, int]], bbox: BoundingBox) -> bool:
        """Check if object is a rectangle"""
        if len(cells) != bbox.area:
            return False
        
        # Check if all cells within bounding box are present
        for r in range(bbox.min_row, bbox.max_row + 1):
            for c in range(bbox.min_col, bbox.max_col + 1):
                if (r, c) not in cells:
                    return False
        return True
    
    def _is_square(self, cells: Set[Tuple[int, int]], bbox: BoundingBox) -> bool:
        """Check if object is a square"""
        return self._is_rectangle(cells, bbox) and bbox.width == bbox.height
    
    def _is_horizontal_line(self, cells: Set[Tuple[int, int]], bbox: BoundingBox) -> bool:
        """Check if object is a horizontal line"""
        if bbox.height != 1:
            return False
        
        # Check if all cells in the row are present
        expected_cells = {(bbox.min_row, c) for c in range(bbox.min_col, bbox.max_col + 1)}
        return cells == expected_cells
    
    def _is_vertical_line(self, cells: Set[Tuple[int, int]], bbox: BoundingBox) -> bool:
        """Check if object is a vertical line"""
        if bbox.width != 1:
            return False
        
        # Check if all cells in the column are present
        expected_cells = {(r, bbox.min_col) for r in range(bbox.min_row, bbox.max_row + 1)}
        return cells == expected_cells
    
    def _is_diagonal_line(self, cells: Set[Tuple[int, int]], bbox: BoundingBox) -> bool:
        """Check if object is a diagonal line"""
        if len(cells) < 3:
            return False
        
        # Check if cells form a diagonal pattern
        cells_list = sorted(list(cells))
        
        # Check main diagonal (top-left to bottom-right)
        dr, dc = 1, 1
        start_r, start_c = cells_list[0]
        diagonal_cells = {(start_r + i * dr, start_c + i * dc) for i in range(len(cells_list))}
        if cells == diagonal_cells:
            return True
        
        # Check anti-diagonal (top-right to bottom-left)
        dr, dc = 1, -1
        start_r, start_c = min(cells_list, key=lambda x: (x[0], -x[1]))
        diagonal_cells = {(start_r + i * dr, start_c + i * dc) for i in range(len(cells_list))}
        return cells == diagonal_cells
    
    def _is_l_shape(self, cells: Set[Tuple[int, int]], bbox: BoundingBox) -> bool:
        """Check if object is L-shaped"""
        if len(cells) < 4 or bbox.area < 4:
            return False
        
        # L-shape has two perpendicular arms
        # Try different L orientations
        for corner_r, corner_c in cells:
            # Check if this could be the corner of an L
            horizontal_arm = {(corner_r, c) for c in range(bbox.min_col, bbox.max_col + 1) if (corner_r, c) in cells}
            vertical_arm = {(r, corner_c) for r in range(bbox.min_row, bbox.max_row + 1) if (r, corner_c) in cells}
            
            # L-shape should be the union of horizontal and vertical arms
            if len(horizontal_arm) >= 2 and len(vertical_arm) >= 2:
                l_cells = horizontal_arm | vertical_arm
                if cells == l_cells:
                    return True
        
        return False
    
    def _is_t_shape(self, cells: Set[Tuple[int, int]], bbox: BoundingBox) -> bool:
        """Check if object is T-shaped"""
        if len(cells) < 5:
            return False
        
        # T-shape has a horizontal bar and vertical stem
        for r in range(bbox.min_row, bbox.max_row + 1):
            for c in range(bbox.min_col, bbox.max_col + 1):
                if (r, c) not in cells:
                    continue
                
                # Try this as the intersection point
                horizontal_line = {(r, col) for col in range(bbox.min_col, bbox.max_col + 1) if (r, col) in cells}
                vertical_line = {(row, c) for row in range(bbox.min_row, bbox.max_row + 1) if (row, c) in cells}
                
                if len(horizontal_line) >= 3 and len(vertical_line) >= 3:
                    t_cells = horizontal_line | vertical_line
                    if cells == t_cells:
                        return True
        
        return False
    
    def _is_cross(self, cells: Set[Tuple[int, int]], bbox: BoundingBox) -> bool:
        """Check if object is cross-shaped"""
        if len(cells) < 5:
            return False
        
        # Cross has both horizontal and vertical lines through center
        center_r = (bbox.min_row + bbox.max_row) // 2
        center_c = (bbox.min_col + bbox.max_col) // 2
        
        if (center_r, center_c) not in cells:
            return False
        
        horizontal_line = {(center_r, c) for c in range(bbox.min_col, bbox.max_col + 1) if (center_r, c) in cells}
        vertical_line = {(r, center_c) for r in range(bbox.min_row, bbox.max_row + 1) if (r, center_c) in cells}
        
        cross_cells = horizontal_line | vertical_line
        return cells == cross_cells and len(horizontal_line) >= 3 and len(vertical_line) >= 3
    
    def _analyze_spatial_relations(self, objects: List[GridObject]) -> Dict[Tuple[int, int], List[SpatialRelation]]:
        """Analyze spatial relationships between objects"""
        relations = {}
        
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects):
                if i >= j:  # Avoid duplicates and self-relations
                    continue
                
                obj_relations = []
                
                # Check basic spatial relations
                if obj1.overlaps_with(obj2):
                    obj_relations.append(SpatialRelation.OVERLAPPING)
                elif obj1.is_adjacent_to(obj2):
                    obj_relations.append(SpatialRelation.TOUCHING)
                else:
                    obj_relations.append(SpatialRelation.SEPARATE)
                
                # Check directional relations
                center1 = obj1.bounding_box.center
                center2 = obj2.bounding_box.center
                
                if center1[0] < center2[0]:  # obj1 is above obj2
                    obj_relations.append(SpatialRelation.ABOVE)
                elif center1[0] > center2[0]:  # obj1 is below obj2
                    obj_relations.append(SpatialRelation.BELOW)
                
                if center1[1] < center2[1]:  # obj1 is left of obj2
                    obj_relations.append(SpatialRelation.LEFT_OF)
                elif center1[1] > center2[1]:  # obj1 is right of obj2
                    obj_relations.append(SpatialRelation.RIGHT_OF)
                
                # Check containment
                if self._is_inside(obj1, obj2):
                    obj_relations.append(SpatialRelation.INSIDE)
                elif self._is_inside(obj2, obj1):
                    obj_relations.append(SpatialRelation.OUTSIDE)
                
                relations[(obj1.object_id, obj2.object_id)] = obj_relations
        
        return relations
    
    def _is_inside(self, obj1: GridObject, obj2: GridObject) -> bool:
        """Check if obj1 is inside obj2"""
        return (obj1.bounding_box.min_row >= obj2.bounding_box.min_row and
                obj1.bounding_box.max_row <= obj2.bounding_box.max_row and
                obj1.bounding_box.min_col >= obj2.bounding_box.min_col and
                obj1.bounding_box.max_col <= obj2.bounding_box.max_col)
    
    def _detect_patterns(self, grid: np.ndarray, objects: List[GridObject]) -> List[GridPattern]:
        """Detect various patterns in the grid"""
        patterns = []
        
        # Color patterns
        patterns.extend(self._detect_color_patterns(objects))
        
        # Size patterns
        patterns.extend(self._detect_size_patterns(objects))
        
        # Shape patterns
        patterns.extend(self._detect_shape_patterns(objects))
        
        # Spatial arrangement patterns
        patterns.extend(self._detect_spatial_patterns(objects))
        
        # Repetition patterns
        patterns.extend(self._detect_repetition_patterns(grid, objects))
        
        return patterns
    
    def _detect_color_patterns(self, objects: List[GridObject]) -> List[GridPattern]:
        """Detect color-based patterns"""
        patterns = []
        
        if not objects:
            return patterns
        
        # Analyze color distribution
        color_counts = defaultdict(int)
        for obj in objects:
            color_counts[obj.color] += 1
        
        # Single color pattern
        if len(color_counts) == 1:
            color = list(color_counts.keys())[0]
            patterns.append(GridPattern(
                pattern_type="uniform_color",
                elements=[color],
                confidence=0.9,
                description=f"All objects have the same color: {color}"
            ))
        
        # Alternating colors
        if len(objects) >= 4 and len(color_counts) == 2:
            colors = list(color_counts.keys())
            obj_colors = [obj.color for obj in objects]
            
            # Check for alternating pattern
            alternating = True
            for i in range(1, len(obj_colors)):
                if obj_colors[i] == obj_colors[i-1]:
                    alternating = False
                    break
            
            if alternating:
                patterns.append(GridPattern(
                    pattern_type="alternating_colors",
                    elements=colors,
                    confidence=0.8,
                    description=f"Objects alternate between colors {colors[0]} and {colors[1]}"
                ))
        
        return patterns
    
    def _detect_size_patterns(self, objects: List[GridObject]) -> List[GridPattern]:
        """Detect size-based patterns"""
        patterns = []
        
        if not objects:
            return patterns
        
        sizes = [obj.size for obj in objects]
        
        # Uniform size
        if len(set(sizes)) == 1:
            patterns.append(GridPattern(
                pattern_type="uniform_size",
                elements=[sizes[0]],
                confidence=0.9,
                description=f"All objects have the same size: {sizes[0]}"
            ))
        
        # Increasing/decreasing sizes
        if len(objects) >= 3:
            if sizes == sorted(sizes):
                patterns.append(GridPattern(
                    pattern_type="increasing_size",
                    elements=sizes,
                    confidence=0.8,
                    description="Object sizes increase in order"
                ))
            elif sizes == sorted(sizes, reverse=True):
                patterns.append(GridPattern(
                    pattern_type="decreasing_size",
                    elements=sizes,
                    confidence=0.8,
                    description="Object sizes decrease in order"
                ))
        
        return patterns
    
    def _detect_shape_patterns(self, objects: List[GridObject]) -> List[GridPattern]:
        """Detect shape-based patterns"""
        patterns = []
        
        if not objects:
            return patterns
        
        shapes = [obj.shape for obj in objects]
        shape_counts = defaultdict(int)
        for shape in shapes:
            shape_counts[shape] += 1
        
        # Uniform shape
        if len(shape_counts) == 1:
            shape = list(shape_counts.keys())[0]
            patterns.append(GridPattern(
                pattern_type="uniform_shape",
                elements=[shape],
                confidence=0.9,
                description=f"All objects have the same shape: {shape.value}"
            ))
        
        return patterns
    
    def _detect_spatial_patterns(self, objects: List[GridObject]) -> List[GridPattern]:
        """Detect spatial arrangement patterns"""
        patterns = []
        
        if len(objects) < 2:
            return patterns
        
        # Linear arrangements
        centers = [obj.bounding_box.center for obj in objects]
        
        # Check for horizontal alignment
        if len(set(center[0] for center in centers)) == 1:
            patterns.append(GridPattern(
                pattern_type="horizontal_alignment",
                elements=centers,
                confidence=0.8,
                description="Objects are horizontally aligned"
            ))
        
        # Check for vertical alignment
        if len(set(center[1] for center in centers)) == 1:
            patterns.append(GridPattern(
                pattern_type="vertical_alignment",
                elements=centers,
                confidence=0.8,
                description="Objects are vertically aligned"
            ))
        
        # Check for diagonal alignment
        if len(objects) >= 3:
            sorted_centers = sorted(centers)
            if self._is_diagonal_alignment(sorted_centers):
                patterns.append(GridPattern(
                    pattern_type="diagonal_alignment",
                    elements=sorted_centers,
                    confidence=0.7,
                    description="Objects are diagonally aligned"
                ))
        
        return patterns
    
    def _is_diagonal_alignment(self, centers: List[Tuple[float, float]]) -> bool:
        """Check if centers form a diagonal line"""
        if len(centers) < 3:
            return False
        
        # Calculate slopes between consecutive points
        slopes = []
        for i in range(len(centers) - 1):
            dx = centers[i+1][1] - centers[i][1]
            dy = centers[i+1][0] - centers[i][0]
            if abs(dx) < 1e-6:  # Vertical line
                slope = float('inf')
            else:
                slope = dy / dx
            slopes.append(slope)
        
        # Check if all slopes are approximately equal
        if len(set(slopes)) <= 2:  # Allow for some variance
            return True
        
        return False
    
    def _detect_repetition_patterns(self, grid: np.ndarray, objects: List[GridObject]) -> List[GridPattern]:
        """Detect repetition patterns"""
        patterns = []
        
        # This is a simplified implementation
        # In a full implementation, you'd look for repeated subgrids
        
        return patterns
    
    def _detect_symmetries(self, grid: np.ndarray) -> List[str]:
        """Detect symmetries in the grid"""
        symmetries = []
        
        # Vertical symmetry (left-right mirror)
        if np.array_equal(grid, np.fliplr(grid)):
            symmetries.append("vertical_symmetry")
        
        # Horizontal symmetry (top-bottom mirror)
        if np.array_equal(grid, np.flipud(grid)):
            symmetries.append("horizontal_symmetry")
        
        # Rotational symmetry (90, 180, 270 degrees)
        rotated_90 = np.rot90(grid, 1)
        rotated_180 = np.rot90(grid, 2)
        rotated_270 = np.rot90(grid, 3)
        
        if np.array_equal(grid, rotated_90):
            symmetries.append("90_degree_rotational")
        if np.array_equal(grid, rotated_180):
            symmetries.append("180_degree_rotational")
        if np.array_equal(grid, rotated_270):
            symmetries.append("270_degree_rotational")
        
        # Diagonal symmetry (transpose)
        if grid.shape[0] == grid.shape[1] and np.array_equal(grid, grid.T):
            symmetries.append("diagonal_symmetry")
        
        return symmetries
    
    def _build_connectivity_graph(self, objects: List[GridObject]) -> Dict[int, Set[int]]:
        """Build graph of object connectivity"""
        connectivity = defaultdict(set)
        
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects):
                if i != j and (obj1.is_adjacent_to(obj2) or obj1.overlaps_with(obj2)):
                    connectivity[obj1.object_id].add(obj2.object_id)
                    connectivity[obj2.object_id].add(obj1.object_id)
        
        return dict(connectivity)