"""
Visual Feature Extraction for ARC-AGI Processing

Advanced feature extraction for comprehensive visual analysis:
- Shape feature analysis (geometry, topology, symmetry)
- Color and texture features
- Spatial relationship features
- Pattern-based features
- Multi-scale hierarchical features
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, Counter
from scipy import ndimage
from sklearn.cluster import DBSCAN

from .grid_processor import GridObject, BoundingBox, GridRepresentation, ObjectShape, SpatialRelation


class FeatureType(Enum):
    """Types of visual features"""
    # Shape features
    GEOMETRIC = "geometric"
    TOPOLOGICAL = "topological"
    SYMMETRY = "symmetry"
    
    # Color features
    COLOR_DISTRIBUTION = "color_distribution"
    COLOR_PATTERN = "color_pattern"
    
    # Spatial features
    POSITION = "position"
    RELATIONSHIP = "relationship"
    ARRANGEMENT = "arrangement"
    
    # Pattern features
    REPETITION = "repetition"
    SEQUENCE = "sequence"
    HIERARCHICAL = "hierarchical"
    
    # Global features
    STATISTICAL = "statistical"
    STRUCTURAL = "structural"


@dataclass
class ShapeFeatures:
    """Shape-based features for objects"""
    # Basic geometric properties
    area: int
    perimeter: int
    width: int
    height: int
    aspect_ratio: float
    
    # Shape complexity
    compactness: float  # 4π*area/perimeter²
    solidity: float     # area/convex_hull_area
    extent: float       # area/bounding_box_area
    
    # Shape descriptors
    shape_type: ObjectShape
    rectangularity: float
    circularity: float
    
    # Symmetry properties
    has_vertical_symmetry: bool
    has_horizontal_symmetry: bool
    has_rotational_symmetry: bool
    symmetry_count: int
    
    # Topological properties
    euler_number: int  # Number of connected components - holes
    connected_components: int
    holes_count: int
    
    def __post_init__(self):
        """Calculate derived features"""
        if self.perimeter > 0:
            self.compactness = (4 * np.pi * self.area) / (self.perimeter ** 2)
        else:
            self.compactness = 0.0


@dataclass
class ColorFeatures:
    """Color-based features"""
    # Basic color properties
    dominant_color: int
    color_count: int
    unique_colors: Set[int]
    color_distribution: Dict[int, int]
    
    # Color patterns
    color_entropy: float
    color_variance: float
    most_common_color: int
    least_common_color: int
    
    # Color relationships
    color_contrast: float
    color_harmony_score: float


@dataclass
class SpatialFeatures:
    """Spatial arrangement features"""
    # Position features
    center_x: float
    center_y: float
    relative_position: Tuple[float, float]  # Relative to grid center
    
    # Spatial relationships
    nearest_neighbors: List[int]  # Object IDs
    spatial_density: float
    clustering_coefficient: float
    
    # Grid-based features
    grid_quadrant: int  # 0-3 for quadrants
    border_distance: float
    central_tendency: float  # How close to center
    
    # Alignment features
    horizontal_alignment: float
    vertical_alignment: float
    diagonal_alignment: float


@dataclass
class PatternFeatures:
    """Pattern-based features"""
    # Repetition features
    is_part_of_pattern: bool
    pattern_type: Optional[str]
    pattern_period: Optional[int]
    pattern_confidence: float
    
    # Sequence features
    sequence_position: Optional[int]
    sequence_length: Optional[int]
    sequence_direction: Optional[str]
    
    # Hierarchical features
    nesting_level: int
    parent_object_id: Optional[int]
    child_object_ids: List[int]


@dataclass
class FeatureVector:
    """Complete feature vector for visual analysis"""
    object_id: Optional[int]  # None for global features
    feature_type: FeatureType
    
    # Feature categories
    shape_features: Optional[ShapeFeatures] = None
    color_features: Optional[ColorFeatures] = None
    spatial_features: Optional[SpatialFeatures] = None
    pattern_features: Optional[PatternFeatures] = None
    
    # Raw numerical features
    numerical_features: Dict[str, float] = field(default_factory=dict)
    categorical_features: Dict[str, str] = field(default_factory=dict)
    
    # Metadata
    extraction_confidence: float = 1.0
    feature_importance: float = 1.0
    
    def get_numerical_vector(self) -> np.ndarray:
        """Get numerical feature vector"""
        features = []
        
        if self.shape_features:
            features.extend([
                self.shape_features.area,
                self.shape_features.perimeter,
                self.shape_features.aspect_ratio,
                self.shape_features.compactness,
                self.shape_features.solidity,
                self.shape_features.extent,
                self.shape_features.rectangularity,
                self.shape_features.circularity,
                float(self.shape_features.has_vertical_symmetry),
                float(self.shape_features.has_horizontal_symmetry),
                self.shape_features.euler_number,
                self.shape_features.connected_components
            ])
        
        if self.spatial_features:
            features.extend([
                self.spatial_features.center_x,
                self.spatial_features.center_y,
                self.spatial_features.spatial_density,
                self.spatial_features.clustering_coefficient,
                self.spatial_features.border_distance,
                self.spatial_features.central_tendency,
                self.spatial_features.horizontal_alignment,
                self.spatial_features.vertical_alignment,
                self.spatial_features.diagonal_alignment
            ])
        
        if self.color_features:
            features.extend([
                self.color_features.dominant_color,
                self.color_features.color_count,
                self.color_features.color_entropy,
                self.color_features.color_variance,
                self.color_features.color_contrast
            ])
        
        # Add custom numerical features
        features.extend(list(self.numerical_features.values()))
        
        return np.array(features, dtype=np.float32)


class VisualFeatureExtractor:
    """Advanced visual feature extraction engine"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Feature extraction settings
        self.min_object_size = 1
        self.spatial_neighborhood_radius = 3.0
        
        # Feature normalization ranges
        self.feature_ranges = {
            'area': (1, 100),
            'aspect_ratio': (0.1, 10.0),
            'compactness': (0.0, 1.0),
            'spatial_density': (0.0, 1.0)
        }
    
    def extract_all_features(self, grid_repr: GridRepresentation) -> Dict[str, List[FeatureVector]]:
        """Extract all types of features from grid representation"""
        try:
            feature_vectors = {
                'object_features': [],
                'global_features': [],
                'relational_features': []
            }
            
            # Extract object-level features
            for obj in grid_repr.objects:
                obj_features = self.extract_object_features(obj, grid_repr)
                feature_vectors['object_features'].extend(obj_features)
            
            # Extract global features
            global_features = self.extract_global_features(grid_repr)
            feature_vectors['global_features'].extend(global_features)
            
            # Extract relational features
            relational_features = self.extract_relational_features(grid_repr)
            feature_vectors['relational_features'].extend(relational_features)
            
            return feature_vectors
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {e}")
            return {'object_features': [], 'global_features': [], 'relational_features': []}
    
    def extract_object_features(self, obj: GridObject, grid_repr: GridRepresentation) -> List[FeatureVector]:
        """Extract features for a single object"""
        feature_vectors = []
        
        try:
            # Shape features
            shape_features = self._extract_shape_features(obj, grid_repr)
            shape_vector = FeatureVector(
                object_id=obj.object_id,
                feature_type=FeatureType.GEOMETRIC,
                shape_features=shape_features,
                extraction_confidence=0.9
            )
            feature_vectors.append(shape_vector)
            
            # Color features
            color_features = self._extract_color_features(obj, grid_repr)
            color_vector = FeatureVector(
                object_id=obj.object_id,
                feature_type=FeatureType.COLOR_DISTRIBUTION,
                color_features=color_features,
                extraction_confidence=0.95
            )
            feature_vectors.append(color_vector)
            
            # Spatial features
            spatial_features = self._extract_spatial_features(obj, grid_repr)
            spatial_vector = FeatureVector(
                object_id=obj.object_id,
                feature_type=FeatureType.POSITION,
                spatial_features=spatial_features,
                extraction_confidence=0.9
            )
            feature_vectors.append(spatial_vector)
            
            # Pattern features
            pattern_features = self._extract_pattern_features(obj, grid_repr)
            pattern_vector = FeatureVector(
                object_id=obj.object_id,
                feature_type=FeatureType.REPETITION,
                pattern_features=pattern_features,
                extraction_confidence=0.7
            )
            feature_vectors.append(pattern_vector)
            
        except Exception as e:
            self.logger.error(f"Error extracting object features for object {obj.object_id}: {e}")
        
        return feature_vectors
    
    def extract_global_features(self, grid_repr: GridRepresentation) -> List[FeatureVector]:
        """Extract grid-level global features"""
        feature_vectors = []
        
        try:
            # Statistical features
            stats_features = self._extract_statistical_features(grid_repr)
            stats_vector = FeatureVector(
                object_id=None,
                feature_type=FeatureType.STATISTICAL,
                numerical_features=stats_features,
                extraction_confidence=0.95
            )
            feature_vectors.append(stats_vector)
            
            # Structural features
            structural_features = self._extract_structural_features(grid_repr)
            structural_vector = FeatureVector(
                object_id=None,
                feature_type=FeatureType.STRUCTURAL,
                numerical_features=structural_features,
                extraction_confidence=0.8
            )
            feature_vectors.append(structural_vector)
            
        except Exception as e:
            self.logger.error(f"Error extracting global features: {e}")
        
        return feature_vectors
    
    def extract_relational_features(self, grid_repr: GridRepresentation) -> List[FeatureVector]:
        """Extract features describing relationships between objects"""
        feature_vectors = []
        
        try:
            # Spatial relationship features
            for (obj1_id, obj2_id), relations in grid_repr.spatial_relations.items():
                relation_features = self._extract_relationship_features(
                    obj1_id, obj2_id, relations, grid_repr
                )
                
                relation_vector = FeatureVector(
                    object_id=None,  # Relational features don't belong to single object
                    feature_type=FeatureType.RELATIONSHIP,
                    numerical_features=relation_features,
                    categorical_features={
                        'object1_id': str(obj1_id),
                        'object2_id': str(obj2_id),
                        'primary_relation': relations[0].value if relations else 'none'
                    },
                    extraction_confidence=0.8
                )
                feature_vectors.append(relation_vector)
                
        except Exception as e:
            self.logger.error(f"Error extracting relational features: {e}")
        
        return feature_vectors
    
    def _extract_shape_features(self, obj: GridObject, grid_repr: GridRepresentation) -> ShapeFeatures:
        """Extract shape-based features for an object"""
        bbox = obj.bounding_box
        
        # Basic geometric properties
        area = obj.size
        perimeter = self._calculate_perimeter(obj.cells)
        width = bbox.width
        height = bbox.height
        aspect_ratio = width / height if height > 0 else 1.0
        
        # Shape complexity measures
        extent = area / bbox.area if bbox.area > 0 else 0.0
        
        # Rectangularity (how close to rectangle)
        rectangularity = area / bbox.area if bbox.area > 0 else 0.0
        
        # Circularity approximation
        circularity = self._calculate_circularity(obj.cells)
        
        # Symmetry detection
        vertical_sym, horizontal_sym, rotational_sym = self._detect_object_symmetries(obj)
        
        # Topological properties
        euler_number = self._calculate_euler_number(obj.cells)
        connected_components = 1  # Single object is one component
        holes_count = max(0, connected_components - euler_number)
        
        return ShapeFeatures(
            area=area,
            perimeter=perimeter,
            width=width,
            height=height,
            aspect_ratio=aspect_ratio,
            compactness=0.0,  # Will be calculated in __post_init__
            solidity=1.0,  # Simplified - could calculate convex hull
            extent=extent,
            shape_type=obj.shape,
            rectangularity=rectangularity,
            circularity=circularity,
            has_vertical_symmetry=vertical_sym,
            has_horizontal_symmetry=horizontal_sym,
            has_rotational_symmetry=rotational_sym,
            symmetry_count=sum([vertical_sym, horizontal_sym, rotational_sym]),
            euler_number=euler_number,
            connected_components=connected_components,
            holes_count=holes_count
        )
    
    def _extract_color_features(self, obj: GridObject, grid_repr: GridRepresentation) -> ColorFeatures:
        """Extract color-based features for an object"""
        # For single-colored objects, features are simple
        dominant_color = obj.color
        color_count = 1
        unique_colors = {obj.color}
        color_distribution = {obj.color: obj.size}
        
        # Color entropy (0 for single color)
        color_entropy = 0.0
        color_variance = 0.0
        
        # Color contrast with background
        background_color = grid_repr.background_color
        color_contrast = abs(dominant_color - background_color) / 9.0  # Normalized assuming colors 0-9
        
        return ColorFeatures(
            dominant_color=dominant_color,
            color_count=color_count,
            unique_colors=unique_colors,
            color_distribution=color_distribution,
            color_entropy=color_entropy,
            color_variance=color_variance,
            most_common_color=dominant_color,
            least_common_color=dominant_color,
            color_contrast=color_contrast,
            color_harmony_score=1.0  # Perfect harmony for single color
        )
    
    def _extract_spatial_features(self, obj: GridObject, grid_repr: GridRepresentation) -> SpatialFeatures:
        """Extract spatial features for an object"""
        center = obj.bounding_box.center
        grid_center = (grid_repr.height / 2, grid_repr.width / 2)
        
        # Position features
        center_x, center_y = center[1], center[0]  # Convert to x,y coordinates
        relative_x = (center_x - grid_center[1]) / grid_repr.width
        relative_y = (center_y - grid_center[0]) / grid_repr.height
        
        # Spatial relationships
        nearest_neighbors = self._find_nearest_neighbors(obj, grid_repr.objects)
        spatial_density = self._calculate_spatial_density(obj, grid_repr.objects)
        
        # Grid position features
        grid_quadrant = self._determine_quadrant(center, grid_center)
        border_distance = min(
            center_x, center_y,
            grid_repr.width - center_x,
            grid_repr.height - center_y
        ) / max(grid_repr.width, grid_repr.height)
        
        central_tendency = 1.0 - np.sqrt(relative_x**2 + relative_y**2) / np.sqrt(2)
        
        # Alignment features
        h_align, v_align, d_align = self._calculate_alignment_features(obj, grid_repr.objects)
        
        return SpatialFeatures(
            center_x=center_x,
            center_y=center_y,
            relative_position=(relative_x, relative_y),
            nearest_neighbors=nearest_neighbors,
            spatial_density=spatial_density,
            clustering_coefficient=0.5,  # Simplified
            grid_quadrant=grid_quadrant,
            border_distance=border_distance,
            central_tendency=central_tendency,
            horizontal_alignment=h_align,
            vertical_alignment=v_align,
            diagonal_alignment=d_align
        )
    
    def _extract_pattern_features(self, obj: GridObject, grid_repr: GridRepresentation) -> PatternFeatures:
        """Extract pattern-based features for an object"""
        # Analyze if object is part of detected patterns
        is_pattern_part = False
        pattern_type = None
        pattern_confidence = 0.0
        
        for pattern in grid_repr.detected_patterns:
            if any(obj.object_id in str(element) for element in pattern.elements):
                is_pattern_part = True
                pattern_type = pattern.pattern_type
                pattern_confidence = pattern.confidence
                break
        
        # Hierarchical features (nesting)
        nesting_level = 0
        parent_id = None
        child_ids = []
        
        for other_obj in grid_repr.objects:
            if other_obj.object_id != obj.object_id:
                if self._is_nested_in(obj, other_obj):
                    nesting_level = 1
                    parent_id = other_obj.object_id
                elif self._is_nested_in(other_obj, obj):
                    child_ids.append(other_obj.object_id)
        
        return PatternFeatures(
            is_part_of_pattern=is_pattern_part,
            pattern_type=pattern_type,
            pattern_period=None,  # Could be enhanced
            pattern_confidence=pattern_confidence,
            sequence_position=None,  # Could be enhanced
            sequence_length=None,
            sequence_direction=None,
            nesting_level=nesting_level,
            parent_object_id=parent_id,
            child_object_ids=child_ids
        )
    
    def _extract_statistical_features(self, grid_repr: GridRepresentation) -> Dict[str, float]:
        """Extract statistical features from the grid"""
        features = {}
        
        # Basic statistics
        features['total_objects'] = len(grid_repr.objects)
        features['grid_density'] = len(grid_repr.objects) / (grid_repr.width * grid_repr.height)
        features['background_ratio'] = grid_repr.color_distribution.get(grid_repr.background_color, 0) / (grid_repr.width * grid_repr.height)
        
        # Color statistics
        features['unique_colors'] = len(grid_repr.colors)
        features['color_entropy'] = self._calculate_color_entropy(grid_repr.color_distribution)
        
        # Size statistics
        if grid_repr.objects:
            sizes = [obj.size for obj in grid_repr.objects]
            features['mean_object_size'] = np.mean(sizes)
            features['std_object_size'] = np.std(sizes)
            features['max_object_size'] = np.max(sizes)
            features['min_object_size'] = np.min(sizes)
            features['size_range'] = features['max_object_size'] - features['min_object_size']
        else:
            features.update({
                'mean_object_size': 0,
                'std_object_size': 0,
                'max_object_size': 0,
                'min_object_size': 0,
                'size_range': 0
            })
        
        # Symmetry statistics
        features['has_global_vertical_symmetry'] = float('vertical_symmetry' in grid_repr.symmetries)
        features['has_global_horizontal_symmetry'] = float('horizontal_symmetry' in grid_repr.symmetries)
        features['symmetry_count'] = len(grid_repr.symmetries)
        
        return features
    
    def _extract_structural_features(self, grid_repr: GridRepresentation) -> Dict[str, float]:
        """Extract structural features from the grid"""
        features = {}
        
        # Connectivity features
        if grid_repr.objects:
            # Average connectivity
            connectivities = [len(neighbors) for neighbors in grid_repr.connectivity_graph.values()]
            features['mean_connectivity'] = np.mean(connectivities) if connectivities else 0
            features['max_connectivity'] = np.max(connectivities) if connectivities else 0
            features['connectivity_variance'] = np.var(connectivities) if connectivities else 0
            
            # Clustering coefficient
            features['global_clustering_coefficient'] = self._calculate_global_clustering(grid_repr.connectivity_graph)
        else:
            features.update({
                'mean_connectivity': 0,
                'max_connectivity': 0,
                'connectivity_variance': 0,
                'global_clustering_coefficient': 0
            })
        
        # Pattern complexity
        features['pattern_count'] = len(grid_repr.detected_patterns)
        features['pattern_density'] = len(grid_repr.detected_patterns) / max(1, len(grid_repr.objects))
        
        # Spatial distribution
        if grid_repr.objects:
            centers = [obj.bounding_box.center for obj in grid_repr.objects]
            features['spatial_dispersion'] = self._calculate_spatial_dispersion(centers)
            features['spatial_clustering'] = self._calculate_spatial_clustering_score(centers)
        else:
            features['spatial_dispersion'] = 0
            features['spatial_clustering'] = 0
        
        return features
    
    def _extract_relationship_features(self, obj1_id: int, obj2_id: int, 
                                     relations: List[SpatialRelation],
                                     grid_repr: GridRepresentation) -> Dict[str, float]:
        """Extract features describing relationship between two objects"""
        features = {}
        
        obj1 = grid_repr.object_by_id.get(obj1_id)
        obj2 = grid_repr.object_by_id.get(obj2_id)
        
        if not obj1 or not obj2:
            return features
        
        # Distance features
        center1 = obj1.bounding_box.center
        center2 = obj2.bounding_box.center
        distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        
        features['euclidean_distance'] = distance
        features['manhattan_distance'] = abs(center1[0] - center2[0]) + abs(center1[1] - center2[1])
        features['normalized_distance'] = distance / max(grid_repr.width, grid_repr.height)
        
        # Size relationship
        size_ratio = obj1.size / obj2.size if obj2.size > 0 else 1.0
        features['size_ratio'] = size_ratio
        features['size_difference'] = abs(obj1.size - obj2.size)
        
        # Color relationship
        features['same_color'] = float(obj1.color == obj2.color)
        features['color_difference'] = abs(obj1.color - obj2.color)
        
        # Shape relationship
        features['same_shape'] = float(obj1.shape == obj2.shape)
        
        # Relationship type encoding
        for relation in SpatialRelation:
            features[f'relation_{relation.value}'] = float(relation in relations)
        
        return features
    
    # Helper methods
    
    def _calculate_perimeter(self, cells: Set[Tuple[int, int]]) -> int:
        """Calculate perimeter of object"""
        perimeter = 0
        for row, col in cells:
            # Count edges that are on the boundary
            neighbors = [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]
            for nr, nc in neighbors:
                if (nr, nc) not in cells:
                    perimeter += 1
        return perimeter
    
    def _calculate_circularity(self, cells: Set[Tuple[int, int]]) -> float:
        """Calculate how circular the object is"""
        if not cells:
            return 0.0
        
        # Find centroid
        rows = [r for r, c in cells]
        cols = [c for r, c in cells]
        centroid_r = np.mean(rows)
        centroid_c = np.mean(cols)
        
        # Calculate distances from centroid
        distances = [np.sqrt((r - centroid_r)**2 + (c - centroid_c)**2) for r, c in cells]
        
        # Circularity is inverse of variance of distances
        if len(distances) > 1:
            distance_var = np.var(distances)
            return 1.0 / (1.0 + distance_var)
        else:
            return 1.0
    
    def _detect_object_symmetries(self, obj: GridObject) -> Tuple[bool, bool, bool]:
        """Detect symmetries in object shape"""
        # Convert cells to local coordinate system
        min_row = obj.bounding_box.min_row
        min_col = obj.bounding_box.min_col
        
        local_cells = {(r - min_row, c - min_col) for r, c in obj.cells}
        
        # Check vertical symmetry
        max_col = max(c for r, c in local_cells) if local_cells else 0
        vertical_sym = all((r, max_col - c) in local_cells for r, c in local_cells)
        
        # Check horizontal symmetry  
        max_row = max(r for r, c in local_cells) if local_cells else 0
        horizontal_sym = all((max_row - r, c) in local_cells for r, c in local_cells)
        
        # Check rotational symmetry (180 degrees)
        rotational_sym = all((max_row - r, max_col - c) in local_cells for r, c in local_cells)
        
        return vertical_sym, horizontal_sym, rotational_sym
    
    def _calculate_euler_number(self, cells: Set[Tuple[int, int]]) -> int:
        """Calculate Euler number (topological invariant)"""
        # Simplified: assume single connected component with no holes
        # Euler number = 1 for simply connected region
        return 1
    
    def _find_nearest_neighbors(self, obj: GridObject, all_objects: List[GridObject], k: int = 3) -> List[int]:
        """Find k nearest neighbor objects"""
        center = obj.bounding_box.center
        distances = []
        
        for other_obj in all_objects:
            if other_obj.object_id != obj.object_id:
                other_center = other_obj.bounding_box.center
                distance = np.sqrt((center[0] - other_center[0])**2 + (center[1] - other_center[1])**2)
                distances.append((distance, other_obj.object_id))
        
        # Sort by distance and return k nearest
        distances.sort()
        return [obj_id for _, obj_id in distances[:k]]
    
    def _calculate_spatial_density(self, obj: GridObject, all_objects: List[GridObject]) -> float:
        """Calculate local spatial density around object"""
        center = obj.bounding_box.center
        nearby_count = 0
        
        for other_obj in all_objects:
            if other_obj.object_id != obj.object_id:
                other_center = other_obj.bounding_box.center
                distance = np.sqrt((center[0] - other_center[0])**2 + (center[1] - other_center[1])**2)
                
                if distance <= self.spatial_neighborhood_radius:
                    nearby_count += 1
        
        # Normalize by area of neighborhood
        neighborhood_area = np.pi * self.spatial_neighborhood_radius ** 2
        return nearby_count / neighborhood_area
    
    def _determine_quadrant(self, center: Tuple[float, float], grid_center: Tuple[float, float]) -> int:
        """Determine which quadrant the center point is in"""
        if center[0] <= grid_center[0] and center[1] <= grid_center[1]:
            return 0  # Top-left
        elif center[0] <= grid_center[0] and center[1] > grid_center[1]:
            return 1  # Top-right
        elif center[0] > grid_center[0] and center[1] <= grid_center[1]:
            return 2  # Bottom-left
        else:
            return 3  # Bottom-right
    
    def _calculate_alignment_features(self, obj: GridObject, all_objects: List[GridObject]) -> Tuple[float, float, float]:
        """Calculate alignment features with other objects"""
        if not all_objects:
            return 0.0, 0.0, 0.0
        
        center = obj.bounding_box.center
        h_alignments = []
        v_alignments = []
        d_alignments = []
        
        for other_obj in all_objects:
            if other_obj.object_id != obj.object_id:
                other_center = other_obj.bounding_box.center
                
                # Horizontal alignment (same row)
                h_alignments.append(abs(center[0] - other_center[0]))
                
                # Vertical alignment (same column)
                v_alignments.append(abs(center[1] - other_center[1]))
                
                # Diagonal alignment
                d_alignments.append(abs(abs(center[0] - other_center[0]) - abs(center[1] - other_center[1])))
        
        # Convert to alignment scores (inverse of distance)
        h_align = 1.0 / (1.0 + min(h_alignments)) if h_alignments else 0.0
        v_align = 1.0 / (1.0 + min(v_alignments)) if v_alignments else 0.0
        d_align = 1.0 / (1.0 + min(d_alignments)) if d_alignments else 0.0
        
        return h_align, v_align, d_align
    
    def _is_nested_in(self, inner_obj: GridObject, outer_obj: GridObject) -> bool:
        """Check if inner object is nested within outer object"""
        inner_bbox = inner_obj.bounding_box
        outer_bbox = outer_obj.bounding_box
        
        return (inner_bbox.min_row >= outer_bbox.min_row and
                inner_bbox.max_row <= outer_bbox.max_row and
                inner_bbox.min_col >= outer_bbox.min_col and
                inner_bbox.max_col <= outer_bbox.max_col and
                inner_obj.size < outer_obj.size)
    
    def _calculate_color_entropy(self, color_distribution: Dict[int, int]) -> float:
        """Calculate color entropy"""
        if not color_distribution:
            return 0.0
        
        total_pixels = sum(color_distribution.values())
        entropy = 0.0
        
        for count in color_distribution.values():
            if count > 0:
                p = count / total_pixels
                entropy -= p * np.log2(p)
        
        return entropy
    
    def _calculate_global_clustering(self, connectivity_graph: Dict[int, Set[int]]) -> float:
        """Calculate global clustering coefficient"""
        if not connectivity_graph:
            return 0.0
        
        clustering_coeffs = []
        
        for node, neighbors in connectivity_graph.items():
            if len(neighbors) < 2:
                clustering_coeffs.append(0.0)
                continue
            
            # Count connections between neighbors
            neighbor_connections = 0
            for n1 in neighbors:
                for n2 in neighbors:
                    if n1 != n2 and n2 in connectivity_graph.get(n1, set()):
                        neighbor_connections += 1
            
            # Clustering coefficient for this node
            possible_connections = len(neighbors) * (len(neighbors) - 1)
            if possible_connections > 0:
                clustering_coeffs.append(neighbor_connections / possible_connections)
            else:
                clustering_coeffs.append(0.0)
        
        return np.mean(clustering_coeffs) if clustering_coeffs else 0.0
    
    def _calculate_spatial_dispersion(self, centers: List[Tuple[float, float]]) -> float:
        """Calculate spatial dispersion of object centers"""
        if len(centers) < 2:
            return 0.0
        
        # Calculate mean center
        mean_center = (np.mean([c[0] for c in centers]), np.mean([c[1] for c in centers]))
        
        # Calculate average distance from mean center
        distances = [np.sqrt((c[0] - mean_center[0])**2 + (c[1] - mean_center[1])**2) for c in centers]
        
        return np.mean(distances)
    
    def _calculate_spatial_clustering_score(self, centers: List[Tuple[float, float]]) -> float:
        """Calculate spatial clustering score using DBSCAN-inspired method"""
        if len(centers) < 3:
            return 0.0
        
        # Calculate pairwise distances
        distances = []
        for i, c1 in enumerate(centers):
            for c2 in centers[i+1:]:
                distance = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
                distances.append(distance)
        
        # Use median distance as clustering threshold
        threshold = np.median(distances)
        
        # Count points within threshold of each point
        cluster_sizes = []
        for center in centers:
            nearby_count = sum(1 for other_center in centers 
                             if np.sqrt((center[0] - other_center[0])**2 + (center[1] - other_center[1])**2) <= threshold)
            cluster_sizes.append(nearby_count)
        
        # Normalize clustering score
        max_possible_cluster_size = len(centers)
        avg_cluster_size = np.mean(cluster_sizes)
        
        return avg_cluster_size / max_possible_cluster_size
    
    def normalize_features(self, feature_vector: FeatureVector) -> FeatureVector:
        """Normalize features to standard ranges"""
        normalized_vector = feature_vector
        
        # Normalize numerical features
        for feature_name, value in feature_vector.numerical_features.items():
            if feature_name in self.feature_ranges:
                min_val, max_val = self.feature_ranges[feature_name]
                normalized_value = (value - min_val) / (max_val - min_val) if max_val > min_val else 0.0
                normalized_value = np.clip(normalized_value, 0.0, 1.0)
                normalized_vector.numerical_features[feature_name] = normalized_value
        
        return normalized_vector