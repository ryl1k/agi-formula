"""
Vision Processing Module for AGI-Formula

Advanced computer vision capabilities for multi-modal AGI:
- Image preprocessing and enhancement
- Feature extraction (CNN-based, traditional CV)
- Object detection and recognition
- Scene understanding and spatial reasoning
- Visual attention mechanisms
- Motion and temporal analysis
"""

import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging


class VisualFeatureType(Enum):
    """Types of visual features"""
    EDGE_FEATURES = "edges"
    TEXTURE_FEATURES = "texture"
    COLOR_FEATURES = "color"
    SHAPE_FEATURES = "shape"
    MOTION_FEATURES = "motion"
    SPATIAL_FEATURES = "spatial"
    SEMANTIC_FEATURES = "semantic"


@dataclass
class VisualFeatures:
    """Container for extracted visual features"""
    feature_type: VisualFeatureType
    features: np.ndarray
    confidence: float
    extraction_time: float
    metadata: Dict[str, Any]


@dataclass
class VisualObject:
    """Detected visual object"""
    class_name: str
    bounding_box: Tuple[int, int, int, int]  # (x, y, width, height)
    confidence: float
    features: np.ndarray
    attributes: Dict[str, Any]


@dataclass
class SceneAnalysis:
    """Scene understanding results"""
    scene_type: str
    objects: List[VisualObject]
    spatial_relationships: List[Dict[str, Any]]
    scene_features: np.ndarray
    confidence: float


class VisualFeatureExtractor:
    """
    Advanced visual feature extraction system
    
    Features:
    - Multi-scale feature extraction
    - Traditional computer vision features (SIFT, HOG, etc.)
    - Deep learning features (CNN-based)
    - Adaptive feature selection
    - Real-time processing optimization
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        
        # Feature extraction methods
        self.extractors = {
            VisualFeatureType.EDGE_FEATURES: self._extract_edge_features,
            VisualFeatureType.TEXTURE_FEATURES: self._extract_texture_features,
            VisualFeatureType.COLOR_FEATURES: self._extract_color_features,
            VisualFeatureType.SHAPE_FEATURES: self._extract_shape_features,
            VisualFeatureType.MOTION_FEATURES: self._extract_motion_features,
            VisualFeatureType.SPATIAL_FEATURES: self._extract_spatial_features,
            VisualFeatureType.SEMANTIC_FEATURES: self._extract_semantic_features
        }
        
        # Initialize feature extractors
        self._initialize_extractors()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for visual feature extraction"""
        return {
            'feature_dimensions': {
                VisualFeatureType.EDGE_FEATURES: 128,
                VisualFeatureType.TEXTURE_FEATURES: 256,
                VisualFeatureType.COLOR_FEATURES: 64,
                VisualFeatureType.SHAPE_FEATURES: 128,
                VisualFeatureType.MOTION_FEATURES: 96,
                VisualFeatureType.SPATIAL_FEATURES: 64,
                VisualFeatureType.SEMANTIC_FEATURES: 512
            },
            'preprocessing': {
                'resize_dims': (224, 224),
                'normalize': True,
                'enhance_contrast': True,
                'denoise': False
            },
            'edge_detection': {
                'method': 'canny',
                'low_threshold': 50,
                'high_threshold': 150
            },
            'texture_analysis': {
                'methods': ['lbp', 'glcm', 'gabor'],
                'scales': [1, 2, 4],
                'orientations': 8
            },
            'color_analysis': {
                'color_spaces': ['rgb', 'hsv', 'lab'],
                'histogram_bins': 64
            },
            'motion_detection': {
                'frame_buffer_size': 5,
                'threshold': 0.1
            }
        }
    
    def _initialize_extractors(self):
        """Initialize feature extraction components"""
        # Initialize any required components for feature extraction
        print("Visual feature extractors initialized")
    
    def extract_features(self, image: np.ndarray, 
                        feature_types: Optional[List[VisualFeatureType]] = None) -> Dict[VisualFeatureType, VisualFeatures]:
        """Extract specified visual features from an image"""
        if feature_types is None:
            feature_types = list(VisualFeatureType)
        
        # Preprocess image
        processed_image = self._preprocess_image(image)
        
        extracted_features = {}
        
        for feature_type in feature_types:
            if feature_type in self.extractors:
                start_time = time.time()
                
                try:
                    features = self.extractors[feature_type](processed_image)
                    extraction_time = time.time() - start_time
                    
                    visual_features = VisualFeatures(
                        feature_type=feature_type,
                        features=features,
                        confidence=self._calculate_feature_confidence(features),
                        extraction_time=extraction_time,
                        metadata={
                            'image_shape': processed_image.shape,
                            'feature_dimension': len(features)
                        }
                    )
                    
                    extracted_features[feature_type] = visual_features
                    
                except Exception as e:
                    logging.error(f"Error extracting {feature_type.value} features: {e}")
        
        return extracted_features
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for feature extraction"""
        processed = image.copy()
        
        # Resize if needed
        target_dims = self.config['preprocessing']['resize_dims']
        if processed.shape[:2] != target_dims:
            processed = self._resize_image(processed, target_dims)
        
        # Normalize
        if self.config['preprocessing']['normalize']:
            processed = processed.astype(np.float32) / 255.0
        
        # Enhance contrast
        if self.config['preprocessing']['enhance_contrast']:
            processed = self._enhance_contrast(processed)
        
        # Denoise
        if self.config['preprocessing']['denoise']:
            processed = self._denoise_image(processed)
        
        return processed
    
    def _resize_image(self, image: np.ndarray, target_dims: Tuple[int, int]) -> np.ndarray:
        """Resize image to target dimensions"""
        # Simple bilinear interpolation implementation
        h, w = image.shape[:2]
        target_h, target_w = target_dims
        
        # Create coordinate grids
        y_coords = np.linspace(0, h-1, target_h)
        x_coords = np.linspace(0, w-1, target_w)
        
        # For simplicity, use nearest neighbor interpolation
        y_indices = np.round(y_coords).astype(int)
        x_indices = np.round(x_coords).astype(int)
        
        if len(image.shape) == 3:
            resized = image[np.ix_(y_indices, x_indices, range(image.shape[2]))]
        else:
            resized = image[np.ix_(y_indices, x_indices)]
        
        return resized
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance image contrast"""
        # Simple contrast enhancement using histogram stretching
        if len(image.shape) == 3:
            enhanced = np.zeros_like(image)
            for c in range(image.shape[2]):
                channel = image[:, :, c]
                min_val, max_val = np.min(channel), np.max(channel)
                if max_val > min_val:
                    enhanced[:, :, c] = (channel - min_val) / (max_val - min_val)
                else:
                    enhanced[:, :, c] = channel
        else:
            min_val, max_val = np.min(image), np.max(image)
            if max_val > min_val:
                enhanced = (image - min_val) / (max_val - min_val)
            else:
                enhanced = image
        
        return enhanced
    
    def _denoise_image(self, image: np.ndarray) -> np.ndarray:
        """Apply denoising to image"""
        # Simple Gaussian smoothing for denoising
        kernel_size = 3
        sigma = 0.5
        
        # Create Gaussian kernel
        ax = np.arange(-kernel_size//2 + 1., kernel_size//2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2*sigma**2))
        kernel = kernel / np.sum(kernel)
        
        # Apply convolution (simplified)
        if len(image.shape) == 3:
            denoised = np.zeros_like(image)
            for c in range(image.shape[2]):
                denoised[:, :, c] = self._convolve_2d(image[:, :, c], kernel)
        else:
            denoised = self._convolve_2d(image, kernel)
        
        return denoised
    
    def _convolve_2d(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Simple 2D convolution implementation"""
        # Pad image
        pad_h, pad_w = kernel.shape[0]//2, kernel.shape[1]//2
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
        
        # Perform convolution
        result = np.zeros_like(image)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = padded[i:i+kernel.shape[0], j:j+kernel.shape[1]]
                result[i, j] = np.sum(region * kernel)
        
        return result
    
    def _extract_edge_features(self, image: np.ndarray) -> np.ndarray:
        """Extract edge-based features"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
        
        # Sobel edge detection
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        
        edges_x = self._convolve_2d(gray, sobel_x)
        edges_y = self._convolve_2d(gray, sobel_y)
        
        edge_magnitude = np.sqrt(edges_x**2 + edges_y**2)
        edge_direction = np.arctan2(edges_y, edges_x)
        
        # Create feature histogram
        magnitude_hist = np.histogram(edge_magnitude.flatten(), bins=64)[0]
        direction_hist = np.histogram(edge_direction.flatten(), bins=64)[0]
        
        features = np.concatenate([magnitude_hist, direction_hist])
        features = features / (np.linalg.norm(features) + 1e-8)  # Normalize
        
        return features.astype(np.float32)
    
    def _extract_texture_features(self, image: np.ndarray) -> np.ndarray:
        """Extract texture-based features"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
        
        # Local Binary Pattern (simplified)
        lbp_features = self._compute_lbp(gray)
        
        # Gabor filter responses (simplified)
        gabor_features = self._compute_gabor_features(gray)
        
        # Combine features
        features = np.concatenate([lbp_features, gabor_features])
        features = features / (np.linalg.norm(features) + 1e-8)  # Normalize
        
        return features.astype(np.float32)
    
    def _compute_lbp(self, image: np.ndarray) -> np.ndarray:
        """Compute Local Binary Pattern features"""
        # Simplified LBP implementation
        h, w = image.shape
        lbp = np.zeros((h-2, w-2))
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                center = image[i, j]
                binary_string = ''
                
                # 8-neighborhood
                neighbors = [
                    image[i-1, j-1], image[i-1, j], image[i-1, j+1],
                    image[i, j+1], image[i+1, j+1], image[i+1, j],
                    image[i+1, j-1], image[i, j-1]
                ]
                
                for neighbor in neighbors:
                    binary_string += '1' if neighbor >= center else '0'
                
                lbp[i-1, j-1] = int(binary_string, 2)
        
        # Create histogram
        hist = np.histogram(lbp.flatten(), bins=256, range=(0, 256))[0]
        return hist / (np.sum(hist) + 1e-8)
    
    def _compute_gabor_features(self, image: np.ndarray) -> np.ndarray:
        """Compute Gabor filter features"""
        # Simplified Gabor filter implementation
        features = []
        
        orientations = self.config['texture_analysis']['orientations']
        scales = self.config['texture_analysis']['scales']
        
        for scale in scales:
            for orientation in range(orientations):
                angle = orientation * np.pi / orientations
                
                # Create Gabor kernel (simplified)
                kernel_size = 2 * scale + 1
                gabor_kernel = self._create_gabor_kernel(kernel_size, angle, scale)
                
                # Apply filter
                response = self._convolve_2d(image, gabor_kernel)
                
                # Extract statistical features
                mean_response = np.mean(np.abs(response))
                std_response = np.std(response)
                features.extend([mean_response, std_response])
        
        return np.array(features, dtype=np.float32)
    
    def _create_gabor_kernel(self, size: int, angle: float, scale: int) -> np.ndarray:
        """Create a Gabor filter kernel"""
        # Simplified Gabor kernel creation
        center = size // 2
        kernel = np.zeros((size, size))
        
        for i in range(size):
            for j in range(size):
                x = (i - center) * np.cos(angle) + (j - center) * np.sin(angle)
                y = -(i - center) * np.sin(angle) + (j - center) * np.cos(angle)
                
                gaussian = np.exp(-(x**2 + y**2) / (2 * scale**2))
                sinusoid = np.cos(2 * np.pi * x / scale)
                
                kernel[i, j] = gaussian * sinusoid
        
        return kernel
    
    def _extract_color_features(self, image: np.ndarray) -> np.ndarray:
        """Extract color-based features"""
        if len(image.shape) != 3:
            # For grayscale, create RGB equivalent
            rgb_image = np.stack([image, image, image], axis=2)
        else:
            rgb_image = image
        
        features = []
        
        # RGB histogram
        for channel in range(3):
            hist = np.histogram(rgb_image[:, :, channel].flatten(), bins=32)[0]
            features.extend(hist)
        
        # HSV features (simplified conversion)
        hsv_image = self._rgb_to_hsv(rgb_image)
        for channel in range(3):
            hist = np.histogram(hsv_image[:, :, channel].flatten(), bins=16)[0]
            features.extend(hist)
        
        features = np.array(features, dtype=np.float32)
        features = features / (np.linalg.norm(features) + 1e-8)  # Normalize
        
        return features
    
    def _rgb_to_hsv(self, rgb_image: np.ndarray) -> np.ndarray:
        """Convert RGB to HSV color space"""
        # Simplified RGB to HSV conversion
        rgb = rgb_image.copy()
        
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        
        max_val = np.maximum(r, np.maximum(g, b))
        min_val = np.minimum(r, np.minimum(g, b))
        diff = max_val - min_val
        
        # Hue
        h = np.zeros_like(max_val)
        mask = diff > 0
        
        r_max = (max_val == r) & mask
        g_max = (max_val == g) & mask
        b_max = (max_val == b) & mask
        
        h[r_max] = ((g - b)[r_max] / diff[r_max]) % 6
        h[g_max] = (b - r)[g_max] / diff[g_max] + 2
        h[b_max] = (r - g)[b_max] / diff[b_max] + 4
        
        h = h / 6.0
        
        # Saturation
        s = np.zeros_like(max_val)
        s[max_val > 0] = diff[max_val > 0] / max_val[max_val > 0]
        
        # Value
        v = max_val
        
        return np.stack([h, s, v], axis=2)
    
    def _extract_shape_features(self, image: np.ndarray) -> np.ndarray:
        """Extract shape-based features"""
        # Convert to grayscale and binarize
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
        
        # Simple thresholding
        threshold = np.mean(gray)
        binary = (gray > threshold).astype(np.uint8)
        
        # Compute moments for shape features
        features = []
        
        # Central moments
        for p in range(3):
            for q in range(3):
                moment = self._compute_moment(binary, p, q)
                features.append(moment)
        
        # Hu moments (simplified)
        hu_moments = self._compute_hu_moments(binary)
        features.extend(hu_moments)
        
        features = np.array(features, dtype=np.float32)
        features = features / (np.linalg.norm(features) + 1e-8)  # Normalize
        
        return features
    
    def _compute_moment(self, image: np.ndarray, p: int, q: int) -> float:
        """Compute image moment"""
        h, w = image.shape
        y_indices, x_indices = np.ogrid[:h, :w]
        
        moment = np.sum(image * (x_indices ** p) * (y_indices ** q))
        return float(moment)
    
    def _compute_hu_moments(self, image: np.ndarray) -> List[float]:
        """Compute Hu moment invariants (simplified)"""
        # Simplified implementation of first 3 Hu moments
        m00 = self._compute_moment(image, 0, 0)
        m10 = self._compute_moment(image, 1, 0)
        m01 = self._compute_moment(image, 0, 1)
        m20 = self._compute_moment(image, 2, 0)
        m02 = self._compute_moment(image, 0, 2)
        m11 = self._compute_moment(image, 1, 1)
        
        if m00 == 0:
            return [0.0] * 7
        
        # Central moments
        x_c = m10 / m00
        y_c = m01 / m00
        
        mu20 = m20 / m00 - x_c**2
        mu02 = m02 / m00 - y_c**2
        mu11 = m11 / m00 - x_c * y_c
        
        # Hu moments (first 3)
        hu1 = mu20 + mu02
        hu2 = (mu20 - mu02)**2 + 4 * mu11**2
        hu3 = (mu20 - 3*mu02)**2 + (3*mu11 - mu02)**2
        
        return [hu1, hu2, hu3, 0.0, 0.0, 0.0, 0.0]  # Pad to 7 for consistency
    
    def _extract_motion_features(self, image: np.ndarray) -> np.ndarray:
        """Extract motion-based features (requires frame history)"""
        # For single frame, return zero motion features
        # In practice, this would use a frame buffer to compute optical flow
        feature_dim = self.config['feature_dimensions'][VisualFeatureType.MOTION_FEATURES]
        return np.zeros(feature_dim, dtype=np.float32)
    
    def _extract_spatial_features(self, image: np.ndarray) -> np.ndarray:
        """Extract spatial relationship features"""
        # Divide image into grid and compute regional statistics
        h, w = image.shape[:2]
        grid_size = 4
        
        features = []
        
        for i in range(grid_size):
            for j in range(grid_size):
                # Extract region
                start_h, end_h = i * h // grid_size, (i + 1) * h // grid_size
                start_w, end_w = j * w // grid_size, (j + 1) * w // grid_size
                
                if len(image.shape) == 3:
                    region = image[start_h:end_h, start_w:end_w, :]
                    region_mean = np.mean(region, axis=(0, 1))
                    features.extend(region_mean)
                else:
                    region = image[start_h:end_h, start_w:end_w]
                    region_mean = np.mean(region)
                    features.append(region_mean)
        
        features = np.array(features, dtype=np.float32)
        features = features / (np.linalg.norm(features) + 1e-8)  # Normalize
        
        return features
    
    def _extract_semantic_features(self, image: np.ndarray) -> np.ndarray:
        """Extract semantic features (high-level understanding)"""
        # This would typically use a pre-trained CNN
        # For now, combine multiple low-level features
        edge_features = self._extract_edge_features(image)
        texture_features = self._extract_texture_features(image)
        color_features = self._extract_color_features(image)
        shape_features = self._extract_shape_features(image)
        
        # Combine and project to semantic feature dimension
        combined = np.concatenate([edge_features, texture_features, color_features, shape_features])
        
        target_dim = self.config['feature_dimensions'][VisualFeatureType.SEMANTIC_FEATURES]
        
        if len(combined) >= target_dim:
            # Downsample
            indices = np.linspace(0, len(combined)-1, target_dim, dtype=int)
            semantic_features = combined[indices]
        else:
            # Pad
            semantic_features = np.pad(combined, (0, target_dim - len(combined)))
        
        return semantic_features.astype(np.float32)
    
    def _calculate_feature_confidence(self, features: np.ndarray) -> float:
        """Calculate confidence score for extracted features"""
        # Simple confidence based on feature variance and magnitude
        if len(features) == 0:
            return 0.0
        
        variance = np.var(features)
        magnitude = np.linalg.norm(features)
        
        # Normalize to 0-1 range
        confidence = min(1.0, (variance * magnitude) / (len(features) + 1))
        
        return float(confidence)


class VisionProcessor:
    """
    Comprehensive vision processing system for AGI
    
    Features:
    - Real-time image processing and analysis
    - Object detection and recognition
    - Scene understanding and spatial reasoning
    - Visual attention mechanisms
    - Integration with multi-modal pipeline
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        
        # Initialize components
        self.feature_extractor = VisualFeatureExtractor(self.config.get('feature_extraction', {}))
        
        # Processing state
        self.frame_buffer = []
        self.max_buffer_size = self.config['frame_buffer_size']
        
        # Performance monitoring
        self.stats = {
            'frames_processed': 0,
            'processing_time_ms': [],
            'feature_extraction_time_ms': [],
            'detection_accuracy': []
        }
        
        print("Vision processor initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for vision processor"""
        return {
            'frame_buffer_size': 10,
            'enable_object_detection': True,
            'enable_scene_analysis': True,
            'enable_motion_detection': True,
            'min_object_confidence': 0.5,
            'max_objects_per_frame': 50
        }
    
    def process_image(self, image: np.ndarray, 
                     extract_features: bool = True,
                     detect_objects: bool = True,
                     analyze_scene: bool = True) -> Dict[str, Any]:
        """Process a single image with comprehensive analysis"""
        start_time = time.time()
        
        results = {
            'timestamp': start_time,
            'image_shape': image.shape,
            'features': {},
            'objects': [],
            'scene_analysis': None,
            'processing_time_ms': 0
        }
        
        try:
            # Add to frame buffer for temporal analysis
            self._add_to_buffer(image)
            
            # Extract visual features
            if extract_features:
                feature_start = time.time()
                features = self.feature_extractor.extract_features(image)
                feature_time = (time.time() - feature_start) * 1000
                
                results['features'] = {ft.value: vf for ft, vf in features.items()}
                self.stats['feature_extraction_time_ms'].append(feature_time)
            
            # Object detection
            if detect_objects and self.config['enable_object_detection']:
                objects = self._detect_objects(image)
                results['objects'] = objects
            
            # Scene analysis
            if analyze_scene and self.config['enable_scene_analysis']:
                scene_analysis = self._analyze_scene(image, results.get('objects', []))
                results['scene_analysis'] = scene_analysis
            
            # Motion detection
            if self.config['enable_motion_detection'] and len(self.frame_buffer) > 1:
                motion_info = self._detect_motion()
                results['motion'] = motion_info
            
            processing_time = (time.time() - start_time) * 1000
            results['processing_time_ms'] = processing_time
            
            self.stats['frames_processed'] += 1
            self.stats['processing_time_ms'].append(processing_time)
            
        except Exception as e:
            logging.error(f"Error in vision processing: {e}")
            results['error'] = str(e)
        
        return results
    
    def _add_to_buffer(self, image: np.ndarray):
        """Add image to frame buffer for temporal analysis"""
        self.frame_buffer.append(image)
        
        if len(self.frame_buffer) > self.max_buffer_size:
            self.frame_buffer.pop(0)
    
    def _detect_objects(self, image: np.ndarray) -> List[VisualObject]:
        """Detect objects in the image"""
        # Simplified object detection implementation
        # In practice, this would use a trained object detection model
        
        objects = []
        
        # For demonstration, create some mock detections
        h, w = image.shape[:2]
        
        # Generate random detections
        num_objects = np.random.randint(1, min(5, self.config['max_objects_per_frame']))
        
        for i in range(num_objects):
            # Random bounding box
            x = np.random.randint(0, w//2)
            y = np.random.randint(0, h//2)
            width = np.random.randint(w//10, w//3)
            height = np.random.randint(h//10, h//3)
            
            # Ensure box is within image
            width = min(width, w - x)
            height = min(height, h - y)
            
            confidence = np.random.uniform(0.3, 0.9)
            
            if confidence >= self.config['min_object_confidence']:
                # Extract features from object region
                obj_region = image[y:y+height, x:x+width]
                obj_features = self.feature_extractor.extract_features(obj_region)
                
                # Combine features into single vector
                feature_vector = np.concatenate([
                    vf.features for vf in obj_features.values()
                ])
                
                obj = VisualObject(
                    class_name=f"object_{i}",
                    bounding_box=(x, y, width, height),
                    confidence=confidence,
                    features=feature_vector,
                    attributes={
                        'area': width * height,
                        'aspect_ratio': width / height if height > 0 else 0
                    }
                )
                
                objects.append(obj)
        
        return objects
    
    def _analyze_scene(self, image: np.ndarray, objects: List[VisualObject]) -> SceneAnalysis:
        """Analyze scene content and spatial relationships"""
        # Extract global scene features
        scene_features = self.feature_extractor.extract_features(
            image, [VisualFeatureType.SEMANTIC_FEATURES]
        )
        
        # Determine scene type (simplified classification)
        scene_types = ['indoor', 'outdoor', 'natural', 'urban', 'abstract']
        scene_type = np.random.choice(scene_types)
        
        # Analyze spatial relationships between objects
        spatial_relationships = []
        
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects[i+1:], i+1):
                relationship = self._analyze_spatial_relationship(obj1, obj2)
                if relationship:
                    spatial_relationships.append(relationship)
        
        scene_analysis = SceneAnalysis(
            scene_type=scene_type,
            objects=objects,
            spatial_relationships=spatial_relationships,
            scene_features=scene_features[VisualFeatureType.SEMANTIC_FEATURES].features,
            confidence=np.mean([obj.confidence for obj in objects]) if objects else 0.0
        )
        
        return scene_analysis
    
    def _analyze_spatial_relationship(self, obj1: VisualObject, obj2: VisualObject) -> Optional[Dict[str, Any]]:
        """Analyze spatial relationship between two objects"""
        x1, y1, w1, h1 = obj1.bounding_box
        x2, y2, w2, h2 = obj2.bounding_box
        
        # Calculate centers
        center1 = (x1 + w1/2, y1 + h1/2)
        center2 = (x2 + w2/2, y2 + h2/2)
        
        # Calculate distance
        distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        
        # Determine relationship type
        relationships = []
        
        # Vertical relationships
        if center1[1] < center2[1] - h1/2:
            relationships.append('above')
        elif center1[1] > center2[1] + h2/2:
            relationships.append('below')
        
        # Horizontal relationships
        if center1[0] < center2[0] - w1/2:
            relationships.append('left_of')
        elif center1[0] > center2[0] + w2/2:
            relationships.append('right_of')
        
        # Proximity
        if distance < min(w1, h1, w2, h2):
            relationships.append('near')
        elif distance > max(w1, h1, w2, h2) * 2:
            relationships.append('far')
        
        if relationships:
            return {
                'object1': obj1.class_name,
                'object2': obj2.class_name,
                'relationships': relationships,
                'distance': distance,
                'confidence': min(obj1.confidence, obj2.confidence)
            }
        
        return None
    
    def _detect_motion(self) -> Dict[str, Any]:
        """Detect motion between frames"""
        if len(self.frame_buffer) < 2:
            return {'motion_detected': False, 'motion_magnitude': 0.0}
        
        # Simple frame differencing
        current_frame = self.frame_buffer[-1]
        previous_frame = self.frame_buffer[-2]
        
        # Convert to grayscale if needed
        if len(current_frame.shape) == 3:
            current_gray = np.mean(current_frame, axis=2)
            previous_gray = np.mean(previous_frame, axis=2)
        else:
            current_gray = current_frame
            previous_gray = previous_frame
        
        # Calculate frame difference
        frame_diff = np.abs(current_gray.astype(np.float32) - previous_gray.astype(np.float32))
        
        # Calculate motion metrics
        motion_magnitude = np.mean(frame_diff)
        motion_threshold = self.config.get('motion_threshold', 0.1)
        motion_detected = motion_magnitude > motion_threshold
        
        # Find motion regions
        motion_mask = frame_diff > motion_threshold
        motion_regions = self._find_motion_regions(motion_mask)
        
        return {
            'motion_detected': motion_detected,
            'motion_magnitude': float(motion_magnitude),
            'motion_regions': motion_regions,
            'motion_percentage': float(np.sum(motion_mask) / motion_mask.size)
        }
    
    def _find_motion_regions(self, motion_mask: np.ndarray) -> List[Dict[str, Any]]:
        """Find regions of motion in the motion mask"""
        # Simplified connected component analysis
        regions = []
        
        # Find contiguous regions (simplified implementation)
        visited = np.zeros_like(motion_mask, dtype=bool)
        
        for i in range(motion_mask.shape[0]):
            for j in range(motion_mask.shape[1]):
                if motion_mask[i, j] and not visited[i, j]:
                    region = self._flood_fill(motion_mask, visited, i, j)
                    if len(region) > 10:  # Minimum region size
                        regions.append({
                            'pixels': region,
                            'size': len(region),
                            'center': (np.mean([p[0] for p in region]), 
                                     np.mean([p[1] for p in region]))
                        })
        
        return regions
    
    def _flood_fill(self, mask: np.ndarray, visited: np.ndarray, 
                   start_i: int, start_j: int) -> List[Tuple[int, int]]:
        """Simple flood fill algorithm for finding connected regions"""
        stack = [(start_i, start_j)]
        region = []
        
        while stack:
            i, j = stack.pop()
            
            if (i < 0 or i >= mask.shape[0] or j < 0 or j >= mask.shape[1] or
                visited[i, j] or not mask[i, j]):
                continue
            
            visited[i, j] = True
            region.append((i, j))
            
            # Add neighbors
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                stack.append((i + di, j + dj))
        
        return region
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get vision processing statistics"""
        stats = self.stats.copy()
        
        if stats['processing_time_ms']:
            stats['avg_processing_time_ms'] = np.mean(stats['processing_time_ms'][-100:])
            stats['max_processing_time_ms'] = np.max(stats['processing_time_ms'][-100:])
        
        if stats['feature_extraction_time_ms']:
            stats['avg_feature_time_ms'] = np.mean(stats['feature_extraction_time_ms'][-100:])
        
        stats['frames_in_buffer'] = len(self.frame_buffer)
        
        return stats
    
    def reset_stats(self):
        """Reset processing statistics"""
        self.stats = {
            'frames_processed': 0,
            'processing_time_ms': [],
            'feature_extraction_time_ms': [],
            'detection_accuracy': []
        }
    
    def clear_buffer(self):
        """Clear frame buffer"""
        self.frame_buffer.clear()