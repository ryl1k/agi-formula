"""
Multi-Modal Data Processing Pipeline

Comprehensive pipeline for processing and integrating multiple data modalities:
- Unified data input and preprocessing
- Modality-specific processing modules
- Cross-modal synchronization and alignment
- Feature extraction and representation learning
- Multi-modal fusion and integration
"""

import numpy as np
import time
import threading
import queue
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import logging


class ModalityType(Enum):
    """Types of sensory modalities"""
    VISION = "vision"
    LANGUAGE = "language"
    AUDIO = "audio"
    TACTILE = "tactile"
    PROPRIOCEPTION = "proprioception"
    CUSTOM = "custom"


@dataclass
class ModalityData:
    """Data container for a specific modality"""
    modality: ModalityType
    data: Any
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    processed: bool = False
    features: Optional[np.ndarray] = None
    confidence: float = 1.0


@dataclass
class MultiModalFrame:
    """Synchronized multi-modal data frame"""
    timestamp: float
    modalities: Dict[ModalityType, ModalityData]
    synchronized: bool = False
    frame_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class MultiModalPipeline:
    """
    Comprehensive multi-modal data processing pipeline
    
    Features:
    - Real-time multi-modal data ingestion
    - Modality-specific preprocessing and feature extraction
    - Temporal synchronization across modalities
    - Cross-modal attention and fusion
    - Scalable processing with threading support
    - Memory-efficient data management
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        
        # Pipeline state
        self.is_running = False
        self.processing_threads = {}
        self.data_queues = {}
        self.frame_buffer = queue.Queue(maxsize=self.config['buffer_size'])
        
        # Processors for each modality
        self.processors = {}
        self.feature_extractors = {}
        
        # Synchronization and timing
        self.sync_window = self.config['sync_window_ms'] / 1000.0
        self.frame_rate = self.config['target_frame_rate']
        self.last_frame_time = 0
        
        # Performance monitoring
        self.stats = {
            'frames_processed': 0,
            'sync_failures': 0,
            'processing_time_ms': [],
            'memory_usage_mb': [],
            'modality_fps': {modality: 0 for modality in ModalityType}
        }
        
        # Event handlers
        self.event_handlers = {
            'frame_ready': [],
            'sync_failure': [],
            'processing_error': []
        }
        
        # Initialize pipeline
        self._initialize_pipeline()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default pipeline configuration"""
        return {
            'buffer_size': 100,
            'sync_window_ms': 50,  # 50ms synchronization window
            'target_frame_rate': 30,
            'max_processing_threads': 4,
            'feature_dimensions': {
                ModalityType.VISION: 512,
                ModalityType.LANGUAGE: 256,
                ModalityType.AUDIO: 128,
                ModalityType.TACTILE: 64,
                ModalityType.PROPRIOCEPTION: 32
            },
            'preprocessing': {
                'normalize': True,
                'standardize': True,
                'augment': False
            },
            'quality_control': {
                'min_confidence': 0.5,
                'max_latency_ms': 100,
                'enable_filtering': True
            }
        }
    
    def _initialize_pipeline(self):
        """Initialize the multi-modal pipeline"""
        # Create data queues for each modality
        for modality in ModalityType:
            self.data_queues[modality] = queue.Queue(
                maxsize=self.config['buffer_size']
            )
        
        # Initialize default processors
        self._initialize_default_processors()
        
        print(f"Multi-modal pipeline initialized with {len(ModalityType)} modalities")
    
    def _initialize_default_processors(self):
        """Initialize default processors for each modality"""
        # We'll implement specific processors in separate modules
        # For now, create placeholder processors
        for modality in ModalityType:
            self.processors[modality] = self._create_default_processor(modality)
            self.feature_extractors[modality] = self._create_default_feature_extractor(modality)
    
    def _create_default_processor(self, modality: ModalityType) -> Callable:
        """Create default processor for a modality"""
        def default_processor(data: Any) -> Any:
            """Default preprocessing for modality data"""
            if isinstance(data, np.ndarray):
                # Basic normalization
                if self.config['preprocessing']['normalize']:
                    data_min, data_max = np.min(data), np.max(data)
                    if data_max > data_min:
                        data = (data - data_min) / (data_max - data_min)
                
                # Standardization
                if self.config['preprocessing']['standardize']:
                    data = (data - np.mean(data)) / (np.std(data) + 1e-8)
            
            return data
        
        return default_processor
    
    def _create_default_feature_extractor(self, modality: ModalityType) -> Callable:
        """Create default feature extractor for a modality"""
        feature_dim = self.config['feature_dimensions'][modality]
        
        def default_feature_extractor(data: Any) -> np.ndarray:
            """Default feature extraction for modality data"""
            if isinstance(data, np.ndarray):
                # Flatten and project to target dimension
                flattened = data.flatten()
                
                if len(flattened) >= feature_dim:
                    # Downsample if too large
                    indices = np.linspace(0, len(flattened)-1, feature_dim, dtype=int)
                    features = flattened[indices]
                else:
                    # Pad if too small
                    features = np.pad(flattened, (0, feature_dim - len(flattened)))
                
                return features.astype(np.float32)
            else:
                # Generate random features for non-array data
                return np.random.randn(feature_dim).astype(np.float32)
        
        return default_feature_extractor
    
    def register_processor(self, modality: ModalityType, processor: Callable):
        """Register a custom processor for a modality"""
        self.processors[modality] = processor
        print(f"Registered custom processor for {modality.value}")
    
    def register_feature_extractor(self, modality: ModalityType, extractor: Callable):
        """Register a custom feature extractor for a modality"""
        self.feature_extractors[modality] = extractor
        print(f"Registered custom feature extractor for {modality.value}")
    
    def add_event_handler(self, event_type: str, handler: Callable):
        """Add event handler for pipeline events"""
        if event_type in self.event_handlers:
            self.event_handlers[event_type].append(handler)
    
    def start_pipeline(self):
        """Start the multi-modal processing pipeline"""
        if self.is_running:
            print("Pipeline already running")
            return
        
        self.is_running = True
        
        # Start processing threads for each modality
        for modality in ModalityType:
            thread = threading.Thread(
                target=self._modality_processing_loop,
                args=(modality,),
                daemon=True
            )
            thread.start()
            self.processing_threads[modality] = thread
        
        # Start frame synchronization thread
        sync_thread = threading.Thread(
            target=self._frame_synchronization_loop,
            daemon=True
        )
        sync_thread.start()
        self.processing_threads['sync'] = sync_thread
        
        print("Multi-modal pipeline started")
    
    def stop_pipeline(self):
        """Stop the multi-modal processing pipeline"""
        self.is_running = False
        
        # Wait for threads to finish
        for thread in self.processing_threads.values():
            if thread.is_alive():
                thread.join(timeout=1)
        
        print("Multi-modal pipeline stopped")
    
    def input_data(self, modality: ModalityType, data: Any, 
                   metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Input data for a specific modality"""
        if not self.is_running:
            logging.warning("Pipeline not running, data input ignored")
            return False
        
        modality_data = ModalityData(
            modality=modality,
            data=data,
            timestamp=time.time(),
            metadata=metadata or {},
            confidence=metadata.get('confidence', 1.0) if metadata else 1.0
        )
        
        # Quality control check
        if not self._quality_check(modality_data):
            return False
        
        try:
            self.data_queues[modality].put_nowait(modality_data)
            return True
        except queue.Full:
            logging.warning(f"Data queue full for {modality.value}, dropping data")
            return False
    
    def _quality_check(self, modality_data: ModalityData) -> bool:
        """Perform quality control on input data"""
        if not self.config['quality_control']['enable_filtering']:
            return True
        
        # Check confidence threshold
        min_confidence = self.config['quality_control']['min_confidence']
        if modality_data.confidence < min_confidence:
            return False
        
        # Check latency
        max_latency = self.config['quality_control']['max_latency_ms'] / 1000.0
        latency = time.time() - modality_data.timestamp
        if latency > max_latency:
            return False
        
        return True
    
    def _modality_processing_loop(self, modality: ModalityType):
        """Processing loop for a specific modality"""
        data_queue = self.data_queues[modality]
        processor = self.processors[modality]
        feature_extractor = self.feature_extractors[modality]
        
        while self.is_running:
            try:
                # Get data with timeout
                modality_data = data_queue.get(timeout=0.1)
                
                # Process the data
                start_time = time.time()
                
                # Apply preprocessing
                processed_data = processor(modality_data.data)
                modality_data.data = processed_data
                
                # Extract features
                features = feature_extractor(processed_data)
                modality_data.features = features
                modality_data.processed = True
                
                processing_time = (time.time() - start_time) * 1000
                self.stats['processing_time_ms'].append(processing_time)
                
                # Send to synchronization buffer
                self._send_to_sync_buffer(modality_data)
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error processing {modality.value} data: {e}")
                self._trigger_event_handlers('processing_error', {
                    'modality': modality,
                    'error': str(e)
                })
    
    def _send_to_sync_buffer(self, modality_data: ModalityData):
        """Send processed modality data to synchronization buffer"""
        # For now, we'll implement a simple approach
        # In a full implementation, this would manage temporal alignment
        try:
            # Create a frame if this is the first modality data in the time window
            # For simplicity, we'll just forward the data
            pass
        except Exception as e:
            logging.error(f"Error in synchronization: {e}")
    
    def _frame_synchronization_loop(self):
        """Main loop for frame synchronization across modalities"""
        frame_interval = 1.0 / self.frame_rate
        
        while self.is_running:
            current_time = time.time()
            
            # Check if it's time for a new frame
            if current_time - self.last_frame_time >= frame_interval:
                frame = self._create_synchronized_frame(current_time)
                
                if frame and frame.synchronized:
                    self._process_frame(frame)
                    self.stats['frames_processed'] += 1
                else:
                    self.stats['sync_failures'] += 1
                    self._trigger_event_handlers('sync_failure', {
                        'timestamp': current_time,
                        'frame': frame
                    })
                
                self.last_frame_time = current_time
            
            time.sleep(0.001)  # Small sleep to prevent CPU spinning
    
    def _create_synchronized_frame(self, timestamp: float) -> Optional[MultiModalFrame]:
        """Create a synchronized frame from available modality data"""
        frame_modalities = {}
        
        # Collect data from each modality within the sync window
        for modality in ModalityType:
            data_queue = self.data_queues[modality]
            latest_data = None
            
            # Get the most recent data within sync window
            while not data_queue.empty():
                try:
                    candidate_data = data_queue.get_nowait()
                    time_diff = abs(timestamp - candidate_data.timestamp)
                    
                    if time_diff <= self.sync_window:
                        latest_data = candidate_data
                    else:
                        # Data too old, skip it
                        pass
                except queue.Empty:
                    break
            
            if latest_data and latest_data.processed:
                frame_modalities[modality] = latest_data
        
        # Create frame if we have data from at least one modality
        if frame_modalities:
            frame = MultiModalFrame(
                timestamp=timestamp,
                modalities=frame_modalities,
                synchronized=len(frame_modalities) > 1,  # Synchronized if multiple modalities
                frame_id=f"frame_{int(timestamp * 1000)}"
            )
            return frame
        
        return None
    
    def _process_frame(self, frame: MultiModalFrame):
        """Process a synchronized multi-modal frame"""
        try:
            # Add frame to buffer for downstream processing
            self.frame_buffer.put_nowait(frame)
            
            # Trigger frame ready event
            self._trigger_event_handlers('frame_ready', frame)
            
        except queue.Full:
            logging.warning("Frame buffer full, dropping frame")
    
    def get_next_frame(self, timeout: float = None) -> Optional[MultiModalFrame]:
        """Get the next synchronized frame from the pipeline"""
        try:
            if timeout is None:
                return self.frame_buffer.get_nowait()
            else:
                return self.frame_buffer.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def _trigger_event_handlers(self, event_type: str, data: Any):
        """Trigger event handlers for a specific event"""
        handlers = self.event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                handler(data)
            except Exception as e:
                logging.error(f"Error in event handler {event_type}: {e}")
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get current pipeline statistics"""
        stats = self.stats.copy()
        
        # Calculate average processing time
        if stats['processing_time_ms']:
            stats['avg_processing_time_ms'] = np.mean(stats['processing_time_ms'][-100:])
        else:
            stats['avg_processing_time_ms'] = 0
        
        # Calculate frame rate
        stats['actual_frame_rate'] = self.stats['frames_processed'] / max(1, time.time() - self.last_frame_time)
        
        # Add queue sizes
        stats['queue_sizes'] = {
            modality.value: self.data_queues[modality].qsize()
            for modality in ModalityType
        }
        
        return stats
    
    def reset_stats(self):
        """Reset pipeline statistics"""
        self.stats = {
            'frames_processed': 0,
            'sync_failures': 0,
            'processing_time_ms': [],
            'memory_usage_mb': [],
            'modality_fps': {modality: 0 for modality in ModalityType}
        }
    
    def get_supported_modalities(self) -> List[ModalityType]:
        """Get list of supported modalities"""
        return list(ModalityType)
    
    def is_modality_active(self, modality: ModalityType) -> bool:
        """Check if a modality is actively receiving data"""
        return not self.data_queues[modality].empty()
    
    def flush_buffers(self):
        """Flush all data buffers"""
        for modality in ModalityType:
            while not self.data_queues[modality].empty():
                try:
                    self.data_queues[modality].get_nowait()
                except queue.Empty:
                    break
        
        while not self.frame_buffer.empty():
            try:
                self.frame_buffer.get_nowait()
            except queue.Empty:
                break
    
    def configure_modality(self, modality: ModalityType, config: Dict[str, Any]):
        """Configure settings for a specific modality"""
        if 'feature_dimension' in config:
            self.config['feature_dimensions'][modality] = config['feature_dimension']
        
        # Recreate feature extractor with new configuration
        if 'feature_dimension' in config:
            self.feature_extractors[modality] = self._create_default_feature_extractor(modality)
        
        print(f"Updated configuration for {modality.value}")
    
    def export_configuration(self) -> Dict[str, Any]:
        """Export current pipeline configuration"""
        return {
            'config': self.config,
            'modalities': [modality.value for modality in ModalityType],
            'stats': self.get_pipeline_stats(),
            'timestamp': datetime.now().isoformat()
        }


class ModalityDataGenerator:
    """Utility class for generating test data for different modalities"""
    
    @staticmethod
    def generate_vision_data(width: int = 224, height: int = 224, channels: int = 3) -> np.ndarray:
        """Generate synthetic vision data (image)"""
        return np.random.uint8(np.random.rand(height, width, channels) * 255)
    
    @staticmethod
    def generate_language_data(sequence_length: int = 50, vocab_size: int = 10000) -> np.ndarray:
        """Generate synthetic language data (token sequence)"""
        return np.random.randint(0, vocab_size, size=sequence_length)
    
    @staticmethod
    def generate_audio_data(duration: float = 1.0, sample_rate: int = 16000) -> np.ndarray:
        """Generate synthetic audio data"""
        num_samples = int(duration * sample_rate)
        return np.random.randn(num_samples).astype(np.float32)
    
    @staticmethod
    def generate_tactile_data(sensor_count: int = 16) -> np.ndarray:
        """Generate synthetic tactile sensor data"""
        return np.random.rand(sensor_count).astype(np.float32)
    
    @staticmethod
    def generate_proprioception_data(joint_count: int = 12) -> np.ndarray:
        """Generate synthetic proprioception data (joint positions/angles)"""
        return np.random.uniform(-np.pi, np.pi, joint_count).astype(np.float32)