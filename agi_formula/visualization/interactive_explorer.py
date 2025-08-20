"""
Interactive Network Explorer for AGI-Formula

Advanced interactive exploration of AGI networks:
- Interactive 3D network navigation
- Neuron-level inspection and analysis
- Causal path tracing and visualization
- Dynamic network manipulation
- Real-time concept exploration
"""

import json
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ExplorationSession:
    """Data structure for exploration sessions"""
    session_id: str
    start_time: datetime
    selected_neurons: Set[int]
    traced_paths: List[List[int]]
    bookmarks: List[Dict[str, Any]]
    annotations: Dict[int, str]


class InteractiveExplorer:
    """
    Interactive explorer for AGI network analysis
    
    Features:
    - 3D interactive network navigation
    - Neuron inspection and detailed analysis
    - Causal path tracing and visualization
    - Network manipulation and experimentation
    - Concept exploration and mapping
    - Session management and bookmarking
    """
    
    def __init__(self, network, webgl_renderer=None, config: Optional[Dict[str, Any]] = None):
        self.network = network
        self.webgl_renderer = webgl_renderer
        self.config = config or self._get_default_config()
        
        # Exploration state
        self.current_session = None
        self.exploration_history = []
        self.selected_neurons = set()
        self.highlighted_paths = []
        
        # Analysis components
        self.path_analyzer = PathAnalyzer(self.network)
        self.concept_mapper = ConceptMapper(self.network)
        self.influence_analyzer = InfluenceAnalyzer(self.network)
        
        # Interactive tools
        self.interaction_modes = {
            'explore': ExploreMode(self),
            'trace': TraceMode(self),
            'analyze': AnalyzeMode(self),
            'manipulate': ManipulateMode(self)
        }
        
        self.current_mode = 'explore'
        
        # Initialize explorer
        self._initialize_explorer()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default explorer configuration"""
        return {
            'interaction_settings': {
                'selection_radius': 2.0,
                'path_trace_depth': 5,
                'influence_threshold': 0.1,
                'animation_duration': 1000,
                'hover_delay': 500
            },
            'visualization_settings': {
                'highlight_color': 0xffff00,
                'selection_color': 0xff0000,
                'path_colors': [0xff6b35, 0x35ff6b, 0x6b35ff, 0xff35ff],
                'fade_unselected': True,
                'show_labels': True,
                'show_connections': True
            },
            'analysis_settings': {
                'max_path_length': 10,
                'min_influence_strength': 0.05,
                'concept_similarity_threshold': 0.7,
                'enable_real_time_analysis': True
            },
            'session_settings': {
                'auto_save_interval': 300,  # 5 minutes
                'max_history_length': 100,
                'enable_bookmarks': True
            }
        }
    
    def _initialize_explorer(self):
        """Initialize interactive explorer"""
        # Create initial exploration session
        self.start_new_session()
        
        # Setup interaction handlers
        self._setup_interaction_handlers()
        
        print(f"Interactive explorer initialized for network with {self.network.config.num_neurons} neurons")
    
    def start_new_session(self, session_name: Optional[str] = None) -> str:
        """Start a new exploration session"""
        session_id = session_name or f"session_{int(time.time())}"
        
        self.current_session = ExplorationSession(
            session_id=session_id,
            start_time=datetime.now(),
            selected_neurons=set(),
            traced_paths=[],
            bookmarks=[],
            annotations={}
        )
        
        # Clear previous exploration state
        self.selected_neurons.clear()
        self.highlighted_paths.clear()
        
        print(f"Started new exploration session: {session_id}")
        return session_id
    
    def set_interaction_mode(self, mode: str):
        """Set current interaction mode"""
        if mode in self.interaction_modes:
            self.current_mode = mode
            self.interaction_modes[mode].activate()
            print(f"Switched to {mode} mode")
        else:
            raise ValueError(f"Unknown interaction mode: {mode}")
    
    def select_neuron(self, neuron_id: int, add_to_selection: bool = False):
        """Select a neuron for detailed analysis"""
        if not add_to_selection:
            self.selected_neurons.clear()
        
        self.selected_neurons.add(neuron_id)
        self.current_session.selected_neurons.add(neuron_id)
        
        # Perform analysis on selected neuron
        analysis = self._analyze_neuron(neuron_id)
        
        # Update visualization
        self._update_selection_visualization()
        
        return analysis
    
    def trace_causal_path(self, start_neuron: int, direction: str = 'forward', max_depth: int = None) -> List[List[int]]:
        """Trace causal paths from a neuron"""
        max_depth = max_depth or self.config['analysis_settings']['max_path_length']
        
        paths = self.path_analyzer.trace_paths(
            start_neuron=start_neuron,
            direction=direction,
            max_depth=max_depth,
            min_strength=self.config['analysis_settings']['min_influence_strength']
        )
        
        # Add paths to current session
        self.current_session.traced_paths.extend(paths)
        self.highlighted_paths = paths
        
        # Update visualization
        self._update_path_visualization(paths)
        
        return paths
    
    def analyze_influence_network(self, target_neuron: int, radius: int = 2) -> Dict[str, Any]:
        """Analyze the influence network around a neuron"""
        influence_data = self.influence_analyzer.analyze_local_influence(
            target_neuron=target_neuron,
            radius=radius
        )
        
        # Highlight influenced neurons
        influenced_neurons = influence_data['influenced_neurons']
        for neuron_id in influenced_neurons:
            self.selected_neurons.add(neuron_id)
        
        self._update_selection_visualization()
        
        return influence_data
    
    def explore_concept_relationships(self, concept_name: str) -> Dict[str, Any]:
        """Explore relationships between concepts"""
        concept_analysis = self.concept_mapper.analyze_concept_relationships(concept_name)
        
        # Highlight neurons related to concept
        related_neurons = concept_analysis.get('related_neurons', [])
        for neuron_id in related_neurons:
            self.selected_neurons.add(neuron_id)
        
        self._update_selection_visualization()
        
        return concept_analysis
    
    def create_bookmark(self, name: str, description: str = "") -> str:
        """Create a bookmark of current exploration state"""
        bookmark = {
            'id': f"bookmark_{int(time.time())}",
            'name': name,
            'description': description,
            'timestamp': datetime.now().isoformat(),
            'selected_neurons': list(self.selected_neurons),
            'highlighted_paths': self.highlighted_paths.copy(),
            'current_mode': self.current_mode,
            'camera_position': self._get_camera_position(),
            'annotations': dict(self.current_session.annotations)
        }
        
        self.current_session.bookmarks.append(bookmark)
        print(f"Created bookmark: {name}")
        
        return bookmark['id']
    
    def load_bookmark(self, bookmark_id: str):
        """Load a previously created bookmark"""
        bookmark = None
        for b in self.current_session.bookmarks:
            if b['id'] == bookmark_id:
                bookmark = b
                break
        
        if not bookmark:
            raise ValueError(f"Bookmark not found: {bookmark_id}")
        
        # Restore exploration state
        self.selected_neurons = set(bookmark['selected_neurons'])
        self.highlighted_paths = bookmark['highlighted_paths']
        self.current_mode = bookmark['current_mode']
        
        # Restore annotations
        self.current_session.annotations.update(bookmark['annotations'])
        
        # Update visualization
        self._update_selection_visualization()
        self._update_path_visualization(self.highlighted_paths)
        
        print(f"Loaded bookmark: {bookmark['name']}")
    
    def annotate_neuron(self, neuron_id: int, annotation: str):
        """Add annotation to a neuron"""
        self.current_session.annotations[neuron_id] = annotation
        print(f"Added annotation to neuron {neuron_id}: {annotation}")
    
    def manipulate_network(self, manipulation_type: str, **kwargs) -> Dict[str, Any]:
        """Perform network manipulation for experimentation"""
        manipulator = self.interaction_modes['manipulate']
        
        if manipulation_type == 'strengthen_connection':
            return manipulator.strengthen_connection(
                source=kwargs['source'],
                target=kwargs['target'],
                factor=kwargs.get('factor', 1.5)
            )
        elif manipulation_type == 'weaken_connection':
            return manipulator.weaken_connection(
                source=kwargs['source'],
                target=kwargs['target'],
                factor=kwargs.get('factor', 0.5)
            )
        elif manipulation_type == 'inject_activation':
            return manipulator.inject_activation(
                neuron_id=kwargs['neuron_id'],
                activation_value=kwargs['activation_value']
            )
        elif manipulation_type == 'disable_neuron':
            return manipulator.disable_neuron(kwargs['neuron_id'])
        else:
            raise ValueError(f"Unknown manipulation type: {manipulation_type}")
    
    def _analyze_neuron(self, neuron_id: int) -> Dict[str, Any]:
        """Perform detailed analysis of a neuron"""
        analysis = {
            'neuron_id': neuron_id,
            'timestamp': datetime.now().isoformat(),
            'basic_properties': self._get_neuron_properties(neuron_id),
            'connections': self._analyze_connections(neuron_id),
            'influence': self._analyze_influence(neuron_id),
            'concepts': self._analyze_concepts(neuron_id),
            'activation_history': self._get_activation_history(neuron_id)
        }
        
        return analysis
    
    def _get_neuron_properties(self, neuron_id: int) -> Dict[str, Any]:
        """Get basic properties of a neuron"""
        properties = {
            'id': neuron_id,
            'type': self._get_neuron_type(neuron_id),
            'current_activation': 0.0,  # Placeholder
            'position': self._get_neuron_position(neuron_id)
        }
        
        # Get additional properties if available
        if hasattr(self.network, 'neurons') and neuron_id < len(self.network.neurons):
            neuron = self.network.neurons[neuron_id]
            properties.update({
                'concept_type': getattr(neuron, 'concept_type', 'unknown'),
                'activation_function': getattr(neuron, 'activation_function', 'sigmoid'),
                'learning_rate': getattr(neuron, 'learning_rate', 0.01)
            })
        
        return properties
    
    def _analyze_connections(self, neuron_id: int) -> Dict[str, Any]:
        """Analyze connections of a neuron"""
        connections = {
            'incoming': [],
            'outgoing': [],
            'total_strength': 0.0,
            'avg_strength': 0.0
        }
        
        if hasattr(self.network, 'neurons') and neuron_id < len(self.network.neurons):
            neuron = self.network.neurons[neuron_id]
            
            # Analyze outgoing connections
            if hasattr(neuron, 'neighbors'):
                for neighbor_id in neuron.neighbors:
                    if neighbor_id < len(self.network.neurons):
                        strength = np.random.uniform(0.1, 1.0)  # Placeholder
                        connections['outgoing'].append({
                            'target': neighbor_id,
                            'strength': strength,
                            'type': 'excitatory' if strength > 0 else 'inhibitory'
                        })
                        connections['total_strength'] += abs(strength)
            
            # Calculate average strength
            if connections['outgoing']:
                connections['avg_strength'] = connections['total_strength'] / len(connections['outgoing'])
        
        return connections
    
    def _analyze_influence(self, neuron_id: int) -> Dict[str, Any]:
        """Analyze influence patterns of a neuron"""
        return self.influence_analyzer.analyze_neuron_influence(neuron_id)
    
    def _analyze_concepts(self, neuron_id: int) -> Dict[str, Any]:
        """Analyze concept associations of a neuron"""
        return self.concept_mapper.analyze_neuron_concepts(neuron_id)
    
    def _get_activation_history(self, neuron_id: int) -> List[Dict[str, Any]]:
        """Get activation history for a neuron"""
        # Placeholder implementation
        history = []
        for i in range(10):
            history.append({
                'timestamp': (datetime.now().timestamp() - i * 10),
                'activation': np.random.uniform(0, 1),
                'context': f'step_{i}'
            })
        return history
    
    def _get_neuron_type(self, neuron_id: int) -> str:
        """Get type of neuron"""
        if neuron_id < self.network.config.input_size:
            return 'input'
        elif neuron_id >= self.network.config.num_neurons - self.network.config.output_size:
            return 'output'
        else:
            return 'hidden'
    
    def _get_neuron_position(self, neuron_id: int) -> List[float]:
        """Get 3D position of neuron"""
        if self.webgl_renderer and neuron_id < len(self.webgl_renderer.scene_data['nodes']):
            return self.webgl_renderer.scene_data['nodes'][neuron_id]['position']
        else:
            # Generate placeholder position
            return [np.random.uniform(-10, 10) for _ in range(3)]
    
    def _get_camera_position(self) -> Dict[str, float]:
        """Get current camera position"""
        # Placeholder implementation
        return {'x': 0, 'y': 0, 'z': 20}
    
    def _update_selection_visualization(self):
        """Update visualization to show current selection"""
        if self.webgl_renderer:
            # Update node colors for selected neurons
            for node_data in self.webgl_renderer.scene_data['nodes']:
                node_id = node_data['id']
                if node_id in self.selected_neurons:
                    node_data['style']['color'] = self.config['visualization_settings']['selection_color']
                else:
                    # Reset to original color
                    node_type = node_data['type']
                    original_style = self.webgl_renderer.scene_data['styles']['nodes'][node_type]
                    node_data['style']['color'] = original_style['color']
    
    def _update_path_visualization(self, paths: List[List[int]]):
        """Update visualization to show traced paths"""
        if not self.webgl_renderer:
            return
        
        # Clear existing path edges
        self.webgl_renderer.scene_data['edges'] = [
            e for e in self.webgl_renderer.scene_data['edges'] 
            if e.get('path_id') is None
        ]
        
        # Add path edges
        path_colors = self.config['visualization_settings']['path_colors']
        
        for path_idx, path in enumerate(paths):
            color = path_colors[path_idx % len(path_colors)]
            
            for i in range(len(path) - 1):
                source = path[i]
                target = path[i + 1]
                
                path_edge = {
                    'source': source,
                    'target': target,
                    'weight': 1.0,
                    'type': 'path',
                    'path_id': path_idx,
                    'style': {
                        'color': color,
                        'opacity': 0.8
                    },
                    'animated': True
                }
                
                self.webgl_renderer.scene_data['edges'].append(path_edge)
    
    def _setup_interaction_handlers(self):
        """Setup interaction event handlers"""
        # Placeholder for interaction handlers
        print("Interaction handlers setup complete")
    
    def get_exploration_summary(self) -> Dict[str, Any]:
        """Get summary of current exploration"""
        if not self.current_session:
            return {}
        
        summary = {
            'session_id': self.current_session.session_id,
            'duration': (datetime.now() - self.current_session.start_time).total_seconds(),
            'selected_neurons_count': len(self.selected_neurons),
            'traced_paths_count': len(self.current_session.traced_paths),
            'bookmarks_count': len(self.current_session.bookmarks),
            'annotations_count': len(self.current_session.annotations),
            'current_mode': self.current_mode,
            'network_coverage': len(self.selected_neurons) / self.network.config.num_neurons
        }
        
        return summary
    
    def export_exploration_data(self, output_file: str = "exploration_export.json") -> str:
        """Export exploration session data"""
        if not self.current_session:
            raise ValueError("No active exploration session")
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'session_data': {
                'session_id': self.current_session.session_id,
                'start_time': self.current_session.start_time.isoformat(),
                'selected_neurons': list(self.current_session.selected_neurons),
                'traced_paths': self.current_session.traced_paths,
                'bookmarks': self.current_session.bookmarks,
                'annotations': self.current_session.annotations
            },
            'network_info': {
                'num_neurons': self.network.config.num_neurons,
                'input_size': self.network.config.input_size,
                'output_size': self.network.config.output_size
            },
            'analysis_results': self._get_session_analysis()
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        return output_file
    
    def _get_session_analysis(self) -> Dict[str, Any]:
        """Get analysis results for current session"""
        analysis = {
            'most_analyzed_neurons': [],
            'frequent_paths': [],
            'concept_discoveries': [],
            'influence_patterns': []
        }
        
        # Analyze most frequently selected neurons
        neuron_counts = {}
        for neuron_id in self.current_session.selected_neurons:
            neuron_counts[neuron_id] = neuron_counts.get(neuron_id, 0) + 1
        
        analysis['most_analyzed_neurons'] = sorted(
            neuron_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        return analysis
    
    def generate_interactive_html(self, output_file: str = "interactive_explorer.html") -> str:
        """Generate interactive exploration interface"""
        html_template = self._get_explorer_template()
        
        # Embed network data and configuration
        network_data = self._prepare_network_data()
        config_json = json.dumps(self.config, indent=2, default=str)
        
        # Generate complete HTML
        html_content = html_template.format(
            network_data=json.dumps(network_data, indent=2, default=str),
            config=config_json,
            title=f"AGI Network Interactive Explorer - {self.network.config.num_neurons} neurons"
        )
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Interactive explorer saved to: {output_file}")
        return output_file
    
    def _prepare_network_data(self) -> Dict[str, Any]:
        """Prepare network data for visualization"""
        data = {
            'neurons': [],
            'connections': [],
            'concepts': [],
            'current_selection': list(self.selected_neurons),
            'traced_paths': self.highlighted_paths
        }
        
        # Add neuron data
        for i in range(self.network.config.num_neurons):
            neuron_data = {
                'id': i,
                'type': self._get_neuron_type(i),
                'position': self._get_neuron_position(i),
                'properties': self._get_neuron_properties(i)
            }
            data['neurons'].append(neuron_data)
        
        # Add connection data
        if hasattr(self.network, 'neurons'):
            for i, neuron in enumerate(self.network.neurons):
                if hasattr(neuron, 'neighbors'):
                    for neighbor_id in neuron.neighbors:
                        if neighbor_id < len(self.network.neurons):
                            connection = {
                                'source': i,
                                'target': neighbor_id,
                                'strength': np.random.uniform(0.1, 1.0)  # Placeholder
                            }
                            data['connections'].append(connection)
        
        return data
    
    def _get_explorer_template(self) -> str:
        """Get HTML template for interactive explorer"""
        return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/dat-gui/0.7.9/dat.gui.min.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 0;
            background: #0a0a0a;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: white;
            overflow: hidden;
        }}
        
        #container {{
            position: relative;
            width: 100vw;
            height: 100vh;
        }}
        
        #exploration-panel {{
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(0, 0, 0, 0.9);
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #333;
            min-width: 300px;
            max-height: 80vh;
            overflow-y: auto;
            z-index: 1000;
        }}
        
        .mode-selector {{
            display: flex;
            gap: 5px;
            margin-bottom: 15px;
        }}
        
        .mode-btn {{
            padding: 8px 12px;
            background: #333;
            border: none;
            color: white;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
        }}
        
        .mode-btn.active {{
            background: #4a90e2;
        }}
        
        .analysis-section {{
            margin: 15px 0;
            padding: 10px;
            background: #1a1a1a;
            border-radius: 4px;
        }}
        
        .neuron-info {{
            font-size: 12px;
            line-height: 1.4;
        }}
        
        .path-trace {{
            margin: 10px 0;
        }}
        
        .path-item {{
            background: #2a2a2a;
            padding: 5px;
            margin: 2px 0;
            border-radius: 3px;
            font-size: 11px;
        }}
        
        .bookmark-list {{
            max-height: 200px;
            overflow-y: auto;
        }}
        
        .bookmark-item {{
            background: #2a2a2a;
            padding: 8px;
            margin: 3px 0;
            border-radius: 3px;
            cursor: pointer;
            font-size: 12px;
        }}
        
        .bookmark-item:hover {{
            background: #3a3a3a;
        }}
        
        input, textarea, select {{
            width: 100%;
            padding: 5px;
            background: #222;
            border: 1px solid #444;
            color: white;
            border-radius: 3px;
            margin: 5px 0;
        }}
        
        button {{
            background: #4a90e2;
            color: white;
            border: none;
            padding: 6px 10px;
            border-radius: 3px;
            cursor: pointer;
            margin: 2px;
            font-size: 11px;
        }}
        
        button:hover {{
            background: #5aa0f2;
        }}
        
        h4 {{
            margin: 10px 0 5px 0;
            color: #4a90e2;
            font-size: 14px;
        }}
        
        .stats {{
            font-size: 11px;
            color: #888;
            margin: 5px 0;
        }}
    </style>
</head>
<body>
    <div id="container">
        <div id="exploration-panel">
            <h3>Network Explorer</h3>
            
            <div class="mode-selector">
                <button class="mode-btn active" data-mode="explore">Explore</button>
                <button class="mode-btn" data-mode="trace">Trace</button>
                <button class="mode-btn" data-mode="analyze">Analyze</button>
                <button class="mode-btn" data-mode="manipulate">Manipulate</button>
            </div>
            
            <div class="analysis-section">
                <h4>Selection Info</h4>
                <div id="selection-info" class="neuron-info">
                    Click on neurons to select and analyze them
                </div>
                <button onclick="clearSelection()">Clear Selection</button>
            </div>
            
            <div class="analysis-section" id="trace-section" style="display: none;">
                <h4>Path Tracing</h4>
                <select id="trace-direction">
                    <option value="forward">Forward</option>
                    <option value="backward">Backward</option>
                    <option value="both">Both</option>
                </select>
                <input type="number" id="trace-depth" placeholder="Max depth" value="5" min="1" max="10">
                <button onclick="traceFromSelected()">Trace Path</button>
                
                <div id="traced-paths" class="path-trace">
                    <!-- Traced paths will appear here -->
                </div>
            </div>
            
            <div class="analysis-section" id="analyze-section" style="display: none;">
                <h4>Analysis Tools</h4>
                <button onclick="analyzeInfluence()">Analyze Influence</button>
                <button onclick="findConcepts()">Find Concepts</button>
                <button onclick="calculateCentrality()">Calculate Centrality</button>
                
                <div id="analysis-results">
                    <!-- Analysis results will appear here -->
                </div>
            </div>
            
            <div class="analysis-section" id="manipulate-section" style="display: none;">
                <h4>Network Manipulation</h4>
                <button onclick="strengthenConnections()">Strengthen</button>
                <button onclick="weakenConnections()">Weaken</button>
                <button onclick="injectActivation()">Inject Signal</button>
                <button onclick="disableNeurons()">Disable</button>
                
                <div id="manipulation-controls">
                    <!-- Manipulation controls will appear here -->
                </div>
            </div>
            
            <div class="analysis-section">
                <h4>Bookmarks</h4>
                <input type="text" id="bookmark-name" placeholder="Bookmark name">
                <button onclick="createBookmark()">Save Bookmark</button>
                
                <div id="bookmark-list" class="bookmark-list">
                    <!-- Bookmarks will appear here -->
                </div>
            </div>
            
            <div class="analysis-section">
                <h4>Session Stats</h4>
                <div class="stats" id="session-stats">
                    Selected: 0 neurons<br>
                    Paths traced: 0<br>
                    Bookmarks: 0
                </div>
                <button onclick="exportSession()">Export Session</button>
            </div>
        </div>
    </div>

    <script>
        // Network data and configuration
        const networkData = {network_data};
        const config = {config};
        
        // Global variables
        let scene, camera, renderer, controls;
        let networkNodes = [], networkEdges = [];
        let selectedNeurons = new Set();
        let currentMode = 'explore';
        let tracedPaths = [];
        let bookmarks = [];
        
        // Raycaster for mouse interaction
        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2();
        
        // Initialize explorer
        function init() {{
            createScene();
            createNetworkVisualization();
            setupEventListeners();
            setupModeHandlers();
            
            console.log('Interactive explorer initialized');
        }}
        
        function createScene() {{
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x0a0a0a);
            
            camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
            camera.position.set(0, 0, 20);
            
            renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(window.innerWidth, window.innerHeight);
            document.getElementById('container').appendChild(renderer.domElement);
            
            // Lighting
            const ambientLight = new THREE.AmbientLight(0x404040, 0.4);
            scene.add(ambientLight);
            
            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
            directionalLight.position.set(10, 10, 5);
            scene.add(directionalLight);
            
            // Controls
            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
        }}
        
        function createNetworkVisualization() {{
            // Create neurons
            networkData.neurons.forEach((neuronData, index) => {{
                const geometry = new THREE.SphereGeometry(0.5, 16, 16);
                const material = new THREE.MeshPhongMaterial({{
                    color: getTypeColor(neuronData.type),
                    transparent: true,
                    opacity: 0.8
                }});
                
                const mesh = new THREE.Mesh(geometry, material);
                mesh.position.set(...neuronData.position);
                mesh.userData = neuronData;
                mesh.userData.originalColor = getTypeColor(neuronData.type);
                
                scene.add(mesh);
                networkNodes.push(mesh);
            }});
            
            // Create connections
            networkData.connections.forEach(connData => {{
                const sourcePos = networkData.neurons[connData.source].position;
                const targetPos = networkData.neurons[connData.target].position;
                
                const geometry = new THREE.BufferGeometry().setFromPoints([
                    new THREE.Vector3(...sourcePos),
                    new THREE.Vector3(...targetPos)
                ]);
                
                const material = new THREE.LineBasicMaterial({{
                    color: 0x444444,
                    transparent: true,
                    opacity: 0.3
                }});
                
                const line = new THREE.Line(geometry, material);
                line.userData = connData;
                
                scene.add(line);
                networkEdges.push(line);
            }});
            
            animate();
        }}
        
        function getTypeColor(type) {{
            const colors = {{
                'input': 0x4a90e2,
                'hidden': 0x7ed321,
                'output': 0xf5a623
            }};
            return colors[type] || 0x888888;
        }}
        
        function setupEventListeners() {{
            window.addEventListener('resize', onWindowResize, false);
            
            renderer.domElement.addEventListener('click', onMouseClick, false);
            renderer.domElement.addEventListener('mousemove', onMouseMove, false);
        }}
        
        function setupModeHandlers() {{
            document.querySelectorAll('.mode-btn').forEach(btn => {{
                btn.addEventListener('click', (e) => {{
                    switchMode(e.target.dataset.mode);
                }});
            }});
        }}
        
        function switchMode(mode) {{
            currentMode = mode;
            
            // Update UI
            document.querySelectorAll('.mode-btn').forEach(btn => {{
                btn.classList.remove('active');
            }});
            document.querySelector(`[data-mode="${{mode}}"]`).classList.add('active');
            
            // Show/hide relevant sections
            document.querySelectorAll('.analysis-section').forEach(section => {{
                if (section.id === `${{mode}}-section`) {{
                    section.style.display = 'block';
                }} else if (section.id.endsWith('-section')) {{
                    section.style.display = 'none';
                }}
            }});
            
            console.log(`Switched to ${{mode}} mode`);
        }}
        
        function onMouseClick(event) {{
            mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
            mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
            
            raycaster.setFromCamera(mouse, camera);
            const intersects = raycaster.intersectObjects(networkNodes);
            
            if (intersects.length > 0) {{
                const selectedObject = intersects[0].object;
                const neuronId = selectedObject.userData.id;
                
                if (event.ctrlKey || event.metaKey) {{
                    // Add to selection
                    if (selectedNeurons.has(neuronId)) {{
                        deselectNeuron(neuronId);
                    }} else {{
                        selectNeuron(neuronId, true);
                    }}
                }} else {{
                    // Single selection
                    clearSelection();
                    selectNeuron(neuronId);
                }}
            }}
        }}
        
        function onMouseMove(event) {{
            // Hover effects can be added here
        }}
        
        function selectNeuron(neuronId, addToSelection = false) {{
            if (!addToSelection) {{
                selectedNeurons.clear();
            }}
            
            selectedNeurons.add(neuronId);
            updateSelectionVisualization();
            updateSelectionInfo();
        }}
        
        function deselectNeuron(neuronId) {{
            selectedNeurons.delete(neuronId);
            updateSelectionVisualization();
            updateSelectionInfo();
        }}
        
        function clearSelection() {{
            selectedNeurons.clear();
            updateSelectionVisualization();
            updateSelectionInfo();
        }}
        
        function updateSelectionVisualization() {{
            networkNodes.forEach(node => {{
                const neuronId = node.userData.id;
                if (selectedNeurons.has(neuronId)) {{
                    node.material.color.setHex(0xff0000);
                    node.scale.set(1.5, 1.5, 1.5);
                }} else {{
                    node.material.color.setHex(node.userData.originalColor);
                    node.scale.set(1, 1, 1);
                }}
            }});
        }}
        
        function updateSelectionInfo() {{
            const infoDiv = document.getElementById('selection-info');
            
            if (selectedNeurons.size === 0) {{
                infoDiv.innerHTML = 'Click on neurons to select and analyze them';
            }} else if (selectedNeurons.size === 1) {{
                const neuronId = Array.from(selectedNeurons)[0];
                const neuronData = networkData.neurons[neuronId];
                infoDiv.innerHTML = `
                    <strong>Neuron ${{neuronId}}</strong><br>
                    Type: ${{neuronData.type}}<br>
                    Position: [${{neuronData.position.map(p => p.toFixed(1)).join(', ')}}]<br>
                    Properties: ${{Object.keys(neuronData.properties).length}} items
                `;
            }} else {{
                infoDiv.innerHTML = `
                    <strong>${{selectedNeurons.size}} neurons selected</strong><br>
                    Types: ${{getSelectionTypes()}}<br>
                    Ready for batch analysis
                `;
            }}
            
            updateSessionStats();
        }}
        
        function getSelectionTypes() {{
            const types = new Set();
            selectedNeurons.forEach(id => {{
                types.add(networkData.neurons[id].type);
            }});
            return Array.from(types).join(', ');
        }}
        
        function traceFromSelected() {{
            if (selectedNeurons.size === 0) {{
                alert('Please select a neuron first');
                return;
            }}
            
            const direction = document.getElementById('trace-direction').value;
            const depth = parseInt(document.getElementById('trace-depth').value);
            
            selectedNeurons.forEach(neuronId => {{
                const paths = tracePaths(neuronId, direction, depth);
                tracedPaths.push(...paths);
                displayTracedPaths(paths);
            }});
        }}
        
        function tracePaths(startNeuron, direction, maxDepth) {{
            // Simplified path tracing implementation
            const paths = [];
            const visited = new Set();
            
            function dfs(neuronId, currentPath, depth) {{
                if (depth >= maxDepth || visited.has(neuronId)) {{
                    return;
                }}
                
                visited.add(neuronId);
                currentPath.push(neuronId);
                
                // Find connections
                const connections = networkData.connections.filter(conn => {{
                    return (direction === 'forward' && conn.source === neuronId) ||
                           (direction === 'backward' && conn.target === neuronId) ||
                           (direction === 'both' && (conn.source === neuronId || conn.target === neuronId));
                }});
                
                connections.forEach(conn => {{
                    const nextNeuron = conn.source === neuronId ? conn.target : conn.source;
                    if (!visited.has(nextNeuron)) {{
                        dfs(nextNeuron, [...currentPath], depth + 1);
                    }}
                }});
                
                if (currentPath.length > 1) {{
                    paths.push([...currentPath]);
                }}
                
                visited.delete(neuronId);
            }}
            
            dfs(startNeuron, [], 0);
            return paths;
        }}
        
        function displayTracedPaths(paths) {{
            const pathsDiv = document.getElementById('traced-paths');
            
            paths.forEach((path, index) => {{
                const pathDiv = document.createElement('div');
                pathDiv.className = 'path-item';
                pathDiv.innerHTML = `Path ${{tracedPaths.length + index}}: ${{path.join(' -> ')}}`;
                pathsDiv.appendChild(pathDiv);
            }});
        }}
        
        function createBookmark() {{
            const name = document.getElementById('bookmark-name').value;
            if (!name) {{
                alert('Please enter a bookmark name');
                return;
            }}
            
            const bookmark = {{
                id: `bookmark_${{Date.now()}}`,
                name: name,
                selectedNeurons: Array.from(selectedNeurons),
                timestamp: new Date().toISOString()
            }};
            
            bookmarks.push(bookmark);
            displayBookmarks();
            
            document.getElementById('bookmark-name').value = '';
        }}
        
        function displayBookmarks() {{
            const bookmarkList = document.getElementById('bookmark-list');
            bookmarkList.innerHTML = '';
            
            bookmarks.forEach(bookmark => {{
                const bookmarkDiv = document.createElement('div');
                bookmarkDiv.className = 'bookmark-item';
                bookmarkDiv.innerHTML = `
                    <strong>${{bookmark.name}}</strong><br>
                    ${{bookmark.selectedNeurons.length}} neurons<br>
                    <small>${{new Date(bookmark.timestamp).toLocaleString()}}</small>
                `;
                bookmarkDiv.onclick = () => loadBookmark(bookmark);
                bookmarkList.appendChild(bookmarkDiv);
            }});
        }}
        
        function loadBookmark(bookmark) {{
            clearSelection();
            bookmark.selectedNeurons.forEach(id => selectNeuron(id, true));
            console.log(`Loaded bookmark: ${{bookmark.name}}`);
        }}
        
        function updateSessionStats() {{
            const statsDiv = document.getElementById('session-stats');
            statsDiv.innerHTML = `
                Selected: ${{selectedNeurons.size}} neurons<br>
                Paths traced: ${{tracedPaths.length}}<br>
                Bookmarks: ${{bookmarks.length}}
            `;
        }}
        
        function exportSession() {{
            const sessionData = {{
                selectedNeurons: Array.from(selectedNeurons),
                tracedPaths: tracedPaths,
                bookmarks: bookmarks,
                timestamp: new Date().toISOString()
            }};
            
            const dataStr = JSON.stringify(sessionData, null, 2);
            const dataBlob = new Blob([dataStr], {{type: 'application/json'}});
            const url = URL.createObjectURL(dataBlob);
            
            const link = document.createElement('a');
            link.href = url;
            link.download = 'exploration_session.json';
            link.click();
            
            URL.revokeObjectURL(url);
        }}
        
        function onWindowResize() {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }}
        
        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }}
        
        // Placeholder functions for analysis and manipulation
        function analyzeInfluence() {{ console.log('Analyze influence'); }}
        function findConcepts() {{ console.log('Find concepts'); }}
        function calculateCentrality() {{ console.log('Calculate centrality'); }}
        function strengthenConnections() {{ console.log('Strengthen connections'); }}
        function weakenConnections() {{ console.log('Weaken connections'); }}
        function injectActivation() {{ console.log('Inject activation'); }}
        function disableNeurons() {{ console.log('Disable neurons'); }}
        
        // Initialize when page loads
        window.addEventListener('load', init);
    </script>
</body>
</html>
        '''


class PathAnalyzer:
    """Analyze causal paths in the network"""
    
    def __init__(self, network):
        self.network = network
    
    def trace_paths(self, start_neuron: int, direction: str = 'forward', 
                   max_depth: int = 5, min_strength: float = 0.1) -> List[List[int]]:
        """Trace causal paths from a starting neuron"""
        paths = []
        visited = set()
        
        def dfs(neuron_id: int, current_path: List[int], depth: int):
            if depth >= max_depth or neuron_id in visited:
                return
            
            visited.add(neuron_id)
            current_path.append(neuron_id)
            
            # Get connections based on direction
            connections = self._get_connections(neuron_id, direction)
            
            for target_id, strength in connections:
                if strength >= min_strength and target_id not in visited:
                    dfs(target_id, current_path.copy(), depth + 1)
            
            if len(current_path) > 1:
                paths.append(current_path.copy())
            
            visited.remove(neuron_id)
        
        dfs(start_neuron, [], 0)
        return paths
    
    def _get_connections(self, neuron_id: int, direction: str) -> List[Tuple[int, float]]:
        """Get connections for a neuron in specified direction"""
        connections = []
        
        if hasattr(self.network, 'neurons') and neuron_id < len(self.network.neurons):
            neuron = self.network.neurons[neuron_id]
            
            if direction in ['forward', 'both'] and hasattr(neuron, 'neighbors'):
                for neighbor_id in neuron.neighbors:
                    if neighbor_id < len(self.network.neurons):
                        strength = np.random.uniform(0.1, 1.0)  # Placeholder
                        connections.append((neighbor_id, strength))
        
        return connections


class ConceptMapper:
    """Map and analyze concept relationships"""
    
    def __init__(self, network):
        self.network = network
    
    def analyze_concept_relationships(self, concept_name: str) -> Dict[str, Any]:
        """Analyze relationships for a specific concept"""
        analysis = {
            'concept_name': concept_name,
            'related_neurons': [],
            'concept_strength': 0.0,
            'related_concepts': [],
            'concept_clusters': []
        }
        
        # Find neurons related to concept
        for i in range(self.network.config.num_neurons):
            if self._is_neuron_related_to_concept(i, concept_name):
                analysis['related_neurons'].append(i)
        
        # Calculate concept strength
        if analysis['related_neurons']:
            analysis['concept_strength'] = len(analysis['related_neurons']) / self.network.config.num_neurons
        
        return analysis
    
    def analyze_neuron_concepts(self, neuron_id: int) -> Dict[str, Any]:
        """Analyze concept associations for a neuron"""
        analysis = {
            'neuron_id': neuron_id,
            'primary_concepts': [],
            'concept_strengths': {},
            'concept_relationships': []
        }
        
        # Placeholder implementation
        if hasattr(self.network, 'neurons') and neuron_id < len(self.network.neurons):
            neuron = self.network.neurons[neuron_id]
            concept_type = getattr(neuron, 'concept_type', 'unknown')
            
            if concept_type != 'unknown':
                analysis['primary_concepts'].append(concept_type)
                analysis['concept_strengths'][concept_type] = np.random.uniform(0.5, 1.0)
        
        return analysis
    
    def _is_neuron_related_to_concept(self, neuron_id: int, concept_name: str) -> bool:
        """Check if neuron is related to a concept"""
        if hasattr(self.network, 'neurons') and neuron_id < len(self.network.neurons):
            neuron = self.network.neurons[neuron_id]
            neuron_concept = getattr(neuron, 'concept_type', None)
            return neuron_concept == concept_name
        return False


class InfluenceAnalyzer:
    """Analyze influence patterns in the network"""
    
    def __init__(self, network):
        self.network = network
    
    def analyze_local_influence(self, target_neuron: int, radius: int = 2) -> Dict[str, Any]:
        """Analyze influence within a local radius"""
        analysis = {
            'target_neuron': target_neuron,
            'radius': radius,
            'influenced_neurons': [],
            'influence_strengths': {},
            'total_influence': 0.0
        }
        
        # Find neurons within radius
        for i in range(self.network.config.num_neurons):
            if i != target_neuron:
                distance = self._calculate_influence_distance(target_neuron, i)
                if distance <= radius:
                    influence_strength = self._calculate_influence_strength(target_neuron, i)
                    if influence_strength > 0.05:  # Threshold
                        analysis['influenced_neurons'].append(i)
                        analysis['influence_strengths'][i] = influence_strength
                        analysis['total_influence'] += influence_strength
        
        return analysis
    
    def analyze_neuron_influence(self, neuron_id: int) -> Dict[str, Any]:
        """Analyze influence patterns for a specific neuron"""
        analysis = {
            'neuron_id': neuron_id,
            'outgoing_influence': 0.0,
            'incoming_influence': 0.0,
            'influence_centrality': 0.0,
            'influence_targets': [],
            'influence_sources': []
        }
        
        # Calculate influence metrics
        for i in range(self.network.config.num_neurons):
            if i != neuron_id:
                # Outgoing influence
                outgoing = self._calculate_influence_strength(neuron_id, i)
                if outgoing > 0.05:
                    analysis['outgoing_influence'] += outgoing
                    analysis['influence_targets'].append({'neuron': i, 'strength': outgoing})
                
                # Incoming influence
                incoming = self._calculate_influence_strength(i, neuron_id)
                if incoming > 0.05:
                    analysis['incoming_influence'] += incoming
                    analysis['influence_sources'].append({'neuron': i, 'strength': incoming})
        
        # Calculate centrality
        total_connections = len(analysis['influence_targets']) + len(analysis['influence_sources'])
        analysis['influence_centrality'] = total_connections / max(1, self.network.config.num_neurons - 1)
        
        return analysis
    
    def _calculate_influence_distance(self, source: int, target: int) -> float:
        """Calculate influence distance between neurons"""
        # Placeholder implementation using network topology
        return np.random.uniform(1, 5)
    
    def _calculate_influence_strength(self, source: int, target: int) -> float:
        """Calculate influence strength between neurons"""
        # Placeholder implementation
        return np.random.uniform(0, 1)


class ExploreMode:
    """Exploration mode for network navigation"""
    
    def __init__(self, explorer):
        self.explorer = explorer
    
    def activate(self):
        """Activate exploration mode"""
        print("Exploration mode activated - Click neurons to select and explore")


class TraceMode:
    """Path tracing mode"""
    
    def __init__(self, explorer):
        self.explorer = explorer
    
    def activate(self):
        """Activate trace mode"""
        print("Trace mode activated - Select neurons to trace causal paths")


class AnalyzeMode:
    """Analysis mode for detailed network analysis"""
    
    def __init__(self, explorer):
        self.explorer = explorer
    
    def activate(self):
        """Activate analysis mode"""
        print("Analysis mode activated - Advanced analysis tools available")


class ManipulateMode:
    """Network manipulation mode"""
    
    def __init__(self, explorer):
        self.explorer = explorer
    
    def activate(self):
        """Activate manipulation mode"""
        print("Manipulation mode activated - Network manipulation tools available")
    
    def strengthen_connection(self, source: int, target: int, factor: float = 1.5) -> Dict[str, Any]:
        """Strengthen connection between neurons"""
        result = {
            'action': 'strengthen_connection',
            'source': source,
            'target': target,
            'factor': factor,
            'success': True,
            'original_strength': np.random.uniform(0.1, 1.0),  # Placeholder
            'new_strength': None
        }
        
        result['new_strength'] = min(1.0, result['original_strength'] * factor)
        print(f"Strengthened connection {source} -> {target} by factor {factor}")
        
        return result
    
    def weaken_connection(self, source: int, target: int, factor: float = 0.5) -> Dict[str, Any]:
        """Weaken connection between neurons"""
        result = {
            'action': 'weaken_connection',
            'source': source,
            'target': target,
            'factor': factor,
            'success': True,
            'original_strength': np.random.uniform(0.1, 1.0),  # Placeholder
            'new_strength': None
        }
        
        result['new_strength'] = max(0.0, result['original_strength'] * factor)
        print(f"Weakened connection {source} -> {target} by factor {factor}")
        
        return result
    
    def inject_activation(self, neuron_id: int, activation_value: float) -> Dict[str, Any]:
        """Inject activation into a neuron"""
        result = {
            'action': 'inject_activation',
            'neuron_id': neuron_id,
            'activation_value': activation_value,
            'success': True,
            'previous_activation': np.random.uniform(0, 1)  # Placeholder
        }
        
        print(f"Injected activation {activation_value} into neuron {neuron_id}")
        
        return result
    
    def disable_neuron(self, neuron_id: int) -> Dict[str, Any]:
        """Disable a neuron"""
        result = {
            'action': 'disable_neuron',
            'neuron_id': neuron_id,
            'success': True,
            'previous_state': 'active'
        }
        
        print(f"Disabled neuron {neuron_id}")
        
        return result