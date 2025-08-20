"""
WebGL-Based 3D Network Renderer for AGI-Formula

High-performance 3D visualization using WebGL and Three.js:
- Real-time network topology rendering
- Interactive neuron exploration
- Dynamic causal chain visualization
- Live attention flow patterns
"""

import json
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import base64
from datetime import datetime


class WebGLRenderer:
    """
    Advanced WebGL-based 3D renderer for AGI networks
    
    Features:
    - High-performance 3D network visualization
    - Real-time updates and animations
    - Interactive controls and exploration
    - Customizable visual themes and layouts
    """
    
    def __init__(self, network, config: Optional[Dict[str, Any]] = None):
        self.network = network
        self.config = config or self._get_default_config()
        
        # Rendering state
        self.scene_data = {}
        self.animation_frames = []
        self.layout_cache = {}
        
        # Initialize scene
        self._initialize_scene()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default rendering configuration"""
        return {
            'canvas_width': 1200,
            'canvas_height': 800,
            'background_color': 0x0a0a0a,
            'node_size_scale': 1.0,
            'edge_width_scale': 1.0,
            'animation_speed': 1.0,
            'physics_enabled': True,
            'auto_layout': True,
            'quality_settings': 'high',  # low, medium, high, ultra
            'theme': 'dark',  # dark, light, neon, scientific
            'camera_controls': True,
            'real_time_updates': True
        }
    
    def _initialize_scene(self):
        """Initialize 3D scene data"""
        self.scene_data = {
            'metadata': {
                'created': datetime.now().isoformat(),
                'network_id': getattr(self.network, 'id', 'unknown'),
                'num_neurons': self.network.config.num_neurons,
                'renderer_config': self.config
            },
            'nodes': [],
            'edges': [],
            'groups': [],
            'layouts': {},
            'animations': [],
            'styles': self._get_visual_styles()
        }
        
        self._generate_network_geometry()
    
    def _get_visual_styles(self) -> Dict[str, Any]:
        """Get visual styling configuration"""
        theme = self.config['theme']
        
        themes = {
            'dark': {
                'background': 0x0a0a0a,
                'nodes': {
                    'input': {'color': 0x4a90e2, 'emissive': 0x001a33},
                    'hidden': {'color': 0x7ed321, 'emissive': 0x1a3300},
                    'output': {'color': 0xf5a623, 'emissive': 0x331f00},
                    'composite': {'color': 0xd0021b, 'emissive': 0x330005}
                },
                'edges': {
                    'default': {'color': 0x444444, 'opacity': 0.6},
                    'causal': {'color': 0xff6b35, 'opacity': 0.8},
                    'attention': {'color': 0x35ff6b, 'opacity': 0.9}
                },
                'particles': {
                    'activation': {'color': 0xffffff, 'size': 2.0},
                    'signal': {'color': 0x35a3ff, 'size': 1.5}
                }
            },
            'light': {
                'background': 0xf5f5f5,
                'nodes': {
                    'input': {'color': 0x2196f3, 'emissive': 0x000000},
                    'hidden': {'color': 0x4caf50, 'emissive': 0x000000},
                    'output': {'color': 0xff9800, 'emissive': 0x000000},
                    'composite': {'color': 0xe91e63, 'emissive': 0x000000}
                },
                'edges': {
                    'default': {'color': 0x666666, 'opacity': 0.7},
                    'causal': {'color': 0xff5722, 'opacity': 0.9},
                    'attention': {'color': 0x4caf50, 'opacity': 0.9}
                },
                'particles': {
                    'activation': {'color': 0x333333, 'size': 2.0},
                    'signal': {'color': 0x2196f3, 'size': 1.5}
                }
            },
            'neon': {
                'background': 0x000000,
                'nodes': {
                    'input': {'color': 0x00ffff, 'emissive': 0x003333},
                    'hidden': {'color': 0xff00ff, 'emissive': 0x330033},
                    'output': {'color': 0xffff00, 'emissive': 0x333300},
                    'composite': {'color': 0xff0040, 'emissive': 0x330008}
                },
                'edges': {
                    'default': {'color': 0x004080, 'opacity': 0.8},
                    'causal': {'color': 0xff0080, 'opacity': 1.0},
                    'attention': {'color': 0x80ff00, 'opacity': 1.0}
                },
                'particles': {
                    'activation': {'color': 0xffffff, 'size': 3.0},
                    'signal': {'color': 0x00ffff, 'size': 2.0}
                }
            }
        }
        
        return themes.get(theme, themes['dark'])
    
    def _generate_network_geometry(self):
        """Generate 3D geometry for network nodes and edges"""
        # Generate node positions using force-directed layout
        positions = self._calculate_3d_layout()
        
        # Create node data
        for i in range(self.network.config.num_neurons):
            node_type = self._get_neuron_type(i)
            
            node_data = {
                'id': i,
                'type': node_type,
                'position': positions[i].tolist(),
                'size': self._calculate_node_size(i),
                'activation': 0.0,
                'style': self.scene_data['styles']['nodes'][node_type],
                'metadata': self._get_node_metadata(i)
            }
            
            self.scene_data['nodes'].append(node_data)
        
        # Create edge data
        edges = self._get_network_edges()
        for edge in edges:
            edge_data = {
                'source': edge['source'],
                'target': edge['target'],
                'weight': edge['weight'],
                'type': edge.get('type', 'default'),
                'style': self.scene_data['styles']['edges'][edge.get('type', 'default')],
                'animated': edge.get('animated', False)
            }
            
            self.scene_data['edges'].append(edge_data)
    
    def _calculate_3d_layout(self) -> np.ndarray:
        """Calculate 3D positions for neurons using force-directed algorithm"""
        num_neurons = self.network.config.num_neurons
        
        # Initialize random positions
        positions = np.random.uniform(-10, 10, (num_neurons, 3))
        
        # Apply force-directed layout
        if self.config['auto_layout']:
            positions = self._force_directed_layout_3d(positions)
        
        # Apply special layouts for different neuron types
        positions = self._apply_type_based_layout(positions)
        
        return positions
    
    def _force_directed_layout_3d(self, positions: np.ndarray, iterations: int = 100) -> np.ndarray:
        """3D force-directed layout algorithm"""
        num_neurons = len(positions)
        
        # Get connection matrix
        connections = self._get_connection_matrix()
        
        # Force-directed simulation
        for iteration in range(iterations):
            forces = np.zeros_like(positions)
            
            # Repulsive forces between all nodes
            for i in range(num_neurons):
                for j in range(i + 1, num_neurons):
                    diff = positions[i] - positions[j]
                    dist = np.linalg.norm(diff)
                    
                    if dist > 0:
                        force_magnitude = 1.0 / (dist ** 2)
                        force_direction = diff / dist
                        
                        forces[i] += force_direction * force_magnitude
                        forces[j] -= force_direction * force_magnitude
            
            # Attractive forces for connected nodes
            for i in range(num_neurons):
                for j in range(num_neurons):
                    if connections[i, j] > 0:
                        diff = positions[j] - positions[i]
                        dist = np.linalg.norm(diff)
                        
                        if dist > 0:
                            force_magnitude = connections[i, j] * dist * 0.1
                            force_direction = diff / dist
                            
                            forces[i] += force_direction * force_magnitude
            
            # Apply forces with damping
            damping = 0.9 - (iteration / iterations) * 0.5
            positions += forces * 0.01 * damping
        
        return positions
    
    def _apply_type_based_layout(self, positions: np.ndarray) -> np.ndarray:
        """Apply layout constraints based on neuron types"""
        # Input neurons in bottom layer
        for i in range(self.network.config.input_size):
            positions[i, 2] = -5  # Bottom layer
            positions[i, 0] = (i - self.network.config.input_size/2) * 2
        
        # Output neurons in top layer
        output_start = self.network.config.num_neurons - self.network.config.output_size
        for i in range(output_start, self.network.config.num_neurons):
            positions[i, 2] = 5  # Top layer
            idx = i - output_start
            positions[i, 0] = (idx - self.network.config.output_size/2) * 2
        
        return positions
    
    def _get_neuron_type(self, neuron_id: int) -> str:
        """Get neuron type for styling"""
        if neuron_id < self.network.config.input_size:
            return 'input'
        elif neuron_id >= self.network.config.num_neurons - self.network.config.output_size:
            return 'output'
        elif hasattr(self.network, 'neurons') and neuron_id < len(self.network.neurons):
            neuron = self.network.neurons[neuron_id]
            if getattr(neuron, 'composite', False):
                return 'composite'
        
        return 'hidden'
    
    def _calculate_node_size(self, neuron_id: int) -> float:
        """Calculate node size based on properties"""
        base_size = 1.0 * self.config['node_size_scale']
        
        # Size based on neuron type
        neuron_type = self._get_neuron_type(neuron_id)
        type_multipliers = {
            'input': 1.2,
            'output': 1.4,
            'hidden': 1.0,
            'composite': 1.6
        }
        
        return base_size * type_multipliers.get(neuron_type, 1.0)
    
    def _get_node_metadata(self, neuron_id: int) -> Dict[str, Any]:
        """Get metadata for neuron"""
        metadata = {
            'id': neuron_id,
            'type': self._get_neuron_type(neuron_id)
        }
        
        # Add neuron-specific metadata if available
        if hasattr(self.network, 'neurons') and neuron_id < len(self.network.neurons):
            neuron = self.network.neurons[neuron_id]
            metadata.update({
                'concept_type': getattr(neuron, 'concept_type', 'unknown'),
                'activation_function': getattr(neuron, 'activation_function', 'sigmoid'),
                'num_connections': len(getattr(neuron, 'neighbors', []))
            })
        
        return metadata
    
    def _get_network_edges(self) -> List[Dict[str, Any]]:
        """Get network edge data"""
        edges = []
        
        if hasattr(self.network, 'neurons'):
            for i, neuron in enumerate(self.network.neurons):
                if hasattr(neuron, 'neighbors'):
                    for neighbor_id in neuron.neighbors:
                        if neighbor_id < len(self.network.neurons):
                            weight = np.random.uniform(0.1, 1.0)  # Placeholder
                            
                            edges.append({
                                'source': i,
                                'target': neighbor_id,
                                'weight': weight,
                                'type': 'default'
                            })
        
        return edges
    
    def _get_connection_matrix(self) -> np.ndarray:
        """Get connection matrix for layout algorithm"""
        num_neurons = self.network.config.num_neurons
        matrix = np.zeros((num_neurons, num_neurons))
        
        edges = self._get_network_edges()
        for edge in edges:
            matrix[edge['source'], edge['target']] = edge['weight']
        
        return matrix
    
    def update_activations(self, activations: np.ndarray):
        """Update neuron activations for visualization"""
        if len(activations) != len(self.scene_data['nodes']):
            raise ValueError("Activation array size doesn't match number of nodes")
        
        for i, activation in enumerate(activations):
            self.scene_data['nodes'][i]['activation'] = float(activation)
    
    def update_causal_chains(self, causal_data: Dict[str, Any]):
        """Update causal chain visualization"""
        # Clear existing causal edges
        self.scene_data['edges'] = [e for e in self.scene_data['edges'] if e['type'] != 'causal']
        
        # Add new causal edges
        for chain in causal_data.get('chains', []):
            for i in range(len(chain) - 1):
                source = chain[i]['neuron_id']
                target = chain[i + 1]['neuron_id']
                strength = chain[i + 1].get('contribution', 0.5)
                
                causal_edge = {
                    'source': source,
                    'target': target,
                    'weight': strength,
                    'type': 'causal',
                    'style': self.scene_data['styles']['edges']['causal'],
                    'animated': True
                }
                
                self.scene_data['edges'].append(causal_edge)
    
    def update_attention_flows(self, attention_data: Dict[str, Any]):
        """Update attention flow visualization"""
        # Clear existing attention edges
        self.scene_data['edges'] = [e for e in self.scene_data['edges'] if e['type'] != 'attention']
        
        # Add new attention edges
        for flow in attention_data.get('flows', []):
            source = flow['source']
            target = flow['target']
            strength = flow.get('strength', 0.5)
            
            attention_edge = {
                'source': source,
                'target': target,
                'weight': strength,
                'type': 'attention',
                'style': self.scene_data['styles']['edges']['attention'],
                'animated': True
            }
            
            self.scene_data['edges'].append(attention_edge)
    
    def create_animation_frame(self, timestamp: Optional[str] = None) -> Dict[str, Any]:
        """Create animation frame for current state"""
        frame = {
            'timestamp': timestamp or datetime.now().isoformat(),
            'activations': [node['activation'] for node in self.scene_data['nodes']],
            'active_edges': [
                {'source': e['source'], 'target': e['target'], 'strength': e['weight']}
                for e in self.scene_data['edges'] 
                if e.get('animated', False)
            ]
        }
        
        self.animation_frames.append(frame)
        return frame
    
    def generate_webgl_html(self, output_file: str = "agi_network_3d.html") -> str:
        """Generate complete WebGL visualization HTML"""
        html_template = self._get_html_template()
        
        # Embed scene data
        scene_json = json.dumps(self.scene_data, indent=2, default=str)
        
        # Generate complete HTML
        html_content = html_template.format(
            scene_data=scene_json,
            canvas_width=self.config['canvas_width'],
            canvas_height=self.config['canvas_height'],
            background_color=hex(self.config['background_color']),
            animation_speed=self.config['animation_speed'],
            title=f"AGI Network 3D Visualization - {self.network.config.num_neurons} neurons"
        )
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"WebGL visualization saved to: {output_file}")
        return output_file
    
    def _get_html_template(self) -> str:
        """Get HTML template with embedded Three.js visualization"""
        return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            margin: 0;
            padding: 0;
            background: #000;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: white;
            overflow: hidden;
        }}
        
        #container {{
            position: relative;
            width: 100vw;
            height: 100vh;
        }}
        
        #canvas {{
            display: block;
        }}
        
        #ui-panel {{
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(0, 0, 0, 0.8);
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #333;
            min-width: 250px;
            max-height: 80vh;
            overflow-y: auto;
        }}
        
        .control-group {{
            margin-bottom: 15px;
        }}
        
        .control-group label {{
            display: block;
            margin-bottom: 5px;
            font-size: 12px;
            color: #ccc;
        }}
        
        .control-group input, .control-group select {{
            width: 100%;
            padding: 5px;
            background: #222;
            border: 1px solid #444;
            color: white;
            border-radius: 3px;
        }}
        
        #info-panel {{
            position: absolute;
            bottom: 10px;
            right: 10px;
            background: rgba(0, 0, 0, 0.8);
            padding: 10px;
            border-radius: 8px;
            border: 1px solid #333;
            font-size: 12px;
        }}
        
        .stats {{
            margin: 5px 0;
        }}
        
        #loading {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            text-align: center;
            font-size: 18px;
        }}
        
        .hidden {{
            display: none;
        }}
        
        button {{
            background: #0066cc;
            color: white;
            border: none;
            padding: 8px 12px;
            border-radius: 4px;
            cursor: pointer;
            margin: 2px;
        }}
        
        button:hover {{
            background: #0080ff;
        }}
        
        button:disabled {{
            background: #666;
            cursor: not-allowed;
        }}
    </style>
</head>
<body>
    <div id="container">
        <div id="loading">
            <div>Loading AGI Network Visualization...</div>
            <div style="margin-top: 10px; color: #666;">Initializing WebGL renderer</div>
        </div>
        
        <div id="ui-panel" class="hidden">
            <h3 style="margin-top: 0;">AGI Network Controls</h3>
            
            <div class="control-group">
                <label>Animation Speed</label>
                <input type="range" id="animation-speed" min="0.1" max="3.0" step="0.1" value="{animation_speed}">
                <span id="speed-value">{animation_speed}</span>
            </div>
            
            <div class="control-group">
                <label>Node Size</label>
                <input type="range" id="node-size" min="0.5" max="3.0" step="0.1" value="1.0">
                <span id="size-value">1.0</span>
            </div>
            
            <div class="control-group">
                <label>Visualization Mode</label>
                <select id="viz-mode">
                    <option value="network">Network Topology</option>
                    <option value="activations">Live Activations</option>
                    <option value="causal">Causal Chains</option>
                    <option value="attention">Attention Flow</option>
                </select>
            </div>
            
            <div class="control-group">
                <button id="play-pause">⏸️ Pause</button>
                <button id="reset-view">🏠 Reset View</button>
                <button id="screenshot">📷 Screenshot</button>
            </div>
            
            <div class="control-group">
                <label>Selected Neuron</label>
                <div id="selected-neuron">None</div>
            </div>
        </div>
        
        <div id="info-panel" class="hidden">
            <div class="stats">FPS: <span id="fps">0</span></div>
            <div class="stats">Neurons: <span id="neuron-count">0</span></div>
            <div class="stats">Connections: <span id="edge-count">0</span></div>
            <div class="stats">Rendered: <span id="rendered-objects">0</span></div>
        </div>
    </div>

    <!-- Three.js CDN -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/dat-gui/0.7.9/dat.gui.min.js"></script>
    
    <script>
        // Scene data embedded from Python
        const sceneData = {scene_data};
        
        // Global variables
        let scene, camera, renderer, controls;
        let networkNodes = [], networkEdges = [];
        let animationId;
        let isPlaying = true;
        let selectedNeuron = null;
        
        // Performance monitoring
        let frameCount = 0;
        let lastTime = performance.now();
        
        // Initialize visualization
        function init() {{
            // Create scene
            scene = new THREE.Scene();
            scene.background = new THREE.Color({background_color});
            
            // Create camera
            camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
            camera.position.set(0, 0, 20);
            
            // Create renderer
            renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.shadowMap.enabled = true;
            renderer.shadowMap.type = THREE.PCFSoftShadowMap;
            
            document.getElementById('container').appendChild(renderer.domElement);
            
            // Add lighting
            setupLighting();
            
            // Create network geometry
            createNetworkVisualization();
            
            // Setup controls
            setupControls();
            
            // Setup event listeners
            setupEventListeners();
            
            // Hide loading, show UI
            document.getElementById('loading').classList.add('hidden');
            document.getElementById('ui-panel').classList.remove('hidden');
            document.getElementById('info-panel').classList.remove('hidden');
            
            // Start animation loop
            animate();
        }}
        
        function setupLighting() {{
            // Ambient light
            const ambientLight = new THREE.AmbientLight(0x404040, 0.4);
            scene.add(ambientLight);
            
            // Directional light
            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
            directionalLight.position.set(10, 10, 5);
            directionalLight.castShadow = true;
            scene.add(directionalLight);
            
            // Point lights for effects
            const pointLight1 = new THREE.PointLight(0x0066ff, 0.5, 100);
            pointLight1.position.set(-10, 10, 10);
            scene.add(pointLight1);
            
            const pointLight2 = new THREE.PointLight(0xff6600, 0.5, 100);
            pointLight2.position.set(10, -10, 10);
            scene.add(pointLight2);
        }}
        
        function createNetworkVisualization() {{
            createNodes();
            createEdges();
            updateStats();
        }}
        
        function createNodes() {{
            sceneData.nodes.forEach((nodeData, index) => {{
                const geometry = new THREE.SphereGeometry(nodeData.size, 16, 16);
                const material = new THREE.MeshPhongMaterial({{
                    color: nodeData.style.color,
                    emissive: nodeData.style.emissive,
                    transparent: true,
                    opacity: 0.9
                }});
                
                const mesh = new THREE.Mesh(geometry, material);
                mesh.position.set(...nodeData.position);
                mesh.castShadow = true;
                mesh.receiveShadow = true;
                
                // Store node data
                mesh.userData = nodeData;
                mesh.userData.originalColor = nodeData.style.color;
                mesh.userData.index = index;
                
                scene.add(mesh);
                networkNodes.push(mesh);
            }});
        }}
        
        function createEdges() {{
            sceneData.edges.forEach(edgeData => {{
                const sourcePos = sceneData.nodes[edgeData.source].position;
                const targetPos = sceneData.nodes[edgeData.target].position;
                
                const geometry = new THREE.BufferGeometry().setFromPoints([
                    new THREE.Vector3(...sourcePos),
                    new THREE.Vector3(...targetPos)
                ]);
                
                const material = new THREE.LineBasicMaterial({{
                    color: edgeData.style.color,
                    transparent: true,
                    opacity: edgeData.style.opacity * edgeData.weight
                }});
                
                const line = new THREE.Line(geometry, material);
                line.userData = edgeData;
                
                scene.add(line);
                networkEdges.push(line);
            }});
        }}
        
        function setupControls() {{
            // Mouse controls for camera
            const controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            
            // UI controls
            document.getElementById('animation-speed').addEventListener('input', (e) => {{
                const speed = parseFloat(e.target.value);
                document.getElementById('speed-value').textContent = speed;
                // Update animation speed
            }});
            
            document.getElementById('node-size').addEventListener('input', (e) => {{
                const size = parseFloat(e.target.value);
                document.getElementById('size-value').textContent = size;
                updateNodeSizes(size);
            }});
            
            document.getElementById('play-pause').addEventListener('click', () => {{
                isPlaying = !isPlaying;
                document.getElementById('play-pause').textContent = isPlaying ? '⏸️ Pause' : '▶️ Play';
            }});
            
            document.getElementById('reset-view').addEventListener('click', () => {{
                camera.position.set(0, 0, 20);
                controls.reset();
            }});
            
            document.getElementById('screenshot').addEventListener('click', () => {{
                takeScreenshot();
            }});
        }}
        
        function setupEventListeners() {{
            // Window resize
            window.addEventListener('resize', onWindowResize, false);
            
            // Mouse interaction
            const raycaster = new THREE.Raycaster();
            const mouse = new THREE.Vector2();
            
            renderer.domElement.addEventListener('click', (event) => {{
                mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
                mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
                
                raycaster.setFromCamera(mouse, camera);
                const intersects = raycaster.intersectObjects(networkNodes);
                
                if (intersects.length > 0) {{
                    selectNeuron(intersects[0].object);
                }}
            }});
        }}
        
        function selectNeuron(neuronMesh) {{
            // Deselect previous
            if (selectedNeuron) {{
                selectedNeuron.material.color.setHex(selectedNeuron.userData.originalColor);
                selectedNeuron.scale.set(1, 1, 1);
            }}
            
            // Select new
            selectedNeuron = neuronMesh;
            selectedNeuron.material.color.setHex(0xffff00); // Highlight in yellow
            selectedNeuron.scale.set(1.5, 1.5, 1.5);
            
            // Update UI
            const neuronData = selectedNeuron.userData;
            document.getElementById('selected-neuron').innerHTML = `
                <strong>Neuron ${{neuronData.id}}</strong><br>
                Type: ${{neuronData.type}}<br>
                Activation: ${{neuronData.activation.toFixed(3)}}<br>
                Connections: ${{neuronData.metadata.num_connections || 0}}
            `;
        }}
        
        function updateNodeSizes(scale) {{
            networkNodes.forEach(node => {{
                const originalSize = node.userData.size;
                node.scale.set(scale, scale, scale);
            }});
        }}
        
        function updateActivations() {{
            // Simulate activation updates
            networkNodes.forEach(node => {{
                const activation = node.userData.activation;
                const intensity = Math.max(0.3, activation);
                
                // Update emissive intensity based on activation
                node.material.emissive.setHex(node.userData.style.emissive);
                node.material.emissiveIntensity = intensity;
                
                // Add pulsing effect for high activation
                if (activation > 0.8) {{
                    const pulse = Math.sin(Date.now() * 0.01) * 0.2 + 0.8;
                    node.scale.set(pulse, pulse, pulse);
                }}
            }});
        }}
        
        function updateStats() {{
            document.getElementById('neuron-count').textContent = networkNodes.length;
            document.getElementById('edge-count').textContent = networkEdges.length;
            document.getElementById('rendered-objects').textContent = scene.children.length;
        }}
        
        function updateFPS() {{
            frameCount++;
            const currentTime = performance.now();
            
            if (currentTime - lastTime >= 1000) {{
                const fps = Math.round(frameCount * 1000 / (currentTime - lastTime));
                document.getElementById('fps').textContent = fps;
                frameCount = 0;
                lastTime = currentTime;
            }}
        }}
        
        function takeScreenshot() {{
            const canvas = renderer.domElement;
            const link = document.createElement('a');
            link.download = 'agi_network_screenshot.png';
            link.href = canvas.toDataURL();
            link.click();
        }}
        
        function onWindowResize() {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }}
        
        function animate() {{
            animationId = requestAnimationFrame(animate);
            
            if (isPlaying) {{
                updateActivations();
                updateFPS();
            }}
            
            renderer.render(scene, camera);
        }}
        
        // Initialize when page loads
        window.addEventListener('load', init);
    </script>
</body>
</html>
        '''
    
    def export_scene_data(self, output_file: str = "scene_data.json") -> str:
        """Export scene data as JSON"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.scene_data, f, indent=2, default=str)
        
        return output_file
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get rendering performance metrics"""
        return {
            'nodes_count': len(self.scene_data['nodes']),
            'edges_count': len(self.scene_data['edges']),
            'animation_frames': len(self.animation_frames),
            'scene_complexity': self._calculate_scene_complexity(),
            'estimated_render_time': self._estimate_render_time()
        }
    
    def _calculate_scene_complexity(self) -> float:
        """Calculate scene complexity score"""
        nodes = len(self.scene_data['nodes'])
        edges = len(self.scene_data['edges'])
        
        # Simple complexity metric
        complexity = (nodes * 0.1) + (edges * 0.05)
        
        return min(10.0, complexity)  # Cap at 10
    
    def _estimate_render_time(self) -> float:
        """Estimate rendering time in milliseconds"""
        complexity = self._calculate_scene_complexity()
        
        # Simple estimation based on complexity
        base_time = 16.67  # 60 FPS target
        return base_time * (1 + complexity * 0.1)