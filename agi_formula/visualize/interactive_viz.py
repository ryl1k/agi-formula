"""
Interactive Visualization Tools for AGI-Formula

Provides real-time, interactive visualizations of network behavior including:
- Network topology and connections
- Causal reasoning chains
- Attention flow patterns
- Training progress
- Concept composition
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import time
import threading
import queue

# Try to import visualization libraries
try:
    import plotly.graph_objects as go
    import plotly.subplots as sp
    from plotly.offline import plot
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    
try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.widgets import Slider, Button, CheckButtons
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


@dataclass
class VisualizationConfig:
    """Configuration for visualization settings"""
    width: int = 1200
    height: int = 800
    node_size_scale: float = 300
    edge_width_scale: float = 3
    color_scheme: str = 'viridis'
    animation_speed: float = 100  # milliseconds
    show_labels: bool = True
    interactive: bool = True


class InteractiveNetworkVisualizer:
    """
    Interactive network topology visualizer
    
    Features:
    - Real-time network structure display
    - Interactive node selection and information
    - Dynamic layout updates
    - Attention flow visualization
    - Causal chain highlighting
    """
    
    def __init__(self, network, config: Optional[VisualizationConfig] = None):
        self.network = network
        self.config = config or VisualizationConfig()
        self.graph = None
        self.pos = None
        self.fig = None
        self.animation_running = False
        self.selected_neuron = None
        
        # Data for real-time updates
        self.activation_history = []
        self.attention_history = []
        self.causal_history = []
        
        self._initialize_graph()
    
    def _initialize_graph(self):
        """Initialize NetworkX graph from AGI network"""
        self.graph = nx.DiGraph()
        
        # Add nodes (neurons)
        for i in range(self.network.config.num_neurons):
            neuron_type = 'input' if i < self.network.config.input_size else 'hidden'
            if i >= self.network.config.num_neurons - self.network.config.output_size:
                neuron_type = 'output'
            
            self.graph.add_node(i, 
                              type=neuron_type,
                              activation=0.0,
                              concept=getattr(self.network.neurons[i], 'concept_type', 'unknown') if hasattr(self.network, 'neurons') and i < len(self.network.neurons) else 'unknown')
        
        # Add edges (connections)
        if hasattr(self.network, 'neurons'):
            for i, neuron in enumerate(self.network.neurons):
                if hasattr(neuron, 'neighbors'):
                    for neighbor_id in neuron.neighbors:
                        if neighbor_id < len(self.network.neurons):
                            weight = np.random.uniform(0.1, 1.0)  # Placeholder weight
                            self.graph.add_edge(i, neighbor_id, weight=weight, active=False)
        
        # Compute layout
        self.pos = nx.spring_layout(self.graph, k=2, iterations=50)
    
    def create_interactive_plot(self) -> str:
        """Create interactive visualization"""
        if not PLOTLY_AVAILABLE:
            return self._create_simple_text_viz()
            
        try:
            # Prepare node data
            node_trace = self._create_node_trace()
            edge_trace = self._create_edge_trace()
            
            # Create figure
            fig = go.Figure(data=[edge_trace, node_trace],
                           layout=go.Layout(
                               title='AGI Network Interactive Visualization',
                               titlefont_size=16,
                               showlegend=False,
                               hovermode='closest',
                               margin=dict(b=20,l=5,r=5,t=40),
                               annotations=[ dict(
                                   text="Click on nodes to explore causal chains",
                                   showarrow=False,
                                   xref="paper", yref="paper",
                                   x=0.005, y=-0.002,
                                   xanchor='left', yanchor='bottom',
                                   font=dict(color='gray', size=12)
                               )],
                               xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                               yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                               width=self.config.width,
                               height=self.config.height))
            
            # Add interactivity
            fig.update_traces(selector=dict(type='scatter', mode='markers'), 
                             hovertemplate='<b>Neuron %{customdata[0]}</b><br>' +
                                         'Type: %{customdata[1]}<br>' +
                                         'Activation: %{customdata[2]:.3f}<br>' +
                                         'Concept: %{customdata[3]}<extra></extra>')
            
            # Save as HTML
            output_file = 'agi_network_visualization.html'
            plot(fig, filename=output_file, auto_open=False)
            
            return output_file
        except Exception as e:
            return self._create_simple_text_viz()
    
    def _create_simple_text_viz(self) -> str:
        """Create simple text-based visualization when plotting libraries unavailable"""
        output_file = 'agi_network_simple.html'
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head><title>AGI Network Visualization</title></head>
        <body>
            <h1>AGI Network Structure</h1>
            <p>Network with {self.network.config.num_neurons} neurons</p>
            <p>Input size: {self.network.config.input_size}</p>
            <p>Output size: {self.network.config.output_size}</p>
            <p>Note: Full visualization requires plotly. Install with: pip install plotly</p>
            
            <h2>Network Nodes:</h2>
            <ul>
        """
        
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            node_type = node_data.get('type', 'unknown')
            html_content += f"<li>Neuron {node}: {node_type}</li>"
        
        html_content += """
            </ul>
        </body>
        </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        return output_file
    
    def _create_node_trace(self):
        """Create node trace for Plotly"""
        node_x = []
        node_y = []
        node_info = []
        node_colors = []
        node_sizes = []
        
        for node in self.graph.nodes():
            x, y = self.pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Node information
            node_data = self.graph.nodes[node]
            activation = node_data.get('activation', 0.0)
            node_type = node_data.get('type', 'unknown')
            concept = node_data.get('concept', 'unknown')
            
            node_info.append([node, node_type, activation, concept])
            
            # Color based on type
            if node_type == 'input':
                node_colors.append('lightblue')
            elif node_type == 'output':
                node_colors.append('lightcoral')
            else:
                node_colors.append('lightgreen')
            
            # Size based on activation
            size = 10 + activation * 20
            node_sizes.append(size)
        
        return go.Scatter(x=node_x, y=node_y,
                         mode='markers',
                         marker=dict(size=node_sizes,
                                   color=node_colors,
                                   line=dict(width=2, color='black')),
                         customdata=node_info,
                         name='Neurons')
    
    def _create_edge_trace(self):
        """Create edge trace for Plotly"""
        edge_x = []
        edge_y = []
        
        for edge in self.graph.edges():
            x0, y0 = self.pos[edge[0]]
            x1, y1 = self.pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        return go.Scatter(x=edge_x, y=edge_y,
                         line=dict(width=1, color='gray'),
                         hoverinfo='none',
                         mode='lines',
                         name='Connections')
    
    def create_causal_chain_viz(self, target_neuron: int) -> str:
        """Create causal chain visualization for specific neuron"""
        if not hasattr(self.network, 'get_causal_explanation'):
            return "Causal explanation not available"
        
        try:
            # Get causal explanation
            explanation = self.network.get_causal_explanation(target_neuron)
            reasoning_path = explanation.get('reasoning_path', [])
            
            if not reasoning_path:
                return "No causal chain found"
            
            # Create causal chain visualization
            fig = go.Figure()
            
            # Add nodes for each step in reasoning
            x_positions = list(range(len(reasoning_path)))
            y_position = 0
            
            for i, step in enumerate(reasoning_path):
                contribution = step.get('contribution', 0)
                neuron_id = step.get('neuron_id', i)
                concept_type = step.get('concept_type', 'unknown')
                
                # Node size based on contribution
                size = 20 + abs(contribution) * 50
                color = 'red' if contribution < 0 else 'green'
                
                fig.add_trace(go.Scatter(
                    x=[i], y=[y_position],
                    mode='markers+text',
                    marker=dict(size=size, color=color, opacity=0.7),
                    text=f"N{neuron_id}<br>{concept_type}<br>{contribution:.3f}",
                    textposition="top center",
                    name=f"Step {i+1}"
                ))
                
                # Add arrow to next step
                if i < len(reasoning_path) - 1:
                    fig.add_annotation(
                        x=i+0.4, y=y_position,
                        ax=i, ay=y_position,
                        xref='x', yref='y',
                        axref='x', ayref='y',
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=2,
                        arrowcolor='blue'
                    )
            
            fig.update_layout(
                title=f'Causal Reasoning Chain for Neuron {target_neuron}',
                xaxis_title='Reasoning Steps',
                yaxis=dict(showticklabels=False),
                showlegend=False,
                width=self.config.width,
                height=400
            )
            
            output_file = f'causal_chain_neuron_{target_neuron}.html'
            plot(fig, filename=output_file, auto_open=False)
            
            return output_file
            
        except Exception as e:
            return f"Error creating causal visualization: {e}"
    
    def create_attention_flow_viz(self) -> str:
        """Create attention flow visualization"""
        fig = go.Figure()
        
        # Get attention data if available
        attention_data = {}
        if hasattr(self.network, 'attention_module'):
            if hasattr(self.network.attention_module, 'last_attention_scores'):
                attention_data = self.network.attention_module.last_attention_scores or {}
        
        # Create heatmap of attention scores
        if attention_data:
            neurons = sorted(attention_data.keys())
            scores = [attention_data[n] for n in neurons]
            
            fig.add_trace(go.Bar(
                x=neurons,
                y=scores,
                marker_color='viridis',
                name='Attention Scores'
            ))
            
            fig.update_layout(
                title='Attention Flow Patterns',
                xaxis_title='Neuron ID',
                yaxis_title='Attention Score',
                width=self.config.width,
                height=400
            )
        else:
            # Create placeholder visualization
            fig.add_annotation(
                text="No attention data available<br>Run network forward pass first",
                x=0.5, y=0.5,
                xref='paper', yref='paper',
                showarrow=False,
                font=dict(size=16)
            )
        
        output_file = 'attention_flow.html'
        plot(fig, filename=output_file, auto_open=False)
        
        return output_file
    
    def create_concept_composition_viz(self) -> str:
        """Create concept composition visualization"""
        fig = go.Figure()
        
        # Check if network has concept system
        if hasattr(self.network, 'concept_registry'):
            try:
                concepts = self.network.concept_registry.get_all_concepts()
                
                # Create concept relationship graph
                concept_graph = nx.Graph()
                
                for concept in concepts:
                    concept_graph.add_node(concept)
                
                # Add edges based on compatibility
                for i, concept_a in enumerate(concepts):
                    for concept_b in concepts[i+1:]:
                        compatibility = self.network.concept_registry.get_compatibility(concept_a, concept_b)
                        if compatibility > 0.5:  # Threshold for visualization
                            concept_graph.add_edge(concept_a, concept_b, weight=compatibility)
                
                # Create layout
                pos = nx.spring_layout(concept_graph)
                
                # Add nodes
                for node in concept_graph.nodes():
                    x, y = pos[node]
                    fig.add_trace(go.Scatter(
                        x=[x], y=[y],
                        mode='markers+text',
                        marker=dict(size=30, color='lightblue'),
                        text=node,
                        textposition='middle center',
                        name=node
                    ))
                
                # Add edges
                for edge in concept_graph.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    weight = concept_graph.edges[edge]['weight']
                    
                    fig.add_trace(go.Scatter(
                        x=[x0, x1], y=[y0, y1],
                        mode='lines',
                        line=dict(width=weight*5, color='gray'),
                        showlegend=False,
                        hoverinfo='none'
                    ))
                
                fig.update_layout(
                    title='Concept Composition Network',
                    showlegend=False,
                    width=self.config.width,
                    height=self.config.height
                )
                
            except Exception as e:
                fig.add_annotation(
                    text=f"Error loading concepts: {e}",
                    x=0.5, y=0.5,
                    xref='paper', yref='paper',
                    showarrow=False
                )
        else:
            fig.add_annotation(
                text="Concept system not available in this network",
                x=0.5, y=0.5,
                xref='paper', yref='paper',
                showarrow=False,
                font=dict(size=16)
            )
        
        output_file = 'concept_composition.html'
        plot(fig, filename=output_file, auto_open=False)
        
        return output_file


class RealTimeTrainingVisualizer:
    """
    Real-time training progress visualizer
    
    Features:
    - Live loss and accuracy curves
    - Network activation patterns
    - Causal reasoning development
    - Performance metrics dashboard
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        self.data_queue = queue.Queue()
        self.is_recording = False
        
        # Training data
        self.epochs = []
        self.losses = []
        self.accuracies = []
        self.agi_scores = []
        self.causal_scores = []
        
    def start_recording(self):
        """Start recording training data"""
        self.is_recording = True
        self.epochs.clear()
        self.losses.clear()
        self.accuracies.clear()
        self.agi_scores.clear()
        self.causal_scores.clear()
    
    def stop_recording(self):
        """Stop recording training data"""
        self.is_recording = False
    
    def record_epoch(self, epoch: int, metrics: Dict[str, float]):
        """Record metrics for current epoch"""
        if not self.is_recording:
            return
        
        self.epochs.append(epoch)
        self.losses.append(metrics.get('loss', 0))
        self.accuracies.append(metrics.get('accuracy', 0))
        self.agi_scores.append(metrics.get('agi_score', 0))
        self.causal_scores.append(metrics.get('causal_score', 0))
    
    def create_training_dashboard(self) -> str:
        """Create interactive training dashboard"""
        if not self.epochs:
            fig = go.Figure()
            fig.add_annotation(
                text="No training data recorded yet<br>Start training to see live metrics",
                x=0.5, y=0.5,
                xref='paper', yref='paper',
                showarrow=False,
                font=dict(size=16)
            )
        else:
            # Create subplots
            fig = sp.make_subplots(
                rows=2, cols=2,
                subplot_titles=('Training Loss', 'Accuracy', 'AGI Score', 'Causal Reasoning'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Loss curve
            fig.add_trace(
                go.Scatter(x=self.epochs, y=self.losses, name='Loss', line=dict(color='red')),
                row=1, col=1
            )
            
            # Accuracy curve
            fig.add_trace(
                go.Scatter(x=self.epochs, y=self.accuracies, name='Accuracy', line=dict(color='blue')),
                row=1, col=2
            )
            
            # AGI score curve
            fig.add_trace(
                go.Scatter(x=self.epochs, y=self.agi_scores, name='AGI Score', line=dict(color='green')),
                row=2, col=1
            )
            
            # Causal reasoning curve
            fig.add_trace(
                go.Scatter(x=self.epochs, y=self.causal_scores, name='Causal Score', line=dict(color='purple')),
                row=2, col=2
            )
        
        fig.update_layout(
            title='AGI Training Dashboard',
            showlegend=False,
            width=self.config.width,
            height=self.config.height
        )
        
        output_file = 'training_dashboard.html'
        plot(fig, filename=output_file, auto_open=False)
        
        return output_file


class PerformanceAnalysisVisualizer:
    """
    Performance analysis and profiling visualizer
    
    Features:
    - Function timing analysis
    - Memory usage patterns
    - Bottleneck identification
    - Optimization recommendations
    """
    
    def __init__(self, profiling_data: Dict[str, Any], config: Optional[VisualizationConfig] = None):
        self.profiling_data = profiling_data
        self.config = config or VisualizationConfig()
    
    def create_performance_analysis(self) -> str:
        """Create comprehensive performance analysis dashboard"""
        fig = sp.make_subplots(
            rows=2, cols=2,
            subplot_titles=('Function Timing', 'Memory Usage', 'Call Frequency', 'Bottlenecks'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "treemap"}]]
        )
        
        # Extract timing data
        timing_data = self.profiling_data.get('timing', {})
        if timing_data:
            functions = list(timing_data.keys())
            times = [timing_data[f]['total_time'] for f in functions]
            
            fig.add_trace(
                go.Bar(x=functions, y=times, name='Execution Time'),
                row=1, col=1
            )
        
        # Memory usage over time
        memory_data = self.profiling_data.get('memory', [])
        if memory_data:
            timestamps = list(range(len(memory_data)))
            fig.add_trace(
                go.Scatter(x=timestamps, y=memory_data, name='Memory Usage'),
                row=1, col=2
            )
        
        # Call frequency
        call_data = self.profiling_data.get('calls', {})
        if call_data:
            functions = list(call_data.keys())
            counts = [call_data[f] for f in functions]
            
            fig.add_trace(
                go.Bar(x=functions, y=counts, name='Call Count'),
                row=2, col=1
            )
        
        # Bottlenecks treemap
        bottleneck_data = self.profiling_data.get('bottlenecks', {})
        if bottleneck_data:
            fig.add_trace(
                go.Treemap(
                    labels=list(bottleneck_data.keys()),
                    values=list(bottleneck_data.values()),
                    name='Bottlenecks'
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title='Performance Analysis Dashboard',
            showlegend=False,
            width=self.config.width,
            height=self.config.height
        )
        
        output_file = 'performance_analysis.html'
        plot(fig, filename=output_file, auto_open=False)
        
        return output_file


class AGIVisualizationSuite:
    """
    Comprehensive AGI visualization suite
    
    Combines all visualization tools into a unified interface
    """
    
    def __init__(self, network, config: Optional[VisualizationConfig] = None):
        self.network = network
        self.config = config or VisualizationConfig()
        
        # Initialize visualizers
        self.network_viz = InteractiveNetworkVisualizer(network, config)
        self.training_viz = RealTimeTrainingVisualizer(config)
        
    def create_comprehensive_dashboard(self) -> Dict[str, str]:
        """Create comprehensive visualization dashboard"""
        dashboard_files = {}
        
        print("🎨 Creating AGI visualization dashboard...")
        
        # Network topology
        print("  📊 Network topology...")
        dashboard_files['network'] = self.network_viz.create_interactive_plot()
        
        # Attention flow
        print("  🔍 Attention patterns...")
        dashboard_files['attention'] = self.network_viz.create_attention_flow_viz()
        
        # Concept composition
        print("  🧩 Concept relationships...")
        dashboard_files['concepts'] = self.network_viz.create_concept_composition_viz()
        
        # Training dashboard
        print("  📈 Training metrics...")
        dashboard_files['training'] = self.training_viz.create_training_dashboard()
        
        # Create main index page
        dashboard_files['index'] = self._create_index_page(dashboard_files)
        
        print("✅ Visualization dashboard created!")
        return dashboard_files
    
    def _create_index_page(self, dashboard_files: Dict[str, str]) -> str:
        """Create main index page for dashboard"""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>AGI-Formula Visualization Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
                .header { text-align: center; color: #333; margin-bottom: 30px; }
                .dashboard-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
                .card { background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .card h3 { color: #2c3e50; margin-top: 0; }
                .card p { color: #666; line-height: 1.6; }
                .btn { display: inline-block; background: #3498db; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; margin-top: 10px; }
                .btn:hover { background: #2980b9; }
                .emoji { font-size: 24px; margin-right: 10px; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🧠 AGI-Formula Visualization Dashboard</h1>
                <p>Interactive exploration of your AGI network</p>
            </div>
            
            <div class="dashboard-grid">
        """
        
        # Add cards for each visualization
        visualizations = [
            ('network', '🕸️', 'Network Topology', 'Explore the structure and connections of your AGI network'),
            ('attention', '🔍', 'Attention Patterns', 'Visualize how attention flows through the network'),
            ('concepts', '🧩', 'Concept Composition', 'See how concepts relate and combine'),
            ('training', '📈', 'Training Progress', 'Monitor real-time training metrics and progress')
        ]
        
        for key, emoji, title, description in visualizations:
            if key in dashboard_files:
                html_content += f"""
                <div class="card">
                    <h3><span class="emoji">{emoji}</span>{title}</h3>
                    <p>{description}</p>
                    <a href="{dashboard_files[key]}" class="btn">Open Visualization</a>
                </div>
                """
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        index_file = 'agi_dashboard_index.html'
        with open(index_file, 'w') as f:
            f.write(html_content)
        
        return index_file
    
    def start_real_time_monitoring(self, trainer=None):
        """Start real-time monitoring during training"""
        if trainer:
            self.training_viz.start_recording()
            
            # Hook into trainer to record metrics
            original_train_step = getattr(trainer, 'train_step', None)
            
            if original_train_step:
                def monitored_train_step(*args, **kwargs):
                    result = original_train_step(*args, **kwargs)
                    
                    # Extract metrics from result
                    if isinstance(result, dict):
                        epoch = kwargs.get('epoch', 0)
                        self.training_viz.record_epoch(epoch, result)
                    
                    return result
                
                trainer.train_step = monitored_train_step