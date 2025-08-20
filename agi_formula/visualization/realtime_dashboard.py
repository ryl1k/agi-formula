"""
Real-time Dashboard for AGI-Formula Network Monitoring

Provides live monitoring and visualization of AGI network activities:
- Real-time performance metrics
- Live neuron activation tracking
- Causal chain visualization
- Training progress monitoring
- Interactive control panels
"""

import json
import time
import threading
import queue
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from pathlib import Path
import base64
from dataclasses import dataclass


@dataclass
class MetricData:
    """Structure for metric data points"""
    timestamp: float
    value: float
    metric_type: str
    metadata: Dict[str, Any] = None


class RealtimeDashboard:
    """
    Real-time dashboard for AGI network monitoring
    
    Features:
    - Live performance metric collection and display
    - Real-time neuron activation visualization
    - Causal chain tracking and analysis
    - Training progress monitoring
    - WebSocket-based live updates
    """
    
    def __init__(self, network, config: Optional[Dict[str, Any]] = None):
        self.network = network
        self.config = config or self._get_default_config()
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread = None
        self.data_queue = queue.Queue()
        
        # Metrics storage
        self.metrics_history = {
            'agi_score': [],
            'activation_levels': [],
            'causal_strength': [],
            'training_loss': [],
            'memory_usage': [],
            'fps': []
        }
        
        # Performance tracking
        self.performance_tracker = PerformanceTracker()
        self.activation_tracker = ActivationTracker(self.network)
        self.causal_tracker = CausalTracker()
        
        # WebSocket server for live updates
        self.websocket_server = None
        self.connected_clients = set()
        
        # Initialize dashboard
        self._initialize_dashboard()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default dashboard configuration"""
        return {
            'update_interval': 0.1,  # 10 FPS
            'max_history_points': 1000,
            'websocket_port': 8765,
            'auto_save_interval': 60,  # seconds
            'metrics_enabled': {
                'agi_score': True,
                'activations': True,
                'causal_chains': True,
                'memory': True,
                'performance': True
            },
            'visualization_settings': {
                'theme': 'dark',
                'animation_speed': 1.0,
                'show_legends': True,
                'auto_scale': True
            }
        }
    
    def _initialize_dashboard(self):
        """Initialize dashboard components"""
        # Setup data collection
        self._setup_metric_collectors()
        
        # Initialize visualization components
        self.chart_manager = ChartManager(self.config['visualization_settings'])
        self.alert_system = AlertSystem()
        
        print(f"Real-time dashboard initialized for network with {self.network.config.num_neurons} neurons")
    
    def _setup_metric_collectors(self):
        """Setup metric collection systems"""
        self.metric_collectors = {
            'agi_score': self._collect_agi_score,
            'activation_levels': self._collect_activation_levels,
            'causal_strength': self._collect_causal_strength,
            'training_loss': self._collect_training_loss,
            'memory_usage': self._collect_memory_usage,
            'fps': self._collect_fps
        }
    
    def start_monitoring(self):
        """Start real-time monitoring"""
        if self.is_monitoring:
            print("Monitoring already active")
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        # Start WebSocket server for live updates
        self._start_websocket_server()
        
        print("Real-time monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.is_monitoring = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1)
        
        if self.websocket_server:
            self.websocket_server.shutdown()
        
        print("Real-time monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        last_update = time.time()
        
        while self.is_monitoring:
            current_time = time.time()
            
            if current_time - last_update >= self.config['update_interval']:
                # Collect all enabled metrics
                for metric_name, collector in self.metric_collectors.items():
                    if self.config['metrics_enabled'].get(metric_name, True):
                        try:
                            metric_data = collector()
                            if metric_data is not None:
                                self._add_metric_data(metric_name, metric_data)
                        except Exception as e:
                            print(f"Error collecting {metric_name}: {e}")
                
                # Process alerts
                self._check_alerts()
                
                # Send updates to connected clients
                self._broadcast_updates()
                
                last_update = current_time
            
            time.sleep(0.01)  # Small sleep to prevent CPU spinning
    
    def _collect_agi_score(self) -> Optional[MetricData]:
        """Collect current AGI score"""
        try:
            # Get current network performance metrics
            if hasattr(self.network, 'get_performance_metrics'):
                metrics = self.network.get_performance_metrics()
                agi_score = metrics.get('agi_score', 0.0)
            else:
                # Fallback calculation
                agi_score = self._calculate_simple_agi_score()
            
            return MetricData(
                timestamp=time.time(),
                value=agi_score,
                metric_type='agi_score',
                metadata={'network_id': getattr(self.network, 'id', 'unknown')}
            )
        except Exception as e:
            print(f"Error collecting AGI score: {e}")
            return None
    
    def _collect_activation_levels(self) -> Optional[MetricData]:
        """Collect neuron activation levels"""
        try:
            activations = self.activation_tracker.get_current_activations()
            if activations is not None:
                avg_activation = np.mean(activations)
                max_activation = np.max(activations)
                
                return MetricData(
                    timestamp=time.time(),
                    value=avg_activation,
                    metric_type='activation_levels',
                    metadata={
                        'max_activation': max_activation,
                        'active_neurons': np.sum(activations > 0.1),
                        'total_neurons': len(activations)
                    }
                )
        except Exception as e:
            print(f"Error collecting activations: {e}")
            return None
    
    def _collect_causal_strength(self) -> Optional[MetricData]:
        """Collect causal reasoning strength"""
        try:
            causal_data = self.causal_tracker.get_current_strength()
            if causal_data is not None:
                return MetricData(
                    timestamp=time.time(),
                    value=causal_data['average_strength'],
                    metric_type='causal_strength',
                    metadata={
                        'active_chains': causal_data['active_chains'],
                        'max_strength': causal_data['max_strength']
                    }
                )
        except Exception as e:
            print(f"Error collecting causal strength: {e}")
            return None
    
    def _collect_training_loss(self) -> Optional[MetricData]:
        """Collect current training loss"""
        try:
            if hasattr(self.network, 'current_loss'):
                loss = self.network.current_loss
                return MetricData(
                    timestamp=time.time(),
                    value=loss,
                    metric_type='training_loss'
                )
        except Exception as e:
            print(f"Error collecting training loss: {e}")
            return None
    
    def _collect_performance_metrics(self) -> Optional[MetricData]:
        """Collect performance metrics"""
        try:
            import psutil
            process = psutil.Process()
            cpu_percent = process.cpu_percent()
            
            return MetricData(
                timestamp=time.time(),
                value=cpu_percent,
                metric_type='performance_metrics',
                metadata={'unit': 'percent'}
            )
        except Exception as e:
            print(f"Error collecting performance metrics: {e}")
            return None
    
    def _collect_memory_usage(self) -> Optional[MetricData]:
        """Collect memory usage metrics"""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            
            return MetricData(
                timestamp=time.time(),
                value=memory_mb,
                metric_type='memory_usage',
                metadata={'unit': 'MB'}
            )
        except Exception as e:
            print(f"Error collecting memory usage: {e}")
            return None
    
    def _collect_fps(self) -> Optional[MetricData]:
        """Collect frame rate metrics"""
        fps = self.performance_tracker.get_fps()
        if fps is not None:
            return MetricData(
                timestamp=time.time(),
                value=fps,
                metric_type='fps'
            )
        return None
    
    def _calculate_simple_agi_score(self) -> float:
        """Calculate a simple AGI score based on network activity"""
        try:
            # Get activation diversity
            activations = self.activation_tracker.get_current_activations()
            if activations is None:
                return 0.0
            
            # Calculate activity metrics
            active_ratio = np.sum(activations > 0.1) / len(activations)
            activation_variance = np.var(activations)
            
            # Simple AGI score combining activity and diversity
            agi_score = (active_ratio * 0.5 + activation_variance * 0.5)
            return min(1.0, max(0.0, agi_score))
            
        except Exception:
            return 0.0
    
    def _add_metric_data(self, metric_name: str, data: MetricData):
        """Add metric data to history"""
        if metric_name not in self.metrics_history:
            self.metrics_history[metric_name] = []
        
        history = self.metrics_history[metric_name]
        history.append(data)
        
        # Maintain max history length
        max_points = self.config['max_history_points']
        if len(history) > max_points:
            self.metrics_history[metric_name] = history[-max_points:]
    
    def _check_alerts(self):
        """Check for alert conditions"""
        # Add alert logic based on metric thresholds
        for metric_name, history in self.metrics_history.items():
            if len(history) > 0:
                latest = history[-1]
                self.alert_system.check_metric(metric_name, latest)
    
    def _broadcast_updates(self):
        """Broadcast updates to connected WebSocket clients"""
        if not self.connected_clients:
            return
        
        # Prepare update data
        update_data = {
            'timestamp': datetime.now().isoformat(),
            'metrics': self._get_latest_metrics(),
            'network_status': self._get_network_status()
        }
        
        # Send to all connected clients
        message = json.dumps(update_data, default=str)
        for client in list(self.connected_clients):
            try:
                # Placeholder for WebSocket send
                pass
            except Exception as e:
                print(f"Error sending to client: {e}")
                self.connected_clients.discard(client)
    
    def _get_latest_metrics(self) -> Dict[str, Any]:
        """Get latest metrics for broadcast"""
        latest = {}
        for metric_name, history in self.metrics_history.items():
            if history:
                latest_data = history[-1]
                latest[metric_name] = {
                    'value': latest_data.value,
                    'timestamp': latest_data.timestamp,
                    'metadata': latest_data.metadata
                }
        return latest
    
    def _get_network_status(self) -> Dict[str, Any]:
        """Get current network status"""
        return {
            'num_neurons': self.network.config.num_neurons,
            'is_training': getattr(self.network, 'is_training', False),
            'monitoring_active': self.is_monitoring,
            'uptime': time.time() - getattr(self, 'start_time', time.time())
        }
    
    def _start_websocket_server(self):
        """Start WebSocket server for live updates"""
        # Placeholder for WebSocket server implementation
        print(f"WebSocket server would start on port {self.config['websocket_port']}")
    
    def get_dashboard_html(self, output_file: str = "realtime_dashboard.html") -> str:
        """Generate real-time dashboard HTML"""
        html_template = self._get_dashboard_template()
        
        # Embed configuration and initial data
        config_json = json.dumps(self.config, indent=2, default=str)
        initial_data = json.dumps(self._get_latest_metrics(), default=str)
        
        # Generate complete HTML
        html_content = html_template.format(
            config=config_json,
            initial_data=initial_data,
            websocket_port=self.config['websocket_port'],
            title=f"AGI Network Real-time Dashboard - {self.network.config.num_neurons} neurons"
        )
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Real-time dashboard saved to: {output_file}")
        return output_file
    
    def _get_dashboard_template(self) -> str:
        """Get HTML template for real-time dashboard"""
        return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 0;
            background: #0a0a0a;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: white;
            overflow-x: hidden;
        }}
        
        .dashboard {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            grid-template-rows: auto 1fr 1fr;
            height: 100vh;
            gap: 10px;
            padding: 10px;
        }}
        
        .header {{
            grid-column: 1 / -1;
            background: rgba(0, 0, 0, 0.8);
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #333;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .metric-panel {{
            background: rgba(0, 0, 0, 0.8);
            border: 1px solid #333;
            border-radius: 8px;
            padding: 15px;
            overflow: hidden;
        }}
        
        .chart-container {{
            position: relative;
            height: 200px;
            margin-top: 10px;
        }}
        
        .status-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-top: 10px;
        }}
        
        .status-item {{
            background: #1a1a1a;
            padding: 10px;
            border-radius: 4px;
            text-align: center;
        }}
        
        .status-value {{
            font-size: 24px;
            font-weight: bold;
            color: #4a90e2;
        }}
        
        .status-label {{
            font-size: 12px;
            color: #888;
            margin-top: 5px;
        }}
        
        .connection-status {{
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 12px;
        }}
        
        .connected {{
            background: #2d5a2d;
            color: #7ed321;
        }}
        
        .disconnected {{
            background: #5a2d2d;
            color: #e32121;
        }}
        
        .alert {{
            background: #5a2d2d;
            border: 1px solid #e32121;
            border-radius: 4px;
            padding: 10px;
            margin: 5px 0;
            font-size: 14px;
        }}
        
        h3 {{
            margin: 0 0 10px 0;
            color: #4a90e2;
        }}
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="header">
            <div>
                <h1>AGI Network Real-time Dashboard</h1>
                <div id="network-info">Loading network information...</div>
            </div>
            <div>
                <span id="connection-status" class="connection-status disconnected">Disconnected</span>
                <span id="last-update"></span>
            </div>
        </div>
        
        <div class="metric-panel">
            <h3>AGI Score & Performance</h3>
            <div class="chart-container">
                <canvas id="agi-chart"></canvas>
            </div>
            <div class="status-grid">
                <div class="status-item">
                    <div class="status-value" id="current-agi">0.000</div>
                    <div class="status-label">Current AGI Score</div>
                </div>
                <div class="status-item">
                    <div class="status-value" id="current-fps">0</div>
                    <div class="status-label">FPS</div>
                </div>
            </div>
        </div>
        
        <div class="metric-panel">
            <h3>Neuron Activations</h3>
            <div class="chart-container">
                <canvas id="activation-chart"></canvas>
            </div>
            <div class="status-grid">
                <div class="status-item">
                    <div class="status-value" id="active-neurons">0</div>
                    <div class="status-label">Active Neurons</div>
                </div>
                <div class="status-item">
                    <div class="status-value" id="avg-activation">0.000</div>
                    <div class="status-label">Avg Activation</div>
                </div>
            </div>
        </div>
        
        <div class="metric-panel">
            <h3>Causal Reasoning</h3>
            <div class="chart-container">
                <canvas id="causal-chart"></canvas>
            </div>
            <div class="status-grid">
                <div class="status-item">
                    <div class="status-value" id="causal-strength">0.000</div>
                    <div class="status-label">Causal Strength</div>
                </div>
                <div class="status-item">
                    <div class="status-value" id="active-chains">0</div>
                    <div class="status-label">Active Chains</div>
                </div>
            </div>
        </div>
        
        <div class="metric-panel">
            <h3>System Status</h3>
            <div class="status-grid">
                <div class="status-item">
                    <div class="status-value" id="memory-usage">0</div>
                    <div class="status-label">Memory (MB)</div>
                </div>
                <div class="status-item">
                    <div class="status-value" id="uptime">0s</div>
                    <div class="status-label">Uptime</div>
                </div>
            </div>
            <div id="alerts-container">
                <!-- Alerts will appear here -->
            </div>
        </div>
    </div>

    <script>
        // Dashboard configuration
        const config = {config};
        const websocketPort = {websocket_port};
        
        // Chart instances
        let charts = {{}};
        
        // WebSocket connection
        let ws = null;
        let reconnectInterval = null;
        
        // Data storage
        let metricsData = {{
            agi_score: [],
            activation_levels: [],
            causal_strength: [],
            memory_usage: []
        }};
        
        // Initialize dashboard
        function initDashboard() {{
            createCharts();
            connectWebSocket();
            
            // Initial data
            const initialData = {initial_data};
            updateMetrics(initialData);
            
            console.log('Real-time dashboard initialized');
        }}
        
        function createCharts() {{
            // AGI Score Chart
            const agiCtx = document.getElementById('agi-chart').getContext('2d');
            charts.agi = new Chart(agiCtx, {{
                type: 'line',
                data: {{
                    labels: [],
                    datasets: [{{
                        label: 'AGI Score',
                        data: [],
                        borderColor: '#4a90e2',
                        backgroundColor: 'rgba(74, 144, 226, 0.1)',
                        fill: true,
                        tension: 0.4
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        y: {{ min: 0, max: 1 }}
                    }},
                    plugins: {{
                        legend: {{ display: false }}
                    }}
                }}
            }});
            
            // Activation Chart
            const activationCtx = document.getElementById('activation-chart').getContext('2d');
            charts.activation = new Chart(activationCtx, {{
                type: 'line',
                data: {{
                    labels: [],
                    datasets: [{{
                        label: 'Avg Activation',
                        data: [],
                        borderColor: '#7ed321',
                        backgroundColor: 'rgba(126, 211, 33, 0.1)',
                        fill: true,
                        tension: 0.4
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{ display: false }}
                    }}
                }}
            }});
            
            // Causal Chart
            const causalCtx = document.getElementById('causal-chart').getContext('2d');
            charts.causal = new Chart(causalCtx, {{
                type: 'line',
                data: {{
                    labels: [],
                    datasets: [{{
                        label: 'Causal Strength',
                        data: [],
                        borderColor: '#f5a623',
                        backgroundColor: 'rgba(245, 166, 35, 0.1)',
                        fill: true,
                        tension: 0.4
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{ display: false }}
                    }}
                }}
            }});
        }}
        
        function connectWebSocket() {{
            try {{
                ws = new WebSocket(`ws://localhost:${{websocketPort}}`);
                
                ws.onopen = function() {{
                    console.log('WebSocket connected');
                    updateConnectionStatus(true);
                    if (reconnectInterval) {{
                        clearInterval(reconnectInterval);
                        reconnectInterval = null;
                    }}
                }};
                
                ws.onmessage = function(event) {{
                    const data = JSON.parse(event.data);
                    updateMetrics(data.metrics);
                    updateNetworkStatus(data.network_status);
                    updateLastUpdate();
                }};
                
                ws.onclose = function() {{
                    console.log('WebSocket disconnected');
                    updateConnectionStatus(false);
                    
                    // Auto-reconnect
                    if (!reconnectInterval) {{
                        reconnectInterval = setInterval(connectWebSocket, 5000);
                    }}
                }};
                
                ws.onerror = function(error) {{
                    console.error('WebSocket error:', error);
                }};
                
            }} catch (error) {{
                console.error('Failed to connect WebSocket:', error);
                updateConnectionStatus(false);
            }}
        }}
        
        function updateMetrics(metrics) {{
            const now = new Date().toLocaleTimeString();
            
            // Update AGI score
            if (metrics.agi_score) {{
                const value = metrics.agi_score.value;
                updateChart(charts.agi, now, value);
                document.getElementById('current-agi').textContent = value.toFixed(3);
            }}
            
            // Update activations
            if (metrics.activation_levels) {{
                const value = metrics.activation_levels.value;
                const metadata = metrics.activation_levels.metadata || {{}};
                updateChart(charts.activation, now, value);
                document.getElementById('avg-activation').textContent = value.toFixed(3);
                document.getElementById('active-neurons').textContent = metadata.active_neurons || 0;
            }}
            
            // Update causal strength
            if (metrics.causal_strength) {{
                const value = metrics.causal_strength.value;
                const metadata = metrics.causal_strength.metadata || {{}};
                updateChart(charts.causal, now, value);
                document.getElementById('causal-strength').textContent = value.toFixed(3);
                document.getElementById('active-chains').textContent = metadata.active_chains || 0;
            }}
            
            // Update FPS
            if (metrics.fps) {{
                document.getElementById('current-fps').textContent = Math.round(metrics.fps.value);
            }}
            
            // Update memory usage
            if (metrics.memory_usage) {{
                document.getElementById('memory-usage').textContent = Math.round(metrics.memory_usage.value);
            }}
        }}
        
        function updateChart(chart, label, value) {{
            chart.data.labels.push(label);
            chart.data.datasets[0].data.push(value);
            
            // Keep only last 50 points
            if (chart.data.labels.length > 50) {{
                chart.data.labels.shift();
                chart.data.datasets[0].data.shift();
            }}
            
            chart.update('none');
        }}
        
        function updateConnectionStatus(connected) {{
            const statusElement = document.getElementById('connection-status');
            if (connected) {{
                statusElement.className = 'connection-status connected';
                statusElement.textContent = 'Connected';
            }} else {{
                statusElement.className = 'connection-status disconnected';
                statusElement.textContent = 'Disconnected';
            }}
        }}
        
        function updateNetworkStatus(status) {{
            if (status) {{
                const uptime = Math.round(status.uptime || 0);
                document.getElementById('uptime').textContent = `${{uptime}}s`;
                
                const networkInfo = `${{status.num_neurons}} neurons | Training: ${{status.is_training ? 'Yes' : 'No'}}`;
                document.getElementById('network-info').textContent = networkInfo;
            }}
        }}
        
        function updateLastUpdate() {{
            const now = new Date().toLocaleTimeString();
            document.getElementById('last-update').textContent = `Last update: ${{now}}`;
        }}
        
        // Initialize when page loads
        window.addEventListener('load', initDashboard);
    </script>
</body>
</html>
        '''
    
    def export_metrics_data(self, output_file: str = "metrics_export.json") -> str:
        """Export all collected metrics data"""
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'network_info': {
                'num_neurons': self.network.config.num_neurons,
                'config': self.config
            },
            'metrics_history': {}
        }
        
        # Convert MetricData objects to dictionaries
        for metric_name, history in self.metrics_history.items():
            export_data['metrics_history'][metric_name] = [
                {
                    'timestamp': data.timestamp,
                    'value': data.value,
                    'metric_type': data.metric_type,
                    'metadata': data.metadata
                }
                for data in history
            ]
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        return output_file


class PerformanceTracker:
    """Track performance metrics"""
    
    def __init__(self):
        self.frame_times = []
        self.last_frame_time = time.time()
    
    def mark_frame(self):
        """Mark a frame for FPS calculation"""
        current_time = time.time()
        frame_time = current_time - self.last_frame_time
        self.frame_times.append(frame_time)
        
        # Keep only last 60 frames
        if len(self.frame_times) > 60:
            self.frame_times.pop(0)
        
        self.last_frame_time = current_time
    
    def get_fps(self) -> Optional[float]:
        """Get current FPS"""
        if len(self.frame_times) < 2:
            return None
        
        avg_frame_time = np.mean(self.frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0


class ActivationTracker:
    """Track neuron activations"""
    
    def __init__(self, network):
        self.network = network
        self.current_activations = None
    
    def update_activations(self, activations: np.ndarray):
        """Update current activations"""
        self.current_activations = activations.copy()
    
    def get_current_activations(self) -> Optional[np.ndarray]:
        """Get current activation levels"""
        # Try to get activations from network
        if hasattr(self.network, 'get_activations'):
            return self.network.get_activations()
        elif self.current_activations is not None:
            return self.current_activations
        else:
            # Generate placeholder activations
            return np.random.uniform(0, 1, self.network.config.num_neurons)


class CausalTracker:
    """Track causal reasoning strength"""
    
    def __init__(self):
        self.current_strength = 0.0
        self.active_chains = 0
    
    def update_causal_data(self, strength: float, chains: int):
        """Update causal tracking data"""
        self.current_strength = strength
        self.active_chains = chains
    
    def get_current_strength(self) -> Optional[Dict[str, Any]]:
        """Get current causal strength data"""
        return {
            'average_strength': self.current_strength,
            'active_chains': self.active_chains,
            'max_strength': self.current_strength * 1.2  # Placeholder
        }


class ChartManager:
    """Manage chart configurations and themes"""
    
    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings
    
    def get_chart_config(self, chart_type: str) -> Dict[str, Any]:
        """Get configuration for specific chart type"""
        base_config = {
            'responsive': True,
            'maintainAspectRatio': False,
            'animation': {
                'duration': 200 * self.settings.get('animation_speed', 1.0)
            }
        }
        
        return base_config


class AlertSystem:
    """Handle alerting for metric thresholds"""
    
    def __init__(self):
        self.alert_thresholds = {
            'agi_score': {'min': 0.1, 'max': 1.0},
            'memory_usage': {'max': 1000},  # MB
            'fps': {'min': 10}
        }
        self.active_alerts = set()
    
    def check_metric(self, metric_name: str, data: MetricData):
        """Check metric against thresholds"""
        if metric_name not in self.alert_thresholds:
            return
        
        thresholds = self.alert_thresholds[metric_name]
        value = data.value
        
        # Check min threshold
        if 'min' in thresholds and value < thresholds['min']:
            alert_id = f"{metric_name}_low"
            if alert_id not in self.active_alerts:
                self.active_alerts.add(alert_id)
                print(f"ALERT: {metric_name} below threshold: {value}")
        
        # Check max threshold
        if 'max' in thresholds and value > thresholds['max']:
            alert_id = f"{metric_name}_high"
            if alert_id not in self.active_alerts:
                self.active_alerts.add(alert_id)
                print(f"ALERT: {metric_name} above threshold: {value}")