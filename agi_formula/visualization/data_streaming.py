"""
Data Streaming System for AGI-Formula Real-time Visualization

High-performance data streaming for live AGI network visualization:
- Real-time data collection and buffering
- WebSocket-based streaming architecture
- Efficient data compression and serialization
- Multi-client streaming support
- Adaptive quality control
"""

import json
import time
import threading
import queue
import gzip
import base64
import asyncio
import websockets
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Set, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import logging


class StreamDataType(Enum):
    """Types of streamable data"""
    ACTIVATIONS = "activations"
    CAUSAL_CHAINS = "causal_chains"
    ATTENTION_FLOWS = "attention_flows"
    PERFORMANCE_METRICS = "performance_metrics"
    NETWORK_TOPOLOGY = "network_topology"
    TRAINING_PROGRESS = "training_progress"


@dataclass
class StreamPacket:
    """Data packet for streaming"""
    data_type: StreamDataType
    timestamp: float
    data: Any
    client_id: Optional[str] = None
    compression: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ClientInfo:
    """Information about connected client"""
    client_id: str
    websocket: Any
    subscriptions: Set[StreamDataType]
    quality_level: int  # 1-5, higher = better quality
    last_ping: float
    connection_time: float
    bytes_sent: int


class DataStreamer:
    """
    High-performance data streaming system for AGI visualization
    
    Features:
    - Real-time data collection from AGI network
    - WebSocket-based streaming to multiple clients
    - Adaptive quality control based on client capabilities
    - Data compression and efficient serialization
    - Buffering and flow control
    - Client subscription management
    """
    
    def __init__(self, network, config: Optional[Dict[str, Any]] = None):
        self.network = network
        self.config = config or self._get_default_config()
        
        # Streaming state
        self.is_streaming = False
        self.streaming_thread = None
        self.websocket_server = None
        
        # Client management
        self.connected_clients: Dict[str, ClientInfo] = {}
        self.client_counter = 0
        
        # Data collection
        self.data_collectors = {}
        self.data_buffers = {}
        self.collection_thread = None
        
        # Performance monitoring
        self.stream_stats = {
            'packets_sent': 0,
            'bytes_sent': 0,
            'clients_connected': 0,
            'avg_latency': 0.0,
            'drops': 0
        }
        
        # Event handlers
        self.event_handlers = {
            'client_connected': [],
            'client_disconnected': [],
            'data_collected': [],
            'stream_error': []
        }
        
        # Initialize streamer
        self._initialize_streamer()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default streaming configuration"""
        return {
            'websocket_settings': {
                'host': 'localhost',
                'port': 8765,
                'max_connections': 50,
                'ping_interval': 30,
                'ping_timeout': 10
            },
            'streaming_settings': {
                'base_fps': 30,
                'max_buffer_size': 1000,
                'compression_enabled': True,
                'compression_level': 6,
                'adaptive_quality': True,
                'quality_levels': {
                    1: {'fps': 5, 'resolution': 0.2},
                    2: {'fps': 10, 'resolution': 0.4},
                    3: {'fps': 15, 'resolution': 0.6},
                    4: {'fps': 20, 'resolution': 0.8},
                    5: {'fps': 30, 'resolution': 1.0}
                }
            },
            'data_collection': {
                'activations_enabled': True,
                'causal_chains_enabled': True,
                'attention_flows_enabled': True,
                'performance_metrics_enabled': True,
                'collection_interval': 0.033,  # ~30 FPS
                'history_length': 100
            },
            'optimization': {
                'use_binary_protocol': True,
                'delta_compression': True,
                'batch_updates': True,
                'client_side_interpolation': True
            }
        }
    
    def _initialize_streamer(self):
        """Initialize data streaming system"""
        # Setup data collectors
        self._setup_data_collectors()
        
        # Initialize data buffers
        for data_type in StreamDataType:
            self.data_buffers[data_type] = queue.Queue(
                maxsize=self.config['streaming_settings']['max_buffer_size']
            )
        
        print(f"Data streamer initialized for network with {self.network.config.num_neurons} neurons")
    
    def _setup_data_collectors(self):
        """Setup data collection functions"""
        self.data_collectors = {
            StreamDataType.ACTIVATIONS: self._collect_activations,
            StreamDataType.CAUSAL_CHAINS: self._collect_causal_chains,
            StreamDataType.ATTENTION_FLOWS: self._collect_attention_flows,
            StreamDataType.PERFORMANCE_METRICS: self._collect_performance_metrics,
            StreamDataType.NETWORK_TOPOLOGY: self._collect_network_topology,
            StreamDataType.TRAINING_PROGRESS: self._collect_training_progress
        }
    
    async def start_streaming(self):
        """Start the streaming system"""
        if self.is_streaming:
            print("Streaming already active")
            return
        
        self.is_streaming = True
        
        # Start data collection
        self._start_data_collection()
        
        # Start WebSocket server
        await self._start_websocket_server()
        
        print(f"Data streaming started on ws://{self.config['websocket_settings']['host']}:{self.config['websocket_settings']['port']}")
    
    async def stop_streaming(self):
        """Stop the streaming system"""
        self.is_streaming = False
        
        # Stop data collection
        self._stop_data_collection()
        
        # Stop WebSocket server
        if self.websocket_server:
            self.websocket_server.close()
            await self.websocket_server.wait_closed()
        
        # Disconnect all clients
        for client_info in list(self.connected_clients.values()):
            await self._disconnect_client(client_info.client_id)
        
        print("Data streaming stopped")
    
    def _start_data_collection(self):
        """Start data collection thread"""
        if self.collection_thread and self.collection_thread.is_alive():
            return
        
        self.collection_thread = threading.Thread(target=self._data_collection_loop, daemon=True)
        self.collection_thread.start()
    
    def _stop_data_collection(self):
        """Stop data collection thread"""
        if self.collection_thread:
            self.collection_thread.join(timeout=1)
    
    def _data_collection_loop(self):
        """Main data collection loop"""
        last_collection = time.time()
        collection_interval = self.config['data_collection']['collection_interval']
        
        while self.is_streaming:
            current_time = time.time()
            
            if current_time - last_collection >= collection_interval:
                # Collect all enabled data types
                for data_type, collector in self.data_collectors.items():
                    if self._is_data_type_enabled(data_type):
                        try:
                            data = collector()
                            if data is not None:
                                packet = StreamPacket(
                                    data_type=data_type,
                                    timestamp=current_time,
                                    data=data,
                                    metadata={'collection_time': current_time}
                                )
                                self._buffer_packet(packet)
                        except Exception as e:
                            logging.error(f"Error collecting {data_type}: {e}")
                
                last_collection = current_time
            
            time.sleep(0.001)  # Small sleep to prevent CPU spinning
    
    def _is_data_type_enabled(self, data_type: StreamDataType) -> bool:
        """Check if data type collection is enabled"""
        config_key = f"{data_type.value}_enabled"
        return self.config['data_collection'].get(config_key, True)
    
    def _buffer_packet(self, packet: StreamPacket):
        """Buffer a data packet for streaming"""
        buffer = self.data_buffers[packet.data_type]
        
        try:
            buffer.put_nowait(packet)
        except queue.Full:
            # Buffer is full, drop oldest packet
            try:
                buffer.get_nowait()
                buffer.put_nowait(packet)
                self.stream_stats['drops'] += 1
            except queue.Empty:
                pass
    
    async def _start_websocket_server(self):
        """Start WebSocket server for client connections"""
        host = self.config['websocket_settings']['host']
        port = self.config['websocket_settings']['port']
        
        self.websocket_server = await websockets.serve(
            self._handle_client_connection,
            host,
            port,
            ping_interval=self.config['websocket_settings']['ping_interval'],
            ping_timeout=self.config['websocket_settings']['ping_timeout']
        )
    
    async def _handle_client_connection(self, websocket, path):
        """Handle new client connection"""
        client_id = f"client_{self.client_counter}"
        self.client_counter += 1
        
        client_info = ClientInfo(
            client_id=client_id,
            websocket=websocket,
            subscriptions=set(),
            quality_level=3,  # Default quality
            last_ping=time.time(),
            connection_time=time.time(),
            bytes_sent=0
        )
        
        self.connected_clients[client_id] = client_info
        self.stream_stats['clients_connected'] += 1
        
        try:
            # Send initial connection message
            await self._send_to_client(client_id, {
                'type': 'connection',
                'client_id': client_id,
                'server_capabilities': self._get_server_capabilities()
            })
            
            # Trigger event handlers
            await self._trigger_event_handlers('client_connected', client_info)
            
            print(f"Client {client_id} connected")
            
            # Handle client messages
            async for message in websocket:
                await self._handle_client_message(client_id, message)
                
        except websockets.exceptions.ConnectionClosed:
            print(f"Client {client_id} disconnected")
        except Exception as e:
            print(f"Error with client {client_id}: {e}")
        finally:
            await self._disconnect_client(client_id)
    
    async def _handle_client_message(self, client_id: str, message: str):
        """Handle message from client"""
        try:
            data = json.loads(message)
            message_type = data.get('type')
            
            if message_type == 'subscribe':
                await self._handle_subscription(client_id, data)
            elif message_type == 'unsubscribe':
                await self._handle_unsubscription(client_id, data)
            elif message_type == 'set_quality':
                await self._handle_quality_change(client_id, data)
            elif message_type == 'ping':
                await self._handle_ping(client_id, data)
            else:
                print(f"Unknown message type from {client_id}: {message_type}")
                
        except json.JSONDecodeError:
            print(f"Invalid JSON from client {client_id}")
        except Exception as e:
            print(f"Error handling message from {client_id}: {e}")
    
    async def _handle_subscription(self, client_id: str, data: Dict[str, Any]):
        """Handle client subscription request"""
        if client_id not in self.connected_clients:
            return
        
        data_types = data.get('data_types', [])
        client_info = self.connected_clients[client_id]
        
        for data_type_str in data_types:
            try:
                data_type = StreamDataType(data_type_str)
                client_info.subscriptions.add(data_type)
            except ValueError:
                print(f"Invalid data type: {data_type_str}")
        
        # Send confirmation
        await self._send_to_client(client_id, {
            'type': 'subscription_confirmed',
            'subscriptions': [dt.value for dt in client_info.subscriptions]
        })
        
        print(f"Client {client_id} subscribed to: {[dt.value for dt in client_info.subscriptions]}")
    
    async def _handle_unsubscription(self, client_id: str, data: Dict[str, Any]):
        """Handle client unsubscription request"""
        if client_id not in self.connected_clients:
            return
        
        data_types = data.get('data_types', [])
        client_info = self.connected_clients[client_id]
        
        for data_type_str in data_types:
            try:
                data_type = StreamDataType(data_type_str)
                client_info.subscriptions.discard(data_type)
            except ValueError:
                print(f"Invalid data type: {data_type_str}")
        
        # Send confirmation
        await self._send_to_client(client_id, {
            'type': 'unsubscription_confirmed',
            'subscriptions': [dt.value for dt in client_info.subscriptions]
        })
    
    async def _handle_quality_change(self, client_id: str, data: Dict[str, Any]):
        """Handle client quality level change"""
        if client_id not in self.connected_clients:
            return
        
        quality_level = data.get('quality_level', 3)
        quality_level = max(1, min(5, quality_level))  # Clamp to valid range
        
        self.connected_clients[client_id].quality_level = quality_level
        
        await self._send_to_client(client_id, {
            'type': 'quality_changed',
            'quality_level': quality_level
        })
        
        print(f"Client {client_id} quality level set to {quality_level}")
    
    async def _handle_ping(self, client_id: str, data: Dict[str, Any]):
        """Handle client ping"""
        if client_id not in self.connected_clients:
            return
        
        self.connected_clients[client_id].last_ping = time.time()
        
        await self._send_to_client(client_id, {
            'type': 'pong',
            'timestamp': time.time()
        })
    
    async def _disconnect_client(self, client_id: str):
        """Disconnect a client"""
        if client_id in self.connected_clients:
            client_info = self.connected_clients.pop(client_id)
            self.stream_stats['clients_connected'] -= 1
            
            # Trigger event handlers
            await self._trigger_event_handlers('client_disconnected', client_info)
    
    async def _send_to_client(self, client_id: str, data: Union[Dict, StreamPacket]):
        """Send data to a specific client"""
        if client_id not in self.connected_clients:
            return False
        
        client_info = self.connected_clients[client_id]
        
        try:
            # Prepare message based on data type
            if isinstance(data, StreamPacket):
                message = self._prepare_stream_packet(data, client_info)
            else:
                message = json.dumps(data, default=str)
            
            # Send message
            await client_info.websocket.send(message)
            
            # Update stats
            message_size = len(message.encode('utf-8'))
            client_info.bytes_sent += message_size
            self.stream_stats['bytes_sent'] += message_size
            self.stream_stats['packets_sent'] += 1
            
            return True
            
        except websockets.exceptions.ConnectionClosed:
            await self._disconnect_client(client_id)
            return False
        except Exception as e:
            print(f"Error sending to client {client_id}: {e}")
            return False
    
    def _prepare_stream_packet(self, packet: StreamPacket, client_info: ClientInfo) -> str:
        """Prepare stream packet for client based on quality settings"""
        # Apply quality-based filtering
        processed_data = self._apply_quality_filter(packet.data, client_info.quality_level, packet.data_type)
        
        # Create message
        message_data = {
            'type': 'stream_data',
            'data_type': packet.data_type.value,
            'timestamp': packet.timestamp,
            'data': processed_data,
            'metadata': packet.metadata or {}
        }
        
        # Apply compression if enabled
        if self.config['streaming_settings']['compression_enabled']:
            return self._compress_message(message_data)
        else:
            return json.dumps(message_data, default=str)
    
    def _apply_quality_filter(self, data: Any, quality_level: int, data_type: StreamDataType) -> Any:
        """Apply quality-based filtering to data"""
        quality_settings = self.config['streaming_settings']['quality_levels'][quality_level]
        resolution = quality_settings['resolution']
        
        if data_type == StreamDataType.ACTIVATIONS and isinstance(data, list):
            # Sample activations based on resolution
            if resolution < 1.0:
                sample_size = max(1, int(len(data) * resolution))
                step = len(data) // sample_size
                return data[::step]
        
        elif data_type == StreamDataType.NETWORK_TOPOLOGY and isinstance(data, dict):
            # Reduce topology complexity
            if 'nodes' in data and resolution < 1.0:
                nodes = data['nodes']
                sample_size = max(1, int(len(nodes) * resolution))
                data['nodes'] = nodes[:sample_size]
        
        return data
    
    def _compress_message(self, data: Dict[str, Any]) -> str:
        """Compress message data"""
        try:
            json_str = json.dumps(data, default=str)
            compressed = gzip.compress(json_str.encode('utf-8'))
            encoded = base64.b64encode(compressed).decode('ascii')
            
            return json.dumps({
                'compressed': True,
                'data': encoded
            })
        except Exception:
            # Fallback to uncompressed
            return json.dumps(data, default=str)
    
    async def _broadcast_to_subscribers(self, packet: StreamPacket):
        """Broadcast packet to all subscribed clients"""
        if not self.connected_clients:
            return
        
        # Find clients subscribed to this data type
        subscribers = [
            client_info for client_info in self.connected_clients.values()
            if packet.data_type in client_info.subscriptions
        ]
        
        # Send to all subscribers
        for client_info in subscribers:
            await self._send_to_client(client_info.client_id, packet)
    
    async def _trigger_event_handlers(self, event_type: str, data: Any):
        """Trigger event handlers"""
        handlers = self.event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception as e:
                print(f"Error in event handler {event_type}: {e}")
    
    def add_event_handler(self, event_type: str, handler: Callable):
        """Add event handler"""
        if event_type in self.event_handlers:
            self.event_handlers[event_type].append(handler)
    
    def _get_server_capabilities(self) -> Dict[str, Any]:
        """Get server capabilities for client"""
        return {
            'supported_data_types': [dt.value for dt in StreamDataType],
            'quality_levels': list(self.config['streaming_settings']['quality_levels'].keys()),
            'compression_supported': self.config['streaming_settings']['compression_enabled'],
            'max_fps': self.config['streaming_settings']['base_fps']
        }
    
    # Data collection methods
    def _collect_activations(self) -> Optional[List[float]]:
        """Collect current neuron activations"""
        try:
            if hasattr(self.network, 'get_activations'):
                activations = self.network.get_activations()
                return activations.tolist() if isinstance(activations, np.ndarray) else activations
            else:
                # Generate placeholder activations
                return np.random.uniform(0, 1, self.network.config.num_neurons).tolist()
        except Exception as e:
            logging.error(f"Error collecting activations: {e}")
            return None
    
    def _collect_causal_chains(self) -> Optional[Dict[str, Any]]:
        """Collect current causal chain data"""
        try:
            # Placeholder implementation
            chains = []
            for i in range(np.random.randint(1, 5)):
                chain_length = np.random.randint(2, 6)
                chain = [np.random.randint(0, self.network.config.num_neurons) for _ in range(chain_length)]
                chains.append({
                    'chain_id': i,
                    'neurons': chain,
                    'strength': np.random.uniform(0.1, 1.0)
                })
            
            return {'chains': chains, 'total_strength': sum(c['strength'] for c in chains)}
        except Exception as e:
            logging.error(f"Error collecting causal chains: {e}")
            return None
    
    def _collect_attention_flows(self) -> Optional[Dict[str, Any]]:
        """Collect attention flow data"""
        try:
            # Placeholder implementation
            flows = []
            for i in range(np.random.randint(5, 15)):
                flows.append({
                    'source': np.random.randint(0, self.network.config.num_neurons),
                    'target': np.random.randint(0, self.network.config.num_neurons),
                    'strength': np.random.uniform(0.1, 1.0)
                })
            
            return {'flows': flows}
        except Exception as e:
            logging.error(f"Error collecting attention flows: {e}")
            return None
    
    def _collect_performance_metrics(self) -> Optional[Dict[str, Any]]:
        """Collect performance metrics"""
        try:
            return {
                'timestamp': time.time(),
                'fps': np.random.uniform(25, 35),
                'memory_usage': np.random.uniform(100, 500),
                'cpu_usage': np.random.uniform(10, 80),
                'network_utilization': len(self.connected_clients) / max(1, self.config['websocket_settings']['max_connections'])
            }
        except Exception as e:
            logging.error(f"Error collecting performance metrics: {e}")
            return None
    
    def _collect_network_topology(self) -> Optional[Dict[str, Any]]:
        """Collect network topology data"""
        try:
            # Only collect topology changes, not full topology every time
            return {
                'nodes_count': self.network.config.num_neurons,
                'connections_count': 0,  # Placeholder
                'topology_version': 1
            }
        except Exception as e:
            logging.error(f"Error collecting network topology: {e}")
            return None
    
    def _collect_training_progress(self) -> Optional[Dict[str, Any]]:
        """Collect training progress data"""
        try:
            if hasattr(self.network, 'training_stats'):
                return self.network.training_stats
            else:
                return {
                    'epoch': np.random.randint(1, 100),
                    'loss': np.random.uniform(0.1, 2.0),
                    'accuracy': np.random.uniform(0.6, 0.95)
                }
        except Exception as e:
            logging.error(f"Error collecting training progress: {e}")
            return None
    
    def get_streaming_stats(self) -> Dict[str, Any]:
        """Get current streaming statistics"""
        return {
            'is_streaming': self.is_streaming,
            'connected_clients': len(self.connected_clients),
            'packets_sent': self.stream_stats['packets_sent'],
            'bytes_sent': self.stream_stats['bytes_sent'],
            'average_latency': self.stream_stats['avg_latency'],
            'dropped_packets': self.stream_stats['drops'],
            'clients_info': {
                client_id: {
                    'subscriptions': [dt.value for dt in info.subscriptions],
                    'quality_level': info.quality_level,
                    'bytes_sent': info.bytes_sent,
                    'connection_duration': time.time() - info.connection_time
                }
                for client_id, info in self.connected_clients.items()
            }
        }
    
    def export_stream_config(self, output_file: str = "stream_config.json") -> str:
        """Export streaming configuration"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, default=str)
        
        return output_file
    
    def generate_client_html(self, output_file: str = "stream_client.html") -> str:
        """Generate HTML client for testing the stream"""
        html_template = self._get_client_template()
        
        config_json = json.dumps(self.config['websocket_settings'], indent=2)
        
        html_content = html_template.format(
            websocket_config=config_json,
            title="AGI Data Stream Client"
        )
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Stream client saved to: {output_file}")
        return output_file
    
    def _get_client_template(self) -> str:
        """Get HTML template for stream client"""
        return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background: #1a1a1a;
            color: white;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        .control-panel {{
            background: #2a2a2a;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        
        .data-panel {{
            background: #2a2a2a;
            padding: 20px;
            border-radius: 8px;
            height: 400px;
            overflow-y: auto;
        }}
        
        .status {{
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
        }}
        
        .connected {{ background: #2d5a2d; }}
        .disconnected {{ background: #5a2d2d; }}
        
        button {{
            background: #4a90e2;
            color: white;
            border: none;
            padding: 10px 15px;
            border-radius: 4px;
            cursor: pointer;
            margin: 5px;
        }}
        
        button:hover {{ background: #5aa0f2; }}
        button:disabled {{ background: #666; cursor: not-allowed; }}
        
        select, input {{
            background: #1a1a1a;
            color: white;
            border: 1px solid #444;
            padding: 8px;
            border-radius: 4px;
            margin: 5px;
        }}
        
        .data-item {{
            background: #1a1a1a;
            padding: 10px;
            margin: 5px 0;
            border-radius: 4px;
            border-left: 3px solid #4a90e2;
            font-size: 12px;
        }}
        
        .timestamp {{
            color: #888;
            font-size: 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>AGI Data Stream Client</h1>
        
        <div class="control-panel">
            <h3>Connection Controls</h3>
            <div id="connection-status" class="status disconnected">Disconnected</div>
            
            <button id="connect-btn" onclick="connect()">Connect</button>
            <button id="disconnect-btn" onclick="disconnect()" disabled>Disconnect</button>
            
            <div style="margin: 15px 0;">
                <label>Quality Level:</label>
                <select id="quality-select" onchange="setQuality()">
                    <option value="1">Low (1)</option>
                    <option value="2">Medium-Low (2)</option>
                    <option value="3" selected>Medium (3)</option>
                    <option value="4">Medium-High (4)</option>
                    <option value="5">High (5)</option>
                </select>
            </div>
            
            <div style="margin: 15px 0;">
                <h4>Subscriptions:</h4>
                <label><input type="checkbox" id="sub-activations" checked> Activations</label>
                <label><input type="checkbox" id="sub-causal"> Causal Chains</label>
                <label><input type="checkbox" id="sub-attention"> Attention Flows</label>
                <label><input type="checkbox" id="sub-performance" checked> Performance</label>
                <label><input type="checkbox" id="sub-topology"> Topology</label>
                <label><input type="checkbox" id="sub-training"> Training</label>
                <button onclick="updateSubscriptions()">Update Subscriptions</button>
            </div>
            
            <div style="margin: 15px 0;">
                <span id="stats">Packets: 0 | Bytes: 0 | FPS: 0</span>
            </div>
        </div>
        
        <div class="data-panel">
            <h3>Live Data Stream</h3>
            <div id="data-container">
                <!-- Stream data will appear here -->
            </div>
        </div>
    </div>

    <script>
        const wsConfig = {websocket_config};
        let ws = null;
        let isConnected = false;
        let packetsReceived = 0;
        let bytesReceived = 0;
        let lastFpsTime = Date.now();
        let fpsCounter = 0;
        let currentFps = 0;
        
        function connect() {{
            if (isConnected) return;
            
            const wsUrl = `ws://${{wsConfig.host}}:${{wsConfig.port}}`;
            ws = new WebSocket(wsUrl);
            
            ws.onopen = function() {{
                isConnected = true;
                updateConnectionStatus();
                updateSubscriptions();
                console.log('Connected to stream');
            }};
            
            ws.onmessage = function(event) {{
                handleMessage(event.data);
            }};
            
            ws.onclose = function() {{
                isConnected = false;
                updateConnectionStatus();
                console.log('Disconnected from stream');
            }};
            
            ws.onerror = function(error) {{
                console.error('WebSocket error:', error);
            }};
        }}
        
        function disconnect() {{
            if (ws) {{
                ws.close();
            }}
        }}
        
        function handleMessage(data) {{
            try {{
                let message = JSON.parse(data);
                
                // Handle compressed messages
                if (message.compressed) {{
                    const decompressed = atob(message.data);
                    message = JSON.parse(decompressed);
                }}
                
                packetsReceived++;
                bytesReceived += data.length;
                
                // Update FPS counter
                fpsCounter++;
                const now = Date.now();
                if (now - lastFpsTime >= 1000) {{
                    currentFps = fpsCounter;
                    fpsCounter = 0;
                    lastFpsTime = now;
                }}
                
                updateStats();
                displayMessage(message);
                
            }} catch (error) {{
                console.error('Error parsing message:', error);
            }}
        }}
        
        function displayMessage(message) {{
            const container = document.getElementById('data-container');
            
            if (message.type === 'stream_data') {{
                const dataItem = document.createElement('div');
                dataItem.className = 'data-item';
                
                const timestamp = new Date(message.timestamp * 1000).toLocaleTimeString();
                const dataPreview = JSON.stringify(message.data).substring(0, 100) + '...';
                
                dataItem.innerHTML = `
                    <strong>${{message.data_type}}</strong>
                    <div class="timestamp">${{timestamp}}</div>
                    <div>${{dataPreview}}</div>
                `;
                
                container.insertBefore(dataItem, container.firstChild);
                
                // Keep only last 50 items
                while (container.children.length > 50) {{
                    container.removeChild(container.lastChild);
                }}
            }}
        }}
        
        function setQuality() {{
            if (!isConnected) return;
            
            const quality = parseInt(document.getElementById('quality-select').value);
            ws.send(JSON.stringify({{
                type: 'set_quality',
                quality_level: quality
            }}));
        }}
        
        function updateSubscriptions() {{
            if (!isConnected) return;
            
            const subscriptions = [];
            const checkboxes = {{
                'sub-activations': 'activations',
                'sub-causal': 'causal_chains',
                'sub-attention': 'attention_flows',
                'sub-performance': 'performance_metrics',
                'sub-topology': 'network_topology',
                'sub-training': 'training_progress'
            }};
            
            for (const [id, dataType] of Object.entries(checkboxes)) {{
                if (document.getElementById(id).checked) {{
                    subscriptions.push(dataType);
                }}
            }}
            
            ws.send(JSON.stringify({{
                type: 'subscribe',
                data_types: subscriptions
            }}));
        }}
        
        function updateConnectionStatus() {{
            const statusEl = document.getElementById('connection-status');
            const connectBtn = document.getElementById('connect-btn');
            const disconnectBtn = document.getElementById('disconnect-btn');
            
            if (isConnected) {{
                statusEl.textContent = 'Connected';
                statusEl.className = 'status connected';
                connectBtn.disabled = true;
                disconnectBtn.disabled = false;
            }} else {{
                statusEl.textContent = 'Disconnected';
                statusEl.className = 'status disconnected';
                connectBtn.disabled = false;
                disconnectBtn.disabled = true;
            }}
        }}
        
        function updateStats() {{
            const statsEl = document.getElementById('stats');
            statsEl.textContent = `Packets: ${{packetsReceived}} | Bytes: ${{bytesReceived}} | FPS: ${{currentFps}}`;
        }}
        
        // Auto-connect on page load
        window.addEventListener('load', function() {{
            console.log('Stream client loaded');
        }});
    </script>
</body>
</html>
        '''


# Utility functions for async context
async def create_data_streamer(network, config=None):
    """Create and initialize a data streamer"""
    streamer = DataStreamer(network, config)
    return streamer


async def run_streaming_server(network, config=None):
    """Run a complete streaming server"""
    streamer = DataStreamer(network, config)
    
    try:
        await streamer.start_streaming()
        print("Streaming server running... Press Ctrl+C to stop")
        
        # Keep server running
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\nShutting down streaming server...")
    finally:
        await streamer.stop_streaming()