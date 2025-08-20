"""Safety controller for secure self-modification in AGI-Formula."""

from typing import Dict, List, Optional, Tuple, Set, Any, Callable
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict, deque
import copy
import time
import hashlib
import json


@dataclass
class SafetyBounds:
    """Define safety boundaries for network modifications."""
    max_neurons: int = 1000
    max_connections_per_neuron: int = 50
    min_network_size: int = 10
    max_weight_magnitude: float = 10.0
    min_performance_threshold: float = 0.1
    max_memory_usage_mb: float = 500.0
    max_modification_frequency: int = 10  # per minute
    stability_window: int = 50  # epochs to check stability


@dataclass
class SafetyViolation:
    """Represents a detected safety violation."""
    violation_type: str
    severity: str  # "low", "medium", "high", "critical"
    description: str
    affected_neurons: List[int]
    timestamp: float
    auto_rollback: bool = False


@dataclass
class NetworkSnapshot:
    """Complete snapshot of network state for rollback."""
    snapshot_id: str
    timestamp: float
    network_state: Dict[str, Any]
    performance_metrics: Dict[str, float]
    metadata: Dict[str, Any]
    verification_hash: str


class SafetyController:
    """
    Advanced safety controller that ensures secure self-modification.
    
    Key safety features:
    - Bounded modifications within safe parameters
    - Performance monitoring and automatic rollback
    - Network state snapshots for recovery
    - Anomaly detection in network behavior
    - Resource usage monitoring
    - Modification rate limiting
    """
    
    def __init__(self, network: 'Network', safety_bounds: SafetyBounds = None):
        """Initialize safety controller."""
        self.network = network
        self.safety_bounds = safety_bounds or SafetyBounds()
        
        # Safety monitoring
        self.violations: List[SafetyViolation] = []
        self.safety_events: deque = deque(maxlen=1000)
        
        # Network snapshots for rollback
        self.snapshots: Dict[str, NetworkSnapshot] = {}
        self.current_snapshot_id: Optional[str] = None
        self.auto_snapshot_interval = 10  # Create snapshot every N modifications
        self.modification_counter = 0
        
        # Performance monitoring
        self.performance_baseline: Optional[Dict[str, float]] = None
        self.performance_history: deque = deque(maxlen=100)
        self.stability_violations = 0
        
        # Anomaly detection
        self.behavior_patterns: Dict[str, List[float]] = defaultdict(list)
        self.anomaly_threshold = 3.0  # Standard deviations
        
        # Rate limiting
        self.modification_timestamps: deque = deque(maxlen=100)
        
        # Safety locks
        self.emergency_stop = False
        self.safety_mode = "normal"  # "normal", "cautious", "emergency"
        self.prohibited_modifications: Set[str] = set()
        
        # Verification
        self.integrity_checks_enabled = True
        self.verification_failures = 0
        
    def validate_modification_proposal(self, proposal: 'ModificationProposal') -> Tuple[bool, List[str]]:
        """
        Validate a modification proposal against safety constraints.
        
        Args:
            proposal: The proposed modification
            
        Returns:
            Tuple of (is_safe, list_of_concerns)
        """
        concerns = []
        
        # Check emergency stop
        if self.emergency_stop:
            return False, ["Emergency stop is active"]
        
        # Check prohibited modifications
        if proposal.modification_type in self.prohibited_modifications:
            return False, [f"Modification type '{proposal.modification_type}' is prohibited"]
        
        # Rate limiting
        if not self._check_modification_rate():
            concerns.append("Modification rate limit exceeded")
        
        # Resource bounds checking
        resource_concerns = self._check_resource_bounds(proposal)
        concerns.extend(resource_concerns)
        
        # Network integrity checks
        integrity_concerns = self._check_network_integrity(proposal)
        concerns.extend(integrity_concerns)
        
        # Performance impact assessment
        performance_concerns = self._assess_performance_impact(proposal)
        concerns.extend(performance_concerns)
        
        # Safety score validation
        if proposal.safety_score < 0.5:
            concerns.append(f"Safety score too low: {proposal.safety_score}")
        
        # Critical concerns make modification unsafe
        critical_concerns = [c for c in concerns if "critical" in c.lower() or "emergency" in c.lower()]
        is_safe = len(critical_concerns) == 0
        
        # Log safety assessment
        self._log_safety_event("modification_validation", {
            "proposal_id": proposal.modification_id,
            "is_safe": is_safe,
            "concerns": concerns,
            "safety_score": proposal.safety_score
        })
        
        return is_safe, concerns
    
    def _check_modification_rate(self) -> bool:
        """Check if modification rate is within safe limits."""
        current_time = time.time()
        
        # Remove old timestamps (older than 1 minute)
        cutoff_time = current_time - 60.0
        while self.modification_timestamps and self.modification_timestamps[0] < cutoff_time:
            self.modification_timestamps.popleft()
        
        # Check if we're within rate limit
        if len(self.modification_timestamps) >= self.safety_bounds.max_modification_frequency:
            return False
        
        return True
    
    def _check_resource_bounds(self, proposal: 'ModificationProposal') -> List[str]:
        """Check resource usage bounds."""
        concerns = []
        
        current_neurons = len(self.network.neurons)
        
        # Check neuron count limits
        if proposal.modification_type.startswith('add_'):
            estimated_new_neurons = proposal.parameters.get('num_neurons', 1)
            if current_neurons + estimated_new_neurons > self.safety_bounds.max_neurons:
                concerns.append(f"CRITICAL: Would exceed max neurons ({self.safety_bounds.max_neurons})")
        
        # Check minimum network size
        if proposal.modification_type.startswith('remove_'):
            neurons_to_remove = len(proposal.target_neurons)
            if current_neurons - neurons_to_remove < self.safety_bounds.min_network_size:
                concerns.append(f"CRITICAL: Would violate minimum network size ({self.safety_bounds.min_network_size})")
        
        # Check memory usage (estimated)
        estimated_memory = self._estimate_memory_usage(proposal)
        if estimated_memory > self.safety_bounds.max_memory_usage_mb:
            concerns.append(f"Memory usage concern: {estimated_memory:.1f}MB > {self.safety_bounds.max_memory_usage_mb}MB")
        
        return concerns
    
    def _check_network_integrity(self, proposal: 'ModificationProposal') -> List[str]:
        """Check network structural integrity."""
        concerns = []
        
        # Don't modify input/output neurons
        for neuron_id in proposal.target_neurons:
            if neuron_id in self.network.input_neurons:
                concerns.append(f"CRITICAL: Cannot modify input neuron {neuron_id}")
            if neuron_id in self.network.output_neurons:
                concerns.append(f"CRITICAL: Cannot modify output neuron {neuron_id}")
        
        # Check if target neurons exist
        for neuron_id in proposal.target_neurons:
            if neuron_id not in self.network.neurons:
                concerns.append(f"Target neuron {neuron_id} does not exist")
        
        # Check for circular dependencies in proposed modifications
        if self._would_create_cycles(proposal):
            concerns.append("Proposed modification would create circular dependencies")
        
        return concerns
    
    def _assess_performance_impact(self, proposal: 'ModificationProposal') -> List[str]:
        """Assess potential performance impact."""
        concerns = []
        
        # Check if network performance is already below threshold
        if self.performance_baseline:
            current_performance = self._get_current_performance()
            baseline_accuracy = self.performance_baseline.get('prediction_accuracy', 0.0)
            current_accuracy = current_performance.get('prediction_accuracy', 0.0)
            
            if current_accuracy < self.safety_bounds.min_performance_threshold:
                concerns.append(f"CRITICAL: Current performance below threshold ({current_accuracy:.3f} < {self.safety_bounds.min_performance_threshold})")
        
        # Check stability window
        if len(self.performance_history) >= self.safety_bounds.stability_window:
            stability = self._assess_performance_stability()
            if stability < 0.3:
                concerns.append(f"Performance instability detected (stability: {stability:.3f})")
        
        return concerns
    
    def create_safety_snapshot(self, reason: str = "auto") -> str:
        """Create a complete snapshot of current network state."""
        snapshot_id = f"snapshot_{int(time.time())}_{len(self.snapshots)}"
        
        # Create deep copy of network state
        network_state = self._serialize_network_state()
        
        # Get current performance metrics
        performance_metrics = self._get_current_performance()
        
        # Create verification hash
        verification_data = {
            'network_state': network_state,
            'performance_metrics': performance_metrics,
            'timestamp': time.time()
        }
        verification_hash = hashlib.sha256(
            json.dumps(verification_data, sort_keys=True).encode()
        ).hexdigest()
        
        snapshot = NetworkSnapshot(
            snapshot_id=snapshot_id,
            timestamp=time.time(),
            network_state=network_state,
            performance_metrics=performance_metrics,
            metadata={'reason': reason, 'neuron_count': len(self.network.neurons)},
            verification_hash=verification_hash
        )
        
        self.snapshots[snapshot_id] = snapshot
        self.current_snapshot_id = snapshot_id
        
        # Cleanup old snapshots (keep last 10)
        if len(self.snapshots) > 10:
            oldest_id = min(self.snapshots.keys(), key=lambda x: self.snapshots[x].timestamp)
            del self.snapshots[oldest_id]
        
        self._log_safety_event("snapshot_created", {
            "snapshot_id": snapshot_id,
            "reason": reason,
            "neuron_count": len(self.network.neurons)
        })
        
        return snapshot_id
    
    def rollback_to_snapshot(self, snapshot_id: str, reason: str = "safety_violation") -> bool:
        """Rollback network to a previous safe state."""
        if snapshot_id not in self.snapshots:
            print(f"Snapshot {snapshot_id} not found")
            return False
        
        snapshot = self.snapshots[snapshot_id]
        
        # Verify snapshot integrity
        if not self._verify_snapshot_integrity(snapshot):
            print(f"Snapshot {snapshot_id} failed integrity check")
            return False
        
        try:
            # Restore network state
            self._restore_network_state(snapshot.network_state)
            
            # Log rollback
            self._log_safety_event("rollback_executed", {
                "snapshot_id": snapshot_id,
                "reason": reason,
                "timestamp": snapshot.timestamp
            })
            
            print(f"Successfully rolled back to snapshot {snapshot_id}")
            return True
            
        except Exception as e:
            print(f"Error during rollback: {e}")
            return False
    
    def monitor_modification_execution(self, proposal: 'ModificationProposal') -> None:
        """Monitor a modification as it's being executed."""
        # Record modification start
        self.modification_timestamps.append(time.time())
        self.modification_counter += 1
        
        # Create automatic snapshot if needed
        if self.modification_counter % self.auto_snapshot_interval == 0:
            self.create_safety_snapshot("auto_before_modification")
        
        # Enable monitoring mode
        self._enable_modification_monitoring(proposal)
    
    def evaluate_modification_safety(self, proposal: 'ModificationProposal', post_performance: Dict[str, float]) -> bool:
        """Evaluate safety after modification has been applied."""
        violations = []
        
        # Performance degradation check
        if self.performance_baseline:
            baseline_accuracy = self.performance_baseline.get('prediction_accuracy', 0.0)
            current_accuracy = post_performance.get('prediction_accuracy', 0.0)
            
            degradation = baseline_accuracy - current_accuracy
            if degradation > 0.2:  # 20% performance drop
                violations.append(SafetyViolation(
                    violation_type="performance_degradation",
                    severity="high",
                    description=f"Performance dropped by {degradation:.3f}",
                    affected_neurons=proposal.target_neurons,
                    timestamp=time.time(),
                    auto_rollback=True
                ))
        
        # Anomaly detection
        anomalies = self._detect_behavioral_anomalies(post_performance)
        violations.extend(anomalies)
        
        # Network integrity check
        integrity_violations = self._check_post_modification_integrity()
        violations.extend(integrity_violations)
        
        # Record violations
        self.violations.extend(violations)
        
        # Auto-rollback for critical violations
        critical_violations = [v for v in violations if v.severity == "critical" or v.auto_rollback]
        
        if critical_violations:
            print(f"Critical safety violations detected, initiating rollback")
            for violation in critical_violations:
                print(f"  - {violation.description}")
            
            if self.current_snapshot_id:
                return not self.rollback_to_snapshot(self.current_snapshot_id, "critical_violation")
        
        return len(violations) == 0
    
    def set_safety_mode(self, mode: str) -> None:
        """Set safety operation mode."""
        valid_modes = ["normal", "cautious", "emergency"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid safety mode: {mode}. Valid modes: {valid_modes}")
        
        old_mode = self.safety_mode
        self.safety_mode = mode
        
        # Adjust safety parameters based on mode
        if mode == "cautious":
            self.safety_bounds.max_modification_frequency = max(1, self.safety_bounds.max_modification_frequency // 2)
            self.anomaly_threshold = min(2.0, self.anomaly_threshold - 0.5)
        elif mode == "emergency":
            self.emergency_stop = True
            self.safety_bounds.max_modification_frequency = 0
        else:  # normal
            self.emergency_stop = False
            # Reset to defaults
        
        self._log_safety_event("safety_mode_changed", {
            "old_mode": old_mode,
            "new_mode": mode
        })
    
    def emergency_shutdown(self, reason: str) -> None:
        """Initiate emergency shutdown of all modifications."""
        self.emergency_stop = True
        self.safety_mode = "emergency"
        
        # Create emergency snapshot
        emergency_snapshot = self.create_safety_snapshot("emergency_shutdown")
        
        self._log_safety_event("emergency_shutdown", {
            "reason": reason,
            "snapshot_id": emergency_snapshot,
            "timestamp": time.time()
        })
        
        print(f"EMERGENCY SHUTDOWN: {reason}")
    
    # Helper methods
    def _estimate_memory_usage(self, proposal: 'ModificationProposal') -> float:
        """Estimate memory usage of proposed modification."""
        base_neuron_memory = 0.01  # MB per neuron (rough estimate)
        
        if proposal.modification_type.startswith('add_'):
            num_neurons = proposal.parameters.get('num_neurons', 1)
            return num_neurons * base_neuron_memory
        
        return 0.0
    
    def _would_create_cycles(self, proposal: 'ModificationProposal') -> bool:
        """Check if proposal would create circular dependencies."""
        # Simplified cycle detection - in practice would be more sophisticated
        return False
    
    def _assess_performance_stability(self) -> float:
        """Assess stability of recent performance."""
        if len(self.performance_history) < 5:
            return 1.0
        
        # Calculate coefficient of variation for recent performance
        recent_accuracies = [p.get('prediction_accuracy', 0.0) for p in list(self.performance_history)[-10:]]
        
        if not recent_accuracies:
            return 0.0
        
        mean_acc = np.mean(recent_accuracies)
        std_acc = np.std(recent_accuracies)
        
        if mean_acc == 0:
            return 0.0
        
        cv = std_acc / mean_acc
        stability = 1.0 / (1.0 + cv)  # Higher stability = lower coefficient of variation
        
        return stability
    
    def _serialize_network_state(self) -> Dict[str, Any]:
        """Serialize current network state for snapshot."""
        # This would serialize the complete network state
        # For now, return a placeholder
        return {
            'neurons': {nid: {'weights': neuron.weights.tolist() if hasattr(neuron, 'weights') else []} 
                       for nid, neuron in self.network.neurons.items()},
            'config': {
                'num_neurons': len(self.network.neurons),
                'input_neurons': list(self.network.input_neurons),
                'output_neurons': list(self.network.output_neurons)
            }
        }
    
    def _restore_network_state(self, network_state: Dict[str, Any]) -> None:
        """Restore network from serialized state."""
        # This would restore the complete network state
        # Implementation would depend on network architecture
        print("Network state restoration not fully implemented - placeholder")
    
    def _verify_snapshot_integrity(self, snapshot: NetworkSnapshot) -> bool:
        """Verify snapshot integrity using hash."""
        verification_data = {
            'network_state': snapshot.network_state,
            'performance_metrics': snapshot.performance_metrics,
            'timestamp': snapshot.timestamp
        }
        
        current_hash = hashlib.sha256(
            json.dumps(verification_data, sort_keys=True).encode()
        ).hexdigest()
        
        return current_hash == snapshot.verification_hash
    
    def _get_current_performance(self) -> Dict[str, float]:
        """Get current performance metrics."""
        # Placeholder - would get from network/trainer
        return {
            'prediction_accuracy': 0.75,
            'causal_consistency': 0.68,
            'attention_efficiency': 0.82
        }
    
    def _enable_modification_monitoring(self, proposal: 'ModificationProposal') -> None:
        """Enable enhanced monitoring during modification."""
        # Enhanced monitoring would be implemented here
        pass
    
    def _detect_behavioral_anomalies(self, performance: Dict[str, float]) -> List[SafetyViolation]:
        """Detect anomalies in network behavior."""
        violations = []
        
        for metric_name, value in performance.items():
            if metric_name not in self.behavior_patterns:
                self.behavior_patterns[metric_name] = []
            
            patterns = self.behavior_patterns[metric_name]
            patterns.append(value)
            
            # Keep only recent patterns
            if len(patterns) > 50:
                patterns.pop(0)
            
            # Anomaly detection using z-score
            if len(patterns) >= 10:
                mean_val = np.mean(patterns)
                std_val = np.std(patterns)
                
                if std_val > 0:
                    z_score = abs(value - mean_val) / std_val
                    
                    if z_score > self.anomaly_threshold:
                        violations.append(SafetyViolation(
                            violation_type="behavioral_anomaly",
                            severity="medium" if z_score < 4.0 else "high",
                            description=f"Anomaly in {metric_name}: z-score {z_score:.2f}",
                            affected_neurons=[],
                            timestamp=time.time()
                        ))
        
        return violations
    
    def _check_post_modification_integrity(self) -> List[SafetyViolation]:
        """Check network integrity after modification."""
        violations = []
        
        # Check for disconnected neurons
        disconnected = self._find_disconnected_neurons()
        if disconnected:
            violations.append(SafetyViolation(
                violation_type="disconnected_neurons",
                severity="medium",
                description=f"Found {len(disconnected)} disconnected neurons",
                affected_neurons=disconnected,
                timestamp=time.time()
            ))
        
        # Check for NaN values in weights
        nan_neurons = self._find_nan_weights()
        if nan_neurons:
            violations.append(SafetyViolation(
                violation_type="nan_weights",
                severity="critical",
                description=f"NaN weights detected in {len(nan_neurons)} neurons",
                affected_neurons=nan_neurons,
                timestamp=time.time(),
                auto_rollback=True
            ))
        
        return violations
    
    def _find_disconnected_neurons(self) -> List[int]:
        """Find neurons that are not connected to the network."""
        # Placeholder implementation
        return []
    
    def _find_nan_weights(self) -> List[int]:
        """Find neurons with NaN weights."""
        nan_neurons = []
        
        for neuron_id, neuron in self.network.neurons.items():
            if hasattr(neuron, 'weights') and np.any(np.isnan(neuron.weights)):
                nan_neurons.append(neuron_id)
        
        return nan_neurons
    
    def _log_safety_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log safety-related events."""
        event = {
            'type': event_type,
            'timestamp': time.time(),
            'data': data
        }
        self.safety_events.append(event)
    
    def get_safety_statistics(self) -> Dict[str, Any]:
        """Get comprehensive safety statistics."""
        violation_counts = defaultdict(int)
        for violation in self.violations:
            violation_counts[violation.violation_type] += 1
        
        return {
            'safety_mode': self.safety_mode,
            'emergency_stop': self.emergency_stop,
            'total_violations': len(self.violations),
            'violation_types': dict(violation_counts),
            'snapshots_created': len(self.snapshots),
            'modifications_monitored': self.modification_counter,
            'current_snapshot': self.current_snapshot_id,
            'stability_score': self._assess_performance_stability(),
            'safety_bounds': {
                'max_neurons': self.safety_bounds.max_neurons,
                'min_network_size': self.safety_bounds.min_network_size,
                'max_modification_frequency': self.safety_bounds.max_modification_frequency
            },
            'recent_events': list(self.safety_events)[-10:] if self.safety_events else []
        }