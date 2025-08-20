"""
Production-Grade Error Handling and Fault Tolerance System

Comprehensive error management with:
- Structured exception hierarchy
- Circuit breaker patterns
- Retry mechanisms with exponential backoff
- Graceful degradation strategies
- Error correlation and tracking
"""

import time
import logging
import traceback
import threading
from typing import Dict, List, Any, Optional, Callable, Type, Union
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from collections import deque, defaultdict
import uuid
import asyncio


class ErrorSeverity(Enum):
    """Error severity levels"""
    CRITICAL = "critical"
    HIGH = "high" 
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ErrorCategory(Enum):
    """Error categories for classification"""
    SYSTEM = "system"
    RESOURCE = "resource"
    SECURITY = "security"
    VALIDATION = "validation"
    BUSINESS_LOGIC = "business_logic"
    EXTERNAL_SERVICE = "external_service"
    PERFORMANCE = "performance"


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit broken, failing fast
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class ErrorContext:
    """Context information for errors"""
    error_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    correlation_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    component: Optional[str] = None
    operation: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetryConfig:
    """Configuration for retry policies"""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retry_exceptions: tuple = (Exception,)
    stop_exceptions: tuple = ()


# Custom Exception Hierarchy

class AGIError(Exception):
    """Base exception for all AGI-Formula errors"""
    
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 category: ErrorCategory = ErrorCategory.SYSTEM,
                 context: Optional[ErrorContext] = None,
                 cause: Optional[Exception] = None,
                 recoverable: bool = True):
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.category = category
        self.context = context or ErrorContext()
        self.cause = cause
        self.recoverable = recoverable
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/serialization"""
        return {
            'error_id': self.context.error_id,
            'message': self.message,
            'severity': self.severity.value,
            'category': self.category.value,
            'recoverable': self.recoverable,
            'timestamp': self.timestamp,
            'correlation_id': self.context.correlation_id,
            'component': self.context.component,
            'operation': self.context.operation,
            'cause': str(self.cause) if self.cause else None,
            'traceback': traceback.format_exc(),
            'additional_data': self.context.additional_data
        }


class SystemError(AGIError):
    """System-level errors (infrastructure, runtime, etc.)"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.SYSTEM, **kwargs)


class ResourceError(AGIError):
    """Resource-related errors (memory, GPU, storage, etc.)"""
    
    def __init__(self, message: str, resource_type: str = None, **kwargs):
        super().__init__(message, category=ErrorCategory.RESOURCE, **kwargs)
        if resource_type:
            self.context.additional_data['resource_type'] = resource_type


class SecurityError(AGIError):
    """Security-related errors"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message, 
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.SECURITY,
            recoverable=False,
            **kwargs
        )


class ValidationError(AGIError):
    """Input validation and data integrity errors"""
    
    def __init__(self, message: str, field: str = None, value: Any = None, **kwargs):
        super().__init__(message, category=ErrorCategory.VALIDATION, **kwargs)
        if field:
            self.context.additional_data['field'] = field
        if value is not None:
            self.context.additional_data['value'] = str(value)


class PerformanceError(AGIError):
    """Performance-related errors (timeouts, slow responses, etc.)"""
    
    def __init__(self, message: str, operation_time: float = None, threshold: float = None, **kwargs):
        super().__init__(message, category=ErrorCategory.PERFORMANCE, **kwargs)
        if operation_time:
            self.context.additional_data['operation_time'] = operation_time
        if threshold:
            self.context.additional_data['threshold'] = threshold


class ExternalServiceError(AGIError):
    """External service integration errors"""
    
    def __init__(self, message: str, service_name: str = None, status_code: int = None, **kwargs):
        super().__init__(message, category=ErrorCategory.EXTERNAL_SERVICE, **kwargs)
        if service_name:
            self.context.additional_data['service_name'] = service_name
        if status_code:
            self.context.additional_data['status_code'] = status_code


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance"""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: float = 60.0,
                 expected_exception: Type[Exception] = Exception,
                 name: str = None):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.name = name or "unknown"
        
        self._failure_count = 0
        self._last_failure_time = None
        self._state = CircuitState.CLOSED
        self._lock = threading.Lock()
        
        # Statistics
        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap function with circuit breaker"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self._call_with_circuit_breaker(func, *args, **kwargs)
        return wrapper
    
    def _call_with_circuit_breaker(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        with self._lock:
            self._total_requests += 1
            
            if self._state == CircuitState.OPEN:
                if time.time() - self._last_failure_time >= self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    logging.info(f"Circuit breaker {self.name}: transitioning to HALF_OPEN")
                else:
                    raise SystemError(
                        f"Circuit breaker {self.name} is OPEN",
                        severity=ErrorSeverity.HIGH,
                        context=ErrorContext(
                            component="circuit_breaker",
                            operation="call_blocked",
                            additional_data={
                                'circuit_name': self.name,
                                'state': self._state.value,
                                'failure_count': self._failure_count
                            }
                        )
                    )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        """Handle successful execution"""
        with self._lock:
            self._failure_count = 0
            self._successful_requests += 1
            
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
                logging.info(f"Circuit breaker {self.name}: recovered, transitioning to CLOSED")
    
    def _on_failure(self):
        """Handle failed execution"""
        with self._lock:
            self._failure_count += 1
            self._failed_requests += 1
            self._last_failure_time = time.time()
            
            if self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                logging.warning(f"Circuit breaker {self.name}: threshold reached, transitioning to OPEN")
    
    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state information"""
        with self._lock:
            return {
                'name': self.name,
                'state': self._state.value,
                'failure_count': self._failure_count,
                'failure_threshold': self.failure_threshold,
                'total_requests': self._total_requests,
                'successful_requests': self._successful_requests,
                'failed_requests': self._failed_requests,
                'success_rate': self._successful_requests / max(1, self._total_requests),
                'last_failure_time': self._last_failure_time
            }


class RetryPolicy:
    """Intelligent retry mechanism with exponential backoff"""
    
    def __init__(self, config: RetryConfig = None):
        self.config = config or RetryConfig()
        self._retry_stats = defaultdict(list)
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to add retry logic to function"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self._execute_with_retry(func, *args, **kwargs)
        return wrapper
    
    def _execute_with_retry(self, func: Callable, *args, **kwargs):
        """Execute function with retry logic"""
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                result = func(*args, **kwargs)
                if attempt > 0:
                    logging.info(f"Function {func.__name__} succeeded on attempt {attempt + 1}")
                return result
                
            except self.config.stop_exceptions as e:
                # Don't retry for stop exceptions
                raise
                
            except self.config.retry_exceptions as e:
                last_exception = e
                
                if attempt < self.config.max_attempts - 1:  # Not the last attempt
                    delay = self._calculate_delay(attempt)
                    logging.warning(
                        f"Function {func.__name__} failed on attempt {attempt + 1}, "
                        f"retrying in {delay:.2f}s: {str(e)}"
                    )
                    time.sleep(delay)
                    self._retry_stats[func.__name__].append({
                        'attempt': attempt + 1,
                        'delay': delay,
                        'exception': str(e),
                        'timestamp': time.time()
                    })
        
        # All attempts failed
        logging.error(f"Function {func.__name__} failed after {self.config.max_attempts} attempts")
        raise SystemError(
            f"Operation failed after {self.config.max_attempts} retry attempts",
            severity=ErrorSeverity.HIGH,
            context=ErrorContext(
                component="retry_policy",
                operation=func.__name__,
                additional_data={
                    'attempts': self.config.max_attempts,
                    'last_exception': str(last_exception),
                    'retry_history': self._retry_stats[func.__name__][-10:]  # Last 10 retries
                }
            ),
            cause=last_exception
        )
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for exponential backoff"""
        delay = self.config.base_delay * (self.config.exponential_base ** attempt)
        delay = min(delay, self.config.max_delay)
        
        if self.config.jitter:
            import random
            delay *= (0.5 + random.random())  # Add 50-100% jitter
        
        return delay
    
    def get_retry_stats(self, func_name: str = None) -> Dict[str, Any]:
        """Get retry statistics"""
        if func_name:
            return {
                'function': func_name,
                'retry_history': self._retry_stats.get(func_name, []),
                'total_retries': len(self._retry_stats.get(func_name, []))
            }
        else:
            return {
                'total_functions': len(self._retry_stats),
                'functions': {
                    func: {
                        'total_retries': len(retries),
                        'recent_retries': retries[-5:]  # Last 5 retries
                    }
                    for func, retries in self._retry_stats.items()
                }
            }


class ErrorHandler:
    """Central error handling and reporting system"""
    
    def __init__(self):
        self.error_history = deque(maxlen=10000)
        self.error_counts = defaultdict(int)
        self.circuit_breakers = {}
        self.retry_policies = {}
        self._lock = threading.Lock()
        
        # Setup logging
        self.logger = logging.getLogger('agi_error_handler')
        self.logger.setLevel(logging.INFO)
        
        # Error analysis
        self.error_patterns = defaultdict(list)
        self.alert_thresholds = {
            ErrorSeverity.CRITICAL: 1,
            ErrorSeverity.HIGH: 5,
            ErrorSeverity.MEDIUM: 10,
            ErrorSeverity.LOW: 50
        }
    
    def handle_error(self, error: Exception, context: Optional[ErrorContext] = None) -> str:
        """Handle and log error with context"""
        # Convert to AGI error if needed
        if isinstance(error, AGIError):
            agi_error = error
        else:
            agi_error = SystemError(
                f"Unhandled exception: {str(error)}",
                context=context,
                cause=error,
                severity=ErrorSeverity.MEDIUM
            )
        
        # Record error
        with self._lock:
            self.error_history.append(agi_error)
            self.error_counts[agi_error.category] += 1
        
        # Log error
        self._log_error(agi_error)
        
        # Analyze error patterns
        self._analyze_error_pattern(agi_error)
        
        # Check for alert conditions
        self._check_alert_conditions(agi_error)
        
        return agi_error.context.error_id
    
    def _log_error(self, error: AGIError):
        """Log error with appropriate level"""
        error_dict = error.to_dict()
        
        if error.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(f"CRITICAL ERROR: {error.message}", extra=error_dict)
        elif error.severity == ErrorSeverity.HIGH:
            self.logger.error(f"HIGH SEVERITY: {error.message}", extra=error_dict)
        elif error.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(f"MEDIUM SEVERITY: {error.message}", extra=error_dict)
        else:
            self.logger.info(f"LOW SEVERITY: {error.message}", extra=error_dict)
    
    def _analyze_error_pattern(self, error: AGIError):
        """Analyze error patterns for insights"""
        pattern_key = f"{error.category.value}:{error.context.component}"
        self.error_patterns[pattern_key].append({
            'timestamp': error.timestamp,
            'message': error.message,
            'severity': error.severity.value
        })
        
        # Keep only recent patterns (last 100 per pattern)
        if len(self.error_patterns[pattern_key]) > 100:
            self.error_patterns[pattern_key] = self.error_patterns[pattern_key][-100:]
    
    def _check_alert_conditions(self, error: AGIError):
        """Check if error should trigger alerts"""
        # Count recent errors of this severity
        recent_window = time.time() - 300  # Last 5 minutes
        recent_errors = [
            e for e in self.error_history
            if e.severity == error.severity and e.timestamp >= recent_window
        ]
        
        threshold = self.alert_thresholds.get(error.severity, 100)
        
        if len(recent_errors) >= threshold:
            self._trigger_alert(error, len(recent_errors))
    
    def _trigger_alert(self, error: AGIError, count: int):
        """Trigger alert for error condition"""
        alert_message = (
            f"ERROR THRESHOLD EXCEEDED: {count} {error.severity.value} errors in 5 minutes. "
            f"Latest: {error.message}"
        )
        
        self.logger.critical(alert_message, extra={
            'alert_type': 'error_threshold',
            'severity': error.severity.value,
            'count': count,
            'component': error.context.component
        })
        
        # Here you would integrate with alerting systems (PagerDuty, Slack, etc.)
    
    def get_error_summary(self, window_minutes: int = 60) -> Dict[str, Any]:
        """Get error summary for specified time window"""
        cutoff_time = time.time() - (window_minutes * 60)
        recent_errors = [e for e in self.error_history if e.timestamp >= cutoff_time]
        
        summary = {
            'total_errors': len(recent_errors),
            'window_minutes': window_minutes,
            'errors_by_severity': defaultdict(int),
            'errors_by_category': defaultdict(int),
            'errors_by_component': defaultdict(int),
            'most_common_errors': [],
            'error_rate': len(recent_errors) / window_minutes  # errors per minute
        }
        
        # Aggregate statistics
        for error in recent_errors:
            summary['errors_by_severity'][error.severity.value] += 1
            summary['errors_by_category'][error.category.value] += 1
            if error.context.component:
                summary['errors_by_component'][error.context.component] += 1
        
        # Find most common error messages
        error_messages = defaultdict(int)
        for error in recent_errors:
            error_messages[error.message] += 1
        
        summary['most_common_errors'] = sorted(
            error_messages.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        return dict(summary)
    
    def create_circuit_breaker(self, name: str, **kwargs) -> CircuitBreaker:
        """Create and register circuit breaker"""
        circuit_breaker = CircuitBreaker(name=name, **kwargs)
        self.circuit_breakers[name] = circuit_breaker
        return circuit_breaker
    
    def create_retry_policy(self, name: str, config: RetryConfig = None) -> RetryPolicy:
        """Create and register retry policy"""
        retry_policy = RetryPolicy(config)
        self.retry_policies[name] = retry_policy
        return retry_policy
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health based on error patterns"""
        recent_errors = [e for e in self.error_history if e.timestamp >= time.time() - 300]
        
        health_score = 100.0
        
        # Reduce health score based on recent errors
        for error in recent_errors:
            if error.severity == ErrorSeverity.CRITICAL:
                health_score -= 10
            elif error.severity == ErrorSeverity.HIGH:
                health_score -= 5
            elif error.severity == ErrorSeverity.MEDIUM:
                health_score -= 1
        
        health_score = max(0.0, health_score)
        
        return {
            'health_score': health_score,
            'status': 'healthy' if health_score >= 90 else 'degraded' if health_score >= 70 else 'unhealthy',
            'recent_error_count': len(recent_errors),
            'circuit_breakers': {
                name: cb.get_state() 
                for name, cb in self.circuit_breakers.items()
            },
            'error_summary': self.get_error_summary(5)  # Last 5 minutes
        }


# Global error handler instance
global_error_handler = ErrorHandler()


# Convenience decorators
def with_circuit_breaker(name: str, **kwargs):
    """Decorator to add circuit breaker to function"""
    circuit_breaker = global_error_handler.create_circuit_breaker(name, **kwargs)
    return circuit_breaker


def with_retry(config: RetryConfig = None):
    """Decorator to add retry logic to function"""
    retry_policy = RetryPolicy(config)
    return retry_policy


def handle_errors(func: Callable) -> Callable:
    """Decorator to automatically handle and log errors"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            context = ErrorContext(
                component=func.__module__,
                operation=func.__name__
            )
            global_error_handler.handle_error(e, context)
            raise
    return wrapper