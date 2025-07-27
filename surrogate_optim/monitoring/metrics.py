"""Metrics collection and monitoring utilities."""

import time
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class MetricValue:
    """A single metric value with timestamp."""
    value: float
    timestamp: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)


class Counter:
    """Thread-safe counter metric."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._value = 0.0
        self._lock = threading.Lock()
    
    def increment(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Increment the counter."""
        with self._lock:
            self._value += amount
        
        # Record in metrics collector if available
        MetricsCollector.get_instance().record(
            self.name, self._value, labels or {}
        )
    
    def get_value(self) -> float:
        """Get current counter value."""
        with self._lock:
            return self._value
    
    def reset(self):
        """Reset counter to zero."""
        with self._lock:
            self._value = 0.0


class Gauge:
    """Thread-safe gauge metric."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._value = 0.0
        self._lock = threading.Lock()
    
    def set(self, value: float, labels: Optional[Dict[str, str]] = None):
        """Set the gauge value."""
        with self._lock:
            self._value = value
        
        # Record in metrics collector if available
        MetricsCollector.get_instance().record(
            self.name, self._value, labels or {}
        )
    
    def increment(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Increment the gauge."""
        with self._lock:
            self._value += amount
        
        MetricsCollector.get_instance().record(
            self.name, self._value, labels or {}
        )
    
    def decrement(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Decrement the gauge."""
        self.increment(-amount, labels)
    
    def get_value(self) -> float:
        """Get current gauge value."""
        with self._lock:
            return self._value


class Timer:
    """Timer metric for measuring durations."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._durations = deque(maxlen=1000)  # Keep last 1000 measurements
        self._lock = threading.Lock()
        self._start_time = None
    
    def start(self):
        """Start timing."""
        self._start_time = time.time()
        return self
    
    def stop(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Stop timing and record duration."""
        if self._start_time is None:
            logger.warning(f"Timer {self.name} was not started")
            return 0.0
        
        duration = time.time() - self._start_time
        self._start_time = None
        
        with self._lock:
            self._durations.append(duration)
        
        # Record in metrics collector
        MetricsCollector.get_instance().record(
            f"{self.name}_duration_seconds", duration, labels or {}
        )
        
        return duration
    
    def __enter__(self):
        """Context manager entry."""
        return self.start()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
    
    def get_stats(self) -> Dict[str, float]:
        """Get timing statistics."""
        with self._lock:
            if not self._durations:
                return {"count": 0}
            
            durations = list(self._durations)
        
        return {
            "count": len(durations),
            "mean": sum(durations) / len(durations),
            "min": min(durations),
            "max": max(durations),
            "p50": self._percentile(durations, 0.5),
            "p95": self._percentile(durations, 0.95),
            "p99": self._percentile(durations, 0.99),
        }
    
    @staticmethod
    def _percentile(values: List[float], percentile: float) -> float:
        """Calculate percentile of values."""
        sorted_values = sorted(values)
        index = int(percentile * (len(sorted_values) - 1))
        return sorted_values[index]


class MetricsCollector:
    """Central metrics collector using singleton pattern."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self):
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self._counters: Dict[str, Counter] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._timers: Dict[str, Timer] = {}
        self._lock = threading.Lock()
    
    @classmethod
    def get_instance(cls) -> 'MetricsCollector':
        """Get singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def record(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a metric value."""
        metric_value = MetricValue(value=value, labels=labels or {})
        
        with self._lock:
            self._metrics[name].append(metric_value)
    
    def get_counter(self, name: str, description: str = "") -> Counter:
        """Get or create a counter metric."""
        if name not in self._counters:
            self._counters[name] = Counter(name, description)
        return self._counters[name]
    
    def get_gauge(self, name: str, description: str = "") -> Gauge:
        """Get or create a gauge metric."""
        if name not in self._gauges:
            self._gauges[name] = Gauge(name, description)
        return self._gauges[name]
    
    def get_timer(self, name: str, description: str = "") -> Timer:
        """Get or create a timer metric."""
        if name not in self._timers:
            self._timers[name] = Timer(name, description)
        return self._timers[name]
    
    def get_metrics(self, name: Optional[str] = None) -> Dict[str, List[MetricValue]]:
        """Get collected metrics."""
        with self._lock:
            if name is not None:
                return {name: list(self._metrics.get(name, []))}
            else:
                return {k: list(v) for k, v in self._metrics.items()}
    
    def get_latest_values(self) -> Dict[str, float]:
        """Get latest value for each metric."""
        with self._lock:
            latest = {}
            for name, values in self._metrics.items():
                if values:
                    latest[name] = values[-1].value
            return latest
    
    def clear(self):
        """Clear all metrics."""
        with self._lock:
            self._metrics.clear()
    
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        # Export counters
        for counter in self._counters.values():
            lines.append(f"# HELP {counter.name} {counter.description}")
            lines.append(f"# TYPE {counter.name} counter")
            lines.append(f"{counter.name} {counter.get_value()}")
        
        # Export gauges
        for gauge in self._gauges.values():
            lines.append(f"# HELP {gauge.name} {gauge.description}")
            lines.append(f"# TYPE {gauge.name} gauge")
            lines.append(f"{gauge.name} {gauge.get_value()}")
        
        # Export timer statistics
        for timer in self._timers.values():
            stats = timer.get_stats()
            if stats.get("count", 0) > 0:
                base_name = timer.name.replace("_duration_seconds", "")
                lines.append(f"# HELP {base_name}_duration_seconds {timer.description}")
                lines.append(f"# TYPE {base_name}_duration_seconds summary")
                lines.append(f"{base_name}_duration_seconds_count {stats['count']}")
                lines.append(f"{base_name}_duration_seconds_sum {stats['mean'] * stats['count']}")
                
                # Quantiles
                for quantile in [0.5, 0.95, 0.99]:
                    key = f"p{int(quantile * 100)}"
                    if key in stats:
                        lines.append(f"{base_name}_duration_seconds{{quantile=\"{quantile}\"}} {stats[key]}")
        
        return "\n".join(lines) + "\n"
    
    def export_json(self) -> str:
        """Export metrics in JSON format."""
        data = {
            "timestamp": time.time(),
            "counters": {name: counter.get_value() for name, counter in self._counters.items()},
            "gauges": {name: gauge.get_value() for name, gauge in self._gauges.items()},
            "timers": {name: timer.get_stats() for name, timer in self._timers.items()},
            "raw_metrics": {
                name: [{"value": mv.value, "timestamp": mv.timestamp, "labels": mv.labels} 
                       for mv in values]
                for name, values in self._metrics.items()
            }
        }
        return json.dumps(data, indent=2)


# Global metrics instances for common use cases
TRAINING_METRICS = MetricsCollector.get_instance()

# Pre-defined metrics
training_time = TRAINING_METRICS.get_timer("surrogate_training", "Time spent training surrogate models")
prediction_time = TRAINING_METRICS.get_timer("surrogate_prediction", "Time spent making predictions")
optimization_time = TRAINING_METRICS.get_timer("optimization", "Time spent in optimization")

training_iterations = TRAINING_METRICS.get_counter("training_iterations_total", "Total training iterations")
predictions_made = TRAINING_METRICS.get_counter("predictions_total", "Total predictions made")
optimizations_run = TRAINING_METRICS.get_counter("optimizations_total", "Total optimizations run")

current_loss = TRAINING_METRICS.get_gauge("current_training_loss", "Current training loss")
gradient_norm = TRAINING_METRICS.get_gauge("gradient_norm", "Gradient norm during training")
learning_rate = TRAINING_METRICS.get_gauge("learning_rate", "Current learning rate")


# Decorators for easy metrics collection
def time_function(metric_name: Optional[str] = None, labels: Optional[Dict[str, str]] = None):
    """Decorator to time function execution."""
    def decorator(func):
        name = metric_name or f"{func.__module__}.{func.__name__}"
        timer = TRAINING_METRICS.get_timer(name, f"Execution time for {func.__name__}")
        
        def wrapper(*args, **kwargs):
            with timer:
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


def count_calls(metric_name: Optional[str] = None, labels: Optional[Dict[str, str]] = None):
    """Decorator to count function calls."""
    def decorator(func):
        name = metric_name or f"{func.__module__}.{func.__name__}_calls_total"
        counter = TRAINING_METRICS.get_counter(name, f"Total calls to {func.__name__}")
        
        def wrapper(*args, **kwargs):
            counter.increment(labels=labels)
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


# Context managers for inline metrics
class measure_time:
    """Context manager to measure elapsed time."""
    
    def __init__(self, metric_name: str, labels: Optional[Dict[str, str]] = None):
        self.timer = TRAINING_METRICS.get_timer(metric_name)
        self.labels = labels
    
    def __enter__(self):
        self.timer.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.timer.stop(self.labels)


# Performance monitoring utilities
def track_memory_usage(func_name: str = "unknown"):
    """Track memory usage of a function."""
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        def decorator(func):
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                
                memory_after = process.memory_info().rss / 1024 / 1024  # MB
                memory_used = memory_after - memory_before
                
                # Record memory usage
                memory_gauge = TRAINING_METRICS.get_gauge(f"memory_usage_mb")
                memory_gauge.set(memory_after)
                
                if memory_used > 0:
                    TRAINING_METRICS.record(f"memory_delta_mb_{func_name}", memory_used)
                
                return result
            return wrapper
        return decorator
        
    except ImportError:
        # If psutil is not available, return a no-op decorator
        def decorator(func):
            return func
        return decorator