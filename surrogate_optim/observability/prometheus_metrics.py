"""Prometheus metrics integration for advanced monitoring."""

import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        Summary,
        CollectorRegistry,
        generate_latest,
        start_http_server,
        CONTENT_TYPE_LATEST,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = Gauge = Histogram = Summary = None
    CollectorRegistry = None

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of Prometheus metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricConfig:
    """Configuration for a Prometheus metric."""
    name: str
    description: str
    metric_type: MetricType
    labels: List[str] = None
    buckets: List[float] = None  # For histograms
    quantiles: Dict[float, float] = None  # For summaries


class PrometheusMetrics:
    """Centralized Prometheus metrics management."""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus client not available, metrics disabled")
            return
        
        self.registry = registry or CollectorRegistry()
        self._metrics: Dict[str, Any] = {}
        self._lock = threading.Lock()
        
        # Initialize core surrogate optimization metrics
        self._initialize_core_metrics()
    
    def _initialize_core_metrics(self):
        """Initialize core metrics for surrogate optimization."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        # Training metrics
        self.register_metric(MetricConfig(
            name="surrogate_training_duration_seconds",
            description="Time spent training surrogate models",
            metric_type=MetricType.HISTOGRAM,
            buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0]
        ))
        
        self.register_metric(MetricConfig(
            name="surrogate_training_iterations_total",
            description="Total number of training iterations",
            metric_type=MetricType.COUNTER,
            labels=["model_type", "dataset_size"]
        ))
        
        self.register_metric(MetricConfig(
            name="surrogate_training_loss",
            description="Current training loss",
            metric_type=MetricType.GAUGE,
            labels=["model_type", "loss_type"]
        ))
        
        # Prediction metrics
        self.register_metric(MetricConfig(
            name="surrogate_predictions_total",
            description="Total number of predictions made",
            metric_type=MetricType.COUNTER,
            labels=["model_type"]
        ))
        
        self.register_metric(MetricConfig(
            name="surrogate_prediction_duration_seconds",
            description="Time to make predictions",
            metric_type=MetricType.HISTOGRAM,
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
        ))
        
        self.register_metric(MetricConfig(
            name="surrogate_prediction_error",
            description="Prediction error metrics",
            metric_type=MetricType.GAUGE,
            labels=["model_type", "error_type"]
        ))
        
        # Optimization metrics
        self.register_metric(MetricConfig(
            name="optimization_runs_total",
            description="Total optimization runs",
            metric_type=MetricType.COUNTER,
            labels=["optimizer_type", "status"]
        ))
        
        self.register_metric(MetricConfig(
            name="optimization_duration_seconds",
            description="Time spent in optimization",
            metric_type=MetricType.HISTOGRAM,
            buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0]
        ))
        
        self.register_metric(MetricConfig(
            name="optimization_function_evaluations",
            description="Number of function evaluations",
            metric_type=MetricType.HISTOGRAM,
            buckets=[10, 50, 100, 500, 1000, 5000, 10000]
        ))
        
        self.register_metric(MetricConfig(
            name="optimization_convergence_rate",
            description="Rate of convergence",
            metric_type=MetricType.GAUGE,
            labels=["optimizer_type"]
        ))
        
        # System metrics
        self.register_metric(MetricConfig(
            name="system_memory_usage_bytes",
            description="Memory usage in bytes",
            metric_type=MetricType.GAUGE,
            labels=["component"]
        ))
        
        self.register_metric(MetricConfig(
            name="system_cpu_usage_percent",
            description="CPU usage percentage",
            metric_type=MetricType.GAUGE,
            labels=["component"]
        ))
        
        self.register_metric(MetricConfig(
            name="system_gpu_usage_percent",
            description="GPU usage percentage",
            metric_type=MetricType.GAUGE,
            labels=["gpu_id"]
        ))
        
        # Data quality metrics
        self.register_metric(MetricConfig(
            name="data_quality_score",
            description="Data quality assessment score",
            metric_type=MetricType.GAUGE,
            labels=["dataset", "quality_metric"]
        ))
        
        self.register_metric(MetricConfig(
            name="data_preprocessing_duration_seconds",
            description="Time spent in data preprocessing",
            metric_type=MetricType.HISTOGRAM
        ))
        
        # Model performance metrics
        self.register_metric(MetricConfig(
            name="model_accuracy",
            description="Model accuracy metrics",
            metric_type=MetricType.GAUGE,
            labels=["model_type", "metric_type"]
        ))
        
        self.register_metric(MetricConfig(
            name="model_inference_latency_seconds",
            description="Model inference latency",
            metric_type=MetricType.SUMMARY,
            quantiles={0.5: 0.05, 0.9: 0.01, 0.99: 0.001}
        ))
    
    def register_metric(self, config: MetricConfig):
        """Register a new metric."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        with self._lock:
            if config.name in self._metrics:
                logger.warning(f"Metric {config.name} already registered")
                return
            
            labels = config.labels or []
            
            if config.metric_type == MetricType.COUNTER:
                metric = Counter(
                    config.name,
                    config.description,
                    labelnames=labels,
                    registry=self.registry
                )
            elif config.metric_type == MetricType.GAUGE:
                metric = Gauge(
                    config.name,
                    config.description,
                    labelnames=labels,
                    registry=self.registry
                )
            elif config.metric_type == MetricType.HISTOGRAM:
                buckets = config.buckets or [0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
                metric = Histogram(
                    config.name,
                    config.description,
                    labelnames=labels,
                    buckets=buckets,
                    registry=self.registry
                )
            elif config.metric_type == MetricType.SUMMARY:
                quantiles = config.quantiles or {0.5: 0.05, 0.9: 0.01, 0.99: 0.001}
                metric = Summary(
                    config.name,
                    config.description,
                    labelnames=labels,
                    registry=self.registry
                )
            else:
                raise ValueError(f"Unknown metric type: {config.metric_type}")
            
            self._metrics[config.name] = metric
    
    def get_metric(self, name: str):
        """Get a registered metric by name."""
        with self._lock:
            return self._metrics.get(name)
    
    def increment_counter(self, name: str, labels: Optional[Dict[str, str]] = None, amount: float = 1.0):
        """Increment a counter metric."""
        metric = self.get_metric(name)
        if metric and hasattr(metric, 'inc'):
            if labels:
                metric.labels(**labels).inc(amount)
            else:
                metric.inc(amount)
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set a gauge metric value."""
        metric = self.get_metric(name)
        if metric and hasattr(metric, 'set'):
            if labels:
                metric.labels(**labels).set(value)
            else:
                metric.set(value)
    
    def observe_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Observe a value in a histogram."""
        metric = self.get_metric(name)
        if metric and hasattr(metric, 'observe'):
            if labels:
                metric.labels(**labels).observe(value)
            else:
                metric.observe(value)
    
    def time_histogram(self, name: str, labels: Optional[Dict[str, str]] = None):
        """Context manager to time and observe a histogram."""
        class TimerContext:
            def __init__(self, metric, labels):
                self.metric = metric
                self.labels = labels
                self.start_time = None
            
            def __enter__(self):
                self.start_time = time.time()
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.metric and hasattr(self.metric, 'observe'):
                    duration = time.time() - self.start_time
                    if self.labels:
                        self.metric.labels(**self.labels).observe(duration)
                    else:
                        self.metric.observe(duration)
        
        metric = self.get_metric(name)
        return TimerContext(metric, labels)
    
    def observe_summary(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Observe a value in a summary."""
        metric = self.get_metric(name)
        if metric and hasattr(metric, 'observe'):
            if labels:
                metric.labels(**labels).observe(value)
            else:
                metric.observe(value)
    
    def get_metrics_text(self) -> str:
        """Get metrics in Prometheus text format."""
        if not PROMETHEUS_AVAILABLE:
            return ""
        return generate_latest(self.registry).decode('utf-8')
    
    def clear_metrics(self):
        """Clear all metrics."""
        with self._lock:
            self._metrics.clear()
            # Create new registry
            self.registry = CollectorRegistry()


# Global metrics instance
_global_metrics: Optional[PrometheusMetrics] = None
_metrics_lock = threading.Lock()


def get_global_metrics() -> PrometheusMetrics:
    """Get the global metrics instance."""
    global _global_metrics
    if _global_metrics is None:
        with _metrics_lock:
            if _global_metrics is None:
                _global_metrics = PrometheusMetrics()
    return _global_metrics


def register_metric(config: MetricConfig):
    """Register a metric with the global metrics instance."""
    get_global_metrics().register_metric(config)


def start_metrics_server(port: int = 8000, addr: str = '0.0.0.0'):
    """Start Prometheus metrics HTTP server."""
    if not PROMETHEUS_AVAILABLE:
        logger.warning("Cannot start metrics server: Prometheus client not available")
        return
    
    try:
        start_http_server(port, addr, registry=get_global_metrics().registry)
        logger.info(f"Prometheus metrics server started on {addr}:{port}")
    except Exception as e:
        logger.error(f"Failed to start metrics server: {e}")


# Convenient metric functions
def track_training_time(model_type: str = "unknown"):
    """Context manager to track training time."""
    metrics = get_global_metrics()
    return metrics.time_histogram(
        "surrogate_training_duration_seconds",
        labels={"model_type": model_type}
    )


def track_prediction_time(model_type: str = "unknown"):
    """Context manager to track prediction time."""
    metrics = get_global_metrics()
    return metrics.time_histogram(
        "surrogate_prediction_duration_seconds", 
        labels={"model_type": model_type}
    )


def track_optimization_time(optimizer_type: str = "unknown"):
    """Context manager to track optimization time."""
    metrics = get_global_metrics()
    return metrics.time_histogram(
        "optimization_duration_seconds",
        labels={"optimizer_type": optimizer_type}
    )


def increment_training_iterations(model_type: str = "unknown", dataset_size: str = "unknown"):
    """Increment training iterations counter."""
    metrics = get_global_metrics()
    metrics.increment_counter(
        "surrogate_training_iterations_total",
        labels={"model_type": model_type, "dataset_size": dataset_size}
    )


def set_training_loss(loss: float, model_type: str = "unknown", loss_type: str = "mse"):
    """Set current training loss."""
    metrics = get_global_metrics()
    metrics.set_gauge(
        "surrogate_training_loss",
        loss,
        labels={"model_type": model_type, "loss_type": loss_type}
    )


def increment_predictions(model_type: str = "unknown"):
    """Increment predictions counter."""
    metrics = get_global_metrics()
    metrics.increment_counter(
        "surrogate_predictions_total",
        labels={"model_type": model_type}
    )


def increment_optimizations(optimizer_type: str = "unknown", status: str = "success"):
    """Increment optimization runs counter."""
    metrics = get_global_metrics()
    metrics.increment_counter(
        "optimization_runs_total",
        labels={"optimizer_type": optimizer_type, "status": status}
    )


def set_system_memory_usage(memory_bytes: float, component: str = "total"):
    """Set system memory usage."""
    metrics = get_global_metrics()
    metrics.set_gauge(
        "system_memory_usage_bytes",
        memory_bytes,
        labels={"component": component}
    )


def set_model_accuracy(accuracy: float, model_type: str = "unknown", metric_type: str = "r2"):
    """Set model accuracy metric."""
    metrics = get_global_metrics()
    metrics.set_gauge(
        "model_accuracy",
        accuracy,
        labels={"model_type": model_type, "metric_type": metric_type}
    )


# Decorators for automatic metrics collection
def track_function_calls(metric_name: Optional[str] = None, labels: Optional[Dict[str, str]] = None):
    """Decorator to track function call counts and timing.""" 
    def decorator(func):
        name = metric_name or f"{func.__module__}_{func.__name__}"
        
        # Register metrics if they don't exist
        counter_config = MetricConfig(
            name=f"{name}_calls_total",
            description=f"Total calls to {func.__name__}",
            metric_type=MetricType.COUNTER,
            labels=list(labels.keys()) if labels else []
        )
        
        histogram_config = MetricConfig( 
            name=f"{name}_duration_seconds",
            description=f"Execution time for {func.__name__}",
            metric_type=MetricType.HISTOGRAM,
            labels=list(labels.keys()) if labels else []
        )
        
        metrics = get_global_metrics()
        metrics.register_metric(counter_config)
        metrics.register_metric(histogram_config)
        
        def wrapper(*args, **kwargs):
            # Increment call counter
            metrics.increment_counter(f"{name}_calls_total", labels)
            
            # Time execution
            with metrics.time_histogram(f"{name}_duration_seconds", labels):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator