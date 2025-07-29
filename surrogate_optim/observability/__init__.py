"""Advanced observability and monitoring module for surrogate optimization systems."""

from .tracing import (
    get_tracer,
    trace_function,
    create_span,
    set_span_attribute,
    set_span_status,
    SpanStatus,
    TracingConfig
)

from .prometheus_metrics import (
    PrometheusMetrics,
    start_metrics_server,
    register_metric,
    MetricType
)

from .health_monitoring import (
    HealthChecker,
    HealthStatus,
    ComponentHealth,
    register_health_check,
    get_system_health
)

from .alerting import (
    AlertManager,
    Alert,
    AlertSeverity,
    NotificationChannel,
    send_alert
)

__all__ = [
    # Tracing
    'get_tracer',
    'trace_function', 
    'create_span',
    'set_span_attribute',
    'set_span_status',
    'SpanStatus',
    'TracingConfig',
    
    # Metrics
    'PrometheusMetrics',
    'start_metrics_server',
    'register_metric', 
    'MetricType',
    
    # Health
    'HealthChecker',
    'HealthStatus',
    'ComponentHealth',
    'register_health_check',
    'get_system_health',
    
    # Alerting
    'AlertManager',
    'Alert',
    'AlertSeverity', 
    'NotificationChannel',
    'send_alert'
]