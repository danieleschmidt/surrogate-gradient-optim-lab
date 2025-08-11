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

from .monitoring_dashboard import (
    AlertSeverity,
    AlertStatus,
    Alert,
    MetricSnapshot,
    MetricAggregator,
    AlertManager,
    MonitoringDashboard,
    get_global_dashboard,
    start_monitoring,
    stop_monitoring,
    get_monitoring_report,
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
    
    # Monitoring dashboard
    'AlertSeverity',
    'AlertStatus',
    'Alert',
    'MetricSnapshot',
    'MetricAggregator',
    'AlertManager',
    'MonitoringDashboard',
    'get_global_dashboard',
    'start_monitoring',
    'stop_monitoring',
    'get_monitoring_report',
]