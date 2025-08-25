"""Advanced observability and monitoring module for surrogate optimization systems."""

from .monitoring_dashboard import (
    Alert,
    AlertManager,
    AlertSeverity,
    AlertStatus,
    MetricAggregator,
    MetricSnapshot,
    MonitoringDashboard,
    get_global_dashboard,
    get_monitoring_report,
    start_monitoring,
    stop_monitoring,
)
from .prometheus_metrics import (
    MetricType,
    PrometheusMetrics,
    register_metric,
    start_metrics_server,
)
from .tracing import (
    SpanStatus,
    TracingConfig,
    create_span,
    get_tracer,
    set_span_attribute,
    set_span_status,
    trace_function,
)

__all__ = [
    # Tracing
    "get_tracer",
    "trace_function",
    "create_span",
    "set_span_attribute",
    "set_span_status",
    "SpanStatus",
    "TracingConfig",

    # Metrics
    "PrometheusMetrics",
    "start_metrics_server",
    "register_metric",
    "MetricType",

    # Monitoring dashboard
    "AlertSeverity",
    "AlertStatus",
    "Alert",
    "MetricSnapshot",
    "MetricAggregator",
    "AlertManager",
    "MonitoringDashboard",
    "get_global_dashboard",
    "start_monitoring",
    "stop_monitoring",
    "get_monitoring_report",
]
