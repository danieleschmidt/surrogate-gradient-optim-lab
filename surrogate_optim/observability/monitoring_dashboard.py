"""Advanced monitoring dashboard for surrogate optimization systems."""

from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .prometheus_metrics import PROMETHEUS_AVAILABLE, get_global_metrics
from .tracing import get_tracer

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class AlertStatus(Enum):
    """Alert status states."""
    ACTIVE = "active"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class Alert:
    """Alert data structure."""
    id: str
    name: str
    description: str
    severity: AlertSeverity
    status: AlertStatus
    timestamp: datetime
    metric_name: str
    threshold: float
    current_value: float
    labels: Dict[str, str]
    resolved_at: Optional[datetime] = None
    suppressed_until: Optional[datetime] = None


@dataclass
class MetricSnapshot:
    """Snapshot of metric values at a point in time."""
    timestamp: datetime
    metrics: Dict[str, float]
    labels: Dict[str, Dict[str, str]]


class MetricAggregator:
    """Aggregates and analyzes metrics over time windows."""

    def __init__(self, window_size: int = 1000):
        """Initialize metric aggregator.
        
        Args:
            window_size: Maximum number of data points to keep in memory
        """
        self.window_size = window_size
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.lock = threading.Lock()

    def add_metric(self, name: str, value: float, timestamp: Optional[datetime] = None):
        """Add a metric value."""
        if timestamp is None:
            timestamp = datetime.now()

        with self.lock:
            self.metrics_history[name].append((timestamp, value))

    def get_recent_values(self, name: str, duration_minutes: int = 10) -> List[Tuple[datetime, float]]:
        """Get recent values for a metric."""
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)

        with self.lock:
            if name not in self.metrics_history:
                return []

            return [(ts, val) for ts, val in self.metrics_history[name] if ts >= cutoff_time]

    def calculate_statistics(self, name: str, duration_minutes: int = 10) -> Dict[str, float]:
        """Calculate statistics for a metric over a time window."""
        recent_values = self.get_recent_values(name, duration_minutes)

        if not recent_values:
            return {"count": 0}

        values = [val for _, val in recent_values]
        values_array = np.array(values)

        return {
            "count": len(values),
            "mean": float(np.mean(values_array)),
            "median": float(np.median(values_array)),
            "std": float(np.std(values_array)),
            "min": float(np.min(values_array)),
            "max": float(np.max(values_array)),
            "p95": float(np.percentile(values_array, 95)),
            "p99": float(np.percentile(values_array, 99)),
        }

    def detect_anomalies(self, name: str, z_threshold: float = 3.0, duration_minutes: int = 60) -> List[Tuple[datetime, float]]:
        """Detect anomalies using z-score analysis."""
        recent_values = self.get_recent_values(name, duration_minutes)

        if len(recent_values) < 10:  # Need minimum data points
            return []

        values = [val for _, val in recent_values]
        values_array = np.array(values)

        mean_val = np.mean(values_array)
        std_val = np.std(values_array)

        if std_val == 0:  # No variation
            return []

        anomalies = []
        for ts, val in recent_values:
            z_score = abs(val - mean_val) / std_val
            if z_score > z_threshold:
                anomalies.append((ts, val))

        return anomalies


class AlertManager:
    """Manages alerts and notifications for monitoring."""

    def __init__(self):
        """Initialize alert manager."""
        self.alerts: Dict[str, Alert] = {}
        self.alert_rules: List[Dict[str, Any]] = []
        self.alert_history: deque = deque(maxlen=10000)
        self.lock = threading.Lock()
        self.notification_callbacks = []

        # Load default alert rules
        self._initialize_default_rules()

    def _initialize_default_rules(self):
        """Initialize default alert rules for surrogate optimization."""
        default_rules = [
            {
                "name": "high_training_time",
                "metric": "surrogate_training_duration_seconds",
                "condition": "mean",
                "threshold": 300.0,  # 5 minutes
                "severity": AlertSeverity.WARNING,
                "description": "Surrogate training time is unusually high",
                "duration_minutes": 5,
            },
            {
                "name": "training_failure_rate",
                "metric": "surrogate_training_iterations_total",
                "condition": "failure_rate",
                "threshold": 0.1,  # 10% failure rate
                "severity": AlertSeverity.CRITICAL,
                "description": "High training failure rate detected",
                "duration_minutes": 10,
            },
            {
                "name": "prediction_latency_high",
                "metric": "surrogate_prediction_duration_seconds",
                "condition": "p95",
                "threshold": 0.1,  # 100ms
                "severity": AlertSeverity.WARNING,
                "description": "High prediction latency detected",
                "duration_minutes": 5,
            },
            {
                "name": "memory_usage_high",
                "metric": "system_memory_usage_bytes",
                "condition": "max",
                "threshold": 8e9,  # 8GB
                "severity": AlertSeverity.CRITICAL,
                "description": "System memory usage is critically high",
                "duration_minutes": 2,
            },
            {
                "name": "model_accuracy_low",
                "metric": "model_accuracy",
                "condition": "mean",
                "threshold": 0.8,
                "severity": AlertSeverity.WARNING,
                "description": "Model accuracy has dropped below acceptable threshold",
                "duration_minutes": 15,
                "invert": True,  # Alert when below threshold
            },
            {
                "name": "optimization_convergence_poor",
                "metric": "optimization_convergence_rate",
                "condition": "mean",
                "threshold": 0.01,
                "severity": AlertSeverity.WARNING,
                "description": "Poor optimization convergence detected",
                "duration_minutes": 10,
                "invert": True,
            },
        ]

        for rule in default_rules:
            self.add_alert_rule(**rule)

    def add_alert_rule(
        self,
        name: str,
        metric: str,
        condition: str,
        threshold: float,
        severity: AlertSeverity,
        description: str,
        duration_minutes: int = 5,
        labels_filter: Optional[Dict[str, str]] = None,
        invert: bool = False,
    ):
        """Add an alert rule.
        
        Args:
            name: Alert rule name
            metric: Metric name to monitor
            condition: Condition to evaluate (mean, max, p95, etc.)
            threshold: Threshold value
            severity: Alert severity
            description: Alert description
            duration_minutes: Time window for evaluation
            labels_filter: Optional label filters
            invert: If True, alert when value is below threshold
        """
        rule = {
            "name": name,
            "metric": metric,
            "condition": condition,
            "threshold": threshold,
            "severity": severity,
            "description": description,
            "duration_minutes": duration_minutes,
            "labels_filter": labels_filter or {},
            "invert": invert,
        }

        with self.lock:
            # Remove existing rule with same name
            self.alert_rules = [r for r in self.alert_rules if r["name"] != name]
            self.alert_rules.append(rule)

    def evaluate_rules(self, aggregator: MetricAggregator):
        """Evaluate all alert rules against current metrics."""
        current_time = datetime.now()

        for rule in self.alert_rules:
            try:
                self._evaluate_single_rule(rule, aggregator, current_time)
            except Exception as e:
                logger.error(f"Error evaluating alert rule {rule['name']}: {e}")

    def _evaluate_single_rule(self, rule: Dict[str, Any], aggregator: MetricAggregator, current_time: datetime):
        """Evaluate a single alert rule."""
        metric_name = rule["metric"]
        condition = rule["condition"]
        threshold = rule["threshold"]
        duration_minutes = rule["duration_minutes"]
        invert = rule.get("invert", False)

        # Get metric statistics
        stats = aggregator.calculate_statistics(metric_name, duration_minutes)

        if stats["count"] == 0:
            return  # No data available

        # Evaluate condition
        if condition in stats:
            current_value = stats[condition]
        else:
            logger.warning(f"Unknown condition '{condition}' for rule '{rule['name']}'")
            return

        # Check threshold
        if invert:
            triggered = current_value < threshold
        else:
            triggered = current_value > threshold

        alert_id = f"{rule['name']}_{metric_name}"

        if triggered:
            self._trigger_alert(rule, alert_id, current_value, current_time)
        else:
            self._resolve_alert(alert_id, current_time)

    def _trigger_alert(self, rule: Dict[str, Any], alert_id: str, current_value: float, current_time: datetime):
        """Trigger an alert."""
        with self.lock:
            if alert_id in self.alerts and self.alerts[alert_id].status == AlertStatus.ACTIVE:
                # Alert already active, just update current value
                self.alerts[alert_id].current_value = current_value
                return

            alert = Alert(
                id=alert_id,
                name=rule["name"],
                description=rule["description"],
                severity=rule["severity"],
                status=AlertStatus.ACTIVE,
                timestamp=current_time,
                metric_name=rule["metric"],
                threshold=rule["threshold"],
                current_value=current_value,
                labels=rule.get("labels_filter", {}),
            )

            self.alerts[alert_id] = alert
            self.alert_history.append(alert)

            # Notify callbacks
            for callback in self.notification_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Error in alert notification callback: {e}")

            logger.warning(f"ALERT TRIGGERED: {alert.name} - {alert.description} (value: {current_value:.3f}, threshold: {rule['threshold']:.3f})")

    def _resolve_alert(self, alert_id: str, current_time: datetime):
        """Resolve an alert."""
        with self.lock:
            if alert_id in self.alerts and self.alerts[alert_id].status == AlertStatus.ACTIVE:
                alert = self.alerts[alert_id]
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = current_time

                logger.info(f"ALERT RESOLVED: {alert.name}")

    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        with self.lock:
            return [alert for alert in self.alerts.values() if alert.status == AlertStatus.ACTIVE]

    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get recent alert history."""
        with self.lock:
            return list(self.alert_history)[-limit:]

    def suppress_alert(self, alert_id: str, duration_minutes: int = 60):
        """Suppress an alert for a specified duration."""
        with self.lock:
            if alert_id in self.alerts:
                alert = self.alerts[alert_id]
                alert.status = AlertStatus.SUPPRESSED
                alert.suppressed_until = datetime.now() + timedelta(minutes=duration_minutes)

    def add_notification_callback(self, callback):
        """Add a callback function for alert notifications."""
        self.notification_callbacks.append(callback)


class MonitoringDashboard:
    """Comprehensive monitoring dashboard for surrogate optimization."""

    def __init__(
        self,
        update_interval_seconds: int = 30,
        enable_web_interface: bool = False,
        web_port: int = 8080,
    ):
        """Initialize monitoring dashboard.
        
        Args:
            update_interval_seconds: How often to update metrics
            enable_web_interface: Whether to enable web interface
            web_port: Port for web interface
        """
        self.update_interval = update_interval_seconds
        self.enable_web_interface = enable_web_interface
        self.web_port = web_port

        # Core components
        self.aggregator = MetricAggregator()
        self.alert_manager = AlertManager()
        self.tracer = get_tracer("monitoring_dashboard")

        # State
        self.is_running = False
        self.monitoring_thread = None
        self.web_server = None

        # Dashboard data
        self.dashboard_data = {
            "system_status": "unknown",
            "active_alerts": [],
            "metric_summaries": {},
            "performance_trends": {},
            "health_checks": {},
        }

        # Setup alert notifications
        self.alert_manager.add_notification_callback(self._handle_alert_notification)

    def start(self):
        """Start the monitoring dashboard."""
        if self.is_running:
            logger.warning("Monitoring dashboard already running")
            return

        self.is_running = True

        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()

        # Start web interface if enabled
        if self.enable_web_interface:
            self._start_web_interface()

        logger.info(f"Monitoring dashboard started (update interval: {self.update_interval}s)")

    def stop(self):
        """Stop the monitoring dashboard."""
        self.is_running = False

        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)

        if self.web_server:
            self._stop_web_interface()

        logger.info("Monitoring dashboard stopped")

    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_running:
            try:
                self._update_metrics()
                self._evaluate_alerts()
                self._update_dashboard_data()
                self._run_health_checks()

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")

            time.sleep(self.update_interval)

    def _update_metrics(self):
        """Update metrics from Prometheus."""
        if not PROMETHEUS_AVAILABLE:
            return

        try:
            # Get current metrics from Prometheus
            metrics = get_global_metrics()

            # Simulate metric collection (in real implementation, would query actual metrics)
            current_time = datetime.now()

            # Add some sample metrics
            self.aggregator.add_metric("system_health", 1.0, current_time)

            # In a real implementation, you would query actual Prometheus metrics here

        except Exception as e:
            logger.error(f"Error updating metrics: {e}")

    def _evaluate_alerts(self):
        """Evaluate alert rules."""
        try:
            self.alert_manager.evaluate_rules(self.aggregator)
        except Exception as e:
            logger.error(f"Error evaluating alerts: {e}")

    def _update_dashboard_data(self):
        """Update dashboard data structure."""
        try:
            # Update active alerts
            active_alerts = self.alert_manager.get_active_alerts()
            self.dashboard_data["active_alerts"] = [asdict(alert) for alert in active_alerts]

            # Update system status
            if any(alert.severity == AlertSeverity.CRITICAL for alert in active_alerts):
                self.dashboard_data["system_status"] = "critical"
            elif any(alert.severity == AlertSeverity.WARNING for alert in active_alerts):
                self.dashboard_data["system_status"] = "warning"
            else:
                self.dashboard_data["system_status"] = "healthy"

            # Update metric summaries
            key_metrics = [
                "surrogate_training_duration_seconds",
                "surrogate_prediction_duration_seconds",
                "optimization_duration_seconds",
                "system_memory_usage_bytes",
                "model_accuracy",
            ]

            metric_summaries = {}
            for metric in key_metrics:
                stats = self.aggregator.calculate_statistics(metric, duration_minutes=60)
                if stats["count"] > 0:
                    metric_summaries[metric] = stats

            self.dashboard_data["metric_summaries"] = metric_summaries

        except Exception as e:
            logger.error(f"Error updating dashboard data: {e}")

    def _run_health_checks(self):
        """Run system health checks."""
        health_checks = {}

        try:
            # Check metric collection
            health_checks["metric_collection"] = {
                "status": "healthy" if len(self.aggregator.metrics_history) > 0 else "unhealthy",
                "details": f"{len(self.aggregator.metrics_history)} metrics tracked"
            }

            # Check alert system
            health_checks["alert_system"] = {
                "status": "healthy",
                "details": f"{len(self.alert_manager.alert_rules)} rules configured"
            }

            # Check memory usage
            try:
                import psutil
                memory = psutil.virtual_memory()
                memory_status = "healthy" if memory.percent < 80 else "warning" if memory.percent < 90 else "critical"
                health_checks["system_memory"] = {
                    "status": memory_status,
                    "details": f"{memory.percent:.1f}% used ({memory.used // (1024**3):.1f}GB / {memory.total // (1024**3):.1f}GB)"
                }
            except ImportError:
                health_checks["system_memory"] = {
                    "status": "unknown",
                    "details": "psutil not available"
                }

            # Check Prometheus metrics
            health_checks["prometheus_metrics"] = {
                "status": "healthy" if PROMETHEUS_AVAILABLE else "unhealthy",
                "details": "Prometheus client available" if PROMETHEUS_AVAILABLE else "Prometheus client not available"
            }

            self.dashboard_data["health_checks"] = health_checks

        except Exception as e:
            logger.error(f"Error running health checks: {e}")

    def _handle_alert_notification(self, alert: Alert):
        """Handle alert notifications."""
        # Log alert
        severity_emoji = {"critical": "🚨", "warning": "⚠️", "info": "ℹ️"}
        emoji = severity_emoji.get(alert.severity.value, "")

        logger.warning(f"{emoji} ALERT: {alert.name} - {alert.description} (value: {alert.current_value:.3f})")

        # Add tracing span for alert
        with self.tracer.trace("alert_triggered") as span:
            span.set_attribute("alert.id", alert.id)
            span.set_attribute("alert.name", alert.name)
            span.set_attribute("alert.severity", alert.severity.value)
            span.set_attribute("alert.metric", alert.metric_name)
            span.set_attribute("alert.threshold", alert.threshold)
            span.set_attribute("alert.current_value", alert.current_value)

    def _start_web_interface(self):
        """Start web interface for dashboard."""
        try:
            # This would start a web server (Flask, FastAPI, etc.)
            # For now, just log that it would start
            logger.info(f"Web interface would start on port {self.web_port}")
            logger.info("Web interface implementation requires additional dependencies")
        except Exception as e:
            logger.error(f"Failed to start web interface: {e}")

    def _stop_web_interface(self):
        """Stop web interface."""
        if self.web_server:
            logger.info("Stopping web interface")

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data."""
        return self.dashboard_data.copy()

    def get_metric_statistics(self, metric_name: str, duration_minutes: int = 60) -> Dict[str, float]:
        """Get statistics for a specific metric."""
        return self.aggregator.calculate_statistics(metric_name, duration_minutes)

    def detect_metric_anomalies(self, metric_name: str, z_threshold: float = 3.0, duration_minutes: int = 60) -> List[Tuple[datetime, float]]:
        """Detect anomalies in a specific metric."""
        return self.aggregator.detect_anomalies(metric_name, z_threshold, duration_minutes)

    def add_custom_alert_rule(self, **kwargs):
        """Add a custom alert rule."""
        self.alert_manager.add_alert_rule(**kwargs)

    def suppress_alert(self, alert_id: str, duration_minutes: int = 60):
        """Suppress a specific alert."""
        self.alert_manager.suppress_alert(alert_id, duration_minutes)

    def export_dashboard_report(self) -> str:
        """Export dashboard data as a formatted report."""
        data = self.get_dashboard_data()

        report = []
        report.append("SURROGATE OPTIMIZATION MONITORING REPORT")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"System Status: {data['system_status'].upper()}")
        report.append("")

        # Active alerts
        active_alerts = data["active_alerts"]
        if active_alerts:
            report.append(f"ACTIVE ALERTS ({len(active_alerts)}):")
            report.append("-" * 20)
            for alert in active_alerts:
                report.append(f"• {alert['name']} ({alert['severity']}) - {alert['description']}")
                report.append(f"  Metric: {alert['metric_name']} = {alert['current_value']:.3f} (threshold: {alert['threshold']:.3f})")
                report.append("")
        else:
            report.append("ACTIVE ALERTS: None")
            report.append("")

        # Health checks
        health_checks = data["health_checks"]
        if health_checks:
            report.append("HEALTH CHECKS:")
            report.append("-" * 15)
            for check_name, check_data in health_checks.items():
                status_emoji = {"healthy": "✅", "warning": "⚠️", "critical": "🚨", "unhealthy": "❌", "unknown": "❓"}
                emoji = status_emoji.get(check_data["status"], "")
                report.append(f"{emoji} {check_name}: {check_data['status']} - {check_data['details']}")
            report.append("")

        # Metric summaries
        metric_summaries = data["metric_summaries"]
        if metric_summaries:
            report.append("KEY METRICS (last 60 minutes):")
            report.append("-" * 30)
            for metric_name, stats in metric_summaries.items():
                report.append(f"{metric_name}:")
                report.append(f"  Mean: {stats['mean']:.3f}, P95: {stats['p95']:.3f}, Max: {stats['max']:.3f}")
                report.append(f"  Count: {stats['count']}, Std: {stats['std']:.3f}")
                report.append("")

        return "\n".join(report)


# Global monitoring dashboard instance
_global_dashboard: Optional[MonitoringDashboard] = None
_dashboard_lock = threading.Lock()


def get_global_dashboard() -> MonitoringDashboard:
    """Get the global monitoring dashboard instance."""
    global _global_dashboard
    if _global_dashboard is None:
        with _dashboard_lock:
            if _global_dashboard is None:
                _global_dashboard = MonitoringDashboard()
    return _global_dashboard


def start_monitoring(
    update_interval_seconds: int = 30,
    enable_web_interface: bool = False,
    web_port: int = 8080,
):
    """Start global monitoring dashboard."""
    dashboard = get_global_dashboard()
    dashboard.update_interval = update_interval_seconds
    dashboard.enable_web_interface = enable_web_interface
    dashboard.web_port = web_port
    dashboard.start()


def stop_monitoring():
    """Stop global monitoring dashboard."""
    dashboard = get_global_dashboard()
    dashboard.stop()


def get_monitoring_report() -> str:
    """Get current monitoring report."""
    dashboard = get_global_dashboard()
    return dashboard.export_dashboard_report()
