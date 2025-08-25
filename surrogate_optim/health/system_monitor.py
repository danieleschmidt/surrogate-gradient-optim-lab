"""System health monitoring for robust operation."""

from dataclasses import dataclass, field
from enum import Enum
import threading
import time
from typing import Any, Dict, List, Optional

import jax
import psutil

from ..monitoring.enhanced_logging import enhanced_logger


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILURE = "failure"


@dataclass
class HealthMetrics:
    """System health metrics."""
    timestamp: float = field(default_factory=time.time)
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    memory_available: float = 0.0
    gpu_available: bool = False
    gpu_memory_usage: float = 0.0
    jax_backend: str = "unknown"
    response_time: float = 0.0
    error_rate: float = 0.0

    @property
    def overall_status(self) -> HealthStatus:
        """Determine overall health status."""
        if self.error_rate > 0.5 or self.memory_usage > 0.95:
            return HealthStatus.FAILURE
        if self.error_rate > 0.2 or self.memory_usage > 0.85 or self.cpu_usage > 0.9:
            return HealthStatus.CRITICAL
        if self.error_rate > 0.1 or self.memory_usage > 0.7 or self.cpu_usage > 0.7:
            return HealthStatus.WARNING
        return HealthStatus.HEALTHY


class SystemMonitor:
    """Monitors system health and performance."""

    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self.metrics_history: List[HealthMetrics] = []
        self.is_monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Error tracking
        self._total_requests = 0
        self._failed_requests = 0
        self._lock = threading.Lock()

    def start_monitoring(self) -> None:
        """Start background health monitoring."""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        enhanced_logger.info("System health monitoring started")

    def stop_monitoring(self) -> None:
        """Stop background health monitoring."""
        if not self.is_monitoring:
            return

        self.is_monitoring = False
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        enhanced_logger.info("System health monitoring stopped")

    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while not self._stop_event.wait(self.check_interval):
            try:
                metrics = self.collect_metrics()
                self.metrics_history.append(metrics)

                # Keep only last 100 metrics
                if len(self.metrics_history) > 100:
                    self.metrics_history = self.metrics_history[-100:]

                # Log warnings for unhealthy states
                if metrics.overall_status in [HealthStatus.CRITICAL, HealthStatus.FAILURE]:
                    enhanced_logger.warning(f"System health: {metrics.overall_status.value} - "
                                          f"CPU: {metrics.cpu_usage:.1%}, "
                                          f"Memory: {metrics.memory_usage:.1%}, "
                                          f"Error Rate: {metrics.error_rate:.1%}")

            except Exception as e:
                enhanced_logger.error(f"Health monitoring error: {e}")

    def collect_metrics(self) -> HealthMetrics:
        """Collect current system metrics."""
        metrics = HealthMetrics()

        # CPU and memory
        metrics.cpu_usage = psutil.cpu_percent() / 100.0
        memory = psutil.virtual_memory()
        metrics.memory_usage = memory.percent / 100.0
        metrics.memory_available = memory.available / (1024**3)  # GB

        # GPU info
        try:
            devices = jax.devices()
            metrics.gpu_available = any(d.device_kind == "gpu" for d in devices)
            metrics.jax_backend = str(jax.default_backend())
        except Exception:
            metrics.gpu_available = False
            metrics.jax_backend = "unknown"

        # Error rate
        with self._lock:
            if self._total_requests > 0:
                metrics.error_rate = self._failed_requests / self._total_requests
            else:
                metrics.error_rate = 0.0

        return metrics

    def record_request(self, success: bool = True) -> None:
        """Record a request outcome."""
        with self._lock:
            self._total_requests += 1
            if not success:
                self._failed_requests += 1

    def get_current_status(self) -> HealthStatus:
        """Get current health status."""
        if not self.metrics_history:
            return HealthStatus.HEALTHY
        return self.metrics_history[-1].overall_status

    def get_health_report(self) -> Dict[str, Any]:
        """Generate health report."""
        if not self.metrics_history:
            return {"status": "no_data", "message": "No metrics available"}

        latest = self.metrics_history[-1]

        # Calculate averages over last 10 metrics
        recent_metrics = self.metrics_history[-10:]
        avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
        avg_error_rate = sum(m.error_rate for m in recent_metrics) / len(recent_metrics)

        return {
            "status": latest.overall_status.value,
            "timestamp": latest.timestamp,
            "current": {
                "cpu_usage": latest.cpu_usage,
                "memory_usage": latest.memory_usage,
                "memory_available_gb": latest.memory_available,
                "gpu_available": latest.gpu_available,
                "jax_backend": latest.jax_backend,
                "error_rate": latest.error_rate,
            },
            "averages": {
                "cpu_usage": avg_cpu,
                "memory_usage": avg_memory,
                "error_rate": avg_error_rate,
            },
            "total_requests": self._total_requests,
            "failed_requests": self._failed_requests,
        }


# Global system monitor instance
system_monitor = SystemMonitor()
