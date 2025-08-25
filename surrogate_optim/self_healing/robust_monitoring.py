"""Robust monitoring system with advanced error handling and resilience."""

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import os
import signal
import threading
import time
import traceback
from typing import Any, Callable, Dict, List, Optional

import jax
from loguru import logger
import numpy as np
import psutil

from .pipeline_monitor import HealthStatus, PipelineHealth, PipelineMonitor
from .recovery_engine import RecoveryEngine


class MonitoringLevel(Enum):
    """Monitoring intensity levels."""
    MINIMAL = "minimal"
    STANDARD = "standard"
    INTENSIVE = "intensive"
    PARANOID = "paranoid"


class ErrorSeverity(Enum):
    """Error severity classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorEvent:
    """Structured error event with context."""
    timestamp: float
    severity: ErrorSeverity
    error_type: str
    message: str
    context: Dict[str, Any]
    stack_trace: Optional[str] = None
    recovery_attempted: bool = False
    recovery_successful: bool = False


@dataclass
class MonitoringConfig:
    """Configuration for robust monitoring system."""
    level: MonitoringLevel = MonitoringLevel.STANDARD
    check_interval: float = 10.0
    error_window: int = 100
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    enable_predictive_monitoring: bool = True
    enable_resource_protection: bool = True
    max_memory_percent: float = 90.0
    max_cpu_percent: float = 95.0
    enable_graceful_degradation: bool = True


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""

    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time >= self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)

            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
                logger.info("Circuit breaker reset to CLOSED")

            return result

        except Exception:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                logger.error(f"Circuit breaker opened after {self.failure_count} failures")

            raise


class RobustMonitor:
    """Enhanced monitoring system with comprehensive error handling and resilience."""

    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or MonitoringConfig()

        # Core monitoring components
        self.pipeline_monitor = PipelineMonitor(
            check_interval=self.config.check_interval,
            alert_callback=self._handle_alert,
            enable_auto_recovery=True
        )

        self.recovery_engine = RecoveryEngine()

        # Error tracking
        self._error_history: List[ErrorEvent] = []
        self._error_counts: Dict[str, int] = {}

        # Circuit breakers for critical operations
        self._circuit_breakers: Dict[str, CircuitBreaker] = {
            "optimization": CircuitBreaker(self.config.circuit_breaker_threshold, self.config.circuit_breaker_timeout),
            "model_training": CircuitBreaker(3, 120.0),
            "data_collection": CircuitBreaker(5, 30.0)
        }

        # Resource protection
        self._resource_monitor_active = False
        self._resource_monitor_thread: Optional[threading.Thread] = None

        # Graceful shutdown handling
        self._shutdown_requested = False
        self._register_signal_handlers()

        # Performance tracking
        self._performance_metrics: Dict[str, List[float]] = {}

        logger.info(f"Robust monitor initialized with {self.config.level.value} monitoring")

    def _register_signal_handlers(self) -> None:
        """Register signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self._shutdown_requested = True

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    @contextmanager
    def monitored_operation(self, operation_name: str, circuit_breaker: bool = True):
        """Context manager for monitored operations with error handling."""
        start_time = time.time()
        operation_id = f"{operation_name}_{int(start_time)}"

        try:
            logger.debug(f"Starting monitored operation: {operation_name}")

            # Check resource constraints
            if self.config.enable_resource_protection:
                self._check_resource_constraints()

            # Yield control to the operation
            if circuit_breaker and operation_name in self._circuit_breakers:
                breaker = self._circuit_breakers[operation_name]
                yield lambda func, *args, **kwargs: breaker.call(func, *args, **kwargs)
            else:
                yield lambda func, *args, **kwargs: func(*args, **kwargs)

        except Exception as e:
            # Record error event
            error_event = self._create_error_event(operation_name, e)
            self._record_error(error_event)

            # Attempt recovery based on error severity
            if error_event.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
                self._attempt_operation_recovery(operation_name, error_event)

            raise

        finally:
            # Record performance metrics
            duration = time.time() - start_time
            self._record_performance_metric(operation_name, duration)

            logger.debug(f"Completed monitored operation: {operation_name} ({duration:.3f}s)")

    def _check_resource_constraints(self) -> None:
        """Check system resource constraints before operations."""
        memory_usage = psutil.virtual_memory().percent
        cpu_usage = psutil.cpu_percent(interval=1)

        if memory_usage > self.config.max_memory_percent:
            raise ResourceWarning(f"Memory usage too high: {memory_usage:.1f}%")

        if cpu_usage > self.config.max_cpu_percent:
            raise ResourceWarning(f"CPU usage too high: {cpu_usage:.1f}%")

    def _create_error_event(self, operation_name: str, error: Exception) -> ErrorEvent:
        """Create structured error event from exception."""
        # Determine severity based on error type and context
        severity = self._classify_error_severity(error)

        # Gather context
        context = {
            "operation": operation_name,
            "error_class": error.__class__.__name__,
            "pid": os.getpid(),
            "memory_usage": psutil.virtual_memory().percent,
            "cpu_usage": psutil.cpu_percent(),
        }

        # Add JAX-specific context if available
        try:
            context["jax_devices"] = [str(d) for d in jax.devices()]
            context["jax_memory"] = jax.lib.xla_bridge.get_backend().get_memory_info(jax.devices()[0]) if jax.devices() else None
        except Exception:
            pass

        return ErrorEvent(
            timestamp=time.time(),
            severity=severity,
            error_type=error.__class__.__name__,
            message=str(error),
            context=context,
            stack_trace=traceback.format_exc(),
        )

    def _classify_error_severity(self, error: Exception) -> ErrorSeverity:
        """Classify error severity based on type and context."""
        critical_errors = [SystemExit, KeyboardInterrupt, MemoryError, OSError]
        high_errors = [ValueError, RuntimeError, ImportError]
        medium_errors = [TypeError, AttributeError, IndexError]

        error_type = type(error)

        if any(issubclass(error_type, critical) for critical in critical_errors):
            return ErrorSeverity.CRITICAL
        if any(issubclass(error_type, high) for high in high_errors):
            return ErrorSeverity.HIGH
        if any(issubclass(error_type, medium) for medium in medium_errors):
            return ErrorSeverity.MEDIUM
        return ErrorSeverity.LOW

    def _record_error(self, error_event: ErrorEvent) -> None:
        """Record error event in history."""
        self._error_history.append(error_event)

        # Maintain rolling window
        if len(self._error_history) > self.config.error_window:
            self._error_history = self._error_history[-self.config.error_window:]

        # Update error counts
        error_key = f"{error_event.error_type}:{error_event.context.get('operation', 'unknown')}"
        self._error_counts[error_key] = self._error_counts.get(error_key, 0) + 1

        # Log based on severity
        if error_event.severity == ErrorSeverity.CRITICAL:
            logger.critical(f"CRITICAL ERROR: {error_event.message}")
        elif error_event.severity == ErrorSeverity.HIGH:
            logger.error(f"HIGH SEVERITY: {error_event.message}")
        elif error_event.severity == ErrorSeverity.MEDIUM:
            logger.warning(f"MEDIUM SEVERITY: {error_event.message}")
        else:
            logger.debug(f"LOW SEVERITY: {error_event.message}")

    def _attempt_operation_recovery(self, operation_name: str, error_event: ErrorEvent) -> None:
        """Attempt recovery for failed operation."""
        logger.info(f"Attempting recovery for operation: {operation_name}")

        try:
            # Create mock health for recovery
            current_health = self.pipeline_monitor.get_current_health()
            if current_health:
                recovery_results = self.recovery_engine.execute_recovery(current_health)

                # Update error event with recovery info
                error_event.recovery_attempted = True
                error_event.recovery_successful = any(r.success for r in recovery_results)

                if error_event.recovery_successful:
                    logger.info(f"Recovery successful for operation: {operation_name}")
                else:
                    logger.warning(f"Recovery failed for operation: {operation_name}")

        except Exception as recovery_error:
            logger.error(f"Recovery attempt failed: {recovery_error}")
            error_event.recovery_attempted = True
            error_event.recovery_successful = False

    def _record_performance_metric(self, operation_name: str, duration: float) -> None:
        """Record performance metric for operation."""
        if operation_name not in self._performance_metrics:
            self._performance_metrics[operation_name] = []

        self._performance_metrics[operation_name].append(duration)

        # Keep only recent metrics
        if len(self._performance_metrics[operation_name]) > 100:
            self._performance_metrics[operation_name] = self._performance_metrics[operation_name][-100:]

    def start_robust_monitoring(self) -> None:
        """Start comprehensive monitoring with all protection systems."""
        logger.info("Starting robust monitoring system...")

        # Start pipeline monitoring
        self.pipeline_monitor.start_monitoring()

        # Start resource monitoring if enabled
        if self.config.enable_resource_protection:
            self._start_resource_monitoring()

        # Start predictive monitoring if enabled
        if self.config.enable_predictive_monitoring:
            self._start_predictive_monitoring()

        logger.info("Robust monitoring system active")

    def stop_robust_monitoring(self) -> None:
        """Stop all monitoring systems gracefully."""
        logger.info("Stopping robust monitoring system...")

        self._shutdown_requested = True

        # Stop pipeline monitoring
        self.pipeline_monitor.stop_monitoring()

        # Stop resource monitoring
        if self._resource_monitor_active:
            self._resource_monitor_active = False
            if self._resource_monitor_thread:
                self._resource_monitor_thread.join(timeout=5.0)

        logger.info("Robust monitoring system stopped")

    def _start_resource_monitoring(self) -> None:
        """Start dedicated resource monitoring thread."""
        if self._resource_monitor_active:
            return

        self._resource_monitor_active = True
        self._resource_monitor_thread = threading.Thread(target=self._resource_monitor_loop, daemon=True)
        self._resource_monitor_thread.start()

        logger.info("Resource monitoring started")

    def _resource_monitor_loop(self) -> None:
        """Resource monitoring loop with protection mechanisms."""
        while self._resource_monitor_active and not self._shutdown_requested:
            try:
                # Check memory pressure
                memory_usage = psutil.virtual_memory().percent
                if memory_usage > self.config.max_memory_percent * 0.8:  # 80% of max
                    logger.warning(f"High memory usage: {memory_usage:.1f}%")

                    # Trigger preemptive cleanup
                    self._trigger_memory_cleanup()

                # Check CPU usage
                cpu_usage = psutil.cpu_percent(interval=1)
                if cpu_usage > self.config.max_cpu_percent * 0.8:  # 80% of max
                    logger.warning(f"High CPU usage: {cpu_usage:.1f}%")

                # Check disk space
                disk_usage = psutil.disk_usage("/").percent
                if disk_usage > 85.0:
                    logger.warning(f"High disk usage: {disk_usage:.1f}%")

                # Check system load
                load_avg = os.getloadavg()[0] if hasattr(os, "getloadavg") else 0
                cpu_count = psutil.cpu_count()
                if load_avg > cpu_count * 2:
                    logger.warning(f"High system load: {load_avg:.2f} (CPUs: {cpu_count})")

            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")

            time.sleep(self.config.check_interval)

    def _trigger_memory_cleanup(self) -> None:
        """Trigger proactive memory cleanup."""
        try:
            import gc

            # Force garbage collection
            collected = gc.collect()
            logger.debug(f"Garbage collection freed {collected} objects")

            # Clear JAX caches
            jax.clear_caches()

            # Clear torch cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

        except Exception as e:
            logger.error(f"Memory cleanup failed: {e}")

    def _start_predictive_monitoring(self) -> None:
        """Start predictive monitoring for anomaly detection."""
        # Register predictive metric collectors
        def memory_trend() -> float:
            history = self._performance_metrics.get("memory_usage", [])
            if len(history) < 5:
                return 0.0
            # Simple trend analysis
            recent = np.mean(history[-5:])
            older = np.mean(history[-10:-5]) if len(history) >= 10 else recent
            return (recent - older) / older if older > 0 else 0.0

        def error_rate() -> float:
            recent_errors = [e for e in self._error_history if time.time() - e.timestamp < 300]  # Last 5 minutes
            return len(recent_errors) / 300.0  # Errors per second

        self.pipeline_monitor.register_metric_collector("memory_trend", memory_trend)
        self.pipeline_monitor.register_metric_collector("error_rate", error_rate)

        logger.info("Predictive monitoring started")

    def _handle_alert(self, health: PipelineHealth) -> None:
        """Enhanced alert handling with robust error management."""
        try:
            alert_severity = self._classify_health_severity(health)

            logger.warning(f"Health alert: {health.overall_status.value} (severity: {alert_severity.value})")

            # Escalate based on severity
            if alert_severity == ErrorSeverity.CRITICAL:
                self._handle_critical_alert(health)
            elif alert_severity == ErrorSeverity.HIGH:
                self._handle_high_alert(health)

        except Exception as e:
            logger.error(f"Alert handling failed: {e}")
            # Create error event for the alert handling failure
            error_event = self._create_error_event("alert_handling", e)
            self._record_error(error_event)

    def _classify_health_severity(self, health: PipelineHealth) -> ErrorSeverity:
        """Classify health status severity."""
        if health.overall_status == HealthStatus.FAILED:
            return ErrorSeverity.CRITICAL
        if health.overall_status == HealthStatus.CRITICAL:
            return ErrorSeverity.HIGH
        if health.overall_status == HealthStatus.DEGRADED:
            return ErrorSeverity.MEDIUM
        return ErrorSeverity.LOW

    def _handle_critical_alert(self, health: PipelineHealth) -> None:
        """Handle critical health alerts."""
        logger.critical("CRITICAL ALERT: Initiating emergency procedures")

        # Immediate actions for critical state
        try:
            # Save system state
            self._save_emergency_state(health)

            # Trigger aggressive recovery
            self.recovery_engine.execute_recovery(health)

            # Activate graceful degradation if enabled
            if self.config.enable_graceful_degradation:
                self._activate_graceful_degradation()

        except Exception as e:
            logger.critical(f"Critical alert handling failed: {e}")

    def _handle_high_alert(self, health: PipelineHealth) -> None:
        """Handle high severity health alerts."""
        logger.error("HIGH ALERT: Initiating recovery procedures")

        try:
            # Standard recovery procedures
            self.recovery_engine.execute_recovery(health)

        except Exception as e:
            logger.error(f"High alert handling failed: {e}")

    def _save_emergency_state(self, health: PipelineHealth) -> None:
        """Save system state for emergency analysis."""
        try:
            emergency_data = {
                "timestamp": time.time(),
                "health_snapshot": health,
                "error_history": self._error_history[-20:],  # Last 20 errors
                "performance_metrics": self._performance_metrics,
                "system_info": {
                    "memory": psutil.virtual_memory()._asdict(),
                    "cpu": psutil.cpu_percent(percpu=True),
                    "disk": psutil.disk_usage("/")._asdict(),
                    "processes": len(psutil.pids())
                }
            }

            # Save to emergency file
            emergency_file = f"/tmp/emergency_state_{int(time.time())}.json"
            import json
            with open(emergency_file, "w") as f:
                json.dump(emergency_data, f, default=str, indent=2)

            logger.info(f"Emergency state saved to: {emergency_file}")

        except Exception as e:
            logger.error(f"Failed to save emergency state: {e}")

    def _activate_graceful_degradation(self) -> None:
        """Activate graceful degradation mode."""
        logger.warning("Activating graceful degradation mode")

        # Reduce monitoring frequency
        self.config.check_interval *= 2

        # Tighten circuit breaker thresholds
        for breaker in self._circuit_breakers.values():
            breaker.failure_threshold = max(1, breaker.failure_threshold // 2)

        # Reduce resource limits
        self.config.max_memory_percent *= 0.8
        self.config.max_cpu_percent *= 0.8

    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive system health summary."""
        return {
            "monitoring_active": self.pipeline_monitor._monitoring,
            "resource_monitoring_active": self._resource_monitor_active,
            "recent_errors": len([e for e in self._error_history if time.time() - e.timestamp < 3600]),
            "error_distribution": self._get_error_distribution(),
            "circuit_breaker_states": {name: breaker.state for name, breaker in self._circuit_breakers.items()},
            "performance_summary": self._get_performance_summary(),
            "current_health": self.pipeline_monitor.get_current_health(),
            "shutdown_requested": self._shutdown_requested
        }

    def _get_error_distribution(self) -> Dict[str, int]:
        """Get distribution of error types."""
        distribution = {}
        for error in self._error_history:
            error_type = error.error_type
            distribution[error_type] = distribution.get(error_type, 0) + 1
        return distribution

    def _get_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics summary."""
        summary = {}
        for operation, metrics in self._performance_metrics.items():
            if metrics:
                summary[operation] = {
                    "mean": np.mean(metrics),
                    "median": np.median(metrics),
                    "p95": np.percentile(metrics, 95),
                    "min": np.min(metrics),
                    "max": np.max(metrics),
                    "count": len(metrics)
                }
        return summary
