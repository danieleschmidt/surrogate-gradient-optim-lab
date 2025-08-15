"""Pipeline monitoring system for autonomous health detection."""

import time
import threading
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import jax.numpy as jnp
import numpy as np
from loguru import logger


class HealthStatus(Enum):
    """Pipeline health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning" 
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILED = "failed"


@dataclass
class HealthMetric:
    """Individual health metric with thresholds."""
    name: str
    value: float
    threshold_warning: float
    threshold_critical: float
    unit: str = ""
    timestamp: float = field(default_factory=time.time)
    
    @property
    def status(self) -> HealthStatus:
        """Determine status based on thresholds."""
        if self.value >= self.threshold_critical:
            return HealthStatus.CRITICAL
        elif self.value >= self.threshold_warning:
            return HealthStatus.WARNING
        return HealthStatus.HEALTHY


@dataclass
class PipelineHealth:
    """Complete pipeline health snapshot."""
    timestamp: float
    overall_status: HealthStatus
    metrics: Dict[str, HealthMetric]
    active_alerts: List[str] = field(default_factory=list)
    performance_score: float = 1.0


class PipelineMonitor:
    """Autonomous pipeline monitoring with real-time health assessment."""
    
    def __init__(
        self,
        check_interval: float = 10.0,
        alert_callback: Optional[Callable] = None,
        enable_auto_recovery: bool = True
    ):
        self.check_interval = check_interval
        self.alert_callback = alert_callback
        self.enable_auto_recovery = enable_auto_recovery
        
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._health_history: List[PipelineHealth] = []
        self._metric_collectors: Dict[str, Callable] = {}
        
        # Health thresholds
        self.thresholds = {
            "memory_usage": {"warning": 0.8, "critical": 0.95},
            "cpu_usage": {"warning": 0.8, "critical": 0.95},
            "error_rate": {"warning": 0.05, "critical": 0.1},
            "latency_p95": {"warning": 1.0, "critical": 5.0},
            "optimization_convergence": {"warning": 0.01, "critical": 0.001},
            "model_accuracy": {"warning": 0.1, "critical": 0.2}
        }
        
        self._register_default_collectors()
        
    def _register_default_collectors(self) -> None:
        """Register default metric collectors."""
        import psutil
        import gc
        
        def memory_usage() -> float:
            return psutil.virtual_memory().percent / 100.0
            
        def cpu_usage() -> float:
            return psutil.cpu_percent(interval=1) / 100.0
            
        def gc_pressure() -> float:
            gc_stats = gc.get_stats()
            return sum(stat.get('collections', 0) for stat in gc_stats) / 1000.0
            
        self._metric_collectors.update({
            "memory_usage": memory_usage,
            "cpu_usage": cpu_usage,
            "gc_pressure": gc_pressure
        })
        
    def register_metric_collector(self, name: str, collector: Callable[[], float]) -> None:
        """Register custom metric collector."""
        self._metric_collectors[name] = collector
        logger.info(f"Registered metric collector: {name}")
        
    def start_monitoring(self) -> None:
        """Start autonomous monitoring."""
        if self._monitoring:
            logger.warning("Monitoring already active")
            return
            
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info(f"Pipeline monitoring started (interval={self.check_interval}s)")
        
    def stop_monitoring(self) -> None:
        """Stop monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("Pipeline monitoring stopped")
        
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._monitoring:
            try:
                health = self._collect_health_metrics()
                self._health_history.append(health)
                
                # Keep only last 1000 health checks
                if len(self._health_history) > 1000:
                    self._health_history = self._health_history[-1000:]
                
                self._process_health_status(health)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                
            time.sleep(self.check_interval)
            
    def _collect_health_metrics(self) -> PipelineHealth:
        """Collect all health metrics."""
        metrics = {}
        
        for name, collector in self._metric_collectors.items():
            try:
                value = collector()
                thresholds = self.thresholds.get(name, {"warning": 0.8, "critical": 0.95})
                
                metric = HealthMetric(
                    name=name,
                    value=value,
                    threshold_warning=thresholds["warning"],
                    threshold_critical=thresholds["critical"]
                )
                metrics[name] = metric
                
            except Exception as e:
                logger.error(f"Failed to collect metric {name}: {e}")
                
        # Determine overall status
        statuses = [metric.status for metric in metrics.values()]
        if HealthStatus.FAILED in statuses:
            overall_status = HealthStatus.FAILED
        elif HealthStatus.CRITICAL in statuses:
            overall_status = HealthStatus.CRITICAL
        elif HealthStatus.DEGRADED in statuses:
            overall_status = HealthStatus.DEGRADED
        elif HealthStatus.WARNING in statuses:
            overall_status = HealthStatus.WARNING
        else:
            overall_status = HealthStatus.HEALTHY
            
        # Calculate performance score
        performance_score = self._calculate_performance_score(metrics)
        
        return PipelineHealth(
            timestamp=time.time(),
            overall_status=overall_status,
            metrics=metrics,
            performance_score=performance_score
        )
        
    def _calculate_performance_score(self, metrics: Dict[str, HealthMetric]) -> float:
        """Calculate overall performance score (0-1)."""
        if not metrics:
            return 1.0
            
        scores = []
        for metric in metrics.values():
            if metric.threshold_critical > 0:
                score = max(0.0, 1.0 - (metric.value / metric.threshold_critical))
                scores.append(score)
                
        return np.mean(scores) if scores else 1.0
        
    def _process_health_status(self, health: PipelineHealth) -> None:
        """Process health status and trigger alerts/recovery."""
        if health.overall_status in [HealthStatus.WARNING, HealthStatus.CRITICAL, HealthStatus.FAILED]:
            logger.warning(f"Pipeline health: {health.overall_status.value} (score: {health.performance_score:.3f})")
            
            # Trigger alert callback
            if self.alert_callback:
                self.alert_callback(health)
                
            # Trigger auto-recovery if enabled
            if self.enable_auto_recovery and health.overall_status in [HealthStatus.CRITICAL, HealthStatus.FAILED]:
                logger.info("Triggering auto-recovery...")
                self._trigger_auto_recovery(health)
                
    def _trigger_auto_recovery(self, health: PipelineHealth) -> None:
        """Trigger automatic recovery actions."""
        try:
            # Import here to avoid circular imports
            from .recovery_engine import RecoveryEngine
            
            recovery = RecoveryEngine()
            recovery.execute_recovery(health)
            
        except Exception as e:
            logger.error(f"Auto-recovery failed: {e}")
            
    def get_current_health(self) -> Optional[PipelineHealth]:
        """Get current health status."""
        return self._health_history[-1] if self._health_history else None
        
    def get_health_history(self, limit: int = 100) -> List[PipelineHealth]:
        """Get recent health history."""
        return self._health_history[-limit:]
        
    def get_performance_trend(self, window_size: int = 10) -> Dict[str, float]:
        """Calculate performance trends."""
        if len(self._health_history) < window_size:
            return {}
            
        recent_scores = [h.performance_score for h in self._health_history[-window_size:]]
        older_scores = [h.performance_score for h in self._health_history[-2*window_size:-window_size]]
        
        if not older_scores:
            return {"trend": 0.0, "current_avg": np.mean(recent_scores)}
            
        trend = np.mean(recent_scores) - np.mean(older_scores)
        
        return {
            "trend": trend,
            "current_avg": np.mean(recent_scores),
            "previous_avg": np.mean(older_scores)
        }