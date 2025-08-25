"""Self-healing surrogate optimizer with autonomous pipeline management."""

from dataclasses import dataclass
import time
from typing import Any, Callable, Dict, List, Optional

import jax
import jax.numpy as jnp
from loguru import logger

from ..core.enhanced_optimizer import EnhancedSurrogateOptimizer
from .health_diagnostics import DiagnosticReport, HealthDiagnostics
from .pipeline_monitor import HealthStatus, PipelineHealth, PipelineMonitor
from .recovery_engine import RecoveryEngine


@dataclass
class OptimizationHealth:
    """Health metrics specific to optimization process."""
    convergence_rate: float
    solution_quality: float
    function_evaluations: int
    gradient_quality: float
    optimization_time: float
    last_improvement: float


class SelfHealingOptimizer(EnhancedSurrogateOptimizer):
    """Surrogate optimizer with autonomous self-healing capabilities."""

    def __init__(
        self,
        surrogate_type: str = "neural_network",
        monitoring_interval: float = 30.0,
        auto_recovery: bool = True,
        health_diagnostics: bool = True,
        **kwargs
    ):
        super().__init__(surrogate_type=surrogate_type, **kwargs)

        self.monitoring_interval = monitoring_interval
        self.auto_recovery = auto_recovery
        self.health_diagnostics_enabled = health_diagnostics

        # Initialize self-healing components
        self.pipeline_monitor = PipelineMonitor(
            check_interval=monitoring_interval,
            alert_callback=self._handle_health_alert,
            enable_auto_recovery=auto_recovery
        )

        self.recovery_engine = RecoveryEngine()

        if health_diagnostics:
            self.diagnostics = HealthDiagnostics(
                history_window=200,
                anomaly_threshold=0.7,
                enable_ml_detection=True
            )
        else:
            self.diagnostics = None

        # Optimization tracking
        self._optimization_active = False
        self._optimization_metrics: Dict[str, float] = {}
        self._last_optimization_health: Optional[OptimizationHealth] = None

        # Register optimization-specific metric collectors
        self._register_optimization_collectors()

        logger.info("Self-healing optimizer initialized with autonomous monitoring")

    def _register_optimization_collectors(self) -> None:
        """Register optimization-specific metric collectors."""

        def convergence_rate() -> float:
            if not self._optimization_active:
                return 1.0
            return self._optimization_metrics.get("convergence_rate", 1.0)

        def solution_quality() -> float:
            if not self._optimization_active:
                return 1.0
            return self._optimization_metrics.get("solution_quality", 1.0)

        def gradient_quality() -> float:
            if not self._optimization_active:
                return 1.0
            return self._optimization_metrics.get("gradient_quality", 1.0)

        def optimization_efficiency() -> float:
            if not self._optimization_active:
                return 1.0
            return self._optimization_metrics.get("efficiency", 1.0)

        # Register with pipeline monitor
        self.pipeline_monitor.register_metric_collector("optimization_convergence", convergence_rate)
        self.pipeline_monitor.register_metric_collector("solution_quality", solution_quality)
        self.pipeline_monitor.register_metric_collector("gradient_quality", gradient_quality)
        self.pipeline_monitor.register_metric_collector("optimization_efficiency", optimization_efficiency)

    def start_monitoring(self) -> None:
        """Start autonomous health monitoring."""
        self.pipeline_monitor.start_monitoring()
        logger.info("Self-healing monitoring started")

    def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        self.pipeline_monitor.stop_monitoring()
        logger.info("Self-healing monitoring stopped")

    def optimize(
        self,
        objective_function: Callable,
        initial_point: jnp.ndarray,
        bounds: Optional[List[tuple]] = None,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        **kwargs
    ) -> Dict[str, Any]:
        """Enhanced optimization with self-healing capabilities."""
        start_time = time.time()
        self._optimization_active = True

        try:
            logger.info(f"Starting self-healing optimization with {max_iterations} max iterations")

            # Start monitoring if not already active
            if not self.pipeline_monitor._monitoring:
                self.start_monitoring()

            # Initialize optimization tracking
            self._initialize_optimization_tracking(objective_function, initial_point)

            # Run optimization with health monitoring
            result = self._run_monitored_optimization(
                objective_function, initial_point, bounds, max_iterations, tolerance, **kwargs
            )

            # Generate final health report
            final_health = self._generate_optimization_health_report(start_time)
            result["optimization_health"] = final_health

            # Generate diagnostic report if enabled
            if self.diagnostics:
                current_health = self.pipeline_monitor.get_current_health()
                if current_health:
                    diagnostic_report = self.diagnostics.analyze_current_health(current_health)
                    result["diagnostic_report"] = diagnostic_report

            logger.info(f"Self-healing optimization completed in {time.time() - start_time:.2f}s")
            return result

        except Exception as e:
            logger.error(f"Optimization failed: {e}")

            # Attempt recovery before re-raising
            if self.auto_recovery:
                self._attempt_optimization_recovery(e)

            raise

        finally:
            self._optimization_active = False

    def _initialize_optimization_tracking(
        self,
        objective_function: Callable,
        initial_point: jnp.ndarray
    ) -> None:
        """Initialize optimization performance tracking."""
        # Evaluate initial point
        try:
            initial_value = objective_function(initial_point)
            self._optimization_metrics = {
                "initial_value": float(initial_value),
                "best_value": float(initial_value),
                "convergence_rate": 1.0,
                "solution_quality": 1.0,
                "gradient_quality": 1.0,
                "efficiency": 1.0,
                "function_evaluations": 1,
                "iterations": 0,
                "last_improvement_iter": 0
            }
        except Exception as e:
            logger.warning(f"Failed to evaluate initial point: {e}")
            self._optimization_metrics = {
                "convergence_rate": 0.5,
                "solution_quality": 0.5,
                "gradient_quality": 0.5,
                "efficiency": 0.5,
                "function_evaluations": 0,
                "iterations": 0,
                "last_improvement_iter": 0
            }

    def _run_monitored_optimization(
        self,
        objective_function: Callable,
        initial_point: jnp.ndarray,
        bounds: Optional[List[tuple]],
        max_iterations: int,
        tolerance: float,
        **kwargs
    ) -> Dict[str, Any]:
        """Run optimization with continuous health monitoring."""
        # Create monitored objective function
        monitored_objective = self._create_monitored_objective(objective_function)

        # Save checkpoint before optimization
        self.recovery_engine.save_checkpoint({
            "initial_point": initial_point,
            "bounds": bounds,
            "max_iterations": max_iterations,
            "tolerance": tolerance,
            "kwargs": kwargs
        })

        # Run base optimization with enhanced monitoring
        result = super().optimize(
            objective_function=monitored_objective,
            initial_point=initial_point,
            bounds=bounds,
            max_iterations=max_iterations,
            tolerance=tolerance,
            **kwargs
        )

        return result

    def _create_monitored_objective(self, objective_function: Callable) -> Callable:
        """Create a monitored version of the objective function."""

        def monitored_objective(x: jnp.ndarray) -> float:
            try:
                # Evaluate original function
                start_time = time.time()
                value = objective_function(x)
                eval_time = time.time() - start_time

                # Update metrics
                self._update_optimization_metrics(x, value, eval_time)

                return value

            except Exception as e:
                logger.error(f"Objective function evaluation failed: {e}")

                # Update error metrics
                self._optimization_metrics["error_rate"] = self._optimization_metrics.get("error_rate", 0.0) + 0.1

                # Attempt recovery if auto-recovery is enabled
                if self.auto_recovery:
                    self._attempt_evaluation_recovery(x, e)

                raise

        return monitored_objective

    def _update_optimization_metrics(
        self,
        x: jnp.ndarray,
        value: float,
        eval_time: float
    ) -> None:
        """Update optimization performance metrics."""
        metrics = self._optimization_metrics
        metrics["function_evaluations"] += 1

        # Track best value and improvements
        if value < metrics.get("best_value", float("inf")):
            improvement = metrics["best_value"] - value
            metrics["best_value"] = value
            metrics["last_improvement_iter"] = metrics["iterations"]
            metrics["last_improvement"] = improvement

        # Update convergence rate (simplified)
        iterations_since_improvement = metrics["iterations"] - metrics["last_improvement_iter"]
        if iterations_since_improvement > 0:
            metrics["convergence_rate"] = 1.0 / (1.0 + iterations_since_improvement * 0.1)
        else:
            metrics["convergence_rate"] = 1.0

        # Update solution quality (based on improvement from initial)
        if "initial_value" in metrics and metrics["initial_value"] != 0:
            improvement_ratio = (metrics["initial_value"] - value) / abs(metrics["initial_value"])
            metrics["solution_quality"] = max(0.0, min(1.0, improvement_ratio + 0.5))
        else:
            metrics["solution_quality"] = 0.5

        # Update efficiency (evaluations per improvement)
        if metrics.get("last_improvement", 0) > 0:
            evals_per_improvement = metrics["function_evaluations"] / max(1, metrics["iterations"] - metrics["last_improvement_iter"] + 1)
            metrics["efficiency"] = max(0.0, min(1.0, 1.0 / evals_per_improvement))
        else:
            metrics["efficiency"] = 0.5

        # Estimate gradient quality (simplified heuristic)
        metrics["gradient_quality"] = min(1.0, metrics["convergence_rate"] + 0.2)

        metrics["iterations"] = metrics.get("iterations", 0) + 1

    def _handle_health_alert(self, health: PipelineHealth) -> None:
        """Handle health alerts during optimization."""
        logger.warning(f"Health alert: {health.overall_status.value}")

        # Log critical metrics
        for metric_name, metric in health.metrics.items():
            if metric.status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                logger.warning(f"Metric {metric_name}: {metric.value:.3f} (threshold: {metric.threshold_warning:.3f})")

        # Generate diagnostic report if available
        if self.diagnostics:
            try:
                report = self.diagnostics.analyze_current_health(health)

                if report.anomalies:
                    logger.warning(f"Detected {len(report.anomalies)} anomalies")
                    for anomaly in report.anomalies[:3]:  # Log top 3
                        logger.warning(f"  {anomaly.type.value}: {anomaly.description} (severity: {anomaly.severity:.2f})")

                if report.recommendations:
                    logger.info("Health recommendations:")
                    for rec in report.recommendations[:3]:  # Show top 3
                        logger.info(f"  - {rec}")

            except Exception as e:
                logger.error(f"Failed to generate diagnostic report: {e}")

    def _attempt_optimization_recovery(self, error: Exception) -> None:
        """Attempt to recover from optimization failure."""
        logger.info(f"Attempting optimization recovery from: {error}")

        # Create mock health for recovery
        current_health = self.pipeline_monitor.get_current_health()
        if not current_health:
            return

        # Execute recovery actions
        recovery_results = self.recovery_engine.execute_recovery(current_health)

        success_count = sum(1 for r in recovery_results if r.success)
        logger.info(f"Recovery completed: {success_count}/{len(recovery_results)} actions successful")

    def _attempt_evaluation_recovery(self, x: jnp.ndarray, error: Exception) -> None:
        """Attempt to recover from objective function evaluation failure."""
        logger.debug(f"Attempting evaluation recovery at point {x}")

        # Simple recovery strategies
        try:
            # Clear JAX caches
            jax.clear_caches()

            # Force garbage collection
            import gc
            gc.collect()

        except Exception as recovery_error:
            logger.warning(f"Evaluation recovery failed: {recovery_error}")

    def _generate_optimization_health_report(self, start_time: float) -> OptimizationHealth:
        """Generate comprehensive optimization health report."""
        metrics = self._optimization_metrics
        total_time = time.time() - start_time

        health = OptimizationHealth(
            convergence_rate=metrics.get("convergence_rate", 0.0),
            solution_quality=metrics.get("solution_quality", 0.0),
            function_evaluations=metrics.get("function_evaluations", 0),
            gradient_quality=metrics.get("gradient_quality", 0.0),
            optimization_time=total_time,
            last_improvement=metrics.get("last_improvement", 0.0)
        )

        self._last_optimization_health = health
        return health

    def get_optimization_health(self) -> Optional[OptimizationHealth]:
        """Get last optimization health report."""
        return self._last_optimization_health

    def get_pipeline_health(self) -> Optional[PipelineHealth]:
        """Get current pipeline health."""
        return self.pipeline_monitor.get_current_health()

    def get_diagnostic_report(self) -> Optional[DiagnosticReport]:
        """Get latest diagnostic report."""
        if not self.diagnostics:
            return None

        current_health = self.pipeline_monitor.get_current_health()
        if not current_health:
            return None

        return self.diagnostics.analyze_current_health(current_health)

    def is_healthy(self) -> bool:
        """Check if optimizer is currently healthy."""
        health = self.pipeline_monitor.get_current_health()
        if not health:
            return True  # Assume healthy if no data

        return health.overall_status in [HealthStatus.HEALTHY, HealthStatus.WARNING]

    def force_recovery(self) -> List[Any]:
        """Manually trigger recovery actions."""
        current_health = self.pipeline_monitor.get_current_health()
        if not current_health:
            logger.warning("No health data available for recovery")
            return []

        return self.recovery_engine.execute_recovery(current_health)

    def __enter__(self):
        """Context manager entry - start monitoring."""
        self.start_monitoring()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - stop monitoring."""
        self.stop_monitoring()
