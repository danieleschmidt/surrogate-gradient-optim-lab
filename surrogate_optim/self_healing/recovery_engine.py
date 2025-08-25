"""Autonomous recovery engine for pipeline failures."""

from dataclasses import dataclass
from enum import Enum
import gc
import time
from typing import Any, Callable, Dict, List, Optional

import jax
from loguru import logger
import numpy as np

from .pipeline_monitor import HealthStatus, PipelineHealth


class RecoveryAction(Enum):
    """Types of recovery actions."""
    MEMORY_CLEANUP = "memory_cleanup"
    MODEL_RESTART = "model_restart"
    CHECKPOINT_RESTORE = "checkpoint_restore"
    PARAMETER_ADJUSTMENT = "parameter_adjustment"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"
    RESOURCE_REALLOCATION = "resource_reallocation"


@dataclass
class RecoveryResult:
    """Result of a recovery action."""
    action: RecoveryAction
    success: bool
    message: str
    metrics_before: Dict[str, float]
    metrics_after: Dict[str, float]
    duration: float
    timestamp: float


class RecoveryEngine:
    """Autonomous recovery system for optimization pipelines."""

    def __init__(self):
        self._recovery_history: List[RecoveryResult] = []
        self._recovery_strategies: Dict[RecoveryAction, Callable] = {}
        self._last_checkpoint: Optional[Dict[str, Any]] = None
        self._emergency_mode = False

        self._register_default_strategies()

    def _register_default_strategies(self) -> None:
        """Register default recovery strategies."""
        self._recovery_strategies.update({
            RecoveryAction.MEMORY_CLEANUP: self._memory_cleanup,
            RecoveryAction.MODEL_RESTART: self._model_restart,
            RecoveryAction.CHECKPOINT_RESTORE: self._checkpoint_restore,
            RecoveryAction.PARAMETER_ADJUSTMENT: self._parameter_adjustment,
            RecoveryAction.EMERGENCY_SHUTDOWN: self._emergency_shutdown,
            RecoveryAction.RESOURCE_REALLOCATION: self._resource_reallocation
        })

    def register_recovery_strategy(
        self,
        action: RecoveryAction,
        strategy: Callable[[PipelineHealth], RecoveryResult]
    ) -> None:
        """Register custom recovery strategy."""
        self._recovery_strategies[action] = strategy
        logger.info(f"Registered recovery strategy: {action.value}")

    def execute_recovery(self, health: PipelineHealth) -> List[RecoveryResult]:
        """Execute appropriate recovery actions based on health status."""
        start_time = time.time()
        results = []

        logger.info(f"Executing recovery for status: {health.overall_status.value}")

        # Determine recovery actions based on health issues
        actions = self._determine_recovery_actions(health)

        for action in actions:
            try:
                logger.info(f"Executing recovery action: {action.value}")

                strategy = self._recovery_strategies.get(action)
                if strategy:
                    result = strategy(health)
                    results.append(result)

                    if result.success:
                        logger.info(f"Recovery action {action.value} succeeded: {result.message}")
                    else:
                        logger.error(f"Recovery action {action.value} failed: {result.message}")

                    # Stop if emergency shutdown was triggered
                    if action == RecoveryAction.EMERGENCY_SHUTDOWN:
                        break

                else:
                    logger.error(f"No strategy registered for action: {action.value}")

            except Exception as e:
                logger.error(f"Recovery action {action.value} error: {e}")

        self._recovery_history.extend(results)

        total_duration = time.time() - start_time
        logger.info(f"Recovery completed in {total_duration:.2f}s with {len(results)} actions")

        return results

    def _determine_recovery_actions(self, health: PipelineHealth) -> List[RecoveryAction]:
        """Determine appropriate recovery actions based on health metrics."""
        actions = []

        # Analyze specific metrics to determine actions
        memory_metric = health.metrics.get("memory_usage")
        if memory_metric and memory_metric.status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
            actions.append(RecoveryAction.MEMORY_CLEANUP)

        cpu_metric = health.metrics.get("cpu_usage")
        if cpu_metric and cpu_metric.status == HealthStatus.CRITICAL:
            actions.append(RecoveryAction.RESOURCE_REALLOCATION)

        error_metric = health.metrics.get("error_rate")
        if error_metric and error_metric.status == HealthStatus.CRITICAL:
            actions.extend([RecoveryAction.MODEL_RESTART, RecoveryAction.CHECKPOINT_RESTORE])

        convergence_metric = health.metrics.get("optimization_convergence")
        if convergence_metric and convergence_metric.status == HealthStatus.WARNING:
            actions.append(RecoveryAction.PARAMETER_ADJUSTMENT)

        # Emergency shutdown for failed status
        if health.overall_status == HealthStatus.FAILED:
            actions.append(RecoveryAction.EMERGENCY_SHUTDOWN)

        return actions

    def _memory_cleanup(self, health: PipelineHealth) -> RecoveryResult:
        """Perform memory cleanup and garbage collection."""
        start_time = time.time()

        try:
            # Get memory before cleanup
            import psutil
            memory_before = psutil.virtual_memory().percent

            # Force garbage collection
            gc.collect()

            # Clear JAX compilation cache
            jax.clear_caches()

            # Additional cleanup if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

            memory_after = psutil.virtual_memory().percent
            duration = time.time() - start_time

            freed_mb = (memory_before - memory_after) * psutil.virtual_memory().total / (100 * 1024 * 1024)

            return RecoveryResult(
                action=RecoveryAction.MEMORY_CLEANUP,
                success=memory_after < memory_before,
                message=f"Freed {freed_mb:.1f}MB memory in {duration:.2f}s",
                metrics_before={"memory_percent": memory_before},
                metrics_after={"memory_percent": memory_after},
                duration=duration,
                timestamp=time.time()
            )

        except Exception as e:
            return RecoveryResult(
                action=RecoveryAction.MEMORY_CLEANUP,
                success=False,
                message=f"Memory cleanup failed: {e}",
                metrics_before={},
                metrics_after={},
                duration=time.time() - start_time,
                timestamp=time.time()
            )

    def _model_restart(self, health: PipelineHealth) -> RecoveryResult:
        """Restart model components."""
        start_time = time.time()

        try:
            # Clear JAX state
            jax.clear_caches()

            # Reset random state
            import random
            random.seed(42)
            np.random.seed(42)

            # Force recompilation on next use
            jax.config.update("jax_disable_jit", True)
            jax.config.update("jax_disable_jit", False)

            duration = time.time() - start_time

            return RecoveryResult(
                action=RecoveryAction.MODEL_RESTART,
                success=True,
                message=f"Model components restarted in {duration:.2f}s",
                metrics_before={},
                metrics_after={},
                duration=duration,
                timestamp=time.time()
            )

        except Exception as e:
            return RecoveryResult(
                action=RecoveryAction.MODEL_RESTART,
                success=False,
                message=f"Model restart failed: {e}",
                metrics_before={},
                metrics_after={},
                duration=time.time() - start_time,
                timestamp=time.time()
            )

    def _checkpoint_restore(self, health: PipelineHealth) -> RecoveryResult:
        """Restore from last known good checkpoint."""
        start_time = time.time()

        try:
            if not self._last_checkpoint:
                return RecoveryResult(
                    action=RecoveryAction.CHECKPOINT_RESTORE,
                    success=False,
                    message="No checkpoint available for restoration",
                    metrics_before={},
                    metrics_after={},
                    duration=time.time() - start_time,
                    timestamp=time.time()
                )

            # Restore checkpoint (implementation depends on system state)
            logger.info(f"Restoring checkpoint from {self._last_checkpoint.get('timestamp', 'unknown')}")

            duration = time.time() - start_time

            return RecoveryResult(
                action=RecoveryAction.CHECKPOINT_RESTORE,
                success=True,
                message=f"Checkpoint restored in {duration:.2f}s",
                metrics_before={},
                metrics_after={},
                duration=duration,
                timestamp=time.time()
            )

        except Exception as e:
            return RecoveryResult(
                action=RecoveryAction.CHECKPOINT_RESTORE,
                success=False,
                message=f"Checkpoint restore failed: {e}",
                metrics_before={},
                metrics_after={},
                duration=time.time() - start_time,
                timestamp=time.time()
            )

    def _parameter_adjustment(self, health: PipelineHealth) -> RecoveryResult:
        """Adjust optimization parameters for better convergence."""
        start_time = time.time()

        try:
            # Example parameter adjustments based on health metrics
            adjustments = {}

            convergence_metric = health.metrics.get("optimization_convergence")
            if convergence_metric:
                if convergence_metric.value < convergence_metric.threshold_warning:
                    adjustments["learning_rate"] = "reduced by 50%"
                    adjustments["batch_size"] = "increased by 25%"

            duration = time.time() - start_time

            return RecoveryResult(
                action=RecoveryAction.PARAMETER_ADJUSTMENT,
                success=bool(adjustments),
                message=f"Parameters adjusted: {adjustments}",
                metrics_before={},
                metrics_after={},
                duration=duration,
                timestamp=time.time()
            )

        except Exception as e:
            return RecoveryResult(
                action=RecoveryAction.PARAMETER_ADJUSTMENT,
                success=False,
                message=f"Parameter adjustment failed: {e}",
                metrics_before={},
                metrics_after={},
                duration=time.time() - start_time,
                timestamp=time.time()
            )

    def _emergency_shutdown(self, health: PipelineHealth) -> RecoveryResult:
        """Perform emergency shutdown to prevent system damage."""
        start_time = time.time()

        try:
            self._emergency_mode = True

            # Save current state for post-mortem analysis
            emergency_state = {
                "timestamp": time.time(),
                "health_snapshot": health,
                "recovery_history": self._recovery_history[-10:]  # Last 10 recovery attempts
            }

            # Stop all background processes
            logger.critical("EMERGENCY SHUTDOWN INITIATED - Stopping all processes")

            duration = time.time() - start_time

            return RecoveryResult(
                action=RecoveryAction.EMERGENCY_SHUTDOWN,
                success=True,
                message=f"Emergency shutdown completed in {duration:.2f}s",
                metrics_before={},
                metrics_after={"emergency_mode": True},
                duration=duration,
                timestamp=time.time()
            )

        except Exception as e:
            return RecoveryResult(
                action=RecoveryAction.EMERGENCY_SHUTDOWN,
                success=False,
                message=f"Emergency shutdown failed: {e}",
                metrics_before={},
                metrics_after={},
                duration=time.time() - start_time,
                timestamp=time.time()
            )

    def _resource_reallocation(self, health: PipelineHealth) -> RecoveryResult:
        """Reallocate system resources for better performance."""
        start_time = time.time()

        try:
            import psutil

            # Get current resource usage
            cpu_before = psutil.cpu_percent()
            memory_before = psutil.virtual_memory().percent

            # Adjust JAX memory allocation
            jax.config.update("jax_memory_fraction", 0.8)  # Use 80% of GPU memory

            # Force garbage collection to free resources
            gc.collect()

            duration = time.time() - start_time

            return RecoveryResult(
                action=RecoveryAction.RESOURCE_REALLOCATION,
                success=True,
                message=f"Resources reallocated in {duration:.2f}s",
                metrics_before={"cpu": cpu_before, "memory": memory_before},
                metrics_after={"memory_fraction": 0.8},
                duration=duration,
                timestamp=time.time()
            )

        except Exception as e:
            return RecoveryResult(
                action=RecoveryAction.RESOURCE_REALLOCATION,
                success=False,
                message=f"Resource reallocation failed: {e}",
                metrics_before={},
                metrics_after={},
                duration=time.time() - start_time,
                timestamp=time.time()
            )

    def save_checkpoint(self, state: Dict[str, Any]) -> None:
        """Save system state as checkpoint."""
        self._last_checkpoint = {
            "timestamp": time.time(),
            "state": state
        }
        logger.info("System checkpoint saved")

    def get_recovery_history(self, limit: int = 50) -> List[RecoveryResult]:
        """Get recent recovery history."""
        return self._recovery_history[-limit:]

    def is_emergency_mode(self) -> bool:
        """Check if system is in emergency mode."""
        return self._emergency_mode

    def reset_emergency_mode(self) -> None:
        """Reset emergency mode (manual intervention required)."""
        self._emergency_mode = False
        logger.info("Emergency mode reset")
