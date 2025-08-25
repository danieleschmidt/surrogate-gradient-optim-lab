"""Auto-scaling capabilities for Generation 3 scalable optimization."""

import concurrent.futures
from dataclasses import dataclass, field
from enum import Enum
import multiprocessing as mp
import threading
import time
from typing import Any, Callable, List, Optional

from jax import Array

from ..health.system_monitor import system_monitor
from ..monitoring.enhanced_logging import enhanced_logger


class ScalingMode(Enum):
    """Auto-scaling modes."""

    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


@dataclass
class ScalingMetrics:
    """Metrics for auto-scaling decisions."""

    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    queue_size: int = 0
    average_processing_time: float = 0.0
    throughput: float = 0.0
    error_rate: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class ScalingConfig:
    """Configuration for auto-scaling."""

    mode: ScalingMode = ScalingMode.BALANCED
    min_workers: int = 1
    max_workers: int = mp.cpu_count()
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.3
    scale_up_cooldown: float = 60.0
    scale_down_cooldown: float = 120.0
    metrics_window: int = 10


class AutoScaler:
    """Intelligent auto-scaling for optimization workloads."""

    def __init__(self, config: ScalingConfig):
        self.config = config
        self.current_workers = config.min_workers
        self.metrics_history: List[ScalingMetrics] = []
        self.last_scale_up: float = 0.0
        self.last_scale_down: float = 0.0
        self._lock = threading.Lock()

        # Worker pool
        self.executor: Optional[concurrent.futures.ProcessPoolExecutor] = None
        self._initialize_workers()

    def _initialize_workers(self) -> None:
        """Initialize worker pool."""
        if self.executor:
            self.executor.shutdown(wait=False)

        self.executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=self.current_workers
        )
        enhanced_logger.info(
            f"Initialized worker pool with {self.current_workers} workers"
        )

    def collect_metrics(self) -> ScalingMetrics:
        """Collect current scaling metrics."""
        health_report = system_monitor.get_health_report()

        metrics = ScalingMetrics(
            cpu_usage=health_report.get("current", {}).get("cpu_usage", 0.0),
            memory_usage=health_report.get("current", {}).get("memory_usage", 0.0),
            error_rate=health_report.get("current", {}).get("error_rate", 0.0),
        )

        # Calculate queue size and throughput if available
        if hasattr(self.executor, "_work_queue"):
            metrics.queue_size = self.executor._work_queue.qsize()

        return metrics

    def should_scale_up(self, metrics: ScalingMetrics) -> bool:
        """Determine if scaling up is needed."""
        current_time = time.time()

        # Check cooldown
        if current_time - self.last_scale_up < self.config.scale_up_cooldown:
            return False

        # Check if at max capacity
        if self.current_workers >= self.config.max_workers:
            return False

        # Check scaling conditions based on mode
        if self.config.mode == ScalingMode.AGGRESSIVE:
            threshold = 0.6
        elif self.config.mode == ScalingMode.CONSERVATIVE:
            threshold = 0.9
        else:  # BALANCED
            threshold = self.config.scale_up_threshold

        # Scale up if CPU/memory usage is high or queue is building up
        return (
            metrics.cpu_usage > threshold
            or metrics.memory_usage > threshold
            or metrics.queue_size > self.current_workers * 2
        )

    def should_scale_down(self, metrics: ScalingMetrics) -> bool:
        """Determine if scaling down is needed."""
        current_time = time.time()

        # Check cooldown
        if current_time - self.last_scale_down < self.config.scale_down_cooldown:
            return False

        # Check if at min capacity
        if self.current_workers <= self.config.min_workers:
            return False

        # Check scaling conditions
        threshold = self.config.scale_down_threshold

        # Scale down if resource usage is consistently low
        if len(self.metrics_history) >= 3:
            recent_metrics = self.metrics_history[-3:]
            avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m.memory_usage for m in recent_metrics) / len(
                recent_metrics
            )

            return (
                avg_cpu < threshold
                and avg_memory < threshold
                and metrics.queue_size == 0
            )

        return False

    def scale_up(self) -> None:
        """Scale up worker pool."""
        with self._lock:
            new_workers = min(self.current_workers + 1, self.config.max_workers)
            if new_workers > self.current_workers:
                self.current_workers = new_workers
                self.last_scale_up = time.time()
                self._initialize_workers()
                enhanced_logger.info(f"Scaled up to {self.current_workers} workers")

    def scale_down(self) -> None:
        """Scale down worker pool."""
        with self._lock:
            new_workers = max(self.current_workers - 1, self.config.min_workers)
            if new_workers < self.current_workers:
                self.current_workers = new_workers
                self.last_scale_down = time.time()
                self._initialize_workers()
                enhanced_logger.info(f"Scaled down to {self.current_workers} workers")

    def auto_scale(self) -> None:
        """Perform auto-scaling decision."""
        metrics = self.collect_metrics()

        # Store metrics
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.config.metrics_window:
            self.metrics_history = self.metrics_history[-self.config.metrics_window :]

        # Make scaling decisions
        if self.should_scale_up(metrics):
            self.scale_up()
        elif self.should_scale_down(metrics):
            self.scale_down()

    def submit_task(self, func: Callable, *args, **kwargs) -> concurrent.futures.Future:
        """Submit task to worker pool with auto-scaling."""
        # Trigger auto-scaling check
        self.auto_scale()

        # Submit task
        if self.executor is None:
            self._initialize_workers()

        return self.executor.submit(func, *args, **kwargs)

    def map_parallel(self, func: Callable, iterable: List[Any]) -> List[Any]:
        """Map function over iterable with auto-scaling."""
        # Check if scaling is beneficial
        if len(iterable) < self.current_workers:
            # Sequential execution for small batches
            return [func(item) for item in iterable]

        # Trigger auto-scaling
        self.auto_scale()

        # Parallel execution
        if self.executor is None:
            self._initialize_workers()

        futures = [self.executor.submit(func, item) for item in iterable]
        return [future.result() for future in futures]

    def shutdown(self) -> None:
        """Shutdown worker pool."""
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None
        enhanced_logger.info("Auto-scaler shutdown complete")


class AdaptiveOptimizer:
    """Adaptive optimization with auto-scaling capabilities."""

    def __init__(self, scaling_config: Optional[ScalingConfig] = None):
        self.scaling_config = scaling_config or ScalingConfig()
        self.auto_scaler = AutoScaler(self.scaling_config)
        self.performance_history: List[float] = []

    def optimize_parallel(
        self,
        objective_function: Callable,
        initial_points: List[Array],
        **optimizer_kwargs,
    ) -> List[Array]:
        """Run optimization from multiple starting points in parallel."""
        enhanced_logger.info(
            f"Starting parallel optimization with {len(initial_points)} initial points"
        )

        def single_optimization(x0: Array) -> Array:
            """Single optimization run."""
            # Import here to avoid circular imports
            from ..optimizers.gradient_descent import GradientDescentOptimizer

            optimizer = GradientDescentOptimizer(**optimizer_kwargs)
            result = optimizer.optimize(objective_function, x0)
            return result.x if hasattr(result, "x") else result

        # Use auto-scaling parallel execution
        start_time = time.time()
        results = self.auto_scaler.map_parallel(single_optimization, initial_points)
        execution_time = time.time() - start_time

        # Track performance
        self.performance_history.append(execution_time)
        enhanced_logger.info(
            f"Parallel optimization completed in {execution_time:.2f}s"
        )

        return results

    def adaptive_batch_size(self, base_size: int, current_load: float) -> int:
        """Determine optimal batch size based on current system load."""
        if current_load > 0.8:
            # Reduce batch size under high load
            return max(1, base_size // 2)
        if current_load < 0.3:
            # Increase batch size under low load
            return min(base_size * 2, 1000)
        return base_size

    def shutdown(self) -> None:
        """Shutdown adaptive optimizer."""
        self.auto_scaler.shutdown()


# Global adaptive optimizer instance
adaptive_optimizer = AdaptiveOptimizer()
