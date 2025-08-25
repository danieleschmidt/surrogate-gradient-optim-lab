"""Intelligent load balancing for scalable surrogate optimization."""

from collections import deque
import concurrent.futures
from dataclasses import dataclass
from enum import Enum
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..monitoring.enhanced_logging import enhanced_logger


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""

    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    ADAPTIVE = "adaptive"
    PERFORMANCE_BASED = "performance_based"


@dataclass
class WorkerNode:
    """Represents a worker node in the cluster."""

    node_id: str
    cpu_cores: int
    memory_gb: float
    gpu_available: bool = False
    current_load: float = 0.0
    performance_score: float = 1.0
    last_response_time: float = 0.0
    total_tasks: int = 0
    failed_tasks: int = 0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_tasks == 0:
            return 1.0
        return (self.total_tasks - self.failed_tasks) / self.total_tasks

    @property
    def availability_score(self) -> float:
        """Calculate availability score."""
        load_factor = max(0.0, 1.0 - self.current_load)
        performance_factor = self.performance_score
        reliability_factor = self.success_rate

        return load_factor * performance_factor * reliability_factor


class IntelligentLoadBalancer:
    """Intelligent load balancer with adaptive routing."""

    def __init__(
        self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE
    ):
        self.strategy = strategy
        self.workers: Dict[str, WorkerNode] = {}
        self.task_queue = deque()
        self.completed_tasks: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._round_robin_index = 0

        # Performance tracking
        self.performance_history: Dict[str, List[float]] = {}

        # Initialize default local worker
        self._add_local_worker()

    def _add_local_worker(self) -> None:
        """Add local worker node."""
        import multiprocessing as mp

        import psutil

        local_worker = WorkerNode(
            node_id="local",
            cpu_cores=mp.cpu_count(),
            memory_gb=psutil.virtual_memory().total / (1024**3),
            gpu_available=len(jax.devices("gpu")) > 0,
        )

        self.workers["local"] = local_worker
        enhanced_logger.info(
            f"Added local worker: {mp.cpu_count()} cores, {local_worker.memory_gb:.1f}GB RAM"
        )

    def add_worker(self, worker: WorkerNode) -> None:
        """Add worker node to the cluster."""
        with self._lock:
            self.workers[worker.node_id] = worker
            self.performance_history[worker.node_id] = []
        enhanced_logger.info(f"Added worker {worker.node_id}")

    def remove_worker(self, node_id: str) -> None:
        """Remove worker node from cluster."""
        with self._lock:
            if node_id in self.workers:
                del self.workers[node_id]
                if node_id in self.performance_history:
                    del self.performance_history[node_id]
        enhanced_logger.info(f"Removed worker {node_id}")

    def select_worker(self) -> Optional[str]:
        """Select optimal worker based on strategy."""
        if not self.workers:
            return None

        with self._lock:
            if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                return self._select_round_robin()
            if self.strategy == LoadBalancingStrategy.LEAST_LOADED:
                return self._select_least_loaded()
            if self.strategy == LoadBalancingStrategy.PERFORMANCE_BASED:
                return self._select_performance_based()
            # ADAPTIVE
            return self._select_adaptive()

    def _select_round_robin(self) -> str:
        """Round-robin worker selection."""
        worker_ids = list(self.workers.keys())
        selected = worker_ids[self._round_robin_index % len(worker_ids)]
        self._round_robin_index += 1
        return selected

    def _select_least_loaded(self) -> str:
        """Select worker with lowest current load."""
        return min(self.workers.keys(), key=lambda w: self.workers[w].current_load)

    def _select_performance_based(self) -> str:
        """Select worker based on performance score."""
        return max(self.workers.keys(), key=lambda w: self.workers[w].performance_score)

    def _select_adaptive(self) -> str:
        """Adaptive worker selection considering multiple factors."""
        # Use availability score that combines load, performance, and reliability
        return max(
            self.workers.keys(), key=lambda w: self.workers[w].availability_score
        )

    def submit_task(
        self,
        func: Callable,
        args: Tuple = (),
        kwargs: Dict[str, Any] = None,
        priority: int = 0,
    ) -> concurrent.futures.Future:
        """Submit task to optimal worker."""
        kwargs = kwargs or {}

        # Select worker
        worker_id = self.select_worker()
        if not worker_id:
            raise RuntimeError("No workers available")

        worker = self.workers[worker_id]

        # Update worker load (simplified)
        worker.current_load = min(1.0, worker.current_load + 0.1)
        worker.total_tasks += 1

        enhanced_logger.debug(f"Submitting task to worker {worker_id}")

        # Submit task (simplified - in practice would use actual distributed execution)
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=worker.cpu_cores)
        future = executor.submit(func, *args, **kwargs)

        # Add callback to update worker metrics
        def update_metrics(future_result):
            try:
                result = future_result.result()
                # Task succeeded
                worker.current_load = max(0.0, worker.current_load - 0.1)

                # Update performance history
                if worker_id not in self.performance_history:
                    self.performance_history[worker_id] = []

                self.performance_history[worker_id].append(time.time())

                # Keep only recent history
                if len(self.performance_history[worker_id]) > 100:
                    self.performance_history[worker_id] = self.performance_history[
                        worker_id
                    ][-100:]

            except Exception:
                # Task failed
                worker.failed_tasks += 1
                worker.current_load = max(0.0, worker.current_load - 0.1)

        future.add_done_callback(update_metrics)
        return future

    def get_cluster_status(self) -> Dict[str, Any]:
        """Get status of the entire cluster."""
        with self._lock:
            status = {
                "total_workers": len(self.workers),
                "strategy": self.strategy.value,
                "workers": {},
                "cluster_load": 0.0,
                "cluster_performance": 0.0,
            }

            total_load = 0.0
            total_performance = 0.0

            for worker_id, worker in self.workers.items():
                status["workers"][worker_id] = {
                    "load": worker.current_load,
                    "performance": worker.performance_score,
                    "success_rate": worker.success_rate,
                    "availability": worker.availability_score,
                    "cpu_cores": worker.cpu_cores,
                    "memory_gb": worker.memory_gb,
                    "gpu_available": worker.gpu_available,
                }

                total_load += worker.current_load
                total_performance += worker.performance_score

            if self.workers:
                status["cluster_load"] = total_load / len(self.workers)
                status["cluster_performance"] = total_performance / len(self.workers)

            return status


# Global load balancer instance
intelligent_load_balancer = IntelligentLoadBalancer()
