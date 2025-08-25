"""Scalable architecture components for distributed self-healing optimization."""

import concurrent.futures
from dataclasses import dataclass, field
from enum import Enum
import multiprocessing as mp
import queue
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import jax
import jax.numpy as jnp
from loguru import logger
import numpy as np

# Distributed computing imports with fallbacks
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    logger.warning("Ray not available. Using local parallel processing.")

try:
    import dask
    from dask.distributed import Client, as_completed
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

from .robust_monitoring import RobustMonitor


class ScalingStrategy(Enum):
    """Scaling strategies for optimization workloads."""
    LOCAL_THREADING = "local_threading"
    LOCAL_MULTIPROCESSING = "local_multiprocessing"
    RAY_DISTRIBUTED = "ray_distributed"
    DASK_DISTRIBUTED = "dask_distributed"
    HYBRID_SCALING = "hybrid_scaling"


class WorkloadType(Enum):
    """Types of optimization workloads."""
    SINGLE_OBJECTIVE = "single_objective"
    MULTI_OBJECTIVE = "multi_objective"
    SURROGATE_TRAINING = "surrogate_training"
    HYPERPARAMETER_SEARCH = "hyperparameter_search"
    BATCH_EVALUATION = "batch_evaluation"


@dataclass
class ScalingConfig:
    """Configuration for scalable optimization."""
    strategy: ScalingStrategy = ScalingStrategy.LOCAL_THREADING
    max_workers: int = 4
    chunk_size: int = 100
    prefetch_factor: int = 2
    memory_limit_gb: float = 8.0
    enable_gpu: bool = True
    enable_jit: bool = True
    distributed_backend: Optional[str] = None
    cluster_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkloadPartition:
    """Partitioned workload for distributed processing."""
    partition_id: str
    workload_type: WorkloadType
    data: Any
    config: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    estimated_duration: float = 0.0
    memory_requirement: float = 0.0


@dataclass
class ScalingMetrics:
    """Metrics for scaling performance."""
    total_workers: int
    active_workers: int
    queued_tasks: int
    completed_tasks: int
    failed_tasks: int
    average_task_duration: float
    throughput_per_second: float
    memory_usage_gb: float
    cpu_utilization: float
    gpu_utilization: float = 0.0


class DistributedTaskManager:
    """Manages distributed task execution with fault tolerance."""

    def __init__(self, config: ScalingConfig):
        self.config = config
        self.strategy = config.strategy

        # Task management
        self._task_queue: queue.Queue = queue.Queue()
        self._result_cache: Dict[str, Any] = {}
        self._task_status: Dict[str, str] = {}

        # Distributed backends
        self._ray_client = None
        self._dask_client = None
        self._executor = None

        # Metrics
        self._metrics = ScalingMetrics(
            total_workers=config.max_workers,
            active_workers=0,
            queued_tasks=0,
            completed_tasks=0,
            failed_tasks=0,
            average_task_duration=0.0,
            throughput_per_second=0.0,
            memory_usage_gb=0.0,
            cpu_utilization=0.0
        )

        self._initialize_backend()

    def _initialize_backend(self) -> None:
        """Initialize the distributed computing backend."""
        try:
            if self.strategy == ScalingStrategy.RAY_DISTRIBUTED and RAY_AVAILABLE:
                self._initialize_ray()
            elif self.strategy == ScalingStrategy.DASK_DISTRIBUTED and DASK_AVAILABLE:
                self._initialize_dask()
            elif self.strategy in [ScalingStrategy.LOCAL_THREADING, ScalingStrategy.LOCAL_MULTIPROCESSING]:
                self._initialize_local()
            else:
                logger.warning(f"Strategy {self.strategy} not available, falling back to local threading")
                self.strategy = ScalingStrategy.LOCAL_THREADING
                self._initialize_local()

        except Exception as e:
            logger.error(f"Failed to initialize backend {self.strategy}: {e}")
            logger.info("Falling back to local threading")
            self.strategy = ScalingStrategy.LOCAL_THREADING
            self._initialize_local()

    def _initialize_ray(self) -> None:
        """Initialize Ray distributed backend."""
        if not ray.is_initialized():
            cluster_config = self.config.cluster_config.get("ray", {})
            ray.init(**cluster_config)

        self._ray_client = ray
        logger.info(f"Ray initialized with {ray.cluster_resources()}")

    def _initialize_dask(self) -> None:
        """Initialize Dask distributed backend."""
        cluster_config = self.config.cluster_config.get("dask", {})

        if "scheduler_address" in cluster_config:
            self._dask_client = Client(cluster_config["scheduler_address"])
        else:
            self._dask_client = Client(n_workers=self.config.max_workers, threads_per_worker=2)

        logger.info(f"Dask initialized: {self._dask_client}")

    def _initialize_local(self) -> None:
        """Initialize local parallel processing."""
        if self.strategy == ScalingStrategy.LOCAL_MULTIPROCESSING:
            self._executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.config.max_workers
            )
        else:
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.config.max_workers
            )

        logger.info(f"Local executor initialized with {self.config.max_workers} workers")

    def submit_batch(self, workloads: List[WorkloadPartition]) -> List[concurrent.futures.Future]:
        """Submit batch of workloads for distributed processing."""
        futures = []

        for workload in workloads:
            future = self.submit_single(workload)
            futures.append(future)

        return futures

    def submit_single(self, workload: WorkloadPartition) -> concurrent.futures.Future:
        """Submit single workload for processing."""
        task_id = workload.partition_id
        self._task_status[task_id] = "submitted"
        self._metrics.queued_tasks += 1

        if self.strategy == ScalingStrategy.RAY_DISTRIBUTED and self._ray_client:
            future = self._submit_ray_task(workload)
        elif self.strategy == ScalingStrategy.DASK_DISTRIBUTED and self._dask_client:
            future = self._submit_dask_task(workload)
        else:
            future = self._submit_local_task(workload)

        return future

    def _submit_ray_task(self, workload: WorkloadPartition) -> Any:
        """Submit task to Ray cluster."""

        @ray.remote
        def process_workload(workload_data):
            return self._process_workload_partition(workload_data)

        future = process_workload.remote(workload)
        return future

    def _submit_dask_task(self, workload: WorkloadPartition) -> Any:
        """Submit task to Dask cluster."""
        future = self._dask_client.submit(self._process_workload_partition, workload)
        return future

    def _submit_local_task(self, workload: WorkloadPartition) -> concurrent.futures.Future:
        """Submit task to local executor."""
        future = self._executor.submit(self._process_workload_partition, workload)
        return future

    def _process_workload_partition(self, workload: WorkloadPartition) -> Dict[str, Any]:
        """Process a single workload partition."""
        start_time = time.time()

        try:
            # Configure JAX for this worker
            if self.config.enable_jit:
                jax.config.update("jax_enable_x64", True)

            if self.config.enable_gpu and jax.devices("gpu"):
                jax.config.update("jax_default_device", jax.devices("gpu")[0])

            # Process based on workload type
            if workload.workload_type == WorkloadType.SINGLE_OBJECTIVE:
                result = self._process_single_objective(workload)
            elif workload.workload_type == WorkloadType.MULTI_OBJECTIVE:
                result = self._process_multi_objective(workload)
            elif workload.workload_type == WorkloadType.SURROGATE_TRAINING:
                result = self._process_surrogate_training(workload)
            elif workload.workload_type == WorkloadType.BATCH_EVALUATION:
                result = self._process_batch_evaluation(workload)
            else:
                raise ValueError(f"Unknown workload type: {workload.workload_type}")

            duration = time.time() - start_time

            return {
                "partition_id": workload.partition_id,
                "status": "success",
                "result": result,
                "duration": duration,
                "memory_used": self._get_memory_usage()
            }

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Workload {workload.partition_id} failed: {e}")

            return {
                "partition_id": workload.partition_id,
                "status": "failed",
                "error": str(e),
                "duration": duration,
                "memory_used": self._get_memory_usage()
            }

    def _process_single_objective(self, workload: WorkloadPartition) -> Any:
        """Process single-objective optimization workload."""
        # Extract optimization parameters
        objective_fn = workload.data.get("objective_function")
        initial_points = workload.data.get("initial_points")
        bounds = workload.data.get("bounds")
        config = workload.config

        results = []
        for i, x0 in enumerate(initial_points):
            try:
                # Run optimization
                result = self._run_optimization(objective_fn, x0, bounds, config)
                results.append(result)
            except Exception as e:
                logger.warning(f"Optimization {i} failed: {e}")
                results.append({"status": "failed", "error": str(e)})

        return {"optimization_results": results}

    def _process_multi_objective(self, workload: WorkloadPartition) -> Any:
        """Process multi-objective optimization workload."""
        # Multi-objective specific processing
        objectives = workload.data.get("objectives")
        initial_population = workload.data.get("initial_population")
        config = workload.config

        # Simplified multi-objective processing
        pareto_front = []
        for individual in initial_population:
            obj_values = [obj(individual) for obj in objectives]
            pareto_front.append({"individual": individual, "objectives": obj_values})

        return {"pareto_front": pareto_front}

    def _process_surrogate_training(self, workload: WorkloadPartition) -> Any:
        """Process surrogate model training workload."""
        training_data = workload.data.get("training_data")
        model_config = workload.config.get("model_config", {})

        # Simplified surrogate training
        X, y = training_data["X"], training_data["y"]

        # Mock training process
        model_params = {
            "weights": np.random.randn(X.shape[1], 10),
            "bias": np.random.randn(10),
            "training_loss": np.random.uniform(0.01, 0.1)
        }

        return {"model_parameters": model_params, "training_metrics": {"loss": 0.05}}

    def _process_batch_evaluation(self, workload: WorkloadPartition) -> Any:
        """Process batch function evaluation workload."""
        function = workload.data.get("function")
        points = workload.data.get("points")

        # Vectorized evaluation
        if self.config.enable_gpu and jax.devices("gpu"):
            points_jax = jnp.array(points)
            values = jax.vmap(function)(points_jax)
            values = np.array(values)
        else:
            values = [function(point) for point in points]

        return {"evaluations": values}

    def _run_optimization(self, objective_fn: Callable, x0: np.ndarray, bounds: List[tuple], config: Dict) -> Dict:
        """Run single optimization with configured parameters."""
        from scipy.optimize import minimize

        # Configure optimization
        method = config.get("method", "L-BFGS-B")
        max_iter = config.get("max_iterations", 100)
        tolerance = config.get("tolerance", 1e-6)

        # Run optimization
        result = minimize(
            objective_fn,
            x0,
            method=method,
            bounds=bounds,
            options={"maxiter": max_iter}
        )

        return {
            "x": result.x.tolist() if hasattr(result.x, "tolist") else result.x,
            "fun": float(result.fun),
            "success": bool(result.success),
            "nfev": int(result.nfev),
            "message": result.message
        }

    def _get_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024**3  # Convert to GB
        except Exception:
            return 0.0

    def gather_results(self, futures: List[concurrent.futures.Future], timeout: Optional[float] = None) -> List[Dict[str, Any]]:
        """Gather results from distributed tasks."""
        results = []

        if self.strategy == ScalingStrategy.RAY_DISTRIBUTED and self._ray_client:
            results = ray.get(futures, timeout=timeout)
        elif self.strategy == ScalingStrategy.DASK_DISTRIBUTED and self._dask_client:
            results = self._dask_client.gather(futures)
        else:
            # Local executor results
            for future in concurrent.futures.as_completed(futures, timeout=timeout):
                try:
                    result = future.result()
                    results.append(result)
                    self._metrics.completed_tasks += 1
                except Exception as e:
                    logger.error(f"Task failed: {e}")
                    self._metrics.failed_tasks += 1
                    results.append({"status": "failed", "error": str(e)})

        return results

    def get_scaling_metrics(self) -> ScalingMetrics:
        """Get current scaling performance metrics."""
        # Update metrics
        try:
            import psutil
            self._metrics.memory_usage_gb = psutil.virtual_memory().used / 1024**3
            self._metrics.cpu_utilization = psutil.cpu_percent()
        except Exception:
            pass

        # Calculate throughput
        if self._metrics.completed_tasks > 0:
            total_time = time.time() - getattr(self, "_start_time", time.time())
            self._metrics.throughput_per_second = self._metrics.completed_tasks / max(1, total_time)

        return self._metrics

    def shutdown(self) -> None:
        """Shutdown distributed resources."""
        if self._executor:
            self._executor.shutdown(wait=True)

        if self._dask_client:
            self._dask_client.close()

        if self._ray_client and ray.is_initialized():
            ray.shutdown()

        logger.info("Distributed task manager shutdown complete")


class AutoScaler:
    """Automatic scaling system for optimization workloads."""

    def __init__(
        self,
        initial_config: ScalingConfig,
        monitor: Optional[RobustMonitor] = None
    ):
        self.config = initial_config
        self.monitor = monitor

        # Scaling history
        self._scaling_history: List[Dict[str, Any]] = []
        self._performance_history: List[ScalingMetrics] = []

        # Auto-scaling parameters
        self.min_workers = 1
        self.max_workers = mp.cpu_count() * 2
        self.scale_up_threshold = 0.8  # CPU/memory usage
        self.scale_down_threshold = 0.3
        self.scale_up_delay = 30.0  # seconds
        self.scale_down_delay = 60.0

        self._last_scale_action = 0.0

    def analyze_workload(self, workloads: List[WorkloadPartition]) -> ScalingConfig:
        """Analyze workloads and determine optimal scaling configuration."""
        # Estimate resource requirements
        total_memory_req = sum(w.memory_requirement for w in workloads)
        total_duration_est = sum(w.estimated_duration for w in workloads)

        # Classify workload characteristics
        workload_types = [w.workload_type for w in workloads]

        # Determine optimal strategy
        if len(workloads) > 100 and RAY_AVAILABLE:
            strategy = ScalingStrategy.RAY_DISTRIBUTED
            max_workers = min(self.max_workers, len(workloads) // 10)
        elif len(workloads) > 50 and DASK_AVAILABLE:
            strategy = ScalingStrategy.DASK_DISTRIBUTED
            max_workers = min(self.max_workers, len(workloads) // 5)
        elif any(wt == WorkloadType.SURROGATE_TRAINING for wt in workload_types):
            # CPU-intensive tasks benefit from multiprocessing
            strategy = ScalingStrategy.LOCAL_MULTIPROCESSING
            max_workers = min(mp.cpu_count(), len(workloads))
        else:
            strategy = ScalingStrategy.LOCAL_THREADING
            max_workers = min(self.config.max_workers, len(workloads))

        # Adjust for memory constraints
        if total_memory_req > self.config.memory_limit_gb:
            max_workers = max(1, int(max_workers * self.config.memory_limit_gb / total_memory_req))

        return ScalingConfig(
            strategy=strategy,
            max_workers=max_workers,
            chunk_size=max(1, len(workloads) // max_workers),
            memory_limit_gb=self.config.memory_limit_gb,
            enable_gpu=self.config.enable_gpu and any(wt == WorkloadType.BATCH_EVALUATION for wt in workload_types),
            enable_jit=self.config.enable_jit
        )

    def should_scale_up(self, metrics: ScalingMetrics) -> bool:
        """Determine if system should scale up."""
        if time.time() - self._last_scale_action < self.scale_up_delay:
            return False

        # Check resource utilization
        high_cpu = metrics.cpu_utilization > self.scale_up_threshold * 100
        high_memory = metrics.memory_usage_gb > self.scale_up_threshold * self.config.memory_limit_gb
        high_queue = metrics.queued_tasks > metrics.active_workers * 2

        return (high_cpu or high_memory or high_queue) and metrics.active_workers < self.max_workers

    def should_scale_down(self, metrics: ScalingMetrics) -> bool:
        """Determine if system should scale down."""
        if time.time() - self._last_scale_action < self.scale_down_delay:
            return False

        # Check for low utilization
        low_cpu = metrics.cpu_utilization < self.scale_down_threshold * 100
        low_memory = metrics.memory_usage_gb < self.scale_down_threshold * self.config.memory_limit_gb
        low_queue = metrics.queued_tasks < metrics.active_workers // 2

        return (low_cpu and low_memory and low_queue) and metrics.active_workers > self.min_workers

    def execute_scaling_decision(self, metrics: ScalingMetrics, task_manager: DistributedTaskManager) -> None:
        """Execute scaling decision based on current metrics."""
        if self.should_scale_up(metrics):
            new_worker_count = min(self.max_workers, metrics.active_workers + 1)
            logger.info(f"Scaling up to {new_worker_count} workers")

            # Update configuration
            self.config.max_workers = new_worker_count
            self._record_scaling_action("scale_up", new_worker_count)

        elif self.should_scale_down(metrics):
            new_worker_count = max(self.min_workers, metrics.active_workers - 1)
            logger.info(f"Scaling down to {new_worker_count} workers")

            # Update configuration
            self.config.max_workers = new_worker_count
            self._record_scaling_action("scale_down", new_worker_count)

    def _record_scaling_action(self, action: str, new_worker_count: int) -> None:
        """Record scaling action in history."""
        self._scaling_history.append({
            "timestamp": time.time(),
            "action": action,
            "worker_count": new_worker_count,
        })

        self._last_scale_action = time.time()

        # Keep only recent history
        if len(self._scaling_history) > 100:
            self._scaling_history = self._scaling_history[-100:]


class ScalableOptimizer:
    """Main scalable optimizer with auto-scaling capabilities."""

    def __init__(
        self,
        scaling_config: Optional[ScalingConfig] = None,
        enable_auto_scaling: bool = True,
        monitor: Optional[RobustMonitor] = None
    ):
        self.scaling_config = scaling_config or ScalingConfig()
        self.enable_auto_scaling = enable_auto_scaling
        self.monitor = monitor

        # Components
        self.task_manager = DistributedTaskManager(self.scaling_config)
        self.auto_scaler = AutoScaler(self.scaling_config, monitor) if enable_auto_scaling else None

        # State tracking
        self._active_optimizations: Dict[str, Any] = {}

        logger.info(f"Scalable optimizer initialized with {self.scaling_config.strategy.value} strategy")

    def optimize_batch(
        self,
        objective_functions: List[Callable],
        initial_points: List[np.ndarray],
        bounds: List[List[tuple]],
        optimization_configs: Optional[List[Dict]] = None
    ) -> List[Dict[str, Any]]:
        """Optimize multiple problems in parallel."""
        if optimization_configs is None:
            optimization_configs = [{}] * len(objective_functions)

        # Create workload partitions
        workloads = []
        for i, (obj_fn, x0, bound, config) in enumerate(zip(
            objective_functions, initial_points, bounds, optimization_configs
        )):
            workload = WorkloadPartition(
                partition_id=f"optimization_{i}",
                workload_type=WorkloadType.SINGLE_OBJECTIVE,
                data={
                    "objective_function": obj_fn,
                    "initial_points": [x0],
                    "bounds": bound
                },
                config=config,
                estimated_duration=config.get("estimated_duration", 60.0),
                memory_requirement=config.get("memory_requirement", 1.0)
            )
            workloads.append(workload)

        return self._execute_workloads(workloads)

    def train_surrogates_parallel(
        self,
        training_datasets: List[Dict[str, np.ndarray]],
        model_configs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Train multiple surrogate models in parallel."""
        workloads = []
        for i, (dataset, config) in enumerate(zip(training_datasets, model_configs)):
            workload = WorkloadPartition(
                partition_id=f"surrogate_{i}",
                workload_type=WorkloadType.SURROGATE_TRAINING,
                data={"training_data": dataset},
                config={"model_config": config},
                estimated_duration=config.get("estimated_duration", 120.0),
                memory_requirement=config.get("memory_requirement", 2.0)
            )
            workloads.append(workload)

        return self._execute_workloads(workloads)

    def evaluate_batch(
        self,
        functions: List[Callable],
        point_batches: List[List[np.ndarray]]
    ) -> List[List[float]]:
        """Evaluate functions at multiple points in parallel."""
        workloads = []
        for i, (func, points) in enumerate(zip(functions, point_batches)):
            workload = WorkloadPartition(
                partition_id=f"evaluation_{i}",
                workload_type=WorkloadType.BATCH_EVALUATION,
                data={"function": func, "points": points},
                config={},
                estimated_duration=len(points) * 0.1,  # Estimate
                memory_requirement=len(points) * 0.001  # Estimate
            )
            workloads.append(workload)

        results = self._execute_workloads(workloads)
        return [r["result"]["evaluations"] for r in results]

    def _execute_workloads(self, workloads: List[WorkloadPartition]) -> List[Dict[str, Any]]:
        """Execute workloads with auto-scaling monitoring."""
        # Analyze workloads and adjust scaling if needed
        if self.auto_scaler:
            optimal_config = self.auto_scaler.analyze_workload(workloads)

            # Update task manager if configuration changed significantly
            if optimal_config.strategy != self.scaling_config.strategy:
                logger.info(f"Switching scaling strategy from {self.scaling_config.strategy.value} to {optimal_config.strategy.value}")
                self.task_manager.shutdown()
                self.task_manager = DistributedTaskManager(optimal_config)
                self.scaling_config = optimal_config

        # Submit workloads
        logger.info(f"Executing {len(workloads)} workloads with {self.scaling_config.strategy.value}")
        futures = self.task_manager.submit_batch(workloads)

        # Monitor and potentially scale during execution
        if self.auto_scaler and self.enable_auto_scaling:
            self._monitor_and_scale(futures)

        # Gather results
        results = self.task_manager.gather_results(futures)

        logger.info(f"Completed {len(results)} workloads")
        return results

    def _monitor_and_scale(self, futures: List[concurrent.futures.Future]) -> None:
        """Monitor execution and apply auto-scaling."""

        # Start monitoring thread
        def monitor_loop():
            while any(not f.done() for f in futures):
                metrics = self.task_manager.get_scaling_metrics()

                if self.auto_scaler:
                    self.auto_scaler.execute_scaling_decision(metrics, self.task_manager)

                time.sleep(10)  # Check every 10 seconds

        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        metrics = self.task_manager.get_scaling_metrics()

        summary = {
            "scaling_strategy": self.scaling_config.strategy.value,
            "current_metrics": metrics,
            "configuration": {
                "max_workers": self.scaling_config.max_workers,
                "chunk_size": self.scaling_config.chunk_size,
                "memory_limit_gb": self.scaling_config.memory_limit_gb,
                "gpu_enabled": self.scaling_config.enable_gpu
            }
        }

        if self.auto_scaler:
            summary["scaling_history"] = self.auto_scaler._scaling_history[-10:]

        return summary

    def shutdown(self) -> None:
        """Shutdown scalable optimizer."""
        self.task_manager.shutdown()
        logger.info("Scalable optimizer shutdown complete")
