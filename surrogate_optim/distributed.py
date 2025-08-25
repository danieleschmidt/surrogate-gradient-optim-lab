"""Distributed computing support for large-scale surrogate optimization."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import pickle
import queue
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from jax import Array
import jax.numpy as jnp


@dataclass
class ComputeTask:
    """Represents a computation task for distributed execution."""
    task_id: str
    task_type: str
    function_name: str
    args: tuple
    kwargs: dict
    priority: int = 0
    created_at: float = 0.0

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()


@dataclass
class TaskResult:
    """Result of a distributed computation task."""
    task_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    worker_id: Optional[str] = None
    completed_at: float = 0.0

    def __post_init__(self):
        if self.completed_at == 0.0:
            self.completed_at = time.time()


class DistributedTaskManager:
    """Manages distributed task execution across multiple workers."""

    def __init__(
        self,
        max_concurrent_tasks: int = 10,
        task_timeout: float = 300.0,
        enable_task_persistence: bool = True,
        task_queue_size: int = 1000,
    ):
        """Initialize distributed task manager.
        
        Args:
            max_concurrent_tasks: Maximum concurrent tasks
            task_timeout: Task timeout in seconds
            enable_task_persistence: Whether to persist tasks to disk
            task_queue_size: Maximum queue size
        """
        self.max_concurrent_tasks = max_concurrent_tasks
        self.task_timeout = task_timeout
        self.enable_task_persistence = enable_task_persistence

        # Task management
        self.task_queue = queue.PriorityQueue(maxsize=task_queue_size)
        self.active_tasks = {}  # task_id -> Future
        self.completed_tasks = {}  # task_id -> TaskResult
        self.failed_tasks = {}  # task_id -> TaskResult

        # Threading
        self.lock = threading.Lock()
        self.running = False
        self.worker_threads = []

        # Statistics
        self.stats = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_execution_time": 0.0,
        }

        # Task persistence
        if enable_task_persistence:
            self.task_storage_dir = Path("distributed_tasks")
            self.task_storage_dir.mkdir(exist_ok=True)

    def start(self):
        """Start the distributed task manager."""
        with self.lock:
            if self.running:
                return

            self.running = True

            # Start worker threads
            for i in range(self.max_concurrent_tasks):
                worker = threading.Thread(
                    target=self._worker_loop,
                    args=(f"worker_{i}",),
                    daemon=True
                )
                worker.start()
                self.worker_threads.append(worker)

    def stop(self):
        """Stop the distributed task manager."""
        with self.lock:
            if not self.running:
                return

            self.running = False

            # Signal workers to stop
            for _ in self.worker_threads:
                try:
                    self.task_queue.put((0, None), timeout=1.0)
                except queue.Full:
                    pass

    def submit_task(
        self,
        task_type: str,
        function_name: str,
        args: tuple = (),
        kwargs: dict = None,
        priority: int = 0,
    ) -> str:
        """Submit a task for distributed execution.
        
        Args:
            task_type: Type of task
            function_name: Name of function to execute
            args: Function arguments
            kwargs: Function keyword arguments
            priority: Task priority (higher = more priority)
            
        Returns:
            Task ID
        """
        kwargs = kwargs or {}

        task_id = f"{task_type}_{int(time.time() * 1000000)}"
        task = ComputeTask(
            task_id=task_id,
            task_type=task_type,
            function_name=function_name,
            args=args,
            kwargs=kwargs,
            priority=priority
        )

        # Persist task if enabled
        if self.enable_task_persistence:
            self._persist_task(task)

        # Add to queue (priority queue uses negative priority for max-heap)
        try:
            self.task_queue.put((-priority, task), timeout=1.0)
            with self.lock:
                self.stats["tasks_submitted"] += 1
            return task_id
        except queue.Full:
            raise RuntimeError("Task queue is full")

    def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> Optional[TaskResult]:
        """Get result of a completed task.
        
        Args:
            task_id: Task ID
            timeout: Timeout for waiting
            
        Returns:
            Task result or None if not ready
        """
        start_time = time.time()

        while True:
            # Check completed tasks
            with self.lock:
                if task_id in self.completed_tasks:
                    return self.completed_tasks[task_id]
                if task_id in self.failed_tasks:
                    return self.failed_tasks[task_id]

            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                return None

            time.sleep(0.1)

    def wait_for_completion(self, task_ids: List[str], timeout: Optional[float] = None) -> Dict[str, TaskResult]:
        """Wait for multiple tasks to complete.
        
        Args:
            task_ids: List of task IDs
            timeout: Timeout for waiting
            
        Returns:
            Dictionary mapping task IDs to results
        """
        results = {}
        start_time = time.time()

        while len(results) < len(task_ids):
            for task_id in task_ids:
                if task_id not in results:
                    result = self.get_task_result(task_id, timeout=0.1)
                    if result:
                        results[task_id] = result

            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                break

            if len(results) < len(task_ids):
                time.sleep(0.1)

        return results

    def _worker_loop(self, worker_id: str):
        """Main worker loop for processing tasks."""
        while self.running:
            try:
                # Get task from queue
                priority, task = self.task_queue.get(timeout=1.0)

                if task is None:  # Stop signal
                    break

                # Execute task
                result = self._execute_task(task, worker_id)

                # Store result
                with self.lock:
                    if result.success:
                        self.completed_tasks[task.task_id] = result
                        self.stats["tasks_completed"] += 1
                    else:
                        self.failed_tasks[task.task_id] = result
                        self.stats["tasks_failed"] += 1

                    self.stats["total_execution_time"] += result.execution_time

                # Persist result if enabled
                if self.enable_task_persistence:
                    self._persist_result(result)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Worker {worker_id} error: {e}")

    def _execute_task(self, task: ComputeTask, worker_id: str) -> TaskResult:
        """Execute a single task.
        
        Args:
            task: Task to execute
            worker_id: ID of executing worker
            
        Returns:
            Task result
        """
        start_time = time.time()

        try:
            # Get function from registry
            func = self._get_function(task.function_name)

            # Execute function
            result = func(*task.args, **task.kwargs)

            execution_time = time.time() - start_time

            return TaskResult(
                task_id=task.task_id,
                success=True,
                result=result,
                execution_time=execution_time,
                worker_id=worker_id
            )

        except Exception as e:
            execution_time = time.time() - start_time

            return TaskResult(
                task_id=task.task_id,
                success=False,
                error=str(e),
                execution_time=execution_time,
                worker_id=worker_id
            )

    def _get_function(self, function_name: str) -> Callable:
        """Get function by name from registry.
        
        Args:
            function_name: Name of function
            
        Returns:
            Function object
        """
        # Function registry for distributed execution
        function_registry = {
            "predict": self._predict_function,
            "gradient": self._gradient_function,
            "uncertainty": self._uncertainty_function,
            "optimize": self._optimize_function,
            "evaluate": self._evaluate_function,
        }

        if function_name not in function_registry:
            raise ValueError(f"Unknown function: {function_name}")

        return function_registry[function_name]

    def _predict_function(self, surrogate, x):
        """Distributed prediction function."""
        return surrogate.predict(x)

    def _gradient_function(self, surrogate, x):
        """Distributed gradient function."""
        return surrogate.gradient(x)

    def _uncertainty_function(self, surrogate, x):
        """Distributed uncertainty function."""
        return surrogate.uncertainty(x)

    def _optimize_function(self, optimizer, surrogate, x0, bounds=None, **kwargs):
        """Distributed optimization function."""
        return optimizer.optimize(surrogate, x0, bounds, **kwargs)

    def _evaluate_function(self, func, x):
        """Distributed function evaluation."""
        return func(x)

    def _persist_task(self, task: ComputeTask):
        """Persist task to disk."""
        if not self.enable_task_persistence:
            return

        task_file = self.task_storage_dir / f"task_{task.task_id}.json"
        with open(task_file, "w") as f:
            json.dump(asdict(task), f, indent=2)

    def _persist_result(self, result: TaskResult):
        """Persist result to disk."""
        if not self.enable_task_persistence:
            return

        result_file = self.task_storage_dir / f"result_{result.task_id}.pkl"
        with open(result_file, "wb") as f:
            pickle.dump(result, f)

    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        with self.lock:
            stats = self.stats.copy()
            stats["active_tasks"] = len(self.active_tasks)
            stats["completed_tasks"] = len(self.completed_tasks)
            stats["failed_tasks"] = len(self.failed_tasks)
            stats["queue_size"] = self.task_queue.qsize()

            if stats["tasks_completed"] > 0:
                stats["average_execution_time"] = stats["total_execution_time"] / stats["tasks_completed"]
            else:
                stats["average_execution_time"] = 0.0

            return stats


class DistributedSurrogateOptimizer:
    """Distributed surrogate optimizer using task-based parallelism."""

    def __init__(
        self,
        base_optimizer,
        task_manager: Optional[DistributedTaskManager] = None,
        enable_distributed_training: bool = True,
        enable_distributed_optimization: bool = True,
    ):
        """Initialize distributed surrogate optimizer.
        
        Args:
            base_optimizer: Base optimizer to distribute
            task_manager: Task manager for distribution
            enable_distributed_training: Enable distributed training
            enable_distributed_optimization: Enable distributed optimization
        """
        self.base_optimizer = base_optimizer
        self.enable_distributed_training = enable_distributed_training
        self.enable_distributed_optimization = enable_distributed_optimization

        if task_manager is None:
            self.task_manager = DistributedTaskManager()
            self.task_manager.start()
        else:
            self.task_manager = task_manager

    def fit_surrogate_distributed(self, datasets: List[Any]) -> List[str]:
        """Fit multiple surrogate models in parallel.
        
        Args:
            datasets: List of datasets for training
            
        Returns:
            List of task IDs
        """
        if not self.enable_distributed_training:
            # Fall back to sequential training
            results = []
            for dataset in datasets:
                self.base_optimizer.fit_surrogate(dataset)
                results.append(None)
            return results

        task_ids = []
        for i, dataset in enumerate(datasets):
            task_id = self.task_manager.submit_task(
                task_type="training",
                function_name="fit_surrogate",
                args=(self.base_optimizer, dataset),
                priority=1
            )
            task_ids.append(task_id)

        return task_ids

    def optimize_distributed(self, surrogate, initial_points: List[Array], bounds=None, **kwargs) -> List[str]:
        """Run optimization from multiple starting points in parallel.
        
        Args:
            surrogate: Trained surrogate model
            initial_points: List of starting points
            bounds: Optimization bounds
            **kwargs: Additional optimization arguments
            
        Returns:
            List of task IDs
        """
        if not self.enable_distributed_optimization:
            # Fall back to sequential optimization
            results = []
            for x0 in initial_points:
                result = self.base_optimizer.optimize(surrogate, x0, bounds, **kwargs)
                results.append(result)
            return results

        task_ids = []
        for i, x0 in enumerate(initial_points):
            task_id = self.task_manager.submit_task(
                task_type="optimization",
                function_name="optimize",
                args=(self.base_optimizer, surrogate, x0, bounds),
                kwargs=kwargs,
                priority=2
            )
            task_ids.append(task_id)

        return task_ids

    def predict_distributed(self, surrogate, x_points: List[Array]) -> List[str]:
        """Distribute prediction across multiple point sets.
        
        Args:
            surrogate: Trained surrogate model
            x_points: List of point arrays for prediction
            
        Returns:
            List of task IDs
        """
        task_ids = []
        for i, x in enumerate(x_points):
            task_id = self.task_manager.submit_task(
                task_type="prediction",
                function_name="predict",
                args=(surrogate, x),
                priority=0
            )
            task_ids.append(task_id)

        return task_ids

    def collect_results(self, task_ids: List[str], timeout: Optional[float] = None) -> Dict[str, Any]:
        """Collect results from distributed tasks.
        
        Args:
            task_ids: List of task IDs to collect
            timeout: Timeout for collection
            
        Returns:
            Dictionary of results
        """
        return self.task_manager.wait_for_completion(task_ids, timeout)

    def get_best_optimization_result(self, task_ids: List[str], timeout: Optional[float] = None):
        """Get best optimization result from parallel runs.
        
        Args:
            task_ids: List of optimization task IDs
            timeout: Timeout for collection
            
        Returns:
            Best optimization result
        """
        results = self.collect_results(task_ids, timeout)

        best_result = None
        best_value = float("inf")

        for task_id, task_result in results.items():
            if task_result.success and hasattr(task_result.result, "fun"):
                if task_result.result.fun < best_value:
                    best_value = task_result.result.fun
                    best_result = task_result.result

        return best_result


class DataDistributor:
    """Manages distribution of large datasets across workers."""

    def __init__(self, chunk_size: int = 1000):
        """Initialize data distributor.
        
        Args:
            chunk_size: Size of data chunks
        """
        self.chunk_size = chunk_size

    def distribute_dataset(self, dataset, n_partitions: int) -> List[Any]:
        """Distribute dataset across partitions.
        
        Args:
            dataset: Dataset to distribute
            n_partitions: Number of partitions
            
        Returns:
            List of dataset partitions
        """
        n_samples = dataset.n_samples
        partition_size = n_samples // n_partitions

        partitions = []
        for i in range(n_partitions):
            start_idx = i * partition_size
            if i == n_partitions - 1:  # Last partition gets remaining samples
                end_idx = n_samples
            else:
                end_idx = start_idx + partition_size

            # Create partition
            from .models.base import Dataset
            partition = Dataset(
                X=dataset.X[start_idx:end_idx],
                y=dataset.y[start_idx:end_idx],
                gradients=dataset.gradients[start_idx:end_idx] if dataset.gradients is not None else None,
                metadata=dataset.metadata.copy()
            )
            partitions.append(partition)

        return partitions

    def merge_predictions(self, prediction_chunks: List[Array]) -> Array:
        """Merge prediction results from multiple chunks.
        
        Args:
            prediction_chunks: List of prediction arrays
            
        Returns:
            Merged predictions
        """
        return jnp.concatenate(prediction_chunks, axis=0)

    def distribute_points(self, points: Array, n_chunks: int) -> List[Array]:
        """Distribute points across chunks for parallel processing.
        
        Args:
            points: Points to distribute
            n_chunks: Number of chunks
            
        Returns:
            List of point chunks
        """
        n_points = points.shape[0]
        chunk_size = n_points // n_chunks

        chunks = []
        for i in range(n_chunks):
            start_idx = i * chunk_size
            if i == n_chunks - 1:  # Last chunk gets remaining points
                end_idx = n_points
            else:
                end_idx = start_idx + chunk_size

            chunks.append(points[start_idx:end_idx])

        return chunks


# Global distributed task manager
_global_task_manager = None

def get_global_task_manager() -> DistributedTaskManager:
    """Get global distributed task manager."""
    global _global_task_manager
    if _global_task_manager is None:
        _global_task_manager = DistributedTaskManager()
        _global_task_manager.start()
    return _global_task_manager


def distributed_optimization(
    optimizer,
    surrogate,
    initial_points: List[Array],
    bounds=None,
    **kwargs
):
    """Convenience function for distributed optimization.
    
    Args:
        optimizer: Optimizer to use
        surrogate: Surrogate model
        initial_points: List of starting points
        bounds: Optimization bounds
        **kwargs: Additional arguments
        
    Returns:
        Best optimization result
    """
    dist_optimizer = DistributedSurrogateOptimizer(optimizer)
    task_ids = dist_optimizer.optimize_distributed(surrogate, initial_points, bounds, **kwargs)
    return dist_optimizer.get_best_optimization_result(task_ids)
