"""GPU acceleration utilities for surrogate optimization using JAX."""

from dataclasses import dataclass
import time
from typing import Any, Dict, List, Optional, Tuple
import warnings

import jax
from jax import Array, device_put, jit, pmap, vmap
from jax.lib import xla_bridge
import jax.numpy as jnp


@dataclass
class GPUStatus:
    """GPU status and capabilities information."""
    has_gpu: bool
    gpu_count: int
    gpu_memory_gb: float
    gpu_compute_capability: str
    driver_version: str
    backend: str
    device_names: List[str]


class GPUManager:
    """Manages GPU resources and optimization strategies."""

    def __init__(self, prefer_gpu: bool = True):
        """Initialize GPU manager.
        
        Args:
            prefer_gpu: Whether to prefer GPU over CPU when available
        """
        self.prefer_gpu = prefer_gpu
        self._status = None
        self._optimal_batch_size = None
        self._memory_limit = None

        # Initialize GPU status
        self._detect_gpu_capabilities()

    def _detect_gpu_capabilities(self) -> GPUStatus:
        """Detect available GPU capabilities."""
        try:
            # Check for GPU devices
            gpu_devices = jax.devices("gpu")
            has_gpu = len(gpu_devices) > 0

            if has_gpu:
                # Get GPU information
                backend = xla_bridge.get_backend().platform
                device_names = [str(device) for device in gpu_devices]

                # Estimate memory (this is approximate)
                # In practice, you'd query actual device memory
                estimated_memory = 8.0  # GB - default estimate

                self._status = GPUStatus(
                    has_gpu=True,
                    gpu_count=len(gpu_devices),
                    gpu_memory_gb=estimated_memory,
                    gpu_compute_capability="unknown",  # Would need CUDA query
                    driver_version="unknown",
                    backend=backend,
                    device_names=device_names
                )
            else:
                self._status = GPUStatus(
                    has_gpu=False,
                    gpu_count=0,
                    gpu_memory_gb=0.0,
                    gpu_compute_capability="",
                    driver_version="",
                    backend=xla_bridge.get_backend().platform,
                    device_names=[]
                )

        except Exception as e:
            warnings.warn(f"Failed to detect GPU capabilities: {e}")
            self._status = GPUStatus(False, 0, 0.0, "", "", "cpu", [])

        return self._status

    def get_status(self) -> GPUStatus:
        """Get current GPU status."""
        return self._status

    def get_optimal_device(self) -> jax.Device:
        """Get the optimal device for computation."""
        if self._status.has_gpu and self.prefer_gpu:
            return jax.devices("gpu")[0]
        return jax.devices("cpu")[0]

    def estimate_optimal_batch_size(self, input_dim: int, model_complexity: str = "medium") -> int:
        """Estimate optimal batch size for GPU computation.
        
        Args:
            input_dim: Input dimension of the problem
            model_complexity: Complexity of the model ("simple", "medium", "complex")
            
        Returns:
            Optimal batch size for current hardware
        """
        if not self._status.has_gpu:
            return 32  # Conservative batch size for CPU

        # Base batch size estimation
        memory_gb = self._status.gpu_memory_gb

        # Complexity multipliers
        complexity_factors = {
            "simple": 1.0,
            "medium": 0.7,
            "complex": 0.4
        }

        factor = complexity_factors.get(model_complexity, 0.7)

        # Estimate based on available memory and input dimension
        base_batch = int(memory_gb * 1000 * factor / max(1, input_dim))

        # Clamp to reasonable bounds
        optimal_batch = max(64, min(2048, base_batch))

        self._optimal_batch_size = optimal_batch
        return optimal_batch

    def configure_memory_growth(self, enable: bool = True):
        """Configure GPU memory growth to avoid pre-allocation.
        
        Args:
            enable: Whether to enable memory growth
        """
        if not self._status.has_gpu:
            return

        # JAX doesn't have direct memory growth control like TensorFlow
        # But we can set memory fraction
        if enable:
            # Set conservative memory usage
            import os
            os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.8")
            os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    def clear_cache(self):
        """Clear JAX compilation cache to free memory."""
        # Clear JAX cache
        from jax.experimental.compilation_cache import clear_cache
        try:
            clear_cache()
        except:
            pass  # Cache clearing might not be available in all JAX versions


class GPUOptimizedSurrogate:
    """GPU-optimized surrogate model with efficient memory management."""

    def __init__(
        self,
        base_surrogate,
        gpu_manager: Optional[GPUManager] = None,
        batch_size: Optional[int] = None,
        mixed_precision: bool = True,
        memory_efficient: bool = True,
    ):
        """Initialize GPU-optimized surrogate.
        
        Args:
            base_surrogate: Base surrogate model
            gpu_manager: GPU manager instance
            batch_size: Batch size for operations
            mixed_precision: Use mixed precision computation
            memory_efficient: Enable memory-efficient operations
        """
        self.base_surrogate = base_surrogate
        self.gpu_manager = gpu_manager or GPUManager()
        self.mixed_precision = mixed_precision
        self.memory_efficient = memory_efficient

        # Set optimal device
        self.device = self.gpu_manager.get_optimal_device()

        # Configure batch size
        if batch_size is None and hasattr(base_surrogate, "input_dim"):
            self.batch_size = self.gpu_manager.estimate_optimal_batch_size(
                base_surrogate.input_dim
            )
        else:
            self.batch_size = batch_size or 256

        # Compiled functions
        self._gpu_predict = None
        self._gpu_gradient = None
        self._gpu_uncertainty = None

        # Performance tracking
        self.gpu_utilization_history = []
        self.memory_usage_history = []

        # State
        self.is_fitted = False

    def fit(self, dataset) -> "GPUOptimizedSurrogate":
        """Fit surrogate with GPU optimizations."""
        # Move data to GPU
        gpu_dataset = self._move_dataset_to_device(dataset)

        # Fit base model
        self.base_surrogate.fit(gpu_dataset)
        self.is_fitted = True

        # Compile GPU functions
        self._compile_gpu_functions(gpu_dataset)

        return self

    def _move_dataset_to_device(self, dataset):
        """Move dataset to optimal device."""
        # Create device-optimized dataset
        gpu_X = device_put(dataset.X, device=self.device)
        gpu_y = device_put(dataset.y, device=self.device)

        gpu_gradients = None
        if dataset.gradients is not None:
            gpu_gradients = device_put(dataset.gradients, device=self.device)

        # Create new dataset with GPU arrays
        from ..models.base import Dataset
        return Dataset(
            X=gpu_X,
            y=gpu_y,
            gradients=gpu_gradients,
            metadata=dataset.metadata
        )

    def _compile_gpu_functions(self, dataset):
        """Compile functions optimized for GPU execution."""
        sample_input = dataset.X[0] if dataset.X.ndim > 1 else dataset.X

        # Prediction function
        @jit
        def gpu_predict_single(x):
            x_gpu = device_put(x, device=self.device)
            if self.mixed_precision:
                x_gpu = x_gpu.astype(jnp.float32)
            result = self.base_surrogate.predict(x_gpu)
            return result

        @jit
        def gpu_predict_batch(x_batch):
            x_gpu = device_put(x_batch, device=self.device)
            if self.mixed_precision:
                x_gpu = x_gpu.astype(jnp.float32)
            return vmap(self.base_surrogate.predict)(x_gpu)

        self._gpu_predict = gpu_predict_batch

        # Gradient function
        @jit
        def gpu_gradient_batch(x_batch):
            x_gpu = device_put(x_batch, device=self.device)
            if self.mixed_precision:
                x_gpu = x_gpu.astype(jnp.float32)
            return vmap(self.base_surrogate.gradient)(x_gpu)

        self._gpu_gradient = gpu_gradient_batch

        # Uncertainty function
        if hasattr(self.base_surrogate, "uncertainty"):
            @jit
            def gpu_uncertainty_batch(x_batch):
                x_gpu = device_put(x_batch, device=self.device)
                if self.mixed_precision:
                    x_gpu = x_gpu.astype(jnp.float32)
                return vmap(self.base_surrogate.uncertainty)(x_gpu)

            self._gpu_uncertainty = gpu_uncertainty_batch

        # Trigger compilation with sample input
        if dataset.X.ndim > 1:
            sample_batch = dataset.X[:min(self.batch_size, len(dataset.X))]
            _ = self._gpu_predict(sample_batch)
            _ = self._gpu_gradient(sample_batch)
            if self._gpu_uncertainty:
                _ = self._gpu_uncertainty(sample_batch)

    def predict(self, x: Array) -> Array:
        """GPU-optimized prediction."""
        if not self.is_fitted:
            return self.base_surrogate.predict(x)

        start_time = time.time()

        # Handle different input shapes
        if x.ndim == 1:
            x_batch = x.reshape(1, -1)
            result = self._predict_gpu_batch(x_batch)
            result = result[0]  # Extract single result
        else:
            result = self._predict_gpu_batch(x)

        # Track performance
        execution_time = time.time() - start_time
        self._track_performance(execution_time, x.shape[0] if x.ndim > 1 else 1)

        return result

    def _predict_gpu_batch(self, x_batch: Array) -> Array:
        """Efficient GPU batch prediction with memory management."""
        n_points = x_batch.shape[0]

        # Process in chunks if too large
        if self.memory_efficient and n_points > self.batch_size:
            results = []
            for i in range(0, n_points, self.batch_size):
                chunk = x_batch[i:i+self.batch_size]
                chunk_result = self._gpu_predict(chunk)
                results.append(chunk_result)
            return jnp.concatenate(results)
        return self._gpu_predict(x_batch)

    def gradient(self, x: Array) -> Array:
        """GPU-optimized gradient computation."""
        if not self.is_fitted:
            return self.base_surrogate.gradient(x)

        start_time = time.time()

        if x.ndim == 1:
            x_batch = x.reshape(1, -1)
            result = self._gradient_gpu_batch(x_batch)
            result = result[0]
        else:
            result = self._gradient_gpu_batch(x)

        execution_time = time.time() - start_time
        self._track_performance(execution_time, x.shape[0] if x.ndim > 1 else 1)

        return result

    def _gradient_gpu_batch(self, x_batch: Array) -> Array:
        """Efficient GPU batch gradient computation."""
        n_points = x_batch.shape[0]

        if self.memory_efficient and n_points > self.batch_size:
            results = []
            for i in range(0, n_points, self.batch_size):
                chunk = x_batch[i:i+self.batch_size]
                chunk_result = self._gpu_gradient(chunk)
                results.append(chunk_result)
            return jnp.concatenate(results)
        return self._gpu_gradient(x_batch)

    def uncertainty(self, x: Array) -> Array:
        """GPU-optimized uncertainty estimation."""
        if not self.is_fitted or self._gpu_uncertainty is None:
            return self.base_surrogate.uncertainty(x)

        if x.ndim == 1:
            x_batch = x.reshape(1, -1)
            result = self._uncertainty_gpu_batch(x_batch)
            result = result[0]
        else:
            result = self._uncertainty_gpu_batch(x)

        return result

    def _uncertainty_gpu_batch(self, x_batch: Array) -> Array:
        """Efficient GPU batch uncertainty computation."""
        n_points = x_batch.shape[0]

        if self.memory_efficient and n_points > self.batch_size:
            results = []
            for i in range(0, n_points, self.batch_size):
                chunk = x_batch[i:i+self.batch_size]
                chunk_result = self._gpu_uncertainty(chunk)
                results.append(chunk_result)
            return jnp.concatenate(results)
        return self._gpu_uncertainty(x_batch)

    def _track_performance(self, execution_time: float, n_points: int):
        """Track performance metrics."""
        throughput = n_points / execution_time if execution_time > 0 else 0

        self.gpu_utilization_history.append({
            "timestamp": time.time(),
            "execution_time": execution_time,
            "throughput": throughput,
            "batch_size": n_points,
        })

        # Keep only recent history
        if len(self.gpu_utilization_history) > 1000:
            self.gpu_utilization_history = self.gpu_utilization_history[-500:]

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        if not self.gpu_utilization_history:
            return {"status": "no_data"}

        recent_history = self.gpu_utilization_history[-10:]
        avg_throughput = sum(h["throughput"] for h in recent_history) / len(recent_history)
        avg_execution_time = sum(h["execution_time"] for h in recent_history) / len(recent_history)

        return {
            "gpu_status": self.gpu_manager.get_status(),
            "device": str(self.device),
            "batch_size": self.batch_size,
            "mixed_precision": self.mixed_precision,
            "avg_throughput": avg_throughput,
            "avg_execution_time": avg_execution_time,
            "total_operations": len(self.gpu_utilization_history),
            "memory_efficient": self.memory_efficient,
        }

    def benchmark_performance(self, test_points: Array, n_trials: int = 5) -> Dict[str, float]:
        """Benchmark GPU performance vs CPU baseline.
        
        Args:
            test_points: Test points for benchmarking
            n_trials: Number of benchmark trials
            
        Returns:
            Performance comparison metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before benchmarking")

        # GPU benchmark
        gpu_times = []
        for _ in range(n_trials):
            start = time.time()
            _ = self.predict(test_points)
            gpu_times.append(time.time() - start)

        # CPU benchmark (if possible)
        cpu_times = []
        try:
            # Move model to CPU temporarily
            cpu_device = jax.devices("cpu")[0]
            test_points_cpu = device_put(test_points, device=cpu_device)

            for _ in range(n_trials):
                start = time.time()
                _ = self.base_surrogate.predict(test_points_cpu)
                cpu_times.append(time.time() - start)
        except:
            cpu_times = [float("inf")]  # CPU benchmark failed

        avg_gpu_time = sum(gpu_times) / len(gpu_times)
        avg_cpu_time = sum(cpu_times) / len(cpu_times)
        speedup = avg_cpu_time / avg_gpu_time if avg_gpu_time > 0 else 1.0

        return {
            "gpu_avg_time": avg_gpu_time,
            "cpu_avg_time": avg_cpu_time,
            "speedup": speedup,
            "n_points": len(test_points),
            "points_per_second_gpu": len(test_points) / avg_gpu_time if avg_gpu_time > 0 else 0,
            "points_per_second_cpu": len(test_points) / avg_cpu_time if avg_cpu_time < float("inf") else 0,
        }


class MultiGPUOptimizer:
    """Multi-GPU optimizer for large-scale surrogate optimization."""

    def __init__(
        self,
        base_optimizer,
        strategy: str = "data_parallel",  # "data_parallel", "model_parallel"
        synchronize_updates: bool = True,
    ):
        """Initialize multi-GPU optimizer.
        
        Args:
            base_optimizer: Base optimization algorithm
            strategy: Parallelization strategy
            synchronize_updates: Whether to synchronize updates across GPUs
        """
        self.base_optimizer = base_optimizer
        self.strategy = strategy
        self.synchronize_updates = synchronize_updates

        # Detect available GPUs
        self.gpu_devices = jax.devices("gpu")
        self.n_gpus = len(self.gpu_devices)

        if self.n_gpus == 0:
            warnings.warn("No GPUs detected. Multi-GPU optimizer will use CPU.")
            self.gpu_devices = jax.devices("cpu")
            self.n_gpus = 1

        # Compiled functions
        self._parallel_predict = None
        self._parallel_optimize = None

    def optimize(
        self,
        surrogate,
        x0_list: List[Array],
        bounds: Optional[List[Tuple[float, float]]] = None,
        **kwargs
    ):
        """Multi-GPU optimization from multiple starting points.
        
        Args:
            surrogate: Surrogate model
            x0_list: List of starting points
            bounds: Optimization bounds
            **kwargs: Additional optimization arguments
            
        Returns:
            Best optimization result from all GPUs
        """
        if self.strategy == "data_parallel":
            return self._optimize_data_parallel(surrogate, x0_list, bounds, **kwargs)
        raise NotImplementedError(f"Strategy {self.strategy} not implemented")

    def _optimize_data_parallel(self, surrogate, x0_list, bounds, **kwargs):
        """Data-parallel optimization across multiple GPUs."""
        # Distribute starting points across GPUs
        n_points_per_gpu = max(1, len(x0_list) // self.n_gpus)
        gpu_x0_batches = []

        for i in range(self.n_gpus):
            start_idx = i * n_points_per_gpu
            end_idx = min((i + 1) * n_points_per_gpu, len(x0_list))

            if start_idx < len(x0_list):
                batch = x0_list[start_idx:end_idx]
                gpu_x0_batches.append(jnp.stack(batch))
            else:
                # Pad with duplicate points if needed
                gpu_x0_batches.append(jnp.stack([x0_list[-1]]))

        # Ensure we have the right number of batches
        while len(gpu_x0_batches) < self.n_gpus:
            gpu_x0_batches.append(gpu_x0_batches[-1])

        # Define single optimization step
        def single_gpu_optimize(x0_batch, device):
            # Move data to specific device
            x0_device = device_put(x0_batch, device=device)

            # Run optimization on each point in the batch
            results = []
            for x0 in x0_device:
                result = self.base_optimizer.optimize(
                    surrogate=surrogate,
                    x0=x0,
                    bounds=bounds,
                    **kwargs
                )
                results.append(result)

            return results

        # Use pmap for parallel execution across devices
        if self.n_gpus > 1:
            try:
                # Create parallel version
                parallel_optimize = pmap(
                    lambda x0_batch: single_gpu_optimize(x0_batch, jax.devices()[0])  # Device will be set by pmap
                )

                # Stack batches for pmap
                x0_array = jnp.stack(gpu_x0_batches)

                # Run parallel optimization
                parallel_results = parallel_optimize(x0_array)

                # Flatten results
                all_results = []
                for gpu_results in parallel_results:
                    all_results.extend(gpu_results)

            except Exception as e:
                warnings.warn(f"Parallel optimization failed: {e}. Falling back to sequential.")
                # Fallback to sequential execution
                all_results = []
                for i, batch in enumerate(gpu_x0_batches):
                    device = self.gpu_devices[i % len(self.gpu_devices)]
                    batch_results = single_gpu_optimize(batch, device)
                    all_results.extend(batch_results)
        else:
            # Single GPU case
            all_results = []
            for batch in gpu_x0_batches:
                batch_results = single_gpu_optimize(batch, self.gpu_devices[0])
                all_results.extend(batch_results)

        # Find best result
        best_result = min(
            all_results,
            key=lambda r: r.fun if hasattr(r, "success") and r.success else float("inf")
        )

        return best_result

    def get_multi_gpu_stats(self) -> Dict[str, Any]:
        """Get multi-GPU utilization statistics."""
        return {
            "n_gpus": self.n_gpus,
            "gpu_devices": [str(device) for device in self.gpu_devices],
            "strategy": self.strategy,
            "synchronize_updates": self.synchronize_updates,
        }


def enable_gpu_optimizations(
    surrogate_model,
    auto_configure: bool = True,
    mixed_precision: bool = True,
    memory_growth: bool = True,
) -> GPUOptimizedSurrogate:
    """Enable GPU optimizations for a surrogate model.
    
    Args:
        surrogate_model: Base surrogate model
        auto_configure: Automatically configure optimal settings
        mixed_precision: Use mixed precision computation
        memory_growth: Enable memory growth instead of pre-allocation
        
    Returns:
        GPU-optimized surrogate model
    """
    gpu_manager = GPUManager()

    # Configure memory growth
    if memory_growth:
        gpu_manager.configure_memory_growth(True)

    # Create GPU-optimized surrogate
    gpu_surrogate = GPUOptimizedSurrogate(
        base_surrogate=surrogate_model,
        gpu_manager=gpu_manager,
        mixed_precision=mixed_precision,
        memory_efficient=True,
    )

    if auto_configure and gpu_manager.get_status().has_gpu:
        print(f"GPU acceleration enabled: {gpu_manager.get_status().gpu_count} GPU(s) detected")
        print(f"Estimated optimal batch size: {gpu_surrogate.batch_size}")
        print(f"Mixed precision: {mixed_precision}")
    elif gpu_manager.get_status().has_gpu:
        print("GPU available but not auto-configured")
    else:
        print("No GPU detected - falling back to CPU optimization")

    return gpu_surrogate


class GPUAccelerator:
    """High-level GPU acceleration interface for surrogate optimization."""

    def __init__(self):
        """Initialize GPU accelerator."""
        self.gpu_manager = GPUManager()
        self.status = self.gpu_manager.get_status()

    def get_training_params(self) -> Dict[str, Any]:
        """Get optimized training parameters for current hardware."""
        if self.status.has_gpu:
            return {
                "batch_size": self.gpu_manager.get_optimal_batch_size(2, "medium"),
                "n_epochs": 200  # Fewer epochs for faster training
            }
        return {
            "batch_size": 32,
            "n_epochs": 200  # Standard training
        }

    def is_available(self) -> bool:
        """Check if GPU acceleration is available."""
        return self.status.has_gpu
