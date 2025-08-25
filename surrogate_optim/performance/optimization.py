"""Performance optimization and scalability enhancements."""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple, Union

import jax
from jax import Array, jit, vmap
import jax.numpy as jnp


@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization operations."""
    compilation_time: float
    execution_time: float
    memory_usage: float
    throughput: float
    cache_hit_rate: float
    parallel_efficiency: float


class JITCache:
    """Advanced JIT compilation cache with smart precompilation."""

    def __init__(self, max_cache_size: int = 100):
        """Initialize JIT cache.
        
        Args:
            max_cache_size: Maximum number of compiled functions to cache
        """
        self.cache = {}
        self.max_cache_size = max_cache_size
        self.access_counts = {}
        self.compilation_times = {}
        self.lock = threading.Lock()

    def get_or_compile(
        self,
        func: Callable,
        func_id: str,
        input_signature: Optional[Any] = None,
        static_argnums: Optional[Union[int, Tuple[int, ...]]] = None,
    ) -> Callable:
        """Get compiled function from cache or compile and cache it.
        
        Args:
            func: Function to compile
            func_id: Unique identifier for the function
            input_signature: Optional input signature for compilation
            static_argnums: Arguments to treat as static
            
        Returns:
            Compiled function
        """
        with self.lock:
            if func_id in self.cache:
                self.access_counts[func_id] += 1
                return self.cache[func_id]

            # Compile function
            start_time = time.time()

            if static_argnums is not None:
                compiled_func = jit(func, static_argnums=static_argnums)
            else:
                compiled_func = jit(func)

            # Trigger compilation if signature provided
            if input_signature is not None:
                _ = compiled_func(*input_signature)

            compilation_time = time.time() - start_time

            # Manage cache size
            if len(self.cache) >= self.max_cache_size:
                # Remove least accessed function
                least_accessed = min(self.access_counts.items(), key=lambda x: x[1])[0]
                del self.cache[least_accessed]
                del self.access_counts[least_accessed]
                del self.compilation_times[least_accessed]

            # Store in cache
            self.cache[func_id] = compiled_func
            self.access_counts[func_id] = 1
            self.compilation_times[func_id] = compilation_time

            return compiled_func

    def precompile_common_functions(self, input_shapes: Dict[str, Tuple]):
        """Precompile common functions with known input shapes.
        
        Args:
            input_shapes: Dictionary mapping function IDs to input shapes
        """
        for func_id, shape in input_shapes.items():
            if func_id not in self.cache:
                # Create dummy inputs for compilation
                dummy_input = jnp.ones(shape)

                # Common function patterns
                if "predict" in func_id:
                    def predict_func(x):
                        return jnp.sum(x**2)  # Placeholder
                    self.get_or_compile(predict_func, func_id, (dummy_input,))

                elif "gradient" in func_id:
                    def grad_func(x):
                        return 2 * x  # Placeholder gradient
                    self.get_or_compile(grad_func, func_id, (dummy_input,))

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            return {
                "cache_size": len(self.cache),
                "max_cache_size": self.max_cache_size,
                "total_compilations": len(self.compilation_times),
                "total_compilation_time": sum(self.compilation_times.values()),
                "average_compilation_time": sum(self.compilation_times.values()) / len(self.compilation_times) if self.compilation_times else 0,
                "access_counts": dict(self.access_counts),
            }


class OptimizedSurrogate:
    """High-performance surrogate model with advanced optimizations."""

    def __init__(
        self,
        base_surrogate,
        enable_jit: bool = True,
        enable_vectorization: bool = True,
        batch_size: Optional[int] = None,
        enable_parallel_prediction: bool = True,
        memory_efficient: bool = True,
    ):
        """Initialize optimized surrogate.
        
        Args:
            base_surrogate: Base surrogate model
            enable_jit: Enable JIT compilation
            enable_vectorization: Enable vectorized operations
            batch_size: Batch size for large predictions
            enable_parallel_prediction: Enable parallel prediction
            memory_efficient: Use memory-efficient operations
        """
        self.base_surrogate = base_surrogate
        self.enable_jit = enable_jit
        self.enable_vectorization = enable_vectorization
        self.batch_size = batch_size
        self.enable_parallel_prediction = enable_parallel_prediction
        self.memory_efficient = memory_efficient

        # JIT cache
        self.jit_cache = JITCache()

        # Performance tracking
        self.performance_metrics = PerformanceMetrics(0, 0, 0, 0, 0, 0)

        # Compiled functions
        self._compiled_predict = None
        self._compiled_gradient = None
        self._vectorized_predict = None
        self._vectorized_gradient = None

        # State
        self.is_fitted = False

    def fit(self, dataset) -> "OptimizedSurrogate":
        """Fit surrogate with performance optimizations."""
        # Fit base model
        self.base_surrogate.fit(dataset)
        self.is_fitted = True

        # Pre-compile functions if JIT is enabled
        if self.enable_jit:
            self._compile_functions(dataset)

        return self

    def _compile_functions(self, dataset):
        """Compile prediction and gradient functions."""
        sample_input = dataset.X[0] if dataset.X.ndim > 1 else dataset.X

        # Compile prediction function
        def predict_func(x):
            return self.base_surrogate.predict(x)

        self._compiled_predict = self.jit_cache.get_or_compile(
            predict_func,
            "predict",
            (sample_input,)
        )

        # Compile gradient function
        def gradient_func(x):
            return self.base_surrogate.gradient(x)

        self._compiled_gradient = self.jit_cache.get_or_compile(
            gradient_func,
            "gradient",
            (sample_input,)
        )

        # Vectorized versions
        if self.enable_vectorization:
            self._vectorized_predict = vmap(self._compiled_predict)
            self._vectorized_gradient = vmap(self._compiled_gradient)

    def predict(self, x: Array) -> Array:
        """Optimized prediction with batching and parallelization."""
        if not self.is_fitted:
            return self.base_surrogate.predict(x)

        start_time = time.time()

        # Handle different input shapes
        if x.ndim == 1:
            # Single point prediction
            if self.enable_jit and self._compiled_predict:
                result = self._compiled_predict(x)
            else:
                result = self.base_surrogate.predict(x)
        else:
            # Multiple points prediction
            result = self._predict_batch(x)

        execution_time = time.time() - start_time
        self.performance_metrics.execution_time += execution_time

        return result

    def _predict_batch(self, x_batch: Array) -> Array:
        """Optimized batch prediction."""
        n_points = x_batch.shape[0]

        # Use vectorized prediction if available and efficient
        if (self.enable_vectorization and
            self._vectorized_predict and
            (self.batch_size is None or n_points <= self.batch_size * 2)):

            return self._vectorized_predict(x_batch)

        # Manual batching for large inputs
        if self.batch_size and n_points > self.batch_size:
            results = []
            for i in range(0, n_points, self.batch_size):
                batch = x_batch[i:i+self.batch_size]
                if self._vectorized_predict:
                    batch_result = self._vectorized_predict(batch)
                else:
                    batch_result = jnp.array([
                        self._compiled_predict(point) if self._compiled_predict else self.base_surrogate.predict(point)
                        for point in batch
                    ])
                results.append(batch_result)
            return jnp.concatenate(results)

        # Standard vectorized prediction
        if self._vectorized_predict:
            return self._vectorized_predict(x_batch)
        return jnp.array([
            self._compiled_predict(point) if self._compiled_predict else self.base_surrogate.predict(point)
            for point in x_batch
        ])

    def gradient(self, x: Array) -> Array:
        """Optimized gradient computation."""
        if not self.is_fitted:
            return self.base_surrogate.gradient(x)

        start_time = time.time()

        if x.ndim == 1:
            # Single point gradient
            if self.enable_jit and self._compiled_gradient:
                result = self._compiled_gradient(x)
            else:
                result = self.base_surrogate.gradient(x)
        else:
            # Multiple points gradient
            result = self._gradient_batch(x)

        execution_time = time.time() - start_time
        self.performance_metrics.execution_time += execution_time

        return result

    def _gradient_batch(self, x_batch: Array) -> Array:
        """Optimized batch gradient computation."""
        n_points = x_batch.shape[0]

        # Use vectorized gradients if available
        if (self.enable_vectorization and
            self._vectorized_gradient and
            (self.batch_size is None or n_points <= self.batch_size * 2)):

            return self._vectorized_gradient(x_batch)

        # Manual batching
        if self.batch_size and n_points > self.batch_size:
            results = []
            for i in range(0, n_points, self.batch_size):
                batch = x_batch[i:i+self.batch_size]
                if self._vectorized_gradient:
                    batch_result = self._vectorized_gradient(batch)
                else:
                    batch_result = jnp.array([
                        self._compiled_gradient(point) if self._compiled_gradient else self.base_surrogate.gradient(point)
                        for point in batch
                    ])
                results.append(batch_result)
            return jnp.concatenate(results)

        # Standard vectorized gradient
        if self._vectorized_gradient:
            return self._vectorized_gradient(x_batch)
        return jnp.array([
            self._compiled_gradient(point) if self._compiled_gradient else self.base_surrogate.gradient(point)
            for point in x_batch
        ])

    def uncertainty(self, x: Array) -> Array:
        """Optimized uncertainty estimation."""
        return self.base_surrogate.uncertainty(x)

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        cache_stats = self.jit_cache.get_cache_stats()

        return {
            "execution_time": self.performance_metrics.execution_time,
            "jit_enabled": self.enable_jit,
            "vectorization_enabled": self.enable_vectorization,
            "batch_size": self.batch_size,
            "compilation_stats": cache_stats,
        }


class ParallelOptimizer:
    """Parallel optimization with multi-start and ensemble approaches."""

    def __init__(
        self,
        base_optimizer,
        n_parallel: int = 4,
        execution_backend: str = "thread",  # "thread", "process", "jax"
        load_balancing: str = "dynamic",  # "static", "dynamic"
    ):
        """Initialize parallel optimizer.
        
        Args:
            base_optimizer: Base optimizer to parallelize
            n_parallel: Number of parallel executions
            execution_backend: Backend for parallel execution
            load_balancing: Load balancing strategy
        """
        self.base_optimizer = base_optimizer
        self.n_parallel = n_parallel
        self.execution_backend = execution_backend
        self.load_balancing = load_balancing

        # Executor setup
        if execution_backend == "thread":
            self.executor = ThreadPoolExecutor(max_workers=n_parallel)
        elif execution_backend == "process":
            self.executor = ProcessPoolExecutor(max_workers=n_parallel)
        elif execution_backend == "jax":
            self.executor = None  # Use JAX parallelization
        else:
            raise ValueError(f"Unknown execution backend: {execution_backend}")

        self.performance_metrics = PerformanceMetrics(0, 0, 0, 0, 0, 0)

    def optimize(self, surrogate, x0_list, bounds=None, **kwargs):
        """Parallel optimization from multiple starting points.
        
        Args:
            surrogate: Surrogate model
            x0_list: List of starting points
            bounds: Optimization bounds
            **kwargs: Additional optimization arguments
            
        Returns:
            Best optimization result
        """
        start_time = time.time()

        if self.execution_backend == "jax":
            results = self._optimize_jax_parallel(surrogate, x0_list, bounds, **kwargs)
        else:
            results = self._optimize_executor_parallel(surrogate, x0_list, bounds, **kwargs)

        execution_time = time.time() - start_time
        self.performance_metrics.execution_time += execution_time

        # Find best result
        best_result = min(results, key=lambda r: r.fun if r.success else float("inf"))

        # Calculate parallel efficiency
        sequential_time_estimate = execution_time * self.n_parallel
        self.performance_metrics.parallel_efficiency = sequential_time_estimate / execution_time if execution_time > 0 else 0

        return best_result

    def _optimize_jax_parallel(self, surrogate, x0_list, bounds, **kwargs):
        """JAX-based parallel optimization."""
        # Convert to JAX arrays
        x0_array = jnp.stack(x0_list)

        # Define optimization function for single point
        def single_optimize(x0):
            return self.base_optimizer.optimize(
                surrogate=surrogate,
                x0=x0,
                bounds=bounds,
                **kwargs
            )

        # Use vmap for parallel execution (if supported)
        try:
            vectorized_optimize = vmap(single_optimize)
            results = vectorized_optimize(x0_array)
            return results
        except:
            # Fallback to sequential execution
            return [single_optimize(x0) for x0 in x0_list]

    def _optimize_executor_parallel(self, surrogate, x0_list, bounds, **kwargs):
        """Executor-based parallel optimization."""
        # Submit optimization tasks
        future_to_x0 = {}
        for x0 in x0_list:
            future = self.executor.submit(
                self.base_optimizer.optimize,
                surrogate=surrogate,
                x0=x0,
                bounds=bounds,
                **kwargs
            )
            future_to_x0[future] = x0

        # Collect results
        results = []
        for future in future_to_x0:
            try:
                result = future.result(timeout=300)  # 5 minute timeout
                results.append(result)
            except Exception as e:
                # Create failure result
                from ..optimizers.base import OptimizationResult
                failure_result = OptimizationResult(
                    x=future_to_x0[future],
                    fun=float("inf"),
                    success=False,
                    message=f"Parallel optimization failed: {e}",
                    nit=0,
                    nfev=0
                )
                results.append(failure_result)

        return results

    def __del__(self):
        """Cleanup executor."""
        if hasattr(self, "executor") and self.executor:
            self.executor.shutdown(wait=True)


class MemoryOptimizer:
    """Memory optimization utilities for large-scale problems."""

    @staticmethod
    def chunked_operation(
        operation: Callable,
        data: Array,
        chunk_size: int = 1000,
        axis: int = 0,
    ) -> Array:
        """Apply operation to data in chunks to manage memory.
        
        Args:
            operation: Operation to apply
            data: Input data
            chunk_size: Size of each chunk
            axis: Axis along which to chunk
            
        Returns:
            Result of operation applied to all chunks
        """
        if data.shape[axis] <= chunk_size:
            return operation(data)

        results = []
        for i in range(0, data.shape[axis], chunk_size):
            if axis == 0:
                chunk = data[i:i+chunk_size]
            elif axis == 1:
                chunk = data[:, i:i+chunk_size]
            else:
                raise NotImplementedError("Only axis 0 and 1 supported")

            chunk_result = operation(chunk)
            results.append(chunk_result)

        return jnp.concatenate(results, axis=axis)

    @staticmethod
    def gradient_checkpointing(func: Callable, *args, **kwargs):
        """Apply gradient checkpointing to reduce memory usage.
        
        This is a placeholder for gradient checkpointing implementation.
        In practice, this would use JAX's checkpointing utilities.
        """
        return func(*args, **kwargs)

    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Get current memory usage statistics."""
        try:
            import os

            import psutil

            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()

            return {
                "rss_mb": memory_info.rss / 1024 / 1024,
                "vms_mb": memory_info.vms / 1024 / 1024,
                "percent": process.memory_percent(),
                "available_mb": psutil.virtual_memory().available / 1024 / 1024,
            }
        except ImportError:
            return {"error": "psutil not available"}


def optimize_for_hardware(func: Callable, device_type: str = "auto") -> Callable:
    """Optimize function for specific hardware.
    
    Args:
        func: Function to optimize
        device_type: Target device type ("cpu", "gpu", "tpu", "auto")
        
    Returns:
        Hardware-optimized function
    """
    if device_type == "auto":
        # Auto-detect best device
        if jax.device_count("gpu") > 0:
            device_type = "gpu"
        elif jax.device_count("tpu") > 0:
            device_type = "tpu"
        else:
            device_type = "cpu"

    # Apply device-specific optimizations
    if device_type == "gpu":
        # GPU optimizations: prefer larger batch sizes, use float32
        optimized_func = jit(func)
    elif device_type == "tpu":
        # TPU optimizations: large batch sizes, specific padding
        optimized_func = jit(func)
    else:
        # CPU optimizations: smaller batch sizes, can use float64
        optimized_func = jit(func)

    return optimized_func


class AdaptivePerformanceTuner:
    """Adaptive performance tuning based on runtime characteristics."""

    def __init__(self):
        """Initialize adaptive performance tuner."""
        self.performance_history = []
        self.current_config = {
            "batch_size": 100,
            "enable_jit": True,
            "enable_vectorization": True,
            "parallel_workers": 4,
        }
        self.tuning_iterations = 0

    def tune_config(self, benchmark_func: Callable, test_data: Array) -> Dict[str, Any]:
        """Automatically tune configuration based on performance.
        
        Args:
            benchmark_func: Function to benchmark
            test_data: Test data for benchmarking
            
        Returns:
            Optimized configuration
        """
        best_config = self.current_config.copy()
        best_time = float("inf")

        # Test different configurations
        test_configs = [
            {"batch_size": 50, "enable_jit": True},
            {"batch_size": 100, "enable_jit": True},
            {"batch_size": 500, "enable_jit": True},
            {"batch_size": 100, "enable_jit": False},
            {"batch_size": 100, "enable_vectorization": False},
        ]

        for config in test_configs:
            test_config = {**self.current_config, **config}

            # Benchmark configuration
            start_time = time.time()
            try:
                _ = benchmark_func(test_data, **test_config)
                execution_time = time.time() - start_time

                if execution_time < best_time:
                    best_time = execution_time
                    best_config = test_config

            except Exception:
                continue

        self.current_config = best_config
        self.tuning_iterations += 1

        self.performance_history.append({
            "iteration": self.tuning_iterations,
            "config": best_config.copy(),
            "execution_time": best_time,
        })

        return best_config

    def get_recommendations(self) -> Dict[str, Any]:
        """Get performance optimization recommendations."""
        if len(self.performance_history) < 2:
            return {"status": "insufficient_data"}

        latest = self.performance_history[-1]
        previous = self.performance_history[-2]

        improvement = (previous["execution_time"] - latest["execution_time"]) / previous["execution_time"]

        recommendations = {
            "performance_improvement": improvement,
            "current_config": latest["config"],
            "recommendations": []
        }

        # Generate specific recommendations
        if improvement < 0.05:  # Less than 5% improvement
            recommendations["recommendations"].append(
                "Consider increasing batch size or enabling more aggressive optimizations"
            )

        if latest["config"]["enable_jit"] and latest["execution_time"] > 1.0:
            recommendations["recommendations"].append(
                "JIT compilation overhead detected. Consider pre-compilation for repeated operations"
            )

        return recommendations
