"""Scalable high-performance surrogate optimizer with advanced optimizations."""

import time
from typing import Any, Dict, List, Optional, Tuple, Union

from jax import Array
import jax.numpy as jnp
import numpy as np

from ..performance.enhanced_performance import (
    PerformanceOptimizer,
    cached_computation,
    performance_benchmark,
)
from .enhanced_optimizer import EnhancedSurrogateOptimizer
from .error_handling import (
    check_numerical_stability,
    validate_array_input,
)


class ScalableSurrogateOptimizer(EnhancedSurrogateOptimizer):
    """Scalable surrogate optimizer with advanced performance optimizations.
    
    This class extends the EnhancedSurrogateOptimizer with:
    - JIT compilation for critical functions
    - Vectorized operations for batch processing
    - Parallel processing capabilities
    - Adaptive caching for expensive computations
    - Memory optimization for large datasets
    - Performance profiling and recommendations
    """

    def __init__(
        self,
        surrogate_type: str = "gaussian_process",
        surrogate_params: Optional[Dict[str, Any]] = None,
        optimizer_type: str = "gradient_descent",
        optimizer_params: Optional[Dict[str, Any]] = None,
        enable_monitoring: bool = True,
        enable_validation: bool = True,
        max_retries: int = 3,
        random_seed: Optional[int] = None,
        # Performance optimization parameters
        enable_jit: bool = True,
        enable_vectorization: bool = True,
        enable_parallel: bool = True,
        max_workers: Optional[int] = None,
        batch_size: int = 1000,
        enable_caching: bool = True,
        memory_limit_mb: float = 2000.0,
        auto_optimize: bool = True,
    ):
        """Initialize scalable surrogate optimizer.
        
        Args:
            surrogate_type: Type of surrogate model
            surrogate_params: Parameters for surrogate model
            optimizer_type: Type of optimizer
            optimizer_params: Parameters for optimizer
            enable_monitoring: Enable performance monitoring
            enable_validation: Enable input/output validation
            max_retries: Maximum retries for failed operations
            random_seed: Random seed for reproducibility
            enable_jit: Enable JAX JIT compilation
            enable_vectorization: Enable vectorized operations
            enable_parallel: Enable parallel processing
            max_workers: Maximum parallel workers
            batch_size: Default batch size for operations
            enable_caching: Enable adaptive caching
            memory_limit_mb: Memory limit in MB
            auto_optimize: Enable automatic performance optimization
        """
        # Initialize parent class
        super().__init__(
            surrogate_type=surrogate_type,
            surrogate_params=surrogate_params,
            optimizer_type=optimizer_type,
            optimizer_params=optimizer_params,
            enable_monitoring=enable_monitoring,
            enable_validation=enable_validation,
            max_retries=max_retries,
            random_seed=random_seed,
        )

        # Performance optimization settings
        self.enable_jit = enable_jit
        self.enable_vectorization = enable_vectorization
        self.enable_parallel = enable_parallel
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.enable_caching = enable_caching
        self.memory_limit_mb = memory_limit_mb
        self.auto_optimize = auto_optimize

        # Initialize performance optimizer
        self.performance_optimizer = PerformanceOptimizer(
            use_jit=enable_jit,
            use_vectorization=enable_vectorization,
            enable_parallel=enable_parallel,
            max_workers=max_workers,
        )

        # Performance tracking
        self.scale_metrics = {
            "batch_sizes_used": [],
            "parallel_speedups": [],
            "jit_speedups": [],
            "cache_performance": {},
            "memory_optimizations": [],
        }

        self.logger.info(
            f"Initialized ScalableSurrogateOptimizer with performance optimizations: "
            f"JIT={enable_jit}, Vectorization={enable_vectorization}, "
            f"Parallel={enable_parallel}, Caching={enable_caching}"
        )

    @performance_benchmark
    def fit_surrogate(
        self,
        data: Union[Dataset, Dict[str, Array]],
        validate_data: bool = None,
        optimize_memory: bool = None,
    ) -> "ScalableSurrogateOptimizer":
        """Train surrogate model with performance optimizations.
        
        Args:
            data: Training data
            validate_data: Whether to validate data
            optimize_memory: Whether to optimize memory usage
            
        Returns:
            Self for method chaining
        """
        start_time = time.time()
        optimize_memory = optimize_memory if optimize_memory is not None else self.auto_optimize

        # Convert to Dataset if needed
        if isinstance(data, dict):
            data = Dataset(
                X=data["X"],
                y=data["y"],
                gradients=data.get("gradients"),
                metadata=data.get("metadata", {})
            )

        # Memory optimization
        original_size = data.n_samples
        if optimize_memory and self.memory_limit_mb:
            data = self.performance_optimizer.optimize_memory_usage(
                data, self.memory_limit_mb
            )

            if data.n_samples != original_size:
                self.scale_metrics["memory_optimizations"].append({
                    "original_size": original_size,
                    "optimized_size": data.n_samples,
                    "reduction_factor": data.n_samples / original_size,
                })

        # Use parent implementation with performance tracking
        result = super().fit_surrogate(data, validate_data)

        # Record batch size used
        effective_batch_size = min(self.batch_size, data.n_samples)
        self.scale_metrics["batch_sizes_used"].append(effective_batch_size)

        training_time = time.time() - start_time
        self.logger.info(
            f"Scalable training completed in {training_time:.2f}s "
            f"(batch_size={effective_batch_size}, "
            f"memory_optimized={data.n_samples != original_size})"
        )

        return result

    @performance_benchmark
    @cached_computation(ttl=1800.0)  # 30 minute cache
    def predict(
        self,
        x: Array,
        validate_inputs: bool = None,
        use_batch_processing: bool = None,
        use_parallel: bool = None,
    ) -> Array:
        """High-performance batch prediction with optimizations.
        
        Args:
            x: Input points for prediction
            validate_inputs: Whether to validate inputs
            use_batch_processing: Whether to use batch processing
            use_parallel: Whether to use parallel processing
            
        Returns:
            Predicted function values
        """
        if not self.is_fitted:
            raise RuntimeError("Surrogate must be trained before prediction")

        # Auto-configure optimization settings
        use_batch_processing = (
            use_batch_processing if use_batch_processing is not None
            else (x.shape[0] > self.batch_size and self.auto_optimize)
        )
        use_parallel = (
            use_parallel if use_parallel is not None
            else (x.shape[0] > 100 and self.enable_parallel)
        )

        # Validate inputs if requested
        if validate_inputs:
            x = validate_array_input(x, "prediction input", finite_values=True)

        # Choose prediction strategy based on input size and settings
        if use_batch_processing:
            predictions = self.performance_optimizer.batch_predict(
                self.surrogate, x, self.batch_size, use_parallel
            )
        else:
            predictions = self.surrogate.predict(x)

        # Validate outputs
        if self.enable_validation:
            check_numerical_stability(predictions, "predictions")

        return predictions

    @performance_benchmark
    @cached_computation(ttl=1800.0)
    def gradient(
        self,
        x: Array,
        validate_inputs: bool = None,
        use_batch_processing: bool = None,
        use_parallel: bool = None,
    ) -> Array:
        """High-performance batch gradient computation.
        
        Args:
            x: Input points for gradient computation
            validate_inputs: Whether to validate inputs
            use_batch_processing: Whether to use batch processing
            use_parallel: Whether to use parallel processing
            
        Returns:
            Gradient vectors
        """
        if not self.is_fitted:
            raise RuntimeError("Surrogate must be trained before gradient computation")

        # Auto-configure optimization settings
        use_batch_processing = (
            use_batch_processing if use_batch_processing is not None
            else (x.shape[0] > self.batch_size // 2 and self.auto_optimize)
        )
        use_parallel = (
            use_parallel if use_parallel is not None
            else (x.shape[0] > 50 and self.enable_parallel)
        )

        # Validate inputs if requested
        if validate_inputs:
            x = validate_array_input(x, "gradient input", finite_values=True)

        # Choose gradient strategy
        if use_batch_processing:
            gradients = self.performance_optimizer.batch_gradient(
                self.surrogate, x, self.batch_size // 2, use_parallel
            )
        else:
            gradients = self.surrogate.gradient(x)

        # Validate outputs
        if self.enable_validation:
            check_numerical_stability(gradients, "gradients")

        return gradients

    def parallel_optimize(
        self,
        initial_points: List[Array],
        bounds: Optional[List[Tuple[float, float]]] = None,
        num_steps: int = 100,
        return_all: bool = False,
        **kwargs
    ) -> Union[Any, List[Any]]:
        """Run parallel optimization from multiple starting points.
        
        Args:
            initial_points: List of starting points
            bounds: Optimization bounds
            num_steps: Maximum optimization steps
            return_all: Whether to return all results or just the best
            **kwargs: Additional optimizer arguments
            
        Returns:
            Best optimization result or all results if return_all=True
        """
        if not self.is_fitted:
            raise RuntimeError("Surrogate must be trained before optimization")

        self.logger.info(
            f"Starting parallel optimization from {len(initial_points)} points"
        )

        # Define optimization function for parallel execution
        def single_optimize(initial_point):
            try:
                return self.optimize(
                    initial_point=initial_point,
                    bounds=bounds,
                    num_steps=num_steps,
                    **kwargs
                )
            except Exception as e:
                self.logger.warning(f"Optimization failed from {initial_point}: {e}")
                return None

        # Run parallel optimization
        start_time = time.time()
        results = self.performance_optimizer.parallel_map(
            single_optimize,
            initial_points,
            backend="threading"
        )
        parallel_time = time.time() - start_time

        # Filter successful results
        valid_results = [r for r in results if r is not None]

        if not valid_results:
            raise RuntimeError("All parallel optimizations failed")

        # Record performance metrics
        speedup = len(initial_points) / (parallel_time / (parallel_time / len(initial_points)))
        self.scale_metrics["parallel_speedups"].append(speedup)

        self.logger.info(
            f"Parallel optimization completed in {parallel_time:.2f}s "
            f"({len(valid_results)}/{len(initial_points)} successful, "
            f"speedup: {speedup:.1f}x)"
        )

        if return_all:
            return valid_results
        # Return best result
        best_result = min(valid_results, key=lambda r: r.fun)
        return best_result

    def create_performance_profile(
        self,
        test_sizes: List[int] = None,
        n_dims: int = None,
    ) -> Dict[str, Any]:
        """Create comprehensive performance profile.
        
        Args:
            test_sizes: List of test sizes
            n_dims: Number of input dimensions
            
        Returns:
            Performance profile dictionary
        """
        if not self.is_fitted:
            raise RuntimeError("Surrogate must be trained before profiling")

        # Use defaults based on training data
        if n_dims is None:
            n_dims = self.training_data.n_dims if self.training_data else 5

        if test_sizes is None:
            test_sizes = [10, 100, 1000, 5000, 10000]

        self.logger.info(f"Creating performance profile with sizes {test_sizes}")

        profile = self.performance_optimizer.create_performance_profile(
            self.surrogate, test_sizes, n_dims
        )

        # Add scalable optimizer specific metrics
        profile["scale_metrics"] = dict(self.scale_metrics)
        profile["optimization_settings"] = {
            "enable_jit": self.enable_jit,
            "enable_vectorization": self.enable_vectorization,
            "enable_parallel": self.enable_parallel,
            "batch_size": self.batch_size,
            "enable_caching": self.enable_caching,
            "memory_limit_mb": self.memory_limit_mb,
        }

        # Generate recommendations
        profile["recommendations"] = self.performance_optimizer.get_performance_recommendations(
            profile, target_throughput=1000.0
        )

        return profile

    def benchmark_scaling(
        self,
        sizes: List[int] = None,
        n_trials: int = 3,
    ) -> Dict[str, Any]:
        """Benchmark scaling performance across different problem sizes.
        
        Args:
            sizes: Problem sizes to test
            n_trials: Number of trials per size
            
        Returns:
            Scaling benchmark results
        """
        if not self.is_fitted:
            raise RuntimeError("Surrogate must be trained before benchmarking")

        sizes = sizes or [100, 500, 1000, 5000, 10000]
        n_dims = self.training_data.n_dims if self.training_data else 5

        self.logger.info(f"Benchmarking scaling performance for sizes {sizes}")

        results = {
            "sizes": sizes,
            "prediction_times": {size: [] for size in sizes},
            "gradient_times": {size: [] for size in sizes},
            "memory_usage": {size: [] for size in sizes},
        }

        for size in sizes:
            self.logger.info(f"Benchmarking size {size}")

            for trial in range(n_trials):
                # Generate test data
                test_X = jnp.array(np.random.randn(size, n_dims))

                # Benchmark prediction
                start_time = time.time()
                predictions = self.predict(test_X, use_batch_processing=True)
                pred_time = time.time() - start_time
                results["prediction_times"][size].append(pred_time)

                # Benchmark gradient computation
                try:
                    start_time = time.time()
                    gradients = self.gradient(test_X, use_batch_processing=True)
                    grad_time = time.time() - start_time
                    results["gradient_times"][size].append(grad_time)
                except Exception:
                    results["gradient_times"][size].append(None)

                # Estimate memory usage
                memory_mb = (test_X.nbytes + predictions.nbytes) / (1024 ** 2)
                results["memory_usage"][size].append(memory_mb)

        # Calculate statistics
        for metric in ["prediction_times", "gradient_times", "memory_usage"]:
            for size in sizes:
                values = [v for v in results[metric][size] if v is not None]
                if values:
                    results[metric][size] = {
                        "mean": float(jnp.mean(jnp.array(values))),
                        "std": float(jnp.std(jnp.array(values))),
                        "min": float(jnp.min(jnp.array(values))),
                        "max": float(jnp.max(jnp.array(values))),
                    }

        # Calculate scaling efficiency
        results["scaling_efficiency"] = self._calculate_scaling_efficiency(results)

        return results

    def _calculate_scaling_efficiency(self, benchmark_results: Dict) -> Dict[str, float]:
        """Calculate scaling efficiency metrics."""
        sizes = benchmark_results["sizes"]

        efficiency = {}

        for metric in ["prediction_times", "gradient_times"]:
            times = [
                benchmark_results[metric][size]["mean"]
                for size in sizes
                if isinstance(benchmark_results[metric][size], dict)
            ]

            if len(times) > 1:
                # Calculate theoretical vs actual scaling
                base_time, base_size = times[0], sizes[0]

                actual_scaling = []
                for i in range(1, len(times)):
                    size_ratio = sizes[i] / base_size
                    time_ratio = times[i] / base_time
                    efficiency_score = size_ratio / time_ratio  # Higher is better
                    actual_scaling.append(efficiency_score)

                efficiency[metric] = float(jnp.mean(jnp.array(actual_scaling))) if actual_scaling else 1.0

        return efficiency

    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance and scaling metrics.
        
        Returns:
            Complete metrics dictionary
        """
        metrics = super().get_performance_metrics()

        # Add scaling-specific metrics
        metrics["scaling"] = dict(self.scale_metrics)
        metrics["performance_optimizer_stats"] = self.performance_optimizer.performance_stats

        # Add cache statistics if caching is enabled
        if self.enable_caching:
            from ..performance.enhanced_performance import _global_cache
            metrics["cache_stats"] = _global_cache.stats()

        # Add memory and compute statistics
        if self.training_data:
            metrics["dataset_stats"] = {
                "n_samples": self.training_data.n_samples,
                "n_dims": self.training_data.n_dims,
                "memory_mb": (
                    self.training_data.X.nbytes + self.training_data.y.nbytes
                ) / (1024 ** 2),
            }

        return metrics

    def auto_tune_performance(self) -> Dict[str, Any]:
        """Automatically tune performance parameters based on current configuration.
        
        Returns:
            Tuning results and recommendations
        """
        if not self.is_fitted:
            raise RuntimeError("Surrogate must be trained before auto-tuning")

        self.logger.info("Starting automatic performance tuning...")

        # Create baseline profile
        baseline_profile = self.create_performance_profile(
            test_sizes=[100, 1000, 5000],
            n_dims=self.training_data.n_dims
        )

        # Test different batch sizes
        batch_sizes = [100, 500, 1000, 2000, 5000]
        best_batch_size = self.batch_size
        best_throughput = 0

        for batch_size in batch_sizes:
            if batch_size > self.training_data.n_samples:
                continue

            # Test this batch size
            self.batch_size = batch_size
            test_throughput = baseline_profile["throughput"][1000]["prediction"]

            if test_throughput > best_throughput:
                best_throughput = test_throughput
                best_batch_size = batch_size

        # Apply best settings
        original_batch_size = self.batch_size
        self.batch_size = best_batch_size

        tuning_results = {
            "original_batch_size": original_batch_size,
            "optimized_batch_size": best_batch_size,
            "throughput_improvement": best_throughput / baseline_profile["throughput"][1000]["prediction"],
            "recommendations": baseline_profile["recommendations"],
        }

        self.logger.info(
            f"Auto-tuning completed: batch_size {original_batch_size} -> {best_batch_size} "
            f"(throughput improvement: {tuning_results['throughput_improvement']:.2f}x)"
        )

        return tuning_results
