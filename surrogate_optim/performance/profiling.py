"""Profiling and performance analysis utilities."""

from contextlib import contextmanager
import cProfile
import functools
from io import StringIO
import pstats
import time
from typing import Any, Callable, Dict, List, Optional

import jax.numpy as jnp

from ..monitoring.logging import get_logger
from .memory import MemoryMonitor


class PerformanceProfiler:
    """Performance profiler for optimization workflows."""

    def __init__(self, enable_memory_profiling: bool = True):
        """Initialize performance profiler.
        
        Args:
            enable_memory_profiling: Whether to profile memory usage
        """
        self.enable_memory_profiling = enable_memory_profiling
        self.memory_monitor = MemoryMonitor() if enable_memory_profiling else None
        self.logger = get_logger()

        # Timing data
        self.timings = {}
        self.call_counts = {}

        # Profiling data
        self.profiler = None
        self.profile_stats = None

    @contextmanager
    def profile_operation(self, operation_name: str, enable_cprofile: bool = False):
        """Context manager for profiling an operation.
        
        Args:
            operation_name: Name of the operation
            enable_cprofile: Whether to use cProfile
        """
        start_time = time.time()

        # Start memory monitoring
        if self.memory_monitor:
            self.memory_monitor.start_monitoring()

        # Start cProfile if requested
        if enable_cprofile:
            self.profiler = cProfile.Profile()
            self.profiler.enable()

        try:
            yield self
        finally:
            duration = time.time() - start_time

            # Stop cProfile
            if self.profiler:
                self.profiler.disable()

                # Capture profile stats
                s = StringIO()
                ps = pstats.Stats(self.profiler, stream=s)
                ps.sort_stats("cumulative").print_stats(20)
                self.profile_stats = s.getvalue()

            # Record timing
            if operation_name not in self.timings:
                self.timings[operation_name] = []
                self.call_counts[operation_name] = 0

            self.timings[operation_name].append(duration)
            self.call_counts[operation_name] += 1

            # Log performance info
            self._log_operation_performance(operation_name, duration)

    def _log_operation_performance(self, operation_name: str, duration: float):
        """Log performance information for an operation."""
        memory_info = ""
        if self.memory_monitor:
            memory_summary = self.memory_monitor.get_summary()
            if "error" not in memory_summary:
                memory_info = f", memory: peak={memory_summary['peak_mb']:.1f}MB"

        self.logger.info(
            f"Performance [{operation_name}]: "
            f"duration={duration:.3f}s{memory_info}"
        )

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        summary = {
            "timing_summary": {},
            "memory_summary": {},
            "profile_available": self.profile_stats is not None,
        }

        # Timing summary
        for operation, times in self.timings.items():
            summary["timing_summary"][operation] = {
                "total_time": sum(times),
                "mean_time": sum(times) / len(times),
                "min_time": min(times),
                "max_time": max(times),
                "call_count": self.call_counts[operation],
            }

        # Memory summary
        if self.memory_monitor:
            memory_info = self.memory_monitor.get_summary()
            if "error" not in memory_info:
                summary["memory_summary"] = memory_info

        return summary

    def get_profile_report(self) -> Optional[str]:
        """Get detailed profiling report."""
        return self.profile_stats

    def reset(self):
        """Reset all profiling data."""
        self.timings.clear()
        self.call_counts.clear()
        self.profile_stats = None
        self.profiler = None

        if self.memory_monitor:
            self.memory_monitor = MemoryMonitor()


class TimingProfiler:
    """Simple timing profiler for function calls."""

    def __init__(self):
        """Initialize timing profiler."""
        self.timings = {}
        self.active_timers = {}

    def start_timer(self, name: str):
        """Start timing an operation."""
        self.active_timers[name] = time.time()

    def end_timer(self, name: str) -> float:
        """End timing an operation and return duration."""
        if name not in self.active_timers:
            raise ValueError(f"Timer '{name}' was not started")

        duration = time.time() - self.active_timers[name]
        del self.active_timers[name]

        if name not in self.timings:
            self.timings[name] = []
        self.timings[name].append(duration)

        return duration

    @contextmanager
    def time_operation(self, name: str):
        """Context manager for timing operations."""
        self.start_timer(name)
        try:
            yield
        finally:
            self.end_timer(name)

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get timing statistics."""
        stats = {}

        for name, times in self.timings.items():
            stats[name] = {
                "count": len(times),
                "total": sum(times),
                "mean": sum(times) / len(times),
                "min": min(times),
                "max": max(times),
                "std": float(jnp.std(jnp.array(times))),
            }

        return stats


def profile_function(
    enable_timing: bool = True,
    enable_memory: bool = False,
    enable_cprofile: bool = False,
):
    """Decorator for profiling function performance.
    
    Args:
        enable_timing: Whether to time function calls
        enable_memory: Whether to profile memory usage
        enable_cprofile: Whether to use cProfile
        
    Returns:
        Function decorator
    """
    def decorator(func: Callable) -> Callable:

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            profiler = PerformanceProfiler(enable_memory)

            with profiler.profile_operation(
                func.__name__,
                enable_cprofile=enable_cprofile
            ):
                result = func(*args, **kwargs)

            # Attach profiling data to function
            if not hasattr(wrapper, "_profile_data"):
                wrapper._profile_data = []

            wrapper._profile_data.append(profiler.get_performance_summary())

            return result

        def get_profile_summary():
            """Get accumulated profiling summary."""
            if not hasattr(wrapper, "_profile_data"):
                return {}

            return wrapper._profile_data

        wrapper.get_profile_summary = get_profile_summary

        return wrapper

    return decorator


class ProfiledOptimizer:
    """Optimizer wrapper that profiles performance."""

    def __init__(
        self,
        base_optimizer,
        profile_level: str = "basic",
        output_file: Optional[str] = None,
    ):
        """Initialize profiled optimizer.
        
        Args:
            base_optimizer: Base optimizer to wrap
            profile_level: Profiling level ("basic", "detailed", "full")
            output_file: Optional file to save profiling results
        """
        self.base_optimizer = base_optimizer
        self.profile_level = profile_level
        self.output_file = output_file
        self.logger = get_logger()

        # Setup profiler based on level
        enable_memory = profile_level in ["detailed", "full"]
        enable_cprofile = profile_level == "full"

        self.profiler = PerformanceProfiler(enable_memory)
        self.enable_cprofile = enable_cprofile

        # Performance metrics
        self.iteration_times = []
        self.function_eval_times = []
        self.gradient_eval_times = []

    def optimize(self, surrogate, x0, bounds=None, **kwargs):
        """Optimize with performance profiling."""
        with self.profiler.profile_operation(
            "full_optimization",
            enable_cprofile=self.enable_cprofile
        ):
            # Wrap surrogate with profiling
            profiled_surrogate = self._wrap_surrogate_with_profiling(surrogate)

            # Run optimization
            result = self.base_optimizer.optimize(
                profiled_surrogate, x0, bounds, **kwargs
            )

            # Add profiling metadata to result
            if result.metadata is None:
                result.metadata = {}

            result.metadata["performance_profile"] = self._get_optimization_profile()

            # Save profiling results if requested
            if self.output_file:
                self._save_profile_results()

            return result

    def _wrap_surrogate_with_profiling(self, surrogate):
        """Wrap surrogate with profiling instrumentation."""
        class ProfiledSurrogate:
            def __init__(self, base_surrogate, profiler_ref):
                self.base = base_surrogate
                self.profiler_ref = profiler_ref

            def predict(self, x):
                start_time = time.time()
                result = self.base.predict(x)
                duration = time.time() - start_time

                self.profiler_ref.function_eval_times.append(duration)
                return result

            def gradient(self, x):
                start_time = time.time()
                result = self.base.gradient(x)
                duration = time.time() - start_time

                self.profiler_ref.gradient_eval_times.append(duration)
                return result

            def __getattr__(self, name):
                return getattr(self.base, name)

        return ProfiledSurrogate(surrogate, self)

    def _get_optimization_profile(self) -> Dict[str, Any]:
        """Get optimization performance profile."""
        profile = {
            "function_evaluations": {
                "count": len(self.function_eval_times),
                "total_time": sum(self.function_eval_times),
                "mean_time": sum(self.function_eval_times) / len(self.function_eval_times)
                           if self.function_eval_times else 0,
            },
            "gradient_evaluations": {
                "count": len(self.gradient_eval_times),
                "total_time": sum(self.gradient_eval_times),
                "mean_time": sum(self.gradient_eval_times) / len(self.gradient_eval_times)
                           if self.gradient_eval_times else 0,
            },
        }

        # Add general performance summary
        profile.update(self.profiler.get_performance_summary())

        return profile

    def _save_profile_results(self):
        """Save profiling results to file."""
        try:
            import json

            profile_data = {
                "optimizer_type": type(self.base_optimizer).__name__,
                "profile_level": self.profile_level,
                "performance_summary": self._get_optimization_profile(),
                "detailed_profile": self.profiler.get_profile_report(),
            }

            with open(self.output_file, "w") as f:
                json.dump(profile_data, f, indent=2, default=str)

            self.logger.info(f"Profile results saved to {self.output_file}")

        except Exception as e:
            self.logger.warning(f"Failed to save profile results: {e}")

    def __getattr__(self, name):
        """Delegate other attributes to base optimizer."""
        return getattr(self.base_optimizer, name)


@contextmanager
def performance_monitor(
    operation_name: str = "operation",
    profile_level: str = "basic",
    log_results: bool = True,
):
    """Context manager for monitoring operation performance.
    
    Args:
        operation_name: Name of the operation
        profile_level: Level of profiling ("basic", "detailed", "full")
        log_results: Whether to log results
        
    Yields:
        PerformanceProfiler instance
    """
    enable_memory = profile_level in ["detailed", "full"]
    enable_cprofile = profile_level == "full"

    profiler = PerformanceProfiler(enable_memory)

    with profiler.profile_operation(operation_name, enable_cprofile):
        yield profiler

    if log_results:
        summary = profiler.get_performance_summary()
        logger = get_logger()

        # Log timing info
        if operation_name in summary["timing_summary"]:
            timing = summary["timing_summary"][operation_name]
            logger.info(
                f"Performance summary [{operation_name}]: "
                f"duration={timing['mean_time']:.3f}s, "
                f"calls={timing['call_count']}"
            )

        # Log memory info if available
        if summary["memory_summary"]:
            memory = summary["memory_summary"]
            logger.info(
                f"Memory usage [{operation_name}]: "
                f"peak={memory.get('peak_mb', 0):.1f}MB, "
                f"increase={memory.get('increase_mb', 0):.1f}MB"
            )


class BenchmarkSuite:
    """Suite for benchmarking surrogate optimization performance."""

    def __init__(self):
        """Initialize benchmark suite."""
        self.results = {}
        self.logger = get_logger()

    def benchmark_data_collection(
        self,
        functions: List[Callable],
        sample_sizes: List[int],
        dimensions: List[int],
        n_runs: int = 3,
    ) -> Dict[str, Any]:
        """Benchmark data collection performance.
        
        Args:
            functions: List of test functions
            sample_sizes: List of sample sizes to test
            dimensions: List of dimensions to test
            n_runs: Number of runs per configuration
            
        Returns:
            Benchmark results
        """
        from ..data.collector import DataCollector

        results = {}

        for func_idx, func in enumerate(functions):
            func_name = getattr(func, "__name__", f"function_{func_idx}")
            results[func_name] = {}

            for dim in dimensions:
                bounds = [(-2.0, 2.0)] * dim
                results[func_name][f"{dim}d"] = {}

                for n_samples in sample_sizes:
                    times = []

                    for run in range(n_runs):
                        collector = DataCollector(func, bounds)

                        start_time = time.time()
                        dataset = collector.collect(n_samples, verbose=False)
                        duration = time.time() - start_time

                        times.append(duration)

                    results[func_name][f"{dim}d"][f"{n_samples}_samples"] = {
                        "mean_time": sum(times) / len(times),
                        "min_time": min(times),
                        "max_time": max(times),
                        "std_time": float(jnp.std(jnp.array(times))),
                    }

        self.results["data_collection"] = results
        return results

    def benchmark_surrogate_training(
        self,
        surrogate_types: List[str],
        dataset_sizes: List[int],
        dimensions: List[int],
        n_runs: int = 3,
    ) -> Dict[str, Any]:
        """Benchmark surrogate model training performance.
        
        Args:
            surrogate_types: List of surrogate model types
            dataset_sizes: List of dataset sizes
            dimensions: List of dimensions
            n_runs: Number of runs per configuration
            
        Returns:
            Benchmark results
        """
        from ..models.base import Dataset
        from ..models.gaussian_process import GPSurrogate
        from ..models.neural import NeuralSurrogate
        from ..models.random_forest import RandomForestSurrogate

        surrogate_classes = {
            "neural": NeuralSurrogate,
            "gp": GPSurrogate,
            "rf": RandomForestSurrogate,
        }

        results = {}

        for surrogate_type in surrogate_types:
            if surrogate_type not in surrogate_classes:
                continue

            results[surrogate_type] = {}

            for dim in dimensions:
                results[surrogate_type][f"{dim}d"] = {}

                for n_samples in dataset_sizes:
                    # Generate synthetic dataset
                    import jax.random
                    key = jax.random.PRNGKey(42)
                    X = jax.random.normal(key, (n_samples, dim))
                    y = jnp.sum(X**2, axis=1)  # Quadratic function
                    dataset = Dataset(X=X, y=y)

                    times = []

                    for run in range(n_runs):
                        # Create fresh surrogate for each run
                        if surrogate_type == "neural":
                            surrogate = NeuralSurrogate(n_epochs=10, random_seed=run)
                        else:
                            surrogate = surrogate_classes[surrogate_type]()

                        start_time = time.time()
                        surrogate.fit(dataset)
                        duration = time.time() - start_time

                        times.append(duration)

                    results[surrogate_type][f"{dim}d"][f"{n_samples}_samples"] = {
                        "mean_time": sum(times) / len(times),
                        "min_time": min(times),
                        "max_time": max(times),
                        "std_time": float(jnp.std(jnp.array(times))),
                    }

        self.results["surrogate_training"] = results
        return results

    def get_benchmark_report(self) -> str:
        """Generate benchmark report."""
        if not self.results:
            return "No benchmark results available."

        report = ["Surrogate Optimization Benchmark Report"]
        report.append("=" * 50)

        for category, results in self.results.items():
            report.append(f"\n{category.upper()} BENCHMARKS:")
            report.append("-" * 30)

            # Format results in a readable way
            for key, value in results.items():
                report.append(f"\n{key}:")
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, dict):
                            report.append(f"  {subkey}:")
                            for metric, val in subvalue.items():
                                if isinstance(val, float):
                                    report.append(f"    {metric}: {val:.4f}s")
                                else:
                                    report.append(f"    {metric}: {val}")
                        else:
                            report.append(f"  {subkey}: {subvalue}")

        return "\n".join(report)
