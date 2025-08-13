"""Enhanced performance optimization features for surrogate optimization."""

import functools
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import multiprocessing

import jax
import jax.numpy as jnp
from jax import Array, jit, pmap, vmap
import numpy as np
from joblib import Parallel, delayed

from ..models.base import Dataset, Surrogate


class PerformanceOptimizer:
    """Performance optimization utilities for surrogate optimization."""
    
    def __init__(
        self,
        use_jit: bool = True,
        use_vectorization: bool = True,
        enable_parallel: bool = True,
        max_workers: Optional[int] = None,
        gpu_memory_fraction: float = 0.9,
    ):
        """Initialize performance optimizer.
        
        Args:
            use_jit: Enable JAX JIT compilation
            use_vectorization: Enable vectorized operations
            enable_parallel: Enable parallel processing
            max_workers: Maximum parallel workers (default: CPU count)
            gpu_memory_fraction: GPU memory allocation fraction
        """
        self.use_jit = use_jit
        self.use_vectorization = use_vectorization
        self.enable_parallel = enable_parallel
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.gpu_memory_fraction = gpu_memory_fraction
        
        self.logger = logging.getLogger(__name__)
        
        # Configure JAX for performance
        self._configure_jax()
        
        # Performance metrics
        self.performance_stats = {
            "jit_compilation_times": [],
            "vectorized_operation_times": [],
            "parallel_execution_times": [],
            "cache_hits": 0,
            "cache_misses": 0,
        }
    
    def _configure_jax(self):
        """Configure JAX for optimal performance."""
        if self.use_jit:
            # Enable 64-bit precision for numerical stability
            jax.config.update("jax_enable_x64", True)
            
            # Configure memory preallocation if GPU available
            try:
                # Check if GPU is available
                if jax.devices('gpu'):
                    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(self.gpu_memory_fraction)
                    self.logger.info(f"GPU detected, memory fraction set to {self.gpu_memory_fraction}")
            except Exception:
                pass  # No GPU available
    
    def jit_compile_function(
        self,
        func: Callable,
        static_argnums: Tuple[int, ...] = (),
        cache_key: Optional[str] = None,
    ) -> Callable:
        """JIT compile a function with caching.
        
        Args:
            func: Function to compile
            static_argnums: Arguments to treat as static
            cache_key: Optional cache key for compiled functions
            
        Returns:
            JIT compiled function
        """
        if not self.use_jit:
            return func
        
        start_time = time.time()
        
        # Create JIT compiled function
        compiled_func = jit(func, static_argnums=static_argnums)
        
        # Record compilation time
        compilation_time = time.time() - start_time
        self.performance_stats["jit_compilation_times"].append(compilation_time)
        
        self.logger.debug(f"JIT compiled {func.__name__} in {compilation_time:.4f}s")
        
        return compiled_func
    
    def vectorize_function(
        self,
        func: Callable,
        in_axes: Union[int, Tuple] = 0,
        out_axes: Union[int, Tuple] = 0,
    ) -> Callable:
        """Vectorize a function using JAX vmap.
        
        Args:
            func: Function to vectorize
            in_axes: Input axes to vectorize over
            out_axes: Output axes to vectorize over
            
        Returns:
            Vectorized function
        """
        if not self.use_vectorization:
            return func
        
        vectorized_func = vmap(func, in_axes=in_axes, out_axes=out_axes)
        
        if self.use_jit:
            vectorized_func = self.jit_compile_function(vectorized_func)
        
        return vectorized_func
    
    def parallel_map(
        self,
        func: Callable,
        inputs: List[Any],
        backend: str = "threading",
        chunk_size: Optional[int] = None,
    ) -> List[Any]:
        """Parallel map function with different backends.
        
        Args:
            func: Function to apply
            inputs: List of inputs
            backend: Backend to use ("threading", "multiprocessing", "joblib")
            chunk_size: Chunk size for processing
            
        Returns:
            List of results
        """
        if not self.enable_parallel or len(inputs) < 2:
            return [func(x) for x in inputs]
        
        start_time = time.time()
        
        if backend == "threading":
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                results = list(executor.map(func, inputs, chunksize=chunk_size or 1))
        
        elif backend == "multiprocessing":
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                results = list(executor.map(func, inputs, chunksize=chunk_size or 1))
        
        elif backend == "joblib":
            n_jobs = min(self.max_workers, len(inputs))
            results = Parallel(n_jobs=n_jobs)(delayed(func)(x) for x in inputs)
        
        else:
            raise ValueError(f"Unknown backend: {backend}")
        
        execution_time = time.time() - start_time
        self.performance_stats["parallel_execution_times"].append(execution_time)
        
        self.logger.debug(
            f"Parallel execution ({backend}) completed in {execution_time:.4f}s "
            f"for {len(inputs)} items"
        )
        
        return results
    
    def batch_predict(
        self,
        surrogate: Surrogate,
        X: Array,
        batch_size: int = 1000,
        parallel: bool = True,
    ) -> Array:
        """Efficient batch prediction with memory management.
        
        Args:
            surrogate: Trained surrogate model
            X: Input points for prediction
            batch_size: Batch size for processing
            parallel: Whether to use parallel processing
            
        Returns:
            Predictions for all inputs
        """
        if X.shape[0] <= batch_size:
            return surrogate.predict(X)
        
        # Split into batches
        n_batches = (X.shape[0] + batch_size - 1) // batch_size
        batches = [X[i*batch_size:(i+1)*batch_size] for i in range(n_batches)]
        
        if parallel and self.enable_parallel:
            # Parallel batch processing
            predict_func = lambda batch: surrogate.predict(batch)
            results = self.parallel_map(predict_func, batches, backend="threading")
        else:
            # Sequential batch processing
            results = [surrogate.predict(batch) for batch in batches]
        
        return jnp.concatenate(results, axis=0)
    
    def batch_gradient(
        self,
        surrogate: Surrogate,
        X: Array,
        batch_size: int = 500,
        parallel: bool = True,
    ) -> Array:
        """Efficient batch gradient computation.
        
        Args:
            surrogate: Trained surrogate model
            X: Input points for gradient computation
            batch_size: Batch size for processing
            parallel: Whether to use parallel processing
            
        Returns:
            Gradients for all inputs
        """
        if X.shape[0] <= batch_size:
            return surrogate.gradient(X)
        
        # Split into batches
        n_batches = (X.shape[0] + batch_size - 1) // batch_size
        batches = [X[i*batch_size:(i+1)*batch_size] for i in range(n_batches)]
        
        if parallel and self.enable_parallel:
            # Parallel batch processing
            gradient_func = lambda batch: surrogate.gradient(batch)
            results = self.parallel_map(gradient_func, batches, backend="threading")
        else:
            # Sequential batch processing
            results = [surrogate.gradient(batch) for batch in batches]
        
        return jnp.concatenate(results, axis=0)
    
    def optimize_memory_usage(
        self,
        dataset: Dataset,
        target_memory_mb: float = 1000.0,
    ) -> Dataset:
        """Optimize dataset memory usage.
        
        Args:
            dataset: Input dataset
            target_memory_mb: Target memory usage in MB
            
        Returns:
            Memory-optimized dataset
        """
        # Calculate current memory usage
        current_memory = (
            dataset.X.nbytes + dataset.y.nbytes +
            (dataset.gradients.nbytes if dataset.gradients is not None else 0)
        ) / (1024 ** 2)  # Convert to MB
        
        if current_memory <= target_memory_mb:
            return dataset  # No optimization needed
        
        # Calculate reduction factor
        reduction_factor = target_memory_mb / current_memory
        new_n_samples = int(dataset.n_samples * reduction_factor)
        
        self.logger.warning(
            f"Reducing dataset size from {dataset.n_samples} to {new_n_samples} "
            f"to fit memory target of {target_memory_mb:.1f}MB"
        )
        
        # Random sampling to reduce size
        indices = np.random.choice(dataset.n_samples, new_n_samples, replace=False)
        indices = jnp.sort(indices)
        
        optimized_dataset = Dataset(
            X=dataset.X[indices],
            y=dataset.y[indices],
            gradients=dataset.gradients[indices] if dataset.gradients is not None else None,
            metadata=dict(dataset.metadata)
        )
        
        optimized_dataset.metadata["memory_optimized"] = True
        optimized_dataset.metadata["original_size"] = dataset.n_samples
        optimized_dataset.metadata["reduction_factor"] = reduction_factor
        
        return optimized_dataset
    
    def create_performance_profile(
        self,
        surrogate: Surrogate,
        test_sizes: List[int] = [10, 100, 1000, 10000],
        n_dims: int = 5,
    ) -> Dict[str, Any]:
        """Create performance profile for a surrogate model.
        
        Args:
            surrogate: Trained surrogate model
            test_sizes: List of test sizes
            n_dims: Number of input dimensions
            
        Returns:
            Performance profile dictionary
        """
        profile = {
            "prediction_times": {},
            "gradient_times": {},
            "throughput": {},
            "memory_usage": {},
        }
        
        for size in test_sizes:
            # Generate test data
            X_test = jnp.array(np.random.randn(size, n_dims))
            
            # Time prediction
            start_time = time.time()
            predictions = self.batch_predict(surrogate, X_test, parallel=False)
            pred_time = time.time() - start_time
            
            # Time gradient computation
            start_time = time.time()
            try:
                gradients = self.batch_gradient(surrogate, X_test, parallel=False)
                grad_time = time.time() - start_time
            except Exception:
                grad_time = None
            
            # Calculate throughput
            pred_throughput = size / pred_time if pred_time > 0 else 0
            grad_throughput = size / grad_time if grad_time and grad_time > 0 else 0
            
            # Estimate memory usage
            memory_usage = (predictions.nbytes + X_test.nbytes) / (1024 ** 2)
            
            profile["prediction_times"][size] = pred_time
            profile["gradient_times"][size] = grad_time
            profile["throughput"][size] = {
                "prediction": pred_throughput,
                "gradient": grad_throughput,
            }
            profile["memory_usage"][size] = memory_usage
        
        return profile
    
    def get_performance_recommendations(
        self,
        profile: Dict[str, Any],
        target_throughput: float = 1000.0,
    ) -> List[str]:
        """Generate performance optimization recommendations.
        
        Args:
            profile: Performance profile from create_performance_profile
            target_throughput: Target throughput (predictions/second)
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Check throughput performance
        throughputs = [
            profile["throughput"][size]["prediction"]
            for size in profile["throughput"].keys()
        ]
        
        max_throughput = max(throughputs) if throughputs else 0
        
        if max_throughput < target_throughput:
            recommendations.append(
                f"Current max throughput ({max_throughput:.1f}/s) is below target "
                f"({target_throughput:.1f}/s). Consider enabling parallelization."
            )
        
        # Check memory usage scaling
        memory_usages = list(profile["memory_usage"].values())
        if len(memory_usages) > 1:
            memory_growth = memory_usages[-1] / memory_usages[0]
            if memory_growth > 100:  # More than 100x memory growth
                recommendations.append(
                    "Memory usage grows significantly with batch size. "
                    "Consider implementing batch processing."
                )
        
        # Check gradient computation availability
        grad_times = [
            time for time in profile["gradient_times"].values() if time is not None
        ]
        if len(grad_times) == 0:
            recommendations.append(
                "Gradient computation failed. Check surrogate model implementation."
            )
        
        # Check performance scaling
        sizes = sorted(profile["prediction_times"].keys())
        if len(sizes) > 1:
            small_size, large_size = sizes[0], sizes[-1]
            small_time = profile["prediction_times"][small_size]
            large_time = profile["prediction_times"][large_size]
            
            expected_time = small_time * (large_size / small_size)
            actual_speedup = expected_time / large_time
            
            if actual_speedup > 5:  # Good vectorization
                recommendations.append("Good vectorization performance detected.")
            elif actual_speedup < 1.2:  # Poor scaling
                recommendations.append(
                    "Poor performance scaling detected. Consider enabling vectorization."
                )
        
        if not recommendations:
            recommendations.append("Performance profile looks good!")
        
        return recommendations


def performance_benchmark(func: Callable) -> Callable:
    """Decorator to benchmark function performance.
    
    Args:
        func: Function to benchmark
        
    Returns:
        Decorated function with performance logging
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = jax.devices()[0].memory_stats()['bytes_in_use'] if jax.devices() else 0
        
        try:
            result = func(*args, **kwargs)
            
            end_time = time.time()
            end_memory = jax.devices()[0].memory_stats()['bytes_in_use'] if jax.devices() else 0
            
            execution_time = end_time - start_time
            memory_delta = (end_memory - start_memory) / (1024 ** 2)  # MB
            
            logging.info(
                f"Performance: {func.__name__} - "
                f"Time: {execution_time:.4f}s, "
                f"Memory: {memory_delta:+.2f}MB"
            )
            
            return result
            
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            
            logging.error(
                f"Performance: {func.__name__} FAILED after {execution_time:.4f}s - {e}"
            )
            raise
    
    return wrapper


class AdaptiveCache:
    """Adaptive cache for expensive computations."""
    
    def __init__(self, max_size: int = 1000, ttl: float = 3600.0):
        """Initialize adaptive cache.
        
        Args:
            max_size: Maximum cache size
            ttl: Time-to-live for cached entries (seconds)
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache = {}
        self.access_times = {}
        self.access_counts = {}
        
        self.hits = 0
        self.misses = 0
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired."""
        if key not in self.access_times:
            return True
        return time.time() - self.access_times[key] > self.ttl
    
    def _evict_oldest(self):
        """Evict oldest cache entry."""
        if not self.cache:
            return
        
        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        self._remove_entry(oldest_key)
    
    def _remove_entry(self, key: str):
        """Remove cache entry."""
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
        self.access_counts.pop(key, None)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        if key in self.cache and not self._is_expired(key):
            self.access_times[key] = time.time()
            self.access_counts[key] = self.access_counts.get(key, 0) + 1
            self.hits += 1
            return self.cache[key]
        
        self.misses += 1
        self._remove_entry(key)  # Remove expired entry
        return None
    
    def put(self, key: str, value: Any):
        """Put value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        # Evict entries if cache is full
        while len(self.cache) >= self.max_size:
            self._evict_oldest()
        
        self.cache[key] = value
        self.access_times[key] = time.time()
        self.access_counts[key] = 1
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.access_times.clear()
        self.access_counts.clear()
        self.hits = 0
        self.misses = 0
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Cache statistics dictionary
        """
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "total_requests": total_requests,
        }


# Global cache instance
_global_cache = AdaptiveCache()


def cached_computation(cache_key_func: Callable = None, ttl: float = 3600.0):
    """Decorator for caching expensive computations.
    
    Args:
        cache_key_func: Function to generate cache key from arguments
        ttl: Time-to-live for cached results
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if cache_key_func:
                cache_key = cache_key_func(*args, **kwargs)
            else:
                # Default key generation
                cache_key = f"{func.__name__}_{hash((args, tuple(sorted(kwargs.items()))))}"
            
            # Check cache
            cached_result = _global_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Compute result
            result = func(*args, **kwargs)
            
            # Cache result
            _global_cache.put(cache_key, result)
            
            return result
        
        return wrapper
    return decorator