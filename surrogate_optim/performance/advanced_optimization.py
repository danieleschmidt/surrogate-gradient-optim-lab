"""Advanced performance optimization and scaling - Generation 3.

This module implements cutting-edge performance optimizations including
GPU acceleration, distributed computing, and advanced caching strategies.
"""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from functools import wraps
import logging
import multiprocessing as mp
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
from jax import Array, device_put, jit, pmap, random, vmap
from jax.experimental import mesh_utils
from jax.lib import xla_bridge
import jax.numpy as jnp

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance tracking metrics."""
    execution_time: float = 0.0
    memory_usage: float = 0.0
    gpu_utilization: float = 0.0
    cache_hit_rate: float = 0.0
    throughput: float = 0.0
    parallel_efficiency: float = 0.0
    numerical_stability: float = 1.0
    scalability_factor: float = 1.0


class PerformanceOptimizer:
    """Advanced performance optimization system."""

    def __init__(self,
                 enable_gpu: bool = True,
                 enable_jit: bool = True,
                 enable_caching: bool = True,
                 cache_size: int = 1000):
        """Initialize performance optimizer.
        
        Args:
            enable_gpu: Enable GPU acceleration
            enable_jit: Enable JIT compilation
            enable_caching: Enable result caching
            cache_size: Maximum cache size
        """
        self.enable_gpu = enable_gpu and self._check_gpu_availability()
        self.enable_jit = enable_jit
        self.enable_caching = enable_caching
        self.cache_size = cache_size

        # Performance tracking
        self.metrics = PerformanceMetrics()
        self.execution_history: List[PerformanceMetrics] = []

        # JIT compilation cache
        self._jit_cache: Dict[str, Callable] = {}

        # Setup devices
        self.devices = jax.devices()
        self.device_count = len(self.devices)

        logger.info(f"Performance optimizer initialized: GPU={self.enable_gpu}, "
                   f"JIT={self.enable_jit}, Devices={self.device_count}")

    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available."""
        try:
            backend = xla_bridge.get_backend()
            platform = backend.platform
            return platform == "gpu"
        except Exception as e:
            logger.warning(f"GPU check failed: {e}")
            return False

    def optimize_function(self, func: Callable,
                         vectorize: bool = True,
                         parallel: bool = True,
                         cache_key: Optional[str] = None) -> Callable:
        """Optimize a function with various acceleration techniques.
        
        Args:
            func: Function to optimize
            vectorize: Apply vectorization
            parallel: Enable parallel execution
            cache_key: Unique cache key for JIT compilation
            
        Returns:
            Optimized function
        """
        optimized_func = func

        # Apply JIT compilation
        if self.enable_jit:
            if cache_key and cache_key in self._jit_cache:
                optimized_func = self._jit_cache[cache_key]
            else:
                try:
                    optimized_func = jit(optimized_func)
                    if cache_key:
                        self._jit_cache[cache_key] = optimized_func
                    logger.debug(f"JIT compiled function: {cache_key or 'anonymous'}")
                except Exception as e:
                    logger.warning(f"JIT compilation failed: {e}")

        # Apply vectorization
        if vectorize:
            try:
                optimized_func = vmap(optimized_func)
                logger.debug("Applied vectorization")
            except Exception as e:
                logger.warning(f"Vectorization failed: {e}")

        # Apply parallelization for multi-device
        if parallel and self.device_count > 1:
            try:
                optimized_func = pmap(optimized_func)
                logger.debug(f"Applied parallelization across {self.device_count} devices")
            except Exception as e:
                logger.warning(f"Parallelization failed: {e}")

        return optimized_func

    def device_put_optimized(self, data: Array, device_id: Optional[int] = None) -> Array:
        """Optimally place data on devices."""
        if not self.enable_gpu:
            return data

        try:
            if device_id is not None and device_id < len(self.devices):
                device = self.devices[device_id]
            else:
                # Use first GPU device
                gpu_devices = [d for d in self.devices if d.device_kind == "gpu"]
                device = gpu_devices[0] if gpu_devices else self.devices[0]

            return device_put(data, device)
        except Exception as e:
            logger.warning(f"Device placement failed: {e}")
            return data

    def batch_process(self, func: Callable, data: Array,
                     batch_size: int = 1000,
                     parallel: bool = True) -> Array:
        """Process data in optimized batches.
        
        Args:
            func: Function to apply
            data: Input data
            batch_size: Size of each batch
            parallel: Process batches in parallel
            
        Returns:
            Processed results
        """
        start_time = time.time()

        # Optimize function
        optimized_func = self.optimize_function(func, vectorize=True, parallel=False)

        # Calculate optimal batch size based on available memory
        if self.enable_gpu:
            # Estimate GPU memory usage and adjust batch size
            estimated_memory_per_sample = data.nbytes / len(data) if len(data) > 0 else 1000
            max_samples_per_batch = min(batch_size, int(2e9 / estimated_memory_per_sample))  # 2GB limit
            batch_size = max(1, max_samples_per_batch)

        # Process in batches
        results = []
        n_batches = (len(data) + batch_size - 1) // batch_size

        logger.debug(f"Processing {len(data)} samples in {n_batches} batches of size {batch_size}")

        if parallel and n_batches > 1:
            # Parallel batch processing
            with ThreadPoolExecutor(max_workers=min(4, n_batches)) as executor:
                futures = []

                for i in range(n_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, len(data))
                    batch = data[start_idx:end_idx]

                    # Place batch on appropriate device
                    batch = self.device_put_optimized(batch, i % self.device_count)

                    future = executor.submit(optimized_func, batch)
                    futures.append(future)

                # Collect results
                for future in futures:
                    result = future.result()
                    results.append(result)
        else:
            # Sequential batch processing
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(data))
                batch = data[start_idx:end_idx]

                # Place batch on device
                batch = self.device_put_optimized(batch)

                result = optimized_func(batch)
                results.append(result)

        # Concatenate results
        final_result = jnp.concatenate(results, axis=0)

        # Update metrics
        execution_time = time.time() - start_time
        throughput = len(data) / execution_time if execution_time > 0 else 0

        self.metrics.execution_time += execution_time
        self.metrics.throughput = throughput

        logger.debug(f"Batch processing completed: {throughput:.1f} samples/sec")

        return final_result


class AdvancedCacheSystem:
    """Sophisticated caching system with LRU and intelligent invalidation."""

    def __init__(self, max_size: int = 1000, ttl_seconds: float = 3600):
        """Initialize cache system.
        
        Args:
            max_size: Maximum number of cached items
            ttl_seconds: Time-to-live for cache entries
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds

        # Multi-level cache
        self._function_cache: Dict[str, Any] = {}
        self._model_cache: Dict[str, Any] = {}
        self._data_cache: Dict[str, Any] = {}

        # Cache metadata
        self._access_times: Dict[str, float] = {}
        self._creation_times: Dict[str, float] = {}
        self._access_counts: Dict[str, int] = {}

        # Cache statistics
        self.hits = 0
        self.misses = 0

        # Thread safety
        self._lock = threading.RLock()

    def _generate_cache_key(self, func_name: str, args: Tuple, kwargs: Dict) -> str:
        """Generate unique cache key."""
        try:
            # Create hashable representation
            args_str = str(tuple(str(arg) for arg in args))
            kwargs_str = str(sorted(kwargs.items()))
            return f"{func_name}:{hash(args_str + kwargs_str)}"
        except Exception:
            # Fallback to simple key
            return f"{func_name}:{time.time()}"

    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired."""
        if key not in self._creation_times:
            return True

        age = time.time() - self._creation_times[key]
        return age > self.ttl_seconds

    def _evict_lru(self):
        """Evict least recently used items."""
        if len(self._function_cache) < self.max_size:
            return

        # Sort by access time (LRU first)
        sorted_keys = sorted(self._access_times.keys(), key=lambda k: self._access_times[k])

        # Evict oldest entries
        keys_to_remove = sorted_keys[:len(sorted_keys) // 4]  # Remove 25%

        for key in keys_to_remove:
            self._remove_entry(key)

        logger.debug(f"Evicted {len(keys_to_remove)} cache entries")

    def _remove_entry(self, key: str):
        """Remove cache entry and metadata."""
        self._function_cache.pop(key, None)
        self._model_cache.pop(key, None)
        self._data_cache.pop(key, None)
        self._access_times.pop(key, None)
        self._creation_times.pop(key, None)
        self._access_counts.pop(key, None)

    def get(self, key: str, default: Any = None) -> Any:
        """Get item from cache."""
        with self._lock:
            # Check if expired
            if self._is_expired(key):
                self._remove_entry(key)
                self.misses += 1
                return default

            # Update access metadata
            current_time = time.time()
            self._access_times[key] = current_time
            self._access_counts[key] = self._access_counts.get(key, 0) + 1

            # Return cached value
            if key in self._function_cache:
                self.hits += 1
                return self._function_cache[key]
            if key in self._model_cache:
                self.hits += 1
                return self._model_cache[key]
            if key in self._data_cache:
                self.hits += 1
                return self._data_cache[key]
            self.misses += 1
            return default

    def put(self, key: str, value: Any, cache_type: str = "function"):
        """Put item in cache."""
        with self._lock:
            current_time = time.time()

            # Evict if necessary
            self._evict_lru()

            # Store in appropriate cache
            if cache_type == "function":
                self._function_cache[key] = value
            elif cache_type == "model":
                self._model_cache[key] = value
            elif cache_type == "data":
                self._data_cache[key] = value

            # Update metadata
            self._creation_times[key] = current_time
            self._access_times[key] = current_time
            self._access_counts[key] = 1

    def cached_function(self, cache_type: str = "function"):
        """Decorator for caching function results."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = self._generate_cache_key(func.__name__, args, kwargs)

                # Try to get from cache
                cached_result = self.get(cache_key)
                if cached_result is not None:
                    return cached_result

                # Execute function and cache result
                start_time = time.time()
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time

                # Only cache if execution took significant time
                if execution_time > 0.01:  # 10ms threshold
                    self.put(cache_key, result, cache_type)

                return result

            return wrapper
        return decorator

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0

        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "total_entries": len(self._function_cache) + len(self._model_cache) + len(self._data_cache),
            "function_cache_size": len(self._function_cache),
            "model_cache_size": len(self._model_cache),
            "data_cache_size": len(self._data_cache),
        }

    def clear(self):
        """Clear all caches."""
        with self._lock:
            self._function_cache.clear()
            self._model_cache.clear()
            self._data_cache.clear()
            self._access_times.clear()
            self._creation_times.clear()
            self._access_counts.clear()
            self.hits = 0
            self.misses = 0


class DistributedComputeManager:
    """Manage distributed computation across multiple devices/nodes."""

    def __init__(self, max_workers: Optional[int] = None):
        """Initialize distributed compute manager.
        
        Args:
            max_workers: Maximum number of worker processes
        """
        self.max_workers = max_workers or min(8, mp.cpu_count())
        self.devices = jax.devices()

        # Setup mesh for multi-device computation
        if len(self.devices) > 1:
            try:
                device_mesh = mesh_utils.create_device_mesh((len(self.devices),))
                self.device_mesh = device_mesh
                logger.info(f"Created device mesh with {len(self.devices)} devices")
            except Exception as e:
                logger.warning(f"Failed to create device mesh: {e}")
                self.device_mesh = None
        else:
            self.device_mesh = None

    def parallel_map(self, func: Callable, data_list: List[Any],
                    use_processes: bool = False) -> List[Any]:
        """Execute function in parallel across data items.
        
        Args:
            func: Function to execute
            data_list: List of data items
            use_processes: Use process pool instead of thread pool
            
        Returns:
            List of results
        """
        if len(data_list) <= 1:
            return [func(item) for item in data_list]

        executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
        max_workers = min(self.max_workers, len(data_list))

        logger.debug(f"Parallel execution with {max_workers} workers")

        with executor_class(max_workers=max_workers) as executor:
            futures = [executor.submit(func, item) for item in data_list]
            results = [future.result() for future in futures]

        return results

    def distributed_batch_process(self, func: Callable, data: Array,
                                 batch_size: int = 1000) -> Array:
        """Process large datasets using distributed batching.
        
        Args:
            func: Function to apply
            data: Input data array
            batch_size: Size of each batch
            
        Returns:
            Processed results
        """
        n_devices = len(self.devices)
        if n_devices <= 1:
            # Fallback to single device processing
            return vmap(func)(data)

        # Split data across devices
        device_batch_size = max(batch_size // n_devices, 1)

        # Create batches for each device
        device_batches = []
        for i in range(n_devices):
            start_idx = i * device_batch_size
            end_idx = min((i + 1) * device_batch_size, len(data))

            if start_idx < len(data):
                batch = data[start_idx:end_idx]
                device_batch = device_put(batch, self.devices[i])
                device_batches.append(device_batch)

        # Process batches in parallel using pmap
        try:
            if device_batches:
                # Pad batches to same size for pmap
                max_batch_size = max(len(batch) for batch in device_batches)

                padded_batches = []
                for batch in device_batches:
                    if len(batch) < max_batch_size:
                        pad_size = max_batch_size - len(batch)
                        padding = jnp.zeros((pad_size,) + batch.shape[1:], dtype=batch.dtype)
                        padded_batch = jnp.concatenate([batch, padding], axis=0)
                    else:
                        padded_batch = batch
                    padded_batches.append(padded_batch)

                # Stack for pmap
                stacked_batches = jnp.stack(padded_batches)

                # Apply function with pmap
                pmap_func = pmap(vmap(func))
                results = pmap_func(stacked_batches)

                # Unpad and concatenate results
                final_results = []
                for i, batch_result in enumerate(results):
                    original_size = len(device_batches[i])
                    final_results.append(batch_result[:original_size])

                return jnp.concatenate(final_results, axis=0)

        except Exception as e:
            logger.warning(f"Distributed processing failed: {e}")

        # Fallback to regular processing
        return vmap(func)(data)


class AdaptiveMemoryManager:
    """Intelligent memory management for large-scale computations."""

    def __init__(self, memory_limit_gb: float = 8.0):
        """Initialize memory manager.
        
        Args:
            memory_limit_gb: Memory limit in gigabytes
        """
        self.memory_limit_bytes = memory_limit_gb * 1024**3
        self.current_usage = 0
        self.allocation_history: List[Tuple[str, int, float]] = []

        logger.info(f"Memory manager initialized with {memory_limit_gb}GB limit")

    def estimate_memory_usage(self, data: Array) -> int:
        """Estimate memory usage of data array."""
        return data.nbytes if hasattr(data, "nbytes") else data.size * 8  # Assume float64

    def can_allocate(self, size_bytes: int) -> bool:
        """Check if allocation is within memory limits."""
        return self.current_usage + size_bytes <= self.memory_limit_bytes

    def allocate(self, data: Array, name: str = "unknown") -> Array:
        """Allocate memory for data with tracking."""
        size_bytes = self.estimate_memory_usage(data)

        if not self.can_allocate(size_bytes):
            logger.warning(f"Memory allocation for {name} ({size_bytes / 1024**2:.1f} MB) "
                         f"would exceed limit. Current usage: {self.current_usage / 1024**2:.1f} MB")
            # Try garbage collection
            import gc
            gc.collect()

            if not self.can_allocate(size_bytes):
                raise MemoryError(f"Cannot allocate {size_bytes / 1024**2:.1f} MB for {name}")

        self.current_usage += size_bytes
        self.allocation_history.append((name, size_bytes, time.time()))

        logger.debug(f"Allocated {size_bytes / 1024**2:.1f} MB for {name}")

        return data

    def deallocate(self, size_bytes: int, name: str = "unknown"):
        """Deallocate memory."""
        self.current_usage = max(0, self.current_usage - size_bytes)
        logger.debug(f"Deallocated {size_bytes / 1024**2:.1f} MB for {name}")

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        usage_mb = self.current_usage / 1024**2
        limit_mb = self.memory_limit_bytes / 1024**2
        usage_pct = (self.current_usage / self.memory_limit_bytes) * 100

        return {
            "current_usage_mb": usage_mb,
            "memory_limit_mb": limit_mb,
            "usage_percentage": usage_pct,
            "allocations_count": len(self.allocation_history),
        }


class ScalabilityOptimizer:
    """Optimize algorithms for scalability across different problem sizes."""

    def __init__(self):
        """Initialize scalability optimizer."""
        self.performance_optimizer = PerformanceOptimizer()
        self.cache_system = AdvancedCacheSystem()
        self.compute_manager = DistributedComputeManager()
        self.memory_manager = AdaptiveMemoryManager()

        # Scalability profiles
        self.algorithm_profiles: Dict[str, Dict] = {}

    def profile_algorithm(self, func: Callable, problem_sizes: List[int],
                         algorithm_name: str = "unknown") -> Dict[str, Any]:
        """Profile algorithm performance across different problem sizes.
        
        Args:
            func: Algorithm function
            problem_sizes: List of problem sizes to test
            algorithm_name: Name for profiling
            
        Returns:
            Performance profile
        """
        logger.info(f"Profiling algorithm {algorithm_name} across {len(problem_sizes)} sizes")

        profile_data = {
            "sizes": [],
            "execution_times": [],
            "memory_usage": [],
            "throughput": [],
            "scalability_factors": [],
        }

        for size in problem_sizes:
            try:
                # Generate test data
                key = random.PRNGKey(42)
                test_data = random.normal(key, (size, 10))  # 10D problem

                # Measure performance
                start_time = time.time()
                memory_before = self.memory_manager.current_usage

                # Execute algorithm
                result = func(test_data)

                execution_time = time.time() - start_time
                memory_after = self.memory_manager.current_usage
                memory_used = memory_after - memory_before

                throughput = size / execution_time if execution_time > 0 else 0

                profile_data["sizes"].append(size)
                profile_data["execution_times"].append(execution_time)
                profile_data["memory_usage"].append(memory_used)
                profile_data["throughput"].append(throughput)

                # Calculate scalability factor
                if len(profile_data["execution_times"]) > 1:
                    prev_time = profile_data["execution_times"][-2]
                    prev_size = profile_data["sizes"][-2]

                    expected_time = prev_time * (size / prev_size)
                    scalability_factor = expected_time / execution_time if execution_time > 0 else 1
                    profile_data["scalability_factors"].append(scalability_factor)
                else:
                    profile_data["scalability_factors"].append(1.0)

                logger.debug(f"Size {size}: {execution_time:.3f}s, "
                           f"{memory_used / 1024**2:.1f} MB, "
                           f"{throughput:.1f} samples/s")

            except Exception as e:
                logger.error(f"Profiling failed for size {size}: {e}")

        # Store profile
        self.algorithm_profiles[algorithm_name] = profile_data

        return profile_data

    def recommend_optimization(self, algorithm_name: str, target_size: int) -> Dict[str, Any]:
        """Recommend optimization strategy based on profiling data.
        
        Args:
            algorithm_name: Name of profiled algorithm
            target_size: Target problem size
            
        Returns:
            Optimization recommendations
        """
        if algorithm_name not in self.algorithm_profiles:
            return {"error": "Algorithm not profiled"}

        profile = self.algorithm_profiles[algorithm_name]

        # Find closest profiled size
        sizes = jnp.array(profile["sizes"])
        closest_idx = jnp.argmin(jnp.abs(sizes - target_size))
        closest_size = sizes[closest_idx]

        # Estimate performance at target size
        execution_times = jnp.array(profile["execution_times"])
        memory_usage = jnp.array(profile["memory_usage"])

        if len(execution_times) > 1:
            # Linear interpolation/extrapolation
            time_slope = (execution_times[-1] - execution_times[0]) / (sizes[-1] - sizes[0])
            estimated_time = execution_times[0] + time_slope * (target_size - sizes[0])

            memory_slope = (memory_usage[-1] - memory_usage[0]) / (sizes[-1] - sizes[0])
            estimated_memory = memory_usage[0] + memory_slope * (target_size - sizes[0])
        else:
            estimated_time = execution_times[0]
            estimated_memory = memory_usage[0]

        recommendations = {
            "estimated_execution_time": float(estimated_time),
            "estimated_memory_usage_mb": float(estimated_memory / 1024**2),
            "recommendations": [],
        }

        # Generate recommendations
        if estimated_memory > self.memory_manager.memory_limit_bytes * 0.8:
            recommendations["recommendations"].append("Use batch processing to reduce memory usage")
            recommendations["recommendations"].append("Consider distributed processing")

        if estimated_time > 60:  # > 1 minute
            recommendations["recommendations"].append("Enable GPU acceleration")
            recommendations["recommendations"].append("Use parallel processing")

        if target_size > 10000:
            recommendations["recommendations"].append("Enable advanced caching")
            recommendations["recommendations"].append("Use JIT compilation")

        # Check scalability
        avg_scalability = jnp.mean(jnp.array(profile["scalability_factors"]))
        if avg_scalability < 0.8:
            recommendations["recommendations"].append("Algorithm shows poor scalability")
            recommendations["recommendations"].append("Consider algorithmic improvements")

        return recommendations

    def optimize_for_scale(self, func: Callable, problem_size: int,
                          algorithm_name: str = "unknown") -> Callable:
        """Automatically optimize function for given problem size.
        
        Args:
            func: Function to optimize
            problem_size: Target problem size
            algorithm_name: Algorithm identifier
            
        Returns:
            Optimized function
        """
        # Get optimization recommendations
        recommendations = self.recommend_optimization(algorithm_name, problem_size)

        # Apply optimizations based on recommendations
        optimized_func = func

        if "Use batch processing" in str(recommendations):
            # Wrap with batch processing
            def batched_func(data):
                return self.performance_optimizer.batch_process(
                    func, data, batch_size=min(1000, problem_size // 10)
                )
            optimized_func = batched_func

        if "Enable GPU acceleration" in str(recommendations):
            # Apply GPU optimizations
            optimized_func = self.performance_optimizer.optimize_function(
                optimized_func, vectorize=True, parallel=True
            )

        if "Enable advanced caching" in str(recommendations):
            # Apply caching
            optimized_func = self.cache_system.cached_function()(optimized_func)

        return optimized_func


# High-level optimization interface
class ProductionOptimizer:
    """Production-ready optimization system combining all performance features."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize production optimizer.
        
        Args:
            config: Optimization configuration
        """
        self.config = config or {}

        # Initialize components
        self.performance_optimizer = PerformanceOptimizer(
            enable_gpu=self.config.get("enable_gpu", True),
            enable_jit=self.config.get("enable_jit", True),
            enable_caching=self.config.get("enable_caching", True),
        )

        self.cache_system = AdvancedCacheSystem(
            max_size=self.config.get("cache_size", 1000),
            ttl_seconds=self.config.get("cache_ttl", 3600),
        )

        self.compute_manager = DistributedComputeManager(
            max_workers=self.config.get("max_workers", None)
        )

        self.memory_manager = AdaptiveMemoryManager(
            memory_limit_gb=self.config.get("memory_limit_gb", 8.0)
        )

        self.scalability_optimizer = ScalabilityOptimizer()

        logger.info("Production optimizer initialized")

    def optimize_surrogate_training(self, training_func: Callable) -> Callable:
        """Optimize surrogate model training."""
        @wraps(training_func)
        def optimized_training(*args, **kwargs):
            # Apply memory management
            if len(args) > 0 and hasattr(args[0], "X"):
                dataset = args[0]
                self.memory_manager.allocate(dataset.X, "training_data_X")
                self.memory_manager.allocate(dataset.y, "training_data_y")

            # Apply performance optimizations
            optimized_func = self.performance_optimizer.optimize_function(
                training_func, vectorize=False, parallel=False, cache_key="training"
            )

            return optimized_func(*args, **kwargs)

        return optimized_training

    def optimize_prediction(self, prediction_func: Callable) -> Callable:
        """Optimize prediction function."""
        # Cache predictions
        cached_func = self.cache_system.cached_function("function")(prediction_func)

        # Apply performance optimizations
        optimized_func = self.performance_optimizer.optimize_function(
            cached_func, vectorize=True, parallel=True, cache_key="prediction"
        )

        return optimized_func

    def optimize_optimization(self, optimization_func: Callable) -> Callable:
        """Optimize the optimization process itself."""
        @wraps(optimization_func)
        def optimized_optimization(*args, **kwargs):
            # Use distributed processing for multiple starting points
            if "n_starts" in kwargs and kwargs["n_starts"] > 1:
                n_starts = kwargs.pop("n_starts")

                # Create multiple starting points
                key = random.PRNGKey(42)
                starts = [random.normal(random.fold_in(key, i), (2,))
                         for i in range(n_starts)]

                # Run optimization in parallel
                results = self.compute_manager.parallel_map(
                    lambda start: optimization_func(*args, initial_point=start, **kwargs),
                    starts
                )

                # Return best result
                best_result = min(results, key=lambda r: r.fun if hasattr(r, "fun") else float("inf"))
                return best_result
            return optimization_func(*args, **kwargs)

        return optimized_optimization

    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system performance statistics."""
        return {
            "performance_metrics": {
                "execution_time": self.performance_optimizer.metrics.execution_time,
                "throughput": self.performance_optimizer.metrics.throughput,
                "device_count": self.performance_optimizer.device_count,
                "gpu_enabled": self.performance_optimizer.enable_gpu,
            },
            "cache_stats": self.cache_system.get_cache_stats(),
            "memory_stats": self.memory_manager.get_memory_stats(),
            "compute_resources": {
                "max_workers": self.compute_manager.max_workers,
                "available_devices": len(self.compute_manager.devices),
            },
        }


# Example usage and testing
if __name__ == "__main__":
    print("🚀 ADVANCED PERFORMANCE OPTIMIZATION DEMONSTRATION")
    print("=" * 70)

    # Initialize production optimizer
    optimizer = ProductionOptimizer({
        "enable_gpu": True,
        "enable_jit": True,
        "cache_size": 500,
        "memory_limit_gb": 4.0,
    })

    print("\\n1. Performance Optimization Demo")

    # Simple test function
    def test_function(x):
        return jnp.sum(x**2)

    # Generate test data
    key = random.PRNGKey(42)
    test_data = random.normal(key, (1000, 10))

    # Benchmark original function
    start_time = time.time()
    results_original = vmap(test_function)(test_data)
    time_original = time.time() - start_time

    # Benchmark optimized function
    optimized_func = optimizer.optimize_prediction(test_function)

    start_time = time.time()
    results_optimized = optimized_func(test_data)
    time_optimized = time.time() - start_time

    speedup = time_original / time_optimized if time_optimized > 0 else 1

    print(f"Original time: {time_original:.4f}s")
    print(f"Optimized time: {time_optimized:.4f}s")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Results match: {jnp.allclose(results_original, results_optimized)}")

    print("\\n2. Cache Performance Demo")

    # Test caching
    cached_func = optimizer.cache_system.cached_function()(test_function)

    # First call (cache miss)
    start_time = time.time()
    result1 = cached_func(test_data[0])
    time_miss = time.time() - start_time

    # Second call (cache hit)
    start_time = time.time()
    result2 = cached_func(test_data[0])
    time_hit = time.time() - start_time

    cache_speedup = time_miss / time_hit if time_hit > 0 else float("inf")

    print(f"Cache miss time: {time_miss:.6f}s")
    print(f"Cache hit time: {time_hit:.6f}s")
    print(f"Cache speedup: {cache_speedup:.1f}x")

    cache_stats = optimizer.cache_system.get_cache_stats()
    print(f"Cache hit rate: {cache_stats['hit_rate']:.1%}")

    print("\\n3. Scalability Analysis")

    # Profile algorithm across different sizes
    sizes = [100, 500, 1000, 2000]
    profile = optimizer.scalability_optimizer.profile_algorithm(
        test_function, sizes, "test_algorithm"
    )

    print("Problem Size | Execution Time | Throughput")
    print("-" * 45)
    for i, size in enumerate(profile["sizes"]):
        exec_time = profile["execution_times"][i]
        throughput = profile["throughput"][i]
        print(f"{size:11d} | {exec_time:13.4f}s | {throughput:10.1f} samples/s")

    print("\\n4. System Statistics")

    system_stats = optimizer.get_system_stats()
    print("Performance Metrics:")
    for key, value in system_stats["performance_metrics"].items():
        print(f"  {key}: {value}")

    print("\\nMemory Statistics:")
    for key, value in system_stats["memory_stats"].items():
        print(f"  {key}: {value}")

    print("\\n" + "="*70)
    print("✅ PERFORMANCE OPTIMIZATION DEMONSTRATION COMPLETED")
    print("="*70)
