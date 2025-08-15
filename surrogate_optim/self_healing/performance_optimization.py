"""Advanced performance optimization for self-healing surrogate optimization."""

import time
import threading
import functools
import gc
from typing import Dict, List, Optional, Callable, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import cProfile
import pstats
import io
from contextlib import contextmanager
import weakref

import jax
import jax.numpy as jnp
import numpy as np
from loguru import logger

# Performance profiling imports
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import line_profiler
    LINE_PROFILER_AVAILABLE = True
except ImportError:
    LINE_PROFILER_AVAILABLE = False

try:
    import memory_profiler
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False


class OptimizationLevel(Enum):
    """Performance optimization levels."""
    MINIMAL = "minimal"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    MAXIMUM = "maximum"


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""
    optimization_level: OptimizationLevel = OptimizationLevel.BALANCED
    enable_jit: bool = True
    enable_gpu: bool = True
    memory_management: bool = True
    cache_size_mb: int = 512
    prefetch_batches: int = 2
    gc_threshold: float = 0.8  # Memory threshold for GC
    profile_execution: bool = False
    enable_vectorization: bool = True
    parallel_workers: int = 4


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking."""
    execution_time: float
    memory_peak_mb: float
    memory_current_mb: float
    cpu_usage_percent: float
    gpu_usage_percent: float
    cache_hit_ratio: float
    compilation_time: float = 0.0
    function_calls: int = 0
    vectorization_ratio: float = 0.0


class AdvancedCache:
    """High-performance caching system with intelligent eviction."""
    
    def __init__(self, max_size_mb: int = 512):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self._cache: Dict[str, Any] = {}
        self._access_times: Dict[str, float] = {}
        self._access_counts: Dict[str, int] = {}
        self._sizes: Dict[str, int] = {}
        self._current_size = 0
        self._hits = 0
        self._misses = 0
        self._lock = threading.RLock()
        
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self._lock:
            if key in self._cache:
                self._access_times[key] = time.time()
                self._access_counts[key] = self._access_counts.get(key, 0) + 1
                self._hits += 1
                return self._cache[key]
            else:
                self._misses += 1
                return None
                
    def put(self, key: str, value: Any) -> None:
        """Put item in cache with intelligent eviction."""
        with self._lock:
            # Estimate size
            try:
                import sys
                size = sys.getsizeof(value)
                if hasattr(value, '__len__'):
                    size += len(value) * 8  # Rough estimate for arrays
            except Exception:
                size = 1024  # Default estimate
                
            # Check if we need to evict
            while self._current_size + size > self.max_size_bytes and self._cache:
                self._evict_lru()
                
            # Add to cache
            if key in self._cache:
                self._current_size -= self._sizes[key]
                
            self._cache[key] = value
            self._access_times[key] = time.time()
            self._access_counts[key] = self._access_counts.get(key, 0) + 1
            self._sizes[key] = size
            self._current_size += size
            
    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if not self._cache:
            return
            
        # Find LRU item with lowest access count as tiebreaker
        lru_key = min(
            self._cache.keys(),
            key=lambda k: (self._access_times[k], self._access_counts[k])
        )
        
        # Remove item
        del self._cache[lru_key]
        self._current_size -= self._sizes[lru_key]
        del self._access_times[lru_key]
        del self._access_counts[lru_key]
        del self._sizes[lru_key]
        
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            self._access_counts.clear()
            self._sizes.clear()
            self._current_size = 0
            
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._hits + self._misses
        hit_ratio = self._hits / max(1, total_requests)
        
        return {
            "size_mb": self._current_size / 1024 / 1024,
            "entries": len(self._cache),
            "hit_ratio": hit_ratio,
            "hits": self._hits,
            "misses": self._misses
        }


class JaxOptimizer:
    """JAX-specific performance optimizations."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self._compiled_functions: Dict[str, Callable] = {}
        self._compilation_cache = AdvancedCache(128)  # Smaller cache for compiled functions
        
        self._configure_jax()
        
    def _configure_jax(self) -> None:
        """Configure JAX for optimal performance."""
        if self.config.enable_jit:
            jax.config.update("jax_enable_x64", True)
            
        if self.config.enable_gpu:
            # Configure GPU memory growth
            jax.config.update("jax_memory_fraction", 0.8)
            
        # Configure compilation cache
        jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
        
        # Enable aggressive optimizations for maximum performance
        if self.config.optimization_level == OptimizationLevel.MAXIMUM:
            jax.config.update("jax_debug_nans", False)
            jax.config.update("jax_debug_infs", False)
            
    @functools.lru_cache(maxsize=128)
    def get_optimized_function(self, func_hash: str, func: Callable) -> Callable:
        """Get JIT-compiled version of function with caching."""
        if not self.config.enable_jit:
            return func
            
        compiled_key = f"jit_{func_hash}"
        
        if compiled_key in self._compiled_functions:
            return self._compiled_functions[compiled_key]
            
        # Compile function
        start_time = time.time()
        
        try:
            compiled_func = jax.jit(func)
            compilation_time = time.time() - start_time
            
            self._compiled_functions[compiled_key] = compiled_func
            
            logger.debug(f"Compiled function {func.__name__} in {compilation_time:.3f}s")
            return compiled_func
            
        except Exception as e:
            logger.warning(f"Failed to compile function {func.__name__}: {e}")
            return func
            
    def vectorize_function(self, func: Callable, batch_size: int = 32) -> Callable:
        """Create vectorized version of function."""
        if not self.config.enable_vectorization:
            return func
            
        @functools.wraps(func)
        def vectorized_func(inputs):
            if isinstance(inputs, (list, tuple)):
                # Batch process inputs
                results = []
                for i in range(0, len(inputs), batch_size):
                    batch = inputs[i:i + batch_size]
                    batch_array = jnp.array(batch)
                    batch_results = jax.vmap(func)(batch_array)
                    results.extend(batch_results.tolist())
                return results
            else:
                return func(inputs)
                
        return vectorized_func
        
    def create_memory_efficient_function(self, func: Callable) -> Callable:
        """Create memory-efficient version using gradient checkpointing."""
        if self.config.optimization_level in [OptimizationLevel.AGGRESSIVE, OptimizationLevel.MAXIMUM]:
            return jax.checkpoint(func)
        return func


class MemoryManager:
    """Advanced memory management for optimization workloads."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self._memory_pools: Dict[str, List[np.ndarray]] = {}
        self._allocation_stats = {"allocated": 0, "deallocated": 0, "reused": 0}
        self._weak_refs: List[weakref.ref] = []
        
    @contextmanager
    def memory_pool(self, pool_name: str):
        """Context manager for memory pool usage."""
        if pool_name not in self._memory_pools:
            self._memory_pools[pool_name] = []
            
        try:
            yield self._memory_pools[pool_name]
        finally:
            # Clean up pool if memory threshold exceeded
            if self._get_memory_usage() > self.config.gc_threshold:
                self._cleanup_pool(pool_name)
                
    def get_array(self, shape: Tuple[int, ...], dtype: np.dtype = np.float32, pool: str = "default") -> np.ndarray:
        """Get array from memory pool or allocate new."""
        if pool in self._memory_pools:
            # Try to reuse existing array
            for i, arr in enumerate(self._memory_pools[pool]):
                if arr.shape == shape and arr.dtype == dtype:
                    # Remove from pool and return
                    self._memory_pools[pool].pop(i)
                    self._allocation_stats["reused"] += 1
                    return arr
                    
        # Allocate new array
        arr = np.empty(shape, dtype=dtype)
        self._allocation_stats["allocated"] += 1
        return arr
        
    def return_array(self, arr: np.ndarray, pool: str = "default") -> None:
        """Return array to memory pool for reuse."""
        if pool not in self._memory_pools:
            self._memory_pools[pool] = []
            
        # Add to pool if not too large
        if len(self._memory_pools[pool]) < 100:  # Limit pool size
            self._memory_pools[pool].append(arr)
        else:
            self._allocation_stats["deallocated"] += 1
            
    def _cleanup_pool(self, pool_name: str) -> None:
        """Clean up memory pool."""
        if pool_name in self._memory_pools:
            count = len(self._memory_pools[pool_name])
            self._memory_pools[pool_name].clear()
            self._allocation_stats["deallocated"] += count
            
            # Force garbage collection
            gc.collect()
            
    def _get_memory_usage(self) -> float:
        """Get current memory usage as fraction of total."""
        if not PSUTIL_AVAILABLE:
            return 0.5  # Conservative estimate
            
        try:
            memory = psutil.virtual_memory()
            return memory.percent / 100.0
        except Exception:
            return 0.5
            
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory management statistics."""
        return {
            "allocation_stats": self._allocation_stats.copy(),
            "pool_sizes": {name: len(pool) for name, pool in self._memory_pools.items()},
            "current_usage": self._get_memory_usage()
        }


class PerformanceProfiler:
    """Comprehensive performance profiling system."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self._profiles: Dict[str, Any] = {}
        self._current_profile: Optional[cProfile.Profile] = None
        
    @contextmanager
    def profile_execution(self, operation_name: str):
        """Context manager for performance profiling."""
        if not self.config.profile_execution:
            yield
            return
            
        # Start profiling
        profiler = cProfile.Profile()
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        profiler.enable()
        
        try:
            yield
        finally:
            profiler.disable()
            
            # Calculate metrics
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            # Store profile
            self._profiles[operation_name] = {
                "profiler": profiler,
                "execution_time": end_time - start_time,
                "memory_delta": end_memory - start_memory,
                "timestamp": time.time()
            }
            
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if not PSUTIL_AVAILABLE:
            return 0.0
            
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0
            
    def get_profile_report(self, operation_name: str) -> Optional[str]:
        """Get profile report for operation."""
        if operation_name not in self._profiles:
            return None
            
        profile_data = self._profiles[operation_name]
        profiler = profile_data["profiler"]
        
        # Generate report
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(20)  # Top 20 functions
        
        report = s.getvalue()
        
        # Add summary
        summary = f"""
Performance Summary for {operation_name}:
- Execution Time: {profile_data['execution_time']:.3f}s
- Memory Delta: {profile_data['memory_delta']:.1f}MB
- Timestamp: {profile_data['timestamp']}

Top Functions:
{report}
        """
        
        return summary
        
    def clear_profiles(self) -> None:
        """Clear all stored profiles."""
        self._profiles.clear()


class PerformanceOptimizer:
    """Main performance optimization orchestrator."""
    
    def __init__(self, config: Optional[PerformanceConfig] = None):
        self.config = config or PerformanceConfig()
        
        # Components
        self.cache = AdvancedCache(self.config.cache_size_mb)
        self.jax_optimizer = JaxOptimizer(self.config)
        self.memory_manager = MemoryManager(self.config)
        self.profiler = PerformanceProfiler(self.config)
        
        # Performance tracking
        self._optimization_metrics: Dict[str, PerformanceMetrics] = {}
        
        logger.info(f"Performance optimizer initialized with {self.config.optimization_level.value} level")
        
    def optimize_function(self, func: Callable, func_name: Optional[str] = None) -> Callable:
        """Apply comprehensive optimizations to function."""
        if func_name is None:
            func_name = func.__name__
            
        # Apply JAX optimizations
        func_hash = hash(func.__code__.co_code)
        optimized_func = self.jax_optimizer.get_optimized_function(str(func_hash), func)
        
        # Add caching if appropriate
        if self.config.optimization_level in [OptimizationLevel.AGGRESSIVE, OptimizationLevel.MAXIMUM]:
            optimized_func = self._add_intelligent_caching(optimized_func, func_name)
            
        # Add vectorization
        if self.config.enable_vectorization:
            optimized_func = self.jax_optimizer.vectorize_function(optimized_func)
            
        # Add memory efficiency
        optimized_func = self.jax_optimizer.create_memory_efficient_function(optimized_func)
        
        # Add performance monitoring
        optimized_func = self._add_performance_monitoring(optimized_func, func_name)
        
        return optimized_func
        
    def _add_intelligent_caching(self, func: Callable, func_name: str) -> Callable:
        """Add intelligent caching to function."""
        
        @functools.wraps(func)
        def cached_func(*args, **kwargs):
            # Create cache key
            key = f"{func_name}_{hash((args, tuple(sorted(kwargs.items()))))}"
            
            # Try cache first
            result = self.cache.get(key)
            if result is not None:
                return result
                
            # Compute and cache
            result = func(*args, **kwargs)
            self.cache.put(key, result)
            
            return result
            
        return cached_func
        
    def _add_performance_monitoring(self, func: Callable, func_name: str) -> Callable:
        """Add performance monitoring to function."""
        
        @functools.wraps(func)
        def monitored_func(*args, **kwargs):
            start_time = time.time()
            start_memory = self._get_memory_usage()
            
            try:
                with self.profiler.profile_execution(func_name):
                    result = func(*args, **kwargs)
                    
                # Record metrics
                end_time = time.time()
                end_memory = self._get_memory_usage()
                
                metrics = PerformanceMetrics(
                    execution_time=end_time - start_time,
                    memory_peak_mb=max(start_memory, end_memory),
                    memory_current_mb=end_memory,
                    cpu_usage_percent=self._get_cpu_usage(),
                    gpu_usage_percent=self._get_gpu_usage(),
                    cache_hit_ratio=self.cache.get_stats()["hit_ratio"],
                    function_calls=1
                )
                
                self._optimization_metrics[func_name] = metrics
                
                return result
                
            except Exception as e:
                logger.error(f"Performance monitoring error in {func_name}: {e}")
                raise
                
        return monitored_func
        
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if not PSUTIL_AVAILABLE:
            return 0.0
            
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0
            
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        if not PSUTIL_AVAILABLE:
            return 0.0
            
        try:
            return psutil.cpu_percent()
        except Exception:
            return 0.0
            
    def _get_gpu_usage(self) -> float:
        """Get current GPU usage percentage."""
        # Simplified GPU usage - would need specific GPU monitoring
        return 0.0
        
    @contextmanager
    def optimized_context(self, context_name: str = "optimization"):
        """Context manager for optimized execution environment."""
        
        # Pre-optimization setup
        if self.config.memory_management:
            gc.disable()  # Disable GC during optimization
            
        # Configure JAX for this context
        original_precision = jax.config.jax_enable_x64
        if self.config.optimization_level == OptimizationLevel.MAXIMUM:
            jax.config.update("jax_enable_x64", True)
            
        try:
            with self.memory_manager.memory_pool(context_name):
                yield
        finally:
            # Cleanup
            if self.config.memory_management:
                gc.enable()
                gc.collect()
                
            # Restore JAX config
            jax.config.update("jax_enable_x64", original_precision)
            
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        return {
            "configuration": {
                "optimization_level": self.config.optimization_level.value,
                "jit_enabled": self.config.enable_jit,
                "gpu_enabled": self.config.enable_gpu,
                "vectorization_enabled": self.config.enable_vectorization
            },
            "cache_stats": self.cache.get_stats(),
            "memory_stats": self.memory_manager.get_memory_stats(),
            "function_metrics": {
                name: {
                    "execution_time": metrics.execution_time,
                    "memory_peak_mb": metrics.memory_peak_mb,
                    "cache_hit_ratio": metrics.cache_hit_ratio,
                    "function_calls": metrics.function_calls
                }
                for name, metrics in self._optimization_metrics.items()
            }
        }
        
    def clear_caches(self) -> None:
        """Clear all performance caches."""
        self.cache.clear()
        jax.clear_caches()
        gc.collect()
        
    def tune_performance(self, workload_characteristics: Dict[str, Any]) -> None:
        """Auto-tune performance based on workload characteristics."""
        
        # Adjust configuration based on workload
        if workload_characteristics.get("memory_intensive", False):
            self.config.gc_threshold = 0.7  # More aggressive GC
            self.config.cache_size_mb = min(256, self.config.cache_size_mb)
            
        if workload_characteristics.get("compute_intensive", False):
            self.config.enable_jit = True
            self.config.optimization_level = OptimizationLevel.AGGRESSIVE
            
        if workload_characteristics.get("batch_size", 1) > 100:
            self.config.enable_vectorization = True
            self.config.parallel_workers = min(8, self.config.parallel_workers * 2)
            
        logger.info(f"Performance tuned for workload: {workload_characteristics}")


# Global performance optimizer instance
global_performance_optimizer = PerformanceOptimizer()

# Convenience decorators
def optimize_performance(func_name: Optional[str] = None, config: Optional[PerformanceConfig] = None):
    """Decorator for automatic performance optimization."""
    def decorator(func: Callable) -> Callable:
        optimizer = PerformanceOptimizer(config) if config else global_performance_optimizer
        return optimizer.optimize_function(func, func_name)
    return decorator

def profile_performance(operation_name: Optional[str] = None):
    """Decorator for performance profiling."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = operation_name or func.__name__
            with global_performance_optimizer.profiler.profile_execution(name):
                return func(*args, **kwargs)
        return wrapper
    return decorator