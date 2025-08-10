"""Research-grade parallel processing for high-performance surrogate optimization."""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Iterator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed, Future
from multiprocessing import cpu_count, shared_memory
import threading
import time
import os
import psutil
import logging
from dataclasses import dataclass, field
import numpy as np
from pathlib import Path
import pickle
import hashlib
import weakref
import gc

import jax
import jax.numpy as jnp
from jax import Array, vmap, pmap, jit, devices
import optax

from ..monitoring.logging import get_logger

logger = get_logger()


@dataclass
class ResourceMetrics:
    """Resource usage metrics for performance monitoring."""
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    gpu_memory_mb: float = 0.0
    execution_time: float = 0.0
    throughput: float = 0.0  # operations per second


@dataclass
class ResearchParallelConfig:
    """Research-grade configuration for parallel processing."""
    max_workers: Optional[int] = None
    use_threads: bool = True
    chunk_size: int = 100
    timeout: Optional[float] = None
    memory_limit_gb: float = 8.0
    enable_gpu: bool = False
    cache_results: bool = True
    adaptive_batching: bool = True
    resource_monitoring: bool = True
    optimization_level: str = "balanced"  # "speed", "memory", "balanced"
    
    def __post_init__(self):
        if self.max_workers is None:
            self.max_workers = min(cpu_count(), 16)  # Reasonable upper bound
        
        # Auto-detect GPU availability
        if self.enable_gpu:
            try:
                gpu_devices = [d for d in jax.devices() if d.device_kind == 'gpu']
                self.enable_gpu = len(gpu_devices) > 0
                logger.info(f"GPU acceleration: {'enabled' if self.enable_gpu else 'disabled'}")
            except:
                self.enable_gpu = False
                logger.info("GPU acceleration: disabled (JAX not available)")


class ResourceMonitor:
    """Monitor and manage computational resources."""
    
    def __init__(self, config: ResearchParallelConfig):
        self.config = config
        self.metrics_history = []
        self.start_time = None
        self.process = psutil.Process()
        
    def start_monitoring(self):
        """Start resource monitoring."""
        self.start_time = time.time()
        
    def stop_monitoring(self) -> ResourceMetrics:
        """Stop monitoring and return metrics."""
        end_time = time.time()
        execution_time = end_time - self.start_time if self.start_time else 0.0
        
        # CPU and memory metrics
        cpu_percent = self.process.cpu_percent()
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        memory_percent = self.process.memory_percent()
        
        # GPU memory (if available)
        gpu_memory_mb = 0.0
        if self.config.enable_gpu:
            try:
                # JAX GPU memory usage (simplified)
                gpu_memory_mb = sum(d.memory_stats()['bytes_in_use'] 
                                  for d in jax.devices() 
                                  if d.device_kind == 'gpu') / 1024 / 1024
            except:
                pass
        
        metrics = ResourceMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_mb=memory_mb,
            gpu_memory_mb=gpu_memory_mb,
            execution_time=execution_time,
            throughput=0.0  # Will be set by caller
        )
        
        self.metrics_history.append(metrics)
        return metrics


class HighPerformanceParallelEvaluator:
    """High-performance parallel function evaluation with GPU acceleration."""
    
    def __init__(self, config: ResearchParallelConfig):
        self.config = config
        self.jax_devices = jax.devices()
        self.gpu_devices = [d for d in self.jax_devices if d.device_kind == 'gpu']
        self.cpu_devices = [d for d in self.jax_devices if d.device_kind == 'cpu']
        
        # Performance cache
        self.evaluation_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Setup JIT compiled functions
        self._setup_compiled_functions()
    
    def _setup_compiled_functions(self):
        """Setup JIT-compiled functions for maximum performance."""
        
        @jit
        def batch_evaluate_jit(func, x_batch):
            """JIT-compiled batch evaluation."""
            return vmap(func)(x_batch)
        
        @jit 
        def batch_gradient_jit(func, x_batch):
            """JIT-compiled batch gradient computation."""
            grad_func = jax.grad(func)
            return vmap(grad_func)(x_batch)
        
        self.batch_evaluate_jit = batch_evaluate_jit
        self.batch_gradient_jit = batch_gradient_jit
    
    def _cache_key(self, x: Array) -> str:
        """Generate cache key for function evaluation."""
        if not self.config.cache_results:
            return None
        return hashlib.md5(x.tobytes()).hexdigest()[:16]
    
    def evaluate_parallel(
        self,
        function: Callable[[Array], float],
        points: Array,
        estimate_gradients: bool = False,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Array]:
        """High-performance parallel evaluation with caching and GPU acceleration."""
        
        n_points = points.shape[0]
        monitor = ResourceMonitor(self.config) if self.config.resource_monitoring else None
        
        if monitor:
            monitor.start_monitoring()
        
        logger.info(f"Evaluating {n_points} points with {self.config.optimization_level} optimization")
        
        # Check cache first
        cached_results = {}
        uncached_points = []
        uncached_indices = []
        
        for i, point in enumerate(points):
            cache_key = self._cache_key(point)
            if cache_key and cache_key in self.evaluation_cache:
                cached_results[i] = self.evaluation_cache[cache_key]
                self.cache_hits += 1
            else:
                uncached_points.append(point)
                uncached_indices.append(i)
                self.cache_misses += 1
        
        if cached_results:
            cache_ratio = len(cached_results) / n_points
            logger.info(f"Cache hit ratio: {cache_ratio:.1%} ({len(cached_results)}/{n_points})")
        
        # Evaluate uncached points
        if uncached_points:
            uncached_array = jnp.stack(uncached_points)
            
            # Choose evaluation strategy based on config and hardware
            if self.config.enable_gpu and self.gpu_devices and len(uncached_points) > 100:
                new_results = self._evaluate_gpu_accelerated(function, uncached_array, estimate_gradients)
            elif len(uncached_points) > self.config.max_workers * 10:
                new_results = self._evaluate_distributed(function, uncached_array, estimate_gradients)
            else:
                new_results = self._evaluate_vectorized(function, uncached_array, estimate_gradients)
            
            # Cache new results
            for i, point in enumerate(uncached_points):
                cache_key = self._cache_key(point)
                if cache_key:
                    self.evaluation_cache[cache_key] = {
                        'values': new_results['values'][i],
                        'gradients': new_results['gradients'][i] if estimate_gradients else None
                    }
        else:
            new_results = {'values': jnp.array([]), 'gradients': jnp.array([])}
        
        # Combine cached and new results
        all_values = jnp.zeros(n_points)
        all_gradients = jnp.zeros((n_points, points.shape[1])) if estimate_gradients else None
        
        # Fill cached results
        for i, result in cached_results.items():
            all_values = all_values.at[i].set(result['values'])
            if estimate_gradients and result['gradients'] is not None:
                all_gradients = all_gradients.at[i].set(result['gradients'])
        
        # Fill new results
        for i, uncached_idx in enumerate(uncached_indices):
            all_values = all_values.at[uncached_idx].set(new_results['values'][i])
            if estimate_gradients:
                all_gradients = all_gradients.at[uncached_idx].set(new_results['gradients'][i])
        
        # Resource metrics
        if monitor:
            final_metrics = monitor.stop_monitoring()
            final_metrics.throughput = n_points / max(final_metrics.execution_time, 1e-6)
            logger.info(f"Evaluation completed: {final_metrics.throughput:.1f} evals/sec, "
                       f"{final_metrics.execution_time:.2f}s total")
        
        result = {
            'values': all_values,
            'gradients': all_gradients,
            'cache_stats': {
                'hits': self.cache_hits,
                'misses': self.cache_misses,
                'hit_ratio': self.cache_hits / max(self.cache_hits + self.cache_misses, 1)
            },
            'resource_metrics': final_metrics if monitor else None
        }
        
        return result
    
    def _evaluate_gpu_accelerated(
        self,
        function: Callable,
        points: Array,
        estimate_gradients: bool
    ) -> Dict[str, Array]:
        """GPU-accelerated evaluation using JAX pmap."""
        
        logger.info(f"Using GPU acceleration with {len(self.gpu_devices)} devices")
        
        # Distribute points across GPU devices
        n_devices = len(self.gpu_devices)
        points_per_device = (points.shape[0] + n_devices - 1) // n_devices
        
        # Reshape for pmap
        padded_size = points_per_device * n_devices
        if points.shape[0] < padded_size:
            # Pad with last point
            padding = jnp.tile(points[-1:], (padded_size - points.shape[0], 1))
            padded_points = jnp.concatenate([points, padding])
        else:
            padded_points = points[:padded_size]
        
        # Reshape: [n_devices, points_per_device, n_dims]
        device_points = padded_points.reshape(n_devices, points_per_device, -1)
        
        # Define parallel evaluation function
        @jit
        def evaluate_on_device(device_batch):
            """Evaluate function on a single device."""
            return vmap(function)(device_batch)
        
        # Parallel evaluation across GPUs
        parallel_evaluate = pmap(evaluate_on_device, devices=self.gpu_devices)
        device_results = parallel_evaluate(device_points)
        
        # Flatten and trim results
        flat_values = device_results.reshape(-1)[:points.shape[0]]
        
        # Gradients if requested
        gradients = None
        if estimate_gradients:
            @jit
            def gradient_on_device(device_batch):
                grad_fn = jax.grad(function)
                return vmap(grad_fn)(device_batch)
            
            parallel_gradient = pmap(gradient_on_device, devices=self.gpu_devices)
            device_gradients = parallel_gradient(device_points)
            gradients = device_gradients.reshape(-1, points.shape[1])[:points.shape[0]]
        
        return {'values': flat_values, 'gradients': gradients}
    
    def _evaluate_distributed(
        self,
        function: Callable,
        points: Array,
        estimate_gradients: bool
    ) -> Dict[str, Array]:
        """Distributed evaluation using multiple processes."""
        
        logger.info(f"Using distributed evaluation with {self.config.max_workers} workers")
        
        def evaluate_batch(batch_data):
            batch_points, estimate_grads = batch_data
            values = jnp.array([function(x) for x in batch_points])
            gradients = None
            
            if estimate_grads:
                gradients = []
                for x in batch_points:
                    # Finite difference gradients
                    grad = jnp.zeros(len(x))
                    eps = 1e-6
                    
                    for j in range(len(x)):
                        x_plus = x.at[j].add(eps)
                        x_minus = x.at[j].add(-eps)
                        grad_j = (function(x_plus) - function(x_minus)) / (2 * eps)
                        grad = grad.at[j].set(grad_j)
                    
                    gradients.append(grad)
                
                gradients = jnp.stack(gradients) if gradients else None
            
            return values, gradients
        
        # Split into batches
        batch_size = max(1, points.shape[0] // (self.config.max_workers * 2))
        batches = []
        
        for i in range(0, points.shape[0], batch_size):
            batch_points = points[i:i+batch_size]
            batches.append((batch_points, estimate_gradients))
        
        # Process batches in parallel
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_batch = {
                executor.submit(evaluate_batch, batch): i
                for i, batch in enumerate(batches)
            }
            
            batch_results = [None] * len(batches)
            for future in as_completed(future_to_batch, timeout=self.config.timeout):
                batch_idx = future_to_batch[future]
                try:
                    batch_results[batch_idx] = future.result()
                except Exception as e:
                    logger.error(f"Batch {batch_idx} failed: {e}")
                    # Fallback to NaN
                    batch_size = len(batches[batch_idx][0])
                    batch_results[batch_idx] = (
                        jnp.full(batch_size, jnp.nan),
                        jnp.full((batch_size, points.shape[1]), jnp.nan) if estimate_gradients else None
                    )
        
        # Combine results
        all_values = []
        all_gradients = [] if estimate_gradients else None
        
        for values, gradients in batch_results:
            all_values.append(values)
            if estimate_gradients:
                all_gradients.append(gradients)
        
        combined_values = jnp.concatenate(all_values)
        combined_gradients = jnp.concatenate(all_gradients) if estimate_gradients else None
        
        return {'values': combined_values, 'gradients': combined_gradients}
    
    def _evaluate_vectorized(
        self,
        function: Callable,
        points: Array,
        estimate_gradients: bool
    ) -> Dict[str, Array]:
        """Vectorized evaluation using JAX vmap."""
        
        logger.info("Using vectorized evaluation")
        
        try:
            # Try JAX vectorization
            values = self.batch_evaluate_jit(function, points)
            
            gradients = None
            if estimate_gradients:
                gradients = self.batch_gradient_jit(function, points)
            
        except Exception as e:
            logger.warning(f"JAX vectorization failed, falling back to sequential: {e}")
            # Sequential fallback
            values = jnp.array([function(x) for x in points])
            
            gradients = None
            if estimate_gradients:
                gradients = []
                for x in points:
                    grad_fn = jax.grad(function)
                    try:
                        grad = grad_fn(x)
                    except:
                        # Finite differences fallback
                        grad = jnp.zeros(len(x))
                        eps = 1e-6
                        for j in range(len(x)):
                            x_plus = x.at[j].add(eps)
                            x_minus = x.at[j].add(-eps)
                            grad_j = (function(x_plus) - function(x_minus)) / (2 * eps)
                            grad = grad.at[j].set(grad_j)
                    gradients.append(grad)
                gradients = jnp.stack(gradients)
        
        return {'values': values, 'gradients': gradients}
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_queries = self.cache_hits + self.cache_misses
        return {
            'cache_size': len(self.evaluation_cache),
            'total_queries': total_queries,
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'hit_ratio': self.cache_hits / max(total_queries, 1),
            'memory_usage_mb': sum(len(pickle.dumps(v)) for v in self.evaluation_cache.values()) / 1024 / 1024
        }
    
    def clear_cache(self):
        """Clear evaluation cache."""
        self.evaluation_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        gc.collect()


class AdaptiveMemoryManager:
    """Adaptive memory management for large-scale research applications."""
    
    def __init__(self, config: ResearchParallelConfig):
        self.config = config
        self.memory_limit_bytes = config.memory_limit_gb * 1024**3
        self.current_usage = 0
        self.allocation_history = []
        
    def estimate_memory_requirement(
        self,
        operation: str,
        data_shape: Tuple,
        dtype=jnp.float32
    ) -> Dict[str, float]:
        """Estimate memory requirements for operations."""
        element_size = jnp.dtype(dtype).itemsize
        data_size_bytes = np.prod(data_shape) * element_size
        
        # Operation-specific multipliers based on research experience
        multipliers = {
            "function_evaluation": 2.0,   # Input + output + overhead
            "gradient_computation": 4.0,  # Input + gradients + jacobian + overhead  
            "surrogate_training": 8.0,    # Data + model + gradients + optimizer state
            "ensemble_prediction": 3.0,   # Input + multiple model outputs
            "optimization": 6.0,          # Candidates + gradients + search history
            "research_experiment": 10.0   # Full experimental pipeline
        }
        
        multiplier = multipliers.get(operation, 3.0)
        estimated_bytes = data_size_bytes * multiplier
        
        return {
            "data_size_gb": data_size_bytes / 1024**3,
            "estimated_total_gb": estimated_bytes / 1024**3,
            "fits_in_memory": estimated_bytes <= self.memory_limit_bytes * 0.8,  # Safety margin
            "recommended_chunks": max(1, int(estimated_bytes / (self.memory_limit_bytes * 0.6)) + 1)
        }
    
    def adaptive_chunk_processing(
        self,
        data: Array,
        process_func: Callable[[Array], Array],
        operation_name: str,
        progress_callback: Optional[Callable] = None
    ) -> Array:
        """Process data with adaptive chunking based on memory pressure."""
        
        # Estimate memory requirements
        memory_info = self.estimate_memory_requirement(operation_name, data.shape)
        
        if memory_info["fits_in_memory"]:
            logger.info(f"Processing {data.shape[0]} samples in single batch")
            return process_func(data)
        
        # Adaptive chunking
        n_chunks = memory_info["recommended_chunks"]
        chunk_size = data.shape[0] // n_chunks
        
        logger.info(f"Processing {data.shape[0]} samples in {n_chunks} adaptive chunks")
        
        results = []
        for i in range(0, data.shape[0], chunk_size):
            chunk_end = min(i + chunk_size, data.shape[0])
            chunk = data[i:chunk_end]
            
            # Monitor memory before processing
            memory_before = psutil.Process().memory_info().rss
            
            # Process chunk
            chunk_result = process_func(chunk)
            results.append(chunk_result)
            
            # Monitor memory after processing and adapt
            memory_after = psutil.Process().memory_info().rss
            memory_used = (memory_after - memory_before) / 1024 / 1024  # MB
            
            # Adaptive adjustment for next chunk
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > 80:
                chunk_size = max(chunk_size // 2, 10)
                logger.warning(f"High memory usage ({memory_percent:.1f}%), reducing chunk size to {chunk_size}")
            elif memory_percent < 30 and chunk_size < data.shape[0] // 2:
                chunk_size = min(chunk_size * 2, data.shape[0] // n_chunks)
            
            # Progress callback
            if progress_callback:
                progress = chunk_end / data.shape[0]
                progress_callback(progress)
            
            # Force garbage collection periodically
            if i % (chunk_size * 5) == 0:
                gc.collect()
        
        return jnp.concatenate(results, axis=0) if results else jnp.array([])


class ResearchBenchmarkRunner:
    """High-performance benchmark runner for research experiments."""
    
    def __init__(self, config: ResearchParallelConfig):
        self.config = config
        self.evaluator = HighPerformanceParallelEvaluator(config)
        self.memory_manager = AdaptiveMemoryManager(config)
        
    def run_research_benchmark(
        self,
        experiments: List[Dict[str, Any]],
        parallel_experiments: bool = True
    ) -> Dict[str, Any]:
        """Run comprehensive research benchmark with parallel execution."""
        
        logger.info(f"Starting research benchmark with {len(experiments)} experiments")
        
        monitor = ResourceMonitor(self.config) if self.config.resource_monitoring else None
        if monitor:
            monitor.start_monitoring()
        
        if parallel_experiments and len(experiments) > 1:
            results = self._run_experiments_parallel(experiments)
        else:
            results = self._run_experiments_sequential(experiments)
        
        # Final metrics
        if monitor:
            final_metrics = monitor.stop_monitoring()
            final_metrics.throughput = len(experiments) / max(final_metrics.execution_time, 1e-6)
            results['benchmark_metrics'] = final_metrics
        
        # Cache statistics
        results['cache_statistics'] = self.evaluator.get_cache_statistics()
        
        logger.info(f"Research benchmark completed: {len(experiments)} experiments")
        
        return results
    
    def _run_experiments_parallel(self, experiments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run experiments in parallel."""
        
        def run_single_experiment(exp_data):
            exp_id, experiment = exp_data
            try:
                return exp_id, self._execute_single_experiment(experiment)
            except Exception as e:
                logger.error(f"Experiment {exp_id} failed: {e}")
                return exp_id, {"error": str(e), "success": False}
        
        # Create experiment tasks
        experiment_tasks = list(enumerate(experiments))
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_exp = {
                executor.submit(run_single_experiment, task): task[0]
                for task in experiment_tasks
            }
            
            experiment_results = {}
            for future in as_completed(future_to_exp, timeout=self.config.timeout):
                exp_id = future_to_exp[future]
                try:
                    result_id, result = future.result()
                    experiment_results[result_id] = result
                except Exception as e:
                    logger.error(f"Experiment execution failed for {exp_id}: {e}")
                    experiment_results[exp_id] = {"error": str(e), "success": False}
        
        return {"experiment_results": experiment_results, "parallel_execution": True}
    
    def _run_experiments_sequential(self, experiments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run experiments sequentially."""
        
        experiment_results = {}
        for i, experiment in enumerate(experiments):
            try:
                result = self._execute_single_experiment(experiment)
                experiment_results[i] = result
            except Exception as e:
                logger.error(f"Experiment {i} failed: {e}")
                experiment_results[i] = {"error": str(e), "success": False}
        
        return {"experiment_results": experiment_results, "parallel_execution": False}
    
    def _execute_single_experiment(self, experiment: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single research experiment."""
        # This would be implemented based on specific experiment requirements
        # For now, return a placeholder structure
        return {
            "success": True,
            "execution_time": time.time(),
            "metrics": experiment.get("expected_metrics", {}),
            "experiment_type": experiment.get("type", "unknown")
        }


# Factory functions for creating optimized configurations
def create_research_config(
    optimization_target: str = "balanced",
    available_memory_gb: float = 8.0,
    enable_gpu: bool = None,
    research_scale: str = "medium"
) -> ResearchParallelConfig:
    """Create research-optimized parallel configuration."""
    
    if enable_gpu is None:
        # Auto-detect GPU
        try:
            gpu_devices = [d for d in jax.devices() if d.device_kind == 'gpu']
            enable_gpu = len(gpu_devices) > 0
        except:
            enable_gpu = False
    
    # Scale-based configurations
    scale_configs = {
        "small": {"workers": 2, "chunk": 50, "memory_factor": 0.6},
        "medium": {"workers": 4, "chunk": 200, "memory_factor": 0.8},
        "large": {"workers": 8, "chunk": 500, "memory_factor": 1.0},
        "xlarge": {"workers": 16, "chunk": 1000, "memory_factor": 1.2}
    }
    
    scale_cfg = scale_configs.get(research_scale, scale_configs["medium"])
    
    base_configs = {
        "speed": ResearchParallelConfig(
            max_workers=min(cpu_count(), scale_cfg["workers"] * 2),
            use_threads=True,
            chunk_size=scale_cfg["chunk"] * 2,
            memory_limit_gb=available_memory_gb * scale_cfg["memory_factor"],
            enable_gpu=enable_gpu,
            adaptive_batching=True,
            optimization_level="speed",
            cache_results=True,
            resource_monitoring=True
        ),
        "memory": ResearchParallelConfig(
            max_workers=max(1, scale_cfg["workers"] // 2),
            use_threads=False,  # Processes for memory isolation
            chunk_size=scale_cfg["chunk"] // 2,
            memory_limit_gb=available_memory_gb * 0.6,
            enable_gpu=False,  # GPU can cause memory issues
            adaptive_batching=True,
            optimization_level="memory",
            cache_results=False,  # Reduce memory usage
            resource_monitoring=True
        ),
        "balanced": ResearchParallelConfig(
            max_workers=scale_cfg["workers"],
            use_threads=True,
            chunk_size=scale_cfg["chunk"],
            memory_limit_gb=available_memory_gb * scale_cfg["memory_factor"] * 0.8,
            enable_gpu=enable_gpu,
            adaptive_batching=True,
            optimization_level="balanced",
            cache_results=True,
            resource_monitoring=True
        )
    }
    
    config = base_configs.get(optimization_target, base_configs["balanced"])
    
    logger.info(f"Created {optimization_target} research config: "
               f"{config.max_workers} workers, {config.memory_limit_gb:.1f}GB limit, "
               f"GPU={'enabled' if config.enable_gpu else 'disabled'}")
    
    return config


# Global research-grade evaluator
_global_evaluator = None


def get_research_evaluator(config: ResearchParallelConfig = None) -> HighPerformanceParallelEvaluator:
    """Get global research evaluator instance."""
    global _global_evaluator
    if _global_evaluator is None or config is not None:
        _global_evaluator = HighPerformanceParallelEvaluator(config or create_research_config())
    return _global_evaluator


def research_parallel_evaluate(
    function: Callable[[Array], float],
    points: Array,
    estimate_gradients: bool = False,
    config: Optional[ResearchParallelConfig] = None,
    progress_callback: Optional[Callable] = None
) -> Dict[str, Array]:
    """High-level interface for research-grade parallel evaluation."""
    
    evaluator = get_research_evaluator(config)
    return evaluator.evaluate_parallel(function, points, estimate_gradients, progress_callback)


if __name__ == "__main__":
    # Performance test
    def test_function(x):
        return jnp.sum(x**2) + 0.1 * jnp.sin(10 * jnp.sum(x))
    
    # Generate test data
    test_points = jnp.random.uniform(-5, 5, (1000, 5))
    
    # Create research config
    config = create_research_config("balanced", research_scale="medium")
    
    # Run evaluation
    results = research_parallel_evaluate(test_function, test_points, estimate_gradients=True, config=config)
    
    print(f"Evaluated {len(test_points)} points")
    print(f"Cache hit ratio: {results['cache_stats']['hit_ratio']:.2%}")
    if results['resource_metrics']:
        print(f"Throughput: {results['resource_metrics'].throughput:.1f} evals/sec")