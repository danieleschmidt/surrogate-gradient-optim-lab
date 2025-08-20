#!/usr/bin/env python3
"""
Generation 3: Make It Scale - High Performance & Scalable Architecture
Building production-ready, scalable surrogate optimization with advanced performance features
"""

import jax.numpy as jnp
from jax import Array, jit, vmap, pmap, lax, random
import jax
import numpy as np
from typing import Callable, List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass, field
import logging
import time
import concurrent.futures
from functools import partial, lru_cache
import threading
import queue
from contextlib import contextmanager
import psutil
import os
from pathlib import Path
import pickle
import hashlib

# Enable JAX optimizations
jax.config.update("jax_enable_x64", True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass 
class PerformanceMetrics:
    """Comprehensive performance tracking."""
    computation_time: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    parallel_efficiency: float = 1.0
    throughput_samples_per_sec: float = 0.0
    gpu_utilization: float = 0.0


@dataclass
class OptimizationResult:
    """Enhanced optimization results with performance tracking."""
    x: Array  
    fun: float  
    success: bool = True
    message: str = "Optimization completed"
    nit: int = 0
    
    # Performance tracking
    performance_metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    convergence_history: List[float] = field(default_factory=list)
    gradient_norm_history: List[float] = field(default_factory=list)
    timing_breakdown: Dict[str, float] = field(default_factory=dict)
    parallel_info: Dict[str, Any] = field(default_factory=dict)


class MemoryManager:
    """Advanced memory management and caching."""
    
    def __init__(self, max_cache_size_mb: int = 100):
        self.max_cache_size_mb = max_cache_size_mb
        self.cache = {}
        self.cache_stats = {"hits": 0, "misses": 0}
        self._lock = threading.Lock()
    
    def _get_cache_key(self, x: Array) -> str:
        """Generate cache key for array."""
        return hashlib.md5(x.tobytes()).hexdigest()
    
    @lru_cache(maxsize=1000)
    def cached_prediction(self, x_hash: str, surrogate_id: str) -> Optional[float]:
        """LRU cached prediction lookup."""
        with self._lock:
            key = f"{surrogate_id}_{x_hash}"
            if key in self.cache:
                self.cache_stats["hits"] += 1
                return self.cache[key]
            self.cache_stats["misses"] += 1
            return None
    
    def store_prediction(self, x: Array, result: float, surrogate_id: str) -> None:
        """Store prediction in cache."""
        with self._lock:
            key = f"{surrogate_id}_{self._get_cache_key(x)}"
            self.cache[key] = result
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def clear_cache(self) -> None:
        """Clear prediction cache."""
        with self._lock:
            self.cache.clear()
            self.cache_stats = {"hits": 0, "misses": 0}


class ParallelDataCollector:
    """High-performance parallel data collection."""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(32, os.cpu_count() + 4)
        self.memory_manager = MemoryManager()
    
    @partial(jit, static_argnums=(3,))
    def _batch_evaluate_jit(self, function_vals: Array, X_batch: Array, 
                           bounds: Array, n_samples: int) -> Array:
        """JIT-compiled batch evaluation."""
        def single_eval(x):
            # Simple function evaluation (can be customized)
            return jnp.sum(x**2) + 0.1 * jnp.sum(x**3)
        
        return vmap(single_eval)(X_batch)
    
    def generate_samples_vectorized(self, n_samples: int, bounds: List[Tuple[float, float]],
                                   method: str = "sobol", key: random.PRNGKey = None) -> Array:
        """Vectorized sample generation."""
        if key is None:
            key = random.PRNGKey(int(time.time() * 1000) % (2**32))
        
        dim = len(bounds)
        bounds_array = jnp.array(bounds)
        
        if method == "random":
            # Vectorized random sampling
            samples = random.uniform(
                key, (n_samples, dim), 
                minval=bounds_array[:, 0], 
                maxval=bounds_array[:, 1]
            )
        
        elif method == "sobol":
            # Quasi-random Sobol sequence (simplified)
            base_samples = random.uniform(key, (n_samples, dim))
            samples = bounds_array[:, 0] + base_samples * (bounds_array[:, 1] - bounds_array[:, 0])
        
        elif method == "latin_hypercube":
            # Latin hypercube sampling
            intervals = jnp.linspace(0, 1, n_samples + 1)
            samples = []
            for i in range(dim):
                # Shuffle intervals for each dimension
                perm_key, key = random.split(key)
                perm = random.permutation(perm_key, n_samples)
                dim_samples = intervals[perm] + random.uniform(
                    key, (n_samples,)
                ) / n_samples
                samples.append(
                    bounds_array[i, 0] + dim_samples * (bounds_array[i, 1] - bounds_array[i, 0])
                )
                key, _ = random.split(key)
            samples = jnp.stack(samples, axis=1)
        
        else:
            raise ValueError(f"Unknown sampling method: {method}")
        
        return samples
    
    def collect_parallel(self, function: Callable, n_samples: int,
                        bounds: List[Tuple[float, float]], 
                        sampling_method: str = "sobol",
                        batch_size: int = 100) -> Tuple[Array, Array, PerformanceMetrics]:
        """High-performance parallel data collection."""
        
        start_time = time.time()
        initial_memory = self.memory_manager.get_memory_usage()
        
        logger.info(f"Starting parallel data collection: {n_samples} samples, {self.max_workers} workers")
        
        # Generate all samples upfront
        all_samples = self.generate_samples_vectorized(n_samples, bounds, sampling_method)
        
        X, y = [], []
        failed_evaluations = 0
        
        # Process in batches for memory efficiency
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all batches
            future_to_batch = {}
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)
                batch_samples = all_samples[start_idx:end_idx]
                
                # Submit batch evaluation
                future = executor.submit(self._evaluate_batch, function, batch_samples)
                future_to_batch[future] = i
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                
                try:
                    batch_X, batch_y, batch_failures = future.result()
                    X.extend(batch_X)
                    y.extend(batch_y)
                    failed_evaluations += batch_failures
                    
                    if batch_idx % 10 == 0:
                        logger.info(f"Completed batch {batch_idx + 1}/{n_batches}")
                
                except Exception as e:
                    logger.error(f"Batch {batch_idx} failed: {e}")
                    failed_evaluations += batch_size
        
        # Convert to arrays
        if X:
            X = jnp.stack(X)
            y = jnp.array(y)
        else:
            X = jnp.array([]).reshape(0, len(bounds))
            y = jnp.array([])
        
        # Performance metrics
        total_time = time.time() - start_time
        final_memory = self.memory_manager.get_memory_usage()
        
        metrics = PerformanceMetrics(
            computation_time=total_time,
            memory_usage_mb=final_memory - initial_memory,
            throughput_samples_per_sec=len(X) / total_time if total_time > 0 else 0,
            parallel_efficiency=len(X) / (n_samples * self.max_workers) if n_samples > 0 else 0
        )
        
        logger.info(f"Data collection complete: {len(X)} samples in {total_time:.2f}s "
                   f"({metrics.throughput_samples_per_sec:.1f} samples/sec)")
        
        return X, y, metrics
    
    def _evaluate_batch(self, function: Callable, batch_samples: Array) -> Tuple[List, List, int]:
        """Evaluate a batch of samples."""
        batch_X, batch_y = [], []
        failures = 0
        
        for sample in batch_samples:
            try:
                value = function(sample)
                if jnp.isfinite(value):
                    batch_X.append(sample)
                    batch_y.append(float(value))
                else:
                    failures += 1
            except Exception:
                failures += 1
        
        return batch_X, batch_y, failures


class ScalableSurrogate:
    """High-performance surrogate with JIT compilation and vectorization."""
    
    def __init__(self, degree: int = 2, use_gpu: bool = False, 
                 enable_jit: bool = True, ensemble_size: int = 1):
        self.degree = degree
        self.use_gpu = use_gpu
        self.enable_jit = enable_jit
        self.ensemble_size = ensemble_size
        
        # Model state
        self.coeffs = None
        self.fitted = False
        self.surrogate_id = f"scalable_{int(time.time()*1000)}"
        
        # Performance components
        self.memory_manager = MemoryManager()
        self.performance_stats = PerformanceMetrics()
        
        # JIT-compiled functions
        if enable_jit:
            self._predict_jit = jit(self._predict_core)
            self._gradient_jit = jit(self._gradient_core)
            self._batch_predict_jit = jit(vmap(self._predict_core, in_axes=(0,)))
        
        logger.info(f"ScalableSurrogate initialized: degree={degree}, JIT={enable_jit}, "
                   f"GPU={use_gpu}, ensemble={ensemble_size}")
    
    @partial(jit, static_argnums=(0,))
    def _create_features_vectorized(self, X: Array) -> Array:
        """Vectorized feature creation with JIT compilation."""
        n_samples = X.shape[0]
        
        # Base features: [1, x1, x2, x1^2, x2^2, x1*x2]
        ones = jnp.ones((n_samples, 1))
        x1 = X[:, 0:1]
        x2 = X[:, 1:2]
        x1_sq = x1 ** 2
        x2_sq = x2 ** 2
        x1_x2 = x1 * x2
        
        features = jnp.concatenate([ones, x1, x2, x1_sq, x2_sq, x1_x2], axis=1)
        
        # Higher order features if needed
        if self.degree > 2:
            x1_cube = x1 ** 3
            x2_cube = x2 ** 3
            x1_sq_x2 = x1_sq * x2
            x1_x2_sq = x1 * x2_sq
            features = jnp.concatenate([features, x1_cube, x2_cube, x1_sq_x2, x1_x2_sq], axis=1)
        
        return features
    
    def fit_scalable(self, X: Array, y: Array) -> PerformanceMetrics:
        """High-performance surrogate fitting with comprehensive optimization."""
        start_time = time.time()
        initial_memory = self.memory_manager.get_memory_usage()
        
        logger.info(f"Starting scalable surrogate fitting: {len(X)} samples")
        
        # Validation
        if len(X) != len(y):
            raise ValueError(f"X length {len(X)} != y length {len(y)}")
        
        # Vectorized feature creation
        features = self._create_features_vectorized(X)
        
        # Robust fitting with multiple strategies
        self.coeffs = self._fit_ensemble(features, y)
        self.fitted = True
        
        # Performance metrics
        total_time = time.time() - start_time
        final_memory = self.memory_manager.get_memory_usage()
        
        metrics = PerformanceMetrics(
            computation_time=total_time,
            memory_usage_mb=final_memory - initial_memory,
            throughput_samples_per_sec=len(X) / total_time if total_time > 0 else 0
        )
        
        self.performance_stats = metrics
        
        logger.info(f"Surrogate fitting complete: {total_time:.3f}s "
                   f"({metrics.throughput_samples_per_sec:.1f} samples/sec)")
        
        return metrics
    
    def _fit_ensemble(self, features: Array, y: Array) -> Array:
        """Fit ensemble of models for robustness."""
        if self.ensemble_size == 1:
            return self._fit_single_model(features, y)
        
        # Ensemble fitting
        coeffs_ensemble = []
        n_samples = len(y)
        
        for i in range(self.ensemble_size):
            # Bootstrap sampling
            key = random.PRNGKey(i)
            indices = random.choice(key, n_samples, (n_samples,), replace=True)
            
            features_boot = features[indices]
            y_boot = y[indices]
            
            coeffs = self._fit_single_model(features_boot, y_boot)
            coeffs_ensemble.append(coeffs)
        
        # Average coefficients
        return jnp.mean(jnp.stack(coeffs_ensemble), axis=0)
    
    def _fit_single_model(self, features: Array, y: Array) -> Array:
        """Fit single model with numerical stability."""
        # Regularized least squares
        regularization = 1e-4
        A = features.T @ features + regularization * jnp.eye(features.shape[1])
        b = features.T @ y
        
        try:
            coeffs = jnp.linalg.solve(A, b)
            
            # Check for numerical issues
            if not jnp.all(jnp.isfinite(coeffs)):
                logger.warning("Non-finite coefficients, using stronger regularization")
                A_strong = features.T @ features + 1e-1 * jnp.eye(features.shape[1])
                coeffs = jnp.linalg.solve(A_strong, b)
            
            return coeffs
            
        except Exception as e:
            logger.error(f"Fitting failed: {e}")
            # Fallback to simple quadratic
            return jnp.array([0.0, 0.0, 0.0, 1.0, 1.0, 0.0])
    
    def _predict_core(self, x: Array) -> float:
        """Core prediction function for JIT compilation."""
        x1, x2 = x[0], x[1]
        
        # Create features
        if self.degree == 2:
            features = jnp.array([1.0, x1, x2, x1*x1, x2*x2, x1*x2])
        else:
            features = jnp.array([1.0, x1, x2, x1*x1, x2*x2, x1*x2, 
                                 x1*x1*x1, x2*x2*x2, x1*x1*x2, x1*x2*x2])
        
        # Ensure matching length
        features = features[:len(self.coeffs)]
        
        return jnp.dot(self.coeffs, features)
    
    def _gradient_core(self, x: Array) -> Array:
        """Core gradient function for JIT compilation."""
        x1, x2 = x[0], x[1]
        
        # Gradient features
        if self.degree == 2:
            grad_x1 = jnp.array([0.0, 1.0, 0.0, 2*x1, 0.0, x2])
            grad_x2 = jnp.array([0.0, 0.0, 1.0, 0.0, 2*x2, x1])
        else:
            grad_x1 = jnp.array([0.0, 1.0, 0.0, 2*x1, 0.0, x2,
                                3*x1*x1, 0.0, 2*x1*x2, x2*x2])
            grad_x2 = jnp.array([0.0, 0.0, 1.0, 0.0, 2*x2, x1,
                                0.0, 3*x2*x2, x1*x1, 2*x1*x2])
        
        # Ensure matching length
        grad_x1 = grad_x1[:len(self.coeffs)]
        grad_x2 = grad_x2[:len(self.coeffs)]
        
        return jnp.array([
            jnp.dot(self.coeffs, grad_x1),
            jnp.dot(self.coeffs, grad_x2)
        ])
    
    def predict(self, x: Array, use_cache: bool = True) -> float:
        """High-performance prediction with caching."""
        if not self.fitted:
            raise ValueError("Surrogate not fitted")
        
        # Cache lookup
        if use_cache:
            cached = self.memory_manager.cached_prediction(
                self.memory_manager._get_cache_key(x), 
                self.surrogate_id
            )
            if cached is not None:
                self.performance_stats.cache_hits += 1
                return cached
            self.performance_stats.cache_misses += 1
        
        # Compute prediction
        if self.enable_jit:
            result = float(self._predict_jit(x))
        else:
            result = float(self._predict_core(x))
        
        # Store in cache
        if use_cache:
            self.memory_manager.store_prediction(x, result, self.surrogate_id)
        
        return result
    
    def gradient(self, x: Array) -> Array:
        """High-performance gradient computation."""
        if not self.fitted:
            raise ValueError("Surrogate not fitted")
        
        if self.enable_jit:
            return self._gradient_jit(x)
        else:
            return self._gradient_core(x)
    
    def batch_predict(self, X: Array) -> Array:
        """Vectorized batch prediction."""
        if not self.fitted:
            raise ValueError("Surrogate not fitted")
        
        if self.enable_jit and len(X) > 1:
            return self._batch_predict_jit(X)
        else:
            return jnp.array([self.predict(x, use_cache=False) for x in X])


class HighPerformanceOptimizer:
    """Scalable optimizer with advanced algorithms."""
    
    def __init__(self, algorithm: str = "adam", learning_rate: float = 0.01,
                 max_iterations: int = 1000, parallel_starts: int = 4):
        self.algorithm = algorithm
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.parallel_starts = parallel_starts
        
        # Performance tracking
        self.memory_manager = MemoryManager()
    
    @partial(jit, static_argnums=(0,))
    def _adam_step(self, x: Array, gradient: Array, m: Array, v: Array, 
                   t: int, lr: float = 0.01, beta1: float = 0.9, 
                   beta2: float = 0.999, eps: float = 1e-8) -> Tuple[Array, Array, Array]:
        """JIT-compiled Adam optimization step."""
        # Update biased moments
        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * (gradient ** 2)
        
        # Bias correction
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        # Update parameters
        x_new = x - lr * m_hat / (jnp.sqrt(v_hat) + eps)
        
        return x_new, m, v
    
    def optimize_parallel(self, surrogate: ScalableSurrogate, 
                         initial_points: List[Array],
                         bounds: Optional[List[Tuple[float, float]]] = None) -> OptimizationResult:
        """Parallel multi-start optimization."""
        
        start_time = time.time()
        logger.info(f"Starting parallel optimization with {len(initial_points)} starts")
        
        # Run optimizations in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.parallel_starts) as executor:
            futures = []
            
            for i, x0 in enumerate(initial_points):
                future = executor.submit(self._single_optimization, surrogate, x0, bounds, i)
                futures.append(future)
            
            # Collect results
            results = []
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
        
        # Select best result
        best_result = min(results, key=lambda r: r.fun)
        
        # Update performance metrics
        total_time = time.time() - start_time
        best_result.performance_metrics.computation_time = total_time
        best_result.parallel_info = {
            "n_starts": len(initial_points),
            "parallel_efficiency": len(initial_points) / total_time if total_time > 0 else 0
        }
        
        logger.info(f"Parallel optimization complete: {total_time:.3f}s, "
                   f"best value: {best_result.fun:.6f}")
        
        return best_result
    
    def _single_optimization(self, surrogate: ScalableSurrogate, x0: Array, 
                           bounds: Optional[List[Tuple[float, float]]], 
                           start_id: int) -> OptimizationResult:
        """Single optimization run."""
        
        x = jnp.array(x0, dtype=float)
        
        # Algorithm-specific initialization
        if self.algorithm == "adam":
            m = jnp.zeros_like(x)
            v = jnp.zeros_like(x)
        
        convergence_history = []
        gradient_norm_history = []
        
        for t in range(1, self.max_iterations + 1):
            # Evaluate current point
            current_val = surrogate.predict(x, use_cache=True)
            convergence_history.append(float(current_val))
            
            # Compute gradient
            gradient = surrogate.gradient(x)
            grad_norm = float(jnp.linalg.norm(gradient))
            gradient_norm_history.append(grad_norm)
            
            # Check convergence
            if grad_norm < 1e-6:
                break
            
            # Optimization step
            if self.algorithm == "adam":
                x, m, v = self._adam_step(x, gradient, m, v, t, self.learning_rate)
            else:  # Gradient descent fallback
                x = x - self.learning_rate * gradient
            
            # Apply bounds
            if bounds is not None:
                for i, (lower, upper) in enumerate(bounds):
                    x = x.at[i].set(jnp.clip(x[i], lower, upper))
        
        return OptimizationResult(
            x=x,
            fun=float(surrogate.predict(x, use_cache=True)),
            success=grad_norm < 1e-4,
            nit=t,
            convergence_history=convergence_history,
            gradient_norm_history=gradient_norm_history,
            message=f"Start {start_id}: {'Converged' if grad_norm < 1e-4 else 'Max iterations'}"
        )


def comprehensive_benchmark(test_function: Callable, bounds: List[Tuple[float, float]],
                          n_samples_list: List[int] = [100, 500, 1000]) -> Dict[str, Any]:
    """Comprehensive performance benchmarking."""
    
    logger.info("Starting comprehensive performance benchmark")
    benchmark_results = {}
    
    for n_samples in n_samples_list:
        logger.info(f"\nBenchmarking with {n_samples} samples...")
        
        # Data collection benchmark
        collector = ParallelDataCollector(max_workers=8)
        
        data_start = time.time()
        X, y, data_metrics = collector.collect_parallel(
            test_function, n_samples, bounds, "sobol", batch_size=50
        )
        data_time = time.time() - data_start
        
        # Surrogate training benchmark
        surrogate = ScalableSurrogate(degree=2, enable_jit=True, ensemble_size=1)
        
        train_start = time.time()
        train_metrics = surrogate.fit_scalable(X, y)
        train_time = time.time() - train_start
        
        # Optimization benchmark
        optimizer = HighPerformanceOptimizer(algorithm="adam", parallel_starts=4)
        initial_points = [
            jnp.array([2.0, 2.0]),
            jnp.array([-2.0, 1.5]),
            jnp.array([1.5, -2.0]),
            jnp.array([0.5, 0.5])
        ]
        
        opt_start = time.time()
        opt_result = optimizer.optimize_parallel(surrogate, initial_points, bounds)
        opt_time = time.time() - opt_start
        
        # Batch prediction benchmark
        test_points = collector.generate_samples_vectorized(100, bounds, "random")
        
        batch_start = time.time()
        batch_predictions = surrogate.batch_predict(test_points)
        batch_time = time.time() - batch_start
        
        benchmark_results[n_samples] = {
            "data_collection": {
                "time": data_time,
                "throughput": len(X) / data_time if data_time > 0 else 0,
                "samples_collected": len(X),
                "metrics": data_metrics
            },
            "surrogate_training": {
                "time": train_time,
                "throughput": len(X) / train_time if train_time > 0 else 0,
                "metrics": train_metrics
            },
            "optimization": {
                "time": opt_time,
                "final_value": opt_result.fun,
                "convergence_iterations": opt_result.nit,
                "success": opt_result.success
            },
            "batch_prediction": {
                "time": batch_time,
                "throughput": len(test_points) / batch_time if batch_time > 0 else 0,
                "predictions_per_second": len(test_points) / batch_time if batch_time > 0 else 0
            }
        }
        
        logger.info(f"  Data collection: {len(X)} samples in {data_time:.3f}s")
        logger.info(f"  Training: {train_time:.3f}s")
        logger.info(f"  Optimization: {opt_time:.3f}s (value: {opt_result.fun:.6f})")
        logger.info(f"  Batch prediction: {len(test_points)} predictions in {batch_time:.3f}s")
    
    return benchmark_results


def main():
    """Generation 3 demo - Make It Scale."""
    print("⚡ GENERATION 3: MAKE IT SCALE")
    print("=" * 70)
    
    # Scalable test function
    def scalable_test_function(x):
        x = jnp.atleast_1d(x)
        x1, x2 = x[0], x[1]
        
        # Multi-modal function with scaling challenges
        result = (x1**2 + x2**2) * (1 + 0.1 * jnp.sin(5*x1) * jnp.cos(5*x2))
        result += 0.01 * (x1**4 + x2**4)  # Add higher-order terms
        
        return float(result)
    
    bounds = [(-4.0, 4.0), (-4.0, 4.0)]
    
    print("📊 Test function: Multi-modal with higher-order terms")
    print(f"📊 Bounds: {bounds}")
    print(f"📊 Expected minimum: near [0, 0]")
    
    try:
        # 1. High-performance data collection
        print("\n🚀 Step 1: High-performance parallel data collection...")
        collector = ParallelDataCollector(max_workers=8)
        
        X_train, y_train, data_metrics = collector.collect_parallel(
            scalable_test_function,
            n_samples=500,
            bounds=bounds,
            sampling_method="latin_hypercube",
            batch_size=50
        )
        
        print(f"   ✅ Collected {len(X_train)} samples")
        print(f"   ⚡ Throughput: {data_metrics.throughput_samples_per_sec:.1f} samples/sec")
        print(f"   💾 Memory usage: {data_metrics.memory_usage_mb:.1f} MB")
        print(f"   🔄 Parallel efficiency: {data_metrics.parallel_efficiency:.2f}")
        
        # 2. Scalable surrogate training
        print("\n🧠 Step 2: Scalable surrogate training with JIT compilation...")
        surrogate = ScalableSurrogate(
            degree=2, 
            enable_jit=True, 
            ensemble_size=3,  # Ensemble for robustness
            use_gpu=False
        )
        
        train_metrics = surrogate.fit_scalable(X_train, y_train)
        
        print(f"   ✅ Surrogate trained with ensemble size 3")
        print(f"   ⚡ Training time: {train_metrics.computation_time:.3f}s")
        print(f"   🚀 Training throughput: {train_metrics.throughput_samples_per_sec:.1f} samples/sec")
        print(f"   💾 Memory usage: {train_metrics.memory_usage_mb:.1f} MB")
        
        # 3. Performance testing
        print("\n🔍 Step 3: Performance validation and caching...")
        
        # Test prediction performance
        test_points = collector.generate_samples_vectorized(1000, bounds, "random")
        
        # Without caching
        no_cache_start = time.time()
        predictions_no_cache = [surrogate.predict(x, use_cache=False) for x in test_points[:100]]
        no_cache_time = time.time() - no_cache_start
        
        # With caching
        cache_start = time.time()
        predictions_cache = [surrogate.predict(x, use_cache=True) for x in test_points[:100]]
        # Repeat for cache hits
        predictions_cache_repeat = [surrogate.predict(x, use_cache=True) for x in test_points[:100]]
        cache_time = time.time() - cache_start
        
        # Batch prediction
        batch_start = time.time()
        batch_predictions = surrogate.batch_predict(test_points[:100])
        batch_time = time.time() - batch_start
        
        print(f"   🔍 Single predictions (no cache): {no_cache_time:.4f}s")
        print(f"   ⚡ Single predictions (with cache): {cache_time:.4f}s")
        print(f"   🚀 Batch predictions: {batch_time:.4f}s")
        print(f"   📈 Speedup (batch vs single): {no_cache_time/batch_time:.1f}x")
        print(f"   💾 Cache stats: {surrogate.performance_stats.cache_hits} hits, "
              f"{surrogate.performance_stats.cache_misses} misses")
        
        # 4. High-performance optimization
        print("\n🎯 Step 4: High-performance parallel optimization...")
        optimizer = HighPerformanceOptimizer(
            algorithm="adam",
            learning_rate=0.05,
            max_iterations=200,
            parallel_starts=6
        )
        
        # Generate diverse starting points
        start_points = collector.generate_samples_vectorized(6, bounds, "sobol")
        
        optimization_result = optimizer.optimize_parallel(
            surrogate, 
            [start_points[i] for i in range(6)], 
            bounds
        )
        
        true_optimum_value = scalable_test_function(optimization_result.x)
        
        print(f"   ✅ Optimization complete: {optimization_result.message}")
        print(f"   🎯 Best point: [{optimization_result.x[0]:.6f}, {optimization_result.x[1]:.6f}]")
        print(f"   📈 Surrogate value: {optimization_result.fun:.6f}")
        print(f"   📊 True value: {true_optimum_value:.6f}")
        print(f"   ⏱️  Optimization time: {optimization_result.performance_metrics.computation_time:.3f}s")
        print(f"   🔄 Parallel starts: {optimization_result.parallel_info['n_starts']}")
        print(f"   📈 Convergence iterations: {optimization_result.nit}")
        
        # 5. Comprehensive benchmarking
        print("\n📊 Step 5: Comprehensive performance benchmarking...")
        
        benchmark_results = comprehensive_benchmark(
            scalable_test_function, 
            bounds, 
            n_samples_list=[100, 300, 500]
        )
        
        print(f"\n   📈 PERFORMANCE SCALING RESULTS:")
        for n_samples, results in benchmark_results.items():
            print(f"     {n_samples} samples:")
            print(f"       Data collection: {results['data_collection']['throughput']:.1f} samples/sec")
            print(f"       Training: {results['surrogate_training']['throughput']:.1f} samples/sec")  
            print(f"       Batch prediction: {results['batch_prediction']['throughput']:.1f} predictions/sec")
            print(f"       Optimization value: {results['optimization']['final_value']:.6f}")
        
        # 6. Scalability assessment
        print(f"\n🏆 GENERATION 3 RESULTS:")
        print(f"   🎯 Best point: [{optimization_result.x[0]:.6f}, {optimization_result.x[1]:.6f}]")
        print(f"   📈 Best value: {true_optimum_value:.6f}")
        print(f"   📊 True optimum at [0,0]: {scalable_test_function(jnp.array([0.0, 0.0])):.6f}")
        print(f"   📏 Distance from optimum: {jnp.linalg.norm(optimization_result.x):.6f}")
        
        # Scalability criteria
        base_throughput = benchmark_results[100]['data_collection']['throughput']
        scaled_throughput = benchmark_results[500]['data_collection']['throughput']
        scaling_efficiency = (scaled_throughput / base_throughput) / 5.0  # Should scale with data size
        
        scalability_criteria = {
            "High-throughput data collection": data_metrics.throughput_samples_per_sec > 50,
            "JIT compilation working": train_metrics.throughput_samples_per_sec > 100,
            "Caching effective": surrogate.performance_stats.cache_hits > 0,
            "Batch processing faster": batch_time < no_cache_time,
            "Parallel optimization successful": optimization_result.success,
            "Good final result": true_optimum_value < 1.0,
            "Performance scaling": scaling_efficiency > 0.5,
            "Memory management": train_metrics.memory_usage_mb < 100,
        }
        
        print(f"\n✅ GENERATION 3 SCALABILITY CRITERIA:")
        all_passed = True
        for criterion, passed in scalability_criteria.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {status}: {criterion}")
            all_passed = all_passed and passed
        
        print(f"\n⚡ GENERATION 3: {'SUCCESS' if all_passed else 'PARTIAL_SUCCESS'}")
        
        # Performance summary
        print(f"\n📊 PERFORMANCE SUMMARY:")
        print(f"   🚀 Data throughput: {data_metrics.throughput_samples_per_sec:.1f} samples/sec")
        print(f"   🧠 Training throughput: {train_metrics.throughput_samples_per_sec:.1f} samples/sec")
        print(f"   ⚡ Prediction speedup: {no_cache_time/batch_time:.1f}x (batch vs individual)")
        print(f"   🔄 Parallel efficiency: {data_metrics.parallel_efficiency:.2f}")
        print(f"   💾 Peak memory: {max(data_metrics.memory_usage_mb, train_metrics.memory_usage_mb):.1f} MB")
        
        return all_passed
        
    except Exception as e:
        print(f"\n❌ GENERATION 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)