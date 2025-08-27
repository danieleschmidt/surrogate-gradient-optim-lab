#!/usr/bin/env python3
"""
Generation 3: Scalable & Optimized Surrogate Optimization
Enhanced with performance optimization, caching, parallel processing, and auto-scaling
"""

import numpy as np
import logging
import time
import json
import multiprocessing as mp
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from functools import lru_cache, partial
import pickle
import hashlib
from scipy.optimize import minimize
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import psutil
import warnings
warnings.filterwarnings('ignore')

# Configure performance logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(processName)s] %(message)s',
    handlers=[
        logging.FileHandler('scalable_surrogate_optim.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    total_time: float
    fit_time: float
    optimization_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    cache_hits: int
    cache_misses: int
    parallel_workers: int
    throughput_samples_per_sec: float

@dataclass
class ScalabilityConfig:
    """Auto-scaling configuration."""
    enable_parallel: bool = True
    max_workers: Optional[int] = None
    enable_caching: bool = True
    cache_size: int = 1000
    enable_gpu: bool = False
    memory_limit_gb: float = 8.0
    auto_tune_workers: bool = True

class AdvancedCache:
    """High-performance caching with TTL and memory management."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: float = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_times = {}
        self.hits = 0
        self.misses = 0
        
    def _generate_key(self, x: np.ndarray) -> str:
        """Generate cache key from input."""
        return hashlib.md5(x.tobytes()).hexdigest()
        
    def get(self, x: np.ndarray) -> Optional[float]:
        """Get cached prediction."""
        key = self._generate_key(x)
        current_time = time.time()
        
        if key in self.cache:
            cached_time, value = self.cache[key]
            if current_time - cached_time < self.ttl_seconds:
                self.access_times[key] = current_time
                self.hits += 1
                return value
            else:
                # Expired
                del self.cache[key]
                del self.access_times[key]
                
        self.misses += 1
        return None
        
    def put(self, x: np.ndarray, value: float) -> None:
        """Cache prediction with LRU eviction."""
        key = self._generate_key(x)
        current_time = time.time()
        
        # Evict if at capacity
        if len(self.cache) >= self.max_size:
            # Remove oldest accessed item
            oldest_key = min(self.access_times.items(), key=lambda x: x[1])[0]
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
            
        self.cache[key] = (current_time, value)
        self.access_times[key] = current_time
        
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache)
        }

class ResourceMonitor:
    """System resource monitoring and auto-scaling."""
    
    def __init__(self):
        self.process = psutil.Process()
        
    def get_metrics(self) -> Dict[str, float]:
        """Get current resource metrics."""
        return {
            "cpu_percent": self.process.cpu_percent(),
            "memory_mb": self.process.memory_info().rss / 1024 / 1024,
            "memory_percent": self.process.memory_percent(),
            "num_threads": self.process.num_threads()
        }
        
    def optimal_workers(self) -> int:
        """Determine optimal number of workers based on system resources."""
        cpu_count = mp.cpu_count()
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        # Conservative approach: limit workers based on memory
        memory_based_workers = max(1, int(available_memory_gb / 2))  # 2GB per worker
        cpu_based_workers = cpu_count
        
        return min(memory_based_workers, cpu_based_workers, 8)  # Cap at 8

class ScalableSurrogateOptimizer:
    """Generation 3 - Scalable and optimized surrogate optimizer."""
    
    def __init__(self, 
                 surrogate_type: str = "neural_network",
                 config: ScalabilityConfig = None):
        
        self.surrogate_type = surrogate_type
        self.config = config or ScalabilityConfig()
        
        # Initialize components
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Performance components
        self.cache = AdvancedCache(self.config.cache_size) if self.config.enable_caching else None
        self.resource_monitor = ResourceMonitor()
        
        # Auto-tune workers
        if self.config.auto_tune_workers:
            self.config.max_workers = self.resource_monitor.optimal_workers()
        elif self.config.max_workers is None:
            self.config.max_workers = mp.cpu_count()
            
        # Setup logging
        self.logger = logging.getLogger(f"ScalableOptimizer_{id(self)}")
        
        # Performance tracking
        self.start_time = time.time()
        self.metrics = PerformanceMetrics(
            total_time=0, fit_time=0, optimization_time=0,
            memory_usage_mb=0, cpu_usage_percent=0,
            cache_hits=0, cache_misses=0,
            parallel_workers=self.config.max_workers,
            throughput_samples_per_sec=0
        )
        
        self.logger.info(f"Initialized scalable optimizer: {surrogate_type}, "
                        f"workers={self.config.max_workers}, caching={self.config.enable_caching}")
        
    def _parallel_function_evaluation(self, func, X: np.ndarray) -> np.ndarray:
        """Parallel function evaluation with load balancing."""
        
        if not self.config.enable_parallel or len(X) < 50:
            # Sequential for small datasets
            return np.array([func(x) for x in X])
            
        # Parallel evaluation
        chunk_size = max(1, len(X) // (self.config.max_workers * 4))  # Dynamic chunking
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit chunks
            futures = []
            for i in range(0, len(X), chunk_size):
                chunk = X[i:i+chunk_size]
                future = executor.submit(self._evaluate_chunk, func, chunk)
                futures.append(future)
                
            # Collect results
            results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    chunk_results = future.result(timeout=60)  # 1 minute timeout
                    results.extend(chunk_results)
                except Exception as e:
                    self.logger.error(f"Parallel evaluation failed: {e}")
                    # Fallback to sequential
                    return np.array([func(x) for x in X])
                    
        return np.array(results)
        
    @staticmethod
    def _evaluate_chunk(func, chunk):
        """Evaluate function on a chunk of data."""
        return [func(x) for x in chunk]
        
    def collect_data(self, func, bounds: List[Tuple[float, float]], 
                    n_samples: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        """Scalable data collection with parallel evaluation."""
        
        start_time = time.time()
        
        # Advanced sampling strategies
        dim = len(bounds)
        self.logger.info(f"Collecting {n_samples} samples in {dim}D space with {self.config.max_workers} workers")
        
        # Use Latin Hypercube Sampling for better space coverage
        try:
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(d=dim)
            samples = sampler.random(n_samples)
        except ImportError:
            # Fallback to random
            samples = np.random.random((n_samples, dim))
            
        # Scale to bounds
        X = np.array([
            [bounds[j][0] + s[j] * (bounds[j][1] - bounds[j][0]) for j in range(dim)]
            for s in samples
        ])
        
        # Parallel function evaluation
        y = self._parallel_function_evaluation(func, X)
        
        collection_time = time.time() - start_time
        throughput = n_samples / collection_time
        self.metrics.throughput_samples_per_sec = throughput
        
        self.logger.info(f"Data collection: {n_samples} samples in {collection_time:.2f}s "
                        f"({throughput:.1f} samples/sec)")
        
        return X, y
        
    def _create_optimized_model(self, X: np.ndarray, y: np.ndarray):
        """Create optimized model based on data characteristics."""
        
        n_samples, n_features = X.shape
        
        if self.surrogate_type == "neural_network":
            # Adaptive architecture based on data size
            if n_samples < 500:
                hidden_layers = (64, 32)
                max_iter = 500
            elif n_samples < 2000:
                hidden_layers = (128, 64, 32)
                max_iter = 1000
            else:
                hidden_layers = (256, 128, 64)
                max_iter = 2000
                
            model = MLPRegressor(
                hidden_layer_sizes=hidden_layers,
                max_iter=max_iter,
                alpha=0.001,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                learning_rate_init=0.001,
                random_state=42
            )
            
        elif self.surrogate_type == "gaussian_process":
            # Memory-efficient GP for large datasets
            if n_samples > 1000:
                # Use sparse GP approximation
                model = GaussianProcessRegressor(
                    kernel=RBF(length_scale=1.0),
                    alpha=1e-6,
                    n_restarts_optimizer=3,  # Reduced for speed
                    random_state=42
                )
            else:
                model = GaussianProcessRegressor(
                    kernel=RBF(length_scale=1.0),
                    alpha=1e-6,
                    n_restarts_optimizer=5,
                    random_state=42
                )
                
        elif self.surrogate_type == "random_forest":
            # Scalable random forest
            n_estimators = min(200, max(50, n_samples // 10))
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=10,
                min_samples_split=5,
                n_jobs=self.config.max_workers,
                random_state=42
            )
            
        else:
            raise ValueError(f"Unknown surrogate type: {self.surrogate_type}")
            
        return model
        
    def fit_surrogate(self, X: np.ndarray, y: np.ndarray) -> Any:
        """Scalable surrogate training with optimization."""
        
        start_time = time.time()
        
        # Memory-efficient preprocessing
        X_scaled = self.scaler.fit_transform(X)
        
        # Create optimized model
        self.model = self._create_optimized_model(X, y)
        
        self.logger.info(f"Training optimized {self.surrogate_type} on {len(X)} samples...")
        
        # Monitor resources during training
        initial_memory = self.resource_monitor.get_metrics()["memory_mb"]
        
        # Fit model
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        # Update metrics
        final_memory = self.resource_monitor.get_metrics()["memory_mb"]
        fit_time = time.time() - start_time
        
        self.metrics.fit_time = fit_time
        self.metrics.memory_usage_mb = final_memory - initial_memory
        
        self.logger.info(f"Training completed in {fit_time:.2f}s, "
                        f"memory delta: {self.metrics.memory_usage_mb:.1f}MB")
        
        return self.model
        
    def predict(self, x: np.ndarray, use_cache: bool = True) -> float:
        """Optimized prediction with caching."""
        
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
            
        # Check cache first
        if use_cache and self.cache:
            cached_result = self.cache.get(x)
            if cached_result is not None:
                return cached_result
                
        # Compute prediction
        x = np.atleast_2d(x)
        x_scaled = self.scaler.transform(x)
        prediction = float(self.model.predict(x_scaled)[0])
        
        # Cache result
        if use_cache and self.cache:
            self.cache.put(x.flatten(), prediction)
            
        return prediction
        
    def batch_predict(self, X: np.ndarray) -> np.ndarray:
        """Efficient batch prediction."""
        
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
            
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
        
    def _adaptive_gradient(self, x: np.ndarray) -> np.ndarray:
        """Adaptive gradient computation with error estimation."""
        
        # Multiple epsilon values for Richardson extrapolation
        epsilons = [1e-5, 1e-6, 1e-7]
        gradients = []
        
        for eps in epsilons:
            grad = np.zeros_like(x)
            f0 = self.predict(x, use_cache=True)
            
            for i in range(len(x)):
                x_plus = x.copy()
                x_plus[i] += eps
                f_plus = self.predict(x_plus, use_cache=True)
                grad[i] = (f_plus - f0) / eps
                
            gradients.append(grad)
            
        # Richardson extrapolation for better accuracy
        grad_refined = 4 * gradients[1] - gradients[0]  # O(h^3) approximation
        return grad_refined / 3
        
    def optimize(self, initial_point: np.ndarray, bounds: Optional[List[Tuple[float, float]]] = None,
                method: str = "L-BFGS-B", max_iterations: int = 1000) -> Dict[str, Any]:
        """Scalable optimization with adaptive strategies."""
        
        start_time = time.time()
        
        # Multi-start optimization with resource-aware parallelization
        n_starts = min(10, self.config.max_workers * 2)
        
        def single_optimization(start_point):
            def objective(x):
                return -self.predict(x, use_cache=True)
                
            def grad_objective(x):
                return -self._adaptive_gradient(x)
                
            try:
                result = minimize(
                    objective,
                    start_point,
                    method=method,
                    jac=grad_objective,
                    bounds=bounds,
                    options={'maxiter': max_iterations // n_starts}  # Distribute iterations
                )
                return result
            except Exception as e:
                self.logger.warning(f"Optimization failed for start point {start_point}: {e}")
                return None
                
        # Generate start points
        if bounds:
            start_points = []
            for _ in range(n_starts):
                point = np.array([
                    np.random.uniform(b[0], b[1]) for b in bounds
                ])
                start_points.append(point)
        else:
            # Perturb initial point
            start_points = [
                initial_point + np.random.normal(0, 0.1, size=initial_point.shape)
                for _ in range(n_starts)
            ]
            
        # Parallel optimization
        if self.config.enable_parallel and n_starts > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = [executor.submit(single_optimization, sp) for sp in start_points]
                results = [f.result() for f in concurrent.futures.as_completed(futures)]
        else:
            results = [single_optimization(sp) for sp in start_points]
            
        # Select best result
        valid_results = [r for r in results if r is not None and r.success]
        
        if not valid_results:
            raise RuntimeError("All optimization attempts failed")
            
        best_result = min(valid_results, key=lambda r: r.fun)
        
        optimization_time = time.time() - start_time
        self.metrics.optimization_time = optimization_time
        
        # Update cache statistics
        if self.cache:
            cache_stats = self.cache.stats()
            self.metrics.cache_hits = cache_stats["hits"]
            self.metrics.cache_misses = cache_stats["misses"]
            
        result = {
            "x_optimal": best_result.x,
            "f_optimal": -best_result.fun,
            "n_iterations": best_result.nit,
            "convergence_time": optimization_time,
            "n_starts": n_starts,
            "success": best_result.success,
            "function_evaluations": best_result.nfev,
            "cache_stats": self.cache.stats() if self.cache else None
        }
        
        self.logger.info(f"Multi-start optimization completed: {n_starts} starts, "
                        f"{optimization_time:.2f}s, f*={result['f_optimal']:.3f}")
        
        return result
        
    def get_performance_report(self) -> Dict[str, Any]:
        """Comprehensive performance report."""
        
        total_time = time.time() - self.start_time
        resource_metrics = self.resource_monitor.get_metrics()
        
        return {
            "configuration": asdict(self.config),
            "performance_metrics": {
                "total_runtime": total_time,
                "fit_time": self.metrics.fit_time,
                "optimization_time": self.metrics.optimization_time,
                "throughput_samples_per_sec": self.metrics.throughput_samples_per_sec,
            },
            "resource_usage": resource_metrics,
            "caching": self.cache.stats() if self.cache else None,
            "scalability": {
                "parallel_workers": self.config.max_workers,
                "memory_limit_gb": self.config.memory_limit_gb,
                "auto_scaling_enabled": self.config.auto_tune_workers
            }
        }

def demo_scalable_optimization():
    """Demonstrate scalable surrogate optimization."""
    print("⚡ Generation 3: Scalable & Optimized Surrogate Optimization")
    print("="*65)
    
    # Multi-modal test function (Rastrigin)
    def rastrigin(x):
        n = len(x)
        A = 10
        return -(A * n + sum(xi**2 - A * np.cos(2 * np.pi * xi) for xi in x))
    
    bounds = [(-5.12, 5.12)] * 3  # 3D optimization
    
    # Test different configurations
    configs = [
        ("Standard", ScalabilityConfig(enable_parallel=False, enable_caching=False)),
        ("Cached", ScalabilityConfig(enable_parallel=False, enable_caching=True)),
        ("Parallel", ScalabilityConfig(enable_parallel=True, enable_caching=False)),
        ("Full Optimization", ScalabilityConfig(enable_parallel=True, enable_caching=True, auto_tune_workers=True))
    ]
    
    for config_name, config in configs:
        print(f"\n🚀 Testing {config_name} Configuration")
        print("-" * 50)
        
        for surrogate_type in ["neural_network", "random_forest"]:
            print(f"\n🧠 {surrogate_type.replace('_', ' ').title()} Surrogate")
            
            try:
                # Initialize scalable optimizer
                optimizer = ScalableSurrogateOptimizer(
                    surrogate_type=surrogate_type,
                    config=config
                )
                
                # Large dataset for scalability testing
                print("📊 Collecting large training dataset...")
                X, y = optimizer.collect_data(rastrigin, bounds, n_samples=1000)
                print(f"   ✅ {len(X)} samples collected")
                
                # Scalable training
                print("🎓 Scalable model training...")
                train_start = time.time()
                optimizer.fit_surrogate(X, y)
                train_time = time.time() - train_start
                print(f"   ✅ Training completed in {train_time:.2f}s")
                
                # Multi-start optimization
                print("⚡ Multi-start scalable optimization...")
                initial_point = np.array([2.0, 2.0, 2.0])
                result = optimizer.optimize(initial_point, bounds, max_iterations=2000)
                
                print(f"   🎯 Results:")
                print(f"      Best point: [{', '.join(f'{x:.3f}' for x in result['x_optimal'])}]")
                print(f"      Best value: {result['f_optimal']:.3f}")
                print(f"      Optimization time: {result['convergence_time']:.2f}s")
                print(f"      Function evaluations: {result['function_evaluations']}")
                print(f"      Multiple starts: {result['n_starts']}")
                
                # Performance report
                report = optimizer.get_performance_report()
                print(f"   📈 Performance:")
                print(f"      Throughput: {report['performance_metrics']['throughput_samples_per_sec']:.1f} samples/sec")
                print(f"      Memory usage: {report['resource_usage']['memory_mb']:.1f} MB")
                print(f"      Workers: {report['scalability']['parallel_workers']}")
                
                if report['caching']:
                    print(f"      Cache hit rate: {report['caching']['hit_rate']:.1%}")
                    
            except Exception as e:
                print(f"   ❌ ERROR: {e}")

def benchmark_scalability():
    """Benchmark scalability across different dataset sizes."""
    print("\n🏁 Scalability Benchmark")
    print("="*30)
    
    def simple_quadratic(x):
        return -sum(xi**2 for xi in x)
    
    bounds = [(-5, 5)] * 2
    sample_sizes = [100, 500, 1000, 2000]
    
    for n_samples in sample_sizes:
        print(f"\n📊 Dataset size: {n_samples} samples")
        
        # Test with full optimization
        config = ScalabilityConfig(enable_parallel=True, enable_caching=True)
        optimizer = ScalableSurrogateOptimizer("neural_network", config)
        
        # Measure performance
        start_time = time.time()
        
        X, y = optimizer.collect_data(simple_quadratic, bounds, n_samples)
        collection_time = time.time() - start_time
        
        fit_start = time.time()
        optimizer.fit_surrogate(X, y)
        fit_time = time.time() - fit_start
        
        opt_start = time.time()
        result = optimizer.optimize(np.array([1.0, 1.0]), bounds)
        opt_time = time.time() - opt_start
        
        total_time = time.time() - start_time
        
        print(f"   Collection: {collection_time:.2f}s ({n_samples/collection_time:.1f} samples/sec)")
        print(f"   Training: {fit_time:.2f}s")
        print(f"   Optimization: {opt_time:.2f}s")
        print(f"   Total: {total_time:.2f}s")
        print(f"   Result: f*={result['f_optimal']:.3f}")

if __name__ == "__main__":
    demo_scalable_optimization()
    benchmark_scalability()