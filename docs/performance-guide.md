# Performance Guide

This guide covers optimization strategies, performance monitoring, and best practices for scaling the Surrogate Gradient Optimization Lab.

## Table of Contents

1. [Performance Overview](#performance-overview)
2. [Profiling and Monitoring](#profiling-and-monitoring)
3. [Memory Management](#memory-management)
4. [Caching Strategies](#caching-strategies)
5. [Parallel Processing](#parallel-processing)
6. [Optimization Tuning](#optimization-tuning)
7. [Scaling Guidelines](#scaling-guidelines)
8. [Benchmarking](#benchmarking)

---

## Performance Overview

### Key Performance Factors

1. **Surrogate Model Choice**: Different models have different computational characteristics
2. **Data Size**: Training sample size affects both accuracy and computational cost
3. **Dimensionality**: Higher dimensions require more sophisticated approaches
4. **Function Evaluation Cost**: Balance between surrogate accuracy and true function calls

### Performance Hierarchy (Typical)

**Training Speed** (Fast → Slow):
1. Random Forest
2. Neural Network (small)
3. Gaussian Process (small data)
4. Neural Network (large)
5. Gaussian Process (large data)

**Prediction Speed** (Fast → Slow):
1. Neural Network
2. Random Forest
3. Gaussian Process

**Memory Usage** (Low → High):
1. Random Forest
2. Neural Network
3. Gaussian Process (scales O(n³))

---

## Profiling and Monitoring

### Basic Performance Monitoring

```python
from surrogate_optim.performance import performance_monitor
from surrogate_optim import SurrogateOptimizer

def expensive_function(x):
    import time
    time.sleep(0.01)  # Simulate expensive computation
    return sum(x**2)

bounds = [(-5, 5)] * 10

# Monitor the entire workflow
with performance_monitor("full_optimization", profile_level="detailed") as profiler:
    
    with performance_monitor("data_collection") as data_profiler:
        from surrogate_optim.data import collect_data
        dataset = collect_data(expensive_function, n_samples=200, bounds=bounds)
    
    with performance_monitor("surrogate_training") as train_profiler:
        optimizer = SurrogateOptimizer(
            surrogate_type="neural_network",
            surrogate_params={"n_epochs": 100}
        )
        optimizer.fit_surrogate(dataset)
    
    with performance_monitor("optimization") as opt_profiler:
        result = optimizer.optimize(jnp.zeros(10), bounds)

# Get performance summary
summary = profiler.get_performance_summary()
print(f"Total time: {summary['timing_summary']['full_optimization']['mean_time']:.3f}s")
print(f"Peak memory: {summary['memory_summary'].get('peak_mb', 0):.1f}MB")
```

### Detailed Profiling

```python
from surrogate_optim.performance import ProfiledOptimizer
from surrogate_optim.optimizers import GradientDescentOptimizer

# Wrap optimizer with profiling
base_optimizer = GradientDescentOptimizer(max_iterations=100)
profiled_optimizer = ProfiledOptimizer(
    base_optimizer,
    profile_level="full",  # Include cProfile
    output_file="optimization_profile.json"
)

# Use as normal optimizer
result = profiled_optimizer.optimize(surrogate, x0, bounds)

# Access detailed profiling data
performance_data = result.metadata["performance_profile"]
print(f"Function evaluations: {performance_data['function_evaluations']['count']}")
print(f"Average eval time: {performance_data['function_evaluations']['mean_time']:.6f}s")
```

### Custom Performance Metrics

```python
from surrogate_optim.performance import PerformanceProfiler
import time

class CustomProfiler(PerformanceProfiler):
    def __init__(self):
        super().__init__(enable_memory_profiling=True)
        self.custom_metrics = {}
    
    def track_surrogate_accuracy(self, surrogate, true_function, test_points):
        """Track surrogate prediction accuracy over time"""
        start_time = time.time()
        
        # Evaluate both surrogate and true function
        surrogate_preds = surrogate.predict(test_points)
        true_values = jnp.array([true_function(x) for x in test_points])
        
        # Calculate metrics
        mse = float(jnp.mean((surrogate_preds - true_values)**2))
        mae = float(jnp.mean(jnp.abs(surrogate_preds - true_values)))
        
        duration = time.time() - start_time
        
        self.custom_metrics["accuracy_check"] = {
            "mse": mse,
            "mae": mae,
            "check_duration": duration,
            "n_test_points": len(test_points)
        }
    
    def get_extended_summary(self):
        """Get summary including custom metrics"""
        base_summary = self.get_performance_summary()
        base_summary["custom_metrics"] = self.custom_metrics
        return base_summary

# Usage
profiler = CustomProfiler()
# ... use profiler in optimization ...
```

---

## Memory Management

### Memory Monitoring

```python
from surrogate_optim.performance import MemoryMonitor
import jax.numpy as jnp

def memory_intensive_optimization():
    with MemoryMonitor() as monitor:
        # Large dataset
        n_samples = 10000
        n_dims = 50
        
        # Monitor memory during data generation
        X = jax.random.normal(jax.random.PRNGKey(0), (n_samples, n_dims))
        y = jnp.sum(X**2, axis=1)
        
        # Monitor during training
        from surrogate_optim.models.neural import NeuralSurrogate
        surrogate = NeuralSurrogate(hidden_layers=[512, 256, 128], n_epochs=200)
        
        dataset = Dataset(X=X, y=y)
        surrogate.fit(dataset)
    
    print(f"Peak memory usage: {monitor.get_peak_memory():.1f} MB")
    print(f"Memory increase: {monitor.get_memory_increase():.1f} MB")
    return monitor.get_summary()

summary = memory_intensive_optimization()
```

### Memory Optimization Strategies

```python
# Strategy 1: Batch Processing for Large Datasets
class BatchedNeuralSurrogate(NeuralSurrogate):
    def __init__(self, max_batch_size=1000, **kwargs):
        self.max_batch_size = max_batch_size
        super().__init__(**kwargs)
    
    def fit(self, dataset):
        """Train in batches to reduce memory usage"""
        if dataset.n_samples <= self.max_batch_size:
            return super().fit(dataset)
        
        # Train on batches
        n_batches = (dataset.n_samples + self.max_batch_size - 1) // self.max_batch_size
        
        for epoch in range(self.n_epochs):
            indices = jnp.arange(dataset.n_samples)
            shuffled_indices = jax.random.permutation(
                jax.random.PRNGKey(epoch), indices
            )
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * self.max_batch_size
                end_idx = min(start_idx + self.max_batch_size, dataset.n_samples)
                batch_indices = shuffled_indices[start_idx:end_idx]
                
                batch_X = dataset.X[batch_indices]
                batch_y = dataset.y[batch_indices]
                
                # Train on batch (implement batch training logic)
                self._train_batch(batch_X, batch_y)

# Strategy 2: Memory-Efficient Data Loading
class MemoryEfficientDataCollector:
    def __init__(self, function, bounds, chunk_size=100):
        self.function = function
        self.bounds = bounds
        self.chunk_size = chunk_size
    
    def collect_streaming(self, n_samples, sampling="random"):
        """Collect data in chunks to save memory"""
        all_X = []
        all_y = []
        
        for chunk_start in range(0, n_samples, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, n_samples)
            chunk_size = chunk_end - chunk_start
            
            # Generate chunk
            if sampling == "random":
                key = jax.random.PRNGKey(chunk_start)
                X_chunk = jax.random.uniform(
                    key, (chunk_size, len(self.bounds)),
                    minval=jnp.array([b[0] for b in self.bounds]),
                    maxval=jnp.array([b[1] for b in self.bounds])
                )
            
            # Evaluate chunk
            y_chunk = jnp.array([self.function(x) for x in X_chunk])
            
            all_X.append(X_chunk)
            all_y.append(y_chunk)
            
            # Clear chunk from memory explicitly
            del X_chunk, y_chunk
        
        return Dataset(X=jnp.vstack(all_X), y=jnp.concatenate(all_y))
```

### Memory-Conscious Model Selection

```python
def select_memory_conscious_surrogate(n_samples, n_dims, memory_budget_mb=1000):
    """Select surrogate based on memory constraints"""
    
    # Rough memory estimates (highly approximate)
    gp_memory = n_samples**2 * 8 / 1e6  # GP needs O(n²) storage
    nn_memory = (n_dims * 128 + 128 * 64 + 64 * 1) * 4 / 1e6  # NN parameters
    rf_memory = n_samples * n_dims * 8 / 1e6 * 0.7  # RF stores subset of data
    
    print(f"Memory estimates (MB):")
    print(f"  GP: {gp_memory:.1f}")
    print(f"  Neural Network: {nn_memory:.1f}")
    print(f"  Random Forest: {rf_memory:.1f}")
    
    if gp_memory < memory_budget_mb and n_samples < 5000:
        return "gaussian_process", {}
    elif nn_memory < memory_budget_mb:
        return "neural_network", {"hidden_layers": [128, 64]}
    else:
        return "random_forest", {"n_estimators": min(100, memory_budget_mb // 10)}

# Usage
surrogate_type, params = select_memory_conscious_surrogate(
    n_samples=2000, 
    n_dims=20, 
    memory_budget_mb=500
)
print(f"Selected: {surrogate_type} with params {params}")
```

---

## Caching Strategies

### Basic Caching

```python
from surrogate_optim.performance import SurrogateCache, enable_caching

# Method 1: Wrap surrogate with cache
optimizer = SurrogateOptimizer(surrogate_type="neural_network")
optimizer.fit_surrogate(dataset)

# Add caching
optimizer.surrogate = enable_caching(
    optimizer.surrogate,
    cache_type="lru",
    max_size=1000
)

# Method 2: Manual cache management
cache = SurrogateCache(max_size=5000)

def cached_predict(surrogate, x):
    # Check cache first
    cached_result = cache.get(x, "predict")
    if cached_result is not None:
        return cached_result
    
    # Compute and cache
    result = surrogate.predict(x)
    cache.set(x, "predict", result)
    return result
```

### Persistent Caching

```python
from surrogate_optim.performance.caching import PersistentCache
import os

class PersistentOptimizer(SurrogateOptimizer):
    def __init__(self, cache_dir="./surrogate_cache", **kwargs):
        super().__init__(**kwargs)
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize persistent cache
        self.persistent_cache = PersistentCache(
            cache_dir=cache_dir,
            max_size_mb=1000  # 1GB cache limit
        )
    
    def fit_surrogate(self, data):
        # Check if surrogate already trained
        cache_key = self._generate_cache_key(data)
        cached_surrogate = self.persistent_cache.load_surrogate(cache_key)
        
        if cached_surrogate is not None:
            print("Loaded surrogate from cache")
            self.surrogate = cached_surrogate
            self.is_fitted = True
            self.training_data = data
            return self
        
        # Train and cache
        super().fit_surrogate(data)
        self.persistent_cache.save_surrogate(cache_key, self.surrogate)
        return self
    
    def _generate_cache_key(self, data):
        """Generate cache key based on data and model configuration"""
        data_hash = hash((data.X.tobytes(), data.y.tobytes()))
        config_hash = hash(str(sorted(self.surrogate_params.items())))
        return f"{self.surrogate_type}_{data_hash}_{config_hash}"

# Usage with persistent caching
optimizer = PersistentOptimizer(
    cache_dir="./my_cache",
    surrogate_type="neural_network",
    surrogate_params={"n_epochs": 200}
)

# First run: trains and caches
optimizer.fit_surrogate(dataset)

# Subsequent runs: loads from cache
optimizer2 = PersistentOptimizer(
    cache_dir="./my_cache", 
    surrogate_type="neural_network",
    surrogate_params={"n_epochs": 200}
)
optimizer2.fit_surrogate(dataset)  # Loads from cache
```

### Smart Cache Strategies

```python
class SmartCache:
    """Cache with intelligent eviction and precomputation"""
    
    def __init__(self, max_size=1000, precompute_neighbors=True):
        self.cache = {}
        self.access_count = {}
        self.max_size = max_size
        self.precompute_neighbors = precompute_neighbors
    
    def get(self, x, operation):
        key = self._make_key(x, operation)
        
        if key in self.cache:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return self.cache[key]
        
        return None
    
    def set(self, x, operation, value):
        key = self._make_key(x, operation)
        
        # Evict if necessary
        if len(self.cache) >= self.max_size:
            self._evict_least_valuable()
        
        self.cache[key] = value
        self.access_count[key] = 1
        
        # Precompute nearby points if beneficial
        if self.precompute_neighbors and operation == "predict":
            self._precompute_neighbors(x, operation)
    
    def _evict_least_valuable(self):
        """Evict based on access frequency and recency"""
        if not self.cache:
            return
        
        # Score each item (higher = more valuable)
        scores = {}
        for key in self.cache:
            access_freq = self.access_count.get(key, 1)
            scores[key] = access_freq  # Simple frequency-based scoring
        
        # Remove lowest scoring item
        worst_key = min(scores.keys(), key=lambda k: scores[k])
        del self.cache[worst_key]
        del self.access_count[worst_key]
    
    def _precompute_neighbors(self, x, operation):
        """Precompute predictions for nearby points"""
        # This would require access to the surrogate model
        pass
    
    def _make_key(self, x, operation):
        return f"{operation}_{hash(x.tobytes())}"
```

---

## Parallel Processing

### Data Collection Parallelization

```python
from surrogate_optim.performance import ParallelDataCollector
import multiprocessing as mp

def expensive_black_box(x):
    """Simulate expensive computation"""
    import time
    time.sleep(0.1)
    return sum(x**2) + 0.1 * sum(jnp.sin(10 * x))

bounds = [(-2, 2)] * 5

# Sequential collection (baseline)
start_time = time.time()
sequential_collector = DataCollector(expensive_black_box, bounds)
dataset_seq = sequential_collector.collect(n_samples=100)
sequential_time = time.time() - start_time

# Parallel collection
start_time = time.time()
parallel_collector = ParallelDataCollector(
    expensive_black_box, 
    bounds,
    n_jobs=mp.cpu_count()
)
dataset_par = parallel_collector.collect(n_samples=100)
parallel_time = time.time() - start_time

print(f"Sequential time: {sequential_time:.2f}s")
print(f"Parallel time: {parallel_time:.2f}s")
print(f"Speedup: {sequential_time / parallel_time:.2f}x")
```

### Vectorized Operations

```python
from surrogate_optim.performance import VectorizedSurrogate

class HighPerformanceOptimizer(SurrogateOptimizer):
    def fit_surrogate(self, data):
        super().fit_surrogate(data)
        
        # Wrap surrogate for vectorized operations
        self.surrogate = VectorizedSurrogate(
            self.surrogate, 
            batch_size=1000  # Process 1000 points at once
        )
        return self

# Usage for batch predictions
optimizer = HighPerformanceOptimizer(surrogate_type="neural_network")
optimizer.fit_surrogate(dataset)

# These operations are now vectorized and faster
test_points = jax.random.normal(jax.random.PRNGKey(0), (5000, 10))

start_time = time.time()
predictions = optimizer.predict(test_points)  # Vectorized
vectorized_time = time.time() - start_time

print(f"Vectorized prediction time: {vectorized_time:.3f}s for {len(test_points)} points")
```

### Parallel Multi-Start Optimization

```python
from concurrent.futures import ProcessPoolExecutor, as_completed
import jax.numpy as jnp

def parallel_multi_start_optimization(
    surrogate, bounds, n_starts=10, n_jobs=None
):
    """Parallel multi-start optimization"""
    
    if n_jobs is None:
        n_jobs = min(n_starts, mp.cpu_count())
    
    # Generate starting points
    key = jax.random.PRNGKey(42)
    starts = []
    for i in range(n_starts):
        key, subkey = jax.random.split(key)
        start = jax.random.uniform(
            subkey, (len(bounds),),
            minval=jnp.array([b[0] for b in bounds]),
            maxval=jnp.array([b[1] for b in bounds])
        )
        starts.append(start)
    
    # Define optimization task
    def optimize_from_start(start_point):
        from surrogate_optim.optimizers import GradientDescentOptimizer
        optimizer = GradientDescentOptimizer(max_iterations=100)
        return optimizer.optimize(surrogate, start_point, bounds)
    
    # Run parallel optimization
    results = []
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # Submit all tasks
        future_to_start = {
            executor.submit(optimize_from_start, start): start 
            for start in starts
        }
        
        # Collect results
        for future in as_completed(future_to_start):
            start = future_to_start[future]
            try:
                result = future.result()
                results.append((start, result))
            except Exception as e:
                print(f"Optimization from {start} failed: {e}")
    
    # Find best result
    successful_results = [(s, r) for s, r in results if r.success]
    if successful_results:
        best_start, best_result = min(successful_results, key=lambda x: x[1].fun)
        print(f"Best result from {len(successful_results)}/{n_starts} successful starts")
        return best_result
    else:
        print("No successful optimizations")
        return None

# Usage
result = parallel_multi_start_optimization(
    optimizer.surrogate, 
    bounds, 
    n_starts=20, 
    n_jobs=8
)
```

---

## Optimization Tuning

### Adaptive Learning Rates

```python
from surrogate_optim.optimizers.gradient_descent import GradientDescentOptimizer

class AdaptiveGradientDescent(GradientDescentOptimizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss_history = []
        self.lr_history = []
    
    def _update_learning_rate(self, iteration, loss_value):
        """Adaptive learning rate based on loss improvement"""
        if len(self.loss_history) > 0:
            improvement = self.loss_history[-1] - loss_value
            
            if improvement > 1e-3:  # Good improvement
                self.learning_rate *= 1.05  # Increase
            elif improvement < 1e-6:  # Poor improvement
                self.learning_rate *= 0.95  # Decrease
            
            # Keep within reasonable bounds
            self.learning_rate = jnp.clip(self.learning_rate, 1e-6, 1.0)
        
        self.loss_history.append(loss_value)
        self.lr_history.append(self.learning_rate)

# Usage
optimizer = SurrogateOptimizer(
    surrogate_type="neural_network",
    optimizer_type="gradient_descent",
    optimizer_params={"learning_rate": 0.01, "adaptive_lr": True}
)

# Override with custom adaptive optimizer
optimizer.optimizer = AdaptiveGradientDescent(
    learning_rate=0.01,
    max_iterations=200
)
```

### Early Stopping

```python
class EarlyStoppingOptimizer:
    def __init__(self, base_optimizer, patience=10, min_improvement=1e-6):
        self.base_optimizer = base_optimizer
        self.patience = patience
        self.min_improvement = min_improvement
    
    def optimize(self, surrogate, x0, bounds=None, **kwargs):
        """Optimize with early stopping"""
        
        # Track best result
        best_loss = float('inf')
        best_x = x0
        patience_counter = 0
        
        # Override base optimizer to add early stopping logic
        original_max_iter = self.base_optimizer.max_iterations
        self.base_optimizer.max_iterations = 1  # Step-by-step control
        
        current_x = x0
        for iteration in range(original_max_iter):
            # Single step
            step_result = self.base_optimizer.optimize(
                surrogate, current_x, bounds, **kwargs
            )
            
            current_x = step_result.x
            current_loss = float(surrogate.predict(current_x.reshape(1, -1))[0])
            
            # Check for improvement
            if best_loss - current_loss > self.min_improvement:
                best_loss = current_loss
                best_x = current_x
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.patience:
                print(f"Early stopping at iteration {iteration}")
                break
        
        # Restore original max_iterations
        self.base_optimizer.max_iterations = original_max_iter
        
        from surrogate_optim.optimizers.base import OptimizationResult
        return OptimizationResult(
            x=best_x,
            fun=best_loss,
            success=True,
            message=f"Converged with early stopping at iteration {iteration}",
            nit=iteration + 1,
            nfev=iteration + 1
        )

# Usage
base_opt = GradientDescentOptimizer(max_iterations=1000)
early_stopping_opt = EarlyStoppingOptimizer(
    base_opt, patience=20, min_improvement=1e-8
)

result = early_stopping_opt.optimize(surrogate, x0, bounds)
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import ParameterGrid
import time

def tune_surrogate_hyperparameters(function, bounds, param_grid, n_samples=200):
    """Tune surrogate hyperparameters using grid search"""
    
    # Collect test data once
    dataset = collect_data(function, n_samples, bounds, sampling="sobol")
    
    # Split into train/validation
    n_train = int(0.8 * n_samples)
    train_indices = jnp.arange(n_train)
    val_indices = jnp.arange(n_train, n_samples)
    
    train_dataset = Dataset(
        X=dataset.X[train_indices], 
        y=dataset.y[train_indices]
    )
    val_X = dataset.X[val_indices]
    val_y = dataset.y[val_indices]
    
    best_params = None
    best_score = float('inf')
    results = []
    
    for params in ParameterGrid(param_grid):
        print(f"Testing params: {params}")
        
        start_time = time.time()
        
        try:
            # Train surrogate
            if "surrogate_type" in params:
                surrogate_type = params.pop("surrogate_type")
            else:
                surrogate_type = "neural_network"
            
            optimizer = SurrogateOptimizer(
                surrogate_type=surrogate_type,
                surrogate_params=params
            )
            
            optimizer.fit_surrogate(train_dataset)
            
            # Validate
            val_pred = optimizer.predict(val_X)
            mse = float(jnp.mean((val_pred - val_y)**2))
            
            train_time = time.time() - start_time
            
            results.append({
                "params": params.copy(),
                "mse": mse,
                "train_time": train_time,
                "surrogate_type": surrogate_type
            })
            
            if mse < best_score:
                best_score = mse
                best_params = params.copy()
                best_params["surrogate_type"] = surrogate_type
            
            print(f"  MSE: {mse:.6f}, Time: {train_time:.2f}s")
            
        except Exception as e:
            print(f"  Failed: {e}")
    
    return best_params, best_score, results

# Define parameter grid
param_grid = [
    {
        "surrogate_type": ["neural_network"],
        "hidden_layers": [[64, 32], [128, 64], [256, 128, 64]],
        "n_epochs": [50, 100, 200],
        "learning_rate": [0.001, 0.01]
    },
    {
        "surrogate_type": ["gaussian_process"],
        "kernel": ["rbf", "matern"],
        "length_scale": [0.1, 1.0, 10.0]
    }
]

# Tune hyperparameters
best_params, best_score, all_results = tune_surrogate_hyperparameters(
    rosenbrock, bounds, param_grid
)

print(f"\nBest parameters: {best_params}")
print(f"Best validation MSE: {best_score:.6f}")
```

This performance guide provides comprehensive strategies for optimizing the computational efficiency, memory usage, and overall performance of surrogate optimization workflows.