# Examples

This document provides comprehensive examples of using the Surrogate Gradient Optimization Lab for various optimization scenarios.

## Table of Contents

1. [Basic Optimization](#basic-optimization)
2. [Advanced Surrogate Configuration](#advanced-surrogate-configuration)
3. [Multi-Start Global Optimization](#multi-start-global-optimization)
4. [High-Dimensional Problems](#high-dimensional-problems)
5. [Noisy Function Optimization](#noisy-function-optimization)
6. [Constraint Handling](#constraint-handling)
7. [Custom Validation and Analysis](#custom-validation-and-analysis)
8. [Performance Optimization](#performance-optimization)
9. [Visualization Examples](#visualization-examples)

---

## Basic Optimization

### Simple Quadratic Function

```python
import jax.numpy as jnp
from surrogate_optim import quick_optimize

# Define a simple quadratic function
def quadratic(x):
    """Simple quadratic: f(x) = sum(x^2)"""
    return jnp.sum(x**2)

# Optimize with default settings
bounds = [(-5, 5), (-5, 5)]
result = quick_optimize(
    function=quadratic,
    bounds=bounds,
    n_samples=100,
    verbose=True
)

print(f"Optimum: {result.x}")
print(f"Value: {result.fun:.6f}")
```

### Rosenbrock Function

```python
from surrogate_optim import SurrogateOptimizer, DataCollector
import jax.numpy as jnp

def rosenbrock(x):
    """Rosenbrock function: challenging for optimization"""
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

# Collect training data
bounds = [(-2, 2), (-2, 2)]
collector = DataCollector(rosenbrock, bounds)
dataset = collector.collect(n_samples=200, sampling="sobol")

# Configure optimizer
optimizer = SurrogateOptimizer(
    surrogate_type="neural_network",
    surrogate_params={"n_epochs": 150, "hidden_layers": [64, 32]},
    optimizer_type="trust_region"
)

# Train and optimize
optimizer.fit_surrogate(dataset)
result = optimizer.optimize(
    initial_point=jnp.array([-1.0, 1.0]),
    bounds=bounds
)

print(f"Rosenbrock optimum: {result.x}")
print(f"Expected: [1.0, 1.0]")
print(f"Error: {jnp.linalg.norm(result.x - jnp.array([1.0, 1.0])):.6f}")
```

---

## Advanced Surrogate Configuration

### Comparing Surrogate Models

```python
from surrogate_optim import SurrogateOptimizer, DataCollector
import jax.numpy as jnp

# Test function: Ackley function
def ackley(x):
    """Ackley function: multi-modal with many local minima"""
    a, b, c = 20, 0.2, 2 * jnp.pi
    d = len(x)
    sum1 = jnp.sum(x**2)
    sum2 = jnp.sum(jnp.cos(c * x))
    return -a * jnp.exp(-b * jnp.sqrt(sum1 / d)) - jnp.exp(sum2 / d) + a + jnp.e

bounds = [(-5, 5)] * 2
collector = DataCollector(ackley, bounds)
dataset = collector.collect(n_samples=300, sampling="lhs")

# Test different surrogate models
surrogates = {
    "neural_network": {
        "hidden_layers": [128, 64, 32],
        "n_epochs": 200,
        "learning_rate": 0.001
    },
    "gaussian_process": {
        "kernel": "rbf",
        "length_scale": 1.0,
        "optimize_hyperparameters": True
    },
    "random_forest": {
        "n_estimators": 200,
        "max_depth": 15
    }
}

results = {}
initial_point = jnp.array([2.0, 3.0])

for name, params in surrogates.items():
    print(f"\nTesting {name}...")
    
    optimizer = SurrogateOptimizer(
        surrogate_type=name,
        surrogate_params=params,
        optimizer_type="multi_start",
        optimizer_params={"n_starts": 5}
    )
    
    optimizer.fit_surrogate(dataset)
    
    # Validate surrogate
    validation = optimizer.validate(
        ackley, 
        n_test_points=100,
        metrics=["mse", "r2"]
    )
    
    # Optimize
    result = optimizer.optimize(initial_point, bounds)
    
    results[name] = {
        "result": result,
        "validation": validation
    }
    
    print(f"  Validation R²: {validation['r2']:.4f}")
    print(f"  Best value: {result.fun:.6f}")
    print(f"  Success: {result.success}")

# Find best performer
best_surrogate = min(results.items(), key=lambda x: x[1]['result'].fun)
print(f"\nBest surrogate: {best_surrogate[0]} (f = {best_surrogate[1]['result'].fun:.6f})")
```

### Custom Neural Network Architecture

```python
from surrogate_optim.models.neural import NeuralSurrogate
import jax.numpy as jnp

# Complex test function
def branin(x):
    """Branin function: standard benchmark"""
    x1, x2 = x[0], x[1]
    a, b, c, r, s, t = 1, 5.1/(4*jnp.pi**2), 5/jnp.pi, 6, 10, 1/(8*jnp.pi)
    return a * (x2 - b*x1**2 + c*x1 - r)**2 + s*(1-t)*jnp.cos(x1) + s

# Custom neural network with skip connections
class CustomNeuralSurrogate(NeuralSurrogate):
    def __init__(self, **kwargs):
        super().__init__(
            hidden_layers=[256, 128, 64, 32],
            activation="swish",  # Better than ReLU for smooth functions
            learning_rate=0.0005,
            n_epochs=300,
            batch_size=64,
            **kwargs
        )

# Use custom surrogate
bounds = [(-5, 10), (0, 15)]
optimizer = SurrogateOptimizer()
optimizer.surrogate = CustomNeuralSurrogate(random_seed=42)

# Collect data and optimize
collector = DataCollector(branin, bounds)
dataset = collector.collect(n_samples=400, sampling="sobol")

optimizer.fit_surrogate(dataset)
result = optimizer.optimize(jnp.array([0.0, 5.0]), bounds)

print(f"Branin minimum: {result.fun:.6f}")
print(f"Known minimum: 0.397887")
```

---

## Multi-Start Global Optimization

### Finding All Optima

```python
from surrogate_optim import SurrogateOptimizer
import jax.numpy as jnp
import matplotlib.pyplot as plt

def multi_modal(x):
    """Function with multiple local minima"""
    x1, x2 = x[0], x[1]
    return (jnp.sin(3*x1)**2 + (x1-0.5)**2) * (jnp.sin(3*x2)**2 + (x2-0.5)**2) + \
           0.1 * (x1**2 + x2**2)

bounds = [(-2, 2), (-2, 2)]

# Multi-start optimization
optimizer = SurrogateOptimizer(
    surrogate_type="gaussian_process",
    optimizer_type="multi_start",
    optimizer_params={
        "n_starts": 20,
        "local_optimizer": "trust_region",
        "start_method": "sobol",
        "parallel": True
    }
)

# Collect data
from surrogate_optim.data import collect_data
dataset = collect_data(multi_modal, n_samples=500, bounds=bounds, sampling="lhs")

optimizer.fit_surrogate(dataset)
result = optimizer.optimize(jnp.array([0.0, 0.0]), bounds)

# Access all local optima found
if result.metadata and 'local_results' in result.metadata:
    local_results = result.metadata['local_results']
    successful_results = [r for r in local_results if r.success]
    
    print(f"Found {len(successful_results)} successful local optimizations")
    
    # Group similar results (tolerance = 0.1)
    unique_optima = []
    for res in successful_results:
        is_unique = True
        for unique in unique_optima:
            if jnp.linalg.norm(res.x - unique.x) < 0.1:
                if res.fun < unique.fun:  # Better optimum
                    unique_optima.remove(unique)
                    unique_optima.append(res)
                is_unique = False
                break
        if is_unique:
            unique_optima.append(res)
    
    print(f"Found {len(unique_optima)} unique local optima:")
    for i, opt in enumerate(sorted(unique_optima, key=lambda x: x.fun)):
        print(f"  {i+1}: x = {opt.x}, f = {opt.fun:.6f}")
```

### Parallel Multi-Start with Custom Strategy

```python
from surrogate_optim.optimizers.multi_start import MultiStartOptimizer
from surrogate_optim.models.neural import NeuralSurrogate
import jax.numpy as jnp
import jax.random as random

# Expensive black-box function
def expensive_function(x):
    """Simulates an expensive evaluation"""
    import time
    time.sleep(0.01)  # Simulate expensive computation
    
    # Schwefel function
    return 418.9829 * len(x) - jnp.sum(x * jnp.sin(jnp.sqrt(jnp.abs(x))))

bounds = [(-500, 500)] * 3  # 3D problem

# Custom start generation
class CustomMultiStart(MultiStartOptimizer):
    def _generate_starts(self, bounds, n_starts):
        """Generate starts using Latin Hypercube + some known good regions"""
        key = random.PRNGKey(self.random_seed)
        
        # 80% LHS, 20% around known good regions  
        n_lhs = int(0.8 * n_starts)
        n_guided = n_starts - n_lhs
        
        # LHS starts
        from surrogate_optim.data.sampling import latin_hypercube_sampling
        lhs_starts = latin_hypercube_sampling(bounds, n_lhs, key)
        
        # Guided starts around [-420, -420, -420] (known good region for Schwefel)
        guided_starts = []
        for i in range(n_guided):
            key, subkey = random.split(key)
            noise = random.normal(subkey, (len(bounds),)) * 50  # Gaussian noise
            guided_start = jnp.array([-420.0] * len(bounds)) + noise
            
            # Clip to bounds
            guided_start = jnp.clip(
                guided_start, 
                jnp.array([b[0] for b in bounds]),
                jnp.array([b[1] for b in bounds])
            )
            guided_starts.append(guided_start)
        
        all_starts = jnp.vstack([lhs_starts] + guided_starts)
        return all_starts

# Use custom multi-start
optimizer = SurrogateOptimizer(
    surrogate_type="neural_network",
    surrogate_params={
        "hidden_layers": [128, 64, 32], 
        "n_epochs": 150
    }
)

optimizer.optimizer = CustomMultiStart(
    n_starts=12,
    local_optimizer="trust_region", 
    parallel=True,
    n_jobs=4
)

# Limited budget training
dataset = collect_data(
    expensive_function,
    n_samples=200,  # Limited samples for expensive function
    bounds=bounds,
    sampling="sobol",
    verbose=True
)

optimizer.fit_surrogate(dataset)
result = optimizer.optimize(jnp.array([0.0, 0.0, 0.0]), bounds)

print(f"Best value found: {result.fun:.2f}")
print(f"Known global minimum: ~{-418.9829 * 3:.2f}")
```

---

## High-Dimensional Problems

### Scaling to Higher Dimensions

```python
from surrogate_optim import SurrogateOptimizer
import jax.numpy as jnp

def high_dim_function(x):
    """Styblinski-Tang function: scalable to high dimensions"""
    return 0.5 * jnp.sum(x**4 - 16*x**2 + 5*x)

# Test different dimensions
dimensions = [5, 10, 20, 50]
results = {}

for d in dimensions:
    print(f"\nOptimizing in {d} dimensions...")
    bounds = [(-5, 5)] * d
    
    # Adapt sample size to dimension
    n_samples = max(500, 20 * d)
    
    # Use neural network for high-dimensional problems
    optimizer = SurrogateOptimizer(
        surrogate_type="neural_network",
        surrogate_params={
            "hidden_layers": [256, 128, 64],  # Larger network for higher dims
            "n_epochs": 200,
            "learning_rate": 0.0005,
            "batch_size": 64
        },
        optimizer_type="multi_start",
        optimizer_params={"n_starts": 5}
    )
    
    # Collect data
    dataset = collect_data(
        high_dim_function, 
        n_samples=n_samples, 
        bounds=bounds,
        sampling="sobol"
    )
    
    # Train with validation
    optimizer.fit_surrogate(dataset)
    validation = optimizer.validate(
        high_dim_function,
        n_test_points=min(200, 10*d),
        metrics=["mse", "r2"]
    )
    
    # Optimize
    initial_point = jnp.zeros(d)  # Start at origin
    result = optimizer.optimize(initial_point, bounds)
    
    results[d] = {
        "result": result,
        "validation_r2": validation["r2"],
        "known_minimum": -39.16617 * d  # Known minimum per dimension
    }
    
    print(f"  Validation R²: {validation['r2']:.4f}")
    print(f"  Best found: {result.fun:.2f}")
    print(f"  Known minimum: {results[d]['known_minimum']:.2f}")
    print(f"  Error: {abs(result.fun - results[d]['known_minimum']):.2f}")

# Performance analysis
print("\n=== Scaling Analysis ===")
for d, data in results.items():
    error = abs(data["result"].fun - data["known_minimum"])
    print(f"Dim {d:2d}: Error = {error:8.2f}, R² = {data['validation_r2']:.4f}, "
          f"Iterations = {data['result'].nit}")
```

### Dimensionality Reduction Approach

```python
from sklearn.decomposition import PCA
from surrogate_optim import SurrogateOptimizer
import jax.numpy as jnp

class DimensionalityReductionOptimizer:
    """Optimizer with automatic dimensionality reduction"""
    
    def __init__(self, target_dims=10, **surrogate_kwargs):
        self.target_dims = target_dims
        self.surrogate_kwargs = surrogate_kwargs
        self.pca = None
        self.original_bounds = None
        self.reduced_bounds = None
        
    def fit(self, function, bounds, n_samples=1000):
        """Fit PCA and train surrogate on reduced space"""
        self.original_bounds = bounds
        original_dims = len(bounds)
        
        if original_dims <= self.target_dims:
            # No reduction needed
            self.optimizer = SurrogateOptimizer(**self.surrogate_kwargs)
            dataset = collect_data(function, n_samples, bounds, sampling="sobol")
            self.optimizer.fit_surrogate(dataset)
            return
        
        # Collect data in original space
        print(f"Collecting data in {original_dims}D space...")
        dataset = collect_data(function, n_samples, bounds, sampling="lhs")
        
        # Fit PCA
        print(f"Reducing dimensionality: {original_dims}D -> {self.target_dims}D")
        self.pca = PCA(n_components=self.target_dims)
        X_reduced = self.pca.fit_transform(dataset.X)
        
        # Determine bounds in reduced space
        self.reduced_bounds = [
            (float(jnp.min(X_reduced[:, i])), float(jnp.max(X_reduced[:, i])))
            for i in range(self.target_dims)
        ]
        
        # Create reduced dataset
        from surrogate_optim.models.base import Dataset
        reduced_dataset = Dataset(
            X=jnp.array(X_reduced),
            y=dataset.y,
            metadata={"original_bounds": bounds, "pca_components": self.pca.components_}
        )
        
        # Train surrogate on reduced space
        self.optimizer = SurrogateOptimizer(**self.surrogate_kwargs)
        self.optimizer.fit_surrogate(reduced_dataset)
        
        print(f"PCA explained variance: {self.pca.explained_variance_ratio_.sum():.3f}")
    
    def optimize(self, initial_point_original):
        """Optimize in reduced space and map back"""
        if self.pca is None:
            # No reduction was used
            return self.optimizer.optimize(initial_point_original, self.original_bounds)
        
        # Transform initial point to reduced space
        initial_reduced = self.pca.transform(initial_point_original.reshape(1, -1))[0]
        
        # Optimize in reduced space
        result_reduced = self.optimizer.optimize(
            jnp.array(initial_reduced),
            self.reduced_bounds
        )
        
        # Transform result back to original space
        x_original = self.pca.inverse_transform(result_reduced.x.reshape(1, -1))[0]
        
        # Create result in original space
        from surrogate_optim.optimizers.base import OptimizationResult
        return OptimizationResult(
            x=jnp.array(x_original),
            fun=result_reduced.fun,  # Function value should be the same
            success=result_reduced.success,
            message=f"Optimized in {self.target_dims}D reduced space. " + result_reduced.message,
            nit=result_reduced.nit,
            nfev=result_reduced.nfev,
            metadata={
                "reduced_result": result_reduced,
                "pca_variance_explained": self.pca.explained_variance_ratio_.sum(),
                "original_dimensions": len(self.original_bounds),
                "reduced_dimensions": self.target_dims
            }
        )

# Example with 100D problem
def sphere_100d(x):
    """100D sphere function"""
    return jnp.sum(x**2)

bounds_100d = [(-5, 5)] * 100
initial_point_100d = jnp.zeros(100)

# Use dimensionality reduction
dr_optimizer = DimensionalityReductionOptimizer(
    target_dims=15,
    surrogate_type="neural_network",
    surrogate_params={"hidden_layers": [128, 64], "n_epochs": 150}
)

dr_optimizer.fit(sphere_100d, bounds_100d, n_samples=2000)
result = dr_optimizer.optimize(initial_point_100d)

print(f"100D optimization result:")
print(f"  Final value: {result.fun:.6f}")
print(f"  Success: {result.success}")
print(f"  PCA variance explained: {result.metadata['pca_variance_explained']:.3f}")
```

---

## Noisy Function Optimization

### Handling Observation Noise

```python
import jax.numpy as jnp
import jax.random as random
from surrogate_optim import SurrogateOptimizer

def noisy_function(x, noise_level=0.1):
    """Rosenbrock with additive Gaussian noise"""
    true_value = 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
    
    # Add noise
    key = random.PRNGKey(hash(tuple(x.tolist())) % 2**32)  # Deterministic noise based on input
    noise = random.normal(key) * noise_level * (1 + abs(true_value))  # Scale noise with function value
    
    return true_value + noise

bounds = [(-2, 2), (-2, 2)]

# Strategy 1: More data + robust surrogate
print("=== Strategy 1: More Data + GP ===")
optimizer1 = SurrogateOptimizer(
    surrogate_type="gaussian_process",
    surrogate_params={
        "kernel": "rbf",
        "noise_level": 0.2,  # Account for noise in GP
        "optimize_hyperparameters": True
    },
    optimizer_type="trust_region",
    optimizer_params={
        "validation_frequency": 3,  # More frequent validation
        "initial_radius": 0.5
    }
)

# Collect more data for noise robustness
dataset1 = collect_data(
    noisy_function, 
    n_samples=800,  # More data
    bounds=bounds, 
    sampling="lhs"
)

optimizer1.fit_surrogate(dataset1)
result1 = optimizer1.optimize(jnp.array([0.0, 0.0]), bounds)

print(f"GP Result: x = {result1.x}, f = {result1.fun:.6f}")

# Strategy 2: Ensemble method with Random Forest
print("\n=== Strategy 2: Ensemble Method ===")
optimizer2 = SurrogateOptimizer(
    surrogate_type="random_forest",
    surrogate_params={
        "n_estimators": 500,  # Large ensemble for noise robustness
        "max_features": "sqrt",
        "bootstrap": True,
        "min_samples_leaf": 3  # Prevent overfitting
    },
    optimizer_type="multi_start",
    optimizer_params={"n_starts": 8}
)

dataset2 = collect_data(noisy_function, n_samples=600, bounds=bounds, sampling="sobol")
optimizer2.fit_surrogate(dataset2)
result2 = optimizer2.optimize(jnp.array([0.0, 0.0]), bounds)

print(f"RF Result: x = {result2.x}, f = {result2.fun:.6f}")

# Strategy 3: Multiple runs with averaging
print("\n=== Strategy 3: Multiple Runs + Averaging ===")
def average_multiple_runs(optimizer, n_runs=5):
    """Run optimization multiple times and average results"""
    results = []
    
    for i in range(n_runs):
        # Add some randomness to initial point
        key = random.PRNGKey(i)
        noise = random.normal(key, (2,)) * 0.2
        initial = jnp.array([0.0, 0.0]) + noise
        
        result = optimizer.optimize(initial, bounds)
        if result.success:
            results.append(result)
    
    if not results:
        return None
    
    # Average the successful results
    avg_x = jnp.mean(jnp.stack([r.x for r in results]), axis=0)
    avg_fun = jnp.mean(jnp.array([r.fun for r in results]))
    
    return {
        "x": avg_x,
        "fun": avg_fun,
        "n_successful": len(results),
        "std_fun": jnp.std(jnp.array([r.fun for r in results])),
        "individual_results": results
    }

averaged_result = average_multiple_runs(optimizer1, n_runs=8)
if averaged_result:
    print(f"Averaged Result: x = {averaged_result['x']}, f = {averaged_result['fun']:.6f}")
    print(f"  Standard deviation: {averaged_result['std_fun']:.6f}")
    print(f"  Successful runs: {averaged_result['n_successful']}/8")

# Compare to true optimum
true_optimum = jnp.array([1.0, 1.0])
print(f"\n=== Comparison to True Optimum [1, 1] ===")
for name, result in [("GP", result1), ("RF", result2), ("Averaged", averaged_result)]:
    if result is not None:
        if isinstance(result, dict):
            x = result['x']
        else:
            x = result.x
        error = jnp.linalg.norm(x - true_optimum)
        print(f"{name:8}: Error = {error:.6f}")
```

### Noise-Aware Data Collection

```python
from surrogate_optim.data.collector import DataCollector
import jax.numpy as jnp
import jax.random as random

class NoiseAwareCollector(DataCollector):
    """Data collector that handles noisy functions with multiple evaluations"""
    
    def __init__(self, function, bounds, n_repeats=3, **kwargs):
        self.n_repeats = n_repeats
        super().__init__(function, bounds, **kwargs)
    
    def _evaluate_function(self, x):
        """Evaluate function multiple times and return statistics"""
        values = []
        for _ in range(self.n_repeats):
            values.append(self.function(x))
        
        values = jnp.array(values)
        return {
            "mean": jnp.mean(values),
            "std": jnp.std(values),
            "min": jnp.min(values),
            "max": jnp.max(values),
            "values": values
        }
    
    def collect(self, n_samples, sampling="lhs", verbose=True):
        """Collect data with noise statistics"""
        points = self._generate_samples(n_samples, sampling)
        
        evaluations = []
        for i, x in enumerate(points):
            if verbose and (i + 1) % 50 == 0:
                print(f"Evaluating point {i+1}/{n_samples}")
            
            eval_stats = self._evaluate_function(x)
            evaluations.append(eval_stats)
        
        # Use mean values for training
        y_mean = jnp.array([e["mean"] for e in evaluations])
        y_std = jnp.array([e["std"] for e in evaluations])
        
        from surrogate_optim.models.base import Dataset
        return Dataset(
            X=points,
            y=y_mean,
            metadata={
                "bounds": self.bounds,
                "noise_std": y_std,
                "evaluation_stats": evaluations,
                "n_repeats": self.n_repeats
            }
        )

# Use noise-aware collector
collector = NoiseAwareCollector(
    noisy_function, 
    bounds, 
    n_repeats=5  # 5 evaluations per point
)

dataset = collector.collect(n_samples=200, sampling="sobol", verbose=True)
noise_levels = dataset.metadata["noise_std"]

print(f"Average noise level: {jnp.mean(noise_levels):.4f}")
print(f"Max noise level: {jnp.max(noise_levels):.4f}")

# Train GP with estimated noise
optimizer = SurrogateOptimizer(
    surrogate_type="gaussian_process",
    surrogate_params={
        "noise_level": float(jnp.mean(noise_levels)),  # Use estimated noise
        "optimize_hyperparameters": True
    }
)

optimizer.fit_surrogate(dataset)
result = optimizer.optimize(jnp.array([0.0, 0.0]), bounds)
print(f"Noise-aware result: x = {result.x}, f = {result.fun:.6f}")
```

This comprehensive examples documentation provides users with practical, working code for various optimization scenarios. Each example builds in complexity and demonstrates different aspects of the surrogate optimization library.