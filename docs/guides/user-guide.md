# User Guide: Surrogate Gradient Optimization Lab

## Table of Contents

1. [Getting Started](#getting-started)
2. [Core Concepts](#core-concepts)
3. [Basic Usage](#basic-usage)
4. [Advanced Features](#advanced-features)
5. [Troubleshooting](#troubleshooting)
6. [Best Practices](#best-practices)

## Getting Started

### Installation

```bash
# Basic installation
pip install surrogate-gradient-optim-lab

# With GPU support
pip install surrogate-gradient-optim-lab[gpu]

# Full development setup
pip install surrogate-gradient-optim-lab[all]
```

### Quick Verification

```python
import surrogate_optim
print(f"Version: {surrogate_optim.__version__}")

# Run a simple test
from surrogate_optim import quick_test
quick_test()  # Should pass without errors
```

## Core Concepts

### What are Surrogate Gradients?

Surrogate gradients are learned approximations of true gradients for black-box functions. When you have:

- **No access to gradients** (black-box simulators, hardware, human feedback)
- **Expensive function evaluations** (minutes to hours per evaluation)
- **Some offline data** (previous experiments, simulations)

You can learn a differentiable surrogate that enables gradient-based optimization.

### The Optimization Workflow

1. **Collect Data**: Gather function evaluations from your black-box system
2. **Train Surrogate**: Learn a differentiable model that approximates your function
3. **Compute Gradients**: Extract gradients from the surrogate model
4. **Optimize**: Use gradient-based methods to find optimal points
5. **Validate**: Check surrogate predictions against true function when possible

## Basic Usage

### Simple 2D Optimization

```python
from surrogate_optim import SurrogateOptimizer, collect_data
import jax.numpy as jnp

# Define your expensive black-box function
def expensive_simulation(x):
    # This could be a complex simulation, hardware test, etc.
    return -(x[0]**2 + x[1]**2) + 0.1 * jnp.sin(10 * jnp.linalg.norm(x))

# Step 1: Collect some data
data = collect_data(
    function=expensive_simulation,
    n_samples=200,
    bounds=[(-2, 2), (-2, 2)],
    sampling="sobol"  # Space-filling design
)

# Step 2: Create and train surrogate
optimizer = SurrogateOptimizer(
    surrogate_type="neural_network",
    hidden_dims=[64, 64],
    activation="relu"
)

surrogate = optimizer.fit_surrogate(data)

# Step 3: Optimize using learned gradients
x_optimal = optimizer.optimize(
    initial_point=jnp.array([1.0, 1.0]),
    method="L-BFGS-B",
    bounds=[(-2, 2), (-2, 2)]
)

print(f"Optimal point: {x_optimal}")
print(f"Function value: {expensive_simulation(x_optimal):.4f}")
```

### Using Different Surrogate Types

```python
# Gaussian Process (good for smooth functions)
gp_optimizer = SurrogateOptimizer(
    surrogate_type="gaussian_process",
    kernel="rbf",
    noise_level=0.01
)

# Random Forest (good for noisy, irregular functions)
rf_optimizer = SurrogateOptimizer(
    surrogate_type="random_forest",
    n_estimators=100,
    max_depth=10
)

# Hybrid model (combines multiple approaches)
hybrid_optimizer = SurrogateOptimizer(
    surrogate_type="hybrid",
    models={
        "neural_network": {"hidden_dims": [32, 32]},
        "gaussian_process": {"kernel": "matern"},
        "random_forest": {"n_estimators": 50}
    },
    aggregation="weighted_average"
)
```

## Advanced Features

### Data Collection Strategies

```python
from surrogate_optim.data import DataCollector

collector = DataCollector(expensive_simulation)

# Active learning for efficient data collection
data = collector.collect_adaptive(
    initial_samples=50,
    acquisition_function="expected_improvement",
    batch_size=5,
    n_iterations=10,
    bounds=[(-2, 2), (-2, 2)]
)

# Use existing data
data = collector.load_from_csv("experiments.csv")

# Augment with gradient estimates
data_with_grads = collector.estimate_gradients(
    data,
    method="finite_differences",
    epsilon=1e-3
)
```

### Trust Region Optimization

```python
from surrogate_optim.optimizers import TrustRegionOptimizer

# Safer optimization with periodic validation
tr_optimizer = TrustRegionOptimizer(
    surrogate=trained_surrogate,
    true_function=expensive_simulation,  # For validation
    initial_radius=0.5,
    max_radius=2.0
)

trajectory = tr_optimizer.optimize(
    x0=jnp.array([0.5, 0.5]),
    max_iterations=50,
    validate_every=5
)
```

### Multi-Start Global Optimization

```python
from surrogate_optim.optimizers import MultiStartOptimizer

ms_optimizer = MultiStartOptimizer(
    surrogate=trained_surrogate,
    n_starts=20,
    start_method="sobol"
)

global_result = ms_optimizer.optimize_global(
    bounds=[(-5, 5), (-5, 5)],
    parallel=True
)

print(f"Global optimum: {global_result.best_point}")
print(f"Function value: {global_result.best_value}")
```

### Visualization and Analysis

```python
from surrogate_optim.visualization import (
    GradientVisualizer, 
    LandscapeVisualizer,
    OptimizationAnalyzer
)

# Compare gradients
viz = GradientVisualizer()
viz.plot_gradient_fields(
    true_function=expensive_simulation,
    surrogate=trained_surrogate,
    bounds=[(-2, 2), (-2, 2)]
)

# Optimization landscape
landscape = LandscapeVisualizer()
landscape.plot_optimization_landscape(
    surrogate=trained_surrogate,
    optimization_paths=trajectory,
    show_minima=True
)

# Detailed analysis
analyzer = OptimizationAnalyzer()
report = analyzer.analyze_optimization(
    surrogate=trained_surrogate,
    true_function=expensive_simulation,
    optimization_result=global_result
)
print(report)
```

## Troubleshooting

### Common Issues

#### Poor Surrogate Accuracy

```python
# Check data quality
from surrogate_optim.diagnostics import DataDiagnostics

diag = DataDiagnostics()
diag.analyze_data_quality(data)
diag.plot_data_distribution()

# Solutions:
# 1. Collect more data in poorly covered regions
# 2. Try different surrogate types
# 3. Adjust hyperparameters
# 4. Use ensemble methods
```

#### Optimization Gets Stuck

```python
# Use multi-start optimization
ms_optimizer = MultiStartOptimizer(
    surrogate=surrogate,
    n_starts=50,  # Increase number of starting points
    start_method="latin_hypercube"  # Better space coverage
)

# Or try trust region methods
tr_optimizer = TrustRegionOptimizer(
    surrogate=surrogate,
    initial_radius=0.1,  # Start with smaller radius
    shrink_factor=0.5    # More conservative updates
)
```

#### Memory Issues with Large Datasets

```python
# Use batch training
optimizer = SurrogateOptimizer(
    surrogate_type="neural_network",
    batch_size=32,        # Smaller batches
    use_checkpointing=True  # Save memory during training
)

# Or subsample data
from surrogate_optim.data import subsample_data
smaller_data = subsample_data(data, n_samples=1000, method="diverse")
```

### Performance Tips

```python
# Enable GPU acceleration
import jax
print(f"Available devices: {jax.devices()}")

# Use JAX transformations for speed
from jax import jit, vmap

# Vectorize predictions
vectorized_predict = vmap(surrogate.predict)
batch_predictions = vectorized_predict(test_points)

# JIT compile optimization loop
@jit
def fast_optimize_step(x, grad_fn):
    return x - 0.01 * grad_fn(x)
```

## Best Practices

### Data Collection

1. **Use space-filling designs** (Sobol, Latin hypercube) for initial sampling
2. **Collect diverse data** - avoid clustering in similar regions
3. **Include boundary points** - many optima occur at boundaries
4. **Balance exploration vs exploitation** in active learning

### Surrogate Selection

1. **Neural networks** - Good for high-dimensional, complex functions
2. **Gaussian processes** - Best for smooth, low-dimensional functions  
3. **Random forests** - Robust for noisy, irregular functions
4. **Hybrid models** - Combine strengths when unsure

### Optimization Strategy

1. **Always validate** surrogate predictions against true function
2. **Use trust regions** for safety-critical applications
3. **Try multiple starting points** for global optimization
4. **Monitor convergence** and stop early if needed

### Debugging and Validation

```python
# Always validate your surrogate
from surrogate_optim.validation import validate_surrogate

validation_result = validate_surrogate(
    surrogate=trained_surrogate,
    true_function=expensive_simulation,
    test_points=test_data,
    metrics=["mae", "mse", "r2", "gradient_error"]
)

print(f"Validation R²: {validation_result.r2:.3f}")
print(f"Gradient error: {validation_result.gradient_error:.3f}")

# Plot validation results
validation_result.plot_predictions()
validation_result.plot_residuals()
```

---

For more advanced usage and API details, see the [Developer Guide](developer-guide.md) and [API Reference](../api/index.md).