# Surrogate Gradient Optimization Lab - User Guide

## Overview

The Surrogate Gradient Optimization Lab provides a comprehensive toolkit for offline black-box optimization using learned gradient surrogates. This guide walks you through the main features and typical workflows.

## Quick Start

### Basic Usage

```python
from surrogate_optim import quick_optimize
import jax.numpy as jnp

# Define your black-box function
def rosenbrock(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

# Define search bounds
bounds = [(-2, 2), (-2, 2)]

# Run optimization with default settings
result = quick_optimize(
    function=rosenbrock,
    bounds=bounds,
    n_samples=200,  # Training samples
    surrogate_type="neural_network"
)

print(f"Optimum found at {result.x} with value {result.fun:.6f}")
```

### Step-by-Step Workflow

For more control over the optimization process:

```python
from surrogate_optim import SurrogateOptimizer, DataCollector

# 1. Collect training data
collector = DataCollector(rosenbrock, bounds)
dataset = collector.collect(n_samples=200, sampling="sobol")

# 2. Create and configure optimizer
optimizer = SurrogateOptimizer(
    surrogate_type="neural_network",
    surrogate_params={"n_epochs": 100, "hidden_layers": [64, 32]},
    optimizer_type="trust_region"
)

# 3. Train surrogate
optimizer.fit_surrogate(dataset)

# 4. Optimize
initial_point = jnp.array([0.0, 0.0])
result = optimizer.optimize(initial_point, bounds=bounds)
```

## Surrogate Models

### Neural Network Surrogate

Best for complex, non-linear functions with smooth gradients:

```python
surrogate_params = {
    "hidden_layers": [128, 64, 32],  # Network architecture
    "n_epochs": 200,                 # Training epochs
    "learning_rate": 0.001,          # Learning rate
    "random_seed": 42                # For reproducibility
}

optimizer = SurrogateOptimizer(
    surrogate_type="neural_network",
    surrogate_params=surrogate_params
)
```

### Gaussian Process Surrogate

Ideal for smooth functions with uncertainty quantification:

```python
surrogate_params = {
    "kernel": "rbf",           # Kernel type
    "length_scale": 1.0,       # RBF kernel parameter
    "noise_level": 0.1         # Observation noise
}

optimizer = SurrogateOptimizer(
    surrogate_type="gaussian_process",
    surrogate_params=surrogate_params
)
```

### Random Forest Surrogate

Good for non-smooth, discrete, or categorical functions:

```python
surrogate_params = {
    "n_estimators": 100,       # Number of trees
    "max_depth": 10,           # Tree depth
    "min_samples_split": 5     # Minimum samples to split
}

optimizer = SurrogateOptimizer(
    surrogate_type="random_forest",
    surrogate_params=surrogate_params
)
```

## Optimization Methods

### Gradient Descent

Simple and efficient for well-behaved surrogates:

```python
optimizer_params = {
    "learning_rate": 0.01,
    "max_iterations": 100,
    "adaptive_lr": True,
    "momentum": 0.9
}

optimizer = SurrogateOptimizer(
    optimizer_type="gradient_descent",
    optimizer_params=optimizer_params
)
```

### Trust Region

Robust optimization with surrogate validation:

```python
optimizer_params = {
    "initial_radius": 1.0,
    "max_radius": 10.0,
    "validation_frequency": 5,    # Validate every N steps
    "shrink_factor": 0.5,
    "expand_factor": 2.0
}

optimizer = SurrogateOptimizer(
    optimizer_type="trust_region",
    optimizer_params=optimizer_params
)
```

### Multi-Start Global Optimization

For finding global optima in multi-modal functions:

```python
optimizer_params = {
    "n_starts": 10,               # Number of starting points
    "local_optimizer": "gradient_descent",
    "start_method": "sobol",      # How to generate starts
    "parallel": True              # Run starts in parallel
}

optimizer = SurrogateOptimizer(
    optimizer_type="multi_start",
    optimizer_params=optimizer_params
)
```

## Data Collection Strategies

### Sampling Methods

```python
from surrogate_optim.data import collect_data

# Latin Hypercube Sampling (good coverage)
dataset = collect_data(function, bounds, n_samples=100, sampling="lhs")

# Sobol Sequence (low-discrepancy)
dataset = collect_data(function, bounds, n_samples=100, sampling="sobol")

# Random Sampling
dataset = collect_data(function, bounds, n_samples=100, sampling="random")

# Grid Sampling (for low dimensions)
dataset = collect_data(function, bounds, n_samples=100, sampling="grid")
```

### Adaptive Sampling

For expensive functions, use adaptive sampling to focus on promising regions:

```python
from surrogate_optim.data import AdaptiveCollector

collector = AdaptiveCollector(
    function=your_function,
    bounds=bounds,
    acquisition="ei"  # Expected Improvement
)

# Start with initial samples
initial_dataset = collector.collect_initial(n_samples=50)

# Adaptively add samples
for i in range(10):
    new_points = collector.suggest_next(n_points=5)
    collector.evaluate_and_add(new_points)

final_dataset = collector.get_dataset()
```

## Validation and Analysis

### Surrogate Validation

```python
# Validate surrogate accuracy
validation_metrics = optimizer.validate(
    test_function=your_function,
    n_test_points=100,
    metrics=["mse", "mae", "r2", "gradient_error"]
)

print(f"MSE: {validation_metrics['mse']:.4f}")
print(f"R²: {validation_metrics['r2']:.4f}")
```

### Optimization Analysis

```python
from surrogate_optim.validation import validate_convergence

# Analyze convergence properties
convergence_analysis = validate_convergence(result)
print(f"Converged: {convergence_analysis['converged']}")
print(f"Total improvement: {convergence_analysis['total_improvement']:.4f}")

# Efficiency analysis
from surrogate_optim.validation.convergence_validation import analyze_optimization_efficiency

efficiency = analyze_optimization_efficiency(result)
print(f"Improvement per iteration: {efficiency['improvement_per_iteration']:.4f}")
```

## Visualization

### Basic Plotting

```python
from surrogate_optim.visualization import plot_surrogate_comparison, plot_optimization_trajectory

# Compare surrogate vs true function (2D only)
plot_surrogate_comparison(
    surrogate=optimizer.surrogate,
    true_function=your_function,
    bounds=bounds,
    n_points=50
)

# Plot optimization path
plot_optimization_trajectory(
    result=result,
    true_function=your_function,
    bounds=bounds
)
```

### Advanced Visualization

```python
# Plot convergence history
import matplotlib.pyplot as plt

if result.convergence_history:
    plt.figure(figsize=(10, 6))
    plt.plot(result.convergence_history)
    plt.xlabel('Iteration')
    plt.ylabel('Function Value')
    plt.title('Convergence History')
    plt.yscale('log')
    plt.grid(True)
    plt.show()
```

## Performance Optimization

### Caching

Enable caching for repeated evaluations:

```python
from surrogate_optim.performance import enable_caching

# Enable surrogate prediction caching
optimizer.surrogate = enable_caching(
    optimizer.surrogate, 
    cache_type="lru", 
    max_size=1000
)
```

### Parallel Processing

```python
# For data collection
collector = DataCollector(
    your_function, 
    bounds, 
    parallel=True, 
    n_jobs=4
)

# For multi-start optimization
optimizer_params = {"parallel": True, "n_jobs": -1}  # Use all cores
```

### Memory Management

```python
from surrogate_optim.performance import MemoryMonitor

# Monitor memory usage
with MemoryMonitor() as monitor:
    result = optimizer.optimize(initial_point, bounds)

print(f"Peak memory: {monitor.get_peak_memory():.1f} MB")
```

## Best Practices

### Choosing Surrogate Models

- **Neural Networks**: Complex, smooth functions with high dimensionality
- **Gaussian Processes**: Smooth functions where uncertainty matters, lower dimensions
- **Random Forests**: Non-smooth, discrete features, or noisy observations

### Sample Size Guidelines

- Start with 10-20 samples per dimension
- Neural networks benefit from more data (100+ samples)
- GPs work well with fewer samples (50-200)
- Random forests are robust to sample size

### Optimization Method Selection

- **Gradient Descent**: Fast, good surrogates, smooth landscapes
- **Trust Region**: Robust, expensive true function evaluations
- **Multi-Start**: Multi-modal functions, global optimization

### Debugging Tips

1. **Validate your surrogate first**:
   ```python
   metrics = optimizer.validate(your_function, n_test_points=100)
   if metrics['r2'] < 0.8:
       print("Warning: Poor surrogate quality")
   ```

2. **Check convergence**:
   ```python
   if not result.success:
       print(f"Optimization failed: {result.message}")
   ```

3. **Monitor progress**:
   ```python
   # Enable verbose output
   result = optimizer.optimize(initial_point, bounds, verbose=True)
   ```

## Common Issues and Solutions

### Poor Surrogate Quality
- Increase training samples
- Try different surrogate types
- Check data quality and bounds
- Use adaptive sampling

### Optimization Not Converging
- Try different initial points
- Adjust optimizer parameters
- Use trust region method
- Check surrogate gradients

### Slow Performance
- Enable caching
- Use parallel processing
- Reduce surrogate complexity
- Profile memory usage

## Examples

See the `examples/` directory for complete working examples:

- `basic_optimization.py` - Simple optimization workflow
- `advanced_surrogates.py` - Comparing different surrogate models
- `multi_objective.py` - Multi-objective optimization
- `high_dimensional.py` - High-dimensional problems
- `noisy_functions.py` - Handling noisy observations

## API Reference

For detailed API documentation, see:
- [Core API Reference](api-reference.md)
- [Surrogate Models](surrogate-models.md)
- [Optimization Methods](optimization-methods.md)
- [Data Collection](data-collection.md)