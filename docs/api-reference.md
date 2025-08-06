# API Reference

## Core Classes

### SurrogateOptimizer

**`surrogate_optim.core.SurrogateOptimizer`**

Main interface for surrogate gradient optimization.

#### Constructor

```python
SurrogateOptimizer(
    surrogate_type: str = "neural_network",
    surrogate_params: Optional[Dict[str, Any]] = None,
    optimizer_type: str = "gradient_descent", 
    optimizer_params: Optional[Dict[str, Any]] = None
)
```

**Parameters:**
- `surrogate_type`: Type of surrogate model ("neural_network", "gaussian_process", "random_forest")
- `surrogate_params`: Parameters for surrogate model configuration
- `optimizer_type`: Type of optimizer ("gradient_descent", "trust_region", "multi_start")
- `optimizer_params`: Parameters for optimizer configuration

#### Methods

##### `fit_surrogate(data)`

Train the surrogate model on the given data.

**Parameters:**
- `data`: Training data as `Dataset` or dict with 'X' and 'y' keys

**Returns:** Self for method chaining

**Example:**
```python
optimizer = SurrogateOptimizer()
optimizer.fit_surrogate(dataset)
```

##### `optimize(initial_point, bounds=None, method="L-BFGS-B", num_steps=100, **kwargs)`

Optimize using the trained surrogate.

**Parameters:**
- `initial_point`: Starting point for optimization (Array)
- `bounds`: Optional bounds for each dimension (List[Tuple[float, float]])
- `method`: Optimization method (currently ignored, uses configured optimizer)
- `num_steps`: Maximum number of optimization steps (int)
- `**kwargs`: Additional optimizer arguments

**Returns:** `OptimizationResult`

##### `predict(x)`

Predict function values using the trained surrogate.

**Parameters:**
- `x`: Input points for prediction (Array)

**Returns:** Predicted function values (Array)

##### `gradient(x)`

Compute gradients using the trained surrogate.

**Parameters:**
- `x`: Input points for gradient computation (Array)

**Returns:** Gradient vectors (Array)

##### `uncertainty(x)`

Estimate prediction uncertainty (if supported by surrogate).

**Parameters:**
- `x`: Input points for uncertainty estimation (Array)

**Returns:** Uncertainty estimates (Array)

##### `validate(test_function, test_points=None, n_test_points=100, metrics=["mse", "gradient_error"])`

Validate surrogate against true function.

**Parameters:**
- `test_function`: True function for validation
- `test_points`: Optional test points (random if None)
- `n_test_points`: Number of test points if test_points is None
- `metrics`: Validation metrics to compute

**Returns:** Dictionary of validation metrics

---

## Surrogate Models

### Base Surrogate

**`surrogate_optim.models.base.Surrogate`**

Abstract base class for surrogate models.

#### Methods

##### `fit(dataset)`
Train the surrogate on the given dataset.

##### `predict(x)`
Make predictions at input points.

##### `gradient(x)`
Compute gradients at input points.

##### `uncertainty(x)`
Estimate prediction uncertainty.

### Neural Network Surrogate

**`surrogate_optim.models.neural.NeuralSurrogate`**

Neural network-based surrogate model using JAX.

#### Constructor

```python
NeuralSurrogate(
    hidden_layers: List[int] = [64, 32],
    activation: str = "tanh",
    learning_rate: float = 0.001,
    n_epochs: int = 100,
    batch_size: int = 32,
    random_seed: int = 42
)
```

### Gaussian Process Surrogate

**`surrogate_optim.models.gaussian_process.GPSurrogate`**

Gaussian Process surrogate with automatic differentiation.

#### Constructor

```python
GPSurrogate(
    kernel: str = "rbf",
    length_scale: float = 1.0,
    noise_level: float = 0.1,
    optimize_hyperparameters: bool = True
)
```

### Random Forest Surrogate

**`surrogate_optim.models.random_forest.RandomForestSurrogate`**

Random Forest surrogate with gradient approximation.

#### Constructor

```python
RandomForestSurrogate(
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    random_seed: int = 42
)
```

---

## Optimization Methods

### Base Optimizer

**`surrogate_optim.optimizers.base.Optimizer`**

Abstract base class for optimization methods.

#### Methods

##### `optimize(surrogate, x0, bounds=None, **kwargs)`

Perform optimization using the given surrogate.

**Returns:** `OptimizationResult`

### Gradient Descent Optimizer

**`surrogate_optim.optimizers.gradient_descent.GradientDescentOptimizer`**

Gradient-based optimization with adaptive learning rate.

#### Constructor

```python
GradientDescentOptimizer(
    learning_rate: float = 0.01,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    adaptive_lr: bool = True,
    momentum: float = 0.0
)
```

### Trust Region Optimizer

**`surrogate_optim.optimizers.trust_region.TrustRegionOptimizer`**

Trust region optimization with surrogate validation.

#### Constructor

```python
TrustRegionOptimizer(
    initial_radius: float = 1.0,
    max_radius: float = 10.0,
    min_radius: float = 1e-6,
    validation_frequency: int = 5,
    shrink_factor: float = 0.25,
    expand_factor: float = 2.0
)
```

### Multi-Start Optimizer

**`surrogate_optim.optimizers.multi_start.MultiStartOptimizer`**

Global optimization using multiple starting points.

#### Constructor

```python
MultiStartOptimizer(
    n_starts: int = 5,
    local_optimizer: str = "gradient_descent",
    start_method: str = "sobol",
    parallel: bool = False,
    n_jobs: int = -1
)
```

---

## Data Collection

### DataCollector

**`surrogate_optim.data.collector.DataCollector`**

Handles sampling and data collection for surrogate training.

#### Constructor

```python
DataCollector(
    function: Callable,
    bounds: List[Tuple[float, float]],
    parallel: bool = False,
    n_jobs: int = 1
)
```

#### Methods

##### `collect(n_samples, sampling="random", verbose=True)`

Collect training data using specified sampling strategy.

**Parameters:**
- `n_samples`: Number of samples to collect
- `sampling`: Sampling method ("random", "lhs", "sobol", "grid")
- `verbose`: Whether to show progress

**Returns:** `Dataset`

### Dataset

**`surrogate_optim.models.base.Dataset`**

Container for training data.

#### Attributes

- `X`: Input data (Array)
- `y`: Output values (Array) 
- `gradients`: Optional gradient data (Array)
- `metadata`: Additional information (Dict)

#### Properties

- `n_samples`: Number of samples
- `n_dims`: Input dimensionality

---

## Results and Analysis

### OptimizationResult

**`surrogate_optim.optimizers.base.OptimizationResult`**

Container for optimization results.

#### Attributes

- `x`: Optimal point found (Array)
- `fun`: Function value at optimum (float)
- `success`: Whether optimization succeeded (bool)
- `message`: Status message (str)
- `nit`: Number of iterations (int)
- `nfev`: Number of function evaluations (int)
- `trajectory`: Optional optimization path (List[Array])
- `convergence_history`: Optional convergence values (List[float])
- `metadata`: Additional information (Dict)

---

## Validation

### Convergence Validation

**`surrogate_optim.validation.convergence_validation.validate_convergence(result, convergence_criteria=None)`**

Validate optimization convergence.

**Parameters:**
- `result`: OptimizationResult to validate
- `convergence_criteria`: Optional convergence criteria

**Returns:** Dictionary with validation results

### Input Validation

**`surrogate_optim.validation.input_validation.validate_inputs(X, y, bounds=None)`**

Validate input data for training.

**Parameters:**
- `X`: Input data
- `y`: Output values  
- `bounds`: Optional bounds

**Returns:** Validation results

---

## Performance Tools

### Caching

**`surrogate_optim.performance.caching.SurrogateCache`**

LRU cache for surrogate predictions.

#### Constructor

```python
SurrogateCache(max_size=1000)
```

### Memory Monitoring

**`surrogate_optim.performance.memory.MemoryMonitor`**

Monitor memory usage during optimization.

#### Usage

```python
with MemoryMonitor() as monitor:
    # Your code here
    pass

peak_memory = monitor.get_peak_memory()
```

### Performance Profiling

**`surrogate_optim.performance.profiling.performance_monitor`**

Context manager for profiling operations.

#### Usage

```python
with performance_monitor("optimization", profile_level="detailed") as profiler:
    result = optimizer.optimize(x0, bounds)

summary = profiler.get_performance_summary()
```

---

## Utility Functions

### Quick Optimization

**`surrogate_optim.core.quick_optimize(function, bounds, n_samples=100, initial_point=None, surrogate_type="neural_network", verbose=True)`**

Convenience function for quick optimization with default settings.

**Parameters:**
- `function`: Black-box function to optimize
- `bounds`: Bounds for each input dimension
- `n_samples`: Number of training samples
- `initial_point`: Starting point (random if None)
- `surrogate_type`: Type of surrogate model
- `verbose`: Whether to print progress

**Returns:** `OptimizationResult`

### Data Collection

**`surrogate_optim.data.collector.collect_data(function, n_samples, bounds, sampling="random", verbose=True)`**

Convenience function for data collection.

**Parameters:**
- `function`: Function to sample
- `n_samples`: Number of samples
- `bounds`: Sampling bounds
- `sampling`: Sampling strategy
- `verbose`: Show progress

**Returns:** `Dataset`

### Optimizer Comparison

**`surrogate_optim.optimizers.utils.compare_optimizers(surrogate, x0, bounds=None, methods=None, options_dict=None)`**

Compare different optimization methods.

**Parameters:**
- `surrogate`: Trained surrogate model
- `x0`: Initial point
- `bounds`: Optional bounds
- `methods`: List of methods to compare
- `options_dict`: Method-specific options

**Returns:** Dictionary mapping method names to results

---

## Exceptions

### ValidationError

**`surrogate_optim.validation.input_validation.ValidationError`**

Raised when input validation fails.

### ValidationWarning

**`surrogate_optim.validation.input_validation.ValidationWarning`**

Warning for potential validation issues.

---

## Configuration

Most classes accept configuration through their constructors. Common patterns:

```python
# Neural network configuration
nn_config = {
    "hidden_layers": [128, 64, 32],
    "learning_rate": 0.001,
    "n_epochs": 200
}

# Optimizer configuration  
opt_config = {
    "learning_rate": 0.01,
    "max_iterations": 100,
    "adaptive_lr": True
}

# Create configured optimizer
optimizer = SurrogateOptimizer(
    surrogate_type="neural_network",
    surrogate_params=nn_config,
    optimizer_type="gradient_descent",
    optimizer_params=opt_config
)
```