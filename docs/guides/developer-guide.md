# Developer Guide: Surrogate Gradient Optimization Lab

## Table of Contents

1. [Development Setup](#development-setup)
2. [Architecture Overview](#architecture-overview)
3. [Contributing Guidelines](#contributing-guidelines)
4. [Adding New Surrogate Models](#adding-new-surrogate-models)
5. [Testing Strategy](#testing-strategy)
6. [Performance Optimization](#performance-optimization)
7. [Release Process](#release-process)

## Development Setup

### Prerequisites

- Python 3.9+
- Git
- Make (optional, for convenience)
- Docker (for containerized development)

### Local Development

```bash
# Clone the repository
git clone https://github.com/terragon-labs/surrogate-gradient-optim-lab
cd surrogate-gradient-optim-lab

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Verify installation
pytest tests/ -v
```

### Development Container

```bash
# Use the devcontainer
code .  # Open in VS Code
# Then: Ctrl/Cmd+Shift+P -> "Dev Containers: Reopen in Container"

# Or manually with Docker
docker build -t surrogate-optim-dev -f .devcontainer/Dockerfile .
docker run -it -v $(pwd):/workspace surrogate-optim-dev bash
```

### Development Workflow

```bash
# Start development
git checkout -b feature/your-feature-name

# Make changes and test
pytest tests/ -v
ruff check .
mypy surrogate_optim/

# Commit with pre-commit hooks
git add .
git commit -m "feat: add your feature"

# Push and create PR
git push origin feature/your-feature-name
```

## Architecture Overview

### Core Components

```
surrogate_optim/
├── core/                 # Core abstractions and interfaces
│   ├── surrogate.py     # Base Surrogate class
│   ├── optimizer.py     # Base Optimizer class
│   └── data.py          # Data structures and utilities
├── models/              # Surrogate model implementations
│   ├── neural.py        # Neural network surrogates
│   ├── gaussian.py      # Gaussian process surrogates
│   ├── forest.py        # Random forest surrogates
│   └── hybrid.py        # Hybrid/ensemble models
├── optimizers/          # Optimization algorithms
│   ├── gradient.py      # Gradient-based optimizers
│   ├── trust_region.py  # Trust region methods
│   └── global_opt.py    # Global optimization strategies
├── data/                # Data collection and processing
│   ├── collectors.py    # Data collection strategies
│   ├── samplers.py      # Sampling methods
│   └── preprocessing.py # Data preprocessing
├── visualization/       # Plotting and analysis tools
│   ├── gradients.py     # Gradient visualization
│   ├── landscapes.py    # Optimization landscapes
│   └── diagnostics.py   # Model diagnostics
├── benchmarks/          # Benchmark functions and evaluation
│   ├── functions.py     # Test functions
│   ├── real_world.py    # Real-world problems
│   └── evaluation.py    # Evaluation metrics
└── utils/               # Utilities and helpers
    ├── math.py          # Mathematical utilities
    ├── io.py            # Input/output utilities
    └── config.py        # Configuration management
```

### Design Principles

1. **Modularity**: Each component has a clear, single responsibility
2. **Extensibility**: Easy to add new surrogate models and optimizers
3. **Performance**: JAX-based implementation for speed and GPU support
4. **Usability**: Simple high-level API with powerful low-level access
5. **Testability**: Comprehensive test coverage with clear test structure

### Key Abstractions

```python
# Base surrogate interface
class Surrogate(ABC):
    @abstractmethod
    def fit(self, X: Array, y: Array) -> None:
        """Train the surrogate model."""
    
    @abstractmethod
    def predict(self, x: Array) -> float:
        """Predict function value at point x."""
    
    @abstractmethod
    def gradient(self, x: Array) -> Array:
        """Compute surrogate gradient at point x."""
    
    def uncertainty(self, x: Array) -> float:
        """Estimate prediction uncertainty (optional)."""
        return 0.0

# Base optimizer interface
class Optimizer(ABC):
    @abstractmethod
    def optimize(self, surrogate: Surrogate, x0: Array, **kwargs) -> OptimizationResult:
        """Optimize using the surrogate model."""
```

## Contributing Guidelines

### Code Style

We use several tools to maintain code quality:

```bash
# Format code
black surrogate_optim/ tests/
isort surrogate_optim/ tests/

# Check style and quality
ruff check surrogate_optim/ tests/
mypy surrogate_optim/

# Security checks
bandit -r surrogate_optim/
safety check
```

### Commit Message Format

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:
```
feat(models): add transformer-based surrogate model
fix(optimizers): handle edge case in trust region update
docs(api): update docstrings for new surrogate interface
```

### Pull Request Process

1. **Create feature branch** from `main`
2. **Make changes** following code style guidelines
3. **Add tests** for new functionality
4. **Update documentation** if needed
5. **Run full test suite** and ensure all checks pass
6. **Create PR** with clear description and link to issues
7. **Address review feedback** promptly
8. **Squash and merge** once approved

## Adding New Surrogate Models

### 1. Create Model Class

```python
# surrogate_optim/models/your_model.py
from typing import Optional
import jax.numpy as jnp
from jax import grad, Array
from ..core.surrogate import Surrogate

class YourSurrogate(Surrogate):
    """Your custom surrogate model.
    
    Args:
        param1: Description of parameter 1
        param2: Description of parameter 2
    """
    
    def __init__(self, param1: float = 1.0, param2: int = 10):
        self.param1 = param1
        self.param2 = param2
        self._is_fitted = False
    
    def fit(self, X: Array, y: Array) -> None:
        """Train the surrogate model."""
        # Implement your training logic here
        self._X_train = X
        self._y_train = y
        # ... training code ...
        self._is_fitted = True
    
    def predict(self, x: Array) -> float:
        """Predict function value at point x."""
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Implement your prediction logic
        # This should be JAX-compatible for automatic differentiation
        return jnp.sum(x**2)  # Example
    
    def gradient(self, x: Array) -> Array:
        """Compute surrogate gradient at point x."""
        # JAX can automatically compute gradients
        grad_fn = grad(self.predict)
        return grad_fn(x)
    
    def uncertainty(self, x: Array) -> float:
        """Estimate prediction uncertainty."""
        # Optional: implement uncertainty quantification
        return 0.0
```

### 2. Register Model

```python
# surrogate_optim/models/__init__.py
from .your_model import YourSurrogate

__all__ = [
    # ... existing models ...
    "YourSurrogate",
]

# Register in model factory
MODEL_REGISTRY = {
    # ... existing models ...
    "your_model": YourSurrogate,
}
```

### 3. Add Tests

```python
# tests/models/test_your_model.py
import pytest
import jax.numpy as jnp
from surrogate_optim.models import YourSurrogate

class TestYourSurrogate:
    @pytest.fixture
    def sample_data(self):
        X = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        y = jnp.array([0.0, 2.0, 8.0])
        return X, y
    
    def test_fit_predict(self, sample_data):
        X, y = sample_data
        model = YourSurrogate()
        
        # Test fitting
        model.fit(X, y)
        assert model._is_fitted
        
        # Test prediction
        pred = model.predict(jnp.array([1.0, 1.0]))
        assert isinstance(pred, float)
    
    def test_gradient_computation(self, sample_data):
        X, y = sample_data
        model = YourSurrogate()
        model.fit(X, y)
        
        x_test = jnp.array([1.0, 1.0])
        grad = model.gradient(x_test)
        
        assert grad.shape == x_test.shape
        assert jnp.all(jnp.isfinite(grad))
    
    def test_unfitted_model_error(self):
        model = YourSurrogate()
        with pytest.raises(ValueError, match="must be fitted"):
            model.predict(jnp.array([0.0, 0.0]))
```

### 4. Add Documentation

```python
# docs/api/models.md - add your model to the API docs

## YourSurrogate

::: surrogate_optim.models.YourSurrogate
    options:
      show_source: true
      show_signature: true
```

### 5. Add Example

```python
# examples/custom_surrogate_example.py
from surrogate_optim.models import YourSurrogate
from surrogate_optim import collect_data
import jax.numpy as jnp

# Example usage of your custom surrogate
def example_function(x):
    return jnp.sum(x**2) + 0.1 * jnp.sin(10 * jnp.linalg.norm(x))

# Collect data
data = collect_data(
    function=example_function,
    n_samples=100,
    bounds=[(-2, 2), (-2, 2)],
    sampling="random"
)

# Use your custom surrogate
surrogate = YourSurrogate(param1=2.0, param2=20)
surrogate.fit(data.X, data.y)

# Test prediction and gradients
test_point = jnp.array([1.0, 1.0])
prediction = surrogate.predict(test_point)
gradient = surrogate.gradient(test_point)

print(f"Prediction: {prediction}")
print(f"Gradient: {gradient}")
```

## Testing Strategy

### Test Structure

```
tests/
├── unit/                # Unit tests for individual components
│   ├── models/         # Surrogate model tests
│   ├── optimizers/     # Optimizer tests
│   └── utils/          # Utility function tests
├── integration/        # Integration tests
│   ├── test_workflows.py  # End-to-end workflows
│   └── test_benchmarks.py # Benchmark evaluations
├── benchmarks/         # Performance benchmarks
│   └── test_performance.py
├── fixtures/           # Test data and fixtures
└── conftest.py         # Pytest configuration
```

### Test Categories

1. **Unit Tests**: Test individual functions and classes
2. **Integration Tests**: Test component interactions
3. **Property Tests**: Use Hypothesis for property-based testing
4. **Performance Tests**: Benchmark critical code paths
5. **GPU Tests**: Test CUDA functionality (if available)

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest -m "not slow"  # Skip slow tests
pytest -m gpu  # Only GPU tests

# Run with coverage
pytest --cov=surrogate_optim --cov-report=html

# Run performance benchmarks
pytest tests/benchmarks/ --benchmark-save=baseline
```

### Writing Good Tests

```python
import pytest
import jax.numpy as jnp
from hypothesis import given, strategies as st

class TestOptimizationFunction:
    """Example of well-structured tests."""
    
    @pytest.fixture
    def simple_quadratic(self):
        """Simple test function."""
        return lambda x: jnp.sum(x**2)
    
    def test_basic_functionality(self, simple_quadratic):
        """Test basic functionality with known inputs."""
        result = optimize_function(simple_quadratic, jnp.array([1.0, 1.0]))
        assert jnp.allclose(result.x, jnp.array([0.0, 0.0]), atol=1e-6)
    
    @given(st.arrays(jnp.float32, shape=(2,), elements=st.floats(-10, 10)))
    def test_property_based(self, x0):
        """Property-based test with random inputs."""
        result = optimize_function(lambda x: jnp.sum(x**2), x0)
        # Properties that should always hold
        assert result.success
        assert result.fun <= (jnp.sum(x0**2) + 1e-6)  # Should improve
    
    @pytest.mark.slow
    def test_large_scale(self):
        """Test with large inputs (marked as slow)."""
        # Expensive test that runs only when explicitly requested
        pass
    
    @pytest.mark.gpu
    def test_gpu_acceleration(self):
        """Test GPU functionality."""
        # Only runs if GPU is available
        pass
```

## Performance Optimization

### JAX Best Practices

```python
from jax import jit, vmap, grad
import jax.numpy as jnp

# 1. JIT compile functions
@jit
def fast_prediction(params, x):
    """JIT-compiled prediction function."""
    return jnp.dot(params, x)

# 2. Vectorize operations
vectorized_predict = vmap(fast_prediction, in_axes=(None, 0))
batch_predictions = vectorized_predict(params, batch_x)

# 3. Use pure functions
def pure_gradient_step(x, grad_fn, learning_rate):
    """Pure function for gradient steps."""
    return x - learning_rate * grad_fn(x)

# 4. Avoid Python loops in hot paths
# Bad:
for i in range(n_steps):
    x = x - lr * grad_fn(x)

# Good:
from jax.lax import scan

def optimization_step(carry, _):
    x, lr = carry
    new_x = x - lr * grad_fn(x)
    return (new_x, lr), new_x

(final_x, _), trajectory = scan(optimization_step, (x0, lr), None, length=n_steps)
```

### Memory Management

```python
# Use gradient checkpointing for large models
from jax import checkpoint

@checkpoint
def expensive_forward_pass(params, x):
    """Checkpointed forward pass saves memory."""
    # ... expensive computation ...
    return result

# Process large datasets in batches
def batch_process(data, batch_size=32):
    """Process data in batches to manage memory."""
    n_batches = len(data) // batch_size
    results = []
    
    for i in range(n_batches):
        batch = data[i*batch_size:(i+1)*batch_size]
        batch_result = process_batch(batch)
        results.append(batch_result)
    
    return jnp.concatenate(results)
```

### Profiling and Debugging

```bash
# Profile with py-spy
py-spy top --pid <python-process-id>
py-spy record -o profile.svg -- python your_script.py

# JAX profiling
python -c "import jax; jax.profiler.start_trace('/tmp/tensorboard'); your_code(); jax.profiler.stop_trace()"
tensorboard --logdir=/tmp/tensorboard

# Memory profiling
mprof run your_script.py
mprof plot
```

## Release Process

### Version Management

We use [semantic versioning](https://semver.org/):
- **Major**: Breaking changes
- **Minor**: New features (backward compatible)
- **Patch**: Bug fixes

### Automated Releases

```bash
# Releases are automated via GitHub Actions
# 1. Create release PR with version bump
git checkout -b release/v0.2.0
# Update version in pyproject.toml
# Update CHANGELOG.md
git commit -m "chore: bump version to 0.2.0"
git push origin release/v0.2.0

# 2. Merge to main triggers release
# 3. GitHub Action builds and publishes to PyPI
# 4. Creates GitHub release with changelog
```

### Manual Release (if needed)

```bash
# Build package
python -m build

# Check package
twine check dist/*

# Upload to PyPI
twine upload dist/*

# Create GitHub release
gh release create v0.2.0 --notes-file CHANGELOG.md
```

---

For more information, see:
- [Architecture Decision Records](../adr/)
- [API Reference](../api/)
- [Contributing Guidelines](../../CONTRIBUTING.md)