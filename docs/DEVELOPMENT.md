# Development Guide

This guide provides comprehensive instructions for setting up a development environment and contributing to the Surrogate Gradient Optimization Lab project.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Development Workflow](#development-workflow)
4. [Code Standards](#code-standards)
5. [Testing](#testing)
6. [Documentation](#documentation)
7. [Debugging](#debugging)
8. [Performance Profiling](#performance-profiling)
9. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **Operating System**: Linux, macOS, or Windows (with WSL2)
- **Python**: 3.9 or higher
- **Memory**: At least 4GB RAM (8GB+ recommended)
- **Disk Space**: At least 2GB free space
- **Git**: Version 2.25 or higher

### Optional Requirements

- **Docker**: For containerized development
- **CUDA**: For GPU acceleration (optional)
- **VS Code**: Recommended IDE with extensions

## Environment Setup

### 1. Clone the Repository

```bash
git clone https://github.com/terragon-labs/surrogate-gradient-optim-lab.git
cd surrogate-gradient-optim-lab
```

### 2. Python Environment Setup

#### Option A: Using Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Upgrade pip and install build tools
pip install --upgrade pip setuptools wheel

# Install development dependencies
pip install -e ".[dev,docs,benchmark,notebook]"
```

#### Option B: Using Conda

```bash
# Create conda environment
conda create -n surrogate-optim python=3.9
conda activate surrogate-optim

# Install development dependencies
pip install -e ".[dev,docs,benchmark,notebook]"
```

#### Option C: Using Docker

```bash
# Build development container
docker-compose up -d surrogate-optim-dev

# Access container shell
docker-compose exec surrogate-optim-dev bash
```

### 3. Install Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Test pre-commit hooks
pre-commit run --all-files
```

### 4. Verify Installation

```bash
# Run basic tests
pytest tests/unit/test_sample.py -v

# Check code quality
make lint

# Verify imports work
python -c "import surrogate_optim; print('Installation successful!')"
```

## Development Workflow

### 1. Branch Management

We use Git Flow for branch management:

- `main`: Production-ready code
- `develop`: Integration branch for features
- `feature/*`: Feature development branches
- `hotfix/*`: Critical bug fixes
- `release/*`: Release preparation branches

#### Creating a Feature Branch

```bash
# Update main branch
git checkout main
git pull origin main

# Create feature branch
git checkout -b feature/your-feature-name

# Work on your feature...
git add .
git commit -m "feat: implement your feature"

# Push feature branch
git push origin feature/your-feature-name
```

### 2. Development Process

1. **Plan**: Create or reference an issue for your work
2. **Branch**: Create a feature branch from `main`
3. **Develop**: Write code following our standards
4. **Test**: Ensure all tests pass and add new tests
5. **Document**: Update documentation as needed
6. **Review**: Create a pull request for code review
7. **Merge**: Merge after approval and CI checks pass

### 3. Commit Message Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```bash
git commit -m "feat(models): add neural network surrogate implementation"
git commit -m "fix(optimization): resolve convergence issue in trust region method"
git commit -m "docs: update getting started guide with GPU setup"
```

## Code Standards

### 1. Python Style Guide

We follow PEP 8 with some modifications:

- **Line length**: 88 characters (Black default)
- **Import order**: Use isort with Black profile
- **Type hints**: Required for all public functions
- **Docstrings**: Google style for all public functions and classes

### 2. Code Formatting

Automated formatting with Black:

```bash
# Format all Python files
black surrogate_optim/ tests/ examples/

# Check formatting without changes
black --check surrogate_optim/ tests/ examples/
```

### 3. Import Sorting

Use isort for consistent import ordering:

```bash
# Sort imports
isort surrogate_optim/ tests/ examples/

# Check import sorting
isort --check-only surrogate_optim/ tests/ examples/
```

### 4. Linting

We use multiple linters for code quality:

```bash
# Run all linting checks
make lint

# Individual linters
ruff check surrogate_optim/ tests/
flake8 surrogate_optim/ tests/
mypy surrogate_optim/
bandit -r surrogate_optim/
```

### 5. Type Checking

Type hints are required for all public APIs:

```python
from typing import Optional, List, Tuple, Dict, Any
import jax
import jax.numpy as jnp

def optimize_surrogate(
    surrogate: "Surrogate",
    initial_point: jax.Array,
    bounds: List[Tuple[float, float]],
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
    callback: Optional[callable] = None
) -> Dict[str, Any]:
    """Optimize using a surrogate model.
    
    Args:
        surrogate: Trained surrogate model
        initial_point: Starting point for optimization
        bounds: Optimization bounds for each dimension
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance
        callback: Optional callback function
        
    Returns:
        Dictionary containing optimization results
        
    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If optimization fails
    """
    # Implementation here
    pass
```

### 6. Documentation Standards

#### Docstring Format

Use Google-style docstrings:

```python
def train_surrogate(
    X: jax.Array, 
    y: jax.Array, 
    model_type: str = "neural_network"
) -> "TrainedSurrogate":
    """Train a surrogate model on the given data.

    This function trains a surrogate model that can approximate the relationship
    between inputs X and outputs y, enabling gradient-based optimization.

    Args:
        X: Input features with shape (n_samples, n_features)
        y: Target values with shape (n_samples,)
        model_type: Type of surrogate model ("neural_network", "gaussian_process", 
            "random_forest")

    Returns:
        Trained surrogate model ready for optimization

    Raises:
        ValueError: If X and y have incompatible shapes
        RuntimeError: If training fails to converge

    Example:
        >>> import jax.numpy as jnp
        >>> from surrogate_optim import train_surrogate
        >>> X = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        >>> y = jnp.array([5.0, 7.0])
        >>> surrogate = train_surrogate(X, y, model_type="neural_network")
    """
    pass
```

#### Code Comments

- Use clear, concise comments for complex logic
- Avoid obvious comments
- Explain the "why", not the "what"
- Use TODO comments for future improvements

```python
# Good: Explains why
# Use trust region to ensure optimization stability with noisy surrogates
trust_radius = min(1.0, previous_step_quality * current_radius)

# Bad: States the obvious
# Set trust_radius to minimum of 1.0 and product
trust_radius = min(1.0, previous_step_quality * current_radius)

# TODO: Implement adaptive trust region sizing based on surrogate uncertainty
```

## Testing

### 1. Test Structure

```
tests/
├── unit/           # Unit tests for individual components
├── integration/    # Integration tests for component interaction
├── benchmarks/     # Performance and benchmark tests
├── fixtures/       # Test data and fixtures
└── conftest.py     # Shared pytest configuration
```

### 2. Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/          # Unit tests only
pytest tests/integration/   # Integration tests only
pytest tests/benchmarks/    # Benchmark tests only

# Run with coverage
pytest --cov=surrogate_optim --cov-report=html

# Run tests in parallel
pytest -n auto

# Run specific test
pytest tests/unit/test_models.py::TestNeuralSurrogate::test_training
```

### 3. Writing Tests

#### Unit Test Example

```python
import pytest
import jax.numpy as jnp
from surrogate_optim.models import NeuralSurrogate

class TestNeuralSurrogate:
    """Test neural network surrogate model."""
    
    def test_training_convergence(self, sample_data_2d):
        """Test that neural surrogate converges during training."""
        X, y = sample_data_2d["X"], sample_data_2d["y"]
        
        surrogate = NeuralSurrogate(
            hidden_dims=[32, 32],
            learning_rate=0.001,
            max_epochs=100
        )
        
        # Train model
        surrogate.fit(X, y)
        
        # Check that model makes reasonable predictions
        predictions = surrogate.predict(X[:10])
        
        assert predictions.shape == (10,)
        assert jnp.isfinite(predictions).all()
        assert jnp.abs(predictions - y[:10]).mean() < 1.0  # Reasonable error
    
    def test_gradient_computation(self, sample_data_2d):
        """Test that gradients are computed correctly."""
        X, y = sample_data_2d["X"], sample_data_2d["y"]
        
        surrogate = NeuralSurrogate()
        surrogate.fit(X, y)
        
        # Compute gradients
        test_point = X[0]
        gradient = surrogate.gradient(test_point)
        
        assert gradient.shape == test_point.shape
        assert jnp.isfinite(gradient).all()
        assert not jnp.allclose(gradient, 0.0)  # Should have non-zero gradients
    
    @pytest.mark.parametrize("hidden_dims", [[16], [32, 32], [64, 32, 16]])
    def test_different_architectures(self, sample_data_2d, hidden_dims):
        """Test training with different network architectures."""
        X, y = sample_data_2d["X"], sample_data_2d["y"]
        
        surrogate = NeuralSurrogate(hidden_dims=hidden_dims)
        
        # Should train without errors
        surrogate.fit(X, y)
        
        # Should make predictions
        predictions = surrogate.predict(X[:5])
        assert predictions.shape == (5,)
```

#### Integration Test Example

```python
class TestSurrogateOptimizationWorkflow:
    """Test complete surrogate optimization workflow."""
    
    def test_end_to_end_optimization(self, optimization_test_functions):
        """Test complete optimization workflow."""
        # Get test function
        rosenbrock = optimization_test_functions["rosenbrock"]
        
        # Generate training data
        from surrogate_optim.data import collect_data
        data = collect_data(
            function=rosenbrock,
            n_samples=100,
            bounds=[(-2, 2), (-2, 2)],
            sampling="sobol"
        )
        
        # Train surrogate
        from surrogate_optim.models import NeuralSurrogate
        surrogate = NeuralSurrogate()
        surrogate.fit(data.X, data.y)
        
        # Optimize using surrogate
        from surrogate_optim.optimizers import SurrogateOptimizer
        optimizer = SurrogateOptimizer(surrogate)
        result = optimizer.optimize(
            initial_point=jnp.array([0.0, 0.0]),
            bounds=[(-2, 2), (-2, 2)],
            max_iterations=50
        )
        
        # Check optimization result
        assert result.success
        assert result.fun < 1.0  # Should find good solution
        assert jnp.linalg.norm(result.x - jnp.array([1.0, 1.0])) < 0.5
```

### 4. Test Fixtures and Utilities

Use pytest fixtures for shared test data:

```python
# In conftest.py
@pytest.fixture
def trained_neural_surrogate(sample_data_2d):
    """Provide a pre-trained neural surrogate for testing."""
    from surrogate_optim.models import NeuralSurrogate
    
    X, y = sample_data_2d["X"], sample_data_2d["y"]
    surrogate = NeuralSurrogate(hidden_dims=[32, 32])
    surrogate.fit(X, y)
    
    return surrogate

# In test files
def test_optimization_with_trained_model(trained_neural_surrogate):
    """Test optimization using pre-trained surrogate."""
    # Use the fixture
    pass
```

## Documentation

### 1. API Documentation

We use Sphinx for API documentation:

```bash
# Build documentation
cd docs/
make html

# View documentation
open _build/html/index.html
```

### 2. Jupyter Notebooks

For tutorials and examples:

```bash
# Start Jupyter Lab
jupyter lab

# Or use the make command
make jupyter
```

### 3. Documentation Guidelines

- All public APIs must have docstrings
- Include examples in docstrings
- Keep documentation up to date with code changes
- Use clear, concise language
- Include diagrams for complex concepts

## Debugging

### 1. Debugging Tools

#### Python Debugger

```python
import pdb; pdb.set_trace()  # Set breakpoint

# Or use ipdb for enhanced debugging
import ipdb; ipdb.set_trace()
```

#### JAX Debugging

```python
# Disable JIT for debugging
import jax
jax.config.update("jax_disable_jit", True)

# Add debugging prints
def debug_function(x):
    print(f"Input: {x}")
    result = some_computation(x)
    print(f"Output: {result}")
    return result
```

#### VS Code Debugging

Add to `.vscode/launch.json`:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "env": {
                "PYTHONPATH": "${workspaceFolder}"
            }
        },
        {
            "name": "Python: Pytest",
            "type": "python",
            "request": "launch",
            "module": "pytest",
            "args": ["${workspaceFolder}/tests/"],
            "console": "integratedTerminal"
        }
    ]
}
```

### 2. Common Issues and Solutions

#### JAX Issues

```python
# Issue: JAX arrays are immutable
# Solution: Use .at[] syntax for updates
x = x.at[0].set(new_value)

# Issue: Shape errors
# Solution: Add explicit shape checking
assert x.shape == expected_shape, f"Expected {expected_shape}, got {x.shape}"

# Issue: Gradient computation errors  
# Solution: Check for NaN/inf values
assert jnp.isfinite(x).all(), "Input contains non-finite values"
```

## Performance Profiling

### 1. Timing Code

```python
import time
from contextlib import contextmanager

@contextmanager
def timer(description):
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"{description}: {elapsed:.3f} seconds")

# Usage
with timer("Training surrogate"):
    surrogate.fit(X, y)
```

### 2. Memory Profiling

```bash
# Install memory profiler
pip install memory-profiler

# Profile memory usage
python -m memory_profiler train_model.py

# Or use in code
from memory_profiler import profile

@profile
def train_model():
    # Your code here
    pass
```

### 3. JAX Profiling

```python
# Profile JAX operations
import jax.profiler

# Start profiler
jax.profiler.start_trace("/tmp/jax-trace")

# Your JAX code here
result = jax.jit(my_function)(data)

# Stop profiler
jax.profiler.stop_trace()

# View trace at chrome://tracing
```

### 4. Benchmark Tests

```python
import pytest

@pytest.mark.benchmark
def test_training_performance(benchmark, sample_data_2d):
    """Benchmark model training performance."""
    X, y = sample_data_2d["X"], sample_data_2d["y"]
    
    def train_model():
        surrogate = NeuralSurrogate()
        surrogate.fit(X, y)
        return surrogate
    
    result = benchmark(train_model)
    assert result is not None
```

## Troubleshooting

### 1. Installation Issues

#### CUDA/GPU Issues

```bash
# Check CUDA installation
nvidia-smi

# Install JAX with CUDA support
pip install jax[cuda11_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Verify GPU availability
python -c "import jax; print(jax.devices())"
```

#### Dependency Conflicts

```bash
# Create clean environment
conda create -n clean-env python=3.9
conda activate clean-env
pip install -e ".[dev]"

# Or use pip-tools
pip-compile pyproject.toml
pip-sync requirements.txt
```

### 2. Test Failures

#### Random Test Failures

```python
# Use fixed random seeds
import jax
key = jax.random.PRNGKey(42)

# Or set environment variable
export PYTHONHASHSEED=42
```

#### Memory Issues

```bash
# Increase available memory
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8

# Or use CPU-only mode
export JAX_PLATFORM_NAME=cpu
```

### 3. Performance Issues

#### Slow Training

```python
# Enable JIT compilation
@jax.jit
def training_step(params, x, y):
    # Training logic here
    pass

# Use vectorization
batched_function = jax.vmap(single_function)
```

#### Memory Leaks

```python
# Clear JAX cache periodically
jax.clear_caches()

# Use context managers for large arrays
with jax.disable_jit():
    # Non-JIT operations
    pass
```

### 4. Getting Help

1. **Check Documentation**: Start with project documentation
2. **Search Issues**: Look for similar issues on GitHub
3. **Ask Questions**: Create a new issue with:
   - Environment details
   - Minimal reproduction case
   - Expected vs actual behavior
   - Error messages and stack traces

4. **Community Resources**:
   - Project Discord/Slack (if available)
   - JAX community forums
   - Stack Overflow with relevant tags

---

This development guide should help you get started with contributing to the project. If you encounter any issues or have suggestions for improvements, please create an issue or submit a pull request.