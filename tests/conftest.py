"""Pytest configuration and shared fixtures for the test suite."""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Return path to test data directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """Create and cleanup temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(scope="session")
def rng_key() -> jax.Array:
    """Provide consistent random number generator key for tests."""
    return random.PRNGKey(42)


@pytest.fixture
def sample_data_1d(rng_key: jax.Array) -> Dict[str, jax.Array]:
    """Generate 1D sample data for testing."""
    n_samples = 100
    x_key, noise_key = random.split(rng_key)
    
    # Generate input points
    X = random.uniform(x_key, (n_samples, 1), minval=-5.0, maxval=5.0)
    
    # Generate function values with noise
    y_true = jnp.sin(X.flatten()) * jnp.exp(-0.1 * X.flatten()**2)
    noise = 0.1 * random.normal(noise_key, (n_samples,))
    y = y_true + noise
    
    return {
        "X": X,
        "y": y,
        "y_true": y_true,
        "noise": noise
    }


@pytest.fixture
def sample_data_2d(rng_key: jax.Array) -> Dict[str, jax.Array]:
    """Generate 2D sample data for testing."""
    n_samples = 200
    x_key, noise_key = random.split(rng_key)
    
    # Generate input points
    X = random.uniform(x_key, (n_samples, 2), minval=-3.0, maxval=3.0)
    
    # Generate function values (Rosenbrock-like function)
    x1, x2 = X[:, 0], X[:, 1]
    y_true = (1 - x1)**2 + 100 * (x2 - x1**2)**2
    
    # Add noise
    noise = 0.1 * random.normal(noise_key, (n_samples,))
    y = y_true + noise
    
    return {
        "X": X,
        "y": y,
        "y_true": y_true,
        "noise": noise
    }


@pytest.fixture
def sample_data_high_dim(rng_key: jax.Array) -> Dict[str, jax.Array]:
    """Generate high-dimensional sample data for testing."""
    n_samples = 500
    n_dims = 10
    x_key, noise_key = random.split(rng_key)
    
    # Generate input points
    X = random.normal(x_key, (n_samples, n_dims))
    
    # Generate function values (quadratic function)
    y_true = jnp.sum(X**2, axis=1) + 0.1 * jnp.sum(X**3, axis=1)
    
    # Add noise
    noise = 0.05 * random.normal(noise_key, (n_samples,))
    y = y_true + noise
    
    return {
        "X": X,
        "y": y,
        "y_true": y_true,
        "noise": noise
    }


@pytest.fixture
def optimization_test_functions() -> Dict[str, callable]:
    """Provide standard optimization test functions."""
    
    def rosenbrock(x: jax.Array) -> float:
        """Rosenbrock function."""
        return jnp.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
    
    def rastrigin(x: jax.Array) -> float:
        """Rastrigin function."""
        n = len(x)
        return 10 * n + jnp.sum(x**2 - 10 * jnp.cos(2 * jnp.pi * x))
    
    def ackley(x: jax.Array) -> float:
        """Ackley function."""
        n = len(x)
        sum_sq = jnp.sum(x**2)
        sum_cos = jnp.sum(jnp.cos(2 * jnp.pi * x))
        return -20 * jnp.exp(-0.2 * jnp.sqrt(sum_sq / n)) - jnp.exp(sum_cos / n) + 20 + jnp.e
    
    def sphere(x: jax.Array) -> float:
        """Sphere function."""
        return jnp.sum(x**2)
    
    return {
        "rosenbrock": rosenbrock,
        "rastrigin": rastrigin,
        "ackley": ackley,
        "sphere": sphere
    }


@pytest.fixture
def benchmark_bounds() -> Dict[str, List[Tuple[float, float]]]:
    """Provide bounds for benchmark functions."""
    return {
        "rosenbrock": [(-5.0, 5.0)] * 2,
        "rastrigin": [(-5.12, 5.12)] * 2,
        "ackley": [(-32.768, 32.768)] * 2,
        "sphere": [(-5.0, 5.0)] * 2
    }


@pytest.fixture
def model_configs() -> Dict[str, Dict[str, Any]]:
    """Provide configurations for different surrogate models."""
    return {
        "neural_network": {
            "hidden_dims": [32, 32],
            "activation": "relu",
            "learning_rate": 0.001,
            "epochs": 100
        },
        "gaussian_process": {
            "kernel": "rbf",
            "length_scale": 1.0,
            "noise_level": 0.1
        },
        "random_forest": {
            "n_estimators": 50,
            "max_depth": 10,
            "random_state": 42
        }
    }


@pytest.fixture(autouse=True)
def setup_jax_config():
    """Configure JAX for testing."""
    # Enable double precision for numerical stability in tests
    jax.config.update("jax_enable_x64", True)
    
    # Disable JIT for easier debugging in tests
    jax.config.update("jax_disable_jit", True)
    
    yield
    
    # Reset to defaults after tests
    jax.config.update("jax_enable_x64", False)
    jax.config.update("jax_disable_jit", False)


@pytest.fixture
def mock_expensive_function():
    """Mock expensive function that tracks number of evaluations."""
    class ExpensiveFunction:
        def __init__(self):
            self.n_evaluations = 0
            self.evaluation_history = []
        
        def __call__(self, x: jax.Array) -> float:
            self.n_evaluations += 1
            self.evaluation_history.append(x.copy())
            
            # Simulate expensive computation with simple function
            result = jnp.sum(x**2) + 0.1 * jnp.sum(jnp.sin(5 * x))
            return float(result)
        
        def reset(self):
            self.n_evaluations = 0
            self.evaluation_history = []
    
    return ExpensiveFunction()


# Performance testing utilities
@pytest.fixture
def performance_thresholds() -> Dict[str, float]:
    """Define performance thresholds for benchmarking."""
    return {
        "max_training_time": 10.0,  # seconds
        "max_prediction_time": 0.1,  # seconds
        "max_memory_usage": 100.0,  # MB
        "min_accuracy": 0.95,  # R² score
        "max_gradient_error": 0.1  # relative error
    }


# Parameterized test fixtures
@pytest.fixture(params=[1, 2, 5, 10])
def dimension(request) -> int:
    """Parameterized fixture for different dimensionalities."""
    return request.param


@pytest.fixture(params=[50, 100, 500])
def sample_size(request) -> int:
    """Parameterized fixture for different sample sizes."""
    return request.param


@pytest.fixture(params=["float32", "float64"])
def dtype(request) -> str:
    """Parameterized fixture for different data types."""
    return request.param


# Slow test marker
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "benchmark: mark test as benchmark")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle markers."""
    if config.getoption("--runslow"):
        # Don't skip slow tests if explicitly requested
        return
    
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def pytest_addoption(parser):
    """Add command line options."""
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--benchmark", action="store_true", default=False, help="run benchmark tests"
    )


# Test data validation utilities
def assert_valid_array(arr: jax.Array, shape: Tuple[int, ...] = None, dtype=None):
    """Assert that array is valid with optional shape and dtype checks."""
    assert isinstance(arr, jax.Array), f"Expected jax.Array, got {type(arr)}"
    assert jnp.isfinite(arr).all(), "Array contains non-finite values"
    
    if shape is not None:
        assert arr.shape == shape, f"Expected shape {shape}, got {arr.shape}"
    
    if dtype is not None:
        assert arr.dtype == dtype, f"Expected dtype {dtype}, got {arr.dtype}"


def assert_gradient_finite(grad: jax.Array):
    """Assert that gradient is finite and non-zero."""
    assert jnp.isfinite(grad).all(), "Gradient contains non-finite values"
    assert not jnp.allclose(grad, 0.0), "Gradient is zero everywhere"


def assert_optimization_convergence(history: List[float], tolerance: float = 1e-6):
    """Assert that optimization history shows convergence."""
    assert len(history) > 1, "Need at least 2 values to check convergence"
    
    # Check that function values are decreasing (for minimization)
    differences = jnp.diff(jnp.array(history))
    assert (differences <= tolerance).all(), "Function values not decreasing"
    
    # Check final convergence
    final_diff = abs(history[-1] - history[-2])
    assert final_diff < tolerance, f"Not converged, final difference: {final_diff}"