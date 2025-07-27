"""Sample unit tests to validate testing framework setup."""

import jax
import jax.numpy as jnp
import pytest


class TestBasicFunctionality:
    """Test basic functionality to ensure test setup works."""
    
    def test_jax_import(self):
        """Test that JAX is properly imported and configured."""
        assert jax is not None
        assert hasattr(jax, 'numpy')
    
    def test_array_creation(self):
        """Test basic JAX array creation and operations."""
        x = jnp.array([1.0, 2.0, 3.0])
        assert x.shape == (3,)
        assert jnp.allclose(x, jnp.array([1.0, 2.0, 3.0]))
    
    def test_gradient_computation(self):
        """Test basic gradient computation with JAX."""
        def f(x):
            return x**2
        
        grad_f = jax.grad(f)
        x = 3.0
        gradient = grad_f(x)
        
        # Analytical gradient of x^2 is 2x
        expected = 2.0 * x
        assert jnp.allclose(gradient, expected)
    
    def test_vectorized_operations(self):
        """Test vectorized operations."""
        x = jnp.array([1.0, 2.0, 3.0, 4.0])
        y = x**2
        expected = jnp.array([1.0, 4.0, 9.0, 16.0])
        assert jnp.allclose(y, expected)
    
    def test_random_number_generation(self):
        """Test reproducible random number generation."""
        key = jax.random.PRNGKey(42)
        x1 = jax.random.normal(key, (10,))
        x2 = jax.random.normal(key, (10,))
        
        # Same key should produce same results
        assert jnp.allclose(x1, x2)
    
    @pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4)])
    def test_array_shapes(self, shape):
        """Test array creation with different shapes."""
        x = jnp.ones(shape)
        assert x.shape == shape
        assert jnp.all(x == 1.0)


class TestMathematicalFunctions:
    """Test mathematical functions used in surrogate modeling."""
    
    def test_rosenbrock_function(self):
        """Test Rosenbrock function implementation."""
        def rosenbrock(x):
            return jnp.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
        
        # Test at global minimum (1, 1)
        x_opt = jnp.array([1.0, 1.0])
        result = rosenbrock(x_opt)
        assert jnp.allclose(result, 0.0, atol=1e-10)
        
        # Test at another point
        x_test = jnp.array([0.0, 0.0])
        result = rosenbrock(x_test)
        assert result > 0.0
    
    def test_gradient_rosenbrock(self):
        """Test gradient computation for Rosenbrock function."""
        def rosenbrock(x):
            return jnp.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
        
        grad_rosenbrock = jax.grad(rosenbrock)
        
        # Test gradient at global minimum
        x_opt = jnp.array([1.0, 1.0])
        gradient = grad_rosenbrock(x_opt)
        assert jnp.allclose(gradient, jnp.zeros_like(x_opt), atol=1e-10)
    
    def test_optimization_bounds(self):
        """Test that optimization bounds are respected."""
        bounds = [(-5.0, 5.0), (-5.0, 5.0)]
        
        # Generate random points within bounds
        key = jax.random.PRNGKey(123)
        n_points = 100
        x = jax.random.uniform(
            key, 
            (n_points, 2), 
            minval=jnp.array([b[0] for b in bounds]),
            maxval=jnp.array([b[1] for b in bounds])
        )
        
        # Check all points are within bounds
        for i, (low, high) in enumerate(bounds):
            assert jnp.all(x[:, i] >= low)
            assert jnp.all(x[:, i] <= high)


class TestNumericalStability:
    """Test numerical stability and edge cases."""
    
    def test_large_numbers(self):
        """Test handling of large numbers."""
        x = jnp.array([1e6, 1e7, 1e8])
        y = jnp.log(x)
        assert jnp.isfinite(y).all()
    
    def test_small_numbers(self):
        """Test handling of small numbers."""
        x = jnp.array([1e-6, 1e-7, 1e-8])
        y = jnp.sqrt(x)
        assert jnp.isfinite(y).all()
        assert (y > 0).all()
    
    def test_inf_and_nan_detection(self):
        """Test detection of infinite and NaN values."""
        # Test NaN detection
        nan_array = jnp.array([1.0, jnp.nan, 3.0])
        assert not jnp.isfinite(nan_array).all()
        assert jnp.isnan(nan_array).any()
        
        # Test infinity detection
        inf_array = jnp.array([1.0, jnp.inf, 3.0])
        assert not jnp.isfinite(inf_array).all()
        assert jnp.isinf(inf_array).any()
    
    def test_numerical_gradients(self):
        """Test numerical gradient computation."""
        def f(x):
            return jnp.sum(x**3)
        
        # Analytical gradient
        grad_f = jax.grad(f)
        x = jnp.array([1.0, 2.0, 3.0])
        analytical_grad = grad_f(x)
        
        # Expected gradient of sum(x^3) is 3*x^2
        expected_grad = 3 * x**2
        assert jnp.allclose(analytical_grad, expected_grad)
    
    def test_matrix_operations(self):
        """Test matrix operations used in surrogate models."""
        # Test matrix multiplication
        A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        B = jnp.array([[5.0, 6.0], [7.0, 8.0]])
        C = A @ B
        expected = jnp.array([[19.0, 22.0], [43.0, 50.0]])
        assert jnp.allclose(C, expected)
        
        # Test matrix inversion
        A_inv = jnp.linalg.inv(A)
        identity = A @ A_inv
        expected_identity = jnp.eye(2)
        assert jnp.allclose(identity, expected_identity, atol=1e-10)


@pytest.mark.slow
class TestPerformance:
    """Test performance characteristics (marked as slow)."""
    
    def test_large_array_operations(self):
        """Test operations on large arrays."""
        n = 10000
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (n,))
        
        # Test that operations complete in reasonable time
        y = jnp.sum(x**2)
        assert jnp.isfinite(y)
    
    def test_gradient_computation_performance(self):
        """Test gradient computation performance."""
        def complex_function(x):
            return jnp.sum(x**4 + jnp.sin(x) + jnp.cos(x**2))
        
        grad_fn = jax.grad(complex_function)
        x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Should complete without issues
        gradient = grad_fn(x)
        assert gradient.shape == x.shape
        assert jnp.isfinite(gradient).all()


class TestFixtures:
    """Test that fixtures work correctly."""
    
    def test_sample_data_1d(self, sample_data_1d):
        """Test 1D sample data fixture."""
        data = sample_data_1d
        assert "X" in data
        assert "y" in data
        assert "y_true" in data
        assert "noise" in data
        
        assert data["X"].shape[1] == 1
        assert len(data["y"]) == len(data["X"])
    
    def test_sample_data_2d(self, sample_data_2d):
        """Test 2D sample data fixture."""
        data = sample_data_2d
        assert data["X"].shape[1] == 2
        assert len(data["y"]) == len(data["X"])
    
    def test_optimization_functions(self, optimization_test_functions):
        """Test optimization function fixtures."""
        functions = optimization_test_functions
        assert "rosenbrock" in functions
        assert "rastrigin" in functions
        assert "ackley" in functions
        assert "sphere" in functions
        
        # Test that functions are callable
        x = jnp.array([1.0, 1.0])
        for name, func in functions.items():
            result = func(x)
            assert jnp.isfinite(result)
    
    def test_rng_key_fixture(self, rng_key):
        """Test RNG key fixture."""
        assert isinstance(rng_key, jax.Array)
        
        # Test reproducibility
        x1 = jax.random.normal(rng_key, (10,))
        x2 = jax.random.normal(rng_key, (10,))
        assert jnp.allclose(x1, x2)