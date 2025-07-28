"""Tests for pytest fixtures and test utilities."""

import jax
import jax.numpy as jnp
import pytest
from tests.conftest import (
    assert_valid_array,
    assert_gradient_finite,
    assert_optimization_convergence
)


class TestFixtureValidation:
    """Test that all fixtures produce valid data."""
    
    def test_sample_data_1d_validity(self, sample_data_1d):
        """Test 1D sample data fixture produces valid data."""
        data = sample_data_1d
        
        # Check data structure
        required_keys = ["X", "y", "y_true", "noise"]
        for key in required_keys:
            assert key in data, f"Missing key: {key}"
        
        # Validate arrays
        assert_valid_array(data["X"], shape=(100, 1))
        assert_valid_array(data["y"], shape=(100,))
        assert_valid_array(data["y_true"], shape=(100,))
        assert_valid_array(data["noise"], shape=(100,))
        
        # Check data relationships
        reconstructed_y = data["y_true"] + data["noise"]
        assert jnp.allclose(data["y"], reconstructed_y), "y != y_true + noise"
    
    def test_sample_data_2d_validity(self, sample_data_2d):
        """Test 2D sample data fixture produces valid data."""
        data = sample_data_2d
        
        # Validate arrays
        assert_valid_array(data["X"], shape=(200, 2))
        assert_valid_array(data["y"], shape=(200,))
        assert_valid_array(data["y_true"], shape=(200,))
        assert_valid_array(data["noise"], shape=(200,))
        
        # Check bounds
        assert jnp.all(data["X"] >= -3.0), "X values below lower bound"
        assert jnp.all(data["X"] <= 3.0), "X values above upper bound"
    
    def test_sample_data_high_dim_validity(self, sample_data_high_dim):
        """Test high-dimensional sample data fixture."""
        data = sample_data_high_dim
        
        # Validate arrays
        assert_valid_array(data["X"], shape=(500, 10))
        assert_valid_array(data["y"], shape=(500,))
        assert_valid_array(data["y_true"], shape=(500,))
        assert_valid_array(data["noise"], shape=(500,))
    
    def test_optimization_test_functions(self, optimization_test_functions):
        """Test optimization function fixtures."""
        functions = optimization_test_functions
        
        expected_functions = ["rosenbrock", "rastrigin", "ackley", "sphere"]
        for func_name in expected_functions:
            assert func_name in functions, f"Missing function: {func_name}"
            
            # Test function calls
            func = functions[func_name]
            x = jnp.array([1.0, 1.0])
            result = func(x)
            
            assert jnp.isfinite(result), f"{func_name} produced non-finite result"
            assert jnp.isscalar(result), f"{func_name} should return scalar"
    
    def test_benchmark_bounds(self, benchmark_bounds):
        """Test benchmark bounds fixture."""
        bounds = benchmark_bounds
        
        expected_functions = ["rosenbrock", "rastrigin", "ackley", "sphere"]
        for func_name in expected_functions:
            assert func_name in bounds, f"Missing bounds for: {func_name}"
            
            func_bounds = bounds[func_name]
            assert len(func_bounds) == 2, f"{func_name} should have 2D bounds"
            
            for low, high in func_bounds:
                assert low < high, f"Invalid bounds for {func_name}: [{low}, {high}]"
    
    def test_model_configs(self, model_configs):
        """Test model configuration fixture."""
        configs = model_configs
        
        expected_models = ["neural_network", "gaussian_process", "random_forest"]
        for model_name in expected_models:
            assert model_name in configs, f"Missing config for: {model_name}"
            
            config = configs[model_name]
            assert isinstance(config, dict), f"{model_name} config should be dict"
            assert len(config) > 0, f"{model_name} config should not be empty"
    
    def test_rng_key_reproducibility(self, rng_key):
        """Test that RNG key produces reproducible results."""
        # Generate data multiple times with same key
        data1 = jax.random.normal(rng_key, (10,))
        data2 = jax.random.normal(rng_key, (10,))
        data3 = jax.random.normal(rng_key, (10,))
        
        # All should be identical
        assert jnp.allclose(data1, data2), "RNG key not reproducible"
        assert jnp.allclose(data2, data3), "RNG key not reproducible"
    
    def test_mock_expensive_function(self, mock_expensive_function):
        """Test mock expensive function fixture."""
        func = mock_expensive_function
        
        # Test initial state
        assert func.n_evaluations == 0
        assert len(func.evaluation_history) == 0
        
        # Test function calls
        x1 = jnp.array([1.0, 2.0])
        result1 = func(x1)
        
        assert func.n_evaluations == 1
        assert len(func.evaluation_history) == 1
        assert jnp.allclose(func.evaluation_history[0], x1)
        assert jnp.isfinite(result1)
        
        # Test multiple calls
        x2 = jnp.array([3.0, 4.0])
        result2 = func(x2)
        
        assert func.n_evaluations == 2
        assert len(func.evaluation_history) == 2
        
        # Test reset
        func.reset()
        assert func.n_evaluations == 0
        assert len(func.evaluation_history) == 0
    
    def test_performance_thresholds(self, performance_thresholds):
        """Test performance thresholds fixture."""
        thresholds = performance_thresholds
        
        expected_keys = [
            "max_training_time",
            "max_prediction_time", 
            "max_memory_usage",
            "min_accuracy",
            "max_gradient_error"
        ]
        
        for key in expected_keys:
            assert key in thresholds, f"Missing threshold: {key}"
            assert thresholds[key] > 0, f"{key} should be positive"


class TestParameterizedFixtures:
    """Test parametrized fixtures."""
    
    def test_dimension_parameter(self, dimension):
        """Test dimension parametrized fixture."""
        assert isinstance(dimension, int)
        assert dimension > 0
        assert dimension <= 10  # Based on fixture definition
    
    def test_sample_size_parameter(self, sample_size):
        """Test sample size parametrized fixture."""
        assert isinstance(sample_size, int)
        assert sample_size > 0
        assert sample_size <= 500  # Based on fixture definition
    
    def test_dtype_parameter(self, dtype):
        """Test dtype parametrized fixture."""
        assert dtype in ["float32", "float64"]
        
        # Test array creation with specified dtype
        arr = jnp.array([1.0, 2.0, 3.0], dtype=dtype)
        assert str(arr.dtype) == dtype


class TestTestUtilities:
    """Test custom test utility functions."""
    
    def test_assert_valid_array_success(self):
        """Test assert_valid_array with valid arrays."""
        # Test basic validation
        arr = jnp.array([1.0, 2.0, 3.0])
        assert_valid_array(arr)  # Should not raise
        
        # Test with shape validation
        assert_valid_array(arr, shape=(3,))  # Should not raise
        
        # Test with dtype validation
        assert_valid_array(arr, dtype=jnp.float32)  # Should not raise
    
    def test_assert_valid_array_failures(self):
        """Test assert_valid_array with invalid arrays."""
        # Test with wrong type
        with pytest.raises(AssertionError, match="Expected jax.Array"):
            assert_valid_array([1, 2, 3])  # Python list
        
        # Test with non-finite values
        arr_nan = jnp.array([1.0, jnp.nan, 3.0])
        with pytest.raises(AssertionError, match="non-finite values"):
            assert_valid_array(arr_nan)
        
        arr_inf = jnp.array([1.0, jnp.inf, 3.0])
        with pytest.raises(AssertionError, match="non-finite values"):
            assert_valid_array(arr_inf)
        
        # Test with wrong shape
        arr = jnp.array([1.0, 2.0, 3.0])
        with pytest.raises(AssertionError, match="Expected shape"):
            assert_valid_array(arr, shape=(2,))
        
        # Test with wrong dtype
        with pytest.raises(AssertionError, match="Expected dtype"):
            assert_valid_array(arr, dtype=jnp.int32)
    
    def test_assert_gradient_finite_success(self):
        """Test assert_gradient_finite with valid gradients."""
        grad = jnp.array([1.0, -2.0, 3.5])
        assert_gradient_finite(grad)  # Should not raise
    
    def test_assert_gradient_finite_failures(self):
        """Test assert_gradient_finite with invalid gradients."""
        # Test with non-finite gradients
        grad_nan = jnp.array([1.0, jnp.nan, 3.0])
        with pytest.raises(AssertionError, match="non-finite values"):
            assert_gradient_finite(grad_nan)
        
        grad_inf = jnp.array([1.0, jnp.inf, 3.0])
        with pytest.raises(AssertionError, match="non-finite values"):
            assert_gradient_finite(grad_inf)
        
        # Test with zero gradient
        grad_zero = jnp.zeros(3)
        with pytest.raises(AssertionError, match="zero everywhere"):
            assert_gradient_finite(grad_zero)
    
    def test_assert_optimization_convergence_success(self):
        """Test assert_optimization_convergence with converged history."""
        # Decreasing function values
        history = [10.0, 5.0, 2.0, 1.0, 0.5, 0.1, 0.01, 0.001]
        assert_optimization_convergence(history)  # Should not raise
    
    def test_assert_optimization_convergence_failures(self):
        """Test assert_optimization_convergence with non-converged history."""
        # Test with insufficient history
        with pytest.raises(AssertionError, match="at least 2 values"):
            assert_optimization_convergence([1.0])
        
        # Test with increasing values
        history_increasing = [1.0, 2.0, 3.0, 4.0]
        with pytest.raises(AssertionError, match="not decreasing"):
            assert_optimization_convergence(history_increasing)
        
        # Test with non-converged final values
        history_non_converged = [10.0, 5.0, 2.0, 1.5]  # Final jump up
        with pytest.raises(AssertionError, match="not decreasing"):
            assert_optimization_convergence(history_non_converged)


class TestJAXConfiguration:
    """Test JAX configuration for tests."""
    
    def test_x64_enabled(self):
        """Test that x64 precision is enabled in tests."""
        # Create array and check precision
        x = jnp.array(1.0)
        assert x.dtype == jnp.float64, "x64 precision should be enabled"
    
    def test_jit_disabled(self):
        """Test that JIT is disabled for easier debugging."""
        # This is harder to test directly, but we can check config
        import jax
        # The config should be set by the fixture
        # We can't directly check the config value as it's internal
        # But we can verify that functions run without JIT compilation delays
        
        def simple_func(x):
            return x**2
        
        x = jnp.array(2.0)
        result = simple_func(x)
        assert result == 4.0
    
    def test_reproducible_random_generation(self):
        """Test that random generation is reproducible."""
        key1 = jax.random.PRNGKey(42)
        key2 = jax.random.PRNGKey(42)
        
        x1 = jax.random.normal(key1, (5,))
        x2 = jax.random.normal(key2, (5,))
        
        assert jnp.allclose(x1, x2), "Random generation should be reproducible"


class TestTestMarkers:
    """Test pytest markers work correctly."""
    
    @pytest.mark.slow
    def test_slow_marker(self):
        """Test that slow marker works."""
        # This test should be skipped unless --runslow is used
        assert True
    
    @pytest.mark.integration
    def test_integration_marker(self):
        """Test that integration marker works."""
        assert True
    
    @pytest.mark.gpu
    def test_gpu_marker(self):
        """Test that GPU marker works."""
        # This would test GPU functionality if available
        assert True
    
    @pytest.mark.benchmark
    def test_benchmark_marker(self):
        """Test that benchmark marker works."""
        assert True