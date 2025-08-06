"""Tests for validation utilities."""

import pytest
import jax.numpy as jnp
import warnings

from surrogate_optim.models.base import Dataset
from surrogate_optim.validation import (
    validate_bounds,
    validate_dataset,
    validate_surrogate_config,
    validate_optimization_inputs,
    validate_function,
    ValidationError,
    ValidationWarning,
)


class TestInputValidation:
    """Test input validation functions."""
    
    def test_validate_bounds_valid(self):
        """Test validation with valid bounds."""
        bounds = [(-1.0, 1.0), (0.0, 10.0), (-5.0, 5.0)]
        result = validate_bounds(bounds)
        
        assert len(result) == 3
        assert result[0] == (-1.0, 1.0)
        assert result[1] == (0.0, 10.0)
        assert result[2] == (-5.0, 5.0)
    
    def test_validate_bounds_invalid(self):
        """Test validation with invalid bounds."""
        # Empty bounds
        with pytest.raises(ValidationError, match="Bounds cannot be empty"):
            validate_bounds([])
        
        # Invalid format
        with pytest.raises(ValidationError, match="must be a 2-tuple"):
            validate_bounds([(1, 2, 3)])
        
        # Lower >= upper
        with pytest.raises(ValidationError, match="lower bound.*upper bound"):
            validate_bounds([(2.0, 1.0)])
        
        # Non-finite values
        with pytest.raises(ValidationError, match="must be finite"):
            validate_bounds([(float('inf'), 1.0)])
    
    def test_validate_bounds_warnings(self):
        """Test bounds validation warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bounds = [(0, 1e7)]  # Very large range
            validate_bounds(bounds)
            
            assert len(w) == 1
            assert issubclass(w[0].category, ValidationWarning)
            assert "very large range" in str(w[0].message)
    
    def test_validate_dataset_valid(self):
        """Test validation with valid dataset."""
        X = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = jnp.array([1.0, 2.0, 3.0])
        gradients = jnp.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        
        dataset = Dataset(X=X, y=y, gradients=gradients)
        result = validate_dataset(dataset)
        
        assert result.n_samples == 3
        assert result.n_dims == 2
    
    def test_validate_dataset_invalid(self):
        """Test validation with invalid dataset."""
        # Empty dataset
        X = jnp.array([]).reshape(0, 2)
        y = jnp.array([])
        dataset = Dataset(X=X, y=y)
        
        with pytest.raises(ValidationError, match="cannot be empty"):
            validate_dataset(dataset)
        
        # Non-finite values
        X = jnp.array([[1.0, jnp.nan], [2.0, 3.0]])
        y = jnp.array([1.0, 2.0])
        dataset = Dataset(X=X, y=y)
        
        with pytest.raises(ValidationError, match="non-finite values"):
            validate_dataset(dataset)
    
    def test_validate_dataset_warnings(self):
        """Test dataset validation warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Small dataset
            X = jnp.array([[1.0, 2.0], [3.0, 4.0]])  # Only 2 samples for 2D
            y = jnp.array([1.0, 2.0])
            dataset = Dataset(X=X, y=y)
            
            validate_dataset(dataset)
            
            # Should warn about small sample size
            assert any("Consider collecting at least" in str(warning.message) for warning in w)
    
    def test_validate_surrogate_config_neural(self):
        """Test neural network configuration validation."""
        surrogate_type = "neural_network"
        params = {"hidden_dims": [64, 32], "learning_rate": 0.001}
        
        result_type, result_params = validate_surrogate_config(surrogate_type, params)
        
        assert result_type == surrogate_type
        assert result_params == params
    
    def test_validate_surrogate_config_invalid_type(self):
        """Test invalid surrogate type."""
        with pytest.raises(ValidationError, match="Unknown surrogate type"):
            validate_surrogate_config("invalid_type", {})
    
    def test_validate_surrogate_config_invalid_params(self):
        """Test invalid parameters."""
        # Invalid hidden_dims
        with pytest.raises(ValidationError, match="hidden_dims must be"):
            validate_surrogate_config("neural_network", {"hidden_dims": []})
        
        # Invalid learning_rate
        with pytest.raises(ValidationError, match="learning_rate must be positive"):
            validate_surrogate_config("neural_network", {"learning_rate": -0.1})
    
    def test_validate_optimization_inputs_valid(self):
        """Test valid optimization inputs."""
        x0 = jnp.array([1.0, 2.0])
        bounds = [(-5.0, 5.0), (-10.0, 10.0)]
        method = "gradient_descent"
        options = {"max_iterations": 100, "tolerance": 1e-6}
        
        result = validate_optimization_inputs(x0, bounds, method, options)
        
        assert len(result) == 4
        assert jnp.allclose(result[0], x0)
        assert result[1] == bounds
        assert result[2] == method
        assert result[3] == options
    
    def test_validate_optimization_inputs_invalid_x0(self):
        """Test invalid initial point."""
        # Non-finite x0
        x0 = jnp.array([1.0, jnp.inf])
        
        with pytest.raises(ValidationError, match="non-finite values"):
            validate_optimization_inputs(x0)
        
        # Wrong dimensions
        x0 = jnp.array([[1.0, 2.0]])  # 2D instead of 1D
        
        with pytest.raises(ValidationError, match="must be 1-dimensional"):
            validate_optimization_inputs(x0)
    
    def test_validate_optimization_inputs_bounds_mismatch(self):
        """Test bounds dimension mismatch."""
        x0 = jnp.array([1.0, 2.0])
        bounds = [(-5.0, 5.0)]  # Only 1 bound for 2D x0
        
        with pytest.raises(ValidationError, match="Bounds length.*x0 length"):
            validate_optimization_inputs(x0, bounds)
    
    def test_validate_optimization_inputs_x0_out_of_bounds(self):
        """Test x0 outside bounds."""
        x0 = jnp.array([10.0])  # Outside bounds
        bounds = [(-5.0, 5.0)]
        
        with pytest.raises(ValidationError, match="violates bounds"):
            validate_optimization_inputs(x0, bounds)
    
    def test_validate_function_valid(self):
        """Test valid function validation."""
        def test_func(x):
            return float(jnp.sum(x**2))
        
        validated_func = validate_function(test_func)
        
        # Should work without errors
        result = validated_func(jnp.array([1.0, 2.0]))
        assert isinstance(result, float)
    
    def test_validate_function_invalid(self):
        """Test invalid function validation."""
        # Not callable
        with pytest.raises(ValidationError, match="must be callable"):
            validate_function("not_a_function")
        
        # Function that raises exception
        def bad_func(x):
            raise ValueError("Test error")
        
        with pytest.raises(ValidationError, match="Function evaluation failed"):
            validate_function(bad_func)
    
    def test_validate_function_with_bounds(self):
        """Test function validation with bounds."""
        def test_func(x):
            return float(jnp.sum(x**2))
        
        bounds = [(-2.0, 2.0), (-2.0, 2.0)]
        validated_func = validate_function(test_func, bounds=bounds)
        
        # Should work within bounds
        result = validated_func(jnp.array([1.0, 1.0]))
        assert isinstance(result, float)


class TestOptimizationValidation:
    """Test optimization-specific validation."""
    
    def test_multi_start_requires_bounds(self):
        """Test that multi-start optimization requires bounds."""
        x0 = jnp.array([1.0, 2.0])
        method = "multi_start"
        
        with pytest.raises(ValidationError, match="requires bounds"):
            validate_optimization_inputs(x0, bounds=None, method=method)
    
    def test_trust_region_valid_params(self):
        """Test trust region parameter validation."""
        x0 = jnp.array([1.0])
        method = "trust_region"
        options = {"initial_radius": 1.0}
        
        # Should not raise error
        validate_optimization_inputs(x0, method=method, options=options)
    
    def test_trust_region_invalid_radius(self):
        """Test invalid trust region radius."""
        x0 = jnp.array([1.0])
        method = "trust_region" 
        options = {"initial_radius": -1.0}
        
        with pytest.raises(ValidationError, match="initial_radius must be positive"):
            validate_optimization_inputs(x0, method=method, options=options)


class TestWarningBehavior:
    """Test warning behavior in validation."""
    
    def test_large_iterations_warning(self):
        """Test warning for large iteration count."""
        x0 = jnp.array([1.0])
        options = {"max_iterations": 50000}
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_optimization_inputs(x0, options=options)
            
            assert len(w) == 1
            assert "very large" in str(w[0].message)
    
    def test_large_tolerance_warning(self):
        """Test warning for large tolerance."""
        x0 = jnp.array([1.0])
        options = {"tolerance": 0.5}
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_optimization_inputs(x0, options=options)
            
            assert len(w) == 1
            assert "quite large" in str(w[0].message)
    
    def test_many_starts_warning(self):
        """Test warning for many multi-starts."""
        x0 = jnp.array([1.0])
        bounds = [(-5.0, 5.0)]
        method = "multi_start"
        options = {"n_starts": 500}
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_optimization_inputs(x0, bounds, method, options)
            
            assert len(w) == 1
            assert "quite large" in str(w[0].message)