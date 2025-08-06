"""Integration tests for robust error handling and recovery."""

import pytest
import jax.numpy as jnp
import warnings
from unittest.mock import Mock, patch

from surrogate_optim import SurrogateOptimizer, collect_data
from surrogate_optim.models import NeuralSurrogate, GPSurrogate
from surrogate_optim.validation import ValidationError, ValidationWarning
from surrogate_optim.optimizers import GradientDescentOptimizer, TrustRegionOptimizer


class TestRobustWorkflows:
    """Test robust error handling in complete workflows."""
    
    def test_data_collection_with_failing_function(self):
        """Test data collection with partially failing function."""
        call_count = 0
        
        def unreliable_function(x):
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 0:  # Fail every 3rd call
                raise ValueError("Simulated function failure")
            return float(jnp.sum(x**2))
        
        bounds = [(-2.0, 2.0), (-2.0, 2.0)]
        
        # Should handle some failures gracefully
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ValidationWarning)
            dataset = collect_data(
                function=unreliable_function,
                n_samples=20,
                bounds=bounds,
                verbose=False
            )
        
        # Should still collect some data
        assert dataset.n_samples > 10  # Some samples should succeed
        assert not jnp.all(jnp.isnan(dataset.y))  # Not all should be NaN
    
    def test_surrogate_training_with_difficult_data(self):
        """Test surrogate training with challenging datasets."""
        # Dataset with very different scales
        X = jnp.array([
            [1e-6, 1e6],
            [2e-6, 2e6], 
            [3e-6, 3e6],
            [4e-6, 4e6],
            [5e-6, 5e6]
        ])
        y = jnp.array([1.0, 4.0, 9.0, 16.0, 25.0])
        
        from surrogate_optim.models.base import Dataset
        dataset = Dataset(X=X, y=y)
        
        # Should handle different scales with warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            surrogate = NeuralSurrogate(n_epochs=10)
            surrogate.fit(dataset)
            
            # Should warn about scale differences
            scale_warnings = [warning for warning in w 
                            if "scale" in str(warning.message)]
            assert len(scale_warnings) > 0
    
    def test_optimization_with_poor_surrogate(self):
        """Test optimization behavior with poorly trained surrogate."""
        # Create a dataset that's hard to fit
        def noisy_function(x):
            return float(jnp.sum(x**2) + 10 * jnp.sin(100 * jnp.linalg.norm(x)))
        
        bounds = [(-1.0, 1.0), (-1.0, 1.0)]
        
        # Collect limited data
        dataset = collect_data(
            function=noisy_function,
            n_samples=15,  # Very small for complex function
            bounds=bounds,
            verbose=False
        )
        
        # Train surrogate (will be poor quality)
        optimizer = SurrogateOptimizer(
            surrogate_type="neural_network",
            surrogate_params={"n_epochs": 10}
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            optimizer.fit_surrogate(dataset)
            
            # Optimization should still work but may not converge well
            result = optimizer.optimize(
                initial_point=jnp.array([0.5, 0.5]),
                bounds=bounds,
                num_steps=20
            )
            
            # Should complete without crashing
            assert result is not None
            assert jnp.isfinite(result.fun)
    
    def test_gradient_computation_failures(self):
        """Test handling of gradient computation failures."""
        # Mock surrogate that sometimes fails gradient computation
        class UnreliableGradientSurrogate(NeuralSurrogate):
            def __init__(self):
                super().__init__(n_epochs=1)
                self.call_count = 0
            
            def gradient(self, x):
                self.call_count += 1
                if self.call_count % 4 == 0:  # Fail every 4th call
                    raise RuntimeError("Gradient computation failed")
                return super().gradient(x)
        
        # Create simple dataset
        X = jnp.array([[0.0], [0.5], [1.0]])
        y = jnp.array([0.0, 0.25, 1.0])
        from surrogate_optim.models.base import Dataset
        dataset = Dataset(X=X, y=y)
        
        surrogate = UnreliableGradientSurrogate()
        surrogate.fit(dataset)
        
        # Optimization should handle gradient failures
        optimizer = GradientDescentOptimizer(max_iterations=10, verbose=False)
        
        # Should fail gracefully when gradients fail
        with pytest.raises((RuntimeError, ValidationError)):
            optimizer.optimize(surrogate, jnp.array([0.5]))
    
    def test_bounds_violation_recovery(self):
        """Test recovery from bounds violations."""
        def quadratic(x):
            return float(jnp.sum((x - 0.5)**2))
        
        bounds = [(-1.0, 1.0), (-1.0, 1.0)]
        dataset = collect_data(quadratic, 20, bounds, verbose=False)
        
        optimizer = SurrogateOptimizer(
            surrogate_type="neural_network",
            surrogate_params={"n_epochs": 10}
        )
        optimizer.fit_surrogate(dataset)
        
        # Start optimization from boundary
        result = optimizer.optimize(
            initial_point=jnp.array([1.0, 1.0]),  # On boundary
            bounds=bounds,
            num_steps=10
        )
        
        # Should handle boundary conditions
        assert jnp.all(result.x >= -1.0)
        assert jnp.all(result.x <= 1.0)
    
    def test_memory_efficient_large_dataset(self):
        """Test handling of large datasets."""
        # Simulate large dataset scenario
        def simple_function(x):
            return float(jnp.sum(x**2))
        
        bounds = [(-2.0, 2.0)] * 10  # 10D problem
        
        # This would create a large dataset in real usage
        dataset = collect_data(
            simple_function,
            n_samples=100,  # Keep reasonable for tests
            bounds=bounds,
            batch_size=20,  # Test batching
            verbose=False
        )
        
        # Should handle batched processing
        assert dataset.n_samples == 100
        assert dataset.n_dims == 10
    
    def test_concurrent_optimization_robustness(self):
        """Test robustness with concurrent optimization attempts."""
        def test_function(x):
            return float(jnp.sum(x**2))
        
        bounds = [(-2.0, 2.0), (-2.0, 2.0)]
        dataset = collect_data(test_function, 30, bounds, verbose=False)
        
        # Multiple optimizers should work independently
        optimizers = []
        for i in range(3):
            opt = SurrogateOptimizer(
                surrogate_type="neural_network",
                surrogate_params={"n_epochs": 5, "random_seed": i}
            )
            opt.fit_surrogate(dataset)
            optimizers.append(opt)
        
        # All should work
        results = []
        for opt in optimizers:
            result = opt.optimize(
                initial_point=jnp.array([1.0, 1.0]),
                bounds=bounds,
                num_steps=10
            )
            results.append(result)
        
        # All should succeed
        assert all(jnp.isfinite(r.fun) for r in results)
    
    def test_trust_region_with_validation(self):
        """Test trust region optimizer with function validation."""
        def test_function(x):
            return float(jnp.sum(x**2))
        
        bounds = [(-2.0, 2.0), (-2.0, 2.0)]
        dataset = collect_data(test_function, 20, bounds, verbose=False)
        
        surrogate = NeuralSurrogate(n_epochs=10)
        surrogate.fit(dataset)
        
        # Trust region with validation
        optimizer = TrustRegionOptimizer(validate_every=3)
        optimizer.set_true_function(test_function)
        
        result = optimizer.optimize(
            surrogate=surrogate,
            x0=jnp.array([1.0, 1.0]),
            bounds=bounds
        )
        
        # Should succeed with validation
        assert result.success or result.nit > 0  # At least made progress
        assert jnp.isfinite(result.fun)
    
    def test_model_validation_integration(self):
        """Test integration with model validation."""
        def quadratic(x):
            return float(jnp.sum(x**2))
        
        bounds = [(-1.0, 1.0), (-1.0, 1.0)]
        train_data = collect_data(quadratic, 30, bounds, verbose=False)
        test_data = collect_data(quadratic, 10, bounds, verbose=False)
        
        surrogate = GPSurrogate()
        surrogate.fit(train_data)
        
        # Test validation
        from surrogate_optim.validation import validate_surrogate_performance
        
        validation_results = validate_surrogate_performance(
            surrogate, test_data
        )
        
        # Should pass validation for this simple case
        assert validation_results["validation"]["passed"]
        assert validation_results["metrics"]["r2"] > 0.8
    
    def test_logging_during_failures(self):
        """Test logging behavior during failures."""
        from surrogate_optim.monitoring.logging import get_logger
        
        logger = get_logger()
        
        # Test with failing function
        def failing_function(x):
            raise RuntimeError("Deliberate failure")
        
        bounds = [(-1.0, 1.0)]
        
        # Should log errors appropriately
        with pytest.raises(Exception):
            collect_data(failing_function, 10, bounds, verbose=False)
        
        # Logger should have recorded the issue (can't easily test this 
        # without complex mocking, but verifies no crashes)
    
    def test_configuration_validation_workflow(self):
        """Test complete workflow with configuration validation."""
        from surrogate_optim.validation import (
            validate_surrogate_config,
            validate_optimization_inputs
        )
        
        # Validate surrogate config
        surrogate_type, surrogate_params = validate_surrogate_config(
            "neural_network",
            {"hidden_dims": [32, 16], "learning_rate": 0.01}
        )
        
        # Validate optimization inputs
        x0 = jnp.array([0.5, 0.5])
        bounds = [(-1.0, 1.0), (-1.0, 1.0)]
        method = "gradient_descent"
        options = {"max_iterations": 50}
        
        validated_inputs = validate_optimization_inputs(x0, bounds, method, options)
        
        # Use validated inputs in workflow
        def test_func(x):
            return float(jnp.sum(x**2))
        
        dataset = collect_data(test_func, 20, bounds, verbose=False)
        
        optimizer = SurrogateOptimizer(
            surrogate_type=surrogate_type,
            surrogate_params=surrogate_params
        )
        optimizer.fit_surrogate(dataset)
        
        result = optimizer.optimize(*validated_inputs[:2], num_steps=options["max_iterations"])
        
        # Should work with validated configuration
        assert jnp.isfinite(result.fun)


class TestErrorRecovery:
    """Test error recovery mechanisms."""
    
    def test_partial_data_recovery(self):
        """Test recovery from partial data collection failures."""
        success_count = 0
        
        def flaky_function(x):
            nonlocal success_count
            success_count += 1
            if success_count <= 5:  # First 5 calls succeed
                return float(jnp.sum(x**2))
            elif success_count <= 8:  # Next 3 fail
                raise RuntimeError("Temporary failure")
            else:  # Rest succeed
                return float(jnp.sum(x**2))
        
        bounds = [(-1.0, 1.0), (-1.0, 1.0)]
        
        # Should recover from temporary failures
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dataset = collect_data(flaky_function, 15, bounds, verbose=False)
        
        # Should have some successful evaluations
        finite_values = jnp.isfinite(dataset.y)
        assert jnp.sum(finite_values) >= 10  # At least 10 successful
    
    def test_surrogate_fallback_behavior(self):
        """Test fallback behavior when primary surrogate fails."""
        # Create dataset
        X = jnp.array([[0.0], [0.5], [1.0], [1.5], [2.0]])
        y = jnp.array([0.0, 0.25, 1.0, 2.25, 4.0])
        from surrogate_optim.models.base import Dataset
        dataset = Dataset(X=X, y=y)
        
        # Try neural network first (might fail with small data)
        try:
            surrogate = NeuralSurrogate(n_epochs=5)
            surrogate.fit(dataset)
            success = True
        except Exception:
            success = False
        
        # If neural network fails, fall back to GP
        if not success:
            surrogate = GPSurrogate()
            surrogate.fit(dataset)
        
        # Should have a working surrogate
        prediction = surrogate.predict(jnp.array([0.75]))
        assert jnp.isfinite(prediction)
    
    def test_optimization_parameter_adjustment(self):
        """Test automatic parameter adjustment on optimization failures."""
        def test_function(x):
            return float(jnp.sum(x**2))
        
        bounds = [(-1.0, 1.0)]
        dataset = collect_data(test_function, 10, bounds, verbose=False)
        
        surrogate = NeuralSurrogate(n_epochs=5)
        surrogate.fit(dataset)
        
        # Start with aggressive parameters
        optimizer = GradientDescentOptimizer(
            learning_rate=10.0,  # Too high
            max_iterations=5
        )
        
        try:
            result = optimizer.optimize(surrogate, jnp.array([0.5]))
            # If it succeeds, good
            assert jnp.isfinite(result.fun)
        except:
            # If it fails, try with conservative parameters
            optimizer = GradientDescentOptimizer(
                learning_rate=0.01,  # More conservative
                max_iterations=10
            )
            result = optimizer.optimize(surrogate, jnp.array([0.5]))
            assert jnp.isfinite(result.fun)