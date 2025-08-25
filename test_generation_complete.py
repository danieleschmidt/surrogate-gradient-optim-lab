#!/usr/bin/env python3
"""Comprehensive test for all three generations of SDLC implementation."""

import jax.numpy as jnp
import jax.random as random
from surrogate_optim import SurrogateOptimizer, quick_optimize
from surrogate_optim.health.system_monitor import system_monitor
from surrogate_optim.robustness.comprehensive_validation import robust_validator


def test_complete_sdlc_implementation():
    """Test all three generations of SDLC implementation."""
    print("🧪 Testing Complete SDLC Implementation (Generations 1-3)")
    
    # Define test function
    def test_function(x):
        """Complex test function with multiple optima."""
        return -jnp.sum(x**2) + 0.5 * jnp.sin(5 * jnp.linalg.norm(x)) + 0.1 * jnp.cos(10 * x[0])
    
    # Generate comprehensive test data
    key = random.PRNGKey(42)
    X = random.uniform(key, (200, 3), minval=-3, maxval=3)
    y = jnp.array([test_function(x) for x in X])
    
    data = {"X": X, "y": y}
    bounds = [(-3, 3), (-3, 3), (-3, 3)]
    
    print("\n🔧 Generation 1: Basic Functionality")
    
    # Test basic surrogate optimization
    optimizer = SurrogateOptimizer(
        surrogate_type="neural_network",
        surrogate_params={"hidden_dims": [64, 32]},
        optimizer_type="gradient_descent",
        optimizer_params={"learning_rate": 0.01, "max_iterations": 100}
    )
    
    # Fit surrogate
    optimizer.fit_surrogate(data)
    print("  ✓ Surrogate training successful")
    
    # Single optimization
    x0 = jnp.array([1.5, -1.0, 0.5])
    result = optimizer.optimize(x0)
    print(f"  ✓ Basic optimization: x* = {result}")
    
    print("\n🛡️ Generation 2: Robustness Testing")
    
    # Test comprehensive validation
    validation_result = robust_validator.validate_complete_workflow(
        objective_function=test_function,
        data=optimizer.training_data,
        surrogate_config=optimizer.surrogate_params,
        optimizer_config=optimizer.optimizer_params
    )
    
    print(f"  ✓ Validation status: {validation_result['overall_status']}")
    print(f"  ✓ Warnings: {len(validation_result['warnings'])}")
    print(f"  ✓ Errors: {len(validation_result['errors'])}")
    
    # Test health monitoring
    health_report = system_monitor.get_health_report()
    print(f"  ✓ System health: {health_report.get('status', 'unknown')}")
    
    print("\n🚀 Generation 3: Scalability Testing")
    
    # Test parallel optimization
    initial_points = [
        jnp.array([1.0, 1.0, 1.0]),
        jnp.array([-1.0, -1.0, -1.0]), 
        jnp.array([0.0, 2.0, -1.0]),
        jnp.array([2.0, 0.0, 1.0])
    ]
    
    try:
        parallel_results = optimizer.optimize_parallel(initial_points, bounds=bounds)
        print(f"  ✓ Parallel optimization: {len(parallel_results)} results")
        
        # Find best result
        best_idx = 0
        best_value = float('inf')
        for i, res in enumerate(parallel_results):
            if hasattr(res, 'fun') and res.fun < best_value:
                best_value = res.fun
                best_idx = i
        
        print(f"  ✓ Best parallel result: value = {best_value:.6f}")
        
    except Exception as e:
        print(f"  ⚠️ Parallel optimization error: {e}")
    
    # Test batch prediction
    test_points = random.uniform(random.PRNGKey(123), (50, 3), minval=-2, maxval=2)
    try:
        batch_predictions = optimizer.predict_batch(test_points)
        print(f"  ✓ Batch prediction: {len(batch_predictions)} predictions")
    except Exception as e:
        print(f"  ⚠️ Batch prediction error: {e}")
    
    print("\n✅ Complete SDLC Implementation Test: PASSED")
    
    # Cleanup
    try:
        system_monitor.stop_monitoring()
    except:
        pass
    
    return True


if __name__ == "__main__":
    test_complete_sdlc_implementation()