#!/usr/bin/env python3
"""Generation 1: Simple functional test to verify core system works."""

import jax.numpy as jnp
import numpy as np
from surrogate_optim import SurrogateOptimizer, quick_optimize

def test_basic_functionality():
    """Test basic surrogate optimization functionality."""
    print("🧪 Testing Generation 1: Basic Functionality")
    
    # Simple test function
    def test_function(x):
        """Simple quadratic function."""
        return -jnp.sum(x**2) + 0.1 * jnp.sin(10 * jnp.linalg.norm(x))
    
    # Generate test data
    key = jax.random.PRNGKey(42)
    X = jax.random.uniform(key, (100, 2), minval=-2, maxval=2)
    y = jnp.array([test_function(x) for x in X])
    
    data = {"X": X, "y": y}
    
    # Test neural network surrogate
    print("  ✓ Testing Neural Network Surrogate...")
    optimizer = SurrogateOptimizer(
        surrogate_type="neural_network",
        surrogate_params={"hidden_dims": [32, 32]},
        optimizer_type="gradient_descent",
        optimizer_params={"learning_rate": 0.01, "max_iterations": 50}
    )
    
    # Fit surrogate
    optimizer.fit_surrogate(data)
    print("  ✓ Surrogate fitted successfully")
    
    # Optimize
    x0 = jnp.array([1.0, 1.0])
    result = optimizer.optimize(x0)
    print(f"  ✓ Optimization completed: x* = {result}")
    
    # Test quick_optimize function
    print("  ✓ Testing quick_optimize...")
    result_quick = quick_optimize(
        function=test_function,
        bounds=[(-2, 2), (-2, 2)],
        n_samples=50
    )
    print(f"  ✓ Quick optimize completed: x* = {result_quick}")
    
    print("✅ Generation 1 Basic Functionality: PASSED")
    return True

if __name__ == "__main__":
    import jax
    test_basic_functionality()