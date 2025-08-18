#!/usr/bin/env python3
"""Test basic functionality of the surrogate optimization package."""

import jax.numpy as jnp
import numpy as np

# Test basic import
try:
    from surrogate_optim.core import SurrogateOptimizer
    print("✅ SurrogateOptimizer imported successfully")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    exit(1)

# Test basic functionality
def simple_quadratic(x):
    """Simple quadratic function for testing."""
    x = jnp.atleast_1d(x)
    return float(jnp.sum(x**2))

def main():
    print("🧪 Testing basic SurrogateOptimizer functionality")
    print("=" * 50)
    
    # Create optimizer
    try:
        optimizer = SurrogateOptimizer(
            surrogate_type="neural_network",
            surrogate_params={"hidden_dims": [32, 32]},
            optimizer_type="gradient_descent"
        )
        print("✅ SurrogateOptimizer created successfully")
    except Exception as e:
        print(f"❌ Failed to create optimizer: {e}")
        return
    
    # Generate some test data
    try:
        from surrogate_optim.data.collector import collect_data
        bounds = [(-2.0, 2.0), (-2.0, 2.0)]
        data = collect_data(
            function=simple_quadratic,
            n_samples=50,
            bounds=bounds,
            sampling="random"
        )
        print("✅ Test data generated successfully")
        print(f"   Data shape: X={data.X.shape}, y={data.y.shape}")
    except Exception as e:
        print(f"❌ Failed to generate data: {e}")
        return
    
    # Test surrogate fitting
    try:
        optimizer.fit_surrogate(data)
        print("✅ Surrogate model fitted successfully")
    except Exception as e:
        print(f"❌ Failed to fit surrogate: {e}")
        return
    
    # Test optimization
    try:
        initial_point = jnp.array([1.0, 1.0])
        result = optimizer.optimize(initial_point)
        print("✅ Optimization completed successfully")
        print(f"   Result: {result}")
        print(f"   Function value: {simple_quadratic(result):.6f}")
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        return
    
    print("\n🎉 All basic functionality tests passed!")

if __name__ == "__main__":
    main()