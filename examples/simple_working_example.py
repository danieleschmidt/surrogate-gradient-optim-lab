#!/usr/bin/env python3
"""
Simple working example for Surrogate Gradient Optimization Lab
Tests basic functionality without complex features
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random
import numpy as np

# Core imports
from surrogate_optim.core import SurrogateOptimizer
from surrogate_optim.data.collector import collect_data


def rosenbrock(x):
    """Simple Rosenbrock function for testing."""
    x = jnp.atleast_1d(x)
    if len(x) != 2:
        # Pad or truncate to 2D
        if len(x) < 2:
            x = jnp.pad(x, (0, 2 - len(x)))
        else:
            x = x[:2]
    return float(100.0 * (x[1] - x[0]**2)**2 + (1 - x[0])**2)


def simple_quadratic(x):
    """Simple quadratic function."""
    x = jnp.atleast_1d(x)
    return float(jnp.sum(x**2) + 0.1 * jnp.sum(x**3))


def main():
    """Simple demonstration."""
    print("🌟 Simple Surrogate Optimization Demo")
    print("=" * 40)
    
    # Test function and bounds
    test_function = simple_quadratic
    bounds = [(-3.0, 3.0), (-3.0, 3.0)]
    
    print(f"🎯 Testing function: {test_function.__name__}")
    print(f"📊 Bounds: {bounds}")
    
    # Collect data
    print("\n📈 Collecting training data...")
    try:
        data = collect_data(
            function=test_function,
            n_samples=100,
            bounds=bounds,
            sampling="random",  # Use random instead of sobol to avoid warning
            verbose=True
        )
        print(f"✅ Collected {data.n_samples} samples")
        print(f"   Input shape: {data.X.shape}")
        print(f"   Output shape: {data.y.shape}")
    except Exception as e:
        print(f"❌ Data collection failed: {e}")
        return
    
    # Try different surrogate types
    surrogate_types = ["gaussian_process"]  # Start with GP which is more stable
    
    for surrogate_type in surrogate_types:
        print(f"\n🧠 Testing {surrogate_type} surrogate...")
        
        try:
            # Create optimizer - use minimal configuration
            optimizer = SurrogateOptimizer(
                surrogate_type=surrogate_type,
                surrogate_params={},  # Use defaults
                optimizer_type="gradient_descent"
            )
            
            print("✅ Optimizer created")
            
            # Train surrogate
            optimizer.fit_surrogate(data)
            print("✅ Surrogate trained successfully")
            
            # Test prediction
            test_point = jnp.array([1.5, 1.5])
            pred = optimizer.predict(test_point)
            true_val = test_function(test_point)
            
            print(f"📊 Test prediction:")
            print(f"   Point: {test_point}")
            print(f"   True value: {true_val:.6f}")
            print(f"   Predicted: {pred:.6f}")
            print(f"   Error: {abs(pred - true_val):.6f}")
            
            # Test gradient
            try:
                grad = optimizer.gradient(test_point)
                print(f"   Gradient: {grad}")
            except Exception as e:
                print(f"   ⚠️  Gradient computation failed: {e}")
            
            # Run optimization
            print("\n🚀 Running optimization...")
            initial_point = jnp.array([2.0, 2.0])
            
            result = optimizer.optimize(
                initial_point=initial_point,
                bounds=bounds,
                num_steps=50
            )
            
            print(f"✅ Optimization completed:")
            print(f"   Initial point: {initial_point}")
            print(f"   Final point: {result.x}")
            print(f"   Final value: {result.fun:.6f}")
            print(f"   True optimum (~[0, 0]): {test_function(jnp.array([0.0, 0.0])):.6f}")
            
            # Validation
            print("\n🔍 Running validation...")
            validation = optimizer.validate(
                test_function=test_function,
                n_test_points=50,
                metrics=["mse", "mae", "r2"]
            )
            
            print(f"   Validation metrics:")
            for metric, value in validation.items():
                print(f"   - {metric}: {value:.6f}")
            
            print(f"\n✅ {surrogate_type} test completed successfully!")
            
        except Exception as e:
            print(f"❌ {surrogate_type} test failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n🎉 Simple demo completed!")


if __name__ == "__main__":
    main()