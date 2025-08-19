#!/usr/bin/env python3
"""
Generation 1: MAKE IT WORK (Simple) - Basic Surrogate Optimization Demo

This demonstrates the fundamental autonomous SDLC Generation 1 implementation:
- Basic functionality that works end-to-end
- Core surrogate optimization workflow
- Minimal viable features with essential error handling
"""

import jax.numpy as jnp
from surrogate_optim import SurrogateOptimizer, collect_data


def black_box_function(x):
    """Example black-box function to optimize.
    
    This could represent any expensive simulation, hardware measurement,
    or other non-differentiable function in real applications.
    """
    return -jnp.sum(x**2) + jnp.sin(5 * jnp.linalg.norm(x))


def generation1_simple_demo():
    """Demonstrate Generation 1: Basic working functionality."""
    print("=" * 60)
    print("GENERATION 1: MAKE IT WORK (Simple)")
    print("Basic Surrogate Optimization Workflow")
    print("=" * 60)
    
    # Define problem bounds
    bounds = [(-3, 3), (-3, 3)]
    
    try:
        # Step 1: Collect training data
        print("\n1. Collecting training data...")
        data = collect_data(
            function=black_box_function,
            n_samples=50,  # Minimal viable sample size
            bounds=bounds,
            sampling="sobol",
            verbose=True
        )
        print(f"✅ Collected {data.n_samples} training samples")
        
        # Step 2: Create and train surrogate optimizer
        print("\n2. Training surrogate model...")
        optimizer = SurrogateOptimizer(
            surrogate_type="neural_network",
            surrogate_params={"hidden_dims": [32, 32]},  # Simple architecture
            optimizer_type="gradient_descent"
        )
        
        optimizer.fit_surrogate(data)
        print("✅ Surrogate model trained successfully")
        
        # Step 3: Optimize using surrogate
        print("\n3. Running optimization...")
        initial_point = jnp.array([1.0, 1.0])
        result = optimizer.optimize(
            initial_point=initial_point,
            bounds=bounds
        )
        
        # Step 4: Validate results
        print("\n4. Validating results...")
        optimal_x = result.x if hasattr(result, 'x') else result
        optimal_value = black_box_function(optimal_x)
        
        print(f"✅ Optimization complete")
        print(f"   Optimal point: {optimal_x}")
        print(f"   Optimal value: {optimal_value:.6f}")
        
        # Step 5: Basic validation against true function
        print("\n5. Basic validation...")
        
        # Simple manual validation
        test_points = jnp.array([[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0]])
        true_values = jnp.array([black_box_function(x) for x in test_points])
        pred_values = jnp.array([optimizer.predict(x) for x in test_points])
        
        mse = float(jnp.mean((pred_values - true_values) ** 2))
        
        print("✅ Basic validation complete:")
        print(f"   Test MSE: {mse:.6f}")
        print(f"   Prediction accuracy: {'Good' if mse < 1.0 else 'Needs improvement'}")
        
        validation_metrics = {"mse": mse}
        
        # Step 6: Display training info
        info = optimizer.get_training_info()
        print("\n6. Training summary:")
        print(f"   Surrogate type: {info['surrogate_type']}")
        print(f"   Training samples: {info['n_training_samples']}")
        print(f"   Input dimensions: {info['input_dimension']}")
        
        print("\n" + "=" * 60)
        print("✅ GENERATION 1 COMPLETE: Basic functionality working!")
        print("Ready to proceed to Generation 2: Enhanced robustness")
        print("=" * 60)
        
        return {
            "success": True,
            "optimal_point": optimal_x,
            "optimal_value": optimal_value,
            "validation_metrics": validation_metrics,
            "training_info": info
        }
        
    except Exception as e:
        print(f"❌ Generation 1 failed: {e}")
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    result = generation1_simple_demo()
    if result["success"]:
        print("\n🚀 Generation 1 autonomous implementation successful!")
    else:
        print(f"\n💥 Generation 1 needs fixes: {result['error']}")