"""Basic usage example for Surrogate Gradient Optimization Lab."""

import jax.numpy as jnp
import matplotlib.pyplot as plt

from surrogate_optim import (
    collect_data,
    NeuralSurrogate,
    GPSurrogate,
    HybridSurrogate, 
    optimize_with_surrogate,
    GradientDescentOptimizer,
    TrustRegionOptimizer,
)


def rosenbrock_function(x):
    """The Rosenbrock function - a classic optimization test case."""
    x = jnp.asarray(x)
    if x.ndim == 0:
        x = jnp.array([x])
    
    result = 0.0
    for i in range(len(x) - 1):
        result += 100.0 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    return result


def rosenbrock_gradient(x):
    """Analytical gradient of the Rosenbrock function."""
    x = jnp.asarray(x)
    if x.ndim == 0:
        x = jnp.array([x])
    
    grad = jnp.zeros_like(x)
    
    # First component
    if len(x) > 1:
        grad = grad.at[0].set(-400.0 * x[0] * (x[1] - x[0]**2) - 2.0 * (1 - x[0]))
    
    # Middle components
    for i in range(1, len(x) - 1):
        grad = grad.at[i].set(200.0 * (x[i] - x[i-1]**2) - 400.0 * x[i] * (x[i+1] - x[i]**2) - 2.0 * (1 - x[i]))
    
    # Last component
    if len(x) > 1:
        grad = grad.at[-1].set(200.0 * (x[-1] - x[-2]**2))
    
    return grad


def main():
    """Demonstrate basic surrogate optimization workflow."""
    print("🚀 Surrogate Gradient Optimization Lab - Basic Example")
    print("=" * 60)
    
    # Problem setup
    bounds = [(-2.0, 2.0), (-1.0, 3.0)]  # 2D Rosenbrock
    n_samples = 100
    
    print(f"\n📊 Problem Setup:")
    print(f"Function: Rosenbrock (2D)")
    print(f"Bounds: {bounds}")
    print(f"Training samples: {n_samples}")
    
    # Step 1: Collect training data
    print(f"\n📈 Step 1: Collecting training data...")
    dataset = collect_data(
        function=rosenbrock_function,
        n_samples=n_samples,
        bounds=bounds,
        sampling="sobol",  # Use Sobol sequence for better coverage
        gradient_function=rosenbrock_gradient,  # Include true gradients
        seed=42
    )
    
    print(f"✓ Collected {dataset.n_samples} samples")
    print(f"  Dataset statistics: y ∈ [{dataset.y.min():.3f}, {dataset.y.max():.3f}]")
    print(f"  Has gradients: {dataset.has_gradients}")
    
    # Step 2: Train different surrogate models
    print(f"\n🧠 Step 2: Training surrogate models...")
    
    # Neural Network Surrogate
    print("  Training Neural Network surrogate...")
    nn_surrogate = NeuralSurrogate(
        hidden_dims=[64, 64],
        activation="relu",
        learning_rate=0.01,
        ensemble_size=3,  # Use ensemble for uncertainty
        normalize_inputs=True
    )
    
    train_dataset, val_dataset = dataset.split(train_ratio=0.8)
    nn_result = nn_surrogate.fit(
        train_dataset.X, train_dataset.y,
        X_val=val_dataset.X, y_val=val_dataset.y,
        epochs=500,
        early_stopping_patience=50,
        verbose=False
    )
    
    print(f"    ✓ Training loss: {nn_result.training_loss:.6f}")
    print(f"    ✓ Validation loss: {nn_result.validation_loss:.6f}")
    print(f"    ✓ Training time: {nn_result.training_time:.2f}s")
    
    # Gaussian Process Surrogate
    print("  Training Gaussian Process surrogate...")
    gp_surrogate = GPSurrogate(
        kernel="rbf",
        normalize_y=True,
        normalize_inputs=True
    )
    
    gp_result = gp_surrogate.fit(
        train_dataset.X, train_dataset.y,
        X_val=val_dataset.X, y_val=val_dataset.y
    )
    
    print(f"    ✓ Training loss: {gp_result.training_loss:.6f}")
    print(f"    ✓ Training time: {gp_result.training_time:.2f}s")
    
    # Hybrid Ensemble
    print("  Training Hybrid Ensemble...")
    hybrid_surrogate = HybridSurrogate(
        models=[
            ("neural_net", nn_surrogate),
            ("gaussian_process", gp_surrogate),
        ],
        aggregation="weighted_average",
        weight_optimization="cv"
    )
    
    hybrid_result = hybrid_surrogate.fit(
        train_dataset.X, train_dataset.y,
        X_val=val_dataset.X, y_val=val_dataset.y
    )
    
    print(f"    ✓ Ensemble weights: {hybrid_surrogate.get_model_weights()}")
    print(f"    ✓ Training time: {hybrid_result.training_time:.2f}s")
    
    # Step 3: Optimization using different methods
    print(f"\n🎯 Step 3: Optimization with surrogate gradients...")
    
    x0 = jnp.array([0.0, 0.0])  # Starting point
    true_optimum = jnp.array([1.0, 1.0])  # Known Rosenbrock optimum
    
    optimizers = [
        ("Gradient Descent (Adam)", lambda s: GradientDescentOptimizer(s, method="adam")),
        ("Trust Region", lambda s: TrustRegionOptimizer(s, true_function=rosenbrock_function)),
        ("Scipy L-BFGS-B", lambda s: None),  # Will use scipy directly
    ]
    
    surrogates = [
        ("Neural Network", nn_surrogate),
        ("Gaussian Process", gp_surrogate), 
        ("Hybrid Ensemble", hybrid_surrogate),
    ]
    
    results = {}
    
    for surrogate_name, surrogate in surrogates:
        print(f"\n  🔧 Using {surrogate_name} surrogate:")
        
        for opt_name, opt_factory in optimizers:
            try:
                if opt_factory is None:
                    # Use scipy L-BFGS-B
                    result = optimize_with_surrogate(
                        surrogate=surrogate,
                        x0=x0,
                        method="L-BFGS-B",
                        bounds=bounds,
                        use_jax=False
                    )
                else:
                    # Use custom optimizer
                    optimizer = opt_factory(surrogate)
                    result = optimizer.optimize(
                        x0=x0,
                        bounds=bounds,
                        max_iterations=1000,
                        tolerance=1e-6,
                        verbose=False
                    )
                
                # Evaluate true function at optimum
                true_value = rosenbrock_function(result.x_opt)
                distance_to_optimum = jnp.linalg.norm(result.x_opt - true_optimum)
                
                results[(surrogate_name, opt_name)] = {
                    "x_opt": result.x_opt,
                    "f_surrogate": result.f_opt,
                    "f_true": true_value,
                    "distance": distance_to_optimum,
                    "converged": result.converged,
                    "iterations": result.n_iterations,
                    "time": result.optimization_time
                }
                
                print(f"    {opt_name:20s}: x* = [{result.x_opt[0]:6.3f}, {result.x_opt[1]:6.3f}], "
                      f"f* = {true_value:8.5f}, dist = {distance_to_optimum:6.3f}")
                
            except Exception as e:
                print(f"    {opt_name:20s}: FAILED ({str(e)[:50]}...)")
    
    # Step 4: Analysis and comparison
    print(f"\n📊 Step 4: Results Analysis")
    print("=" * 80)
    print(f"{'Surrogate':<20} {'Optimizer':<20} {'Distance':<10} {'True f*':<10} {'Time(s)':<8}")
    print("-" * 80)
    
    best_result = None
    best_distance = float('inf')
    
    for (surrogate_name, opt_name), result in results.items():
        print(f"{surrogate_name:<20} {opt_name:<20} {result['distance']:<10.4f} "
              f"{result['f_true']:<10.5f} {result['time']:<8.3f}")
        
        if result['distance'] < best_distance:
            best_distance = result['distance']
            best_result = (surrogate_name, opt_name, result)
    
    if best_result:
        surrogate_name, opt_name, result = best_result
        print(f"\n🏆 Best Result: {surrogate_name} + {opt_name}")
        print(f"   Optimal point: [{result['x_opt'][0]:.6f}, {result['x_opt'][1]:.6f}]")
        print(f"   True optimum:  [1.000000, 1.000000]")
        print(f"   Distance: {result['distance']:.6f}")
        print(f"   Function value: {result['f_true']:.6f}")
    
    # Step 5: Gradient accuracy analysis
    print(f"\n🎯 Step 5: Gradient Accuracy Analysis")
    
    test_points = jnp.array([
        [0.5, 0.5],
        [1.0, 1.0], 
        [-0.5, 1.5],
        [0.0, 0.0]
    ])
    
    print(f"{'Point':<15} {'Surrogate':<15} {'L2 Error':<10} {'Cosine Sim':<12}")
    print("-" * 60)
    
    for i, x in enumerate(test_points):
        true_grad = rosenbrock_gradient(x)
        
        for surrogate_name, surrogate in surrogates:
            surrogate_grad = surrogate.gradient(x)
            
            l2_error = jnp.linalg.norm(surrogate_grad - true_grad)
            cosine_sim = jnp.dot(surrogate_grad, true_grad) / (
                jnp.linalg.norm(surrogate_grad) * jnp.linalg.norm(true_grad) + 1e-12
            )
            
            print(f"[{x[0]:4.1f}, {x[1]:4.1f}]    {surrogate_name:<15} {l2_error:<10.4f} {cosine_sim:<12.4f}")
    
    print(f"\n✅ Example completed successfully!")
    print(f"🔬 This demonstrates the complete surrogate optimization workflow:")
    print(f"   • Data collection with multiple sampling strategies")
    print(f"   • Training various surrogate model types")
    print(f"   • Optimization using learned gradients")
    print(f"   • Performance comparison and analysis")


if __name__ == "__main__":
    main()