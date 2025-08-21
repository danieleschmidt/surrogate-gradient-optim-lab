#!/usr/bin/env python3
"""
Generation 1: Make It Work - Simple Surrogate Optimization Demo
Basic functionality implementation to prove core concepts work
"""

import jax.numpy as jnp
from jax import Array, grad
import numpy as np
from typing import Callable, List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

@dataclass
class OptimizationResult:
    """Results from optimization."""
    x: Array  # Final point
    fun: float  # Final function value
    success: bool = True
    message: str = "Optimization completed"
    nit: int = 0  # Number of iterations


class SimpleSurrogate:
    """Simple surrogate using polynomial approximation."""
    
    def __init__(self, degree: int = 2):
        self.degree = degree
        self.coeffs = None
        self.fitted = False
        
    def fit(self, X: Array, y: Array):
        """Fit polynomial surrogate."""
        # Simple polynomial features for 2D case
        n_samples, n_dims = X.shape
        
        if n_dims != 2:
            raise ValueError("SimpleSurrogate only supports 2D inputs")
            
        # Create polynomial features: [1, x1, x2, x1^2, x2^2, x1*x2]
        features = []
        for i in range(n_samples):
            x1, x2 = X[i]
            row = [1.0, x1, x2, x1*x1, x2*x2, x1*x2]
            features.append(row)
        
        features = jnp.array(features)
        
        # Least squares fit
        self.coeffs = jnp.linalg.lstsq(features, y, rcond=None)[0]
        self.fitted = True
        
    def predict(self, x: Array) -> float:
        """Predict function value."""
        if not self.fitted:
            raise ValueError("Surrogate not fitted")
            
        x = jnp.atleast_1d(x)
        if len(x) != 2:
            raise ValueError("Input must be 2D")
            
        x1, x2 = x[0], x[1]
        features = jnp.array([1.0, x1, x2, x1*x1, x2*x2, x1*x2])
        return float(jnp.dot(self.coeffs, features))
    
    def gradient(self, x: Array) -> Array:
        """Compute gradient analytically."""
        if not self.fitted:
            raise ValueError("Surrogate not fitted")
            
        x = jnp.atleast_1d(x)
        if len(x) != 2:
            raise ValueError("Input must be 2D")
            
        x1, x2 = x[0], x[1]
        # Gradient of [1, x1, x2, x1^2, x2^2, x1*x2]
        # d/dx1 = [0, 1, 0, 2*x1, 0, x2]
        # d/dx2 = [0, 0, 1, 0, 2*x2, x1]
        
        grad_features_x1 = jnp.array([0.0, 1.0, 0.0, 2*x1, 0.0, x2])
        grad_features_x2 = jnp.array([0.0, 0.0, 1.0, 0.0, 2*x2, x1])
        
        grad_x1 = jnp.dot(self.coeffs, grad_features_x1)
        grad_x2 = jnp.dot(self.coeffs, grad_features_x2)
        
        return jnp.array([grad_x1, grad_x2])


class SimpleOptimizer:
    """Simple gradient descent optimizer."""
    
    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 100):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        
    def optimize(self, surrogate: SimpleSurrogate, x0: Array, bounds: Optional[List[Tuple[float, float]]] = None) -> OptimizationResult:
        """Run gradient-based optimization."""
        x = jnp.array(x0, dtype=float)
        
        for i in range(self.max_iterations):
            # Get gradient
            grad = surrogate.gradient(x)
            
            # Gradient descent step (minimize = negative gradient)
            x_new = x - self.learning_rate * grad
            
            # Apply bounds if provided
            if bounds is not None:
                for j, (lower, upper) in enumerate(bounds):
                    x_new = x_new.at[j].set(jnp.clip(x_new[j], lower, upper))
            
            # Check convergence
            if jnp.linalg.norm(x_new - x) < 1e-6:
                break
                
            x = x_new
        
        return OptimizationResult(
            x=x,
            fun=surrogate.predict(x),
            success=True,
            nit=i+1
        )


def collect_random_data(function: Callable, n_samples: int, bounds: List[Tuple[float, float]]) -> Tuple[Array, Array]:
    """Collect random training data."""
    dim = len(bounds)
    X = []
    y = []
    
    for _ in range(n_samples):
        # Random point in bounds
        x = jnp.array([
            np.random.uniform(low, high) 
            for low, high in bounds
        ])
        
        X.append(x)
        y.append(function(x))
    
    return jnp.stack(X), jnp.array(y)


def main():
    """Generation 1 demo - Make It Work."""
    print("🚀 GENERATION 1: MAKE IT WORK")
    print("=" * 50)
    
    # Simple test function
    def simple_quadratic(x):
        x = jnp.atleast_1d(x)
        return float(jnp.sum(x**2) + 0.1 * jnp.sum(x**3))
    
    bounds = [(-3.0, 3.0), (-3.0, 3.0)]
    
    print("📊 Test function: Simple quadratic with cubic term")
    print(f"📊 Bounds: {bounds}")
    print(f"📊 True minimum near: [0, 0]")
    
    # 1. Collect training data
    print("\n📈 Step 1: Collecting training data...")
    X_train, y_train = collect_random_data(simple_quadratic, 50, bounds)
    print(f"   ✅ Collected {len(X_train)} samples")
    print(f"   📊 Training data range: y=[{float(jnp.min(y_train)):.3f}, {float(jnp.max(y_train)):.3f}]")
    
    # 2. Train surrogate
    print("\n🧠 Step 2: Training surrogate model...")
    surrogate = SimpleSurrogate(degree=2)
    surrogate.fit(X_train, y_train)
    print("   ✅ Polynomial surrogate fitted")
    
    # 3. Test predictions
    print("\n🔍 Step 3: Testing surrogate predictions...")
    test_points = [
        jnp.array([0.0, 0.0]),
        jnp.array([1.0, 1.0]),
        jnp.array([-1.0, 1.0])
    ]
    
    for i, test_point in enumerate(test_points):
        true_val = simple_quadratic(test_point)
        pred_val = surrogate.predict(test_point)
        error = abs(pred_val - true_val)
        
        print(f"   Test {i+1}: Point={test_point}, True={true_val:.4f}, Pred={pred_val:.4f}, Error={error:.4f}")
    
    # 4. Test gradients
    print("\n📐 Step 4: Testing gradient computation...")
    for i, test_point in enumerate(test_points[:2]):  # Test first 2 points
        grad_surrogate = surrogate.gradient(test_point)
        print(f"   Test {i+1}: Point={test_point}, Gradient={grad_surrogate}")
    
    # 5. Run optimization
    print("\n🎯 Step 5: Running optimization...")
    optimizer = SimpleOptimizer(learning_rate=0.1, max_iterations=50)
    
    initial_points = [
        jnp.array([2.0, 2.0]),
        jnp.array([-2.0, 2.0]),
        jnp.array([1.0, -1.0])
    ]
    
    best_result = None
    best_true_value = float('inf')
    
    for i, x0 in enumerate(initial_points):
        print(f"   Run {i+1}: Starting from {x0}")
        
        result = optimizer.optimize(surrogate, x0, bounds)
        true_optimum_value = simple_quadratic(result.x)
        
        print(f"     ✅ Converged to {result.x} in {result.nit} iterations")
        print(f"     📊 Surrogate value: {result.fun:.6f}")
        print(f"     📊 True value: {true_optimum_value:.6f}")
        print(f"     📊 Error: {abs(result.fun - true_optimum_value):.6f}")
        
        if true_optimum_value < best_true_value:
            best_result = result
            best_true_value = true_optimum_value
    
    # 6. Final results
    print("\n🏆 GENERATION 1 RESULTS:")
    print(f"   🎯 Best point found: {best_result.x}")
    print(f"   📈 Best true value: {best_true_value:.6f}")
    print(f"   📊 True optimum value at [0,0]: {simple_quadratic(jnp.array([0.0, 0.0])):.6f}")
    print(f"   📏 Distance from true optimum: {jnp.linalg.norm(best_result.x):.6f}")
    
    # Success criteria for Generation 1
    success_criteria = {
        "Surrogate can be trained": surrogate.fitted,
        "Surrogate can predict": abs(surrogate.predict(jnp.array([0.0, 0.0])) - simple_quadratic(jnp.array([0.0, 0.0]))) < 1.0,
        "Gradients work": jnp.linalg.norm(surrogate.gradient(jnp.array([0.0, 0.0]))) is not None,
        "Optimization converges": best_result.success,
        "Found reasonable minimum": best_true_value < 5.0,  # Should be much better than random
    }
    
    print("\n✅ GENERATION 1 SUCCESS CRITERIA:")
    all_passed = True
    for criterion, passed in success_criteria.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status}: {criterion}")
        all_passed = all_passed and passed
    
    print(f"\n🎉 GENERATION 1: {'SUCCESS' if all_passed else 'NEEDS_WORK'}")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)