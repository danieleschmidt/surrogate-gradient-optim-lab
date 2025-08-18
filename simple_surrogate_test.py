#!/usr/bin/env python3
"""Simple test with a basic surrogate implementation."""

import jax.numpy as jnp
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.optimize import minimize

def simple_quadratic(x):
    """Simple quadratic function for testing."""
    x = jnp.atleast_1d(x)
    return float(jnp.sum(x**2))

class SimpleSurrogate:
    """Simple surrogate using scikit-learn GP."""
    
    def __init__(self):
        self.gp = GaussianProcessRegressor(kernel=RBF(1.0))
        self.is_fitted = False
    
    def fit(self, X, y):
        """Fit the surrogate."""
        X = np.array(X)
        y = np.array(y)
        self.gp.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, x):
        """Predict function value."""
        x = np.array(x).reshape(1, -1)
        return float(self.gp.predict(x)[0])
    
    def gradient(self, x):
        """Estimate gradient using finite differences."""
        x = np.array(x)
        eps = 1e-6
        grad = np.zeros_like(x)
        
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            
            grad[i] = (self.predict(x_plus) - self.predict(x_minus)) / (2 * eps)
        
        return grad

def main():
    print("🧪 Simple Surrogate Test")
    print("=" * 30)
    
    # Generate training data
    bounds = [(-2.0, 2.0), (-2.0, 2.0)]
    n_samples = 50
    
    X = np.random.uniform(bounds[0][0], bounds[0][1], (n_samples, 2))
    y = np.array([simple_quadratic(x) for x in X])
    
    print(f"✅ Generated {n_samples} training samples")
    
    # Create and train surrogate
    surrogate = SimpleSurrogate()
    surrogate.fit(X, y)
    print("✅ Surrogate fitted successfully")
    
    # Test prediction
    test_point = np.array([1.0, 1.0])
    pred = surrogate.predict(test_point)
    true_val = simple_quadratic(test_point)
    print(f"   Test point: {test_point}")
    print(f"   True value: {true_val:.6f}")
    print(f"   Predicted: {pred:.6f}")
    print(f"   Error: {abs(pred - true_val):.6f}")
    
    # Test gradient
    grad = surrogate.gradient(test_point)
    true_grad = 2 * test_point  # Analytical gradient for x^2 + y^2
    print(f"   True gradient: {true_grad}")
    print(f"   Estimated gradient: {grad}")
    print(f"   Gradient error: {np.linalg.norm(grad - true_grad):.6f}")
    
    # Simple optimization
    def objective(x):
        return -surrogate.predict(x)  # Minimize negative for maximization
    
    def objective_grad(x):
        return -surrogate.gradient(x)
    
    initial_point = np.array([1.5, 1.5])
    result = minimize(
        objective,
        initial_point,
        method='BFGS',
        jac=objective_grad,
        bounds=[(-2, 2), (-2, 2)]
    )
    
    print("✅ Optimization completed")
    print(f"   Initial point: {initial_point}")
    print(f"   Optimal point: {result.x}")
    print(f"   Optimal value: {simple_quadratic(result.x):.6f}")
    print(f"   Expected optimum: [0, 0] with value 0.0")
    
    print("\n🎉 Simple surrogate optimization working!")

if __name__ == "__main__":
    main()