#!/usr/bin/env python3
"""
Generation 1: Simple Working Demo - Surrogate Gradient Optimization
Basic functionality with minimal dependencies (NumPy/SciPy only)
"""

import numpy as np
from scipy.optimize import minimize
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import warnings
warnings.filterwarnings('ignore')

class SimpleSurrogateOptimizer:
    """Generation 1 - Simple, working surrogate optimizer."""
    
    def __init__(self, surrogate_type="neural_network"):
        self.surrogate_type = surrogate_type
        self.model = None
        self.is_fitted = False
        
    def collect_data(self, func, bounds, n_samples=100):
        """Collect training data from black-box function."""
        dim = len(bounds)
        X = np.random.uniform(
            low=[b[0] for b in bounds],
            high=[b[1] for b in bounds], 
            size=(n_samples, dim)
        )
        y = np.array([func(x) for x in X])
        return X, y
        
    def fit_surrogate(self, X, y):
        """Train surrogate model."""
        if self.surrogate_type == "neural_network":
            self.model = MLPRegressor(
                hidden_layer_sizes=(64, 32),
                max_iter=1000,
                random_state=42
            )
        elif self.surrogate_type == "gaussian_process":
            kernel = RBF(length_scale=1.0)
            self.model = GaussianProcessRegressor(
                kernel=kernel,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown surrogate type: {self.surrogate_type}")
            
        self.model.fit(X, y)
        self.is_fitted = True
        return self.model
        
    def predict(self, x):
        """Predict using surrogate model."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        x = np.atleast_2d(x)
        return self.model.predict(x)[0]
        
    def gradient(self, x, epsilon=1e-6):
        """Numerical gradient of surrogate."""
        grad = np.zeros_like(x)
        f0 = self.predict(x)
        
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += epsilon
            f_plus = self.predict(x_plus)
            grad[i] = (f_plus - f0) / epsilon
            
        return grad
        
    def optimize(self, initial_point, bounds=None, method="L-BFGS-B"):
        """Optimize using surrogate gradients."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
            
        def objective(x):
            return -self.predict(x)  # Maximize by minimizing negative
            
        def grad_objective(x):
            return -self.gradient(x)
            
        result = minimize(
            objective,
            initial_point,
            method=method,
            jac=grad_objective,
            bounds=bounds
        )
        
        return result.x, -result.fun

def demo_optimization():
    """Demonstrate basic surrogate optimization."""
    print("🚀 Generation 1: Simple Surrogate Optimization Demo")
    print("="*50)
    
    # Define a test function (2D Himmelblau's function)
    def himmelblau(x):
        return -((x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2)
    
    bounds = [(-5, 5), (-5, 5)]
    
    # Test both surrogate types
    for surrogate_type in ["neural_network", "gaussian_process"]:
        print(f"\n🧠 Testing {surrogate_type.replace('_', ' ').title()} Surrogate")
        print("-" * 30)
        
        # Initialize optimizer
        optimizer = SimpleSurrogateOptimizer(surrogate_type=surrogate_type)
        
        # Collect data
        print("📊 Collecting training data...")
        X, y = optimizer.collect_data(himmelblau, bounds, n_samples=150)
        print(f"   Collected {len(X)} samples")
        print(f"   Function range: [{y.min():.3f}, {y.max():.3f}]")
        
        # Train surrogate
        print("🎓 Training surrogate model...")
        optimizer.fit_surrogate(X, y)
        
        # Optimize
        print("⚡ Optimizing with surrogate gradients...")
        initial_point = np.array([2.0, 2.0])
        optimal_x, optimal_value = optimizer.optimize(initial_point, bounds)
        
        # Validate with true function
        true_value = himmelblau(optimal_x)
        
        print(f"   Optimum found at: [{optimal_x[0]:.3f}, {optimal_x[1]:.3f}]")
        print(f"   Surrogate value: {optimal_value:.3f}")
        print(f"   True value: {true_value:.3f}")
        print(f"   Approximation error: {abs(optimal_value - true_value):.3f}")
        
        # Known optimal points for Himmelblau's function
        known_optima = [
            np.array([3.0, 2.0]),
            np.array([-2.805118, 3.131312]),
            np.array([-3.779310, -3.283186]),
            np.array([3.584428, -1.848126])
        ]
        
        min_distance = min(np.linalg.norm(optimal_x - opt) for opt in known_optima)
        print(f"   Distance to nearest known optimum: {min_distance:.3f}")
        
        success = min_distance < 1.0 and abs(optimal_value - true_value) < 1.0
        print(f"   ✅ SUCCESS" if success else f"   ⚠️  NEEDS IMPROVEMENT")

if __name__ == "__main__":
    demo_optimization()