"""Gradient descent optimizer using surrogate gradients."""

import time
from typing import Any, Dict, List, Optional, Tuple

import jax.numpy as jnp
from jax import Array

from .base import BaseOptimizer, OptimizationResult


class GradientDescentOptimizer(BaseOptimizer):
    """Gradient descent optimizer using surrogate model gradients.
    
    Implements various gradient descent variants including vanilla GD,
    momentum, Adam, and L-BFGS-style quasi-Newton methods.
    """
    
    def __init__(
        self,
        surrogate=None,
        method: str = "adam",
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        line_search: bool = True,
        name: str = "gradient_descent"
    ):
        """Initialize gradient descent optimizer.
        
        Args:
            surrogate: Surrogate model to optimize.
            method: Optimization method ("sgd", "momentum", "adam", "lbfgs").
            learning_rate: Learning rate for gradient steps.
            momentum: Momentum parameter (for momentum and Adam).
            beta1: First moment decay rate (Adam).
            beta2: Second moment decay rate (Adam).
            epsilon: Numerical stability parameter.
            line_search: Whether to use line search for step size.
            name: Optimizer name.
        """
        super().__init__(surrogate, name)
        self.method = method
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.line_search = line_search
    
    def optimize(
        self,
        x0: Array,
        bounds: Optional[List[Tuple[float, float]]] = None,
        constraints: Optional[List[Dict[str, Any]]] = None,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        patience: int = 50,
        verbose: bool = False,
        **kwargs
    ) -> OptimizationResult:
        """Optimize using gradient descent.
        
        Args:
            x0: Initial point.
            bounds: Bounds for each dimension.
            constraints: Constraints (currently not supported).
            max_iterations: Maximum number of iterations.
            tolerance: Convergence tolerance on gradient norm.
            patience: Early stopping patience.
            verbose: Whether to print progress.
            **kwargs: Additional parameters.
            
        Returns:
            Optimization results.
        """
        start_time = time.time()
        
        self._check_surrogate()
        x0 = jnp.asarray(x0)
        bounds = self._validate_bounds(x0, bounds)
        
        if constraints is not None:
            raise NotImplementedError("Constraints not yet supported")
        
        # Initialize optimization state
        x = x0.copy()
        trajectory = [x]
        function_values = [float(self.surrogate.predict(x))]
        gradient_norms = []
        
        # Method-specific state
        if self.method == "momentum":
            velocity = jnp.zeros_like(x)
        elif self.method == "adam":
            m = jnp.zeros_like(x)  # First moment estimate
            v = jnp.zeros_like(x)  # Second moment estimate
            t = 0  # Time step
        elif self.method == "lbfgs":
            # Simplified L-BFGS state
            s_history = []  # Position differences
            y_history = []  # Gradient differences
            max_history = 10
        
        best_f = function_values[0]
        patience_counter = 0
        n_function_evals = 1
        
        for iteration in range(max_iterations):
            # Compute gradient
            grad = self.surrogate.gradient(x)
            grad_norm = float(jnp.linalg.norm(grad))
            gradient_norms.append(grad_norm)
            
            # Check convergence
            if grad_norm < tolerance:
                if verbose:
                    print(f"Converged at iteration {iteration} with gradient norm {grad_norm:.2e}")
                break
            
            # Compute search direction based on method
            if self.method == "sgd":
                direction = -grad
                
            elif self.method == "momentum":
                velocity = self.momentum * velocity - self.learning_rate * grad
                direction = velocity
                
            elif self.method == "adam":
                t += 1
                m = self.beta1 * m + (1 - self.beta1) * grad
                v = self.beta2 * v + (1 - self.beta2) * grad ** 2
                
                # Bias correction
                m_hat = m / (1 - self.beta1 ** t)
                v_hat = v / (1 - self.beta2 ** t)
                
                direction = -self.learning_rate * m_hat / (jnp.sqrt(v_hat) + self.epsilon)
                
            elif self.method == "lbfgs":
                # Simplified L-BFGS direction computation
                direction = self._lbfgs_direction(grad, s_history, y_history)
                
            else:
                raise ValueError(f"Unknown method: {self.method}")
            
            # Determine step size
            if self.line_search and self.method != "adam":
                # Line search for step size
                if self.method == "sgd":
                    step_size, _ = self._line_search(x, direction, alpha_init=self.learning_rate)
                else:
                    step_size, _ = self._line_search(x, direction, alpha_init=1.0)
                x_new = x + step_size * direction
            else:
                # Use fixed step size or method-specific update
                if self.method == "adam" or self.method == "momentum":
                    x_new = x + direction
                else:
                    x_new = x + self.learning_rate * direction
            
            # Apply bounds
            x_new = self._apply_bounds(x_new, bounds)
            
            # Update L-BFGS history
            if self.method == "lbfgs":
                s = x_new - x
                y = self.surrogate.gradient(x_new) - grad
                
                if len(s_history) >= max_history:
                    s_history.pop(0)
                    y_history.pop(0)
                
                s_history.append(s)
                y_history.append(y)
                n_function_evals += 1
            
            # Update position
            x = x_new
            f_new = float(self.surrogate.predict(x))
            n_function_evals += 1
            
            trajectory.append(x)
            function_values.append(f_new)
            
            # Early stopping check
            if f_new < best_f:
                best_f = f_new
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at iteration {iteration}")
                    break
            
            if verbose and iteration % 100 == 0:
                print(f"Iteration {iteration}: f={f_new:.6f}, grad_norm={grad_norm:.2e}")
        
        optimization_time = time.time() - start_time
        
        return OptimizationResult(
            x_opt=x,
            f_opt=float(self.surrogate.predict(x)),
            n_iterations=iteration + 1,
            n_function_evals=n_function_evals,
            converged=grad_norm < tolerance,
            optimization_time=optimization_time,
            trajectory=trajectory,
            function_values=function_values,
            gradient_norms=gradient_norms,
            metadata={
                "method": self.method,
                "final_gradient_norm": grad_norm,
                "learning_rate": self.learning_rate,
            }
        )
    
    def _lbfgs_direction(
        self,
        grad: Array,
        s_history: List[Array],
        y_history: List[Array]
    ) -> Array:
        """Compute L-BFGS search direction.
        
        Args:
            grad: Current gradient.
            s_history: History of position differences.
            y_history: History of gradient differences.
            
        Returns:
            Search direction.
        """
        if not s_history:
            return -grad
        
        # L-BFGS two-loop recursion (simplified)
        q = grad.copy()
        alphas = []
        
        # First loop (backward)
        for s, y in reversed(list(zip(s_history, y_history))):
            rho = 1.0 / (jnp.dot(y, s) + 1e-10)
            alpha = rho * jnp.dot(s, q)
            q = q - alpha * y
            alphas.append(alpha)
        
        alphas.reverse()
        
        # Initial Hessian approximation (identity scaled)
        if s_history and y_history:
            s_last = s_history[-1]
            y_last = y_history[-1]
            gamma = jnp.dot(s_last, y_last) / (jnp.dot(y_last, y_last) + 1e-10)
            r = gamma * q
        else:
            r = q
        
        # Second loop (forward)
        for i, (s, y) in enumerate(zip(s_history, y_history)):
            rho = 1.0 / (jnp.dot(y, s) + 1e-10)
            beta = rho * jnp.dot(y, r)
            r = r + s * (alphas[i] - beta)
        
        return -r