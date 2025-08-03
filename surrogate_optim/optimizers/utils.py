"""Utility functions for optimization with surrogates."""

from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array
from scipy.optimize import minimize

from ..models.base import Surrogate
from .base import OptimizationResult
from .gradient_descent import GradientDescentOptimizer


def optimize_with_surrogate(
    surrogate: Surrogate,
    x0: Array,
    method: str = "L-BFGS-B",
    bounds: Optional[List[Tuple[float, float]]] = None,
    options: Optional[Dict[str, Any]] = None,
    use_jax: bool = True
) -> OptimizationResult:
    """Optimize surrogate function using scipy or JAX optimizers.
    
    Args:
        surrogate: Fitted surrogate model.
        x0: Initial point.
        method: Optimization method.
        bounds: Bounds for variables.
        options: Optimizer options.
        use_jax: Whether to use JAX-based optimization.
        
    Returns:
        Optimization result.
    """
    if not surrogate.is_fitted:
        raise ValueError("Surrogate must be fitted before optimization")
    
    x0 = jnp.asarray(x0)
    
    if use_jax:
        # Use JAX-based optimization
        return _optimize_with_jax(surrogate, x0, method, bounds, options)
    else:
        # Use scipy optimization
        return _optimize_with_scipy(surrogate, x0, method, bounds, options)


def _optimize_with_jax(
    surrogate: Surrogate,
    x0: Array,
    method: str,
    bounds: Optional[List[Tuple[float, float]]],
    options: Optional[Dict[str, Any]]
) -> OptimizationResult:
    """Optimize using JAX-based methods."""
    options = options or {}
    
    if method.lower() in ["adam", "sgd", "momentum", "lbfgs"]:
        optimizer = GradientDescentOptimizer(
            surrogate=surrogate,
            method=method.lower(),
            **options
        )
        return optimizer.optimize(x0=x0, bounds=bounds)
    
    else:
        # Fall back to scipy for other methods
        return _optimize_with_scipy(surrogate, x0, method, bounds, options)


def _optimize_with_scipy(
    surrogate: Surrogate,
    x0: Array,
    method: str,
    bounds: Optional[List[Tuple[float, float]]],
    options: Optional[Dict[str, Any]]
) -> OptimizationResult:
    """Optimize using scipy methods."""
    import time
    import numpy as np
    
    start_time = time.time()
    
    # Convert JAX arrays to numpy for scipy
    x0_np = np.asarray(x0)
    
    # Define objective function and gradient
    def objective(x):
        x_jax = jnp.asarray(x)
        return float(surrogate.predict(x_jax))
    
    def gradient(x):
        x_jax = jnp.asarray(x)
        grad_jax = surrogate.gradient(x_jax)
        return np.asarray(grad_jax)
    
    # Set up bounds for scipy
    scipy_bounds = None
    if bounds is not None:
        scipy_bounds = [(low, high) for low, high in bounds]
    
    # Run optimization
    result = minimize(
        fun=objective,
        x0=x0_np,
        method=method,
        jac=gradient,
        bounds=scipy_bounds,
        options=options
    )
    
    optimization_time = time.time() - start_time
    
    # Convert result back to JAX format
    return OptimizationResult(
        x_opt=jnp.asarray(result.x),
        f_opt=float(result.fun),
        n_iterations=result.nit if hasattr(result, 'nit') else 0,
        n_function_evals=result.nfev if hasattr(result, 'nfev') else 0,
        converged=result.success,
        optimization_time=optimization_time,
        metadata={
            "scipy_method": method,
            "scipy_message": result.message,
            "scipy_status": result.status if hasattr(result, 'status') else None,
        }
    )


def multi_start_optimize(
    surrogate: Surrogate,
    bounds: List[Tuple[float, float]],
    n_starts: int = 10,
    method: str = "L-BFGS-B",
    start_method: str = "sobol",
    parallel: bool = True,
    **kwargs
) -> 'GlobalOptimizationResult':
    """Convenience function for multi-start optimization.
    
    Args:
        surrogate: Fitted surrogate model.
        bounds: Bounds for each dimension.
        n_starts: Number of starting points.
        method: Local optimization method.
        start_method: Method for generating starting points.
        parallel: Whether to use parallel optimization.
        **kwargs: Additional arguments for local optimizer.
        
    Returns:
        Global optimization result.
    """
    from .multi_start import MultiStartOptimizer
    
    # Create local optimizer based on method
    if method.lower() in ["adam", "sgd", "momentum", "lbfgs"]:
        local_optimizer = GradientDescentOptimizer(
            surrogate=surrogate,
            method=method.lower()
        )
    else:
        # Create a scipy-based optimizer wrapper
        class ScipyOptimizer:
            def __init__(self, surrogate, method):
                self.surrogate = surrogate
                self.method = method
                self.name = f"scipy_{method}"
            
            def set_surrogate(self, surrogate):
                self.surrogate = surrogate
            
            def optimize(self, x0, bounds=None, **kwargs):
                return optimize_with_surrogate(
                    self.surrogate, x0, self.method, bounds, use_jax=False
                )
        
        local_optimizer = ScipyOptimizer(surrogate, method)
    
    # Create multi-start optimizer
    ms_optimizer = MultiStartOptimizer(
        surrogate=surrogate,
        local_optimizer=local_optimizer,
        n_starts=n_starts,
        start_method=start_method,
        parallel=parallel
    )
    
    return ms_optimizer.optimize(bounds=bounds, **kwargs)


def compute_gradient_error(
    surrogate: Surrogate,
    true_gradient_fn: Callable[[Array], Array],
    test_points: Array,
    metric: str = "relative_l2"
) -> float:
    """Compute gradient approximation error.
    
    Args:
        surrogate: Surrogate model.
        true_gradient_fn: Function that computes true gradients.
        test_points: Points at which to evaluate gradients.
        metric: Error metric ("l2", "relative_l2", "cosine").
        
    Returns:
        Gradient error metric.
    """
    test_points = jnp.asarray(test_points)
    if test_points.ndim == 1:
        test_points = test_points.reshape(1, -1)
    
    # Compute surrogate and true gradients
    surrogate_grads = jnp.stack([
        surrogate.gradient(x) for x in test_points
    ])
    
    true_grads = jnp.stack([
        true_gradient_fn(x) for x in test_points
    ])
    
    if metric == "l2":
        errors = jnp.linalg.norm(surrogate_grads - true_grads, axis=1)
        return float(jnp.mean(errors))
    
    elif metric == "relative_l2":
        errors = jnp.linalg.norm(surrogate_grads - true_grads, axis=1)
        true_norms = jnp.linalg.norm(true_grads, axis=1)
        relative_errors = errors / (true_norms + 1e-12)
        return float(jnp.mean(relative_errors))
    
    elif metric == "cosine":
        # Cosine similarity (higher is better, so we return 1 - similarity)
        dot_products = jnp.sum(surrogate_grads * true_grads, axis=1)
        surrogate_norms = jnp.linalg.norm(surrogate_grads, axis=1)
        true_norms = jnp.linalg.norm(true_grads, axis=1)
        cosine_sims = dot_products / (surrogate_norms * true_norms + 1e-12)
        return float(1.0 - jnp.mean(cosine_sims))
    
    else:
        raise ValueError(f"Unknown metric: {metric}")


def line_search_wolfe(
    f: Callable[[Array], float],
    grad_f: Callable[[Array], Array],
    x: Array,
    direction: Array,
    alpha_init: float = 1.0,
    c1: float = 1e-4,
    c2: float = 0.9,
    max_iterations: int = 20
) -> Tuple[float, int, bool]:
    """Wolfe line search for step size selection.
    
    Args:
        f: Function to minimize.
        grad_f: Gradient function.
        x: Current point.
        direction: Search direction.
        alpha_init: Initial step size.
        c1: Armijo parameter.
        c2: Curvature parameter.
        max_iterations: Maximum iterations.
        
    Returns:
        (step_size, iterations, success).
    """
    alpha = alpha_init
    f0 = f(x)
    grad0 = grad_f(x)
    phi_prime_0 = jnp.dot(grad0, direction)
    
    # Check if direction is descent direction
    if phi_prime_0 >= 0:
        return 0.0, 0, False
    
    for i in range(max_iterations):
        x_new = x + alpha * direction
        f_new = f(x_new)
        
        # Armijo condition
        if f_new > f0 + c1 * alpha * phi_prime_0:
            alpha *= 0.5
            continue
        
        # Curvature condition
        grad_new = grad_f(x_new)
        phi_prime_alpha = jnp.dot(grad_new, direction)
        
        if phi_prime_alpha < c2 * phi_prime_0:
            alpha *= 2.0
            continue
        
        # Both conditions satisfied
        return alpha, i + 1, True
    
    # If we get here, line search failed
    return alpha, max_iterations, False