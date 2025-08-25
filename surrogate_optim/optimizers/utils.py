"""Utility functions for optimization with surrogate models."""

from typing import Any, Dict, List, Optional, Tuple

from jax import Array
import jax.numpy as jnp

from ..models.base import Surrogate
from .base import OptimizationResult
from .gradient_descent import GradientDescentOptimizer
from .multi_start import MultiStartOptimizer
from .trust_region import TrustRegionOptimizer


def optimize_with_surrogate(
    surrogate: Surrogate,
    x0: Array,
    method: str = "gradient_descent",
    bounds: Optional[List[Tuple[float, float]]] = None,
    options: Optional[Dict[str, Any]] = None,
) -> OptimizationResult:
    """Convenience function for optimizing with surrogate models.
    
    Args:
        surrogate: Trained surrogate model
        x0: Initial point for optimization
        method: Optimization method ('gradient_descent', 'trust_region', 'multi_start')
        bounds: Optional bounds for each dimension
        options: Optional method-specific parameters
        
    Returns:
        Optimization result
    """
    if options is None:
        options = {}

    if method == "gradient_descent":
        optimizer = GradientDescentOptimizer(**options)
    elif method == "trust_region":
        optimizer = TrustRegionOptimizer(**options)
    elif method == "multi_start":
        optimizer = MultiStartOptimizer(**options)
    else:
        raise ValueError(f"Unknown optimization method: {method}")

    return optimizer.optimize(surrogate, x0, bounds)


def compare_optimizers(
    surrogate: Surrogate,
    x0: Array,
    bounds: Optional[List[Tuple[float, float]]] = None,
    methods: Optional[List[str]] = None,
    options_dict: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, OptimizationResult]:
    """Compare different optimization methods on the same problem.
    
    Args:
        surrogate: Trained surrogate model
        x0: Initial point for optimization
        bounds: Optional bounds for each dimension
        methods: List of methods to compare
        options_dict: Method-specific options
        
    Returns:
        Dictionary mapping method names to optimization results
    """
    if methods is None:
        methods = ["gradient_descent", "trust_region", "multi_start"]

    if options_dict is None:
        options_dict = {}

    results = {}

    for method in methods:
        print(f"Running optimization with {method}...")

        try:
            options = options_dict.get(method, {})
            result = optimize_with_surrogate(
                surrogate=surrogate,
                x0=x0,
                method=method,
                bounds=bounds,
                options=options
            )
            results[method] = result

            status = "✓" if result.success else "✗"
            print(f"  {status} {method}: f = {result.fun:.6f}, iterations = {result.nit}")

        except Exception as e:
            print(f"  ✗ {method}: Failed with error: {e}")
            results[method] = OptimizationResult(
                x=x0,
                fun=float("inf"),
                success=False,
                message=f"Error: {e}",
                nit=0,
                nfev=0
            )

    return results


def create_optimization_summary(result: OptimizationResult) -> str:
    """Create a formatted summary of optimization results.
    
    Args:
        result: Optimization result to summarize
        
    Returns:
        Formatted summary string
    """
    summary = []
    summary.append("=== Optimization Summary ===")
    summary.append(f"Success: {'✓' if result.success else '✗'}")
    summary.append(f"Final value: {result.fun:.8f}")
    summary.append(f"Optimal point: {result.x}")
    summary.append(f"Iterations: {result.nit}")
    summary.append(f"Function evaluations: {result.nfev}")
    summary.append(f"Message: {result.message}")

    if result.metadata:
        summary.append("\n--- Additional Information ---")
        for key, value in result.metadata.items():
            if key not in ["trajectory", "convergence_history", "local_results"]:
                summary.append(f"{key}: {value}")

    if result.trajectory and len(result.trajectory) > 1:
        trajectory_array = jnp.stack([jnp.asarray(point) for point in result.trajectory])
        total_distance = float(jnp.sum(jnp.linalg.norm(
            trajectory_array[1:] - trajectory_array[:-1], axis=1
        )))
        summary.append(f"Total trajectory distance: {total_distance:.6f}")

    return "\n".join(summary)


def analyze_convergence(result: OptimizationResult) -> Dict[str, Any]:
    """Analyze convergence properties of optimization result.
    
    Args:
        result: Optimization result to analyze
        
    Returns:
        Dictionary with convergence analysis
    """
    analysis = {
        "converged": result.success,
        "final_value": float(result.fun),
        "iterations": result.nit,
        "function_evaluations": result.nfev,
    }

    if result.convergence_history and len(result.convergence_history) > 1:
        history = jnp.array(result.convergence_history)

        # Function value improvements
        improvements = history[:-1] - history[1:]
        positive_improvements = improvements[improvements > 0]

        analysis.update({
            "initial_value": float(history[0]),
            "total_improvement": float(history[0] - history[-1]),
            "mean_improvement_per_iteration": float(jnp.mean(improvements)),
            "largest_improvement": float(jnp.max(improvements)),
            "n_improving_steps": int(jnp.sum(improvements > 0)),
            "improvement_rate": float(len(positive_improvements) / len(improvements)),
        })

        # Convergence rate estimation (linear)
        if len(history) > 10:
            # Fit linear trend to log of function values (if positive)
            if jnp.all(history > 0):
                log_history = jnp.log(history)
                n = len(log_history)
                x = jnp.arange(n)

                # Linear regression
                A = jnp.vstack([x, jnp.ones(n)]).T
                slope, intercept = jnp.linalg.lstsq(A, log_history, rcond=None)[0]

                analysis["convergence_rate"] = float(slope)
                analysis["convergence_fit_quality"] = float(
                    1 - jnp.var(log_history - (slope * x + intercept)) / jnp.var(log_history)
                )

    if result.trajectory and len(result.trajectory) > 1:
        trajectory_array = jnp.stack([jnp.asarray(point) for point in result.trajectory])

        # Step sizes
        step_sizes = jnp.linalg.norm(trajectory_array[1:] - trajectory_array[:-1], axis=1)

        analysis.update({
            "mean_step_size": float(jnp.mean(step_sizes)),
            "min_step_size": float(jnp.min(step_sizes)),
            "max_step_size": float(jnp.max(step_sizes)),
            "final_step_size": float(step_sizes[-1]),
        })

    return analysis
