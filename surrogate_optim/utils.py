"""Utility functions for surrogate optimization."""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import jax.numpy as jnp
from scipy.optimize import minimize

from .data import Dataset
from .models import Surrogate

Array = Union[np.ndarray, jnp.ndarray]


def optimize_with_surrogate(
    surrogate: Surrogate,
    x0: Array,
    method: str = "L-BFGS-B",
    bounds: Optional[List[Tuple[float, float]]] = None,
    options: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Optimize using a fitted surrogate model.
    
    Args:
        surrogate: Fitted surrogate model
        x0: Initial point
        method: Optimization method
        bounds: Variable bounds
        options: Optimizer options
        
    Returns:
        Optimization result dictionary
    """
    x0 = np.asarray(x0)
    
    def objective(x):
        """Objective function (negative for maximization)."""
        return -surrogate.predict(x)
        
    def gradient(x):
        """Gradient function."""
        return -surrogate.gradient(x)
        
    # Default options
    if options is None:
        options = {"maxiter": 1000}
        
    result = minimize(
        fun=objective,
        x0=x0,
        method=method,
        jac=gradient,
        bounds=bounds,
        options=options,
    )
    
    return {
        "x": result.x,
        "fun": -result.fun,  # Convert back to maximization
        "success": result.success,
        "message": result.message,
        "nit": result.nit,
        "nfev": result.nfev,
        "njev": result.njev,
    }


def multi_start_optimize(
    surrogate: Surrogate,
    bounds: List[Tuple[float, float]],
    n_starts: int = 10,
    method: str = "L-BFGS-B",
    seed: int = 42,
) -> Dict[str, Any]:
    """Global optimization using multiple random starts.
    
    Args:
        surrogate: Fitted surrogate model
        bounds: Variable bounds
        n_starts: Number of random starts
        method: Local optimization method
        seed: Random seed
        
    Returns:
        Global optimization result dictionary
    """
    np.random.seed(seed)
    bounds = np.array(bounds)
    n_dims = len(bounds)
    
    # Generate random starting points
    starts = np.random.uniform(
        bounds[:, 0], bounds[:, 1], (n_starts, n_dims)
    )
    
    results = []
    for i, start in enumerate(starts):
        try:
            result = optimize_with_surrogate(
                surrogate, start, method=method, bounds=bounds
            )
            result["start_idx"] = i
            result["x0"] = start
            results.append(result)
        except Exception as e:
            print(f"Optimization from start {i} failed: {e}")
            continue
    
    if not results:
        raise RuntimeError("All optimization runs failed")
    
    # Find best result
    best_result = max(results, key=lambda x: x["fun"] if x["success"] else -np.inf)
    
    return {
        "x": best_result["x"],
        "fun": best_result["fun"],
        "best_result": best_result,
        "all_results": results,
        "n_successful": sum(1 for r in results if r["success"]),
        "success_rate": sum(1 for r in results if r["success"]) / len(results),
    }


def benchmark_surrogate(
    surrogate_model: Surrogate,
    test_functions: List[str],
    dimensions: List[int] = [2, 5, 10],
    n_trials: int = 10,
    n_train_samples: int = 100,
    seed: int = 42,
) -> Dict[str, Any]:
    """Benchmark surrogate model on standard test functions.
    
    Args:
        surrogate_model: Surrogate model class (not fitted)
        test_functions: List of test function names
        dimensions: Dimensions to test
        n_trials: Number of trials per configuration
        n_train_samples: Number of training samples
        seed: Random seed
        
    Returns:
        Benchmark results dictionary
    """
    np.random.seed(seed)
    
    # Define test functions
    test_function_map = {
        "rosenbrock": rosenbrock,
        "rastrigin": rastrigin,
        "ackley": ackley,
        "sphere": sphere,
        "griewank": griewank,
    }
    
    results = {
        "function_results": {},
        "summary": {},
    }
    
    all_gaps = []
    all_grad_errors = []
    
    for func_name in test_functions:
        if func_name not in test_function_map:
            print(f"Unknown test function: {func_name}")
            continue
            
        func = test_function_map[func_name]
        func_results = {}
        
        for dim in dimensions:
            dim_results = []
            
            for trial in range(n_trials):
                trial_seed = seed + trial * 100
                result = _benchmark_single(
                    surrogate_model, func, dim, n_train_samples, trial_seed
                )
                dim_results.append(result)
                all_gaps.append(result["optimality_gap"])
                all_grad_errors.append(result["gradient_error"])
                
            func_results[f"dim_{dim}"] = {
                "mean_gap": np.mean([r["optimality_gap"] for r in dim_results]),
                "std_gap": np.std([r["optimality_gap"] for r in dim_results]),
                "mean_grad_error": np.mean([r["gradient_error"] for r in dim_results]),
                "std_grad_error": np.std([r["gradient_error"] for r in dim_results]),
                "trials": dim_results,
            }
            
        results["function_results"][func_name] = func_results
    
    # Overall summary
    results["summary"] = {
        "mean_gap": np.mean(all_gaps),
        "std_gap": np.std(all_gaps),
        "mean_grad_error": np.mean(all_grad_errors),
        "std_grad_error": np.std(all_grad_errors),
        "n_trials_total": len(all_gaps),
    }
    
    return results


def _benchmark_single(
    surrogate_model: Surrogate,
    test_function: Callable,
    dim: int,
    n_train_samples: int,
    seed: int,
) -> Dict[str, float]:
    """Run single benchmark trial."""
    np.random.seed(seed)
    
    # Define bounds for test function
    bounds = [(-5.0, 5.0)] * dim
    bounds_array = np.array(bounds)
    
    # Generate training data
    from .data import collect_data
    data = collect_data(test_function, n_train_samples, bounds, sampling="sobol", random_state=seed)
    
    # Fit surrogate
    import copy
    surrogate = copy.deepcopy(surrogate_model)
    surrogate.fit(data.X, data.y)
    
    # Test optimization
    x0 = np.random.uniform(bounds_array[:, 0], bounds_array[:, 1])
    opt_result = optimize_with_surrogate(surrogate, x0, bounds=bounds)
    
    # Compute true optimum (approximate)
    true_opt_result = minimize(
        lambda x: -test_function(x),
        x0=x0,
        bounds=bounds,
        method="L-BFGS-B",
        options={"maxiter": 1000},
    )
    true_optimum = -true_opt_result.fun
    
    # Compute metrics
    optimality_gap = abs(opt_result["fun"] - true_optimum) / (abs(true_optimum) + 1e-8)
    
    # Gradient error at optimal point
    true_grad = _finite_diff_gradient(test_function, opt_result["x"])
    surrogate_grad = surrogate.gradient(opt_result["x"])
    gradient_error = np.linalg.norm(true_grad - surrogate_grad) / (np.linalg.norm(true_grad) + 1e-8)
    
    return {
        "optimality_gap": optimality_gap,
        "gradient_error": gradient_error,
        "surrogate_optimum": opt_result["fun"],
        "true_optimum": true_optimum,
        "optimization_success": opt_result["success"],
    }


def _finite_diff_gradient(func: Callable, x: Array, eps: float = 1e-6) -> Array:
    """Compute finite difference gradient."""
    x = np.asarray(x)
    grad = np.zeros_like(x)
    
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        grad[i] = (func(x_plus) - func(x_minus)) / (2 * eps)
        
    return grad


# Standard test functions
def rosenbrock(x: Array) -> float:
    """Rosenbrock function."""
    x = np.asarray(x)
    return -np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


def rastrigin(x: Array) -> float:
    """Rastrigin function."""
    x = np.asarray(x)
    n = len(x)
    return -(10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))


def ackley(x: Array) -> float:
    """Ackley function."""
    x = np.asarray(x)
    n = len(x)
    return -(-20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / n)) 
             - np.exp(np.sum(np.cos(2 * np.pi * x)) / n) 
             + 20 + np.e)


def sphere(x: Array) -> float:
    """Sphere function."""
    x = np.asarray(x)
    return -np.sum(x**2)


def griewank(x: Array) -> float:
    """Griewank function."""
    x = np.asarray(x)
    return -(1 + np.sum(x**2) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1)))))


def validate_surrogate(
    surrogate: Surrogate,
    test_function: Callable,
    bounds: List[Tuple[float, float]],
    n_test_points: int = 100,
    seed: int = 42,
) -> Dict[str, float]:
    """Validate surrogate model against true function.
    
    Args:
        surrogate: Fitted surrogate model
        test_function: True function to validate against
        bounds: Variable bounds
        n_test_points: Number of test points
        seed: Random seed
        
    Returns:
        Validation metrics
    """
    np.random.seed(seed)
    bounds = np.array(bounds)
    n_dims = len(bounds)
    
    # Generate test points
    test_X = np.random.uniform(
        bounds[:, 0], bounds[:, 1], (n_test_points, n_dims)
    )
    
    # Compute true and surrogate values
    true_y = np.array([test_function(x) for x in test_X])
    surrogate_y = np.array([surrogate.predict(x) for x in test_X])
    
    # Compute metrics
    mse = np.mean((true_y - surrogate_y)**2)
    mae = np.mean(np.abs(true_y - surrogate_y))
    r2 = 1 - np.sum((true_y - surrogate_y)**2) / np.sum((true_y - np.mean(true_y))**2)
    
    # Gradient validation (sample of points)
    n_grad_test = min(20, n_test_points)
    grad_indices = np.random.choice(n_test_points, n_grad_test, replace=False)
    
    grad_errors = []
    for i in grad_indices:
        x = test_X[i]
        true_grad = _finite_diff_gradient(test_function, x)
        surrogate_grad = surrogate.gradient(x)
        
        grad_error = np.linalg.norm(true_grad - surrogate_grad) / (np.linalg.norm(true_grad) + 1e-8)
        grad_errors.append(grad_error)
    
    return {
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "mean_gradient_error": np.mean(grad_errors),
        "std_gradient_error": np.std(grad_errors),
        "n_test_points": n_test_points,
        "n_gradient_tests": n_grad_test,
    }


def create_benchmark_report(
    results: Dict[str, Any],
    output_file: Optional[str] = None,
) -> str:
    """Create a formatted benchmark report.
    
    Args:
        results: Benchmark results from benchmark_surrogate
        output_file: Optional file to save report
        
    Returns:
        Formatted report string
    """
    report_lines = [
        "# Surrogate Model Benchmark Report",
        "",
        "## Summary",
        f"- Mean optimality gap: {results['summary']['mean_gap']:.4f} ± {results['summary']['std_gap']:.4f}",
        f"- Mean gradient error: {results['summary']['mean_grad_error']:.4f} ± {results['summary']['std_grad_error']:.4f}",
        f"- Total trials: {results['summary']['n_trials_total']}",
        "",
        "## Detailed Results",
        "",
    ]
    
    for func_name, func_results in results["function_results"].items():
        report_lines.append(f"### {func_name.title()}")
        report_lines.append("")
        
        for dim_key, dim_results in func_results.items():
            dim = dim_key.split("_")[1]
            report_lines.extend([
                f"**{dim}D:**",
                f"- Optimality gap: {dim_results['mean_gap']:.4f} ± {dim_results['std_gap']:.4f}",
                f"- Gradient error: {dim_results['mean_grad_error']:.4f} ± {dim_results['std_grad_error']:.4f}",
                "",
            ])
        
        report_lines.append("")
    
    report = "\n".join(report_lines)
    
    if output_file:
        with open(output_file, "w") as f:
            f.write(report)
        print(f"Report saved to {output_file}")
    
    return report