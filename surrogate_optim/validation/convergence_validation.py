"""Convergence validation and analysis utilities."""

from typing import Any, Dict, List, Optional
import warnings

from jax import Array
import jax.numpy as jnp

from ..optimizers.base import OptimizationResult
from .input_validation import ValidationError, ValidationWarning


def validate_convergence(
    result: OptimizationResult,
    convergence_criteria: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Validate optimization convergence.
    
    Args:
        result: Optimization result to validate
        convergence_criteria: Optional convergence criteria
        
    Returns:
        Convergence validation results
        
    Raises:
        ValidationError: If convergence is unacceptable
    """
    if convergence_criteria is None:
        convergence_criteria = {
            "min_improvement": 1e-8,
            "max_stagnation_steps": 50,
            "min_success_rate": 0.8,
            "max_gradient_norm": 1e-3,
        }

    validation_results = {
        "converged": result.success,
        "final_value": float(result.fun),
        "iterations": result.nit,
        "function_evaluations": result.nfev,
        "passed": True,
        "warnings": [],
        "errors": [],
    }

    # Basic success check
    if not result.success:
        validation_results["errors"].append(f"Optimization failed: {result.message}")
        validation_results["passed"] = False

    # Check if result is finite
    if not jnp.isfinite(result.fun):
        validation_results["errors"].append(f"Final function value is not finite: {result.fun}")
        validation_results["passed"] = False

    # Analyze convergence history if available
    if result.convergence_history and len(result.convergence_history) > 1:
        history = jnp.array(result.convergence_history)

        # Total improvement
        total_improvement = float(history[0] - history[-1])
        validation_results["total_improvement"] = total_improvement

        if "min_improvement" in convergence_criteria:
            min_improvement = convergence_criteria["min_improvement"]
            if total_improvement < min_improvement:
                validation_results["warnings"].append(
                    f"Total improvement {total_improvement:.2e} is below threshold {min_improvement:.2e}"
                )

        # Check for stagnation
        if "max_stagnation_steps" in convergence_criteria:
            max_stagnation = convergence_criteria["max_stagnation_steps"]
            stagnation_count = _count_stagnation_steps(history)
            validation_results["stagnation_steps"] = stagnation_count

            if stagnation_count > max_stagnation:
                validation_results["warnings"].append(
                    f"Optimization stagnated for {stagnation_count} steps "
                    f"(threshold: {max_stagnation})"
                )

        # Convergence rate analysis
        convergence_rate = _estimate_convergence_rate(history)
        if convergence_rate is not None:
            validation_results["convergence_rate"] = convergence_rate

            if convergence_rate > -1e-6:  # Should be negative for minimization
                validation_results["warnings"].append(
                    f"Poor convergence rate: {convergence_rate:.2e}"
                )

        # Check for oscillations
        oscillation_measure = _measure_oscillations(history)
        validation_results["oscillation_measure"] = oscillation_measure

        if oscillation_measure > 0.5:
            validation_results["warnings"].append(
                f"High oscillation detected: {oscillation_measure:.3f}"
            )

    # Analyze trajectory if available
    if result.trajectory and len(result.trajectory) > 1:
        trajectory_analysis = _analyze_trajectory(result.trajectory)
        validation_results.update(trajectory_analysis)

        # Check for excessive movement
        if trajectory_analysis.get("total_distance", 0) > 100:
            validation_results["warnings"].append(
                f"Excessive trajectory distance: {trajectory_analysis['total_distance']:.2f}"
            )

    # Method-specific validation
    if result.metadata:
        method = result.metadata.get("method", "unknown")

        if method == "gradient_descent":
            _validate_gradient_descent_result(result, validation_results, convergence_criteria)
        elif method == "trust_region":
            _validate_trust_region_result(result, validation_results, convergence_criteria)
        elif method == "multi_start":
            _validate_multi_start_result(result, validation_results, convergence_criteria)

    # Issue warnings
    for warning in validation_results["warnings"]:
        warnings.warn(warning, ValidationWarning)

    # Check if validation passed
    if validation_results["errors"]:
        validation_results["passed"] = False
        error_msg = "Convergence validation failed: " + "; ".join(validation_results["errors"])
        raise ValidationError(error_msg)

    return validation_results


def _count_stagnation_steps(history: Array, tolerance: float = 1e-8) -> int:
    """Count number of consecutive steps with minimal improvement."""
    if len(history) < 2:
        return 0

    improvements = history[:-1] - history[1:]  # Positive for minimization
    stagnation_steps = 0
    max_stagnation = 0

    for improvement in improvements:
        if improvement < tolerance:
            stagnation_steps += 1
            max_stagnation = max(max_stagnation, stagnation_steps)
        else:
            stagnation_steps = 0

    return max_stagnation


def _estimate_convergence_rate(history: Array) -> Optional[float]:
    """Estimate linear convergence rate from history."""
    if len(history) < 10:
        return None

    # Use last half of history to avoid transient behavior
    recent_history = history[len(history)//2:]

    # Fit linear trend to log of absolute values
    if jnp.all(recent_history > 0):
        log_history = jnp.log(recent_history)
        n = len(log_history)
        x = jnp.arange(n)

        # Linear regression: log(f) = a*x + b
        A = jnp.vstack([x, jnp.ones(n)]).T
        try:
            slope, _ = jnp.linalg.lstsq(A, log_history, rcond=None)[0]
            return float(slope)
        except:
            return None

    return None


def _measure_oscillations(history: Array) -> float:
    """Measure oscillation level in convergence history."""
    if len(history) < 3:
        return 0.0

    # Count direction changes
    diff = jnp.diff(history)
    sign_changes = jnp.sum(jnp.diff(jnp.sign(diff)) != 0)

    # Normalize by possible maximum sign changes
    max_changes = len(diff) - 1

    return float(sign_changes / max_changes) if max_changes > 0 else 0.0


def _analyze_trajectory(trajectory: List[Array]) -> Dict[str, Any]:
    """Analyze optimization trajectory properties."""
    if len(trajectory) < 2:
        return {}

    trajectory_array = jnp.stack([jnp.asarray(point) for point in trajectory])

    # Step sizes
    steps = trajectory_array[1:] - trajectory_array[:-1]
    step_sizes = jnp.linalg.norm(steps, axis=1)

    # Total distance traveled
    total_distance = float(jnp.sum(step_sizes))

    # Displacement (straight-line distance from start to end)
    displacement = float(jnp.linalg.norm(trajectory_array[-1] - trajectory_array[0]))

    # Path efficiency (displacement / total_distance)
    efficiency = displacement / total_distance if total_distance > 1e-12 else 0.0

    return {
        "total_distance": total_distance,
        "displacement": displacement,
        "path_efficiency": efficiency,
        "mean_step_size": float(jnp.mean(step_sizes)),
        "max_step_size": float(jnp.max(step_sizes)),
        "min_step_size": float(jnp.min(step_sizes)),
        "final_step_size": float(step_sizes[-1]),
    }


def _validate_gradient_descent_result(
    result: OptimizationResult,
    validation_results: Dict[str, Any],
    criteria: Dict[str, Any],
) -> None:
    """Validate gradient descent specific results."""
    if result.metadata:
        final_grad_norm = result.metadata.get("final_grad_norm")
        if final_grad_norm is not None:
            validation_results["final_gradient_norm"] = final_grad_norm

            max_grad_norm = criteria.get("max_gradient_norm", 1e-3)
            if final_grad_norm > max_grad_norm:
                validation_results["warnings"].append(
                    f"Final gradient norm {final_grad_norm:.2e} exceeds threshold {max_grad_norm:.2e}"
                )

        final_lr = result.metadata.get("final_lr")
        if final_lr is not None:
            validation_results["final_learning_rate"] = final_lr

            if final_lr < 1e-10:
                validation_results["warnings"].append(
                    f"Learning rate became very small: {final_lr:.2e}"
                )


def _validate_trust_region_result(
    result: OptimizationResult,
    validation_results: Dict[str, Any],
    criteria: Dict[str, Any],
) -> None:
    """Validate trust region specific results."""
    if result.metadata:
        final_radius = result.metadata.get("final_radius")
        if final_radius is not None:
            validation_results["final_trust_radius"] = final_radius

            if final_radius < 1e-10:
                validation_results["warnings"].append(
                    f"Trust region radius became very small: {final_radius:.2e}"
                )

        n_true_evaluations = result.metadata.get("n_true_evaluations", 0)
        validation_results["true_function_evaluations"] = n_true_evaluations

        efficiency = n_true_evaluations / result.nit if result.nit > 0 else 0
        validation_results["validation_efficiency"] = efficiency


def _validate_multi_start_result(
    result: OptimizationResult,
    validation_results: Dict[str, Any],
    criteria: Dict[str, Any],
) -> None:
    """Validate multi-start specific results."""
    if result.metadata:
        n_successful = result.metadata.get("n_successful", 0)
        n_starts = result.metadata.get("n_starts", 1)

        success_rate = n_successful / n_starts if n_starts > 0 else 0.0
        validation_results["local_success_rate"] = success_rate
        validation_results["successful_starts"] = n_successful
        validation_results["total_starts"] = n_starts

        min_success_rate = criteria.get("min_success_rate", 0.5)
        if success_rate < min_success_rate:
            validation_results["warnings"].append(
                f"Local optimization success rate {success_rate:.2f} "
                f"below threshold {min_success_rate:.2f}"
            )

        # Analyze local results if available
        local_results = result.metadata.get("local_results", [])
        if local_results:
            successful_results = [r for r in local_results if r.success and jnp.isfinite(r.fun)]

            if len(successful_results) > 1:
                values = jnp.array([r.fun for r in successful_results])
                value_std = float(jnp.std(values))
                validation_results["result_diversity"] = value_std

                if value_std < 1e-6:
                    validation_results["warnings"].append(
                        "All local optima found are very similar - "
                        "may indicate insufficient exploration"
                    )


def analyze_optimization_efficiency(result: OptimizationResult) -> Dict[str, Any]:
    """Analyze optimization efficiency metrics.
    
    Args:
        result: Optimization result to analyze
        
    Returns:
        Efficiency analysis
    """
    analysis = {
        "iterations": result.nit,
        "function_evaluations": result.nfev,
        "success": result.success,
    }

    if result.nit > 0:
        analysis["evaluations_per_iteration"] = result.nfev / result.nit

    if result.convergence_history and len(result.convergence_history) > 1:
        history = jnp.array(result.convergence_history)

        # Improvement per iteration
        total_improvement = float(history[0] - history[-1])
        analysis["total_improvement"] = total_improvement

        if result.nit > 0:
            analysis["improvement_per_iteration"] = total_improvement / result.nit

        if result.nfev > 0:
            analysis["improvement_per_evaluation"] = total_improvement / result.nfev

        # Time to reach various improvement levels
        improvement_levels = [0.5, 0.9, 0.95, 0.99]
        target_improvements = [level * total_improvement for level in improvement_levels]

        time_to_improvement = {}
        current_improvement = 0.0

        for i, value in enumerate(history[1:], 1):
            current_improvement = float(history[0] - value)

            for level, target in zip(improvement_levels, target_improvements):
                key = f"iterations_to_{int(level*100)}pct"
                if key not in time_to_improvement and current_improvement >= target:
                    time_to_improvement[key] = i

        analysis["convergence_milestones"] = time_to_improvement

    return analysis
