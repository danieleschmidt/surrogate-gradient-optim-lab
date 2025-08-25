"""Trust region optimization for surrogate models."""

from typing import Any, Callable, Dict, List, Optional, Tuple

from jax import Array
import jax.numpy as jnp

from ..models.base import Surrogate
from .base import BaseOptimizer, OptimizationResult


class TrustRegionOptimizer(BaseOptimizer):
    """Trust region optimizer with surrogate model validation.
    
    Uses trust regions to ensure reliability when optimizing with surrogate models
    by periodically validating against the true function.
    """

    def __init__(
        self,
        initial_radius: float = 1.0,
        max_radius: float = 10.0,
        min_radius: float = 1e-6,
        eta: float = 0.15,
        gamma1: float = 0.5,
        gamma2: float = 2.0,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        validate_every: int = 5,
        verbose: bool = False,
    ):
        """Initialize trust region optimizer.
        
        Args:
            initial_radius: Initial trust region radius
            max_radius: Maximum trust region radius
            min_radius: Minimum trust region radius (convergence threshold)
            eta: Acceptance threshold for ratio of actual to predicted reduction
            gamma1: Radius reduction factor for unsuccessful steps
            gamma2: Radius expansion factor for very successful steps
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            validate_every: Validate against true function every N steps (if available)
            verbose: Whether to print progress
        """
        super().__init__(max_iterations, tolerance, verbose)

        self.initial_radius = initial_radius
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.eta = eta
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.validate_every = validate_every

        # Trust region state
        self.radius = initial_radius
        self.true_function = None
        self.n_true_evaluations = 0

    def set_true_function(self, true_function: Callable[[Array], float]):
        """Set true function for validation (optional).
        
        Args:
            true_function: The true black-box function
        """
        self.true_function = true_function

    def optimize(
        self,
        surrogate: Surrogate,
        x0: Array,
        bounds: Optional[List[Tuple[float, float]]] = None,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> OptimizationResult:
        """Optimize using trust region method with surrogate model.
        
        Args:
            surrogate: Trained surrogate model
            x0: Initial optimization point
            bounds: Optional bounds for each dimension
            constraints: Optional constraints (not implemented)
            
        Returns:
            Optimization result
        """
        # Validate inputs
        self._validate_inputs(surrogate, x0, bounds)
        self._reset_state()

        # Initialize optimization variables
        x_current = x0.copy()
        self.radius = self.initial_radius
        self.n_true_evaluations = 0

        # Evaluate initial point
        f_current = float(surrogate.predict(x_current))

        # If true function available, use it for initial evaluation
        if self.true_function is not None:
            f_current_true = float(self.true_function(x_current))
            self.n_true_evaluations += 1
        else:
            f_current_true = f_current

        # Initial update
        self._update_history(x_current, f_current_true)

        # Main optimization loop
        success = False
        message = "Maximum iterations reached"

        for iteration in range(1, self.max_iterations + 1):
            self.current_iteration = iteration

            if self.verbose:
                print(f"Iteration {iteration}: radius = {self.radius:.6f}")

            # Check minimum radius for convergence
            if self.radius < self.min_radius:
                success = True
                message = "Trust region radius below minimum threshold"
                break

            # Solve trust region subproblem
            try:
                candidate, predicted_reduction = self._solve_subproblem(
                    surrogate, x_current, bounds
                )
            except Exception as e:
                message = f"Subproblem solution failed: {e}"
                break

            # Evaluate surrogate at candidate
            f_candidate_surrogate = float(surrogate.predict(candidate))

            # Determine if we should validate with true function
            should_validate = (
                self.true_function is not None and
                (iteration % self.validate_every == 0 or predicted_reduction > 0.1 * self.radius)
            )

            if should_validate:
                f_candidate_true = float(self.true_function(candidate))
                self.n_true_evaluations += 1
                actual_reduction = f_current_true - f_candidate_true
            else:
                f_candidate_true = f_candidate_surrogate
                actual_reduction = f_current - f_candidate_surrogate

            # Compute ratio of actual to predicted reduction
            if abs(predicted_reduction) < 1e-12:
                ratio = 1.0 if abs(actual_reduction) < 1e-12 else float("inf")
            else:
                ratio = actual_reduction / predicted_reduction

            if self.verbose:
                print(f"  Predicted reduction: {predicted_reduction:.6f}")
                print(f"  Actual reduction: {actual_reduction:.6f}")
                print(f"  Ratio: {ratio:.6f}")

            # Update trust region radius and accept/reject step
            if ratio < 0.25:
                # Unsuccessful step - reduce radius
                self.radius *= self.gamma1
                # Don't accept the step
                if self.verbose:
                    print(f"  Step rejected, radius reduced to {self.radius:.6f}")
            else:
                # Accept the step
                x_current = candidate
                f_current = f_candidate_surrogate
                f_current_true = f_candidate_true

                if ratio > 0.75 and jnp.linalg.norm(candidate - x_current) > 0.8 * self.radius:
                    # Very successful step - expand radius
                    self.radius = min(self.gamma2 * self.radius, self.max_radius)
                    if self.verbose:
                        print(f"  Step accepted, radius expanded to {self.radius:.6f}")
                # Moderately successful step - keep radius
                elif self.verbose:
                    print("  Step accepted, radius unchanged")

            # Project to bounds if necessary
            x_current = self._project_to_bounds(x_current, bounds)

            # Update history
            self._update_history(x_current, f_current_true)

            # Check convergence based on radius or function change
            if len(self.convergence_history) > 1:
                f_change = abs(self.convergence_history[-1] - self.convergence_history[-2])
                if f_change < self.tolerance:
                    success = True
                    message = "Function change below tolerance"
                    break

        return OptimizationResult(
            x=x_current,
            fun=f_current_true,
            success=success,
            message=message,
            nit=self.current_iteration,
            nfev=self.n_true_evaluations if self.true_function else self.current_iteration,
            trajectory=self.trajectory.copy(),
            convergence_history=self.convergence_history.copy(),
            metadata={
                "final_radius": self.radius,
                "n_true_evaluations": self.n_true_evaluations,
                "validation_frequency": self.validate_every,
            }
        )

    def _solve_subproblem(
        self,
        surrogate: Surrogate,
        x_center: Array,
        bounds: Optional[List[Tuple[float, float]]],
    ) -> Tuple[Array, float]:
        """Solve trust region subproblem.
        
        Minimizes: m(x) = f(x_center) + g^T(x - x_center) + 0.5*(x - x_center)^T*H*(x - x_center)
        Subject to: ||x - x_center|| <= radius
        
        Simplified implementation using gradient descent with projection.
        """
        # Get gradient at center
        grad = surrogate.gradient(x_center)
        f_center = float(surrogate.predict(x_center))

        # Simple Cauchy point solution: x = x_center - alpha * grad
        # where alpha is chosen to stay within trust region and bounds

        grad_norm = float(jnp.linalg.norm(grad))
        if grad_norm < 1e-12:
            # Gradient is zero, no improvement possible
            return x_center, 0.0

        # Cauchy step length
        alpha_cauchy = self.radius / grad_norm

        # Ensure we don't violate bounds
        if bounds is not None:
            for i, (lower, upper) in enumerate(bounds):
                if grad[i] > 0:  # Moving toward upper bound
                    max_alpha = (upper - x_center[i]) / grad[i]
                    alpha_cauchy = min(alpha_cauchy, max_alpha)
                elif grad[i] < 0:  # Moving toward lower bound
                    max_alpha = (lower - x_center[i]) / grad[i]
                    alpha_cauchy = min(alpha_cauchy, max_alpha)

        # Compute candidate point
        candidate = x_center - alpha_cauchy * grad

        # Project to bounds
        candidate = self._project_to_bounds(candidate, bounds)

        # Ensure within trust region
        step = candidate - x_center
        step_norm = float(jnp.linalg.norm(step))
        if step_norm > self.radius:
            candidate = x_center + (self.radius / step_norm) * step

        # Compute predicted reduction
        f_candidate = float(surrogate.predict(candidate))
        predicted_reduction = f_center - f_candidate

        return candidate, predicted_reduction

    def _reset_state(self):
        """Reset optimizer state for new optimization run."""
        super()._reset_state()
        self.radius = self.initial_radius
        self.n_true_evaluations = 0
