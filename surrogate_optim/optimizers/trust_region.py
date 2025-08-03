"""Trust region optimizer with surrogate validation."""

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax.numpy as jnp
from jax import Array

from .base import BaseOptimizer, OptimizationResult


class TrustRegionOptimizer(BaseOptimizer):
    """Trust region optimizer with optional true function validation.
    
    Uses trust regions to ensure reliable optimization steps, with optional
    validation against the true black-box function to maintain trust in
    the surrogate model.
    """
    
    def __init__(
        self,
        surrogate=None,
        true_function: Optional[Callable[[Array], float]] = None,
        initial_radius: float = 0.5,
        max_radius: float = 2.0,
        eta: float = 0.15,
        gamma1: float = 0.25,
        gamma2: float = 2.0,
        subproblem_solver: str = "cauchy_point",
        validation_frequency: int = 5,
        name: str = "trust_region"
    ):
        """Initialize trust region optimizer.
        
        Args:
            surrogate: Surrogate model to optimize.
            true_function: Optional true function for validation.
            initial_radius: Initial trust region radius.
            max_radius: Maximum trust region radius.
            eta: Acceptance threshold for trust region ratio.
            gamma1: Radius reduction factor.
            gamma2: Radius expansion factor.
            subproblem_solver: Subproblem solver ("cauchy_point", "dogleg").
            validation_frequency: How often to validate against true function.
            name: Optimizer name.
        """
        super().__init__(surrogate, name)
        self.true_function = true_function
        self.initial_radius = initial_radius
        self.max_radius = max_radius
        self.eta = eta
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.subproblem_solver = subproblem_solver
        self.validation_frequency = validation_frequency
    
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
        """Optimize using trust region method.
        
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
        
        # Initialize trust region
        x = x0.copy()
        radius = self.initial_radius
        trajectory = [x]
        function_values = [float(self.surrogate.predict(x))]
        gradient_norms = []
        trust_region_radii = [radius]
        
        n_function_evals = 1
        n_true_function_evals = 0
        best_f = function_values[0]
        patience_counter = 0
        
        for iteration in range(max_iterations):
            # Compute gradient and Hessian approximation
            grad = self.surrogate.gradient(x)
            grad_norm = float(jnp.linalg.norm(grad))
            gradient_norms.append(grad_norm)
            
            # Check convergence
            if grad_norm < tolerance:
                if verbose:
                    print(f"Converged at iteration {iteration} with gradient norm {grad_norm:.2e}")
                break
            
            # Solve trust region subproblem
            if self.subproblem_solver == "cauchy_point":
                step = self._cauchy_point(x, grad, radius, bounds)
            elif self.subproblem_solver == "dogleg":
                step = self._dogleg(x, grad, radius, bounds)
            else:
                raise ValueError(f"Unknown subproblem solver: {self.subproblem_solver}")
            
            x_new = x + step
            x_new = self._apply_bounds(x_new, bounds)
            
            # Compute predicted vs actual reduction
            f_current = self.surrogate.predict(x)
            f_predicted = self.surrogate.predict(x_new)
            predicted_reduction = f_current - f_predicted
            
            # Use true function for validation if available and due
            if (self.true_function is not None and 
                iteration % self.validation_frequency == 0):
                f_true_current = self.true_function(x)
                f_true_new = self.true_function(x_new)
                actual_reduction = f_true_current - f_true_new
                n_true_function_evals += 2
                
                # Compare true vs surrogate accuracy
                surrogate_error = abs(f_current - f_true_current)
                if verbose and surrogate_error > 0.1 * abs(f_true_current):
                    print(f"Warning: Large surrogate error at iteration {iteration}: {surrogate_error:.6f}")
            else:
                # Use surrogate for actual reduction
                actual_reduction = predicted_reduction
            
            # Compute trust region ratio
            if abs(predicted_reduction) < 1e-12:
                rho = 1.0  # Avoid division by zero
            else:
                rho = actual_reduction / predicted_reduction
            
            # Update trust region radius and position
            if rho < self.eta:
                # Reject step and shrink radius
                radius = self.gamma1 * radius
                x_new = x  # Stay at current position
                if verbose:
                    print(f"Iteration {iteration}: Step rejected, radius={radius:.6f}")
            else:
                # Accept step
                x = x_new
                if rho > 0.75:
                    # Expand radius for good steps
                    radius = min(self.gamma2 * radius, self.max_radius)
                # Keep radius unchanged for moderately good steps
                
                if verbose:
                    print(f"Iteration {iteration}: Step accepted, radius={radius:.6f}")
            
            # Update tracking
            f_new = float(self.surrogate.predict(x))
            n_function_evals += 1
            
            trajectory.append(x)
            function_values.append(f_new)
            trust_region_radii.append(radius)
            
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
            
            # Check if radius became too small
            if radius < tolerance:
                if verbose:
                    print(f"Trust region radius too small: {radius:.2e}")
                break
            
            if verbose and iteration % 100 == 0:
                print(f"Iteration {iteration}: f={f_new:.6f}, grad_norm={grad_norm:.2e}, radius={radius:.6f}")
        
        optimization_time = time.time() - start_time
        
        return OptimizationResult(
            x_opt=x,
            f_opt=float(self.surrogate.predict(x)),
            n_iterations=iteration + 1,
            n_function_evals=n_function_evals,
            converged=grad_norm < tolerance or radius < tolerance,
            optimization_time=optimization_time,
            trajectory=trajectory,
            function_values=function_values,
            gradient_norms=gradient_norms,
            metadata={
                "trust_region_radii": trust_region_radii,
                "n_true_function_evals": n_true_function_evals,
                "final_radius": radius,
                "subproblem_solver": self.subproblem_solver,
            }
        )
    
    def _cauchy_point(
        self,
        x: Array,
        grad: Array,
        radius: float,
        bounds: Optional[List[Tuple[float, float]]] = None
    ) -> Array:
        """Compute Cauchy point for trust region subproblem.
        
        Args:
            x: Current point.
            grad: Current gradient.
            radius: Trust region radius.
            bounds: Optional bounds.
            
        Returns:
            Cauchy point step.
        """
        # Steepest descent direction
        p = -grad / (jnp.linalg.norm(grad) + 1e-12)
        
        # Find step length that minimizes quadratic model
        # For now, use simple approach: step to trust region boundary
        step_length = radius
        
        # Apply bounds constraints if given
        if bounds is not None:
            for i, (low, high) in enumerate(bounds):
                if p[i] > 0:
                    max_step = (high - x[i]) / p[i]
                elif p[i] < 0:
                    max_step = (low - x[i]) / p[i]
                else:
                    max_step = float('inf')
                
                step_length = min(step_length, max_step)
        
        return step_length * p
    
    def _dogleg(
        self,
        x: Array,
        grad: Array,
        radius: float,
        bounds: Optional[List[Tuple[float, float]]] = None
    ) -> Array:
        """Compute dogleg step for trust region subproblem.
        
        Args:
            x: Current point.
            grad: Current gradient.
            radius: Trust region radius.
            bounds: Optional bounds.
            
        Returns:
            Dogleg step.
        """
        # For simplicity, use Cauchy point (could be extended to full dogleg)
        return self._cauchy_point(x, grad, radius, bounds)
    
    def plot_trajectory(self, result: OptimizationResult) -> None:
        """Plot optimization trajectory (requires matplotlib).
        
        Args:
            result: Optimization result with trajectory.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib required for plotting")
            return
        
        if result.trajectory is None:
            print("No trajectory data available")
            return
        
        trajectory = jnp.stack(result.trajectory)
        
        if trajectory.shape[1] == 2:
            # 2D trajectory plot
            plt.figure(figsize=(10, 8))
            plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-o', markersize=3)
            plt.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=8, label='Start')
            plt.plot(trajectory[-1, 0], trajectory[-1, 1], 'ro', markersize=8, label='End')
            plt.xlabel('x1')
            plt.ylabel('x2')
            plt.title('Trust Region Optimization Trajectory')
            plt.legend()
            plt.grid(True)
            plt.show()
        else:
            # Multi-dimensional: plot function value and radius over iterations
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
            
            # Function values
            ax1.plot(result.function_values)
            ax1.set_ylabel('Function Value')
            ax1.set_title('Trust Region Optimization Progress')
            ax1.grid(True)
            
            # Gradient norms
            if result.gradient_norms:
                ax2.semilogy(result.gradient_norms)
                ax2.set_ylabel('Gradient Norm')
                ax2.grid(True)
            
            # Trust region radii
            if 'trust_region_radii' in result.metadata:
                ax3.plot(result.metadata['trust_region_radii'])
                ax3.set_ylabel('Trust Region Radius')
                ax3.set_xlabel('Iteration')
                ax3.grid(True)
            
            plt.tight_layout()
            plt.show()