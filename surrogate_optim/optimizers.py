"""Optimization algorithms using surrogate gradients."""

from typing import Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

from .models import Surrogate

Array = Union[np.ndarray, jnp.ndarray]


class SurrogateOptimizer:
    """Main optimizer for surrogate-based optimization."""

    def __init__(
        self,
        surrogate_type: str = "neural_network",
        surrogate_kwargs: Optional[Dict] = None,
    ):
        """Initialize surrogate optimizer.
        
        Args:
            surrogate_type: Type of surrogate ("neural_network", "gp", "random_forest")
            surrogate_kwargs: Additional arguments for surrogate model
        """
        self.surrogate_type = surrogate_type
        self.surrogate_kwargs = surrogate_kwargs or {}
        self.surrogate = None

    def fit_surrogate(self, data) -> Surrogate:
        """Fit surrogate model to data.
        
        Args:
            data: Dataset with X, y attributes
            
        Returns:
            Fitted surrogate model
        """
        from .models import NeuralSurrogate, GPSurrogate, RandomForestSurrogate
        
        if self.surrogate_type == "neural_network":
            self.surrogate = NeuralSurrogate(**self.surrogate_kwargs)
        elif self.surrogate_type == "gp":
            self.surrogate = GPSurrogate(**self.surrogate_kwargs)
        elif self.surrogate_type == "random_forest":
            self.surrogate = RandomForestSurrogate(**self.surrogate_kwargs)
        else:
            raise ValueError(f"Unknown surrogate type: {self.surrogate_type}")
            
        self.surrogate.fit(data.X, data.y)
        return self.surrogate

    def optimize(
        self,
        initial_point: Array,
        method: str = "L-BFGS-B",
        bounds: Optional[List[Tuple[float, float]]] = None,
        num_steps: int = 100,
        **kwargs
    ) -> Array:
        """Optimize using surrogate gradients.
        
        Args:
            initial_point: Starting point for optimization
            method: Optimization method
            bounds: Variable bounds
            num_steps: Maximum number of iterations
            **kwargs: Additional optimizer arguments
            
        Returns:
            Optimal point
        """
        if self.surrogate is None:
            raise ValueError("No surrogate fitted. Call fit_surrogate() first.")
            
        initial_point = np.asarray(initial_point)
        
        def objective(x):
            """Objective function (negative for maximization)."""
            return -self.surrogate.predict(x)
            
        def gradient(x):
            """Gradient function."""
            return -self.surrogate.gradient(x)
            
        result = minimize(
            fun=objective,
            x0=initial_point,
            method=method,
            jac=gradient,
            bounds=bounds,
            options={"maxiter": num_steps},
            **kwargs
        )
        
        return result.x


class TrustRegionOptimizer:
    """Trust region optimization with surrogate validation."""

    def __init__(
        self,
        surrogate: Surrogate,
        true_function: Optional[Callable] = None,
        initial_radius: float = 0.5,
        max_radius: float = 2.0,
        eta: float = 0.15,
        gamma_expand: float = 2.0,
        gamma_shrink: float = 0.5,
    ):
        """Initialize trust region optimizer.
        
        Args:
            surrogate: Fitted surrogate model
            true_function: True function for validation (if available)
            initial_radius: Initial trust region radius
            max_radius: Maximum trust region radius
            eta: Acceptance threshold for ratio
            gamma_expand: Radius expansion factor
            gamma_shrink: Radius shrinkage factor
        """
        self.surrogate = surrogate
        self.true_function = true_function
        self.radius = initial_radius
        self.max_radius = max_radius
        self.eta = eta
        self.gamma_expand = gamma_expand
        self.gamma_shrink = gamma_shrink

    def optimize(
        self,
        x0: Array,
        max_iterations: int = 50,
        validate_every: int = 5,
        tolerance: float = 1e-6,
    ) -> Dict:
        """Optimize using trust region method.
        
        Args:
            x0: Starting point
            max_iterations: Maximum number of iterations
            validate_every: Validate against true function every N iterations
            tolerance: Convergence tolerance
            
        Returns:
            Dictionary with optimization results and trajectory
        """
        x0 = np.asarray(x0)
        x_current = x0.copy()
        trajectory = [x_current.copy()]
        
        for iteration in range(max_iterations):
            # Solve trust region subproblem
            x_new = self._solve_subproblem(x_current)
            
            # Compute actual and predicted reduction
            if self.true_function is not None and iteration % validate_every == 0:
                actual_reduction = self.true_function(x_current) - self.true_function(x_new)
            else:
                actual_reduction = self.surrogate.predict(x_current) - self.surrogate.predict(x_new)
                
            predicted_reduction = self.surrogate.predict(x_current) - self.surrogate.predict(x_new)
            
            # Compute ratio
            if abs(predicted_reduction) < 1e-12:
                ratio = 0
            else:
                ratio = actual_reduction / predicted_reduction
                
            # Update trust region radius
            if ratio < 0.25:
                self.radius *= self.gamma_shrink
            elif ratio > 0.75 and np.linalg.norm(x_new - x_current) >= 0.8 * self.radius:
                self.radius = min(self.gamma_expand * self.radius, self.max_radius)
                
            # Accept or reject step
            if ratio >= self.eta:
                x_current = x_new
                trajectory.append(x_current.copy())
                
            # Check convergence
            if len(trajectory) > 1:
                step_size = np.linalg.norm(trajectory[-1] - trajectory[-2])
                if step_size < tolerance:
                    break
                    
        return {
            "x": x_current,
            "trajectory": np.array(trajectory),
            "iterations": iteration + 1,
            "converged": step_size < tolerance if len(trajectory) > 1 else False,
        }

    def _solve_subproblem(self, x_center: Array) -> Array:
        """Solve trust region subproblem using gradient descent."""
        def objective(x):
            if np.linalg.norm(x - x_center) > self.radius:
                return np.inf
            return self.surrogate.predict(x)
            
        def gradient(x):
            if np.linalg.norm(x - x_center) > self.radius:
                return np.zeros_like(x)
            return self.surrogate.gradient(x)
            
        # Simple gradient step within trust region
        grad = gradient(x_center)
        step = -grad * min(self.radius / (np.linalg.norm(grad) + 1e-8), 0.1)
        x_new = x_center + step
        
        # Project back to trust region if needed
        if np.linalg.norm(x_new - x_center) > self.radius:
            direction = (x_new - x_center) / np.linalg.norm(x_new - x_center)
            x_new = x_center + self.radius * direction
            
        return x_new

    def plot_trajectory(self, trajectory: Array) -> None:
        """Plot optimization trajectory (2D only)."""
        if trajectory.shape[1] != 2:
            print("Trajectory plotting only supported for 2D problems")
            return
            
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(8, 6))
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'bo-', markersize=4)
        plt.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=8, label='Start')
        plt.plot(trajectory[-1, 0], trajectory[-1, 1], 'ro', markersize=8, label='End')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Trust Region Optimization Trajectory')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


class MultiStartOptimizer:
    """Global optimization via multiple random starts."""

    def __init__(
        self,
        surrogate: Surrogate,
        n_starts: int = 20,
        start_method: str = "sobol",
        local_optimizer: str = "L-BFGS-B",
        parallel: bool = False,
    ):
        """Initialize multi-start optimizer.
        
        Args:
            surrogate: Fitted surrogate model
            n_starts: Number of random starts
            start_method: Method for generating starts ("random", "sobol", "grid")
            local_optimizer: Local optimization method
            parallel: Whether to run starts in parallel
        """
        self.surrogate = surrogate
        self.n_starts = n_starts
        self.start_method = start_method
        self.local_optimizer = local_optimizer
        self.parallel = parallel

    def optimize_global(
        self,
        bounds: List[Tuple[float, float]],
        max_iterations: int = 100,
    ) -> Dict:
        """Find global optimum using multiple starts.
        
        Args:
            bounds: Variable bounds
            max_iterations: Maximum iterations per start
            
        Returns:
            Dictionary with global optimization results
        """
        bounds = np.array(bounds)
        n_dims = len(bounds)
        
        # Generate starting points
        starts = self._generate_starts(bounds, n_dims)
        
        results = []
        for i, start in enumerate(starts):
            try:
                result = self._optimize_single(start, bounds, max_iterations)
                results.append({
                    'start_idx': i,
                    'x0': start,
                    'x_opt': result.x,
                    'f_opt': -result.fun,  # Convert back from minimization
                    'success': result.success,
                    'nit': result.nit,
                })
            except Exception as e:
                print(f"Optimization from start {i} failed: {e}")
                continue
                
        if not results:
            raise RuntimeError("All optimization runs failed")
            
        # Find best result
        best_result = max(results, key=lambda x: x['f_opt'] if x['success'] else -np.inf)
        
        return {
            'best_point': best_result['x_opt'],
            'best_value': best_result['f_opt'],
            'all_results': results,
            'n_successful': sum(1 for r in results if r['success']),
        }

    def _generate_starts(self, bounds: Array, n_dims: int) -> Array:
        """Generate starting points."""
        if self.start_method == "random":
            starts = np.random.uniform(
                bounds[:, 0], bounds[:, 1], (self.n_starts, n_dims)
            )
        elif self.start_method == "sobol":
            # Simple quasi-random sequence (could use scipy.stats.qmc.Sobol)
            starts = np.random.uniform(
                bounds[:, 0], bounds[:, 1], (self.n_starts, n_dims)
            )
        elif self.start_method == "grid":
            # Simple grid sampling
            n_per_dim = int(np.ceil(self.n_starts ** (1.0 / n_dims)))
            coords = [np.linspace(bounds[i, 0], bounds[i, 1], n_per_dim) 
                     for i in range(n_dims)]
            grid = np.meshgrid(*coords)
            starts = np.column_stack([g.ravel() for g in grid])[:self.n_starts]
        else:
            raise ValueError(f"Unknown start method: {self.start_method}")
            
        return starts

    def _optimize_single(self, start: Array, bounds: Array, max_iterations: int):
        """Run single optimization from starting point."""
        def objective(x):
            return -self.surrogate.predict(x)  # Minimize negative for maximization
            
        def gradient(x):
            return -self.surrogate.gradient(x)
            
        result = minimize(
            fun=objective,
            x0=start,
            method=self.local_optimizer,
            jac=gradient,
            bounds=bounds,
            options={'maxiter': max_iterations},
        )
        
        return result


class GradientDescentOptimizer:
    """Basic gradient descent optimizer."""

    def __init__(
        self,
        surrogate: Surrogate,
        learning_rate: float = 0.01,
        momentum: float = 0.0,
        adaptive: bool = True,
    ):
        """Initialize gradient descent optimizer.
        
        Args:
            surrogate: Fitted surrogate model
            learning_rate: Learning rate
            momentum: Momentum coefficient
            adaptive: Whether to use adaptive learning rate
        """
        self.surrogate = surrogate
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.adaptive = adaptive

    def optimize(
        self,
        x0: Array,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> Dict:
        """Optimize using gradient descent.
        
        Args:
            x0: Starting point
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
            bounds: Variable bounds
            
        Returns:
            Dictionary with optimization results
        """
        x0 = np.asarray(x0)
        x_current = x0.copy()
        velocity = np.zeros_like(x_current)
        
        trajectory = [x_current.copy()]
        learning_rates = [self.learning_rate]
        
        for iteration in range(max_iterations):
            # Compute gradient
            grad = self.surrogate.gradient(x_current)
            
            # Update velocity with momentum
            velocity = self.momentum * velocity + self.learning_rate * grad
            
            # Update position
            x_new = x_current + velocity
            
            # Apply bounds if specified
            if bounds is not None:
                bounds_array = np.array(bounds)
                x_new = np.clip(x_new, bounds_array[:, 0], bounds_array[:, 1])
                
            # Check convergence
            step_size = np.linalg.norm(x_new - x_current)
            if step_size < tolerance:
                break
                
            # Adaptive learning rate
            if self.adaptive and iteration > 10:
                recent_improvement = (
                    self.surrogate.predict(trajectory[-5]) - 
                    self.surrogate.predict(x_current)
                ) if len(trajectory) >= 5 else 0
                
                if recent_improvement < 0:  # Getting worse
                    self.learning_rate *= 0.9
                elif recent_improvement > 0:  # Improving
                    self.learning_rate *= 1.1
                    
            x_current = x_new
            trajectory.append(x_current.copy())
            learning_rates.append(self.learning_rate)
            
        return {
            'x': x_current,
            'trajectory': np.array(trajectory),
            'learning_rates': np.array(learning_rates),
            'iterations': iteration + 1,
            'converged': step_size < tolerance,
            'final_value': self.surrogate.predict(x_current),
        }