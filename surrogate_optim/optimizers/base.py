"""Base classes for optimization algorithms."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from jax import Array

from ..models.base import Surrogate


@dataclass
class OptimizationResult:
    """Result of an optimization run.
    
    Attributes:
        x: Optimal point found
        fun: Function value at optimal point  
        success: Whether optimization was successful
        message: Status message
        nit: Number of iterations
        nfev: Number of function evaluations
        trajectory: Optimization trajectory (list of points visited)
        convergence_history: History of function values during optimization
        metadata: Additional optimization metadata
    """
    x: Array
    fun: float
    success: bool
    message: str
    nit: int
    nfev: int
    trajectory: Optional[List[Array]] = None
    convergence_history: Optional[List[float]] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseOptimizer(ABC):
    """Abstract base class for optimization algorithms."""
    
    def __init__(
        self,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        verbose: bool = False,
    ):
        """Initialize base optimizer.
        
        Args:
            max_iterations: Maximum number of optimization iterations
            tolerance: Convergence tolerance
            verbose: Whether to print optimization progress
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.verbose = verbose
        
        # Optimization state
        self.current_iteration = 0
        self.trajectory = []
        self.convergence_history = []
    
    @abstractmethod
    def optimize(
        self,
        surrogate: Surrogate,
        x0: Array,
        bounds: Optional[List[Tuple[float, float]]] = None,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> OptimizationResult:
        """Optimize the surrogate function.
        
        Args:
            surrogate: Trained surrogate model
            x0: Initial point for optimization
            bounds: Optional bounds for each dimension
            constraints: Optional constraints dictionary
            
        Returns:
            Optimization result
        """
        pass
    
    def _check_convergence(
        self,
        x_current: Array,
        x_previous: Array,
        f_current: float,
        f_previous: float,
    ) -> bool:
        """Check if optimization has converged.
        
        Args:
            x_current: Current optimization point
            x_previous: Previous optimization point  
            f_current: Current function value
            f_previous: Previous function value
            
        Returns:
            True if converged, False otherwise
        """
        # Check function value change
        f_change = abs(f_current - f_previous)
        if f_change < self.tolerance:
            return True
        
        # Check parameter change
        x_change = float(jnp.linalg.norm(x_current - x_previous))
        if x_change < self.tolerance:
            return True
        
        return False
    
    def _update_history(self, x: Array, f_val: float):
        """Update optimization history.
        
        Args:
            x: Current optimization point
            f_val: Current function value
        """
        self.trajectory.append(x.copy())
        self.convergence_history.append(f_val)
        
        if self.verbose:
            print(f"Iteration {self.current_iteration:3d}: f = {f_val:.6f}, "
                  f"||x|| = {float(jnp.linalg.norm(x)):.6f}")
    
    def _reset_state(self):
        """Reset optimizer state for new optimization run."""
        self.current_iteration = 0
        self.trajectory = []
        self.convergence_history = []
    
    def _validate_inputs(
        self,
        surrogate: Surrogate,
        x0: Array,
        bounds: Optional[List[Tuple[float, float]]],
    ):
        """Validate optimization inputs.
        
        Args:
            surrogate: Surrogate model to optimize
            x0: Initial point
            bounds: Optional bounds
            
        Raises:
            ValueError: If inputs are invalid
        """
        if not hasattr(surrogate, 'predict') or not hasattr(surrogate, 'gradient'):
            raise ValueError("Surrogate must implement predict() and gradient() methods")
        
        if x0.ndim != 1:
            raise ValueError("Initial point x0 must be 1-dimensional")
        
        if bounds is not None:
            if len(bounds) != len(x0):
                raise ValueError(f"Bounds length {len(bounds)} must match x0 length {len(x0)}")
            
            for i, (lower, upper) in enumerate(bounds):
                if lower >= upper:
                    raise ValueError(f"Invalid bounds at dimension {i}: lower={lower} >= upper={upper}")
                
                if not (lower <= x0[i] <= upper):
                    raise ValueError(f"Initial point x0[{i}]={x0[i]} violates bounds [{lower}, {upper}]")
    
    def _project_to_bounds(self, x: Array, bounds: Optional[List[Tuple[float, float]]]) -> Array:
        """Project point to feasible region defined by bounds.
        
        Args:
            x: Point to project
            bounds: Bounds for each dimension
            
        Returns:
            Projected point
        """
        if bounds is None:
            return x
        
        x_proj = x.copy()
        for i, (lower, upper) in enumerate(bounds):
            x_proj = x_proj.at[i].set(jnp.clip(x_proj[i], lower, upper))
        
        return x_proj