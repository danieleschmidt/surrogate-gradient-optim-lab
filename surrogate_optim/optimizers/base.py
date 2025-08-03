"""Base optimizer interface and optimization result structures."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax.numpy as jnp
from jax import Array

from ..models.base import Surrogate


@dataclass
class OptimizationResult:
    """Results from optimization procedure."""
    
    x_opt: Array
    f_opt: float
    n_iterations: int
    n_function_evals: int
    converged: bool
    optimization_time: float
    trajectory: Optional[List[Array]] = None
    function_values: Optional[List[float]] = None
    gradient_norms: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseOptimizer(ABC):
    """Abstract base class for optimization algorithms using surrogates."""
    
    def __init__(
        self,
        surrogate: Optional[Surrogate] = None,
        name: str = "base_optimizer"
    ):
        """Initialize optimizer.
        
        Args:
            surrogate: Surrogate model to optimize.
            name: Optimizer name.
        """
        self.surrogate = surrogate
        self.name = name
    
    @abstractmethod
    def optimize(
        self,
        x0: Array,
        bounds: Optional[List[Tuple[float, float]]] = None,
        constraints: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> OptimizationResult:
        """Optimize the surrogate function.
        
        Args:
            x0: Initial point.
            bounds: Bounds for each dimension as (min, max) tuples.
            constraints: List of constraint dictionaries.
            **kwargs: Algorithm-specific parameters.
            
        Returns:
            Optimization results.
        """
        pass
    
    def set_surrogate(self, surrogate: Surrogate) -> None:
        """Set the surrogate model to optimize.
        
        Args:
            surrogate: Surrogate model.
        """
        self.surrogate = surrogate
    
    def _check_surrogate(self) -> None:
        """Check that surrogate is set and fitted."""
        if self.surrogate is None:
            raise ValueError("Surrogate model must be set before optimization")
        if not self.surrogate.is_fitted:
            raise ValueError("Surrogate model must be fitted before optimization")
    
    def _validate_bounds(
        self, 
        x0: Array, 
        bounds: Optional[List[Tuple[float, float]]]
    ) -> Optional[List[Tuple[float, float]]]:
        """Validate and process bounds.
        
        Args:
            x0: Initial point.
            bounds: Input bounds.
            
        Returns:
            Validated bounds or None.
        """
        if bounds is None:
            return None
        
        if len(bounds) != len(x0):
            raise ValueError(f"Bounds length ({len(bounds)}) must match dimension ({len(x0)})")
        
        for i, (low, high) in enumerate(bounds):
            if low >= high:
                raise ValueError(f"Invalid bounds for dimension {i}: {low} >= {high}")
            if not (low <= x0[i] <= high):
                raise ValueError(f"Initial point x0[{i}]={x0[i]} outside bounds [{low}, {high}]")
        
        return bounds
    
    def _apply_bounds(self, x: Array, bounds: Optional[List[Tuple[float, float]]]) -> Array:
        """Apply bounds to a point by clipping.
        
        Args:
            x: Point to clip.
            bounds: Bounds as (min, max) tuples.
            
        Returns:
            Clipped point.
        """
        if bounds is None:
            return x
        
        x_clipped = x.copy()
        for i, (low, high) in enumerate(bounds):
            x_clipped = x_clipped.at[i].set(jnp.clip(x[i], low, high))
        
        return x_clipped
    
    def _line_search(
        self,
        x: Array,
        direction: Array,
        alpha_init: float = 1.0,
        c1: float = 1e-4,
        c2: float = 0.9,
        max_iterations: int = 20
    ) -> Tuple[float, int]:
        """Perform Wolfe line search.
        
        Args:
            x: Current point.
            direction: Search direction.
            alpha_init: Initial step size.
            c1: Armijo parameter.
            c2: Curvature parameter.
            max_iterations: Maximum line search iterations.
            
        Returns:
            (step_size, n_iterations).
        """
        alpha = alpha_init
        f0 = self.surrogate.predict(x)
        grad0 = self.surrogate.gradient(x)
        directional_derivative = jnp.dot(grad0, direction)
        
        # Armijo condition only (simplified line search)
        for i in range(max_iterations):
            x_new = x + alpha * direction
            f_new = self.surrogate.predict(x_new)
            
            # Armijo condition
            if f_new <= f0 + c1 * alpha * directional_derivative:
                return alpha, i + 1
            
            alpha *= 0.5
        
        return alpha, max_iterations