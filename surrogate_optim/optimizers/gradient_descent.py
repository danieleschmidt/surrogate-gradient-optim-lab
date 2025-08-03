"""Gradient descent optimization using surrogate gradients."""

from typing import Any, Dict, List, Optional, Tuple

import jax.numpy as jnp
from jax import Array

from ..models.base import Surrogate
from .base import BaseOptimizer, OptimizationResult


class GradientDescentOptimizer(BaseOptimizer):
    """Gradient descent optimizer using surrogate gradients.
    
    Implements various gradient descent variants including standard GD,
    momentum, and adaptive learning rates.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        method: str = "momentum",
        momentum: float = 0.9,
        adaptive_lr: bool = True,
        lr_decay: float = 0.95,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        verbose: bool = False,
    ):
        """Initialize gradient descent optimizer.
        
        Args:
            learning_rate: Initial learning rate
            method: Optimization method ('gd', 'momentum', 'adam')
            momentum: Momentum coefficient for momentum methods
            adaptive_lr: Whether to use adaptive learning rate
            lr_decay: Learning rate decay factor
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            verbose: Whether to print progress
        """
        super().__init__(max_iterations, tolerance, verbose)
        
        self.learning_rate = learning_rate
        self.method = method
        self.momentum = momentum
        self.adaptive_lr = adaptive_lr
        self.lr_decay = lr_decay
        
        # Method-specific state
        self.velocity = None
        self.m = None  # First moment estimate (Adam)
        self.v = None  # Second moment estimate (Adam)
        self.beta1 = 0.9  # Adam parameters
        self.beta2 = 0.999
        self.epsilon = 1e-8
        
        # Current learning rate
        self.current_lr = learning_rate
    
    def optimize(
        self,
        surrogate: Surrogate,
        x0: Array,
        bounds: Optional[List[Tuple[float, float]]] = None,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> OptimizationResult:
        """Optimize using gradient descent with surrogate gradients.
        
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
        f_current = float(surrogate.predict(x_current))
        
        # Initialize method-specific state
        self._initialize_method_state(x0)
        
        # Initial update
        self._update_history(x_current, f_current)
        
        # Main optimization loop
        success = False
        message = "Maximum iterations reached"
        
        for iteration in range(1, self.max_iterations + 1):
            self.current_iteration = iteration
            
            # Compute gradient
            try:
                gradient = surrogate.gradient(x_current)
            except Exception as e:
                message = f"Gradient computation failed: {e}"
                break
            
            # Check gradient norm for convergence
            grad_norm = float(jnp.linalg.norm(gradient))
            if grad_norm < self.tolerance:
                success = True
                message = "Gradient norm below tolerance"
                break
            
            # Store previous values for convergence check
            x_previous = x_current.copy()
            f_previous = f_current
            
            # Update using selected method
            if self.method == "gd":
                x_current = self._gradient_descent_step(x_current, gradient)
            elif self.method == "momentum":
                x_current = self._momentum_step(x_current, gradient)
            elif self.method == "adam":
                x_current = self._adam_step(x_current, gradient)
            else:
                raise ValueError(f"Unknown optimization method: {self.method}")
            
            # Project to bounds if necessary
            x_current = self._project_to_bounds(x_current, bounds)
            
            # Evaluate function at new point
            try:
                f_current = float(surrogate.predict(x_current))
            except Exception as e:
                message = f"Function evaluation failed: {e}"
                break
            
            # Update history
            self._update_history(x_current, f_current)
            
            # Check convergence
            if self._check_convergence(x_current, x_previous, f_current, f_previous):
                success = True
                message = "Converged"
                break
            
            # Adaptive learning rate
            if self.adaptive_lr:
                if f_current > f_previous:
                    # Function increased, reduce learning rate
                    self.current_lr *= self.lr_decay
                elif iteration % 50 == 0:
                    # Periodic learning rate decay
                    self.current_lr *= self.lr_decay
        
        return OptimizationResult(
            x=x_current,
            fun=f_current,
            success=success,
            message=message,
            nit=self.current_iteration,
            nfev=self.current_iteration + 1,  # +1 for initial evaluation
            trajectory=self.trajectory.copy(),
            convergence_history=self.convergence_history.copy(),
            metadata={
                "method": self.method,
                "final_lr": self.current_lr,
                "final_grad_norm": grad_norm if 'grad_norm' in locals() else None,
            }
        )
    
    def _initialize_method_state(self, x0: Array):
        """Initialize method-specific state variables."""
        if self.method == "momentum":
            self.velocity = jnp.zeros_like(x0)
        elif self.method == "adam":
            self.m = jnp.zeros_like(x0)  # First moment
            self.v = jnp.zeros_like(x0)  # Second moment
        
        # Reset learning rate
        self.current_lr = self.learning_rate
    
    def _gradient_descent_step(self, x: Array, gradient: Array) -> Array:
        """Standard gradient descent step."""
        return x - self.current_lr * gradient
    
    def _momentum_step(self, x: Array, gradient: Array) -> Array:
        """Momentum gradient descent step."""
        self.velocity = self.momentum * self.velocity - self.current_lr * gradient
        return x + self.velocity
    
    def _adam_step(self, x: Array, gradient: Array) -> Array:
        """Adam optimization step."""
        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        
        # Update biased second raw moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient ** 2)
        
        # Compute bias-corrected first moment estimate
        m_hat = self.m / (1 - self.beta1 ** self.current_iteration)
        
        # Compute bias-corrected second raw moment estimate
        v_hat = self.v / (1 - self.beta2 ** self.current_iteration)
        
        # Update parameters
        update = self.current_lr * m_hat / (jnp.sqrt(v_hat) + self.epsilon)
        
        return x - update
    
    def set_learning_rate(self, lr: float):
        """Set the learning rate during optimization."""
        self.learning_rate = lr
        self.current_lr = lr
    
    def get_learning_rate(self) -> float:
        """Get current learning rate."""
        return self.current_lr