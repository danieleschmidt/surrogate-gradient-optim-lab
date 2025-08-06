"""Basic plotting utilities for surrogate optimization."""

from typing import Callable, List, Optional, Tuple

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import Array

from ..models.base import Surrogate


def plot_surrogate_comparison(
    true_function: Callable[[Array], float],
    surrogate: Surrogate,
    bounds: List[Tuple[float, float]],
    resolution: int = 50,
    save_path: Optional[str] = None,
    show_uncertainty: bool = True,
) -> plt.Figure:
    """Plot comparison between true function and surrogate.
    
    Args:
        true_function: The true black-box function
        surrogate: Trained surrogate model
        bounds: Bounds for plotting [x_min, x_max], [y_min, y_max]
        resolution: Grid resolution for plotting
        save_path: Optional path to save the plot
        show_uncertainty: Whether to show uncertainty bands
        
    Returns:
        matplotlib Figure object
    """
    if len(bounds) != 2:
        raise ValueError("Only 2D plotting is supported")
    
    # Create mesh grid
    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]
    
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate functions on grid
    grid_points = jnp.stack([X.flatten(), Y.flatten()], axis=1)
    
    # True function values
    true_values = jnp.array([true_function(point) for point in grid_points])
    Z_true = true_values.reshape(X.shape)
    
    # Surrogate predictions
    pred_values = surrogate.predict(grid_points)
    Z_pred = pred_values.reshape(X.shape)
    
    # Uncertainty if available
    if show_uncertainty:
        uncertainty_values = surrogate.uncertainty(grid_points)
        Z_uncertainty = uncertainty_values.reshape(X.shape)
    
    # Create subplot figure
    if show_uncertainty:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # True function
    im1 = axes[0].contourf(X, Y, Z_true, levels=20, cmap='viridis')
    axes[0].set_title('True Function')
    axes[0].set_xlabel('x1')
    axes[0].set_ylabel('x2')
    plt.colorbar(im1, ax=axes[0])
    
    # Surrogate prediction
    im2 = axes[1].contourf(X, Y, Z_pred, levels=20, cmap='viridis')
    axes[1].set_title('Surrogate Prediction')
    axes[1].set_xlabel('x1')
    axes[1].set_ylabel('x2')
    plt.colorbar(im2, ax=axes[1])
    
    # Error
    error = jnp.abs(Z_true - Z_pred)
    im3 = axes[2].contourf(X, Y, error, levels=20, cmap='Reds')
    axes[2].set_title('Absolute Error')
    axes[2].set_xlabel('x1')
    axes[2].set_ylabel('x2')
    plt.colorbar(im3, ax=axes[2])
    
    # Uncertainty if requested
    if show_uncertainty and len(axes) > 3:
        im4 = axes[3].contourf(X, Y, Z_uncertainty, levels=20, cmap='Blues')
        axes[3].set_title('Prediction Uncertainty')
        axes[3].set_xlabel('x1')
        axes[3].set_ylabel('x2')
        plt.colorbar(im4, ax=axes[3])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_optimization_trajectory(
    trajectory: List[Array],
    convergence_history: List[float],
    bounds: Optional[List[Tuple[float, float]]] = None,
    true_function: Optional[Callable[[Array], float]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot optimization trajectory and convergence.
    
    Args:
        trajectory: List of points visited during optimization
        convergence_history: List of function values during optimization
        bounds: Optional bounds for 2D trajectory plotting
        true_function: Optional true function for background contour
        save_path: Optional path to save the plot
        
    Returns:
        matplotlib Figure object
    """
    if len(trajectory) == 0:
        raise ValueError("Empty trajectory")
    
    trajectory = [jnp.asarray(point) for point in trajectory]
    
    if trajectory[0].shape[0] == 2 and bounds is not None:
        # 2D trajectory plot with convergence
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Trajectory plot
        if true_function is not None:
            # Plot function contours as background
            x_min, x_max = bounds[0]
            y_min, y_max = bounds[1]
            x = np.linspace(x_min, x_max, 50)
            y = np.linspace(y_min, y_max, 50)
            X, Y = np.meshgrid(x, y)
            
            grid_points = jnp.stack([X.flatten(), Y.flatten()], axis=1)
            Z = jnp.array([true_function(point) for point in grid_points])
            Z = Z.reshape(X.shape)
            
            ax1.contour(X, Y, Z, levels=20, colors='gray', alpha=0.3)
        
        # Plot trajectory
        traj_array = jnp.stack(trajectory)
        ax1.plot(traj_array[:, 0], traj_array[:, 1], 'ro-', markersize=6, linewidth=2)
        ax1.plot(traj_array[0, 0], traj_array[0, 1], 'go', markersize=10, label='Start')
        ax1.plot(traj_array[-1, 0], traj_array[-1, 1], 'bs', markersize=10, label='End')
        
        ax1.set_xlabel('x1')
        ax1.set_ylabel('x2')
        ax1.set_title('Optimization Trajectory')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        if bounds:
            ax1.set_xlim(bounds[0])
            ax1.set_ylim(bounds[1])
    else:
        # Only convergence plot for high-dimensional problems
        fig, ax2 = plt.subplots(1, 1, figsize=(8, 5))
    
    # Convergence plot
    ax2.plot(convergence_history, 'b-', linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Function Value')
    ax2.set_title('Convergence History')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_gradient_field(
    surrogate: Surrogate,
    bounds: List[Tuple[float, float]],
    resolution: int = 20,
    scale: float = 1.0,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot gradient field of the surrogate function.
    
    Args:
        surrogate: Trained surrogate model
        bounds: Bounds for plotting
        resolution: Grid resolution for gradient arrows
        scale: Scale factor for gradient arrows
        save_path: Optional path to save the plot
        
    Returns:
        matplotlib Figure object
    """
    if len(bounds) != 2:
        raise ValueError("Only 2D gradient fields are supported")
    
    # Create mesh grid
    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]
    
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate gradient on grid
    grid_points = jnp.stack([X.flatten(), Y.flatten()], axis=1)
    gradients = surrogate.gradient(grid_points)
    
    # Reshape gradients
    U = gradients[:, 0].reshape(X.shape)
    V = gradients[:, 1].reshape(X.shape)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot function values as background
    function_values = surrogate.predict(grid_points)
    Z = function_values.reshape(X.shape)
    im = ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
    plt.colorbar(im, ax=ax, label='Function Value')
    
    # Plot gradient field
    ax.quiver(X, Y, -U * scale, -V * scale, angles='xy', scale_units='xy', 
              color='red', alpha=0.8, width=0.003)
    
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('Surrogate Function with Negative Gradient Field')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_uncertainty_map(
    surrogate: Surrogate,
    bounds: List[Tuple[float, float]],
    resolution: int = 50,
    training_points: Optional[Array] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot uncertainty map of the surrogate model.
    
    Args:
        surrogate: Trained surrogate model
        bounds: Bounds for plotting
        resolution: Grid resolution for uncertainty map
        training_points: Optional training points to overlay
        save_path: Optional path to save the plot
        
    Returns:
        matplotlib Figure object
    """
    if len(bounds) != 2:
        raise ValueError("Only 2D uncertainty maps are supported")
    
    # Create mesh grid
    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]
    
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate uncertainty on grid
    grid_points = jnp.stack([X.flatten(), Y.flatten()], axis=1)
    uncertainty = surrogate.uncertainty(grid_points)
    Z = uncertainty.reshape(X.shape)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot uncertainty
    im = ax.contourf(X, Y, Z, levels=20, cmap='Blues')
    plt.colorbar(im, ax=ax, label='Prediction Uncertainty')
    
    # Overlay training points if provided
    if training_points is not None:
        ax.scatter(training_points[:, 0], training_points[:, 1], 
                  c='red', s=50, marker='x', label='Training Points')
        ax.legend()
    
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('Surrogate Prediction Uncertainty')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig