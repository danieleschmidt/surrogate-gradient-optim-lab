"""Interactive visualization tools for surrogate optimization."""

from typing import Callable, List, Optional, Tuple

try:
    import plotly.graph_objects as go
    from plotly.offline import plot
    import plotly.subplots as sp
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from jax import Array
import jax.numpy as jnp

from ..models.base import Surrogate


def create_interactive_dashboard(
    surrogate: Surrogate,
    bounds: List[Tuple[float, float]],
    true_function: Optional[Callable[[Array], float]] = None,
    training_data: Optional[Array] = None,
    save_html: Optional[str] = None,
) -> Optional[go.Figure]:
    """Create interactive dashboard for surrogate analysis.
    
    Args:
        surrogate: Trained surrogate model
        bounds: Bounds for visualization
        true_function: Optional true function for comparison
        training_data: Optional training points to display
        save_html: Optional path to save HTML file
        
    Returns:
        Plotly figure object if available, None otherwise
    """
    if not PLOTLY_AVAILABLE:
        print("Warning: Plotly not available. Install with: pip install plotly")
        return None

    if len(bounds) != 2:
        print("Warning: Interactive dashboard only supports 2D visualization")
        return None

    # Create mesh grid for evaluation
    resolution = 50
    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]

    x = jnp.linspace(x_min, x_max, resolution)
    y = jnp.linspace(y_min, y_max, resolution)
    X, Y = jnp.meshgrid(x, y)

    grid_points = jnp.stack([X.flatten(), Y.flatten()], axis=1)

    # Evaluate surrogate
    pred_values = surrogate.predict(grid_points)
    Z_pred = pred_values.reshape(X.shape)

    uncertainty_values = surrogate.uncertainty(grid_points)
    Z_uncertainty = uncertainty_values.reshape(X.shape)

    # Create subplots
    n_plots = 3 if true_function is None else 4
    subplot_titles = ["Surrogate Prediction", "Prediction Uncertainty", "Gradient Field"]
    if true_function is not None:
        subplot_titles.insert(0, "True Function")

    fig = sp.make_subplots(
        rows=2, cols=2,
        subplot_titles=subplot_titles,
        specs=[[{"type": "scene"}, {"type": "scene"}],
               [{"type": "scene"}, {"type": "scene"}]]
    )

    plot_idx = 1

    # True function surface (if available)
    if true_function is not None:
        true_values = jnp.array([true_function(point) for point in grid_points])
        Z_true = true_values.reshape(X.shape)

        fig.add_trace(
            go.Surface(x=X, y=Y, z=Z_true, colorscale="Viridis", name="True Function"),
            row=(plot_idx - 1) // 2 + 1, col=(plot_idx - 1) % 2 + 1
        )
        plot_idx += 1

    # Surrogate prediction surface
    fig.add_trace(
        go.Surface(x=X, y=Y, z=Z_pred, colorscale="Viridis", name="Surrogate"),
        row=(plot_idx - 1) // 2 + 1, col=(plot_idx - 1) % 2 + 1
    )
    plot_idx += 1

    # Uncertainty surface
    fig.add_trace(
        go.Surface(x=X, y=Y, z=Z_uncertainty, colorscale="Blues", name="Uncertainty"),
        row=(plot_idx - 1) // 2 + 1, col=(plot_idx - 1) % 2 + 1
    )
    plot_idx += 1

    # Gradient field (as contour with arrows)
    gradients = surrogate.gradient(grid_points)
    U = gradients[:, 0].reshape(X.shape)
    V = gradients[:, 1].reshape(X.shape)

    # Sample points for arrow display (reduce density for clarity)
    step = max(1, resolution // 10)
    X_arrows = X[::step, ::step]
    Y_arrows = Y[::step, ::step]
    U_arrows = U[::step, ::step]
    V_arrows = V[::step, ::step]

    # Add contour plot
    fig.add_trace(
        go.Contour(x=x, y=y, z=Z_pred, colorscale="Viridis", showscale=False),
        row=(plot_idx - 1) // 2 + 1, col=(plot_idx - 1) % 2 + 1
    )

    # Add gradient arrows (simplified - Plotly doesn't have built-in quiver)
    for i in range(0, X_arrows.shape[0], 2):
        for j in range(0, X_arrows.shape[1], 2):
            x_start = X_arrows[i, j]
            y_start = Y_arrows[i, j]
            dx = -U_arrows[i, j] * 0.1  # Negative for descent direction
            dy = -V_arrows[i, j] * 0.1

            fig.add_trace(
                go.Scatter(
                    x=[x_start, x_start + dx],
                    y=[y_start, y_start + dy],
                    mode="lines",
                    line=dict(color="red", width=2),
                    showlegend=False
                ),
                row=(plot_idx - 1) // 2 + 1, col=(plot_idx - 1) % 2 + 1
            )

    # Add training points if provided
    if training_data is not None:
        for i in range(min(n_plots, 4)):
            row = i // 2 + 1
            col = i % 2 + 1

            fig.add_trace(
                go.Scatter3d(
                    x=training_data[:, 0],
                    y=training_data[:, 1],
                    z=jnp.zeros(len(training_data)),  # Project to base
                    mode="markers",
                    marker=dict(color="red", size=5),
                    name="Training Points",
                    showlegend=(i == 0)
                ),
                row=row, col=col
            )

    # Update layout
    fig.update_layout(
        title="Surrogate Optimization Interactive Dashboard",
        height=800,
        showlegend=True
    )

    if save_html:
        plot(fig, filename=save_html, auto_open=False)
        print(f"Interactive dashboard saved to {save_html}")

    return fig


def plot_interactive_comparison(
    true_function: Callable[[Array], float],
    surrogate: Surrogate,
    bounds: List[Tuple[float, float]],
    training_points: Optional[Array] = None,
    save_html: Optional[str] = None,
) -> Optional[go.Figure]:
    """Create interactive comparison between true and surrogate functions.
    
    Args:
        true_function: The true black-box function
        surrogate: Trained surrogate model  
        bounds: Bounds for visualization
        training_points: Optional training points to display
        save_html: Optional path to save HTML file
        
    Returns:
        Plotly figure object if available, None otherwise
    """
    if not PLOTLY_AVAILABLE:
        print("Warning: Plotly not available. Install with: pip install plotly")
        return None

    if len(bounds) != 2:
        print("Warning: Interactive comparison only supports 2D visualization")
        return None

    # Create mesh grid
    resolution = 50
    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]

    x = jnp.linspace(x_min, x_max, resolution)
    y = jnp.linspace(y_min, y_max, resolution)
    X, Y = jnp.meshgrid(x, y)

    grid_points = jnp.stack([X.flatten(), Y.flatten()], axis=1)

    # Evaluate functions
    true_values = jnp.array([true_function(point) for point in grid_points])
    Z_true = true_values.reshape(X.shape)

    pred_values = surrogate.predict(grid_points)
    Z_pred = pred_values.reshape(X.shape)

    error = jnp.abs(Z_true - Z_pred)
    Z_error = error.reshape(X.shape)

    # Create subplots
    fig = sp.make_subplots(
        rows=1, cols=3,
        subplot_titles=["True Function", "Surrogate Prediction", "Absolute Error"],
        specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]]
    )

    # True function
    fig.add_trace(
        go.Surface(x=X, y=Y, z=Z_true, colorscale="Viridis", name="True"),
        row=1, col=1
    )

    # Surrogate prediction
    fig.add_trace(
        go.Surface(x=X, y=Y, z=Z_pred, colorscale="Viridis", name="Surrogate"),
        row=1, col=2
    )

    # Error
    fig.add_trace(
        go.Surface(x=X, y=Y, z=Z_error, colorscale="Reds", name="Error"),
        row=1, col=3
    )

    # Add training points if provided
    if training_points is not None:
        training_z = jnp.array([true_function(point) for point in training_points])

        for col in range(1, 4):
            fig.add_trace(
                go.Scatter3d(
                    x=training_points[:, 0],
                    y=training_points[:, 1],
                    z=training_z,
                    mode="markers",
                    marker=dict(color="black", size=5),
                    name="Training Points",
                    showlegend=(col == 1)
                ),
                row=1, col=col
            )

    # Update layout
    fig.update_layout(
        title="Interactive Function Comparison",
        height=500,
        showlegend=True
    )

    if save_html:
        plot(fig, filename=save_html, auto_open=False)
        print(f"Interactive comparison saved to {save_html}")

    return fig


def plot_optimization_animation(
    trajectory: List[Array],
    convergence_history: List[float],
    bounds: List[Tuple[float, float]],
    surrogate: Optional[Surrogate] = None,
    save_html: Optional[str] = None,
) -> Optional[go.Figure]:
    """Create animated visualization of optimization trajectory.
    
    Args:
        trajectory: List of points visited during optimization
        convergence_history: List of function values
        bounds: Bounds for visualization
        surrogate: Optional surrogate for background surface
        save_html: Optional path to save HTML file
        
    Returns:
        Plotly figure object if available, None otherwise
    """
    if not PLOTLY_AVAILABLE:
        print("Warning: Plotly not available. Install with: pip install plotly")
        return None

    if len(bounds) != 2 or len(trajectory) == 0:
        print("Warning: Animation only supports 2D trajectories")
        return None

    # Convert trajectory to array
    trajectory_array = jnp.stack([jnp.asarray(point) for point in trajectory])

    # Create figure
    fig = go.Figure()

    # Add background surface if surrogate provided
    if surrogate is not None:
        resolution = 30
        x_min, x_max = bounds[0]
        y_min, y_max = bounds[1]

        x = jnp.linspace(x_min, x_max, resolution)
        y = jnp.linspace(y_min, y_max, resolution)
        X, Y = jnp.meshgrid(x, y)

        grid_points = jnp.stack([X.flatten(), Y.flatten()], axis=1)
        Z_values = surrogate.predict(grid_points)
        Z = Z_values.reshape(X.shape)

        fig.add_trace(
            go.Surface(
                x=X, y=Y, z=Z,
                colorscale="Viridis",
                opacity=0.7,
                name="Surrogate"
            )
        )

    # Add trajectory points as animated scatter
    fig.add_trace(
        go.Scatter3d(
            x=trajectory_array[:1, 0],
            y=trajectory_array[:1, 1],
            z=convergence_history[:1],
            mode="markers+lines",
            marker=dict(color="red", size=8),
            line=dict(color="red", width=4),
            name="Optimization Path"
        )
    )

    # Create animation frames
    frames = []
    for i in range(1, len(trajectory) + 1):
        frame = go.Frame(
            data=[
                go.Scatter3d(
                    x=trajectory_array[:i, 0],
                    y=trajectory_array[:i, 1],
                    z=convergence_history[:i],
                    mode="markers+lines",
                    marker=dict(color="red", size=8),
                    line=dict(color="red", width=4),
                    name="Optimization Path"
                )
            ],
            name=str(i)
        )
        frames.append(frame)

    fig.frames = frames

    # Add play button
    fig.update_layout(
        title="Animated Optimization Trajectory",
        scene=dict(
            xaxis_title="x1",
            yaxis_title="x2",
            zaxis_title="Function Value"
        ),
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "buttons": [{
                "label": "Play",
                "method": "animate",
                "args": [None, {"frame": {"duration": 500}}]
            }]
        }]
    )

    if save_html:
        plot(fig, filename=save_html, auto_open=False)
        print(f"Animated trajectory saved to {save_html}")

    return fig
