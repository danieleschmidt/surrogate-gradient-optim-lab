"""Command-line interface for surrogate optimization."""

import sys
from typing import List, Optional

import typer
import jax.numpy as jnp
from rich.console import Console
from rich.table import Table

from . import collect_data, NeuralSurrogate, optimize_with_surrogate

app = typer.Typer(
    name="surrogate-optim",
    help="Surrogate Gradient Optimization Lab CLI",
    add_completion=False
)
console = Console()


@app.command()
def example(
    function: str = typer.Option("rosenbrock", help="Test function (rosenbrock, sphere, rastrigin)"),
    dim: int = typer.Option(2, help="Problem dimensionality"),
    n_samples: int = typer.Option(100, help="Number of training samples"),
    method: str = typer.Option("adam", help="Optimization method"),
    verbose: bool = typer.Option(False, help="Verbose output")
) -> None:
    """Run optimization example on test function."""
    
    # Define test functions
    def rosenbrock(x):
        """Rosenbrock function."""
        x = jnp.asarray(x)
        if x.ndim == 0:
            x = jnp.array([x])
        result = 0.0
        for i in range(len(x) - 1):
            result += 100.0 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
        return result
    
    def sphere(x):
        """Sphere function."""
        x = jnp.asarray(x)
        return jnp.sum(x**2)
    
    def rastrigin(x):
        """Rastrigin function."""
        x = jnp.asarray(x)
        A = 10.0
        n = len(x)
        return A * n + jnp.sum(x**2 - A * jnp.cos(2 * jnp.pi * x))
    
    functions = {
        "rosenbrock": rosenbrock,
        "sphere": sphere, 
        "rastrigin": rastrigin
    }
    
    if function not in functions:
        console.print(f"[red]Unknown function: {function}[/red]")
        console.print(f"Available functions: {list(functions.keys())}")
        sys.exit(1)
    
    test_function = functions[function]
    bounds = [(-5.0, 5.0)] * dim
    
    console.print(f"[bold blue]Surrogate Optimization Example[/bold blue]")
    console.print(f"Function: {function}")
    console.print(f"Dimensions: {dim}")
    console.print(f"Training samples: {n_samples}")
    console.print(f"Method: {method}")
    console.print()
    
    # Collect training data
    console.print("[yellow]Collecting training data...[/yellow]")
    try:
        dataset = collect_data(
            function=test_function,
            n_samples=n_samples,
            bounds=bounds,
            sampling="sobol"
        )
        console.print(f"✓ Collected {dataset.n_samples} samples")
    except Exception as e:
        console.print(f"[red]Error collecting data: {e}[/red]")
        sys.exit(1)
    
    # Train surrogate model
    console.print("[yellow]Training surrogate model...[/yellow]")
    try:
        surrogate = NeuralSurrogate(
            hidden_dims=[64, 64],
            learning_rate=0.01,
            normalize_inputs=True
        )
        
        result = surrogate.fit(dataset.X, dataset.y, verbose=verbose)
        console.print(f"✓ Training completed in {result.training_time:.2f}s")
        console.print(f"  Final loss: {result.training_loss:.6f}")
    except Exception as e:
        console.print(f"[red]Error training surrogate: {e}[/red]")
        sys.exit(1)
    
    # Optimize using surrogate
    console.print("[yellow]Optimizing with surrogate gradients...[/yellow]")
    try:
        # Random starting point
        x0 = jnp.array([2.0] * dim)
        
        opt_result = optimize_with_surrogate(
            surrogate=surrogate,
            x0=x0,
            method="L-BFGS-B" if method == "lbfgs" else method,
            bounds=bounds,
            use_jax=(method in ["adam", "sgd", "momentum"])
        )
        
        console.print(f"✓ Optimization completed")
        console.print(f"  Iterations: {opt_result.n_iterations}")
        console.print(f"  Function evaluations: {opt_result.n_function_evals}")
        console.print(f"  Converged: {opt_result.converged}")
        console.print(f"  Time: {opt_result.optimization_time:.2f}s")
    except Exception as e:
        console.print(f"[red]Error during optimization: {e}[/red]")
        sys.exit(1)
    
    # Display results
    console.print()
    console.print("[bold green]Results[/bold green]")
    
    table = Table()
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Optimal Point", str(opt_result.x_opt))
    table.add_row("Optimal Value", f"{opt_result.f_opt:.6f}")
    table.add_row("True Function Value", f"{test_function(opt_result.x_opt):.6f}")
    
    # Compute error
    true_value = test_function(opt_result.x_opt)
    error = abs(opt_result.f_opt - true_value)
    table.add_row("Surrogate Error", f"{error:.6f}")
    
    console.print(table)
    
    # Known optima for comparison
    if function == "sphere":
        true_optimum = 0.0
        true_point = jnp.zeros(dim)
        distance_to_optimum = jnp.linalg.norm(opt_result.x_opt - true_point)
        console.print(f"\n[dim]Known optimum: {true_optimum} at {true_point}[/dim]")
        console.print(f"[dim]Distance to true optimum: {distance_to_optimum:.6f}[/dim]")


@app.command()
def benchmark(
    functions: List[str] = typer.Option(["sphere", "rosenbrock"], help="Test functions"),
    dims: List[int] = typer.Option([2, 5], help="Problem dimensions"), 
    n_samples: int = typer.Option(200, help="Training samples"),
    n_trials: int = typer.Option(5, help="Number of trials per setting"),
) -> None:
    """Run benchmark on multiple test functions."""
    console.print("[bold blue]Surrogate Optimization Benchmark[/bold blue]")
    console.print()
    
    # TODO: Implement comprehensive benchmarking
    console.print("[yellow]Benchmark feature coming soon![/yellow]")
    console.print("This will include:")
    console.print("• Multiple test functions")
    console.print("• Performance metrics")
    console.print("• Comparison with baseline methods")
    console.print("• Statistical analysis")


@app.command()
def info() -> None:
    """Display package information."""
    from . import __version__, __author__
    
    console.print(f"[bold blue]Surrogate Gradient Optimization Lab[/bold blue]")
    console.print(f"Version: {__version__}")
    console.print(f"Author: {__author__}")
    console.print()
    console.print("A toolkit for offline black-box optimization using learned gradient surrogates.")
    console.print()
    console.print("[bold]Key Features:[/bold]")
    console.print("• Multiple surrogate model types (Neural Networks, GPs, Random Forests)")
    console.print("• Gradient-based optimization with automatic differentiation")
    console.print("• Trust region and multi-start global optimization")
    console.print("• Uncertainty quantification and active learning")
    console.print("• Comprehensive visualization and benchmarking")


def main() -> None:
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()