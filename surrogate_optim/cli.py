"""Command-line interface for surrogate optimization."""

import sys
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Surrogate Gradient Optimization Lab CLI")
console = Console()


@app.command()
def version():
    """Show version information."""
    from . import __version__
    console.print(f"Surrogate Gradient Optimization Lab v{__version__}")


@app.command()
def info():
    """Show package information."""
    from . import __version__, __author__, __email__
    
    table = Table(title="Package Information")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Version", __version__)
    table.add_row("Author", __author__)
    table.add_row("Email", __email__)
    table.add_row("Python", sys.version.split()[0])
    
    console.print(table)


@app.command()
def example():
    """Run a simple optimization example."""
    import jax.numpy as jnp
    from . import quick_optimize
    
    console.print("🚀 Running surrogate optimization example...")
    
    # Define a simple test function (2D Rosenbrock)
    def rosenbrock(x):
        """2D Rosenbrock function: f(x,y) = (1-x)^2 + 100*(y-x^2)^2"""
        x, y = x[0], x[1]
        return (1 - x)**2 + 100 * (y - x**2)**2
    
    # Define bounds
    bounds = [(-2.0, 2.0), (-1.0, 3.0)]
    
    # Run optimization
    console.print("Optimizing 2D Rosenbrock function...")
    console.print(f"Bounds: {bounds}")
    
    try:
        result = quick_optimize(
            function=rosenbrock,
            bounds=bounds,
            n_samples=50,
            surrogate_type="neural_network",
            verbose=False,
        )
        
        # Display results
        table = Table(title="Optimization Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Success", str(result.success))
        table.add_row("Optimal Point", f"[{result.x[0]:.6f}, {result.x[1]:.6f}]")
        table.add_row("Optimal Value", f"{result.fun:.6f}")
        table.add_row("Iterations", str(result.nit))
        table.add_row("Function Evaluations", str(result.nfev))
        table.add_row("Message", result.message)
        
        console.print(table)
        
        # Known optimum is at (1, 1) with value 0
        true_opt = jnp.array([1.0, 1.0])
        error = float(jnp.linalg.norm(result.x - true_opt))
        console.print(f"\n📊 Distance to true optimum (1, 1): {error:.6f}")
        
        if error < 0.1:
            console.print("✅ Excellent optimization result!")
        elif error < 0.5:
            console.print("✅ Good optimization result!")
        else:
            console.print("⚠️  Could be improved - try more samples or different surrogate")
            
    except Exception as e:
        console.print(f"❌ Optimization failed: {e}")


@app.command()
def benchmark():
    """Run benchmarks on standard test functions."""
    console.print("🏃 Running benchmarks on standard optimization problems...")
    console.print("(This is a placeholder - full benchmarks to be implemented)")


def main():
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()