"""Command-line interface for surrogate optimization."""

import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from . import __version__
from .data import collect_data, Dataset
from .models import NeuralSurrogate, GPSurrogate, RandomForestSurrogate
from .optimizers import SurrogateOptimizer
from .utils import benchmark_surrogate, validate_surrogate

app = typer.Typer(help="Surrogate Gradient Optimization Lab CLI")
console = Console()

# Global CLI state
current_dataset: Optional[Dataset] = None
current_surrogate = None


@app.command()
def version():
    """Show version information."""
    console.print(f"Surrogate Gradient Optimization Lab v{__version__}")
    console.print("A toolkit for offline black-box optimization using learned gradient surrogates.")


@app.command()
def collect(
    function_file: str = typer.Argument(..., help="Python file containing function to optimize"),
    function_name: str = typer.Argument(..., help="Name of function in the file"),
    bounds: str = typer.Argument(..., help="Bounds as JSON: '[[-5,5], [-5,5]]'"),
    n_samples: int = typer.Option(100, "--samples", "-n", help="Number of samples to collect"),
    sampling: str = typer.Option("sobol", "--sampling", "-s", help="Sampling strategy"),
    output: str = typer.Option("data.npz", "--output", "-o", help="Output file"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
):
    """Collect data from a black-box function."""
    global current_dataset
    
    try:
        # Load function
        function = _load_function(function_file, function_name)
        bounds_list = json.loads(bounds)
        
        console.print(f"[bold green]Collecting {n_samples} samples using {sampling} sampling...[/bold green]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Collecting data...", total=None)
            
            current_dataset = collect_data(
                function=function,
                n_samples=n_samples,
                bounds=bounds_list,
                sampling=sampling,
                random_state=seed,
            )
            
        # Save dataset
        np.savez(output, X=current_dataset.X, y=current_dataset.y)
        
        console.print(f"[bold green]✓ Collected {current_dataset.n_samples} samples")
        console.print(f"[bold green]✓ Saved to {output}")
        
        # Show statistics
        _show_dataset_stats(current_dataset)
        
    except Exception as e:
        console.print(f"[bold red]Error: {e}")
        sys.exit(1)


@app.command()
def load(
    data_file: str = typer.Argument(..., help="Data file to load (.npz format)"),
):
    """Load a dataset from file."""
    global current_dataset
    
    try:
        data = np.load(data_file)
        current_dataset = Dataset(X=data['X'], y=data['y'])
        
        console.print(f"[bold green]✓ Loaded dataset from {data_file}")
        _show_dataset_stats(current_dataset)
        
    except Exception as e:
        console.print(f"[bold red]Error loading dataset: {e}")
        sys.exit(1)


@app.command()
def train(
    model_type: str = typer.Option("neural_network", "--model", "-m", help="Surrogate model type"),
    epochs: int = typer.Option(1000, "--epochs", "-e", help="Training epochs (for neural networks)"),
    hidden_dims: str = typer.Option("[64,64]", "--hidden", help="Hidden dimensions as JSON"),
    output: str = typer.Option("surrogate.pkl", "--output", "-o", help="Output model file"),
):
    """Train a surrogate model on the loaded dataset."""
    global current_dataset, current_surrogate
    
    if current_dataset is None:
        console.print("[bold red]Error: No dataset loaded. Use 'collect' or 'load' first.")
        sys.exit(1)
    
    try:
        hidden_dims_list = json.loads(hidden_dims) if model_type == "neural_network" else None
        
        console.print(f"[bold green]Training {model_type} surrogate model...[/bold green]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Training model...", total=None)
            
            if model_type == "neural_network":
                current_surrogate = NeuralSurrogate(
                    hidden_dims=hidden_dims_list,
                    epochs=epochs,
                )
            elif model_type == "gp":
                current_surrogate = GPSurrogate()
            elif model_type == "random_forest":
                current_surrogate = RandomForestSurrogate()
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            current_surrogate.fit(current_dataset.X, current_dataset.y)
        
        # Save model (would need pickle implementation)
        console.print(f"[bold green]✓ Trained {model_type} surrogate model")
        console.print(f"[bold green]✓ Model ready for optimization")
        
    except Exception as e:
        console.print(f"[bold red]Error training model: {e}")
        sys.exit(1)


@app.command()
def optimize(
    x0: str = typer.Argument(..., help="Initial point as JSON: '[1.0, 2.0]'"),
    bounds: Optional[str] = typer.Option(None, "--bounds", "-b", help="Bounds as JSON"),
    method: str = typer.Option("L-BFGS-B", "--method", help="Optimization method"),
    max_iter: int = typer.Option(1000, "--max-iter", help="Maximum iterations"),
):
    """Optimize using the trained surrogate model."""
    global current_surrogate
    
    if current_surrogate is None:
        console.print("[bold red]Error: No surrogate model trained. Use 'train' first.")
        sys.exit(1)
    
    try:
        x0_array = np.array(json.loads(x0))
        bounds_list = json.loads(bounds) if bounds else None
        
        console.print(f"[bold green]Optimizing using {method}...[/bold green]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Optimizing...", total=None)
            
            from .utils import optimize_with_surrogate
            result = optimize_with_surrogate(
                surrogate=current_surrogate,
                x0=x0_array,
                method=method,
                bounds=bounds_list,
                options={"maxiter": max_iter},
            )
        
        # Display results
        _show_optimization_result(result, x0_array)
        
    except Exception as e:
        console.print(f"[bold red]Error during optimization: {e}")
        sys.exit(1)


@app.command()
def benchmark(
    functions: str = typer.Option("rosenbrock,rastrigin", "--functions", "-f", help="Test functions"),
    dimensions: str = typer.Option("2,5", "--dimensions", "-d", help="Dimensions to test"),
    trials: int = typer.Option(5, "--trials", "-t", help="Number of trials"),
    samples: int = typer.Option(100, "--samples", "-s", help="Training samples per trial"),
    model_type: str = typer.Option("neural_network", "--model", "-m", help="Surrogate model type"),
    output: str = typer.Option("benchmark_results.json", "--output", "-o", help="Output file"),
):
    """Benchmark surrogate models on standard test functions."""
    try:
        function_list = functions.split(",")
        dimension_list = [int(d) for d in dimensions.split(",")]
        
        console.print(f"[bold green]Running benchmark on {len(function_list)} functions...")
        console.print(f"Dimensions: {dimension_list}")
        console.print(f"Trials per configuration: {trials}[/bold green]")
        
        # Create surrogate model
        if model_type == "neural_network":
            surrogate_model = NeuralSurrogate(epochs=500)
        elif model_type == "gp":
            surrogate_model = GPSurrogate()
        elif model_type == "random_forest":
            surrogate_model = RandomForestSurrogate()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running benchmark...", total=None)
            
            results = benchmark_surrogate(
                surrogate_model=surrogate_model,
                test_functions=function_list,
                dimensions=dimension_list,
                n_trials=trials,
                n_train_samples=samples,
            )
        
        # Save results
        with open(output, "w") as f:
            json.dump(results, f, indent=2, default=_json_serializer)
        
        console.print(f"[bold green]✓ Benchmark completed")
        console.print(f"[bold green]✓ Results saved to {output}")
        
        # Show summary
        _show_benchmark_results(results)
        
    except Exception as e:
        console.print(f"[bold red]Error during benchmark: {e}")
        sys.exit(1)


@app.command()
def validate(
    function_file: str = typer.Argument(..., help="Python file containing true function"),
    function_name: str = typer.Argument(..., help="Name of function in the file"),
    bounds: str = typer.Argument(..., help="Bounds as JSON: '[[-5,5], [-5,5]]'"),
    n_test: int = typer.Option(100, "--test-points", "-n", help="Number of test points"),
):
    """Validate trained surrogate against true function."""
    global current_surrogate
    
    if current_surrogate is None:
        console.print("[bold red]Error: No surrogate model trained. Use 'train' first.")
        sys.exit(1)
    
    try:
        function = _load_function(function_file, function_name)
        bounds_list = json.loads(bounds)
        
        console.print(f"[bold green]Validating surrogate model...[/bold green]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Validating...", total=None)
            
            results = validate_surrogate(
                surrogate=current_surrogate,
                test_function=function,
                bounds=bounds_list,
                n_test_points=n_test,
            )
        
        _show_validation_results(results)
        
    except Exception as e:
        console.print(f"[bold red]Error during validation: {e}")
        sys.exit(1)


def _load_function(function_file: str, function_name: str):
    """Load function from Python file."""
    import importlib.util
    
    spec = importlib.util.spec_from_file_location("user_module", function_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    if not hasattr(module, function_name):
        raise ValueError(f"Function '{function_name}' not found in {function_file}")
    
    return getattr(module, function_name)


def _show_dataset_stats(dataset: Dataset):
    """Display dataset statistics."""
    table = Table(title="Dataset Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Samples", str(dataset.n_samples))
    table.add_row("Dimensions", str(dataset.n_dims))
    table.add_row("Min value", f"{np.min(dataset.y):.4f}")
    table.add_row("Max value", f"{np.max(dataset.y):.4f}")
    table.add_row("Mean value", f"{np.mean(dataset.y):.4f}")
    table.add_row("Std value", f"{np.std(dataset.y):.4f}")
    
    console.print(table)


def _show_optimization_result(result: dict, x0: np.ndarray):
    """Display optimization results."""
    table = Table(title="Optimization Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Success", "✓" if result["success"] else "✗")
    table.add_row("Optimal value", f"{result['fun']:.6f}")
    table.add_row("Optimal point", str(np.round(result['x'], 4)))
    table.add_row("Initial point", str(np.round(x0, 4)))
    table.add_row("Iterations", str(result.get('nit', 'N/A')))
    table.add_row("Function evals", str(result.get('nfev', 'N/A')))
    
    console.print(table)


def _show_benchmark_results(results: dict):
    """Display benchmark results summary."""
    summary = results["summary"]
    
    panel = Panel(
        f"[bold]Mean Optimality Gap:[/bold] {summary['mean_gap']:.4f} ± {summary['std_gap']:.4f}\\n"
        f"[bold]Mean Gradient Error:[/bold] {summary['mean_grad_error']:.4f} ± {summary['std_grad_error']:.4f}\\n"
        f"[bold]Total Trials:[/bold] {summary['n_trials_total']}",
        title="[bold green]Benchmark Summary[/bold green]",
        expand=False,
    )
    console.print(panel)


def _show_validation_results(results: dict):
    """Display validation results."""
    table = Table(title="Validation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("MSE", f"{results['mse']:.6f}")
    table.add_row("MAE", f"{results['mae']:.6f}")
    table.add_row("R²", f"{results['r2']:.6f}")
    table.add_row("Mean Gradient Error", f"{results['mean_gradient_error']:.6f}")
    table.add_row("Test Points", str(results['n_test_points']))
    
    console.print(table)


def _json_serializer(obj):
    """JSON serializer for numpy objects."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.float32):
        return float(obj)
    if isinstance(obj, np.int32):
        return int(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def main():
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()