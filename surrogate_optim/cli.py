"""Command-line interface for surrogate optimization."""

import sys
from typing import Optional

from rich.console import Console
from rich.table import Table
import typer

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
    from . import __author__, __email__, __version__

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
def benchmark(
    functions: Optional[str] = typer.Option(None, "--functions", "-f", help="Comma-separated list of functions to benchmark"),
    quick: bool = typer.Option(False, "--quick", "-q", help="Run quick benchmark (fewer trials)"),
    output_dir: str = typer.Option("benchmark_results", "--output", "-o", help="Output directory for results"),
    verbose: bool = typer.Option(True, "--verbose/--quiet", "-v/-q", help="Show detailed progress"),
    surrogates: Optional[str] = typer.Option(None, "--surrogates", "-s", help="Comma-separated list of surrogate types"),
    optimizers: Optional[str] = typer.Option(None, "--optimizers", help="Comma-separated list of optimizer types"),
    samples: Optional[str] = typer.Option(None, "--samples", help="Comma-separated list of sample sizes"),
):
    """Run benchmarks on standard test functions."""
    import sys

    from .benchmarks import SurrogateBenchmarkSuite
    sys.path.append("/root/repo")
    from tests.fixtures.benchmark_functions import benchmark_functions

    console.print("🏃 Running surrogate optimization benchmarks...")

    # Parse configuration
    config = {}

    if surrogates:
        config["surrogate_types"] = [s.strip() for s in surrogates.split(",")]

    if optimizers:
        config["optimizer_types"] = [o.strip() for o in optimizers.split(",")]

    if samples:
        config["training_sample_sizes"] = [int(s.strip()) for s in samples.split(",")]

    # Parse function list
    function_names = None
    if functions:
        function_names = [f.strip() for f in functions.split(",")]
        # Validate function names
        invalid_functions = [f for f in function_names if f not in benchmark_functions]
        if invalid_functions:
            console.print(f"❌ Unknown functions: {invalid_functions}")
            console.print(f"Available functions: {list(benchmark_functions.keys())}")
            return

    try:
        # Create benchmark suite
        suite = SurrogateBenchmarkSuite(
            output_dir=output_dir,
            save_results=True,
            verbose=verbose
        )

        # Run appropriate benchmark
        if quick:
            console.print("Running quick benchmark (2-3 functions, 2 trials)...")
            results = suite.run_quick_benchmark()
        else:
            if function_names is None:
                console.print("Running full benchmark suite on 2D functions...")
            else:
                console.print(f"Running benchmark on {len(function_names)} functions...")
            results = suite.run_suite_benchmark(function_names, config)

        # Display summary
        console.print(f"\n✅ Benchmark complete! Results saved to {output_dir}")

        if results.summary_stats:
            console.print("\n📊 Summary Results:")

            table = Table()
            table.add_column("Method", style="cyan")
            table.add_column("Success Rate", style="green")
            table.add_column("Avg Error", style="yellow")
            table.add_column("Avg Time (s)", style="blue")
            table.add_column("Experiments", style="magenta")

            for method, stats in results.summary_stats.items():
                table.add_row(
                    method,
                    f"{stats['success_rate']:.1%}",
                    f"{stats['average_error']:.2e}",
                    f"{stats['average_time']:.2f}",
                    str(stats["num_experiments"])
                )

            console.print(table)

            # Show rankings
            if results.rankings:
                console.print("\n🏆 Method Rankings:")
                for method, rank in sorted(results.rankings.items(), key=lambda x: x[1]):
                    console.print(f"  {rank}. {method}")

    except Exception as e:
        console.print(f"❌ Benchmark failed: {e}")
        if verbose:
            import traceback
            console.print(traceback.format_exc())


@app.command()
def list_functions():
    """List all available benchmark functions."""
    import sys
    sys.path.append("/root/repo")
    from tests.fixtures.benchmark_functions import (
        benchmark_functions,
        get_2d_functions,
        get_multimodal_functions,
    )

    console.print("📋 Available Benchmark Functions:")

    # Show 2D functions
    two_d_funcs = get_2d_functions()
    if two_d_funcs:
        console.print("\n2D Functions (suitable for visualization):")
        for name, func in two_d_funcs.items():
            console.print(f"  • {name}: {func.name}")

    # Show multimodal functions
    multimodal_funcs = get_multimodal_functions()
    if multimodal_funcs:
        console.print("\nMultimodal Functions (multiple local optima):")
        for name, func in multimodal_funcs.items():
            if name not in two_d_funcs:  # Avoid duplicates
                console.print(f"  • {name}: {func.name} ({func.n_dims}D)")

    # Show all functions with details
    console.print(f"\nAll Functions ({len(benchmark_functions)} total):")

    table = Table()
    table.add_column("Name", style="cyan")
    table.add_column("Function", style="green")
    table.add_column("Dimensions", style="yellow")
    table.add_column("Global Optimum", style="blue")

    for name, func in benchmark_functions.items():
        optimum_str = f"[{', '.join([f'{x:.2f}' for x in func.global_optimum[:3]])}{'...' if len(func.global_optimum) > 3 else ''}]"
        table.add_row(name, func.name, str(func.n_dims), optimum_str)

    console.print(table)


@app.command()
def scalability(
    dimensions: str = typer.Option("2,5,10", "--dims", "-d", help="Comma-separated list of dimensions to test"),
    function: str = typer.Option("sphere", "--function", "-f", help="Base function for scalability test"),
    output_dir: str = typer.Option("scalability_results", "--output", "-o", help="Output directory"),
):
    """Run scalability benchmark across different dimensions."""
    from .benchmarks import SurrogateBenchmarkSuite

    console.print("📈 Running scalability benchmark...")

    # Parse dimensions
    try:
        dims = [int(d.strip()) for d in dimensions.split(",")]
    except ValueError:
        console.print("❌ Invalid dimensions format. Use comma-separated integers (e.g., '2,5,10')")
        return

    console.print(f"Testing {function} function in dimensions: {dims}")

    try:
        suite = SurrogateBenchmarkSuite(output_dir=output_dir, verbose=True)
        results = suite.run_scalability_benchmark(dimensions=dims, base_function=function)

        console.print(f"\n✅ Scalability benchmark complete! Results saved to {output_dir}")

        # Show scalability trends
        if results.benchmark_results:
            console.print("\n📊 Scalability Analysis:")

            # Group by dimension
            dim_results = {}
            for result in results.benchmark_results:
                # Extract dimension from function name or metadata
                func_name = result.function_name
                if any(str(d) in func_name for d in dims):
                    for d in dims:
                        if str(d) in func_name:
                            if d not in dim_results:
                                dim_results[d] = []
                            dim_results[d].append(result)
                            break

            table = Table()
            table.add_column("Dimensions", style="cyan")
            table.add_column("Success Rate", style="green")
            table.add_column("Avg Time (s)", style="yellow")
            table.add_column("Avg Error", style="red")

            for dim in sorted(dims):
                if dim in dim_results:
                    dim_res = dim_results[dim]
                    success_rate = sum(r.success for r in dim_res) / len(dim_res)
                    avg_time = sum(r.total_time for r in dim_res) / len(dim_res)
                    finite_errors = [r.final_error for r in dim_res if r.final_error != float("inf")]
                    avg_error = sum(finite_errors) / len(finite_errors) if finite_errors else float("inf")

                    table.add_row(
                        str(dim),
                        f"{success_rate:.1%}",
                        f"{avg_time:.2f}",
                        f"{avg_error:.2e}" if avg_error != float("inf") else "∞"
                    )

            console.print(table)

    except Exception as e:
        console.print(f"❌ Scalability benchmark failed: {e}")


@app.command()
def validate(
    function: str = typer.Argument(..., help="Name of benchmark function to validate against"),
    surrogate: str = typer.Option("neural_network", "--surrogate", "-s", help="Surrogate type to use"),
    n_samples: int = typer.Option(100, "--samples", "-n", help="Number of training samples"),
    n_test: int = typer.Option(50, "--test", "-t", help="Number of test points for validation"),
):
    """Validate a surrogate model against a benchmark function."""
    import sys

    from .core import SurrogateOptimizer
    from .data.collector import collect_data
    sys.path.append("/root/repo")
    from tests.fixtures.benchmark_functions import benchmark_functions

    if function not in benchmark_functions:
        console.print(f"❌ Unknown function: {function}")
        console.print(f"Available: {list(benchmark_functions.keys())}")
        return

    bench_func = benchmark_functions[function]
    console.print(f"🔍 Validating {surrogate} surrogate on {bench_func.name} function...")

    try:
        # Collect training data
        console.print(f"Collecting {n_samples} training samples...")
        train_data = collect_data(
            function=bench_func,
            n_samples=n_samples,
            bounds=bench_func.bounds,
            sampling="sobol",
            verbose=False
        )

        # Train surrogate
        console.print("Training surrogate model...")
        optimizer = SurrogateOptimizer(surrogate_type=surrogate)
        optimizer.fit_surrogate(train_data)

        # Validate
        console.print(f"Validating on {n_test} test points...")
        validation_metrics = optimizer.validate(
            test_function=bench_func,
            n_test_points=n_test,
            metrics=["mse", "mae", "r2", "gradient_error"]
        )

        # Display validation results
        console.print("\n✅ Validation Results:")

        table = Table()
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        for metric, value in validation_metrics.items():
            if metric == "r2":
                table.add_row("R² Score", f"{value:.4f}")
            elif "error" in metric.lower():
                table.add_row(metric.upper(), f"{value:.2e}")
            else:
                table.add_row(metric.upper(), f"{value:.6f}")

        console.print(table)

        # Interpretation
        if "r2" in validation_metrics:
            r2 = validation_metrics["r2"]
            if r2 > 0.95:
                console.print("🎯 Excellent surrogate quality!")
            elif r2 > 0.9:
                console.print("✅ Good surrogate quality")
            elif r2 > 0.8:
                console.print("⚠️  Fair surrogate quality - consider more training data")
            else:
                console.print("❌ Poor surrogate quality - try different model or more data")

    except Exception as e:
        console.print(f"❌ Validation failed: {e}")


def main():
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()
