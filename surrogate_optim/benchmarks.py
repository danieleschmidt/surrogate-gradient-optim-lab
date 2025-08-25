"""Comprehensive benchmarking suite for surrogate optimization."""

from dataclasses import dataclass, field
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Union

from jax import Array
import jax.numpy as jnp

from .core import SurrogateOptimizer
from .data.collector import collect_data

sys.path.append("/root/repo")
from tests.fixtures.benchmark_functions import (
    BenchmarkFunction,
    benchmark_functions,
    get_2d_functions,
)


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    function_name: str
    surrogate_type: str
    optimizer_type: str
    n_training_samples: int
    success: bool
    final_error: float
    optimization_time: float
    total_time: float
    n_iterations: int
    n_function_evaluations: int
    found_optimum: Array
    convergence_history: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """Result from comparing multiple methods on benchmark functions."""
    benchmark_results: List[BenchmarkResult]
    summary_stats: Dict[str, Any] = field(default_factory=dict)
    rankings: Dict[str, int] = field(default_factory=dict)


class SurrogateBenchmarkSuite:
    """Comprehensive benchmarking suite for surrogate optimization methods."""

    def __init__(
        self,
        output_dir: Union[str, Path] = "benchmark_results",
        save_results: bool = True,
        verbose: bool = True,
    ):
        """Initialize benchmark suite.
        
        Args:
            output_dir: Directory to save results
            save_results: Whether to save detailed results
            verbose: Whether to print progress
        """
        self.output_dir = Path(output_dir)
        self.save_results = save_results
        self.verbose = verbose

        if self.save_results:
            self.output_dir.mkdir(exist_ok=True)

        # Default benchmark configuration
        self.default_config = {
            "surrogate_types": ["neural_network", "gaussian_process", "random_forest"],
            "optimizer_types": ["gradient_descent", "trust_region"],
            "training_sample_sizes": [50, 100, 200],
            "n_trials": 5,
            "timeout_seconds": 300,
        }

    def run_single_benchmark(
        self,
        function: BenchmarkFunction,
        surrogate_type: str = "neural_network",
        optimizer_type: str = "gradient_descent",
        n_training_samples: int = 100,
        trial_id: int = 0,
        timeout: float = 300.0,
    ) -> BenchmarkResult:
        """Run a single benchmark experiment.
        
        Args:
            function: Benchmark function to optimize
            surrogate_type: Type of surrogate model
            optimizer_type: Type of optimizer
            n_training_samples: Number of training samples
            trial_id: Trial identifier for reproducibility
            timeout: Timeout in seconds
            
        Returns:
            Benchmark result
        """
        if self.verbose:
            print(f"  Running {surrogate_type} + {optimizer_type} on {function.name} "
                  f"(trial {trial_id + 1}, {n_training_samples} samples)")

        start_time = time.time()
        success = False
        final_error = float("inf")
        optimization_time = 0.0
        n_iterations = 0
        n_function_evaluations = n_training_samples
        found_optimum = jnp.zeros(function.n_dims)
        convergence_history = []
        metadata = {}

        try:
            # Collect training data
            data_start = time.time()
            data = collect_data(
                function=function,
                n_samples=n_training_samples,
                bounds=function.bounds,
                sampling="sobol",
                verbose=False
            )
            data_time = time.time() - data_start

            # Create and train surrogate optimizer
            train_start = time.time()
            optimizer = SurrogateOptimizer(
                surrogate_type=surrogate_type,
                optimizer_type=optimizer_type
            )
            optimizer.fit_surrogate(data)
            train_time = time.time() - train_start

            # Run optimization
            opt_start = time.time()
            # Start from center of bounds as initial point
            initial_point = jnp.array([(low + high) / 2 for low, high in function.bounds])

            result = optimizer.optimize(
                initial_point=initial_point,
                bounds=function.bounds,
                num_steps=100
            )
            optimization_time = time.time() - opt_start

            # Calculate performance metrics
            found_optimum = result.x
            final_error = abs(function(found_optimum) - function.optimal_value)
            success = result.success and final_error < 1e-2  # Success threshold
            n_iterations = result.nit

            if hasattr(result, "convergence_history") and result.convergence_history:
                convergence_history = [float(x) for x in result.convergence_history]

            metadata = {
                "data_collection_time": data_time,
                "surrogate_training_time": train_time,
                "optimization_method": result.message,
                "distance_to_optimum": float(jnp.linalg.norm(found_optimum - function.global_optimum)),
                "function_value": float(function(found_optimum)),
                "optimal_function_value": float(function.optimal_value),
            }

        except Exception as e:
            if self.verbose:
                print(f"    Error: {e}")
            metadata["error"] = str(e)

        total_time = time.time() - start_time

        return BenchmarkResult(
            function_name=function.name,
            surrogate_type=surrogate_type,
            optimizer_type=optimizer_type,
            n_training_samples=n_training_samples,
            success=success,
            final_error=final_error,
            optimization_time=optimization_time,
            total_time=total_time,
            n_iterations=n_iterations,
            n_function_evaluations=n_function_evaluations,
            found_optimum=found_optimum,
            convergence_history=convergence_history,
            metadata=metadata,
        )

    def run_function_benchmark(
        self,
        function: BenchmarkFunction,
        config: Optional[Dict] = None,
    ) -> List[BenchmarkResult]:
        """Run comprehensive benchmark on a single function.
        
        Args:
            function: Benchmark function
            config: Optional configuration override
            
        Returns:
            List of benchmark results
        """
        config = config or self.default_config
        results = []

        if self.verbose:
            print(f"Benchmarking {function.name} (dims: {function.n_dims})")

        for surrogate_type in config["surrogate_types"]:
            for optimizer_type in config["optimizer_types"]:
                for n_samples in config["training_sample_sizes"]:
                    for trial in range(config["n_trials"]):
                        result = self.run_single_benchmark(
                            function=function,
                            surrogate_type=surrogate_type,
                            optimizer_type=optimizer_type,
                            n_training_samples=n_samples,
                            trial_id=trial,
                            timeout=config.get("timeout_seconds", 300)
                        )
                        results.append(result)

        return results

    def run_suite_benchmark(
        self,
        function_names: Optional[List[str]] = None,
        config: Optional[Dict] = None,
    ) -> ComparisonResult:
        """Run benchmark suite on multiple functions.
        
        Args:
            function_names: List of function names to benchmark (default: all 2D functions)
            config: Optional configuration override
            
        Returns:
            Comparison results across all functions and methods
        """
        config = config or self.default_config

        if function_names is None:
            # Default to 2D functions for comprehensive testing
            available_functions = get_2d_functions()
            function_names = list(available_functions.keys())

        all_results = []

        if self.verbose:
            print(f"Running benchmark suite on {len(function_names)} functions")

        for func_name in function_names:
            if func_name not in benchmark_functions:
                print(f"Warning: Unknown function {func_name}, skipping")
                continue

            function = benchmark_functions[func_name]
            results = self.run_function_benchmark(function, config)
            all_results.extend(results)

        # Compute summary statistics
        summary_stats = self._compute_summary_stats(all_results)
        rankings = self._compute_rankings(all_results)

        comparison_result = ComparisonResult(
            benchmark_results=all_results,
            summary_stats=summary_stats,
            rankings=rankings
        )

        if self.save_results:
            self._save_results(comparison_result)

        return comparison_result

    def run_scalability_benchmark(
        self,
        dimensions: List[int] = [2, 5, 10, 20],
        base_function: str = "sphere",
    ) -> ComparisonResult:
        """Run scalability benchmark across different dimensions.
        
        Args:
            dimensions: List of dimensions to test
            base_function: Base function type to scale
            
        Returns:
            Scalability benchmark results
        """
        if self.verbose:
            print(f"Running scalability benchmark on {base_function} function")

        all_results = []

        for n_dims in dimensions:
            # Create function of appropriate dimension
            if base_function == "sphere":
                from ..tests.fixtures.benchmark_functions import Sphere
                function = Sphere(n_dims)
            elif base_function == "rosenbrock":
                from ..tests.fixtures.benchmark_functions import Rosenbrock
                function = Rosenbrock(n_dims)
            elif base_function == "rastrigin":
                from ..tests.fixtures.benchmark_functions import Rastrigin
                function = Rastrigin(n_dims)
            else:
                raise ValueError(f"Unsupported function for scalability: {base_function}")

            # Scale training samples with dimensionality
            config = {
                **self.default_config,
                "training_sample_sizes": [20 * n_dims, 50 * n_dims],
                "n_trials": 3,  # Fewer trials for high-dim
            }

            results = self.run_function_benchmark(function, config)
            all_results.extend(results)

        summary_stats = self._compute_summary_stats(all_results)
        rankings = self._compute_rankings(all_results)

        return ComparisonResult(
            benchmark_results=all_results,
            summary_stats=summary_stats,
            rankings=rankings
        )

    def run_quick_benchmark(
        self,
        n_functions: int = 5,
        n_trials: int = 2,
    ) -> ComparisonResult:
        """Run a quick benchmark for testing purposes.
        
        Args:
            n_functions: Number of functions to test
            n_trials: Number of trials per configuration
            
        Returns:
            Quick benchmark results
        """
        # Use smaller, faster functions
        quick_functions = ["sphere_2d", "rosenbrock_2d", "rastrigin_2d", "ackley_2d", "matyas"]
        function_names = quick_functions[:n_functions]

        config = {
            "surrogate_types": ["neural_network", "gaussian_process"],
            "optimizer_types": ["gradient_descent"],
            "training_sample_sizes": [50],
            "n_trials": n_trials,
            "timeout_seconds": 60,
        }

        return self.run_suite_benchmark(function_names, config)

    def _compute_summary_stats(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Compute summary statistics from benchmark results."""
        if not results:
            return {}

        # Group by method (surrogate + optimizer)
        method_results = {}
        for result in results:
            method = f"{result.surrogate_type}+{result.optimizer_type}"
            if method not in method_results:
                method_results[method] = []
            method_results[method].append(result)

        summary = {}
        for method, method_res in method_results.items():
            success_rate = sum(r.success for r in method_res) / len(method_res)
            avg_error = jnp.mean([r.final_error for r in method_res if jnp.isfinite(r.final_error)])
            avg_time = jnp.mean([r.total_time for r in method_res])
            avg_iterations = jnp.mean([r.n_iterations for r in method_res])

            summary[method] = {
                "success_rate": float(success_rate),
                "average_error": float(avg_error),
                "average_time": float(avg_time),
                "average_iterations": float(avg_iterations),
                "num_experiments": len(method_res),
            }

        return summary

    def _compute_rankings(self, results: List[BenchmarkResult]) -> Dict[str, int]:
        """Compute method rankings based on overall performance."""
        summary = self._compute_summary_stats(results)

        if not summary:
            return {}

        # Rank by success rate (descending), then by average error (ascending)
        methods = list(summary.keys())
        scores = []

        for method in methods:
            stats = summary[method]
            # Composite score: success_rate - log(1 + avg_error)
            score = stats["success_rate"] - jnp.log(1 + stats["average_error"])
            scores.append((score, method))

        # Sort by score (descending)
        scores.sort(reverse=True)

        rankings = {}
        for rank, (score, method) in enumerate(scores, 1):
            rankings[method] = rank

        return rankings

    def _save_results(self, results: ComparisonResult):
        """Save benchmark results to files."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Save detailed results
        results_file = self.output_dir / f"benchmark_results_{timestamp}.json"

        # Convert results to serializable format
        serializable_results = []
        for result in results.benchmark_results:
            result_dict = {
                "function_name": result.function_name,
                "surrogate_type": result.surrogate_type,
                "optimizer_type": result.optimizer_type,
                "n_training_samples": result.n_training_samples,
                "success": result.success,
                "final_error": float(result.final_error),
                "optimization_time": result.optimization_time,
                "total_time": result.total_time,
                "n_iterations": result.n_iterations,
                "n_function_evaluations": result.n_function_evaluations,
                "found_optimum": result.found_optimum.tolist(),
                "convergence_history": result.convergence_history,
                "metadata": result.metadata,
            }
            serializable_results.append(result_dict)

        full_results = {
            "benchmark_results": serializable_results,
            "summary_stats": results.summary_stats,
            "rankings": results.rankings,
            "timestamp": timestamp,
        }

        with open(results_file, "w") as f:
            json.dump(full_results, f, indent=2)

        # Save summary report
        self._save_summary_report(results, timestamp)

        if self.verbose:
            print(f"Results saved to {results_file}")

    def _save_summary_report(self, results: ComparisonResult, timestamp: str):
        """Save a human-readable summary report."""
        report_file = self.output_dir / f"benchmark_report_{timestamp}.md"

        with open(report_file, "w") as f:
            f.write("# Surrogate Optimization Benchmark Report\n\n")
            f.write(f"Generated: {timestamp}\n\n")

            f.write("## Summary Statistics\n\n")
            f.write("| Method | Success Rate | Avg Error | Avg Time (s) | Avg Iterations | Experiments |\n")
            f.write("|--------|--------------|-----------|--------------|----------------|--------------|\n")

            for method, stats in results.summary_stats.items():
                f.write(f"| {method} | {stats['success_rate']:.2%} | "
                       f"{stats['average_error']:.2e} | {stats['average_time']:.2f} | "
                       f"{stats['average_iterations']:.1f} | {stats['num_experiments']} |\n")

            f.write("\n## Method Rankings\n\n")
            f.write("| Rank | Method |\n")
            f.write("|------|--------|\n")

            for method, rank in sorted(results.rankings.items(), key=lambda x: x[1]):
                f.write(f"| {rank} | {method} |\n")

            f.write("\n## Detailed Results\n\n")
            f.write(f"Total experiments: {len(results.benchmark_results)}\n")

            # Group by function
            function_results = {}
            for result in results.benchmark_results:
                if result.function_name not in function_results:
                    function_results[result.function_name] = []
                function_results[result.function_name].append(result)

            for func_name, func_results in function_results.items():
                f.write(f"\n### {func_name}\n\n")
                successful = [r for r in func_results if r.success]
                success_rate = len(successful) / len(func_results)
                f.write(f"- Success Rate: {success_rate:.2%} ({len(successful)}/{len(func_results)})\n")

                if successful:
                    avg_error = jnp.mean([r.final_error for r in successful])
                    f.write(f"- Average Error (successful): {avg_error:.2e}\n")


def run_benchmark_suite(
    function_names: Optional[List[str]] = None,
    config: Optional[Dict] = None,
    output_dir: str = "benchmark_results",
    verbose: bool = True,
) -> ComparisonResult:
    """Convenience function to run benchmark suite.
    
    Args:
        function_names: Functions to benchmark (default: 2D functions)
        config: Benchmark configuration
        output_dir: Output directory for results
        verbose: Whether to print progress
        
    Returns:
        Benchmark results
    """
    suite = SurrogateBenchmarkSuite(output_dir=output_dir, verbose=verbose)
    return suite.run_suite_benchmark(function_names, config)


def run_quick_benchmark(
    output_dir: str = "benchmark_results",
    verbose: bool = True,
) -> ComparisonResult:
    """Run a quick benchmark for testing.
    
    Args:
        output_dir: Output directory for results
        verbose: Whether to print progress
        
    Returns:
        Quick benchmark results
    """
    suite = SurrogateBenchmarkSuite(output_dir=output_dir, verbose=verbose)
    return suite.run_quick_benchmark()
