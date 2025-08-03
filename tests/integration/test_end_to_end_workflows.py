"""Integration tests for complete surrogate optimization workflows."""

import numpy as np
import pytest
import tempfile
import os
from unittest.mock import patch

from surrogate_optim import (
    collect_data,
    NeuralSurrogate,
    GPSurrogate,
    RandomForestSurrogate,
    HybridSurrogate,
    SurrogateOptimizer,
    TrustRegionOptimizer,
    MultiStartOptimizer,
    optimize_with_surrogate,
    benchmark_surrogate,
    validate_surrogate,
)
from surrogate_optim.data import Dataset, ActiveLearner


class TestCompleteOptimizationWorkflow:
    """Test complete optimization workflows from data collection to optimization."""

    def setup_method(self):
        """Set up test fixtures."""
        # Define test problems
        def rosenbrock_2d(x):
            """2D Rosenbrock function."""
            x = np.asarray(x)
            return -(100.0 * (x[1] - x[0]**2)**2 + (1 - x[0])**2)
        
        def ackley_2d(x):
            """2D Ackley function."""
            x = np.asarray(x)
            return -(-20 * np.exp(-0.2 * np.sqrt(0.5 * np.sum(x**2))) 
                     - np.exp(0.5 * np.sum(np.cos(2 * np.pi * x))) + 20 + np.e)
        
        self.test_functions = {
            "rosenbrock": rosenbrock_2d,
            "ackley": ackley_2d,
        }
        
        self.bounds = [(-2.0, 2.0), (-2.0, 2.0)]
        
    def test_neural_surrogate_workflow(self):
        """Test complete workflow with neural network surrogate."""
        function = self.test_functions["rosenbrock"]
        
        # Step 1: Collect data
        dataset = collect_data(
            function=function,
            n_samples=80,
            bounds=self.bounds,
            sampling="sobol",
            random_state=42
        )
        
        assert dataset.n_samples == 80
        assert dataset.n_dims == 2
        
        # Step 2: Train surrogate
        surrogate = NeuralSurrogate(
            hidden_dims=[32, 16],
            epochs=100,
            learning_rate=0.01
        )
        
        with patch('builtins.print'):  # Suppress training output
            surrogate.fit(dataset.X, dataset.y)
        
        # Step 3: Validate surrogate
        validation_results = validate_surrogate(
            surrogate=surrogate,
            test_function=function,
            bounds=self.bounds,
            n_test_points=50,
            seed=123
        )
        
        # Should achieve reasonable validation performance
        assert validation_results["r2"] > 0.5
        assert validation_results["mean_gradient_error"] < 1.0
        
        # Step 4: Optimize
        x0 = np.array([0.0, 0.0])
        result = optimize_with_surrogate(
            surrogate=surrogate,
            x0=x0,
            bounds=self.bounds,
            options={"maxiter": 100}
        )
        
        # Should find reasonable solution
        assert result["success"]
        # Rosenbrock optimum is at (1, 1)
        distance_to_optimum = np.linalg.norm(result["x"] - np.array([1.0, 1.0]))
        assert distance_to_optimum < 0.5
        
    def test_gp_surrogate_workflow(self):
        """Test complete workflow with GP surrogate."""
        function = self.test_functions["ackley"]
        
        # Collect data
        dataset = collect_data(
            function=function,
            n_samples=50,
            bounds=self.bounds,
            sampling="latin_hypercube",
            random_state=456
        )
        
        # Train GP surrogate
        surrogate = GPSurrogate(kernel="rbf", length_scale=0.5)
        surrogate.fit(dataset.X, dataset.y)
        
        # Optimize
        result = optimize_with_surrogate(
            surrogate=surrogate,
            x0=np.array([1.0, 1.0]),
            bounds=self.bounds
        )
        
        # Should find solution close to global optimum (0, 0)
        distance_to_optimum = np.linalg.norm(result["x"])
        assert distance_to_optimum < 1.0
        
    def test_hybrid_surrogate_workflow(self):
        """Test workflow with hybrid ensemble surrogate."""
        function = self.test_functions["rosenbrock"]
        
        # Collect more data for ensemble
        dataset = collect_data(
            function=function,
            n_samples=100,
            bounds=self.bounds,
            sampling="sobol",
            random_state=789
        )
        
        # Create hybrid surrogate
        models = [
            ("nn", NeuralSurrogate(hidden_dims=[16], epochs=20)),
            ("gp", GPSurrogate()),
            ("rf", RandomForestSurrogate(n_estimators=20)),
        ]
        
        surrogate = HybridSurrogate(models=models)
        
        with patch('builtins.print'):
            surrogate.fit(dataset.X, dataset.y)
        
        # Multi-start optimization for better global search
        result = MultiStartOptimizer(
            surrogate=surrogate,
            n_starts=10,
            start_method="random"
        ).optimize_global(bounds=self.bounds)
        
        assert result["n_successful"] > 0
        distance_to_optimum = np.linalg.norm(result["best_point"] - np.array([1.0, 1.0]))
        assert distance_to_optimum < 1.0


class TestActiveLearningWorkflow:
    """Test active learning workflows."""

    def setup_method(self):
        """Set up test fixtures."""
        def complex_function(x):
            """Complex test function with multiple features."""
            x = np.asarray(x)
            return -(x[0]**2 + x[1]**2) + 0.5 * np.sin(4 * x[0]) * np.cos(4 * x[1])
        
        self.function = complex_function
        self.bounds = [(-2.0, 2.0), (-2.0, 2.0)]
        
    def test_active_learning_improves_optimization(self):
        """Test that active learning improves optimization performance."""
        # Initial random dataset
        initial_data = collect_data(
            function=self.function,
            n_samples=20,
            bounds=self.bounds,
            sampling="random",
            random_state=42
        )
        
        # Passive surrogate (trained on initial data only)
        passive_surrogate = GPSurrogate()
        passive_surrogate.fit(initial_data.X, initial_data.y)
        
        # Active learning
        learner = ActiveLearner(
            function=self.function,
            initial_data=initial_data,
            surrogate_type="gp"
        )
        
        with patch('builtins.print'):
            enhanced_data = learner.learn_iteratively(
                n_iterations=5,
                batch_size=4,
                acquisition_function="expected_improvement",
                bounds=self.bounds
            )
        
        # Active surrogate (trained on enhanced data)
        active_surrogate = GPSurrogate()
        active_surrogate.fit(enhanced_data.X, enhanced_data.y)
        
        # Compare optimization performance
        x0 = np.array([1.5, 1.5])
        
        passive_result = optimize_with_surrogate(
            surrogate=passive_surrogate,
            x0=x0,
            bounds=self.bounds
        )
        
        active_result = optimize_with_surrogate(
            surrogate=active_surrogate,
            x0=x0,
            bounds=self.bounds
        )
        
        # Active learning should generally lead to better optimization
        # (or at least not significantly worse)
        assert active_result["fun"] >= passive_result["fun"] - 0.5
        
    def test_different_acquisition_functions(self):
        """Test active learning with different acquisition functions."""
        initial_data = collect_data(
            function=self.function,
            n_samples=15,
            bounds=self.bounds,
            sampling="sobol",
            random_state=123
        )
        
        acquisition_functions = [
            "expected_improvement",
            "upper_confidence_bound",
            "entropy_search"
        ]
        
        results = {}
        
        for acq_func in acquisition_functions:
            learner = ActiveLearner(
                function=self.function,
                initial_data=initial_data,
                surrogate_type="gp"
            )
            
            with patch('builtins.print'):
                enhanced_data = learner.learn_iteratively(
                    n_iterations=3,
                    batch_size=3,
                    acquisition_function=acq_func,
                    bounds=self.bounds
                )
            
            # Train final surrogate and optimize
            surrogate = GPSurrogate()
            surrogate.fit(enhanced_data.X, enhanced_data.y)
            
            opt_result = optimize_with_surrogate(
                surrogate=surrogate,
                x0=np.array([1.0, 1.0]),
                bounds=self.bounds
            )
            
            results[acq_func] = {
                "data_size": enhanced_data.n_samples,
                "opt_value": opt_result["fun"],
                "success": opt_result["success"]
            }
        
        # All should produce valid results
        for acq_func, result in results.items():
            assert result["data_size"] == 15 + 3 * 3  # Initial + iterations * batch
            assert result["success"]


class TestTrustRegionWorkflow:
    """Test trust region optimization workflows."""

    def setup_method(self):
        """Set up test fixtures."""
        def noisy_quadratic(x):
            """Quadratic function with noise."""
            x = np.asarray(x)
            return -np.sum(x**2) + 0.1 * np.sin(10 * np.sum(x))
        
        self.function = noisy_quadratic
        self.bounds = [(-3.0, 3.0), (-3.0, 3.0)]
        
    def test_trust_region_with_validation(self):
        """Test trust region optimization with true function validation."""
        # Collect data and train surrogate
        dataset = collect_data(
            function=self.function,
            n_samples=60,
            bounds=self.bounds,
            sampling="sobol",
            random_state=42
        )
        
        surrogate = NeuralSurrogate(hidden_dims=[32], epochs=50)
        with patch('builtins.print'):
            surrogate.fit(dataset.X, dataset.y)
        
        # Trust region optimization with validation
        optimizer = TrustRegionOptimizer(
            surrogate=surrogate,
            true_function=self.function,
            initial_radius=0.5,
            max_radius=2.0
        )
        
        result = optimizer.optimize(
            x0=np.array([2.0, 2.0]),
            max_iterations=20,
            validate_every=3
        )
        
        assert "x" in result
        assert "trajectory" in result
        assert "iterations" in result
        assert "converged" in result
        
        # Should converge to reasonable solution
        final_point = result["x"]
        assert np.linalg.norm(final_point) < 1.0  # Close to origin
        
    def test_trust_region_radius_adaptation(self):
        """Test that trust region adapts radius appropriately."""
        dataset = collect_data(
            function=self.function,
            n_samples=40,
            bounds=self.bounds,
            sampling="uniform",
            random_state=456
        )
        
        surrogate = GPSurrogate()
        surrogate.fit(dataset.X, dataset.y)
        
        optimizer = TrustRegionOptimizer(
            surrogate=surrogate,
            true_function=self.function,
            initial_radius=0.1,  # Start small
            max_radius=1.0
        )
        
        initial_radius = optimizer.radius
        
        result = optimizer.optimize(
            x0=np.array([1.0, 1.0]),
            max_iterations=15,
            validate_every=2
        )
        
        # Radius should have adapted during optimization
        final_radius = optimizer.radius
        assert final_radius != initial_radius
        assert 0 < final_radius <= optimizer.max_radius


class TestBenchmarkingWorkflow:
    """Test comprehensive benchmarking workflows."""

    def test_multi_function_benchmark(self):
        """Test benchmarking across multiple functions and dimensions."""
        surrogate_model = NeuralSurrogate(hidden_dims=[24], epochs=30)
        
        with patch('builtins.print'):
            results = benchmark_surrogate(
                surrogate_model=surrogate_model,
                test_functions=["sphere", "rosenbrock", "ackley"],
                dimensions=[2, 5],
                n_trials=3,
                n_train_samples=40,
                seed=42
            )
        
        # Check structure
        assert "function_results" in results
        assert "summary" in results
        
        # Check all functions and dimensions tested
        for func in ["sphere", "rosenbrock", "ackley"]:
            assert func in results["function_results"]
            for dim in [2, 5]:
                dim_key = f"dim_{dim}"
                assert dim_key in results["function_results"][func]
                
        # Check summary aggregation
        summary = results["summary"]
        expected_trials = 3 * 3 * 2  # functions * trials * dimensions
        assert summary["n_trials_total"] == expected_trials
        assert 0 <= summary["mean_gap"] <= 1.0
        assert summary["mean_grad_error"] >= 0
        
    def test_surrogate_comparison_benchmark(self):
        """Test comparing different surrogate types."""
        test_function_name = "sphere"
        surrogate_configs = {
            "neural_network": NeuralSurrogate(hidden_dims=[16], epochs=20),
            "gaussian_process": GPSurrogate(),
            "random_forest": RandomForestSurrogate(n_estimators=20),
        }
        
        comparison_results = {}
        
        for name, surrogate_model in surrogate_configs.items():
            with patch('builtins.print'):
                results = benchmark_surrogate(
                    surrogate_model=surrogate_model,
                    test_functions=[test_function_name],
                    dimensions=[2],
                    n_trials=3,
                    n_train_samples=30,
                    seed=123
                )
            
            comparison_results[name] = results["summary"]["mean_gap"]
        
        # All should produce reasonable results
        for name, mean_gap in comparison_results.items():
            assert 0 <= mean_gap <= 1.0, f"{name} produced invalid gap: {mean_gap}"
            
    def test_benchmark_with_validation(self):
        """Test benchmarking combined with validation."""
        surrogate_model = GPSurrogate()
        
        # Benchmark
        with patch('builtins.print'):
            benchmark_results = benchmark_surrogate(
                surrogate_model=surrogate_model,
                test_functions=["sphere"],
                dimensions=[2],
                n_trials=2,
                n_train_samples=30,
                seed=789
            )
        
        # Separate validation
        def sphere_function(x):
            x = np.asarray(x)
            return -np.sum(x**2)
        
        # Train surrogate for validation
        dataset = collect_data(
            function=sphere_function,
            n_samples=30,
            bounds=[(-2.0, 2.0), (-2.0, 2.0)],
            sampling="sobol",
            random_state=789
        )
        
        surrogate = GPSurrogate()
        surrogate.fit(dataset.X, dataset.y)
        
        validation_results = validate_surrogate(
            surrogate=surrogate,
            test_function=sphere_function,
            bounds=[(-2.0, 2.0), (-2.0, 2.0)],
            n_test_points=50,
            seed=789
        )
        
        # Results should be consistent
        benchmark_gap = benchmark_results["function_results"]["sphere"]["dim_2"]["mean_gap"]
        validation_r2 = validation_results["r2"]
        
        # High R² should correspond to low optimality gap
        if validation_r2 > 0.9:
            assert benchmark_gap < 0.2


class TestEndToEndReproducibility:
    """Test reproducibility of complete workflows."""

    def test_reproducible_optimization(self):
        """Test that optimization is reproducible with same seed."""
        def test_function(x):
            x = np.asarray(x)
            return -(x[0] - 1)**2 - (x[1] + 0.5)**2
        
        bounds = [(-2.0, 3.0), (-2.0, 2.0)]
        
        def run_optimization(seed):
            # Data collection
            dataset = collect_data(
                function=test_function,
                n_samples=40,
                bounds=bounds,
                sampling="sobol",
                random_state=seed
            )
            
            # Surrogate training
            surrogate = NeuralSurrogate(hidden_dims=[16], epochs=20)
            with patch('builtins.print'):
                surrogate.fit(dataset.X, dataset.y)
            
            # Optimization
            result = optimize_with_surrogate(
                surrogate=surrogate,
                x0=np.array([0.0, 0.0]),
                bounds=bounds
            )
            
            return result["x"], result["fun"]
        
        # Run twice with same seed
        point1, value1 = run_optimization(seed=42)
        point2, value2 = run_optimization(seed=42)
        
        # Should be very close (allowing for numerical differences)
        assert np.allclose(point1, point2, atol=1e-3)
        assert abs(value1 - value2) < 1e-6
        
    def test_reproducible_active_learning(self):
        """Test reproducible active learning."""
        def test_function(x):
            x = np.asarray(x)
            return np.sum(x**2) + 0.1 * np.sin(5 * np.sum(x))
        
        bounds = [(-1.0, 1.0), (-1.0, 1.0)]
        
        def run_active_learning(seed):
            # Initial data
            initial_data = collect_data(
                function=test_function,
                n_samples=10,
                bounds=bounds,
                sampling="random",
                random_state=seed
            )
            
            # Active learning
            learner = ActiveLearner(
                function=test_function,
                initial_data=initial_data,
                surrogate_type="gp"
            )
            
            with patch('builtins.print'):
                final_data = learner.learn_iteratively(
                    n_iterations=3,
                    batch_size=2,
                    acquisition_function="expected_improvement",
                    bounds=bounds
                )
            
            return final_data.X, final_data.y
        
        # Run twice with same seed
        X1, y1 = run_active_learning(seed=123)
        X2, y2 = run_active_learning(seed=123)
        
        # Should be identical
        assert np.allclose(X1, X2)
        assert np.allclose(y1, y2)


class TestErrorHandlingAndRobustness:
    """Test error handling and robustness of workflows."""

    def test_insufficient_data_handling(self):
        """Test handling of insufficient training data."""
        def simple_function(x):
            return np.sum(np.asarray(x)**2)
        
        # Very small dataset
        dataset = collect_data(
            function=simple_function,
            n_samples=5,  # Very small
            bounds=[(-1.0, 1.0), (-1.0, 1.0)],
            sampling="random",
            random_state=42
        )
        
        # Should still work but may have warnings
        surrogate = NeuralSurrogate(hidden_dims=[8], epochs=10)
        
        with patch('builtins.print'):
            surrogate.fit(dataset.X, dataset.y)
        
        # Should still be able to predict
        prediction = surrogate.predict(np.array([0.5, 0.5]))
        assert isinstance(prediction, (float, np.float64))
        assert not np.isnan(prediction)
        
    def test_optimization_bounds_handling(self):
        """Test proper handling of optimization bounds."""
        def bounded_function(x):
            x = np.asarray(x)
            # Function that prefers edges of domain
            return -(abs(x[0]) + abs(x[1]))
        
        dataset = collect_data(
            function=bounded_function,
            n_samples=30,
            bounds=[(-2.0, 2.0), (-2.0, 2.0)],
            sampling="uniform",
            random_state=123
        )
        
        surrogate = GPSurrogate()
        surrogate.fit(dataset.X, dataset.y)
        
        # Optimize with tight bounds
        tight_bounds = [(-1.0, 1.0), (-1.0, 1.0)]
        result = optimize_with_surrogate(
            surrogate=surrogate,
            x0=np.array([0.0, 0.0]),
            bounds=tight_bounds
        )
        
        # Solution should respect bounds
        for i, (lower, upper) in enumerate(tight_bounds):
            assert lower <= result["x"][i] <= upper
            
    def test_degenerate_function_handling(self):
        """Test handling of degenerate/constant functions."""
        def constant_function(x):
            return 5.0  # Always return constant
        
        dataset = collect_data(
            function=constant_function,
            n_samples=20,
            bounds=[(-1.0, 1.0), (-1.0, 1.0)],
            sampling="uniform",
            random_state=456
        )
        
        # All y values should be the same
        assert np.allclose(dataset.y, 5.0)
        
        # Surrogate should still fit (though gradients may be zero)
        surrogate = RandomForestSurrogate(n_estimators=10)
        surrogate.fit(dataset.X, dataset.y)
        
        prediction = surrogate.predict(np.array([0.5, 0.5]))
        assert abs(prediction - 5.0) < 1.0  # Should predict close to constant


@pytest.fixture
def temp_directory():
    """Create temporary directory for testing file operations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def simple_test_function():
    """Simple test function for integration tests."""
    def func(x):
        x = np.asarray(x)
        return -np.sum(x**2) + 0.1 * np.sin(5 * np.sum(x))
    return func