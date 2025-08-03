"""Unit tests for utility functions."""

import numpy as np
import pytest
from unittest.mock import Mock, patch

from surrogate_optim.utils import (
    optimize_with_surrogate,
    multi_start_optimize,
    benchmark_surrogate,
    validate_surrogate,
    create_benchmark_report,
    rosenbrock,
    rastrigin,
    ackley,
    sphere,
    griewank,
)


class MockSurrogate:
    """Mock surrogate for testing utilities."""
    
    def __init__(self, function=None):
        """Initialize mock surrogate."""
        if function is None:
            # Default to quadratic function
            self.function = lambda x: -np.sum(np.asarray(x)**2)
        else:
            self.function = function
    
    def predict(self, x):
        """Mock prediction."""
        return self.function(x)
    
    def gradient(self, x):
        """Mock gradient (finite differences)."""
        x = np.asarray(x)
        eps = 1e-6
        grad = np.zeros_like(x)
        
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            grad[i] = (self.function(x_plus) - self.function(x_minus)) / (2 * eps)
        
        return grad
    
    def fit(self, X, y):
        """Mock fit method."""
        pass


class TestOptimizeWithSurrogate:
    """Test optimize_with_surrogate function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.surrogate = MockSurrogate()
        
    def test_basic_optimization(self):
        """Test basic optimization."""
        x0 = np.array([2.0, 2.0])
        
        result = optimize_with_surrogate(
            surrogate=self.surrogate,
            x0=x0,
            method="L-BFGS-B"
        )
        
        assert "x" in result
        assert "fun" in result
        assert "success" in result
        assert "nit" in result
        
        assert isinstance(result["x"], np.ndarray)
        assert result["x"].shape == (2,)
        assert isinstance(result["fun"], (float, np.float64))
        
        # Should find optimum near origin for quadratic function
        assert np.linalg.norm(result["x"]) < 0.1
        
    def test_optimization_with_bounds(self):
        """Test optimization with bounds."""
        x0 = np.array([1.0, 1.0])
        bounds = [(-2.0, 2.0), (-2.0, 2.0)]
        
        result = optimize_with_surrogate(
            surrogate=self.surrogate,
            x0=x0,
            bounds=bounds
        )
        
        # Check bounds are respected
        for i, (lower, upper) in enumerate(bounds):
            assert lower <= result["x"][i] <= upper
            
    def test_optimization_with_options(self):
        """Test optimization with custom options."""
        x0 = np.array([1.0, 1.0])
        options = {"maxiter": 10, "ftol": 1e-4}
        
        result = optimize_with_surrogate(
            surrogate=self.surrogate,
            x0=x0,
            options=options
        )
        
        assert result["nit"] <= 10  # Should respect maxiter
        
    def test_different_methods(self):
        """Test different optimization methods."""
        x0 = np.array([1.0, 1.0])
        methods = ["L-BFGS-B", "BFGS", "CG"]
        
        for method in methods:
            result = optimize_with_surrogate(
                surrogate=self.surrogate,
                x0=x0,
                method=method
            )
            
            assert "x" in result
            assert isinstance(result["x"], np.ndarray)


class TestMultiStartOptimize:
    """Test multi_start_optimize function."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create surrogate with multiple local minima
        def multimodal_function(x):
            x = np.asarray(x)
            return -(np.sum(x**2) + 2 * np.sin(5 * x[0]) * np.sin(5 * x[1]))
        
        self.surrogate = MockSurrogate(multimodal_function)
        
    def test_multi_start_optimization(self):
        """Test multi-start optimization."""
        bounds = [(-3.0, 3.0), (-3.0, 3.0)]
        
        result = multi_start_optimize(
            surrogate=self.surrogate,
            bounds=bounds,
            n_starts=10,
            seed=42
        )
        
        assert "x" in result
        assert "fun" in result
        assert "best_result" in result
        assert "all_results" in result
        assert "n_successful" in result
        assert "success_rate" in result
        
        assert isinstance(result["x"], np.ndarray)
        assert result["x"].shape == (2,)
        assert len(result["all_results"]) <= 10
        assert 0 <= result["success_rate"] <= 1
        
    def test_multi_start_finds_better_solution(self):
        """Test that multi-start finds better solutions than single start."""
        bounds = [(-2.0, 2.0), (-2.0, 2.0)]
        
        # Single start from poor initial point
        single_result = optimize_with_surrogate(
            surrogate=self.surrogate,
            x0=np.array([-1.5, -1.5]),
            bounds=bounds
        )
        
        # Multi-start
        multi_result = multi_start_optimize(
            surrogate=self.surrogate,
            bounds=bounds,
            n_starts=20,
            seed=42
        )
        
        # Multi-start should generally find better or equal solution
        assert multi_result["fun"] >= single_result["fun"] - 0.1
        
    def test_different_start_methods(self):
        """Test different methods for generating starting points."""
        bounds = [(-1.0, 1.0), (-1.0, 1.0)]
        
        # This would require modifying multi_start_optimize to accept method parameter
        # For now, just test that it works with default method
        result = multi_start_optimize(
            surrogate=self.surrogate,
            bounds=bounds,
            n_starts=5
        )
        
        assert result["n_successful"] > 0


class TestBenchmarkSurrogate:
    """Test benchmark_surrogate function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.surrogate_model = MockSurrogate()
        
    def test_single_function_benchmark(self):
        """Test benchmarking on single function."""
        with patch('builtins.print'):  # Suppress print statements
            results = benchmark_surrogate(
                surrogate_model=self.surrogate_model,
                test_functions=["sphere"],
                dimensions=[2],
                n_trials=3,
                n_train_samples=20,
                seed=42
            )
        
        assert "function_results" in results
        assert "summary" in results
        assert "sphere" in results["function_results"]
        assert "dim_2" in results["function_results"]["sphere"]
        
        sphere_results = results["function_results"]["sphere"]["dim_2"]
        assert "mean_gap" in sphere_results
        assert "std_gap" in sphere_results
        assert "mean_grad_error" in sphere_results
        assert "trials" in sphere_results
        assert len(sphere_results["trials"]) == 3
        
    def test_multiple_functions_benchmark(self):
        """Test benchmarking on multiple functions."""
        with patch('builtins.print'):
            results = benchmark_surrogate(
                surrogate_model=self.surrogate_model,
                test_functions=["sphere", "rosenbrock"],
                dimensions=[2, 5],
                n_trials=2,
                n_train_samples=15,
                seed=123
            )
        
        assert len(results["function_results"]) == 2
        
        for func_name in ["sphere", "rosenbrock"]:
            assert func_name in results["function_results"]
            func_results = results["function_results"][func_name]
            
            for dim in [2, 5]:
                dim_key = f"dim_{dim}"
                assert dim_key in func_results
                assert len(func_results[dim_key]["trials"]) == 2
        
        # Summary should aggregate all results
        summary = results["summary"]
        expected_total_trials = 2 * 2 * 2  # 2 functions * 2 dimensions * 2 trials
        assert summary["n_trials_total"] == expected_total_trials
        
    def test_benchmark_metrics(self):
        """Test that benchmark computes meaningful metrics."""
        with patch('builtins.print'):
            results = benchmark_surrogate(
                surrogate_model=self.surrogate_model,
                test_functions=["sphere"],
                dimensions=[2],
                n_trials=5,
                seed=456
            )
        
        trial_results = results["function_results"]["sphere"]["dim_2"]["trials"]
        
        for trial in trial_results:
            assert "optimality_gap" in trial
            assert "gradient_error" in trial
            assert "surrogate_optimum" in trial
            assert "true_optimum" in trial
            assert "optimization_success" in trial
            
            # Metrics should be reasonable
            assert trial["optimality_gap"] >= 0
            assert trial["gradient_error"] >= 0
            assert isinstance(trial["optimization_success"], bool)
            
    def test_unknown_function_handling(self):
        """Test handling of unknown test functions."""
        with patch('builtins.print') as mock_print:
            results = benchmark_surrogate(
                surrogate_model=self.surrogate_model,
                test_functions=["unknown_function", "sphere"],
                dimensions=[2],
                n_trials=1
            )
        
        # Should print warning about unknown function
        mock_print.assert_called()
        
        # Should still process known functions
        assert "sphere" in results["function_results"]


class TestValidateSurrogate:
    """Test validate_surrogate function."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create surrogate that approximates sphere function reasonably well
        self.surrogate = MockSurrogate(lambda x: -np.sum(np.asarray(x)**2))
        self.test_function = lambda x: -np.sum(np.asarray(x)**2)  # Same as surrogate
        
    def test_perfect_surrogate_validation(self):
        """Test validation with perfect surrogate."""
        bounds = [(-2.0, 2.0), (-2.0, 2.0)]
        
        results = validate_surrogate(
            surrogate=self.surrogate,
            test_function=self.test_function,
            bounds=bounds,
            n_test_points=50,
            seed=42
        )
        
        assert "mse" in results
        assert "mae" in results
        assert "r2" in results
        assert "mean_gradient_error" in results
        assert "std_gradient_error" in results
        assert "n_test_points" in results
        assert "n_gradient_tests" in results
        
        # Perfect surrogate should have near-zero errors
        assert results["mse"] < 1e-10
        assert results["mae"] < 1e-10
        assert results["r2"] > 0.99
        assert results["mean_gradient_error"] < 1e-5
        
    def test_imperfect_surrogate_validation(self):
        """Test validation with imperfect surrogate."""
        # Different functions for surrogate and test
        def noisy_function(x):
            return -np.sum(np.asarray(x)**2) + 0.1 * np.sin(10 * np.sum(x))
        
        surrogate = MockSurrogate(lambda x: -np.sum(np.asarray(x)**2))  # No noise
        
        bounds = [(-1.0, 1.0), (-1.0, 1.0)]
        
        results = validate_surrogate(
            surrogate=surrogate,
            test_function=noisy_function,
            bounds=bounds,
            n_test_points=30,
            seed=123
        )
        
        # Should have some error due to mismatch
        assert results["mse"] > 0
        assert results["mae"] > 0
        assert results["r2"] < 1.0
        assert results["mean_gradient_error"] > 0
        
    def test_validation_bounds(self):
        """Test that validation respects bounds."""
        bounds = [(-1.0, 1.0), (-1.0, 1.0)]
        
        results = validate_surrogate(
            surrogate=self.surrogate,
            test_function=self.test_function,
            bounds=bounds,
            n_test_points=100,
            seed=789
        )
        
        assert results["n_test_points"] == 100
        assert results["n_gradient_tests"] <= 100


class TestTestFunctions:
    """Test standard test functions."""

    def test_rosenbrock(self):
        """Test Rosenbrock function."""
        # Test known minimum at (1, 1)
        x_min = np.array([1.0, 1.0])
        value_min = rosenbrock(x_min)
        
        # Rosenbrock minimum should be 0 (negated to -0)
        assert abs(value_min) < 1e-10
        
        # Test another point
        x_test = np.array([0.0, 0.0])
        value_test = rosenbrock(x_test)
        assert value_test < value_min  # Should be worse than minimum
        
    def test_rastrigin(self):
        """Test Rastrigin function."""
        # Test known minimum at origin
        x_min = np.array([0.0, 0.0])
        value_min = rastrigin(x_min)
        
        # Rastrigin minimum should be 0 (negated to -0)
        assert abs(value_min) < 1e-10
        
        # Test another point
        x_test = np.array([1.0, 1.0])
        value_test = rastrigin(x_test)
        assert value_test < value_min
        
    def test_ackley(self):
        """Test Ackley function."""
        # Test known minimum at origin
        x_min = np.array([0.0, 0.0])
        value_min = ackley(x_min)
        
        # Ackley minimum should be 0 (negated to -0)
        assert abs(value_min) < 1e-10
        
        # Test another point
        x_test = np.array([2.0, 2.0])
        value_test = ackley(x_test)
        assert value_test < value_min
        
    def test_sphere(self):
        """Test Sphere function."""
        # Test known minimum at origin
        x_min = np.array([0.0, 0.0])
        value_min = sphere(x_min)
        
        # Sphere minimum should be 0 (negated to -0)
        assert abs(value_min) < 1e-10
        
        # Test another point
        x_test = np.array([1.0, 1.0])
        value_test = sphere(x_test)
        assert value_test < value_min
        
    def test_griewank(self):
        """Test Griewank function."""
        # Test known minimum at origin
        x_min = np.array([0.0, 0.0])
        value_min = griewank(x_min)
        
        # Griewank minimum should be 0 (negated to -0)
        assert abs(value_min) < 1e-10
        
        # Test another point
        x_test = np.array([5.0, 5.0])
        value_test = griewank(x_test)
        assert value_test < value_min
        
    def test_functions_different_dimensions(self):
        """Test that functions work with different dimensions."""
        test_functions = [rosenbrock, rastrigin, ackley, sphere, griewank]
        dimensions = [1, 2, 3, 5, 10]
        
        for func in test_functions:
            for dim in dimensions:
                x = np.zeros(dim)
                
                if func == rosenbrock and dim == 1:
                    # Rosenbrock needs at least 2 dimensions
                    continue
                    
                try:
                    value = func(x)
                    assert isinstance(value, (float, np.float64))
                    assert not np.isnan(value)
                    assert not np.isinf(value)
                except Exception as e:
                    pytest.fail(f"Function {func.__name__} failed for dim {dim}: {e}")


class TestCreateBenchmarkReport:
    """Test create_benchmark_report function."""

    def test_create_report(self):
        """Test benchmark report creation."""
        # Create mock benchmark results
        results = {
            "summary": {
                "mean_gap": 0.05,
                "std_gap": 0.02,
                "mean_grad_error": 0.1,
                "std_grad_error": 0.05,
                "n_trials_total": 6
            },
            "function_results": {
                "sphere": {
                    "dim_2": {
                        "mean_gap": 0.03,
                        "std_gap": 0.01,
                        "mean_grad_error": 0.08,
                        "std_grad_error": 0.03
                    },
                    "dim_5": {
                        "mean_gap": 0.07,
                        "std_gap": 0.03,
                        "mean_grad_error": 0.12,
                        "std_grad_error": 0.07
                    }
                }
            }
        }
        
        report = create_benchmark_report(results)
        
        assert isinstance(report, str)
        assert "Surrogate Model Benchmark Report" in report
        assert "Summary" in report
        assert "Detailed Results" in report
        assert "Sphere" in report
        assert "0.05" in report  # Mean gap
        assert "6" in report     # Total trials
        
    def test_create_report_with_file(self):
        """Test creating report and saving to file."""
        results = {
            "summary": {
                "mean_gap": 0.1,
                "std_gap": 0.05,
                "mean_grad_error": 0.2,
                "std_grad_error": 0.1,
                "n_trials_total": 4
            },
            "function_results": {
                "rosenbrock": {
                    "dim_2": {
                        "mean_gap": 0.1,
                        "std_gap": 0.05,
                        "mean_grad_error": 0.2,
                        "std_grad_error": 0.1
                    }
                }
            }
        }
        
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.md') as f:
            temp_filename = f.name
        
        try:
            with patch('builtins.print'):
                report = create_benchmark_report(results, temp_filename)
            
            # Check file was created
            assert os.path.exists(temp_filename)
            
            # Check file contents
            with open(temp_filename, 'r') as f:
                file_content = f.read()
            
            assert file_content == report
            assert "Rosenbrock" in file_content
            
        finally:
            # Clean up
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)


class TestUtilsIntegration:
    """Integration tests for utility functions."""

    def test_optimization_pipeline(self):
        """Test complete optimization pipeline."""
        # Create surrogate that approximates Rosenbrock function
        def approximate_rosenbrock(x):
            x = np.asarray(x)
            # Simplified approximation
            return -(x[0] - 1)**2 - 100*(x[1] - x[0]**2)**2
        
        surrogate = MockSurrogate(approximate_rosenbrock)
        
        # Single optimization
        single_result = optimize_with_surrogate(
            surrogate=surrogate,
            x0=np.array([0.0, 0.0]),
            bounds=[(-2.0, 2.0), (-2.0, 2.0)]
        )
        
        # Multi-start optimization
        multi_result = multi_start_optimize(
            surrogate=surrogate,
            bounds=[(-2.0, 2.0), (-2.0, 2.0)],
            n_starts=10,
            seed=42
        )
        
        # Both should find reasonable solutions
        assert single_result["success"] or multi_result["n_successful"] > 0
        
        # Multi-start should generally be better or equal
        if single_result["success"] and multi_result["n_successful"] > 0:
            assert multi_result["fun"] >= single_result["fun"] - 0.1
            
    def test_benchmark_and_validation_consistency(self):
        """Test that benchmark and validation give consistent results."""
        surrogate = MockSurrogate(lambda x: -np.sum(np.asarray(x)**2))
        
        # Benchmark on sphere function
        with patch('builtins.print'):
            benchmark_results = benchmark_surrogate(
                surrogate_model=surrogate,
                test_functions=["sphere"],
                dimensions=[2],
                n_trials=3,
                n_train_samples=30,
                seed=42
            )
        
        # Validate on sphere function
        validation_results = validate_surrogate(
            surrogate=surrogate,
            test_function=sphere,
            bounds=[(-2.0, 2.0), (-2.0, 2.0)],
            n_test_points=50,
            seed=42
        )
        
        # Both should indicate good performance for matching functions
        sphere_results = benchmark_results["function_results"]["sphere"]["dim_2"]
        
        # Should have low errors in both cases
        assert sphere_results["mean_gap"] < 0.1
        assert validation_results["r2"] > 0.8


@pytest.fixture
def mock_surrogate():
    """Mock surrogate fixture."""
    return MockSurrogate()


@pytest.fixture
def sample_benchmark_results():
    """Sample benchmark results for testing."""
    return {
        "summary": {
            "mean_gap": 0.05,
            "std_gap": 0.02,
            "mean_grad_error": 0.1,
            "std_grad_error": 0.05,
            "n_trials_total": 4
        },
        "function_results": {
            "sphere": {
                "dim_2": {
                    "mean_gap": 0.03,
                    "std_gap": 0.01,
                    "mean_grad_error": 0.08,
                    "std_grad_error": 0.03,
                    "trials": [
                        {"optimality_gap": 0.02, "gradient_error": 0.07},
                        {"optimality_gap": 0.04, "gradient_error": 0.09}
                    ]
                }
            }
        }
    }