"""Comprehensive testing framework for surrogate optimization."""

import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import json

import jax.numpy as jnp
from jax import Array


@dataclass
class TestCase:
    """Represents a single test case."""
    test_id: str
    test_name: str
    description: str
    test_function: Callable
    expected_result: Any = None
    tolerance: float = 1e-6
    timeout: float = 30.0
    requires_gpu: bool = False
    requires_large_memory: bool = False
    tags: List[str] = field(default_factory=list)


@dataclass
class TestResult:
    """Result of a test execution."""
    test_id: str
    test_name: str
    passed: bool
    execution_time: float
    error_message: Optional[str] = None
    actual_result: Any = None
    expected_result: Any = None
    tolerance: float = 1e-6
    metadata: Dict[str, Any] = field(default_factory=dict)


class SurrogateTestSuite:
    """Comprehensive test suite for surrogate optimization components."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize test suite.
        
        Args:
            output_dir: Directory for test output files
        """
        self.output_dir = Path(output_dir) if output_dir else Path("test_results")
        self.output_dir.mkdir(exist_ok=True)
        
        self.test_cases = {}
        self.test_results = []
        self.setup_functions = []
        self.teardown_functions = []
    
    def add_test(self, test_case: TestCase):
        """Add a test case to the suite."""
        self.test_cases[test_case.test_id] = test_case
    
    def add_setup(self, setup_func: Callable):
        """Add a setup function to run before tests."""
        self.setup_functions.append(setup_func)
    
    def add_teardown(self, teardown_func: Callable):
        """Add a teardown function to run after tests."""
        self.teardown_functions.append(teardown_func)
    
    def run_test(self, test_case: TestCase) -> TestResult:
        """Run a single test case.
        
        Args:
            test_case: Test case to run
            
        Returns:
            Test result
        """
        start_time = time.time()
        
        try:
            # Execute test function
            actual_result = test_case.test_function()
            execution_time = time.time() - start_time
            
            # Check result
            if test_case.expected_result is not None:
                passed = self._compare_results(
                    actual_result,
                    test_case.expected_result,
                    test_case.tolerance
                )
                error_message = None if passed else f"Expected {test_case.expected_result}, got {actual_result}"
            else:
                # If no expected result, assume test passes if no exception
                passed = True
                error_message = None
            
            return TestResult(
                test_id=test_case.test_id,
                test_name=test_case.test_name,
                passed=passed,
                execution_time=execution_time,
                error_message=error_message,
                actual_result=actual_result,
                expected_result=test_case.expected_result,
                tolerance=test_case.tolerance
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_id=test_case.test_id,
                test_name=test_case.test_name,
                passed=False,
                execution_time=execution_time,
                error_message=str(e),
                metadata={"traceback": traceback.format_exc()}
            )
    
    def run_all_tests(
        self,
        filter_tags: Optional[List[str]] = None,
        exclude_tags: Optional[List[str]] = None,
        parallel: bool = False
    ) -> Dict[str, TestResult]:
        """Run all test cases.
        
        Args:
            filter_tags: Only run tests with these tags
            exclude_tags: Exclude tests with these tags
            parallel: Whether to run tests in parallel
            
        Returns:
            Dictionary of test results
        """
        # Filter tests
        tests_to_run = []
        for test_case in self.test_cases.values():
            # Check filter tags
            if filter_tags:
                if not any(tag in test_case.tags for tag in filter_tags):
                    continue
            
            # Check exclude tags
            if exclude_tags:
                if any(tag in test_case.tags for tag in exclude_tags):
                    continue
            
            tests_to_run.append(test_case)
        
        # Run setup functions
        for setup_func in self.setup_functions:
            setup_func()
        
        results = {}
        
        try:
            if parallel:
                results = self._run_tests_parallel(tests_to_run)
            else:
                results = self._run_tests_sequential(tests_to_run)
        finally:
            # Run teardown functions
            for teardown_func in self.teardown_functions:
                teardown_func()
        
        self.test_results.extend(results.values())
        return results
    
    def _run_tests_sequential(self, test_cases: List[TestCase]) -> Dict[str, TestResult]:
        """Run tests sequentially."""
        results = {}
        for test_case in test_cases:
            print(f"Running test: {test_case.test_name}")
            result = self.run_test(test_case)
            results[test_case.test_id] = result
            
            status = "PASS" if result.passed else "FAIL"
            print(f"  {status} ({result.execution_time:.3f}s)")
            if not result.passed:
                print(f"  Error: {result.error_message}")
        
        return results
    
    def _run_tests_parallel(self, test_cases: List[TestCase]) -> Dict[str, TestResult]:
        """Run tests in parallel."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        results = {}
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all tests
            future_to_test = {
                executor.submit(self.run_test, test_case): test_case
                for test_case in test_cases
            }
            
            # Collect results
            for future in as_completed(future_to_test):
                test_case = future_to_test[future]
                try:
                    result = future.result()
                    results[test_case.test_id] = result
                    
                    status = "PASS" if result.passed else "FAIL"
                    print(f"{status}: {test_case.test_name} ({result.execution_time:.3f}s)")
                    
                except Exception as e:
                    # Create failure result
                    result = TestResult(
                        test_id=test_case.test_id,
                        test_name=test_case.test_name,
                        passed=False,
                        execution_time=0.0,
                        error_message=f"Test execution failed: {e}"
                    )
                    results[test_case.test_id] = result
                    print(f"FAIL: {test_case.test_name} - {e}")
        
        return results
    
    def _compare_results(self, actual: Any, expected: Any, tolerance: float) -> bool:
        """Compare actual and expected results."""
        if isinstance(expected, (int, float)):
            if isinstance(actual, (int, float)):
                return abs(actual - expected) <= tolerance
            elif hasattr(actual, 'shape') and actual.shape == ():
                return abs(float(actual) - expected) <= tolerance
            else:
                return False
        
        elif hasattr(expected, 'shape'):  # JAX array or numpy array
            if not hasattr(actual, 'shape'):
                return False
            if actual.shape != expected.shape:
                return False
            return jnp.allclose(actual, expected, atol=tolerance)
        
        else:
            return actual == expected
    
    def generate_report(self, results: Dict[str, TestResult]) -> str:
        """Generate a test report.
        
        Args:
            results: Test results
            
        Returns:
            Report as string
        """
        total_tests = len(results)
        passed_tests = sum(1 for r in results.values() if r.passed)
        failed_tests = total_tests - passed_tests
        total_time = sum(r.execution_time for r in results.values())
        
        report = []
        report.append("=" * 60)
        report.append("SURROGATE OPTIMIZATION TEST REPORT")
        report.append("=" * 60)
        report.append(f"Total Tests: {total_tests}")
        report.append(f"Passed: {passed_tests}")
        report.append(f"Failed: {failed_tests}")
        report.append(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
        report.append(f"Total Time: {total_time:.3f}s")
        report.append(f"Average Time: {total_time/total_tests:.3f}s")
        report.append("")
        
        # Failed tests
        if failed_tests > 0:
            report.append("FAILED TESTS:")
            report.append("-" * 30)
            for result in results.values():
                if not result.passed:
                    report.append(f"❌ {result.test_name}")
                    report.append(f"   Error: {result.error_message}")
                    report.append("")
        
        # Passed tests
        if passed_tests > 0:
            report.append("PASSED TESTS:")
            report.append("-" * 30)
            for result in results.values():
                if result.passed:
                    report.append(f"✅ {result.test_name} ({result.execution_time:.3f}s)")
        
        return "\n".join(report)
    
    def save_results(self, results: Dict[str, TestResult], filename: Optional[str] = None):
        """Save test results to file.
        
        Args:
            results: Test results
            filename: Output filename
        """
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"test_results_{timestamp}.json"
        
        output_file = self.output_dir / filename
        
        # Convert results to serializable format
        serializable_results = {}
        for test_id, result in results.items():
            serializable_results[test_id] = {
                "test_id": result.test_id,
                "test_name": result.test_name,
                "passed": result.passed,
                "execution_time": result.execution_time,
                "error_message": result.error_message,
                "metadata": result.metadata,
            }
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Test results saved to: {output_file}")


def create_surrogate_model_tests() -> SurrogateTestSuite:
    """Create test suite for surrogate models."""
    suite = SurrogateTestSuite()
    
    # Test surrogate model initialization
    def test_neural_surrogate_init():
        from .models.neural import NeuralSurrogate
        model = NeuralSurrogate(hidden_dims=[32, 16], activation="relu")
        return model is not None
    
    suite.add_test(TestCase(
        test_id="neural_surrogate_init",
        test_name="Neural Surrogate Initialization",
        description="Test neural surrogate model initialization",
        test_function=test_neural_surrogate_init,
        expected_result=True,
        tags=["surrogate", "neural", "init"]
    ))
    
    # Test surrogate model fitting
    def test_neural_surrogate_fit():
        from .models.neural import NeuralSurrogate
        from .models.base import Dataset
        
        # Create test data
        X = jnp.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        y = jnp.array([1.5, 2.5, 3.5])
        dataset = Dataset(X=X, y=y)
        
        # Fit model
        model = NeuralSurrogate(hidden_dims=[8], activation="relu")
        model.fit(dataset)
        
        # Test prediction
        pred = model.predict(X[0])
        return jnp.isfinite(pred)
    
    suite.add_test(TestCase(
        test_id="neural_surrogate_fit",
        test_name="Neural Surrogate Fitting",
        description="Test neural surrogate model fitting and prediction",
        test_function=test_neural_surrogate_fit,
        expected_result=True,
        tags=["surrogate", "neural", "fit"]
    ))
    
    # Test gradient computation
    def test_neural_surrogate_gradient():
        from .models.neural import NeuralSurrogate
        from .models.base import Dataset
        
        # Create test data
        X = jnp.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        y = jnp.array([1.5, 2.5, 3.5])
        dataset = Dataset(X=X, y=y)
        
        # Fit model
        model = NeuralSurrogate(hidden_dims=[8])
        model.fit(dataset)
        
        # Test gradient
        grad = model.gradient(X[0])
        return grad.shape == (2,) and jnp.isfinite(grad).all()
    
    suite.add_test(TestCase(
        test_id="neural_surrogate_gradient",
        test_name="Neural Surrogate Gradient",
        description="Test neural surrogate gradient computation",
        test_function=test_neural_surrogate_gradient,
        expected_result=True,
        tags=["surrogate", "neural", "gradient"]
    ))
    
    return suite


def create_optimization_tests() -> SurrogateTestSuite:
    """Create test suite for optimization algorithms."""
    suite = SurrogateTestSuite()
    
    # Test basic optimization
    def test_basic_optimization():
        from .core import quick_optimize
        import jax.numpy as jnp
        
        # Simple quadratic function
        def quadratic(x):
            return jnp.sum((x - 1)**2)
        
        try:
            result = quick_optimize(
                function=quadratic,
                bounds=[(-2, 2), (-2, 2)],
                n_samples=20,
                surrogate_type="neural_network",
                verbose=False
            )
            
            # Check if we found a reasonable solution
            error = jnp.linalg.norm(result.x - jnp.array([1.0, 1.0]))
            return error < 0.5  # Allow some tolerance
        except Exception:
            return False
    
    suite.add_test(TestCase(
        test_id="basic_optimization",
        test_name="Basic Optimization",
        description="Test basic optimization workflow",
        test_function=test_basic_optimization,
        expected_result=True,
        timeout=60.0,
        tags=["optimization", "integration"]
    ))
    
    # Test multi-start optimization
    def test_multi_start_optimization():
        from .optimizers.multi_start import MultiStartOptimizer
        from .models.neural import NeuralSurrogate
        from .models.base import Dataset
        
        # Create test surrogate
        X = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        y = jnp.array([0.0, 2.0, 8.0])  # x^2 + y^2
        dataset = Dataset(X=X, y=y)
        
        surrogate = NeuralSurrogate(hidden_dims=[8])
        surrogate.fit(dataset)
        
        # Test multi-start optimization
        optimizer = MultiStartOptimizer(n_starts=3)
        result = optimizer.optimize(
            surrogate=surrogate,
            x0=jnp.array([1.0, 1.0]),
            bounds=[(-2, 2), (-2, 2)]
        )
        
        return result.success
    
    suite.add_test(TestCase(
        test_id="multi_start_optimization",
        test_name="Multi-Start Optimization",
        description="Test multi-start optimization algorithm",
        test_function=test_multi_start_optimization,
        expected_result=True,
        tags=["optimization", "multi_start"]
    ))
    
    return suite


def create_benchmarking_tests() -> SurrogateTestSuite:
    """Create test suite for benchmarking functionality."""
    suite = SurrogateTestSuite()
    
    def test_benchmark_functions():
        import sys
        sys.path.append('/root/repo')
        from tests.fixtures.benchmark_functions import Rosenbrock
        
        # Test benchmark function
        func = Rosenbrock(2)
        x = jnp.array([1.0, 1.0])  # Global optimum
        value = func(x)
        
        # Should be close to 0 at global optimum
        return abs(value) < 1e-6
    
    suite.add_test(TestCase(
        test_id="benchmark_functions",
        test_name="Benchmark Functions",
        description="Test benchmark function implementations",
        test_function=test_benchmark_functions,
        expected_result=True,
        tags=["benchmark", "functions"]
    ))
    
    def test_quick_benchmark():
        from .benchmarks import run_quick_benchmark
        
        try:
            results = run_quick_benchmark()
            return len(results.benchmark_results) > 0
        except Exception:
            return False
    
    suite.add_test(TestCase(
        test_id="quick_benchmark",
        test_name="Quick Benchmark",
        description="Test quick benchmark execution",
        test_function=test_quick_benchmark,
        expected_result=True,
        timeout=120.0,
        tags=["benchmark", "integration", "slow"]
    ))
    
    return suite


def create_robustness_tests() -> SurrogateTestSuite:
    """Create test suite for robustness features."""
    suite = SurrogateTestSuite()
    
    def test_input_validation():
        from .security import InputValidator
        
        validator = InputValidator()
        
        # Test valid input
        valid_x = jnp.array([1.0, 2.0, 3.0])
        violations = validator.validate_array(valid_x, "test_array")
        
        # Should have no violations for valid input
        return len(violations) == 0
    
    suite.add_test(TestCase(
        test_id="input_validation",
        test_name="Input Validation",
        description="Test input validation functionality",
        test_function=test_input_validation,
        expected_result=True,
        tags=["robustness", "security"]
    ))
    
    def test_robust_surrogate():
        from .robustness import RobustSurrogate
        from .models.neural import NeuralSurrogate
        from .models.base import Dataset
        
        # Create test data
        X = jnp.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        y = jnp.array([1.5, 2.5, 3.5])
        dataset = Dataset(X=X, y=y)
        
        # Create robust surrogate
        base_model = NeuralSurrogate(hidden_dims=[8])
        robust_model = RobustSurrogate(base_model)
        
        # Fit and predict
        robust_model.fit(dataset)
        pred = robust_model.predict(X[0])
        
        return jnp.isfinite(pred)
    
    suite.add_test(TestCase(
        test_id="robust_surrogate",
        test_name="Robust Surrogate",
        description="Test robust surrogate wrapper",
        test_function=test_robust_surrogate,
        expected_result=True,
        tags=["robustness", "surrogate"]
    ))
    
    return suite


def run_comprehensive_tests(
    include_slow: bool = False,
    parallel: bool = False,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Run comprehensive test suite.
    
    Args:
        include_slow: Whether to include slow tests
        parallel: Whether to run tests in parallel
        output_dir: Output directory for results
        
    Returns:
        Test summary
    """
    print("Running comprehensive surrogate optimization tests...")
    
    # Create test suites
    suites = [
        create_surrogate_model_tests(),
        create_optimization_tests(),
        create_benchmarking_tests(),
        create_robustness_tests(),
    ]
    
    all_results = {}
    
    for i, suite in enumerate(suites):
        print(f"\n--- Running Test Suite {i+1}/{len(suites)} ---")
        
        # Configure test filters
        exclude_tags = []
        if not include_slow:
            exclude_tags.append("slow")
        
        # Run tests
        results = suite.run_all_tests(
            exclude_tags=exclude_tags,
            parallel=parallel
        )
        
        all_results.update(results)
        
        # Generate and print report
        report = suite.generate_report(results)
        print(report)
        
        # Save results
        if output_dir:
            suite.output_dir = Path(output_dir)
            suite.save_results(results, f"suite_{i+1}_results.json")
    
    # Overall summary
    total_tests = len(all_results)
    passed_tests = sum(1 for r in all_results.values() if r.passed)
    
    summary = {
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "failed_tests": total_tests - passed_tests,
        "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
        "results": all_results
    }
    
    print(f"\n{'='*60}")
    print("COMPREHENSIVE TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
    
    return summary