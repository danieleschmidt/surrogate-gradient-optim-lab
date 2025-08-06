"""Performance regression testing utilities."""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jax.numpy as jnp

from ..monitoring.logging import get_logger
from ..performance.profiling import BenchmarkSuite


class PerformanceBaseline:
    """Store and manage performance baselines."""
    
    def __init__(self, baseline_file: str = "performance_baseline.json"):
        """Initialize performance baseline manager.
        
        Args:
            baseline_file: File to store baseline data
        """
        self.baseline_file = Path(baseline_file)
        self.logger = get_logger()
        self.baselines = self._load_baselines()
    
    def _load_baselines(self) -> Dict[str, Any]:
        """Load existing baselines from file."""
        if self.baseline_file.exists():
            try:
                with open(self.baseline_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load baselines: {e}")
        
        return {
            "metadata": {
                "created": datetime.now().isoformat(),
                "version": "1.0",
            },
            "baselines": {},
        }
    
    def _save_baselines(self):
        """Save baselines to file."""
        try:
            self.baseline_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.baseline_file, 'w') as f:
                json.dump(self.baselines, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save baselines: {e}")
    
    def set_baseline(self, test_name: str, metrics: Dict[str, float], metadata: Optional[Dict] = None):
        """Set performance baseline for a test.
        
        Args:
            test_name: Name of the test
            metrics: Performance metrics to baseline
            metadata: Optional metadata about the baseline
        """
        self.baselines["baselines"][test_name] = {
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        
        self._save_baselines()
        self.logger.info(f"Set performance baseline for {test_name}")
    
    def get_baseline(self, test_name: str) -> Optional[Dict[str, Any]]:
        """Get performance baseline for a test.
        
        Args:
            test_name: Name of the test
            
        Returns:
            Baseline data or None if not found
        """
        return self.baselines["baselines"].get(test_name)
    
    def compare_to_baseline(
        self,
        test_name: str,
        current_metrics: Dict[str, float],
        tolerance: float = 0.1,
    ) -> Dict[str, Any]:
        """Compare current metrics to baseline.
        
        Args:
            test_name: Name of the test
            current_metrics: Current performance metrics
            tolerance: Allowed regression tolerance (0.1 = 10%)
            
        Returns:
            Comparison results
        """
        baseline = self.get_baseline(test_name)
        if not baseline:
            return {
                "has_baseline": False,
                "message": f"No baseline found for {test_name}",
                "regression_detected": False,
            }
        
        baseline_metrics = baseline["metrics"]
        regression_details = []
        improvement_details = []
        regression_detected = False
        
        for metric_name, current_value in current_metrics.items():
            if metric_name not in baseline_metrics:
                continue
            
            baseline_value = baseline_metrics[metric_name]
            
            # Calculate relative change (negative = improvement for time metrics)
            if baseline_value > 0:
                relative_change = (current_value - baseline_value) / baseline_value
            else:
                relative_change = 0.0
            
            # For time metrics, positive change is regression
            # For accuracy metrics (like R²), negative change is regression
            is_time_metric = any(keyword in metric_name.lower() 
                               for keyword in ['time', 'duration', 'latency'])
            
            if is_time_metric:
                is_regression = relative_change > tolerance
                is_improvement = relative_change < -tolerance
            else:
                # Accuracy metrics - lower is worse
                is_regression = relative_change < -tolerance
                is_improvement = relative_change > tolerance
            
            if is_regression:
                regression_detected = True
                regression_details.append({
                    "metric": metric_name,
                    "baseline": baseline_value,
                    "current": current_value,
                    "relative_change": relative_change,
                    "threshold": tolerance,
                })
            elif is_improvement:
                improvement_details.append({
                    "metric": metric_name,
                    "baseline": baseline_value,
                    "current": current_value,
                    "relative_change": relative_change,
                })
        
        return {
            "has_baseline": True,
            "regression_detected": regression_detected,
            "regressions": regression_details,
            "improvements": improvement_details,
            "baseline_timestamp": baseline["timestamp"],
            "comparison_timestamp": datetime.now().isoformat(),
        }


class RegressionTest:
    """Individual regression test."""
    
    def __init__(
        self,
        name: str,
        test_function: callable,
        baseline_metrics: List[str],
        setup_function: Optional[callable] = None,
        teardown_function: Optional[callable] = None,
    ):
        """Initialize regression test.
        
        Args:
            name: Test name
            test_function: Function that runs the test and returns metrics
            baseline_metrics: List of metric names to track for regression
            setup_function: Optional setup function
            teardown_function: Optional teardown function
        """
        self.name = name
        self.test_function = test_function
        self.baseline_metrics = baseline_metrics
        self.setup_function = setup_function
        self.teardown_function = teardown_function
        self.logger = get_logger()
    
    def run(self) -> Dict[str, Any]:
        """Run the regression test.
        
        Returns:
            Test results including metrics
        """
        start_time = time.time()
        
        try:
            # Setup
            if self.setup_function:
                self.setup_function()
            
            # Run test
            result = self.test_function()
            
            # Extract baseline metrics
            baseline_metrics = {}
            for metric_name in self.baseline_metrics:
                if metric_name in result:
                    baseline_metrics[metric_name] = result[metric_name]
            
            return {
                "success": True,
                "execution_time": time.time() - start_time,
                "metrics": baseline_metrics,
                "full_result": result,
            }
            
        except Exception as e:
            self.logger.error(f"Regression test {self.name} failed: {e}")
            return {
                "success": False,
                "execution_time": time.time() - start_time,
                "error": str(e),
                "metrics": {},
            }
        
        finally:
            # Teardown
            if self.teardown_function:
                try:
                    self.teardown_function()
                except Exception as e:
                    self.logger.warning(f"Teardown failed for {self.name}: {e}")


class PerformanceRegressionTester:
    """Test suite for performance regression testing."""
    
    def __init__(
        self,
        baseline_file: str = "performance_baseline.json",
        regression_tolerance: float = 0.1,
    ):
        """Initialize regression tester.
        
        Args:
            baseline_file: File to store baselines
            regression_tolerance: Regression tolerance (0.1 = 10%)
        """
        self.baseline_manager = PerformanceBaseline(baseline_file)
        self.regression_tolerance = regression_tolerance
        self.tests = []
        self.logger = get_logger()
    
    def add_test(self, test: RegressionTest):
        """Add a regression test.
        
        Args:
            test: Regression test to add
        """
        self.tests.append(test)
    
    def run_tests(
        self,
        update_baselines: bool = False,
        test_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run regression tests.
        
        Args:
            update_baselines: Whether to update baselines with current results
            test_names: Optional list of specific tests to run
            
        Returns:
            Test results
        """
        self.logger.info(f"Running {len(self.tests)} regression tests...")
        
        test_results = []
        regressions_detected = []
        
        for test in self.tests:
            if test_names and test.name not in test_names:
                continue
            
            self.logger.info(f"Running regression test: {test.name}")
            
            # Run test
            result = test.run()
            
            if result["success"]:
                # Compare to baseline
                comparison = self.baseline_manager.compare_to_baseline(
                    test.name,
                    result["metrics"],
                    self.regression_tolerance
                )
                
                result["comparison"] = comparison
                
                if comparison["regression_detected"]:
                    regressions_detected.append(test.name)
                    self.logger.warning(f"Regression detected in {test.name}")
                    
                    for regression in comparison["regressions"]:
                        self.logger.warning(
                            f"  {regression['metric']}: "
                            f"{regression['baseline']:.6f} → {regression['current']:.6f} "
                            f"({regression['relative_change']:.1%} change)"
                        )
                
                # Update baseline if requested
                if update_baselines:
                    self.baseline_manager.set_baseline(
                        test.name,
                        result["metrics"],
                        {"test_timestamp": datetime.now().isoformat()}
                    )
            
            test_results.append({
                "test_name": test.name,
                "result": result,
            })
        
        summary = {
            "tests_run": len(test_results),
            "tests_passed": sum(1 for r in test_results if r["result"]["success"]),
            "regressions_detected": len(regressions_detected),
            "regressed_tests": regressions_detected,
            "test_results": test_results,
            "overall_pass": len(regressions_detected) == 0,
        }
        
        if summary["overall_pass"]:
            self.logger.info(f"All regression tests passed ({summary['tests_run']} tests)")
        else:
            self.logger.error(f"Regression tests failed: {len(regressions_detected)} regressions detected")
        
        return summary
    
    def create_standard_tests(self):
        """Create standard regression tests for surrogate optimization."""
        
        # Data collection performance test
        def test_data_collection():
            from ..data.collector import collect_data
            
            def simple_function(x):
                return float(jnp.sum(x**2))
            
            bounds = [(-2.0, 2.0), (-2.0, 2.0)]
            
            start_time = time.time()
            dataset = collect_data(simple_function, 100, bounds, verbose=False)
            collection_time = time.time() - start_time
            
            return {
                "data_collection_time": collection_time,
                "samples_per_second": 100 / collection_time,
                "dataset_size": dataset.n_samples,
            }
        
        self.add_test(RegressionTest(
            "data_collection_performance",
            test_data_collection,
            ["data_collection_time", "samples_per_second"]
        ))
        
        # Neural surrogate training test
        def test_neural_surrogate_training():
            from ..models.neural import NeuralSurrogate
            from ..models.base import Dataset
            import jax.random
            
            # Create synthetic dataset
            key = jax.random.PRNGKey(42)
            X = jax.random.normal(key, (200, 3))
            y = jnp.sum(X**2, axis=1)
            dataset = Dataset(X=X, y=y)
            
            surrogate = NeuralSurrogate(n_epochs=50, random_seed=42)
            
            start_time = time.time()
            surrogate.fit(dataset)
            training_time = time.time() - start_time
            
            # Test prediction time
            test_X = X[:10]
            pred_start = time.time()
            predictions = surrogate.predict(test_X)
            pred_time = time.time() - pred_start
            
            return {
                "neural_training_time": training_time,
                "neural_prediction_time": pred_time,
                "predictions_per_second": len(test_X) / pred_time,
            }
        
        self.add_test(RegressionTest(
            "neural_surrogate_performance",
            test_neural_surrogate_training,
            ["neural_training_time", "neural_prediction_time"]
        ))
        
        # GP surrogate training test
        def test_gp_surrogate_training():
            from ..models.gaussian_process import GPSurrogate
            from ..models.base import Dataset
            import jax.random
            
            # Create synthetic dataset (smaller for GP)
            key = jax.random.PRNGKey(42)
            X = jax.random.normal(key, (50, 2))
            y = jnp.sum(X**2, axis=1)
            dataset = Dataset(X=X, y=y)
            
            surrogate = GPSurrogate()
            
            start_time = time.time()
            surrogate.fit(dataset)
            training_time = time.time() - start_time
            
            # Test prediction time
            test_X = X[:5]
            pred_start = time.time()
            predictions = surrogate.predict(test_X)
            pred_time = time.time() - pred_start
            
            return {
                "gp_training_time": training_time,
                "gp_prediction_time": pred_time,
                "gp_predictions_per_second": len(test_X) / pred_time,
            }
        
        self.add_test(RegressionTest(
            "gp_surrogate_performance",
            test_gp_surrogate_training,
            ["gp_training_time", "gp_prediction_time"]
        ))
        
        # Optimization performance test
        def test_optimization_performance():
            from ..optimizers.gradient_descent import GradientDescentOptimizer
            from ..models.neural import NeuralSurrogate
            from ..models.base import Dataset
            import jax.random
            
            # Setup
            key = jax.random.PRNGKey(42)
            X = jax.random.normal(key, (100, 2))
            y = jnp.sum((X - 0.5)**2, axis=1)  # Minimum at [0.5, 0.5]
            dataset = Dataset(X=X, y=y)
            
            surrogate = NeuralSurrogate(n_epochs=20, random_seed=42)
            surrogate.fit(dataset)
            
            optimizer = GradientDescentOptimizer(max_iterations=50)
            
            start_time = time.time()
            result = optimizer.optimize(surrogate, jnp.array([0.0, 0.0]))
            optimization_time = time.time() - start_time
            
            return {
                "optimization_time": optimization_time,
                "optimization_success": result.success,
                "final_function_value": float(result.fun),
                "iterations_used": result.nit,
            }
        
        self.add_test(RegressionTest(
            "optimization_performance",
            test_optimization_performance,
            ["optimization_time", "iterations_used"]
        ))
    
    def generate_regression_report(self, results: Dict[str, Any]) -> str:
        """Generate regression test report.
        
        Args:
            results: Results from run_tests
            
        Returns:
            Formatted report
        """
        report = []
        report.append("PERFORMANCE REGRESSION TEST REPORT")
        report.append("=" * 50)
        
        # Summary
        overall_status = "PASS" if results["overall_pass"] else "FAIL"
        report.append(f"\nOVERALL STATUS: {overall_status}")
        report.append(f"Tests Run: {results['tests_run']}")
        report.append(f"Tests Passed: {results['tests_passed']}")
        report.append(f"Regressions Detected: {results['regressions_detected']}")
        
        if results["regressed_tests"]:
            report.append(f"Regressed Tests: {', '.join(results['regressed_tests'])}")
        
        # Detailed results
        report.append("\nDETAILED RESULTS:")
        report.append("-" * 30)
        
        for test_result in results["test_results"]:
            test_name = test_result["test_name"]
            result = test_result["result"]
            
            status = "PASS" if result["success"] else "FAIL"
            report.append(f"\n{test_name}: {status}")
            
            if result["success"]:
                report.append(f"  Execution time: {result['execution_time']:.3f}s")
                
                # Show key metrics
                for metric, value in result["metrics"].items():
                    if isinstance(value, float):
                        report.append(f"  {metric}: {value:.6f}")
                    else:
                        report.append(f"  {metric}: {value}")
                
                # Show regression analysis
                comparison = result.get("comparison", {})
                if comparison.get("has_baseline"):
                    if comparison.get("regression_detected"):
                        report.append("  REGRESSION DETECTED:")
                        for regression in comparison["regressions"]:
                            change_pct = regression["relative_change"] * 100
                            report.append(
                                f"    {regression['metric']}: "
                                f"{regression['baseline']:.6f} → {regression['current']:.6f} "
                                f"({change_pct:+.1f}%)"
                            )
                    
                    if comparison.get("improvements"):
                        report.append("  Improvements:")
                        for improvement in comparison["improvements"]:
                            change_pct = improvement["relative_change"] * 100
                            report.append(
                                f"    {improvement['metric']}: "
                                f"{improvement['baseline']:.6f} → {improvement['current']:.6f} "
                                f"({change_pct:+.1f}%)"
                            )
                else:
                    report.append("  No baseline available for comparison")
            else:
                report.append(f"  Error: {result.get('error', 'Unknown error')}")
        
        return "\n".join(report)
    
    def get_performance_trends(self, test_name: str, n_runs: int = 5) -> Dict[str, Any]:
        """Analyze performance trends for a test.
        
        Args:
            test_name: Name of test to analyze
            n_runs: Number of runs to analyze trend
            
        Returns:
            Trend analysis
        """
        # This would require storing historical data
        # For now, return placeholder
        return {
            "test_name": test_name,
            "trend_analysis": "Not implemented - requires historical data storage",
            "n_runs": n_runs,
        }