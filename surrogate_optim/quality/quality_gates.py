"""Quality gates for automated quality assurance."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import time
from typing import Any, Dict, List, Optional

import jax.numpy as jnp

from ..monitoring.logging import get_logger
from ..validation import validate_surrogate_performance


@dataclass
class QualityResult:
    """Result of a quality check."""
    gate_name: str
    passed: bool
    score: float
    message: str
    details: Dict[str, Any]
    execution_time: float
    warnings: List[str]


class QualityGate(ABC):
    """Abstract base class for quality gates."""

    def __init__(self, name: str, threshold: float, weight: float = 1.0):
        """Initialize quality gate.
        
        Args:
            name: Name of the quality gate
            threshold: Minimum score to pass
            weight: Weight for overall quality score
        """
        self.name = name
        self.threshold = threshold
        self.weight = weight
        self.logger = get_logger()

    @abstractmethod
    def check(self, context: Dict[str, Any]) -> QualityResult:
        """Run the quality check.
        
        Args:
            context: Context containing models, data, etc.
            
        Returns:
            Quality check result
        """
        pass

    def _create_result(
        self,
        passed: bool,
        score: float,
        message: str,
        details: Dict[str, Any],
        execution_time: float,
        warnings: List[str] = None,
    ) -> QualityResult:
        """Create quality result."""
        return QualityResult(
            gate_name=self.name,
            passed=passed,
            score=score,
            message=message,
            details=details,
            execution_time=execution_time,
            warnings=warnings or [],
        )


class ModelAccuracyGate(QualityGate):
    """Quality gate for model accuracy."""

    def __init__(
        self,
        name: str = "model_accuracy",
        min_r2: float = 0.8,
        max_mse_ratio: float = 0.1,
        weight: float = 2.0,
    ):
        """Initialize model accuracy gate.
        
        Args:
            name: Gate name
            min_r2: Minimum R² score
            max_mse_ratio: Maximum MSE as ratio of target variance
            weight: Gate weight
        """
        super().__init__(name, min_r2, weight)
        self.min_r2 = min_r2
        self.max_mse_ratio = max_mse_ratio

    def check(self, context: Dict[str, Any]) -> QualityResult:
        """Check model accuracy."""
        start_time = time.time()
        warnings_list = []

        try:
            surrogate = context.get("surrogate")
            test_dataset = context.get("test_dataset")

            if not surrogate or not test_dataset:
                return self._create_result(
                    False, 0.0, "Missing surrogate or test dataset",
                    {}, time.time() - start_time
                )

            # Validate surrogate performance
            validation_results = validate_surrogate_performance(
                surrogate, test_dataset
            )

            metrics = validation_results["metrics"]
            r2_score = metrics["r2"]
            mse = metrics["mse"]

            # Calculate MSE ratio
            target_variance = float(jnp.var(test_dataset.y))
            mse_ratio = mse / target_variance if target_variance > 0 else float("inf")

            # Determine pass/fail
            r2_pass = r2_score >= self.min_r2
            mse_pass = mse_ratio <= self.max_mse_ratio
            overall_pass = r2_pass and mse_pass

            # Calculate composite score
            r2_score_norm = max(0, min(1, r2_score))
            mse_score_norm = max(0, min(1, 1 - mse_ratio))
            composite_score = (r2_score_norm + mse_score_norm) / 2

            # Messages
            if not r2_pass:
                warnings_list.append(f"R² score {r2_score:.3f} below threshold {self.min_r2}")
            if not mse_pass:
                warnings_list.append(f"MSE ratio {mse_ratio:.3f} above threshold {self.max_mse_ratio}")

            message = f"R²={r2_score:.3f}, MSE ratio={mse_ratio:.3f}"
            if overall_pass:
                message = f"PASS: {message}"
            else:
                message = f"FAIL: {message}"

            details = {
                "r2_score": r2_score,
                "mse": mse,
                "mse_ratio": mse_ratio,
                "target_variance": target_variance,
                "r2_pass": r2_pass,
                "mse_pass": mse_pass,
                "all_metrics": metrics,
            }

            return self._create_result(
                overall_pass, composite_score, message, details,
                time.time() - start_time, warnings_list
            )

        except Exception as e:
            return self._create_result(
                False, 0.0, f"Error during accuracy check: {e}",
                {"error": str(e)}, time.time() - start_time
            )


class PerformanceGate(QualityGate):
    """Quality gate for performance benchmarks."""

    def __init__(
        self,
        name: str = "performance",
        max_training_time: float = 300.0,  # 5 minutes
        max_prediction_time_per_sample: float = 0.01,  # 10ms per sample
        weight: float = 1.0,
    ):
        """Initialize performance gate.
        
        Args:
            name: Gate name
            max_training_time: Maximum training time in seconds
            max_prediction_time_per_sample: Maximum prediction time per sample
            weight: Gate weight
        """
        super().__init__(name, 1.0, weight)
        self.max_training_time = max_training_time
        self.max_prediction_time_per_sample = max_prediction_time_per_sample

    def check(self, context: Dict[str, Any]) -> QualityResult:
        """Check performance metrics."""
        start_time = time.time()
        warnings_list = []

        try:
            surrogate = context.get("surrogate")
            test_dataset = context.get("test_dataset")
            performance_data = context.get("performance_data", {})

            if not surrogate or not test_dataset:
                return self._create_result(
                    False, 0.0, "Missing surrogate or test dataset",
                    {}, time.time() - start_time
                )

            # Check training time
            training_time = performance_data.get("training_time", 0)
            training_pass = training_time <= self.max_training_time

            # Benchmark prediction time
            n_test_samples = min(100, test_dataset.n_samples)
            test_points = test_dataset.X[:n_test_samples]

            prediction_start = time.time()
            predictions = surrogate.predict(test_points)
            prediction_duration = time.time() - prediction_start

            time_per_sample = prediction_duration / n_test_samples
            prediction_pass = time_per_sample <= self.max_prediction_time_per_sample

            # Overall pass
            overall_pass = training_pass and prediction_pass

            # Calculate score
            training_score = min(1.0, self.max_training_time / max(training_time, 1e-6))
            prediction_score = min(1.0, self.max_prediction_time_per_sample / max(time_per_sample, 1e-6))
            composite_score = (training_score + prediction_score) / 2

            # Warnings
            if not training_pass:
                warnings_list.append(f"Training time {training_time:.1f}s exceeds {self.max_training_time}s")
            if not prediction_pass:
                warnings_list.append(f"Prediction time {time_per_sample*1000:.1f}ms exceeds {self.max_prediction_time_per_sample*1000:.1f}ms per sample")

            message = f"Training: {training_time:.1f}s, Prediction: {time_per_sample*1000:.1f}ms/sample"
            if overall_pass:
                message = f"PASS: {message}"
            else:
                message = f"FAIL: {message}"

            details = {
                "training_time": training_time,
                "prediction_time_total": prediction_duration,
                "prediction_time_per_sample": time_per_sample,
                "n_test_samples": n_test_samples,
                "training_pass": training_pass,
                "prediction_pass": prediction_pass,
            }

            return self._create_result(
                overall_pass, composite_score, message, details,
                time.time() - start_time, warnings_list
            )

        except Exception as e:
            return self._create_result(
                False, 0.0, f"Error during performance check: {e}",
                {"error": str(e)}, time.time() - start_time
            )


class RobustnessGate(QualityGate):
    """Quality gate for model robustness."""

    def __init__(
        self,
        name: str = "robustness",
        max_sensitivity: float = 1.0,
        perturbation_scale: float = 0.01,
        n_perturbations: int = 20,
        weight: float = 1.5,
    ):
        """Initialize robustness gate.
        
        Args:
            name: Gate name
            max_sensitivity: Maximum allowed sensitivity
            perturbation_scale: Scale of input perturbations
            n_perturbations: Number of perturbations to test
            weight: Gate weight
        """
        super().__init__(name, 1.0, weight)
        self.max_sensitivity = max_sensitivity
        self.perturbation_scale = perturbation_scale
        self.n_perturbations = n_perturbations

    def check(self, context: Dict[str, Any]) -> QualityResult:
        """Check model robustness."""
        start_time = time.time()
        warnings_list = []

        try:
            from ..validation.model_validation import validate_model_robustness

            surrogate = context.get("surrogate")
            test_dataset = context.get("test_dataset")

            if not surrogate or not test_dataset:
                return self._create_result(
                    False, 0.0, "Missing surrogate or test dataset",
                    {}, time.time() - start_time
                )

            # Select test points
            n_test_points = min(10, test_dataset.n_samples)
            test_points = test_dataset.X[:n_test_points]

            # Test robustness
            robustness_results = validate_model_robustness(
                surrogate, test_points,
                perturbation_scale=self.perturbation_scale,
                n_perturbations=self.n_perturbations
            )

            max_sensitivity = robustness_results["max_pred_sensitivity"]
            mean_sensitivity = robustness_results["mean_pred_sensitivity"]

            # Check pass/fail
            sensitivity_pass = max_sensitivity <= self.max_sensitivity

            # Calculate score
            sensitivity_score = min(1.0, self.max_sensitivity / max(max_sensitivity, 1e-6))

            # Warnings
            if not sensitivity_pass:
                warnings_list.append(f"Max sensitivity {max_sensitivity:.3f} exceeds threshold {self.max_sensitivity}")

            if mean_sensitivity > self.max_sensitivity * 0.5:
                warnings_list.append(f"Mean sensitivity {mean_sensitivity:.3f} is relatively high")

            message = f"Max sensitivity: {max_sensitivity:.3f}, Mean: {mean_sensitivity:.3f}"
            if sensitivity_pass:
                message = f"PASS: {message}"
            else:
                message = f"FAIL: {message}"

            details = {
                "max_sensitivity": max_sensitivity,
                "mean_sensitivity": mean_sensitivity,
                "perturbation_scale": self.perturbation_scale,
                "n_test_points": n_test_points,
                "full_results": robustness_results,
            }

            return self._create_result(
                sensitivity_pass, sensitivity_score, message, details,
                time.time() - start_time, warnings_list
            )

        except Exception as e:
            return self._create_result(
                False, 0.0, f"Error during robustness check: {e}",
                {"error": str(e)}, time.time() - start_time
            )


class MemoryGate(QualityGate):
    """Quality gate for memory usage."""

    def __init__(
        self,
        name: str = "memory",
        max_memory_mb: float = 1000.0,
        max_memory_increase_mb: float = 500.0,
        weight: float = 1.0,
    ):
        """Initialize memory gate.
        
        Args:
            name: Gate name
            max_memory_mb: Maximum total memory usage in MB
            max_memory_increase_mb: Maximum memory increase in MB
            weight: Gate weight
        """
        super().__init__(name, 1.0, weight)
        self.max_memory_mb = max_memory_mb
        self.max_memory_increase_mb = max_memory_increase_mb

    def check(self, context: Dict[str, Any]) -> QualityResult:
        """Check memory usage."""
        start_time = time.time()
        warnings_list = []

        try:
            from ..performance.memory import MemoryMonitor

            memory_data = context.get("memory_data")
            if not memory_data:
                # Create fresh memory check
                monitor = MemoryMonitor()
                memory_data = monitor.get_memory_usage()

            current_memory = memory_data.get("rss_mb", 0)
            memory_increase = memory_data.get("increase_mb", 0)

            # Check thresholds
            memory_pass = current_memory <= self.max_memory_mb
            increase_pass = memory_increase <= self.max_memory_increase_mb
            overall_pass = memory_pass and increase_pass

            # Calculate score
            memory_score = min(1.0, self.max_memory_mb / max(current_memory, 1))
            increase_score = min(1.0, self.max_memory_increase_mb / max(abs(memory_increase), 1))
            composite_score = (memory_score + increase_score) / 2

            # Warnings
            if not memory_pass:
                warnings_list.append(f"Memory usage {current_memory:.1f}MB exceeds {self.max_memory_mb}MB")
            if not increase_pass:
                warnings_list.append(f"Memory increase {memory_increase:.1f}MB exceeds {self.max_memory_increase_mb}MB")

            message = f"Usage: {current_memory:.1f}MB, Increase: {memory_increase:.1f}MB"
            if overall_pass:
                message = f"PASS: {message}"
            else:
                message = f"FAIL: {message}"

            details = {
                "current_memory_mb": current_memory,
                "memory_increase_mb": memory_increase,
                "memory_pass": memory_pass,
                "increase_pass": increase_pass,
                "full_memory_data": memory_data,
            }

            return self._create_result(
                overall_pass, composite_score, message, details,
                time.time() - start_time, warnings_list
            )

        except Exception as e:
            return self._create_result(
                False, 0.0, f"Error during memory check: {e}",
                {"error": str(e)}, time.time() - start_time
            )


class AutomatedQualityChecker:
    """Automated quality checker with configurable gates."""

    def __init__(
        self,
        gates: Optional[List[QualityGate]] = None,
        fail_fast: bool = False,
        min_overall_score: float = 0.7,
    ):
        """Initialize automated quality checker.
        
        Args:
            gates: List of quality gates (default gates if None)
            fail_fast: Whether to stop on first failure
            min_overall_score: Minimum weighted average score to pass
        """
        self.gates = gates or self._get_default_gates()
        self.fail_fast = fail_fast
        self.min_overall_score = min_overall_score
        self.logger = get_logger()

    def _get_default_gates(self) -> List[QualityGate]:
        """Get default quality gates."""
        return [
            ModelAccuracyGate(),
            PerformanceGate(),
            RobustnessGate(),
            MemoryGate(),
        ]

    def run_quality_checks(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run all quality checks.
        
        Args:
            context: Context with surrogate, datasets, etc.
            
        Returns:
            Quality check results
        """
        self.logger.info(f"Running {len(self.gates)} quality gates...")

        results = []
        failed_gates = []
        total_weighted_score = 0.0
        total_weight = 0.0

        for gate in self.gates:
            self.logger.info(f"Running quality gate: {gate.name}")

            try:
                result = gate.check(context)
                results.append(result)

                # Update weighted score
                total_weighted_score += result.score * gate.weight
                total_weight += gate.weight

                if not result.passed:
                    failed_gates.append(gate.name)
                    self.logger.warning(f"Quality gate FAILED: {gate.name} - {result.message}")

                    if self.fail_fast:
                        break
                else:
                    self.logger.info(f"Quality gate PASSED: {gate.name} - {result.message}")

                # Log warnings
                for warning in result.warnings:
                    self.logger.warning(f"  Warning: {warning}")

            except Exception as e:
                self.logger.error(f"Quality gate {gate.name} encountered error: {e}")
                failed_gates.append(gate.name)

                if self.fail_fast:
                    break

        # Calculate overall score
        overall_score = total_weighted_score / total_weight if total_weight > 0 else 0.0
        overall_pass = len(failed_gates) == 0 and overall_score >= self.min_overall_score

        summary = {
            "overall_pass": overall_pass,
            "overall_score": overall_score,
            "gates_passed": len(self.gates) - len(failed_gates),
            "gates_total": len(self.gates),
            "failed_gates": failed_gates,
            "gate_results": results,
            "min_required_score": self.min_overall_score,
        }

        if overall_pass:
            self.logger.info(f"Quality check PASSED: {summary['gates_passed']}/{summary['gates_total']} gates, score={overall_score:.3f}")
        else:
            self.logger.error(f"Quality check FAILED: {len(failed_gates)} failures, score={overall_score:.3f}")

        return summary

    def generate_quality_report(self, results: Dict[str, Any]) -> str:
        """Generate human-readable quality report.
        
        Args:
            results: Results from run_quality_checks
            
        Returns:
            Formatted quality report
        """
        report = []
        report.append("SURROGATE OPTIMIZATION QUALITY REPORT")
        report.append("=" * 50)

        # Overall summary
        overall_pass = results["overall_pass"]
        overall_score = results["overall_score"]
        gates_passed = results["gates_passed"]
        gates_total = results["gates_total"]

        status = "PASS" if overall_pass else "FAIL"
        report.append(f"\nOVERALL STATUS: {status}")
        report.append(f"Overall Score: {overall_score:.3f} (min required: {results['min_required_score']:.3f})")
        report.append(f"Gates Passed: {gates_passed}/{gates_total}")

        if results["failed_gates"]:
            report.append(f"Failed Gates: {', '.join(results['failed_gates'])}")

        # Individual gate results
        report.append("\nDETAILED RESULTS:")
        report.append("-" * 30)

        for result in results["gate_results"]:
            status = "PASS" if result.passed else "FAIL"
            report.append(f"\n{result.gate_name}: {status} (score: {result.score:.3f})")
            report.append(f"  Message: {result.message}")
            report.append(f"  Execution time: {result.execution_time:.3f}s")

            if result.warnings:
                report.append("  Warnings:")
                for warning in result.warnings:
                    report.append(f"    - {warning}")

            # Key details
            if result.details:
                key_metrics = {}
                for key, value in result.details.items():
                    if key.endswith("_pass") or key in ["r2_score", "mse", "training_time", "max_sensitivity"]:
                        if isinstance(value, float):
                            key_metrics[key] = f"{value:.3f}"
                        else:
                            key_metrics[key] = str(value)

                if key_metrics:
                    report.append("  Key metrics:")
                    for key, value in key_metrics.items():
                        report.append(f"    {key}: {value}")

        return "\n".join(report)


def run_quality_gates(
    surrogate,
    test_dataset,
    performance_data: Optional[Dict] = None,
    memory_data: Optional[Dict] = None,
    custom_gates: Optional[List[QualityGate]] = None,
) -> Dict[str, Any]:
    """Run quality gates on surrogate model.
    
    Args:
        surrogate: Trained surrogate model
        test_dataset: Test dataset for validation
        performance_data: Optional performance metrics
        memory_data: Optional memory usage data
        custom_gates: Optional custom quality gates
        
    Returns:
        Quality check results
    """
    context = {
        "surrogate": surrogate,
        "test_dataset": test_dataset,
        "performance_data": performance_data or {},
        "memory_data": memory_data or {},
    }

    checker = AutomatedQualityChecker(gates=custom_gates)
    return checker.run_quality_checks(context)


def generate_quality_report(results: Dict[str, Any]) -> str:
    """Generate quality report from results.
    
    Args:
        results: Quality check results
        
    Returns:
        Formatted quality report
    """
    checker = AutomatedQualityChecker()
    return checker.generate_quality_report(results)
