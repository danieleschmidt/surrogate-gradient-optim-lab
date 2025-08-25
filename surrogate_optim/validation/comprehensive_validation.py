"""Comprehensive validation framework for surrogate optimization - Generation 2.

This module provides extensive validation capabilities including statistical tests,
convergence analysis, and model verification for production-grade reliability.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from typing import Any, Dict, List, Optional

from jax import Array, random
import jax.numpy as jnp

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation strictness levels."""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    PRODUCTION = "production"


class ValidationResult(Enum):
    """Validation outcome."""
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"
    CRITICAL = "critical"


@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    validator_name: str
    level: ValidationLevel
    result: ValidationResult
    score: float
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    execution_time: float = 0.0
    recommendations: List[str] = field(default_factory=list)


class BaseValidator(ABC):
    """Abstract base class for all validators."""

    def __init__(self, name: str, level: ValidationLevel = ValidationLevel.STANDARD):
        self.name = name
        self.level = level

    @abstractmethod
    def validate(self, *args, **kwargs) -> ValidationReport:
        """Perform validation and return report."""
        pass

    def _create_report(self, result: ValidationResult, score: float,
                      message: str, details: Dict = None,
                      recommendations: List[str] = None) -> ValidationReport:
        """Create standardized validation report."""
        return ValidationReport(
            validator_name=self.name,
            level=self.level,
            result=result,
            score=score,
            message=message,
            details=details or {},
            recommendations=recommendations or []
        )


class DataQualityValidator(BaseValidator):
    """Validate data quality and integrity."""

    def __init__(self, level: ValidationLevel = ValidationLevel.STANDARD):
        super().__init__("DataQualityValidator", level)

    def validate(self, X: Array, y: Array, **kwargs) -> ValidationReport:
        """Validate input data quality."""
        start_time = time.time()

        try:
            issues = []
            details = {}
            score = 1.0

            # Basic shape validation
            if X.ndim != 2:
                issues.append("X must be 2D array")
                score -= 0.3

            if y.ndim != 1:
                issues.append("y must be 1D array")
                score -= 0.3

            if X.shape[0] != y.shape[0]:
                issues.append("X and y must have same number of samples")
                score -= 0.5

            details["n_samples"] = X.shape[0] if X.ndim >= 1 else 0
            details["n_features"] = X.shape[1] if X.ndim >= 2 else 0

            # Data quality checks
            nan_count_X = jnp.sum(jnp.isnan(X))
            nan_count_y = jnp.sum(jnp.isnan(y))

            if nan_count_X > 0:
                issues.append(f"X contains {nan_count_X} NaN values")
                score -= 0.2

            if nan_count_y > 0:
                issues.append(f"y contains {nan_count_y} NaN values")
                score -= 0.2

            details["nan_count_X"] = int(nan_count_X)
            details["nan_count_y"] = int(nan_count_y)

            # Infinite values
            inf_count_X = jnp.sum(jnp.isinf(X))
            inf_count_y = jnp.sum(jnp.isinf(y))

            if inf_count_X > 0:
                issues.append(f"X contains {inf_count_X} infinite values")
                score -= 0.2

            if inf_count_y > 0:
                issues.append(f"y contains {inf_count_y} infinite values")
                score -= 0.2

            details["inf_count_X"] = int(inf_count_X)
            details["inf_count_y"] = int(inf_count_y)

            # Data range analysis
            if X.size > 0:
                X_range = jnp.max(X) - jnp.min(X)
                details["X_range"] = float(X_range)
                details["X_mean"] = float(jnp.mean(X))
                details["X_std"] = float(jnp.std(X))

                # Check for very large or small ranges
                if X_range > 1e6:
                    issues.append("X has very large range, consider normalization")
                    score -= 0.1
                elif X_range < 1e-6:
                    issues.append("X has very small range, possible numerical issues")
                    score -= 0.1

            if y.size > 0:
                y_range = jnp.max(y) - jnp.min(y)
                details["y_range"] = float(y_range)
                details["y_mean"] = float(jnp.mean(y))
                details["y_std"] = float(jnp.std(y))

                if y_range > 1e6:
                    issues.append("y has very large range, consider normalization")
                    score -= 0.1
                elif y_range < 1e-6:
                    issues.append("y has very small range, possible constant function")
                    score -= 0.1

            # Sample size validation
            min_samples = 10 if self.level == ValidationLevel.BASIC else 50
            if X.shape[0] < min_samples:
                issues.append(f"Sample size {X.shape[0]} is below recommended minimum {min_samples}")
                score -= 0.3

            # Feature dimensionality check
            if X.ndim >= 2 and X.shape[1] > X.shape[0]:
                issues.append("More features than samples (curse of dimensionality)")
                score -= 0.2

            # Determine result
            if score >= 0.8:
                result = ValidationResult.PASS
            elif score >= 0.6:
                result = ValidationResult.WARNING
            elif score >= 0.4:
                result = ValidationResult.FAIL
            else:
                result = ValidationResult.CRITICAL

            message = "Data quality validation completed"
            if issues:
                message += f" with {len(issues)} issues: " + "; ".join(issues[:3])
                if len(issues) > 3:
                    message += f" and {len(issues) - 3} more"

            # Recommendations
            recommendations = []
            if nan_count_X > 0 or nan_count_y > 0:
                recommendations.append("Remove or impute NaN values")
            if inf_count_X > 0 or inf_count_y > 0:
                recommendations.append("Handle infinite values")
            if X.shape[0] < 100:
                recommendations.append("Consider collecting more training data")
            if details.get("X_range", 0) > 1e3 or details.get("y_range", 0) > 1e3:
                recommendations.append("Consider data normalization or scaling")

            report = self._create_report(result, score, message, details, recommendations)
            report.execution_time = time.time() - start_time

            return report

        except Exception as e:
            return self._create_report(
                ValidationResult.CRITICAL, 0.0,
                f"Data validation failed: {e}",
                {"error": str(e)}
            )


class ModelValidationValidator(BaseValidator):
    """Validate surrogate model quality and reliability."""

    def __init__(self, level: ValidationLevel = ValidationLevel.STANDARD):
        super().__init__("ModelValidationValidator", level)

    def validate(self, model, X_test: Array, y_test: Array, **kwargs) -> ValidationReport:
        """Validate model performance and reliability."""
        start_time = time.time()

        try:
            details = {}
            issues = []
            score = 1.0

            # Check if model is fitted
            if not hasattr(model, "predict") or not hasattr(model, "is_fitted"):
                return self._create_report(
                    ValidationResult.CRITICAL, 0.0,
                    "Model does not have required methods",
                    {"error": "Missing predict method or is_fitted attribute"}
                )

            if hasattr(model, "is_fitted") and not model.is_fitted:
                return self._create_report(
                    ValidationResult.CRITICAL, 0.0,
                    "Model is not fitted",
                    {"error": "Model must be trained before validation"}
                )

            # Prediction validation
            try:
                predictions = model.predict(X_test)
                details["prediction_successful"] = True

                # Check prediction shapes
                if predictions.shape != y_test.shape:
                    issues.append(f"Prediction shape {predictions.shape} != target shape {y_test.shape}")
                    score -= 0.3

                # Check for NaN predictions
                nan_predictions = jnp.sum(jnp.isnan(predictions))
                if nan_predictions > 0:
                    issues.append(f"{nan_predictions} NaN predictions")
                    score -= 0.4

                details["nan_predictions"] = int(nan_predictions)

                # Calculate performance metrics
                if nan_predictions == 0:
                    mse = float(jnp.mean((predictions - y_test)**2))
                    mae = float(jnp.mean(jnp.abs(predictions - y_test)))

                    # R² calculation
                    ss_res = jnp.sum((y_test - predictions)**2)
                    ss_tot = jnp.sum((y_test - jnp.mean(y_test))**2)
                    r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

                    details.update({
                        "mse": mse,
                        "mae": mae,
                        "r2": r2,
                        "rmse": float(jnp.sqrt(mse))
                    })

                    # Performance thresholds based on validation level
                    if self.level == ValidationLevel.PRODUCTION:
                        r2_threshold = 0.9
                    elif self.level == ValidationLevel.STRICT:
                        r2_threshold = 0.8
                    else:
                        r2_threshold = 0.6

                    if r2 < r2_threshold:
                        issues.append(f"R² ({r2:.3f}) below threshold ({r2_threshold})")
                        score -= 0.3

                    # Check for overfitting indicators
                    residuals = predictions - y_test
                    residual_std = float(jnp.std(residuals))
                    y_std = float(jnp.std(y_test))

                    if residual_std > 2 * y_std:
                        issues.append("High residual variance suggests poor fit")
                        score -= 0.2

                    details["residual_std"] = residual_std
                    details["target_std"] = y_std

            except Exception as e:
                issues.append(f"Prediction failed: {e}")
                score -= 0.5
                details["prediction_error"] = str(e)

            # Gradient validation (if supported)
            if hasattr(model, "gradient"):
                try:
                    test_point = X_test[0] if len(X_test) > 0 else jnp.zeros(X_test.shape[1])
                    gradient = model.gradient(test_point)

                    # Check gradient properties
                    if jnp.any(jnp.isnan(gradient)):
                        issues.append("Gradient contains NaN values")
                        score -= 0.2

                    if jnp.any(jnp.isinf(gradient)):
                        issues.append("Gradient contains infinite values")
                        score -= 0.2

                    gradient_norm = float(jnp.linalg.norm(gradient))
                    details["gradient_norm"] = gradient_norm

                    if gradient_norm > 1e6:
                        issues.append("Very large gradient norm")
                        score -= 0.1

                    details["gradient_validation"] = "passed"

                except Exception as e:
                    issues.append(f"Gradient computation failed: {e}")
                    score -= 0.2
                    details["gradient_error"] = str(e)

            # Uncertainty validation (if supported)
            if hasattr(model, "uncertainty"):
                try:
                    test_point = X_test[0] if len(X_test) > 0 else jnp.zeros(X_test.shape[1])
                    uncertainty = model.uncertainty(test_point)

                    if jnp.isnan(uncertainty) or jnp.isinf(uncertainty):
                        issues.append("Invalid uncertainty estimate")
                        score -= 0.1
                    elif uncertainty < 0:
                        issues.append("Negative uncertainty estimate")
                        score -= 0.1

                    details["uncertainty_validation"] = "passed"
                    details["sample_uncertainty"] = float(uncertainty)

                except Exception as e:
                    details["uncertainty_error"] = str(e)

            # Determine result
            if score >= 0.85:
                result = ValidationResult.PASS
            elif score >= 0.7:
                result = ValidationResult.WARNING
            elif score >= 0.5:
                result = ValidationResult.FAIL
            else:
                result = ValidationResult.CRITICAL

            message = f"Model validation completed with score {score:.3f}"
            if issues:
                message += f" ({len(issues)} issues found)"

            # Recommendations
            recommendations = []
            if details.get("r2", 0) < 0.8:
                recommendations.append("Consider more training data or model complexity")
            if details.get("nan_predictions", 0) > 0:
                recommendations.append("Debug model for numerical stability issues")
            if "gradient_error" in details:
                recommendations.append("Fix gradient computation for optimization")

            report = self._create_report(result, score, message, details, recommendations)
            report.execution_time = time.time() - start_time

            return report

        except Exception as e:
            return self._create_report(
                ValidationResult.CRITICAL, 0.0,
                f"Model validation failed: {e}",
                {"error": str(e)}
            )


class ConvergenceValidator(BaseValidator):
    """Validate optimization convergence."""

    def __init__(self, level: ValidationLevel = ValidationLevel.STANDARD):
        super().__init__("ConvergenceValidator", level)

    def validate(self, optimization_history: List[float], **kwargs) -> ValidationReport:
        """Validate optimization convergence behavior."""
        start_time = time.time()

        try:
            if not optimization_history or len(optimization_history) < 2:
                return self._create_report(
                    ValidationResult.FAIL, 0.0,
                    "Insufficient optimization history",
                    {"history_length": len(optimization_history)}
                )

            history = jnp.array(optimization_history)
            details = {}
            issues = []
            score = 1.0

            # Basic statistics
            details.update({
                "initial_value": float(history[0]),
                "final_value": float(history[-1]),
                "total_improvement": float(history[0] - history[-1]),
                "n_iterations": len(history)
            })

            # Convergence analysis
            improvements = jnp.diff(history)  # Negative values indicate improvement
            n_improvements = jnp.sum(improvements < 0)
            improvement_rate = float(n_improvements / len(improvements))

            details["improvement_rate"] = improvement_rate
            details["n_improvements"] = int(n_improvements)

            # Check for convergence
            if len(history) >= 10:
                recent_window = history[-10:]
                recent_std = float(jnp.std(recent_window))
                overall_std = float(jnp.std(history))

                details["recent_std"] = recent_std
                details["overall_std"] = overall_std

                # Convergence indicator
                convergence_ratio = recent_std / (overall_std + 1e-8)
                details["convergence_ratio"] = float(convergence_ratio)

                if convergence_ratio > 0.5:
                    issues.append("Poor convergence: high variance in recent iterations")
                    score -= 0.3
                elif convergence_ratio < 0.1:
                    details["converged"] = True
                else:
                    details["converged"] = False

            # Check for stagnation
            if len(history) >= 20:
                last_20_pct = history[-int(len(history) * 0.2):]
                improvement_in_last_20pct = float(jnp.max(last_20_pct) - jnp.min(last_20_pct))
                total_range = float(jnp.max(history) - jnp.min(history))

                if improvement_in_last_20pct < 0.01 * total_range:
                    issues.append("Optimization appears to have stagnated")
                    score -= 0.2

                details["stagnation_indicator"] = improvement_in_last_20pct / (total_range + 1e-8)

            # Check for oscillations
            sign_changes = jnp.sum(jnp.diff(jnp.sign(improvements)) != 0)
            oscillation_rate = float(sign_changes / len(improvements))
            details["oscillation_rate"] = oscillation_rate

            if oscillation_rate > 0.7:
                issues.append("High oscillation rate suggests unstable optimization")
                score -= 0.2

            # Check for adequate improvement
            total_improvement = details["total_improvement"]
            if total_improvement <= 0:
                issues.append("No improvement achieved")
                score -= 0.4
            elif total_improvement < 0.01 * abs(details["initial_value"]):
                issues.append("Very small relative improvement")
                score -= 0.2

            # Early termination check
            if len(history) < 10:
                issues.append("Optimization terminated very early")
                score -= 0.1

            # Determine result
            if score >= 0.8:
                result = ValidationResult.PASS
            elif score >= 0.6:
                result = ValidationResult.WARNING
            elif score >= 0.4:
                result = ValidationResult.FAIL
            else:
                result = ValidationResult.CRITICAL

            message = f"Convergence analysis: {improvement_rate:.1%} improvement rate"
            if issues:
                message += f" with {len(issues)} concerns"

            # Recommendations
            recommendations = []
            if improvement_rate < 0.3:
                recommendations.append("Consider different optimization algorithm")
            if oscillation_rate > 0.5:
                recommendations.append("Reduce learning rate or add momentum")
            if details.get("convergence_ratio", 1.0) > 0.3:
                recommendations.append("Increase maximum iterations or patience")

            report = self._create_report(result, score, message, details, recommendations)
            report.execution_time = time.time() - start_time

            return report

        except Exception as e:
            return self._create_report(
                ValidationResult.CRITICAL, 0.0,
                f"Convergence validation failed: {e}",
                {"error": str(e)}
            )


class StatisticalValidator(BaseValidator):
    """Perform statistical validation tests."""

    def __init__(self, level: ValidationLevel = ValidationLevel.STANDARD):
        super().__init__("StatisticalValidator", level)

    def validate(self, predictions: Array, targets: Array, **kwargs) -> ValidationReport:
        """Perform statistical validation of predictions."""
        start_time = time.time()

        try:
            if len(predictions) != len(targets):
                return self._create_report(
                    ValidationResult.CRITICAL, 0.0,
                    "Predictions and targets have different lengths",
                    {"pred_len": len(predictions), "target_len": len(targets)}
                )

            details = {}
            issues = []
            score = 1.0

            # Basic statistics
            residuals = predictions - targets
            details.update({
                "n_samples": len(predictions),
                "mean_residual": float(jnp.mean(residuals)),
                "std_residual": float(jnp.std(residuals)),
                "max_residual": float(jnp.max(jnp.abs(residuals))),
            })

            # Bias test
            mean_residual = details["mean_residual"]
            if abs(mean_residual) > 0.1 * jnp.std(targets):
                issues.append(f"Significant bias detected: {mean_residual:.3f}")
                score -= 0.2

            # Normality test (simplified)
            # Check for roughly symmetric distribution of residuals
            q25 = float(jnp.percentile(residuals, 25))
            q75 = float(jnp.percentile(residuals, 75))
            median = float(jnp.median(residuals))

            skewness_indicator = (median - q25) / (q75 - q25 + 1e-8) - 0.5
            details["skewness_indicator"] = skewness_indicator

            if abs(skewness_indicator) > 0.3:
                issues.append("Residuals appear non-symmetric")
                score -= 0.1

            # Homoscedasticity test (simplified)
            # Check if residual variance is constant across prediction range
            sorted_indices = jnp.argsort(predictions)
            n_bins = min(5, len(predictions) // 10)

            if n_bins >= 2:
                bin_size = len(predictions) // n_bins
                bin_variances = []

                for i in range(n_bins):
                    start_idx = i * bin_size
                    end_idx = (i + 1) * bin_size if i < n_bins - 1 else len(predictions)
                    bin_indices = sorted_indices[start_idx:end_idx]
                    bin_residuals = residuals[bin_indices]
                    bin_variances.append(float(jnp.var(bin_residuals)))

                variance_ratio = max(bin_variances) / (min(bin_variances) + 1e-8)
                details["variance_ratio"] = variance_ratio

                if variance_ratio > 4.0:
                    issues.append("Heteroscedasticity detected")
                    score -= 0.2

            # Outlier detection
            residual_threshold = 3 * jnp.std(residuals)
            outliers = jnp.sum(jnp.abs(residuals) > residual_threshold)
            outlier_rate = float(outliers / len(residuals))

            details["outlier_count"] = int(outliers)
            details["outlier_rate"] = outlier_rate

            if outlier_rate > 0.1:
                issues.append(f"High outlier rate: {outlier_rate:.1%}")
                score -= 0.2

            # Correlation analysis
            correlation = float(jnp.corrcoef(predictions, targets)[0, 1])
            details["correlation"] = correlation

            min_correlation = 0.9 if self.level == ValidationLevel.PRODUCTION else 0.7
            if correlation < min_correlation:
                issues.append(f"Low correlation: {correlation:.3f}")
                score -= 0.3

            # Distribution comparison
            pred_std = float(jnp.std(predictions))
            target_std = float(jnp.std(targets))
            std_ratio = pred_std / (target_std + 1e-8)

            details["prediction_std"] = pred_std
            details["target_std"] = target_std
            details["std_ratio"] = std_ratio

            if abs(std_ratio - 1.0) > 0.5:
                issues.append(f"Standard deviation mismatch: ratio = {std_ratio:.2f}")
                score -= 0.1

            # Determine result
            if score >= 0.8:
                result = ValidationResult.PASS
            elif score >= 0.6:
                result = ValidationResult.WARNING
            elif score >= 0.4:
                result = ValidationResult.FAIL
            else:
                result = ValidationResult.CRITICAL

            message = f"Statistical validation: correlation = {correlation:.3f}"
            if issues:
                message += f" with {len(issues)} statistical concerns"

            # Recommendations
            recommendations = []
            if correlation < 0.8:
                recommendations.append("Improve model fit or collect more representative data")
            if outlier_rate > 0.05:
                recommendations.append("Investigate and handle outliers")
            if abs(mean_residual) > 0.05:
                recommendations.append("Address systematic bias in predictions")

            report = self._create_report(result, score, message, details, recommendations)
            report.execution_time = time.time() - start_time

            return report

        except Exception as e:
            return self._create_report(
                ValidationResult.CRITICAL, 0.0,
                f"Statistical validation failed: {e}",
                {"error": str(e)}
            )


class ComprehensiveValidationSuite:
    """Complete validation suite for surrogate optimization."""

    def __init__(self, level: ValidationLevel = ValidationLevel.STANDARD):
        """Initialize validation suite.
        
        Args:
            level: Validation strictness level
        """
        self.level = level
        self.validators = {
            "data_quality": DataQualityValidator(level),
            "model_validation": ModelValidationValidator(level),
            "convergence": ConvergenceValidator(level),
            "statistical": StatisticalValidator(level),
        }

        self.validation_history: List[Dict[str, ValidationReport]] = []

    def validate_data(self, X: Array, y: Array) -> ValidationReport:
        """Validate input data quality."""
        return self.validators["data_quality"].validate(X, y)

    def validate_model(self, model, X_test: Array, y_test: Array) -> ValidationReport:
        """Validate trained model."""
        return self.validators["model_validation"].validate(model, X_test, y_test)

    def validate_convergence(self, optimization_history: List[float]) -> ValidationReport:
        """Validate optimization convergence."""
        return self.validators["convergence"].validate(optimization_history)

    def validate_predictions(self, predictions: Array, targets: Array) -> ValidationReport:
        """Validate prediction quality statistically."""
        return self.validators["statistical"].validate(predictions, targets)

    def run_full_validation(self, model, X_train: Array, y_train: Array,
                           X_test: Array, y_test: Array,
                           optimization_history: Optional[List[float]] = None) -> Dict[str, ValidationReport]:
        """Run complete validation suite.
        
        Args:
            model: Trained surrogate model
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            optimization_history: Optimization convergence history
            
        Returns:
            Dictionary of validation reports
        """
        logger.info("Running comprehensive validation suite...")

        reports = {}

        # Data quality validation
        logger.info("Validating training data quality...")
        reports["data_quality_train"] = self.validate_data(X_train, y_train)

        logger.info("Validating test data quality...")
        reports["data_quality_test"] = self.validate_data(X_test, y_test)

        # Model validation
        logger.info("Validating model performance...")
        reports["model_validation"] = self.validate_model(model, X_test, y_test)

        # Prediction validation
        if reports["model_validation"].result != ValidationResult.CRITICAL:
            try:
                predictions = model.predict(X_test)
                logger.info("Validating prediction statistics...")
                reports["statistical_validation"] = self.validate_predictions(predictions, y_test)
            except Exception as e:
                logger.error(f"Prediction validation failed: {e}")
                reports["statistical_validation"] = ValidationReport(
                    validator_name="StatisticalValidator",
                    level=self.level,
                    result=ValidationResult.CRITICAL,
                    score=0.0,
                    message=f"Prediction validation failed: {e}"
                )

        # Convergence validation
        if optimization_history:
            logger.info("Validating optimization convergence...")
            reports["convergence_validation"] = self.validate_convergence(optimization_history)

        # Store validation session
        self.validation_history.append(reports)

        return reports

    def get_overall_score(self, reports: Dict[str, ValidationReport]) -> float:
        """Calculate overall validation score."""
        if not reports:
            return 0.0

        # Weight different validation types
        weights = {
            "data_quality_train": 0.2,
            "data_quality_test": 0.2,
            "model_validation": 0.4,
            "statistical_validation": 0.3,
            "convergence_validation": 0.1,
        }

        total_score = 0.0
        total_weight = 0.0

        for report_name, report in reports.items():
            weight = weights.get(report_name, 0.1)
            total_score += report.score * weight
            total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0

    def get_validation_summary(self, reports: Dict[str, ValidationReport]) -> Dict[str, Any]:
        """Get comprehensive validation summary."""
        if not reports:
            return {"status": "no_validation_performed"}

        overall_score = self.get_overall_score(reports)

        # Count results by type
        result_counts = {}
        for report in reports.values():
            result_counts[report.result.value] = result_counts.get(report.result.value, 0) + 1

        # Collect all issues and recommendations
        all_issues = []
        all_recommendations = []

        for report in reports.values():
            if report.result in [ValidationResult.WARNING, ValidationResult.FAIL, ValidationResult.CRITICAL]:
                all_issues.append(f"{report.validator_name}: {report.message}")
            all_recommendations.extend(report.recommendations)

        # Determine overall status
        if any(r.result == ValidationResult.CRITICAL for r in reports.values()):
            overall_status = "critical"
        elif any(r.result == ValidationResult.FAIL for r in reports.values()):
            overall_status = "failed"
        elif any(r.result == ValidationResult.WARNING for r in reports.values()):
            overall_status = "warning"
        else:
            overall_status = "passed"

        return {
            "overall_status": overall_status,
            "overall_score": overall_score,
            "validation_level": self.level.value,
            "result_distribution": result_counts,
            "total_validations": len(reports),
            "issues_found": len(all_issues),
            "main_issues": all_issues[:5],  # Top 5 issues
            "recommendations": list(set(all_recommendations))[:10],  # Top 10 unique recommendations
            "execution_times": {name: report.execution_time for name, report in reports.items()},
            "total_validation_time": sum(report.execution_time for report in reports.values()),
        }

    def generate_validation_report(self, reports: Dict[str, ValidationReport]) -> str:
        """Generate human-readable validation report."""
        summary = self.get_validation_summary(reports)

        report_lines = [
            "=" * 60,
            "COMPREHENSIVE VALIDATION REPORT",
            "=" * 60,
            f"Overall Status: {summary['overall_status'].upper()}",
            f"Overall Score: {summary['overall_score']:.2f}/1.00",
            f"Validation Level: {summary['validation_level'].upper()}",
            f"Total Validations: {summary['total_validations']}",
            f"Total Execution Time: {summary['total_validation_time']:.3f}s",
            "",
            "VALIDATION RESULTS:",
            "-" * 30,
        ]

        for name, report in reports.items():
            status_symbol = {
                ValidationResult.PASS: "✓",
                ValidationResult.WARNING: "⚠",
                ValidationResult.FAIL: "✗",
                ValidationResult.CRITICAL: "❌"
            }[report.result]

            report_lines.append(
                f"{status_symbol} {name}: {report.result.value.upper()} "
                f"(score: {report.score:.2f}, time: {report.execution_time:.3f}s)"
            )
            report_lines.append(f"   {report.message}")

        if summary["main_issues"]:
            report_lines.extend([
                "",
                "MAIN ISSUES:",
                "-" * 20,
            ])
            for issue in summary["main_issues"]:
                report_lines.append(f"• {issue}")

        if summary["recommendations"]:
            report_lines.extend([
                "",
                "RECOMMENDATIONS:",
                "-" * 20,
            ])
            for rec in summary["recommendations"]:
                report_lines.append(f"• {rec}")

        report_lines.append("=" * 60)

        return "\\n".join(report_lines)


# Example usage
if __name__ == "__main__":
    print("🔍 COMPREHENSIVE VALIDATION FRAMEWORK DEMONSTRATION")
    print("=" * 70)

    # Create validation suite
    validator = ComprehensiveValidationSuite(ValidationLevel.STANDARD)

    # Generate sample data for demonstration
    key = random.PRNGKey(42)
    n_train, n_test = 100, 30
    n_features = 2

    # Training data
    X_train = random.normal(key, (n_train, n_features))
    y_train = jnp.sum(X_train**2, axis=1) + 0.1 * random.normal(random.fold_in(key, 1), (n_train,))

    # Test data
    X_test = random.normal(random.fold_in(key, 2), (n_test, n_features))
    y_test = jnp.sum(X_test**2, axis=1) + 0.1 * random.normal(random.fold_in(key, 3), (n_test,))

    print("\\n1. Data Quality Validation")
    data_report = validator.validate_data(X_train, y_train)
    print(f"Result: {data_report.result.value}, Score: {data_report.score:.3f}")
    print(f"Message: {data_report.message}")

    print("\\n2. Creating Mock Model for Validation")

    # Simple mock model for demonstration
    class MockModel:
        def __init__(self):
            self.is_fitted = True
            self.coef = jnp.array([1.0, 1.0])  # Simple linear coefficients

        def predict(self, X):
            return jnp.sum(X * self.coef, axis=1)

        def gradient(self, x):
            return self.coef

    model = MockModel()

    print("\\n3. Model Performance Validation")
    model_report = validator.validate_model(model, X_test, y_test)
    print(f"Result: {model_report.result.value}, Score: {model_report.score:.3f}")
    print(f"Message: {model_report.message}")

    print("\\n4. Statistical Validation")
    predictions = model.predict(X_test)
    stats_report = validator.validate_predictions(predictions, y_test)
    print(f"Result: {stats_report.result.value}, Score: {stats_report.score:.3f}")
    print(f"Message: {stats_report.message}")

    print("\\n5. Convergence Validation")
    # Mock optimization history
    optimization_history = [10.0, 8.5, 7.2, 6.1, 5.8, 5.5, 5.3, 5.2, 5.15, 5.1]
    conv_report = validator.validate_convergence(optimization_history)
    print(f"Result: {conv_report.result.value}, Score: {conv_report.score:.3f}")
    print(f"Message: {conv_report.message}")

    print("\\n" + "="*70)
    print("✅ VALIDATION FRAMEWORK DEMONSTRATION COMPLETED")
    print("="*70)
