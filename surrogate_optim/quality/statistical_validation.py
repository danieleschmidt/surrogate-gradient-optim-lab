"""Statistical validation utilities for quality gates."""

import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

import jax.numpy as jnp
from jax import Array
import numpy as np
from scipy import stats
from scipy.stats import chi2, normaltest, kstest, shapiro

from .quality_gates import QualityGate, QualityResult


class StatisticalTest(Enum):
    """Types of statistical tests."""
    NORMALITY_SHAPIRO = "shapiro"
    NORMALITY_KOLMOGOROV = "kolmogorov"
    NORMALITY_DAGOSTINO = "dagostino"
    INDEPENDENCE_RUNS = "runs_test"
    HOMOSCEDASTICITY_LEVENE = "levene"
    GOODNESS_OF_FIT_CHI2 = "chi2_gof"
    CONVERGENCE_MANN_KENDALL = "mann_kendall"
    OUTLIER_DETECTION_IQR = "iqr_outliers"
    OUTLIER_DETECTION_ZSCORE = "zscore_outliers"


@dataclass
class StatisticalTestResult:
    """Result of a statistical test."""
    test_name: str
    statistic: float
    p_value: float
    critical_value: Optional[float]
    rejected: bool
    confidence_level: float
    interpretation: str
    details: Dict[str, Any]


class StatisticalValidator:
    """Statistical validation utilities for surrogate models."""
    
    def __init__(self, confidence_level: float = 0.95):
        """Initialize statistical validator.
        
        Args:
            confidence_level: Confidence level for statistical tests
        """
        self.confidence_level = confidence_level
        self.alpha = 1.0 - confidence_level
    
    def test_residual_normality(
        self,
        residuals: Array,
        method: str = "shapiro"
    ) -> StatisticalTestResult:
        """Test if residuals follow normal distribution.
        
        Args:
            residuals: Model residuals
            method: Test method ("shapiro", "kolmogorov", "dagostino")
            
        Returns:
            Statistical test result
        """
        residuals_np = np.array(residuals)
        
        if method == "shapiro":
            if len(residuals_np) > 5000:
                # Shapiro-Wilk test is computationally expensive for large samples
                method = "kolmogorov"
                warnings.warn("Using Kolmogorov-Smirnov test for large sample size")
        
        try:
            if method == "shapiro":
                statistic, p_value = shapiro(residuals_np)
                critical_value = None
                test_name = "Shapiro-Wilk Normality Test"
                
            elif method == "kolmogorov":
                statistic, p_value = kstest(residuals_np, 'norm', 
                                          args=(np.mean(residuals_np), np.std(residuals_np)))
                critical_value = None
                test_name = "Kolmogorov-Smirnov Normality Test"
                
            elif method == "dagostino":
                statistic, p_value = normaltest(residuals_np)
                critical_value = chi2.ppf(self.confidence_level, 2)  # 2 degrees of freedom
                test_name = "D'Agostino-Pearson Normality Test"
                
            else:
                raise ValueError(f"Unknown method: {method}")
            
            rejected = p_value < self.alpha
            
            if rejected:
                interpretation = f"Residuals do NOT follow normal distribution (p={p_value:.6f} < α={self.alpha:.3f})"
            else:
                interpretation = f"Residuals appear to follow normal distribution (p={p_value:.6f} ≥ α={self.alpha:.3f})"
            
            return StatisticalTestResult(
                test_name=test_name,
                statistic=float(statistic),
                p_value=float(p_value),
                critical_value=float(critical_value) if critical_value is not None else None,
                rejected=rejected,
                confidence_level=self.confidence_level,
                interpretation=interpretation,
                details={
                    "method": method,
                    "sample_size": len(residuals_np),
                    "residual_mean": float(np.mean(residuals_np)),
                    "residual_std": float(np.std(residuals_np)),
                }
            )
            
        except Exception as e:
            return StatisticalTestResult(
                test_name=f"Normality Test ({method})",
                statistic=0.0,
                p_value=1.0,
                critical_value=None,
                rejected=True,
                confidence_level=self.confidence_level,
                interpretation=f"Test failed: {str(e)}",
                details={"error": str(e)}
            )
    
    def test_residual_independence(self, residuals: Array) -> StatisticalTestResult:
        """Test if residuals are independent using runs test.
        
        Args:
            residuals: Model residuals
            
        Returns:
            Statistical test result
        """
        try:
            residuals_np = np.array(residuals)
            median_residual = np.median(residuals_np)
            
            # Convert to binary sequence (above/below median)
            binary_sequence = (residuals_np > median_residual).astype(int)
            
            # Count runs
            runs = 1
            for i in range(1, len(binary_sequence)):
                if binary_sequence[i] != binary_sequence[i-1]:
                    runs += 1
            
            n1 = np.sum(binary_sequence == 1)
            n2 = np.sum(binary_sequence == 0)
            n = n1 + n2
            
            # Expected runs and variance
            expected_runs = ((2 * n1 * n2) / n) + 1
            variance_runs = (2 * n1 * n2 * (2 * n1 * n2 - n)) / (n**2 * (n - 1))
            
            # Z-statistic
            if variance_runs > 0:
                z_statistic = (runs - expected_runs) / np.sqrt(variance_runs)
            else:
                z_statistic = 0.0
            
            # P-value (two-tailed test)
            p_value = 2 * (1 - stats.norm.cdf(abs(z_statistic)))
            
            rejected = p_value < self.alpha
            critical_z = stats.norm.ppf(1 - self.alpha/2)
            
            if rejected:
                interpretation = f"Residuals show significant dependence (p={p_value:.6f} < α={self.alpha:.3f})"
            else:
                interpretation = f"Residuals appear independent (p={p_value:.6f} ≥ α={self.alpha:.3f})"
            
            return StatisticalTestResult(
                test_name="Runs Test for Independence",
                statistic=float(z_statistic),
                p_value=float(p_value),
                critical_value=float(critical_z),
                rejected=rejected,
                confidence_level=self.confidence_level,
                interpretation=interpretation,
                details={
                    "runs": int(runs),
                    "expected_runs": float(expected_runs),
                    "n_above_median": int(n1),
                    "n_below_median": int(n2),
                    "total_samples": int(n),
                }
            )
            
        except Exception as e:
            return StatisticalTestResult(
                test_name="Runs Test for Independence",
                statistic=0.0,
                p_value=1.0,
                critical_value=None,
                rejected=True,
                confidence_level=self.confidence_level,
                interpretation=f"Test failed: {str(e)}",
                details={"error": str(e)}
            )
    
    def test_convergence(self, convergence_history: List[float]) -> StatisticalTestResult:
        """Test for convergence using Mann-Kendall trend test.
        
        Args:
            convergence_history: History of objective values
            
        Returns:
            Statistical test result
        """
        try:
            if len(convergence_history) < 3:
                return StatisticalTestResult(
                    test_name="Mann-Kendall Convergence Test",
                    statistic=0.0,
                    p_value=1.0,
                    critical_value=None,
                    rejected=True,
                    confidence_level=self.confidence_level,
                    interpretation="Insufficient data for convergence test",
                    details={"error": "Need at least 3 points"}
                )
            
            values = np.array(convergence_history)
            n = len(values)
            
            # Mann-Kendall statistic
            s = 0
            for i in range(n - 1):
                for j in range(i + 1, n):
                    if values[j] > values[i]:
                        s += 1
                    elif values[j] < values[i]:
                        s -= 1
            
            # Variance
            var_s = n * (n - 1) * (2 * n + 5) / 18
            
            # Z-statistic
            if s > 0:
                z = (s - 1) / np.sqrt(var_s)
            elif s < 0:
                z = (s + 1) / np.sqrt(var_s)
            else:
                z = 0.0
            
            # P-value (two-tailed test)
            p_value = 2 * (1 - stats.norm.cdf(abs(z)))
            
            # For convergence, we want a decreasing trend (negative slope)
            converged = (z < 0) and (p_value < self.alpha)
            
            critical_z = stats.norm.ppf(1 - self.alpha/2)
            
            if converged:
                interpretation = f"Significant convergence trend detected (z={z:.3f}, p={p_value:.6f})"
            else:
                interpretation = f"No significant convergence trend (z={z:.3f}, p={p_value:.6f})"
            
            # Additional convergence metrics
            relative_improvement = abs(values[-1] - values[0]) / abs(values[0]) if values[0] != 0 else 0
            recent_stability = np.std(values[-min(5, n//2):]) if n > 2 else float('inf')
            
            return StatisticalTestResult(
                test_name="Mann-Kendall Convergence Test",
                statistic=float(z),
                p_value=float(p_value),
                critical_value=float(critical_z),
                rejected=not converged,
                confidence_level=self.confidence_level,
                interpretation=interpretation,
                details={
                    "kendall_s": int(s),
                    "trend_direction": "decreasing" if z < 0 else "increasing" if z > 0 else "no trend",
                    "relative_improvement": float(relative_improvement),
                    "recent_stability": float(recent_stability),
                    "initial_value": float(values[0]),
                    "final_value": float(values[-1]),
                    "n_points": int(n),
                }
            )
            
        except Exception as e:
            return StatisticalTestResult(
                test_name="Mann-Kendall Convergence Test",
                statistic=0.0,
                p_value=1.0,
                critical_value=None,
                rejected=True,
                confidence_level=self.confidence_level,
                interpretation=f"Test failed: {str(e)}",
                details={"error": str(e)}
            )
    
    def detect_outliers(
        self,
        data: Array,
        method: str = "iqr"
    ) -> StatisticalTestResult:
        """Detect outliers in data.
        
        Args:
            data: Data array
            method: Detection method ("iqr", "zscore")
            
        Returns:
            Statistical test result with outlier information
        """
        try:
            data_np = np.array(data).flatten()
            n = len(data_np)
            
            if method == "iqr":
                q1 = np.percentile(data_np, 25)
                q3 = np.percentile(data_np, 75)
                iqr = q3 - q1
                
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = (data_np < lower_bound) | (data_np > upper_bound)
                n_outliers = np.sum(outliers)
                outlier_percentage = (n_outliers / n) * 100
                
                # No formal statistical test for IQR method
                statistic = float(outlier_percentage)
                p_value = None
                critical_value = 5.0  # 5% threshold
                
                rejected = outlier_percentage > critical_value
                
                interpretation = f"IQR method: {n_outliers} outliers ({outlier_percentage:.2f}%)"
                if rejected:
                    interpretation += f" - HIGH outlier rate (>{critical_value:.1f}%)"
                
                details = {
                    "method": method,
                    "n_outliers": int(n_outliers),
                    "outlier_percentage": float(outlier_percentage),
                    "q1": float(q1),
                    "q3": float(q3),
                    "iqr": float(iqr),
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound),
                    "outlier_indices": np.where(outliers)[0].tolist(),
                }
                
            elif method == "zscore":
                mean_val = np.mean(data_np)
                std_val = np.std(data_np)
                
                if std_val == 0:
                    z_scores = np.zeros_like(data_np)
                else:
                    z_scores = np.abs((data_np - mean_val) / std_val)
                
                threshold = stats.norm.ppf(1 - self.alpha/2)  # Two-tailed threshold
                outliers = z_scores > threshold
                n_outliers = np.sum(outliers)
                outlier_percentage = (n_outliers / n) * 100
                
                statistic = float(np.max(z_scores))
                p_value = 2 * (1 - stats.norm.cdf(statistic))
                critical_value = float(threshold)
                
                rejected = statistic > critical_value
                
                interpretation = f"Z-score method: {n_outliers} outliers ({outlier_percentage:.2f}%)"
                if rejected:
                    interpretation += f" - Max |z|={statistic:.3f} > {critical_value:.3f}"
                
                details = {
                    "method": method,
                    "n_outliers": int(n_outliers),
                    "outlier_percentage": float(outlier_percentage),
                    "max_zscore": float(statistic),
                    "mean": float(mean_val),
                    "std": float(std_val),
                    "threshold": float(threshold),
                    "outlier_indices": np.where(outliers)[0].tolist(),
                }
                
            else:
                raise ValueError(f"Unknown method: {method}")
            
            return StatisticalTestResult(
                test_name=f"Outlier Detection ({method.upper()})",
                statistic=statistic,
                p_value=p_value,
                critical_value=critical_value,
                rejected=rejected,
                confidence_level=self.confidence_level,
                interpretation=interpretation,
                details=details
            )
            
        except Exception as e:
            return StatisticalTestResult(
                test_name=f"Outlier Detection ({method.upper()})",
                statistic=0.0,
                p_value=1.0,
                critical_value=None,
                rejected=True,
                confidence_level=self.confidence_level,
                interpretation=f"Test failed: {str(e)}",
                details={"error": str(e)}
            )


class StatisticalQualityGate(QualityGate):
    """Quality gate based on statistical validation."""
    
    def __init__(
        self,
        name: str = "statistical_validation",
        confidence_level: float = 0.95,
        required_tests: List[str] = None,
        weight: float = 1.5,
    ):
        """Initialize statistical quality gate.
        
        Args:
            name: Gate name
            confidence_level: Statistical confidence level
            required_tests: List of required statistical tests
            weight: Gate weight
        """
        super().__init__(name, 0.7, weight)
        self.confidence_level = confidence_level
        self.validator = StatisticalValidator(confidence_level)
        
        if required_tests is None:
            self.required_tests = [
                "normality",
                "independence", 
                "convergence",
                "outliers"
            ]
        else:
            self.required_tests = required_tests
    
    def check(self, context: Dict[str, Any]) -> QualityResult:
        """Run statistical validation checks."""
        start_time = time.time()
        warnings_list = []
        
        try:
            surrogate = context.get("surrogate")
            test_dataset = context.get("test_dataset")
            convergence_history = context.get("convergence_history", [])
            
            if not surrogate or not test_dataset:
                return self._create_result(
                    False, 0.0, "Missing surrogate or test dataset",
                    {}, time.time() - start_time
                )
            
            # Compute residuals
            predictions = surrogate.predict(test_dataset.X)
            residuals = test_dataset.y - predictions
            
            test_results = {}
            passed_tests = 0
            total_tests = len(self.required_tests)
            
            # Run statistical tests
            if "normality" in self.required_tests:
                normality_result = self.validator.test_residual_normality(residuals)
                test_results["normality"] = normality_result
                if not normality_result.rejected:
                    passed_tests += 1
                else:
                    warnings_list.append(f"Normality test failed: {normality_result.interpretation}")
            
            if "independence" in self.required_tests:
                independence_result = self.validator.test_residual_independence(residuals)
                test_results["independence"] = independence_result
                if not independence_result.rejected:
                    passed_tests += 1
                else:
                    warnings_list.append(f"Independence test failed: {independence_result.interpretation}")
            
            if "convergence" in self.required_tests and convergence_history:
                convergence_result = self.validator.test_convergence(convergence_history)
                test_results["convergence"] = convergence_result
                if not convergence_result.rejected:
                    passed_tests += 1
                else:
                    warnings_list.append(f"Convergence test failed: {convergence_result.interpretation}")
            
            if "outliers" in self.required_tests:
                outlier_result = self.validator.detect_outliers(residuals)
                test_results["outliers"] = outlier_result
                if not outlier_result.rejected:
                    passed_tests += 1
                else:
                    warnings_list.append(f"Outlier detection failed: {outlier_result.interpretation}")
            
            # Calculate overall score
            score = passed_tests / total_tests if total_tests > 0 else 0.0
            overall_pass = score >= self.threshold
            
            # Generate message
            message = f"Statistical validation: {passed_tests}/{total_tests} tests passed (score: {score:.3f})"
            if overall_pass:
                message = f"PASS: {message}"
            else:
                message = f"FAIL: {message}"
            
            details = {
                "confidence_level": self.confidence_level,
                "passed_tests": passed_tests,
                "total_tests": total_tests,
                "test_results": {
                    name: {
                        "test_name": result.test_name,
                        "statistic": result.statistic,
                        "p_value": result.p_value,
                        "rejected": result.rejected,
                        "interpretation": result.interpretation,
                        "details": result.details,
                    }
                    for name, result in test_results.items()
                },
                "residual_statistics": {
                    "mean": float(jnp.mean(residuals)),
                    "std": float(jnp.std(residuals)),
                    "skewness": float(jnp.mean(((residuals - jnp.mean(residuals)) / jnp.std(residuals)) ** 3)),
                    "kurtosis": float(jnp.mean(((residuals - jnp.mean(residuals)) / jnp.std(residuals)) ** 4) - 3),
                }
            }
            
            return self._create_result(
                overall_pass, score, message, details,
                time.time() - start_time, warnings_list
            )
            
        except Exception as e:
            return self._create_result(
                False, 0.0, f"Error during statistical validation: {e}",
                {"error": str(e)}, time.time() - start_time
            )


class ConvergenceValidationGate(QualityGate):
    """Quality gate specifically for convergence validation."""
    
    def __init__(
        self,
        name: str = "convergence_validation",
        min_improvement_ratio: float = 0.01,
        max_stagnation_steps: int = 10,
        required_confidence: float = 0.95,
        weight: float = 2.0,
    ):
        """Initialize convergence validation gate.
        
        Args:
            name: Gate name
            min_improvement_ratio: Minimum relative improvement required
            max_stagnation_steps: Maximum steps without improvement
            required_confidence: Required statistical confidence for convergence
            weight: Gate weight
        """
        super().__init__(name, 0.8, weight)
        self.min_improvement_ratio = min_improvement_ratio
        self.max_stagnation_steps = max_stagnation_steps
        self.required_confidence = required_confidence
        self.validator = StatisticalValidator(required_confidence)
    
    def check(self, context: Dict[str, Any]) -> QualityResult:
        """Check convergence validation."""
        start_time = time.time()
        warnings_list = []
        
        try:
            convergence_history = context.get("convergence_history", [])
            optimization_result = context.get("optimization_result")
            
            if not convergence_history:
                return self._create_result(
                    False, 0.0, "No convergence history available",
                    {}, time.time() - start_time
                )
            
            if len(convergence_history) < 5:
                warnings_list.append("Limited convergence history for statistical analysis")
                return self._create_result(
                    False, 0.3, "Insufficient convergence data",
                    {"convergence_points": len(convergence_history)},
                    time.time() - start_time, warnings_list
                )
            
            # Statistical convergence test
            convergence_test = self.validator.test_convergence(convergence_history)
            
            # Additional convergence metrics
            values = np.array(convergence_history)
            initial_value = values[0]
            final_value = values[-1]
            
            # Relative improvement
            if abs(initial_value) > 1e-10:
                relative_improvement = abs(final_value - initial_value) / abs(initial_value)
            else:
                relative_improvement = 0.0
            
            # Stagnation analysis
            stagnation_count = 0
            max_stagnation = 0
            improvement_threshold = abs(final_value) * 0.001  # 0.1% improvement threshold
            
            for i in range(1, len(values)):
                if abs(values[i] - values[i-1]) < improvement_threshold:
                    stagnation_count += 1
                    max_stagnation = max(max_stagnation, stagnation_count)
                else:
                    stagnation_count = 0
            
            # Recent stability
            recent_window = min(10, len(values) // 4)
            if recent_window >= 2:
                recent_values = values[-recent_window:]
                recent_stability = np.std(recent_values) / abs(np.mean(recent_values)) if np.mean(recent_values) != 0 else 0
            else:
                recent_stability = float('inf')
            
            # Convergence rate analysis
            if len(values) > 3:
                # Fit exponential decay model
                x = np.arange(len(values))
                try:
                    # Simple linear fit to log of absolute improvements
                    improvements = np.abs(np.diff(values))
                    log_improvements = np.log(improvements + 1e-10)
                    convergence_rate = -np.polyfit(x[1:], log_improvements, 1)[0]
                except:
                    convergence_rate = 0.0
            else:
                convergence_rate = 0.0
            
            # Scoring
            scores = []
            
            # Statistical convergence score
            if not convergence_test.rejected:
                scores.append(1.0)
            else:
                scores.append(0.3)
            
            # Improvement score
            improvement_score = min(1.0, relative_improvement / self.min_improvement_ratio)
            scores.append(improvement_score)
            
            # Stagnation score
            stagnation_score = max(0.0, 1.0 - (max_stagnation / self.max_stagnation_steps))
            scores.append(stagnation_score)
            
            # Stability score (recent stability)
            if recent_stability < 0.01:  # < 1% coefficient of variation
                stability_score = 1.0
            elif recent_stability < 0.05:  # < 5% coefficient of variation
                stability_score = 0.7
            else:
                stability_score = 0.3
            scores.append(stability_score)
            
            # Overall score
            overall_score = sum(scores) / len(scores)
            overall_pass = overall_score >= self.threshold
            
            # Determine primary failure reason
            failure_reasons = []
            if convergence_test.rejected:
                failure_reasons.append("no statistical convergence trend")
            if improvement_score < 0.5:
                failure_reasons.append(f"insufficient improvement ({relative_improvement*100:.2f}%)")
            if stagnation_score < 0.5:
                failure_reasons.append(f"excessive stagnation ({max_stagnation} steps)")
            if stability_score < 0.5:
                failure_reasons.append(f"poor recent stability (CV={recent_stability*100:.1f}%)")
            
            # Generate message
            if overall_pass:
                message = f"PASS: Convergence validated (score: {overall_score:.3f}, improvement: {relative_improvement*100:.2f}%)"
            else:
                message = f"FAIL: Convergence issues - {', '.join(failure_reasons)} (score: {overall_score:.3f})"
            
            # Add warnings
            if max_stagnation > self.max_stagnation_steps // 2:
                warnings_list.append(f"High stagnation detected: {max_stagnation} consecutive steps without improvement")
            
            if relative_improvement < self.min_improvement_ratio:
                warnings_list.append(f"Low improvement ratio: {relative_improvement*100:.3f}% < {self.min_improvement_ratio*100:.1f}%")
            
            details = {
                "convergence_history_length": len(convergence_history),
                "statistical_test": {
                    "test_name": convergence_test.test_name,
                    "statistic": convergence_test.statistic,
                    "p_value": convergence_test.p_value,
                    "rejected": convergence_test.rejected,
                    "interpretation": convergence_test.interpretation,
                },
                "improvement_metrics": {
                    "initial_value": float(initial_value),
                    "final_value": float(final_value),
                    "relative_improvement": float(relative_improvement),
                    "convergence_rate": float(convergence_rate),
                },
                "stagnation_analysis": {
                    "max_consecutive_stagnation": int(max_stagnation),
                    "stagnation_threshold": float(improvement_threshold),
                    "allowed_stagnation": self.max_stagnation_steps,
                },
                "stability_analysis": {
                    "recent_stability_cv": float(recent_stability),
                    "recent_window_size": int(recent_window),
                },
                "component_scores": {
                    "statistical_convergence": float(scores[0]),
                    "improvement": float(improvement_score),
                    "stagnation": float(stagnation_score),
                    "stability": float(stability_score),
                },
            }
            
            return self._create_result(
                overall_pass, overall_score, message, details,
                time.time() - start_time, warnings_list
            )
            
        except Exception as e:
            return self._create_result(
                False, 0.0, f"Error during convergence validation: {e}",
                {"error": str(e)}, time.time() - start_time
            )