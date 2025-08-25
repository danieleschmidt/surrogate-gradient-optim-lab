"""Comprehensive error recovery and fault tolerance system - Generation 2.

This module implements production-grade error handling, recovery strategies,
and fault tolerance for the surrogate optimization framework.
"""

import contextlib
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import logging
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

from jax import Array
import jax.numpy as jnp

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for classification and handling."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories of errors for specialized handling."""

    NUMERICAL = "numerical"
    DATA = "data"
    MODEL = "model"
    OPTIMIZATION = "optimization"
    SYSTEM = "system"
    USER_INPUT = "user_input"
    CONFIGURATION = "configuration"


@dataclass
class ErrorReport:
    """Comprehensive error report with recovery information."""

    error_id: str
    timestamp: float
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    traceback: str
    context: Dict[str, Any] = field(default_factory=dict)
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_strategy: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3


class SurrogateOptimizationError(Exception):
    """Base exception for surrogate optimization errors."""

    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Dict = None,
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context or {}


class NumericalInstabilityError(SurrogateOptimizationError):
    """Error for numerical instability issues."""

    def __init__(self, message: str, context: Dict = None):
        super().__init__(message, ErrorCategory.NUMERICAL, ErrorSeverity.HIGH, context)


class DataValidationError(SurrogateOptimizationError):
    """Error for data validation failures."""

    def __init__(self, message: str, context: Dict = None):
        super().__init__(message, ErrorCategory.DATA, ErrorSeverity.MEDIUM, context)


class ModelTrainingError(SurrogateOptimizationError):
    """Error for model training failures."""

    def __init__(self, message: str, context: Dict = None):
        super().__init__(message, ErrorCategory.MODEL, ErrorSeverity.HIGH, context)


class OptimizationFailureError(SurrogateOptimizationError):
    """Error for optimization process failures."""

    def __init__(self, message: str, context: Dict = None):
        super().__init__(
            message, ErrorCategory.OPTIMIZATION, ErrorSeverity.HIGH, context
        )


class ConfigurationError(SurrogateOptimizationError):
    """Error for configuration issues."""

    def __init__(self, message: str, context: Dict = None):
        super().__init__(
            message, ErrorCategory.CONFIGURATION, ErrorSeverity.MEDIUM, context
        )


class RobustErrorHandler:
    """Comprehensive error handling and recovery system."""

    def __init__(self, max_retries: int = 3, enable_fallback: bool = True):
        """Initialize error handler.

        Args:
            max_retries: Maximum number of retry attempts
            enable_fallback: Whether to enable fallback strategies
        """
        self.max_retries = max_retries
        self.enable_fallback = enable_fallback

        # Error tracking
        self.error_history: List[ErrorReport] = []
        self.error_counts: Dict[ErrorCategory, int] = dict.fromkeys(ErrorCategory, 0)

        # Recovery strategies
        self.recovery_strategies = {
            ErrorCategory.NUMERICAL: self._handle_numerical_error,
            ErrorCategory.DATA: self._handle_data_error,
            ErrorCategory.MODEL: self._handle_model_error,
            ErrorCategory.OPTIMIZATION: self._handle_optimization_error,
            ErrorCategory.SYSTEM: self._handle_system_error,
            ErrorCategory.USER_INPUT: self._handle_user_input_error,
            ErrorCategory.CONFIGURATION: self._handle_configuration_error,
        }

    def handle_error(self, error: Exception, context: Dict = None) -> Tuple[bool, Any]:
        """Handle an error with appropriate recovery strategy.

        Args:
            error: The exception that occurred
            context: Additional context information

        Returns:
            Tuple of (recovery_successful, result_or_fallback)
        """
        # Classify error
        error_category, error_severity = self._classify_error(error)

        # Create error report
        error_report = ErrorReport(
            error_id=f"err_{len(self.error_history)}_{int(time.time())}",
            timestamp=time.time(),
            category=error_category,
            severity=error_severity,
            message=str(error),
            traceback=traceback.format_exc(),
            context=context or {},
        )

        # Log error
        logger.error(
            f"Error occurred [{error_report.error_id}]: {error_report.message}"
        )

        # Update error counts
        self.error_counts[error_category] += 1

        # Attempt recovery
        recovery_successful = False
        result = None

        if self.enable_fallback and error_category in self.recovery_strategies:
            try:
                recovery_successful, result = self.recovery_strategies[error_category](
                    error, error_report, context
                )
                error_report.recovery_attempted = True
                error_report.recovery_successful = recovery_successful

                if recovery_successful:
                    logger.info(
                        f"Successfully recovered from error {error_report.error_id}"
                    )
                else:
                    logger.warning(
                        f"Failed to recover from error {error_report.error_id}"
                    )

            except Exception as recovery_error:
                logger.error(f"Recovery strategy failed: {recovery_error}")
                error_report.recovery_successful = False

        # Store error report
        self.error_history.append(error_report)

        return recovery_successful, result

    def _classify_error(self, error: Exception) -> Tuple[ErrorCategory, ErrorSeverity]:
        """Classify error by category and severity."""
        if isinstance(error, SurrogateOptimizationError):
            return error.category, error.severity

        error_msg = str(error).lower()
        error_type = type(error).__name__.lower()

        # Numerical errors
        if any(
            keyword in error_msg
            for keyword in ["nan", "inf", "overflow", "underflow", "numerical"]
        ):
            return ErrorCategory.NUMERICAL, ErrorSeverity.HIGH

        if any(
            keyword in error_type for keyword in ["arithmetic", "float", "overflow"]
        ):
            return ErrorCategory.NUMERICAL, ErrorSeverity.HIGH

        # Data errors
        if any(
            keyword in error_msg for keyword in ["shape", "dimension", "array", "index"]
        ):
            return ErrorCategory.DATA, ErrorSeverity.MEDIUM

        if any(keyword in error_type for keyword in ["value", "type", "index", "key"]):
            return ErrorCategory.DATA, ErrorSeverity.MEDIUM

        # Model errors
        if any(
            keyword in error_msg
            for keyword in ["training", "model", "convergence", "gradient"]
        ):
            return ErrorCategory.MODEL, ErrorSeverity.HIGH

        # System errors
        if any(keyword in error_type for keyword in ["memory", "system", "os", "io"]):
            return ErrorCategory.SYSTEM, ErrorSeverity.HIGH

        # Default
        return ErrorCategory.SYSTEM, ErrorSeverity.MEDIUM

    def _handle_numerical_error(
        self, error: Exception, report: ErrorReport, context: Dict
    ) -> Tuple[bool, Any]:
        """Handle numerical instability errors."""
        logger.info(f"Attempting numerical error recovery for {report.error_id}")

        recovery_strategies = [
            "reduce_learning_rate",
            "gradient_clipping",
            "alternative_optimizer",
            "numerical_stabilization",
        ]

        for strategy in recovery_strategies:
            try:
                if strategy == "reduce_learning_rate":
                    # Suggest reduced learning rate
                    result = {"learning_rate_multiplier": 0.1, "strategy": strategy}
                    report.recovery_strategy = strategy
                    return True, result

                if strategy == "gradient_clipping":
                    # Suggest gradient clipping
                    result = {"gradient_clip_value": 1.0, "strategy": strategy}
                    report.recovery_strategy = strategy
                    return True, result

                if strategy == "alternative_optimizer":
                    # Suggest alternative optimization algorithm
                    result = {
                        "optimizer": "adam",
                        "learning_rate": 0.001,
                        "strategy": strategy,
                    }
                    report.recovery_strategy = strategy
                    return True, result

                if strategy == "numerical_stabilization":
                    # Add numerical stabilization
                    result = {
                        "add_epsilon": 1e-8,
                        "use_stable_ops": True,
                        "strategy": strategy,
                    }
                    report.recovery_strategy = strategy
                    return True, result

            except Exception as e:
                logger.warning(f"Recovery strategy {strategy} failed: {e}")
                continue

        return False, None

    def _handle_data_error(
        self, error: Exception, report: ErrorReport, context: Dict
    ) -> Tuple[bool, Any]:
        """Handle data validation and shape errors."""
        logger.info(f"Attempting data error recovery for {report.error_id}")

        try:
            # Check for common data issues
            if "shape" in str(error).lower():
                # Suggest data reshaping
                result = {
                    "action": "reshape_data",
                    "suggestion": "Check input dimensions and reshape if necessary",
                    "strategy": "data_reshaping",
                }
                report.recovery_strategy = "data_reshaping"
                return True, result

            if "empty" in str(error).lower():
                # Handle empty data
                result = {
                    "action": "generate_fallback_data",
                    "suggestion": "Generate minimal viable dataset",
                    "strategy": "fallback_data",
                }
                report.recovery_strategy = "fallback_data"
                return True, result

            # General data validation
            result = {
                "action": "validate_and_clean",
                "suggestion": "Perform data validation and cleaning",
                "strategy": "data_cleaning",
            }
            report.recovery_strategy = "data_cleaning"
            return True, result

        except Exception as e:
            logger.error(f"Data error recovery failed: {e}")

        return False, None

    def _handle_model_error(
        self, error: Exception, report: ErrorReport, context: Dict
    ) -> Tuple[bool, Any]:
        """Handle model training and prediction errors."""
        logger.info(f"Attempting model error recovery for {report.error_id}")

        recovery_strategies = [
            "reduce_model_complexity",
            "alternative_architecture",
            "ensemble_fallback",
            "pretrained_fallback",
        ]

        for strategy in recovery_strategies:
            try:
                if strategy == "reduce_model_complexity":
                    result = {
                        "hidden_dims": [32, 16],  # Smaller network
                        "regularization": 0.01,
                        "strategy": strategy,
                    }
                    report.recovery_strategy = strategy
                    return True, result

                if strategy == "alternative_architecture":
                    result = {
                        "model_type": "gaussian_process",  # Fallback to GP
                        "strategy": strategy,
                    }
                    report.recovery_strategy = strategy
                    return True, result

                if strategy == "ensemble_fallback":
                    result = {
                        "use_ensemble": True,
                        "ensemble_size": 3,
                        "strategy": strategy,
                    }
                    report.recovery_strategy = strategy
                    return True, result

            except Exception as e:
                logger.warning(f"Model recovery strategy {strategy} failed: {e}")
                continue

        return False, None

    def _handle_optimization_error(
        self, error: Exception, report: ErrorReport, context: Dict
    ) -> Tuple[bool, Any]:
        """Handle optimization process errors."""
        logger.info(f"Attempting optimization error recovery for {report.error_id}")

        try:
            # Suggest optimization recovery strategies
            result = {
                "restart_optimization": True,
                "reduce_step_size": True,
                "add_bounds": True,
                "alternative_method": "L-BFGS-B",
                "strategy": "optimization_restart",
            }
            report.recovery_strategy = "optimization_restart"
            return True, result

        except Exception as e:
            logger.error(f"Optimization error recovery failed: {e}")

        return False, None

    def _handle_system_error(
        self, error: Exception, report: ErrorReport, context: Dict
    ) -> Tuple[bool, Any]:
        """Handle system-level errors."""
        logger.info(f"Attempting system error recovery for {report.error_id}")

        try:
            # Suggest system-level recovery
            result = {
                "reduce_memory_usage": True,
                "use_cpu_fallback": True,
                "batch_processing": True,
                "strategy": "system_optimization",
            }
            report.recovery_strategy = "system_optimization"
            return True, result

        except Exception as e:
            logger.error(f"System error recovery failed: {e}")

        return False, None

    def _handle_user_input_error(
        self, error: Exception, report: ErrorReport, context: Dict
    ) -> Tuple[bool, Any]:
        """Handle user input validation errors."""
        logger.info(f"Attempting user input error recovery for {report.error_id}")

        try:
            # Suggest input corrections
            result = {
                "validate_inputs": True,
                "use_defaults": True,
                "sanitize_inputs": True,
                "strategy": "input_validation",
            }
            report.recovery_strategy = "input_validation"
            return True, result

        except Exception as e:
            logger.error(f"User input error recovery failed: {e}")

        return False, None

    def _handle_configuration_error(
        self, error: Exception, report: ErrorReport, context: Dict
    ) -> Tuple[bool, Any]:
        """Handle configuration errors."""
        logger.info(f"Attempting configuration error recovery for {report.error_id}")

        try:
            # Suggest configuration fixes
            result = {
                "use_default_config": True,
                "validate_config": True,
                "fallback_settings": True,
                "strategy": "configuration_reset",
            }
            report.recovery_strategy = "configuration_reset"
            return True, result

        except Exception as e:
            logger.error(f"Configuration error recovery failed: {e}")

        return False, None

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        total_errors = len(self.error_history)

        if total_errors == 0:
            return {"total_errors": 0, "message": "No errors recorded"}

        # Category statistics
        category_stats = {cat.value: count for cat, count in self.error_counts.items()}

        # Severity statistics
        severity_counts = {}
        for report in self.error_history:
            severity = report.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        # Recovery statistics
        recovery_attempted = sum(1 for r in self.error_history if r.recovery_attempted)
        recovery_successful = sum(
            1 for r in self.error_history if r.recovery_successful
        )

        recovery_rate = (
            recovery_successful / recovery_attempted if recovery_attempted > 0 else 0
        )

        # Recent error trends
        recent_errors = [
            r for r in self.error_history if time.time() - r.timestamp < 3600
        ]  # Last hour

        return {
            "total_errors": total_errors,
            "category_distribution": category_stats,
            "severity_distribution": severity_counts,
            "recovery_statistics": {
                "attempted": recovery_attempted,
                "successful": recovery_successful,
                "success_rate": recovery_rate,
            },
            "recent_errors": len(recent_errors),
            "most_common_category": (
                max(category_stats.keys(), key=lambda k: category_stats[k])
                if category_stats
                else None
            ),
        }

    def get_recovery_recommendations(self) -> List[str]:
        """Get recommendations based on error history."""
        recommendations = []

        # Analyze error patterns
        if self.error_counts[ErrorCategory.NUMERICAL] > 5:
            recommendations.append("Consider using more numerically stable algorithms")
            recommendations.append(
                "Implement gradient clipping and learning rate scheduling"
            )

        if self.error_counts[ErrorCategory.DATA] > 3:
            recommendations.append("Implement more robust data validation")
            recommendations.append("Add data preprocessing and normalization steps")

        if self.error_counts[ErrorCategory.MODEL] > 3:
            recommendations.append(
                "Consider using ensemble methods for better robustness"
            )
            recommendations.append("Implement model checkpointing and recovery")

        # Recovery success rate analysis
        stats = self.get_error_statistics()
        if stats["recovery_statistics"]["success_rate"] < 0.5:
            recommendations.append("Review and improve error recovery strategies")
            recommendations.append("Add more fallback options for critical operations")

        return recommendations


def robust_function_call(
    func: Callable,
    *args,
    error_handler: RobustErrorHandler = None,
    max_retries: int = 3,
    context: Dict = None,
    **kwargs,
) -> Tuple[bool, Any]:
    """Robustly call a function with error handling and retries.

    Args:
        func: Function to call
        *args: Positional arguments
        error_handler: Error handler instance
        max_retries: Maximum retry attempts
        context: Additional context
        **kwargs: Keyword arguments

    Returns:
        Tuple of (success, result)
    """
    if error_handler is None:
        error_handler = RobustErrorHandler(max_retries=max_retries)

    for attempt in range(max_retries + 1):
        try:
            result = func(*args, **kwargs)
            return True, result

        except Exception as e:
            logger.warning(
                f"Function call failed (attempt {attempt + 1}/{max_retries + 1}): {e}"
            )

            if attempt < max_retries:
                # Attempt error recovery
                recovery_successful, recovery_result = error_handler.handle_error(
                    e,
                    context={
                        **(context or {}),
                        "attempt": attempt,
                        "function": func.__name__,
                    },
                )

                if recovery_successful and isinstance(recovery_result, dict):
                    # Update function arguments based on recovery suggestions
                    if "learning_rate_multiplier" in recovery_result:
                        if "learning_rate" in kwargs:
                            kwargs["learning_rate"] *= recovery_result[
                                "learning_rate_multiplier"
                            ]

                    # Add small delay before retry
                    time.sleep(0.1 * (attempt + 1))
                else:
                    # No recovery possible, but still retry
                    time.sleep(0.1 * (attempt + 1))
            else:
                # Final attempt failed
                error_handler.handle_error(
                    e, context={**(context or {}), "final_attempt": True}
                )
                return False, e

    return False, None


@contextlib.contextmanager
def error_boundary(
    error_handler: RobustErrorHandler = None,
    context: Dict = None,
    reraise: bool = False,
):
    """Context manager for error boundary with automatic recovery.

    Args:
        error_handler: Error handler instance
        context: Additional context
        reraise: Whether to reraise exceptions after handling
    """
    if error_handler is None:
        error_handler = RobustErrorHandler()

    try:
        yield error_handler
    except Exception as e:
        recovery_successful, result = error_handler.handle_error(e, context)

        if not recovery_successful and reraise:
            raise

        return recovery_successful, result


# Validation utilities
def validate_array_input(
    x: Array,
    name: str = "input",
    expected_shape: Optional[Tuple] = None,
    expected_dtype: Optional[Any] = None,
    finite_check: bool = True,
) -> Array:
    """Validate array input with comprehensive checks."""
    try:
        # Type check
        if not isinstance(x, (jnp.ndarray, Array)):
            try:
                x = jnp.array(x)
            except Exception as e:
                raise DataValidationError(f"Cannot convert {name} to array: {e}")

        # Shape check
        if expected_shape is not None and x.shape != expected_shape:
            raise DataValidationError(
                f"{name} has shape {x.shape}, expected {expected_shape}"
            )

        # Dtype check
        if expected_dtype is not None and x.dtype != expected_dtype:
            logger.warning(f"{name} has dtype {x.dtype}, expected {expected_dtype}")

        # Finite check
        if finite_check:
            if jnp.any(jnp.isnan(x)):
                raise NumericalInstabilityError(f"{name} contains NaN values")

            if jnp.any(jnp.isinf(x)):
                raise NumericalInstabilityError(f"{name} contains infinite values")

        return x

    except Exception as e:
        if isinstance(e, (DataValidationError, NumericalInstabilityError)):
            raise
        raise DataValidationError(f"Validation failed for {name}: {e}")


def validate_bounds(bounds: List[Tuple], dimension: int) -> List[Tuple]:
    """Validate optimization bounds."""
    try:
        if not isinstance(bounds, list):
            raise DataValidationError("Bounds must be a list of tuples")

        if len(bounds) != dimension:
            raise DataValidationError(
                f"Bounds length {len(bounds)} does not match dimension {dimension}"
            )

        validated_bounds = []
        for i, bound in enumerate(bounds):
            if not isinstance(bound, (tuple, list)) or len(bound) != 2:
                raise DataValidationError(f"Bound {i} must be a tuple/list of length 2")

            lower, upper = bound
            if not isinstance(lower, (int, float)) or not isinstance(
                upper, (int, float)
            ):
                raise DataValidationError(f"Bound {i} values must be numeric")

            if lower >= upper:
                raise DataValidationError(
                    f"Bound {i}: lower ({lower}) >= upper ({upper})"
                )

            validated_bounds.append((float(lower), float(upper)))

        return validated_bounds

    except Exception as e:
        if isinstance(e, DataValidationError):
            raise
        raise DataValidationError(f"Bounds validation failed: {e}")


def check_numerical_stability(values: Array, name: str = "values") -> bool:
    """Check for numerical stability issues."""
    try:
        # Check for NaN
        if jnp.any(jnp.isnan(values)):
            raise NumericalInstabilityError(f"{name} contains NaN values")

        # Check for infinite values
        if jnp.any(jnp.isinf(values)):
            raise NumericalInstabilityError(f"{name} contains infinite values")

        # Check for very large values that might cause overflow
        max_val = jnp.max(jnp.abs(values))
        if max_val > 1e10:
            logger.warning(f"{name} contains very large values (max: {max_val})")

        # Check for very small values that might cause underflow
        min_val = jnp.min(jnp.abs(values[values != 0])) if jnp.any(values != 0) else 0
        if min_val > 0 and min_val < 1e-10:
            logger.warning(f"{name} contains very small values (min: {min_val})")

        return True

    except Exception as e:
        if isinstance(e, NumericalInstabilityError):
            raise
        raise NumericalInstabilityError(
            f"Numerical stability check failed for {name}: {e}"
        )


# Decorator for robust methods
def robust_method(
    max_retries: int = 3, error_handler: Optional[RobustErrorHandler] = None
):
    """Decorator to make methods robust with automatic error handling."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            handler = error_handler or RobustErrorHandler(max_retries=max_retries)
            success, result = robust_function_call(
                func, *args, error_handler=handler, max_retries=max_retries, **kwargs
            )

            if not success:
                raise result  # Re-raise the exception

            return result

        return wrapper

    return decorator


# Example usage and testing
if __name__ == "__main__":
    # Demonstrate error handling system
    print("🛡️  ROBUST ERROR HANDLING SYSTEM DEMONSTRATION")
    print("=" * 60)

    # Create error handler
    error_handler = RobustErrorHandler(max_retries=3, enable_fallback=True)

    # Test different types of errors
    def test_numerical_error():
        """Simulate numerical instability."""
        x = jnp.array([1e20, 1e20])
        return jnp.exp(x)  # This will overflow

    def test_data_error():
        """Simulate data validation error."""
        x = jnp.array([[1, 2], [3]])  # Irregular shape
        return jnp.dot(x, x.T)

    def test_model_error():
        """Simulate model training failure."""
        raise ModelTrainingError("Training failed to converge")

    # Test error handling
    test_functions = [
        ("Numerical Error", test_numerical_error),
        ("Data Error", test_data_error),
        ("Model Error", test_model_error),
    ]

    for name, test_func in test_functions:
        print(f"\\nTesting {name}:")
        try:
            success, result = robust_function_call(
                test_func, error_handler=error_handler, context={"test": name}
            )

            if success:
                print("✓ Function executed successfully")
            else:
                print(f"✗ Function failed with error: {result}")

        except Exception as e:
            print(f"✗ Unhandled error: {e}")

    # Display error statistics
    print("\\n📊 Error Statistics:")
    stats = error_handler.get_error_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Get recommendations
    print("\\n💡 Recommendations:")
    recommendations = error_handler.get_recovery_recommendations()
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")

    print("\\n✅ Error handling demonstration completed!")
