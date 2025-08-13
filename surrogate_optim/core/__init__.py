"""Core optimization components with enhanced robustness."""

# Import original SurrogateOptimizer - disabled due to circular import
# We'll import it directly from the file when needed
from .enhanced_optimizer import EnhancedSurrogateOptimizer
from .error_handling import (
    SurrogateOptimizationError,
    DataValidationError,
    ModelTrainingError,
    OptimizationError,
    NumericalStabilityError,
    ConfigurationError,
    validate_array_input,
    validate_bounds,
    validate_dataset,
    check_numerical_stability,
    robust_function_call,
    error_boundary,
)

__all__ = [
    "EnhancedSurrogateOptimizer",
    "SurrogateOptimizationError",
    "DataValidationError", 
    "ModelTrainingError",
    "OptimizationError",
    "NumericalStabilityError",
    "ConfigurationError",
    "validate_array_input",
    "validate_bounds",
    "validate_dataset",
    "check_numerical_stability",
    "robust_function_call",
    "error_boundary",
]