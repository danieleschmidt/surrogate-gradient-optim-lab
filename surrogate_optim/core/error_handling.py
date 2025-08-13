"""Enhanced error handling and validation for surrogate optimization."""

import functools
import logging
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings

import jax.numpy as jnp
from jax import Array
import numpy as np

from ..models.base import Dataset


class SurrogateOptimizationError(Exception):
    """Base exception for surrogate optimization errors."""
    pass


class DataValidationError(SurrogateOptimizationError):
    """Raised when data validation fails."""
    pass


class ModelTrainingError(SurrogateOptimizationError):
    """Raised when model training fails."""
    pass


class OptimizationError(SurrogateOptimizationError):
    """Raised when optimization fails."""
    pass


class NumericalStabilityError(SurrogateOptimizationError):
    """Raised when numerical stability issues are detected."""
    pass


class ConfigurationError(SurrogateOptimizationError):
    """Raised when configuration is invalid."""
    pass


def validate_array_input(
    x: Array,
    name: str = "input",
    min_dims: int = 1,
    max_dims: int = 2,
    finite_values: bool = True,
    positive_values: bool = False,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> Array:
    """Validate array input with comprehensive checks.
    
    Args:
        x: Input array to validate
        name: Name of the parameter for error messages
        min_dims: Minimum number of dimensions
        max_dims: Maximum number of dimensions
        finite_values: Whether all values must be finite
        positive_values: Whether all values must be positive
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        
    Returns:
        Validated array
        
    Raises:
        DataValidationError: If validation fails
    """
    if not isinstance(x, (jnp.ndarray, np.ndarray)):
        try:
            x = jnp.asarray(x)
        except Exception as e:
            raise DataValidationError(f"Cannot convert {name} to array: {e}")
    
    # Ensure it's a JAX array
    x = jnp.asarray(x)
    
    # Check dimensions
    if x.ndim < min_dims:
        raise DataValidationError(
            f"{name} must have at least {min_dims} dimensions, got {x.ndim}"
        )
    
    if x.ndim > max_dims:
        raise DataValidationError(
            f"{name} must have at most {max_dims} dimensions, got {x.ndim}"
        )
    
    # Check for finite values
    if finite_values and not jnp.isfinite(x).all():
        raise DataValidationError(f"{name} contains non-finite values")
    
    # Check for positive values
    if positive_values and (x <= 0).any():
        raise DataValidationError(f"{name} must contain only positive values")
    
    # Check value bounds
    if min_value is not None and (x < min_value).any():
        raise DataValidationError(
            f"{name} contains values below minimum {min_value}"
        )
    
    if max_value is not None and (x > max_value).any():
        raise DataValidationError(
            f"{name} contains values above maximum {max_value}"
        )
    
    return x


def validate_bounds(
    bounds: List[Tuple[float, float]],
    n_dims: int,
) -> List[Tuple[float, float]]:
    """Validate optimization bounds.
    
    Args:
        bounds: List of (lower, upper) bound tuples
        n_dims: Expected number of dimensions
        
    Returns:
        Validated bounds
        
    Raises:
        DataValidationError: If bounds are invalid
    """
    if not isinstance(bounds, (list, tuple)):
        raise DataValidationError("bounds must be a list or tuple")
    
    if len(bounds) != n_dims:
        raise DataValidationError(
            f"bounds must have {n_dims} entries, got {len(bounds)}"
        )
    
    validated_bounds = []
    for i, bound in enumerate(bounds):
        if not isinstance(bound, (list, tuple)) or len(bound) != 2:
            raise DataValidationError(
                f"bounds[{i}] must be a tuple/list of length 2, got {bound}"
            )
        
        lower, upper = bound
        if not isinstance(lower, (int, float)) or not isinstance(upper, (int, float)):
            raise DataValidationError(
                f"bounds[{i}] values must be numeric, got {type(lower)}, {type(upper)}"
            )
        
        if not (jnp.isfinite(lower) and jnp.isfinite(upper)):
            raise DataValidationError(
                f"bounds[{i}] values must be finite, got ({lower}, {upper})"
            )
        
        if lower >= upper:
            raise DataValidationError(
                f"bounds[{i}] lower bound must be less than upper bound, "
                f"got ({lower}, {upper})"
            )
        
        validated_bounds.append((float(lower), float(upper)))
    
    return validated_bounds


def validate_dataset(dataset: Dataset) -> Dataset:
    """Validate dataset for training.
    
    Args:
        dataset: Dataset to validate
        
    Returns:
        Validated dataset
        
    Raises:
        DataValidationError: If dataset is invalid
    """
    if not isinstance(dataset, Dataset):
        raise DataValidationError("dataset must be a Dataset instance")
    
    # Validate inputs
    dataset.X = validate_array_input(
        dataset.X, 
        name="dataset.X",
        min_dims=2,
        max_dims=2,
        finite_values=True
    )
    
    # Validate outputs
    dataset.y = validate_array_input(
        dataset.y,
        name="dataset.y", 
        min_dims=1,
        max_dims=1,
        finite_values=True
    )
    
    # Check consistency
    if dataset.X.shape[0] != dataset.y.shape[0]:
        raise DataValidationError(
            f"Inconsistent sample counts: X={dataset.X.shape[0]}, y={dataset.y.shape[0]}"
        )
    
    # Validate gradients if present
    if dataset.gradients is not None:
        dataset.gradients = validate_array_input(
            dataset.gradients,
            name="dataset.gradients",
            min_dims=2,
            max_dims=2,
            finite_values=True
        )
        
        expected_shape = (dataset.X.shape[0], dataset.X.shape[1])
        if dataset.gradients.shape != expected_shape:
            raise DataValidationError(
                f"Invalid gradient shape: {dataset.gradients.shape}, "
                f"expected {expected_shape}"
            )
    
    # Check for minimum number of samples
    if dataset.n_samples < 2:
        raise DataValidationError(
            f"Dataset must have at least 2 samples, got {dataset.n_samples}"
        )
    
    # Check for data quality issues
    if jnp.var(dataset.y) < 1e-12:
        warnings.warn(
            "Dataset has very low output variance, model may not train well",
            UserWarning
        )
    
    # Check for duplicate samples
    if dataset.n_samples > 1:
        # Use approximate duplicate detection for efficiency
        if dataset.n_samples < 1000:
            unique_rows = jnp.unique(dataset.X, axis=0)
            if unique_rows.shape[0] < dataset.n_samples * 0.9:
                warnings.warn(
                    f"Dataset has many duplicate inputs ({unique_rows.shape[0]} unique "
                    f"out of {dataset.n_samples})", 
                    UserWarning
                )
    
    return dataset


def check_numerical_stability(
    values: Array,
    name: str = "values",
    max_condition_number: float = 1e12,
) -> None:
    """Check for numerical stability issues.
    
    Args:
        values: Array to check
        name: Name for error messages
        max_condition_number: Maximum allowed condition number for matrices
        
    Raises:
        NumericalStabilityError: If numerical issues are detected
    """
    if not jnp.isfinite(values).all():
        raise NumericalStabilityError(
            f"{name} contains non-finite values (inf/nan)"
        )
    
    # Check for very large values
    max_val = jnp.max(jnp.abs(values))
    if max_val > 1e10:
        warnings.warn(
            f"{name} contains very large values (max={max_val:.2e})", 
            UserWarning
        )
    
    # Check for very small values
    min_val = jnp.min(jnp.abs(values[values != 0]))
    if jnp.isfinite(min_val) and min_val < 1e-10:
        warnings.warn(
            f"{name} contains very small non-zero values (min={min_val:.2e})",
            UserWarning
        )
    
    # Check condition number for matrices
    if values.ndim == 2 and values.shape[0] == values.shape[1]:
        try:
            cond = jnp.linalg.cond(values)
            if cond > max_condition_number:
                warnings.warn(
                    f"{name} matrix is poorly conditioned (condition number={cond:.2e})",
                    UserWarning
                )
        except Exception:
            pass  # Skip if condition number computation fails


def robust_function_call(
    func: Callable,
    *args,
    max_retries: int = 3,
    backoff_factor: float = 1.0,
    allowed_exceptions: Tuple = (RuntimeError, ValueError),
    **kwargs
) -> Any:
    """Robust function call with retries and error handling.
    
    Args:
        func: Function to call
        *args: Function arguments
        max_retries: Maximum number of retries
        backoff_factor: Backoff factor for retry delays
        allowed_exceptions: Exception types that trigger retries
        **kwargs: Function keyword arguments
        
    Returns:
        Function result
        
    Raises:
        Exception: The last exception if all retries fail
    """
    import time
    
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except allowed_exceptions as e:
            last_exception = e
            if attempt < max_retries:
                wait_time = backoff_factor * (2 ** attempt)
                logging.warning(
                    f"Function {func.__name__} failed on attempt {attempt + 1}, "
                    f"retrying in {wait_time:.2f}s: {e}"
                )
                time.sleep(wait_time)
            else:
                logging.error(
                    f"Function {func.__name__} failed after {max_retries + 1} attempts"
                )
        except Exception as e:
            # Don't retry for non-allowed exceptions
            logging.error(f"Function {func.__name__} failed with non-retryable error: {e}")
            raise
    
    # Re-raise the last exception if all retries failed
    if last_exception is not None:
        raise last_exception


def error_boundary(
    default_return: Any = None,
    log_errors: bool = True,
    reraise_on: Tuple = (),
) -> Callable:
    """Decorator that provides error boundary functionality.
    
    Args:
        default_return: Default value to return on error
        log_errors: Whether to log errors
        reraise_on: Exception types to re-raise instead of catching
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except reraise_on:
                raise  # Re-raise specified exceptions
            except Exception as e:
                if log_errors:
                    logging.error(
                        f"Error in {func.__name__}: {e}\n{traceback.format_exc()}"
                    )
                return default_return
        return wrapper
    return decorator


def validate_optimization_config(
    surrogate_type: str,
    surrogate_params: Dict[str, Any],
    optimizer_type: str,
    optimizer_params: Dict[str, Any],
) -> Tuple[str, Dict[str, Any], str, Dict[str, Any]]:
    """Validate optimization configuration parameters.
    
    Args:
        surrogate_type: Type of surrogate model
        surrogate_params: Surrogate model parameters
        optimizer_type: Type of optimizer
        optimizer_params: Optimizer parameters
        
    Returns:
        Validated configuration tuple
        
    Raises:
        ConfigurationError: If configuration is invalid
    """
    # Validate surrogate type
    valid_surrogate_types = {
        "neural_network", "nn", 
        "gaussian_process", "gp", 
        "random_forest", "rf",
        "hybrid"
    }
    
    if surrogate_type not in valid_surrogate_types:
        raise ConfigurationError(
            f"Invalid surrogate_type '{surrogate_type}'. "
            f"Must be one of: {sorted(valid_surrogate_types)}"
        )
    
    # Validate optimizer type
    valid_optimizer_types = {
        "gradient_descent", "trust_region", "multi_start"
    }
    
    if optimizer_type not in valid_optimizer_types:
        raise ConfigurationError(
            f"Invalid optimizer_type '{optimizer_type}'. "
            f"Must be one of: {sorted(valid_optimizer_types)}"
        )
    
    # Validate surrogate parameters
    if not isinstance(surrogate_params, dict):
        raise ConfigurationError("surrogate_params must be a dictionary")
    
    # Validate optimizer parameters
    if not isinstance(optimizer_params, dict):
        raise ConfigurationError("optimizer_params must be a dictionary")
    
    # Type-specific parameter validation
    if surrogate_type in ["neural_network", "nn"]:
        _validate_neural_network_params(surrogate_params)
    elif surrogate_type in ["gaussian_process", "gp"]:
        _validate_gp_params(surrogate_params)
    elif surrogate_type in ["random_forest", "rf"]:
        _validate_rf_params(surrogate_params)
    
    if optimizer_type == "multi_start":
        _validate_multi_start_params(optimizer_params)
    elif optimizer_type == "trust_region":
        _validate_trust_region_params(optimizer_params)
    
    return surrogate_type, surrogate_params, optimizer_type, optimizer_params


def _validate_neural_network_params(params: Dict[str, Any]) -> None:
    """Validate neural network parameters."""
    if "hidden_dims" in params:
        hidden_dims = params["hidden_dims"]
        if not isinstance(hidden_dims, (list, tuple)):
            raise ConfigurationError("hidden_dims must be a list or tuple")
        if not all(isinstance(dim, int) and dim > 0 for dim in hidden_dims):
            raise ConfigurationError("hidden_dims must contain positive integers")
    
    if "learning_rate" in params:
        lr = params["learning_rate"]
        if not isinstance(lr, (int, float)) or lr <= 0:
            raise ConfigurationError("learning_rate must be a positive number")
    
    if "n_epochs" in params:
        epochs = params["n_epochs"]
        if not isinstance(epochs, int) or epochs <= 0:
            raise ConfigurationError("n_epochs must be a positive integer")


def _validate_gp_params(params: Dict[str, Any]) -> None:
    """Validate Gaussian process parameters."""
    if "length_scale" in params:
        ls = params["length_scale"]
        if not isinstance(ls, (int, float)) or ls <= 0:
            raise ConfigurationError("length_scale must be a positive number")
    
    if "noise_level" in params:
        noise = params["noise_level"]
        if not isinstance(noise, (int, float)) or noise <= 0:
            raise ConfigurationError("noise_level must be a positive number")


def _validate_rf_params(params: Dict[str, Any]) -> None:
    """Validate random forest parameters."""
    if "n_estimators" in params:
        n_est = params["n_estimators"]
        if not isinstance(n_est, int) or n_est <= 0:
            raise ConfigurationError("n_estimators must be a positive integer")
    
    if "max_depth" in params:
        depth = params["max_depth"]
        if depth is not None and (not isinstance(depth, int) or depth <= 0):
            raise ConfigurationError("max_depth must be None or a positive integer")


def _validate_multi_start_params(params: Dict[str, Any]) -> None:
    """Validate multi-start optimizer parameters."""
    if "n_starts" in params:
        n_starts = params["n_starts"]
        if not isinstance(n_starts, int) or n_starts <= 0:
            raise ConfigurationError("n_starts must be a positive integer")


def _validate_trust_region_params(params: Dict[str, Any]) -> None:
    """Validate trust region optimizer parameters."""
    if "initial_radius" in params:
        radius = params["initial_radius"]
        if not isinstance(radius, (int, float)) or radius <= 0:
            raise ConfigurationError("initial_radius must be a positive number")