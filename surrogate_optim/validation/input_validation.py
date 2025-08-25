"""Input validation utilities."""

from typing import Any, Callable, Dict, List, Optional, Tuple
import warnings

from jax import Array
import jax.numpy as jnp

from ..models.base import Dataset


class ValidationError(Exception):
    """Raised when validation fails with critical issues."""
    pass


class ValidationWarning(UserWarning):
    """Warning for non-critical validation issues."""
    pass


def validate_bounds(bounds: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Validate optimization bounds.
    
    Args:
        bounds: List of (min, max) tuples for each dimension
        
    Returns:
        Validated bounds
        
    Raises:
        ValidationError: If bounds are invalid
    """
    if not isinstance(bounds, (list, tuple)):
        raise ValidationError("Bounds must be a list or tuple")

    if len(bounds) == 0:
        raise ValidationError("Bounds cannot be empty")

    validated_bounds = []

    for i, bound in enumerate(bounds):
        if not isinstance(bound, (list, tuple)) or len(bound) != 2:
            raise ValidationError(f"Bound {i} must be a 2-tuple (min, max)")

        lower, upper = bound

        if not isinstance(lower, (int, float)) or not isinstance(upper, (int, float)):
            raise ValidationError(f"Bound {i} values must be numeric")

        if not jnp.isfinite(lower) or not jnp.isfinite(upper):
            raise ValidationError(f"Bound {i} values must be finite")

        if lower >= upper:
            raise ValidationError(f"Bound {i}: lower bound {lower} >= upper bound {upper}")

        # Check for suspiciously large ranges
        range_size = upper - lower
        if range_size > 1e6:
            warnings.warn(
                f"Bound {i} has very large range {range_size:.2e}. "
                "Consider rescaling your problem.",
                ValidationWarning
            )

        validated_bounds.append((float(lower), float(upper)))

    return validated_bounds


def validate_dataset(dataset: Dataset) -> Dataset:
    """Validate a dataset for surrogate training.
    
    Args:
        dataset: Dataset to validate
        
    Returns:
        Validated dataset
        
    Raises:
        ValidationError: If dataset is invalid
    """
    if not isinstance(dataset, Dataset):
        raise ValidationError("Input must be a Dataset instance")

    # Check basic structure
    if dataset.n_samples == 0:
        raise ValidationError("Dataset cannot be empty")

    if dataset.n_dims == 0:
        raise ValidationError("Dataset must have at least one input dimension")

    # Check for NaN or infinite values in inputs
    if not jnp.all(jnp.isfinite(dataset.X)):
        nan_count = jnp.sum(~jnp.isfinite(dataset.X))
        raise ValidationError(f"Dataset.X contains {nan_count} non-finite values")

    # Check for NaN or infinite values in outputs
    if not jnp.all(jnp.isfinite(dataset.y)):
        nan_count = jnp.sum(~jnp.isfinite(dataset.y))
        raise ValidationError(f"Dataset.y contains {nan_count} non-finite values")

    # Check gradients if present
    if dataset.gradients is not None:
        if not jnp.all(jnp.isfinite(dataset.gradients)):
            nan_count = jnp.sum(~jnp.isfinite(dataset.gradients))
            warnings.warn(
                f"Dataset.gradients contains {nan_count} non-finite values. "
                "These will be ignored during training.",
                ValidationWarning
            )

    # Check for sufficient sample size
    min_samples = max(10, dataset.n_dims * 5)
    if dataset.n_samples < min_samples:
        warnings.warn(
            f"Dataset has only {dataset.n_samples} samples for {dataset.n_dims} dimensions. "
            f"Consider collecting at least {min_samples} samples for better surrogate quality.",
            ValidationWarning
        )

    # Check for duplicate points
    unique_points, unique_indices = jnp.unique(
        dataset.X, axis=0, return_index=True, size=dataset.n_samples, fill_value=jnp.nan
    )
    n_unique = len(unique_indices)

    if n_unique < dataset.n_samples:
        n_duplicates = dataset.n_samples - n_unique
        warnings.warn(
            f"Dataset contains {n_duplicates} duplicate input points. "
            "This may reduce surrogate model effectiveness.",
            ValidationWarning
        )

    # Check input range and scaling
    X_min = jnp.min(dataset.X, axis=0)
    X_max = jnp.max(dataset.X, axis=0)
    X_range = X_max - X_min

    # Warn about very small ranges (might indicate insufficient exploration)
    small_range_dims = jnp.where(X_range < 1e-6)[0]
    if len(small_range_dims) > 0:
        warnings.warn(
            f"Input dimensions {small_range_dims.tolist()} have very small ranges. "
            "Consider expanding the exploration region.",
            ValidationWarning
        )

    # Warn about very different scales
    range_ratios = X_range / jnp.median(X_range)
    large_ratio_dims = jnp.where(range_ratios > 100)[0]
    if len(large_ratio_dims) > 0:
        warnings.warn(
            f"Input dimensions {large_ratio_dims.tolist()} have much larger scales "
            "than others. Consider normalizing inputs for better surrogate performance.",
            ValidationWarning
        )

    # Check output distribution
    y_std = jnp.std(dataset.y)
    if y_std < 1e-10:
        warnings.warn(
            "Output values have very small variance. The function might be nearly constant "
            "in the sampled region, making optimization trivial.",
            ValidationWarning
        )

    return dataset


def validate_surrogate_config(
    surrogate_type: str,
    surrogate_params: Dict[str, Any]
) -> Tuple[str, Dict[str, Any]]:
    """Validate surrogate model configuration.
    
    Args:
        surrogate_type: Type of surrogate model
        surrogate_params: Parameters for surrogate model
        
    Returns:
        Validated surrogate type and parameters
        
    Raises:
        ValidationError: If configuration is invalid
    """
    valid_types = ["neural_network", "nn", "gaussian_process", "gp", "random_forest", "rf"]

    if surrogate_type not in valid_types:
        raise ValidationError(f"Unknown surrogate type: {surrogate_type}. Valid types: {valid_types}")

    validated_params = surrogate_params.copy()

    # Type-specific validation
    if surrogate_type in ["neural_network", "nn"]:
        if "hidden_dims" in validated_params:
            hidden_dims = validated_params["hidden_dims"]
            if not isinstance(hidden_dims, (list, tuple)) or len(hidden_dims) == 0:
                raise ValidationError("hidden_dims must be a non-empty list/tuple")

            if not all(isinstance(dim, int) and dim > 0 for dim in hidden_dims):
                raise ValidationError("All hidden_dims must be positive integers")

            # Warn about very large networks
            total_params = sum(hidden_dims) * (len(hidden_dims) + 1)
            if total_params > 10000:
                warnings.warn(
                    f"Neural network has ~{total_params} parameters. "
                    "This might be too complex for limited training data.",
                    ValidationWarning
                )

        if "learning_rate" in validated_params:
            lr = validated_params["learning_rate"]
            if not isinstance(lr, (int, float)) or lr <= 0:
                raise ValidationError("learning_rate must be positive")

            if lr > 0.1:
                warnings.warn(
                    f"Learning rate {lr} is quite high. Consider using a smaller value.",
                    ValidationWarning
                )

        if "n_epochs" in validated_params:
            epochs = validated_params["n_epochs"]
            if not isinstance(epochs, int) or epochs <= 0:
                raise ValidationError("n_epochs must be a positive integer")

            if epochs > 10000:
                warnings.warn(
                    f"Training for {epochs} epochs might take very long.",
                    ValidationWarning
                )

    elif surrogate_type in ["gaussian_process", "gp"]:
        if "kernel" in validated_params:
            kernel = validated_params["kernel"]
            valid_kernels = ["rbf", "matern32", "matern52", "auto"]
            if kernel not in valid_kernels:
                raise ValidationError(f"Unknown kernel: {kernel}. Valid kernels: {valid_kernels}")

        if "length_scale" in validated_params:
            ls = validated_params["length_scale"]
            if not isinstance(ls, (int, float)) or ls <= 0:
                raise ValidationError("length_scale must be positive")

    elif surrogate_type in ["random_forest", "rf"]:
        if "n_estimators" in validated_params:
            n_est = validated_params["n_estimators"]
            if not isinstance(n_est, int) or n_est <= 0:
                raise ValidationError("n_estimators must be a positive integer")

            if n_est > 1000:
                warnings.warn(
                    f"Using {n_est} estimators might be slow. Consider a smaller number.",
                    ValidationWarning
                )

        if "max_depth" in validated_params:
            depth = validated_params["max_depth"]
            if depth is not None and (not isinstance(depth, int) or depth <= 0):
                raise ValidationError("max_depth must be None or a positive integer")

    return surrogate_type, validated_params


def validate_optimization_inputs(
    x0: Array,
    bounds: Optional[List[Tuple[float, float]]] = None,
    method: str = "gradient_descent",
    options: Optional[Dict[str, Any]] = None,
) -> Tuple[Array, Optional[List[Tuple[float, float]]], str, Dict[str, Any]]:
    """Validate optimization inputs.
    
    Args:
        x0: Initial point
        bounds: Optional bounds
        method: Optimization method
        options: Optional method parameters
        
    Returns:
        Validated inputs
        
    Raises:
        ValidationError: If inputs are invalid
    """
    # Validate initial point
    if not isinstance(x0, (jnp.ndarray, list, tuple)):
        raise ValidationError("x0 must be array-like")

    x0 = jnp.asarray(x0)

    if x0.ndim != 1:
        raise ValidationError("x0 must be 1-dimensional")

    if len(x0) == 0:
        raise ValidationError("x0 cannot be empty")

    if not jnp.all(jnp.isfinite(x0)):
        raise ValidationError("x0 contains non-finite values")

    # Validate bounds
    validated_bounds = None
    if bounds is not None:
        validated_bounds = validate_bounds(bounds)

        if len(validated_bounds) != len(x0):
            raise ValidationError(f"Bounds length {len(validated_bounds)} != x0 length {len(x0)}")

        # Check if x0 is within bounds
        for i, (lower, upper) in enumerate(validated_bounds):
            if not (lower <= x0[i] <= upper):
                raise ValidationError(f"x0[{i}] = {x0[i]} violates bounds [{lower}, {upper}]")

    # Validate method
    valid_methods = ["gradient_descent", "trust_region", "multi_start"]
    if method not in valid_methods:
        raise ValidationError(f"Unknown method: {method}. Valid methods: {valid_methods}")

    # Validate options
    validated_options = options.copy() if options else {}

    # Common option validation
    if "max_iterations" in validated_options:
        max_iter = validated_options["max_iterations"]
        if not isinstance(max_iter, int) or max_iter <= 0:
            raise ValidationError("max_iterations must be a positive integer")

        if max_iter > 10000:
            warnings.warn(
                f"max_iterations = {max_iter} is very large. "
                "Optimization might take a long time.",
                ValidationWarning
            )

    if "tolerance" in validated_options:
        tol = validated_options["tolerance"]
        if not isinstance(tol, (int, float)) or tol <= 0:
            raise ValidationError("tolerance must be positive")

        if tol > 0.1:
            warnings.warn(
                f"tolerance = {tol} is quite large. "
                "Optimization might terminate prematurely.",
                ValidationWarning
            )

    # Method-specific validation
    if method == "trust_region":
        if "initial_radius" in validated_options:
            radius = validated_options["initial_radius"]
            if not isinstance(radius, (int, float)) or radius <= 0:
                raise ValidationError("initial_radius must be positive")

    elif method == "multi_start":
        if "n_starts" in validated_options:
            n_starts = validated_options["n_starts"]
            if not isinstance(n_starts, int) or n_starts <= 0:
                raise ValidationError("n_starts must be a positive integer")

            if n_starts > 100:
                warnings.warn(
                    f"n_starts = {n_starts} is quite large. "
                    "This might take a long time.",
                    ValidationWarning
                )

        if bounds is None:
            raise ValidationError("Multi-start optimization requires bounds")

    return x0, validated_bounds, method, validated_options


def validate_function(
    function: Callable[[Array], float],
    test_points: Optional[List[Array]] = None,
    bounds: Optional[List[Tuple[float, float]]] = None,
) -> Callable[[Array], float]:
    """Validate a black-box function.
    
    Args:
        function: Function to validate
        test_points: Optional test points for validation
        bounds: Optional bounds for generating test points
        
    Returns:
        Validated function (potentially wrapped for error handling)
        
    Raises:
        ValidationError: If function is invalid
    """
    if not callable(function):
        raise ValidationError("Function must be callable")

    # Generate test points if not provided
    if test_points is None and bounds is not None:
        from jax import random
        key = random.PRNGKey(42)
        n_dims = len(bounds)
        uniform_samples = random.uniform(key, shape=(5, n_dims))

        test_points = []
        for i in range(5):
            point = jnp.zeros(n_dims)
            for j, (lower, upper) in enumerate(bounds):
                point = point.at[j].set(lower + (upper - lower) * uniform_samples[i, j])
            test_points.append(point)

    if test_points is None:
        test_points = [jnp.array([0.0]), jnp.array([1.0]), jnp.array([-1.0])]

    # Test function on sample points
    for i, point in enumerate(test_points):
        try:
            result = function(point)

            if not isinstance(result, (int, float, jnp.ndarray)):
                raise ValidationError(f"Function must return a scalar, got {type(result)}")

            result = float(result)

            if not jnp.isfinite(result):
                warnings.warn(
                    f"Function returned non-finite value {result} at test point {i}",
                    ValidationWarning
                )

        except Exception as e:
            raise ValidationError(f"Function evaluation failed at test point {i}: {e}")

    def validated_function(x: Array) -> float:
        """Wrapped function with error handling."""
        try:
            result = function(x)
            return float(result)
        except Exception as e:
            warnings.warn(f"Function evaluation failed: {e}", ValidationWarning)
            return float("nan")

    return validated_function
