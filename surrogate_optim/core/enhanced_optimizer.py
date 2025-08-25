"""Enhanced SurrogateOptimizer with robust error handling and monitoring."""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

from jax import Array
import jax.numpy as jnp

# Import the original components
from ..models.base import Dataset, Surrogate
from ..models.gaussian_process import GPSurrogate
from ..models.hybrid import HybridSurrogate
from ..models.neural import NeuralSurrogate
from ..models.random_forest import RandomForestSurrogate
from ..optimizers.base import OptimizationResult
from ..optimizers.gradient_descent import GradientDescentOptimizer
from ..optimizers.multi_start import MultiStartOptimizer
from ..optimizers.trust_region import TrustRegionOptimizer
from .error_handling import (
    ConfigurationError,
    DataValidationError,
    ModelTrainingError,
    NumericalStabilityError,
    OptimizationError,
    check_numerical_stability,
    error_boundary,
    robust_function_call,
    validate_array_input,
    validate_bounds,
    validate_dataset,
    validate_optimization_config,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class EnhancedSurrogateOptimizer:
    """Enhanced surrogate optimizer with robust error handling and monitoring.
    
    This class provides a production-ready interface for surrogate optimization
    with comprehensive error handling, validation, logging, and monitoring.
    """

    def __init__(
        self,
        surrogate_type: str = "gaussian_process",
        surrogate_params: Optional[Dict[str, Any]] = None,
        optimizer_type: str = "gradient_descent",
        optimizer_params: Optional[Dict[str, Any]] = None,
        enable_monitoring: bool = True,
        enable_validation: bool = True,
        max_retries: int = 3,
        random_seed: Optional[int] = None,
    ):
        """Initialize enhanced surrogate optimizer.
        
        Args:
            surrogate_type: Type of surrogate model
            surrogate_params: Parameters for surrogate model
            optimizer_type: Type of optimizer
            optimizer_params: Parameters for optimizer  
            enable_monitoring: Enable performance monitoring
            enable_validation: Enable input/output validation
            max_retries: Maximum retries for failed operations
            random_seed: Random seed for reproducibility
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Validate and store configuration
        try:
            self.surrogate_type, self.surrogate_params, \
            self.optimizer_type, self.optimizer_params = validate_optimization_config(
                surrogate_type,
                surrogate_params or {},
                optimizer_type,
                optimizer_params or {}
            )
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            raise ConfigurationError(f"Invalid configuration: {e}")

        # Configuration flags
        self.enable_monitoring = enable_monitoring
        self.enable_validation = enable_validation
        self.max_retries = max_retries
        self.random_seed = random_seed

        # State tracking
        self.is_fitted = False
        self.training_data = None
        self.surrogate = None
        self.optimizer = None
        self.metrics = {}

        # Performance tracking
        self.performance_history = {
            "training_times": [],
            "optimization_times": [],
            "prediction_times": [],
            "errors": [],
            "warnings": [],
        }

        self.logger.info(
            f"Initialized EnhancedSurrogateOptimizer with "
            f"surrogate={surrogate_type}, optimizer={optimizer_type}"
        )

    def _create_surrogate(self) -> Surrogate:
        """Create surrogate model with error handling."""
        try:
            if self.surrogate_type in ["neural_network", "nn"]:
                # Add random seed if provided
                if self.random_seed is not None:
                    self.surrogate_params["random_seed"] = self.random_seed
                return NeuralSurrogate(**self.surrogate_params)
            if self.surrogate_type in ["gaussian_process", "gp"]:
                return GPSurrogate(**self.surrogate_params)
            if self.surrogate_type in ["random_forest", "rf"]:
                # Add random seed if provided
                if self.random_seed is not None:
                    self.surrogate_params["random_state"] = self.random_seed
                return RandomForestSurrogate(**self.surrogate_params)
            if self.surrogate_type == "hybrid":
                return HybridSurrogate(**self.surrogate_params)
            raise ConfigurationError(f"Unknown surrogate type: {self.surrogate_type}")
        except Exception as e:
            self.logger.error(f"Failed to create surrogate model: {e}")
            raise ModelTrainingError(f"Surrogate creation failed: {e}")

    def _create_optimizer(self):
        """Create optimizer with error handling."""
        try:
            if self.optimizer_type == "gradient_descent":
                return GradientDescentOptimizer(**self.optimizer_params)
            if self.optimizer_type == "trust_region":
                return TrustRegionOptimizer(**self.optimizer_params)
            if self.optimizer_type == "multi_start":
                return MultiStartOptimizer(**self.optimizer_params)
            raise ConfigurationError(f"Unknown optimizer type: {self.optimizer_type}")
        except Exception as e:
            self.logger.error(f"Failed to create optimizer: {e}")
            raise OptimizationError(f"Optimizer creation failed: {e}")

    @error_boundary(default_return=None, reraise_on=(ConfigurationError, DataValidationError))
    def fit_surrogate(
        self,
        data: Union[Dataset, Dict[str, Array]],
        validate_data: bool = None,
    ) -> "EnhancedSurrogateOptimizer":
        """Train the surrogate model with enhanced error handling.
        
        Args:
            data: Training data as Dataset or dict
            validate_data: Whether to validate data (uses instance setting if None)
            
        Returns:
            Self for method chaining
            
        Raises:
            DataValidationError: If data validation fails
            ModelTrainingError: If model training fails
        """
        start_time = time.time()
        validate_data = validate_data if validate_data is not None else self.enable_validation

        try:
            self.logger.info("Starting surrogate model training...")

            # Convert dict to Dataset if necessary
            if isinstance(data, dict):
                if "X" not in data or "y" not in data:
                    raise DataValidationError("Data dict must contain 'X' and 'y' keys")

                data = Dataset(
                    X=data["X"],
                    y=data["y"],
                    gradients=data.get("gradients"),
                    metadata=data.get("metadata", {})
                )

            # Validate dataset
            if validate_data:
                data = validate_dataset(data)
                self.logger.info(f"Dataset validation passed: {data.n_samples} samples, {data.n_dims} dims")

            # Check for numerical stability
            if self.enable_validation:
                check_numerical_stability(data.X, "training inputs")
                check_numerical_stability(data.y, "training outputs")
                if data.gradients is not None:
                    check_numerical_stability(data.gradients, "training gradients")

            # Create surrogate model if not already created
            if self.surrogate is None:
                self.surrogate = self._create_surrogate()

            # Robust training with retries
            def train_surrogate():
                self.surrogate.fit(data)
                return self.surrogate

            surrogate = robust_function_call(
                train_surrogate,
                max_retries=self.max_retries,
                allowed_exceptions=(RuntimeError, ValueError, NumericalStabilityError)
            )

            # Store training data and update state
            self.training_data = data
            self.is_fitted = True

            # Record performance metrics
            training_time = time.time() - start_time
            if self.enable_monitoring:
                self.performance_history["training_times"].append(training_time)
                self.metrics["last_training_time"] = training_time
                self.metrics["training_samples"] = data.n_samples
                self.metrics["input_dimensions"] = data.n_dims

            self.logger.info(
                f"Surrogate training completed in {training_time:.2f}s "
                f"({data.n_samples} samples, {data.n_dims}D)"
            )

            return self

        except (DataValidationError, ConfigurationError):
            # Re-raise validation errors without modification
            raise
        except Exception as e:
            error_msg = f"Surrogate training failed: {e}"
            self.logger.error(error_msg)
            if self.enable_monitoring:
                self.performance_history["errors"].append(error_msg)
            raise ModelTrainingError(error_msg) from e

    @error_boundary(reraise_on=(OptimizationError, DataValidationError))
    def optimize(
        self,
        initial_point: Array,
        bounds: Optional[List[Tuple[float, float]]] = None,
        method: str = "auto",
        num_steps: int = 100,
        validate_inputs: bool = None,
        **kwargs
    ) -> OptimizationResult:
        """Run optimization with enhanced error handling.
        
        Args:
            initial_point: Starting point for optimization
            bounds: Optional bounds for each dimension
            method: Optimization method (ignored, uses configured optimizer)
            num_steps: Maximum number of optimization steps
            validate_inputs: Whether to validate inputs (uses instance setting if None)
            **kwargs: Additional optimizer arguments
            
        Returns:
            Optimization result
            
        Raises:
            OptimizationError: If optimization fails
            DataValidationError: If input validation fails
        """
        start_time = time.time()
        validate_inputs = validate_inputs if validate_inputs is not None else self.enable_validation

        try:
            if not self.is_fitted:
                raise OptimizationError("Surrogate must be trained before optimization")

            self.logger.info(f"Starting optimization from point {initial_point}")

            # Validate inputs
            if validate_inputs:
                initial_point = validate_array_input(
                    initial_point,
                    "initial_point",
                    min_dims=1,
                    max_dims=1,
                    finite_values=True
                )

                if bounds is not None:
                    bounds = validate_bounds(bounds, len(initial_point))
                    # Check if initial point is within bounds
                    for i, (lower, upper) in enumerate(bounds):
                        if not (lower <= initial_point[i] <= upper):
                            warnings.warn(
                                f"Initial point[{i}]={initial_point[i]} is outside "
                                f"bounds ({lower}, {upper})"
                            )

            # Create optimizer if not already created
            if self.optimizer is None:
                self.optimizer = self._create_optimizer()

            # Update optimizer parameters
            if hasattr(self.optimizer, "max_iterations"):
                self.optimizer.max_iterations = num_steps

            # Robust optimization with retries
            def run_optimization():
                return self.optimizer.optimize(
                    surrogate=self.surrogate,
                    x0=initial_point,
                    bounds=bounds,
                    **kwargs
                )

            result = robust_function_call(
                run_optimization,
                max_retries=self.max_retries,
                allowed_exceptions=(RuntimeError, ValueError, OptimizationError)
            )

            # Validate result
            if self.enable_validation:
                result.x = validate_array_input(
                    result.x,
                    "optimization result",
                    finite_values=True
                )

                if not jnp.isfinite(result.fun):
                    raise OptimizationError(f"Optimization returned non-finite value: {result.fun}")

            # Record performance metrics
            optimization_time = time.time() - start_time
            if self.enable_monitoring:
                self.performance_history["optimization_times"].append(optimization_time)
                self.metrics["last_optimization_time"] = optimization_time
                self.metrics["last_final_value"] = float(result.fun)

            self.logger.info(
                f"Optimization completed in {optimization_time:.2f}s. "
                f"Final value: {result.fun:.6f} at {result.x}"
            )

            return result

        except (DataValidationError, OptimizationError):
            # Re-raise specific errors
            raise
        except Exception as e:
            error_msg = f"Optimization failed: {e}"
            self.logger.error(error_msg)
            if self.enable_monitoring:
                self.performance_history["errors"].append(error_msg)
            raise OptimizationError(error_msg) from e

    @error_boundary(default_return=jnp.array([jnp.nan]))
    def predict(self, x: Array, validate_inputs: bool = None) -> Array:
        """Make predictions with enhanced error handling.
        
        Args:
            x: Input points for prediction
            validate_inputs: Whether to validate inputs
            
        Returns:
            Predicted function values
        """
        start_time = time.time()
        validate_inputs = validate_inputs if validate_inputs is not None else self.enable_validation

        try:
            if not self.is_fitted:
                raise ModelTrainingError("Surrogate must be trained before prediction")

            # Validate inputs
            if validate_inputs:
                x = validate_array_input(x, "prediction input", finite_values=True)

            # Make prediction
            predictions = self.surrogate.predict(x)

            # Validate outputs
            if self.enable_validation:
                check_numerical_stability(predictions, "predictions")

            # Record performance
            if self.enable_monitoring:
                pred_time = time.time() - start_time
                self.performance_history["prediction_times"].append(pred_time)

            return predictions

        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            if self.enable_monitoring:
                self.performance_history["errors"].append(f"Prediction error: {e}")
            return jnp.full(x.shape[0] if x.ndim > 1 else 1, jnp.nan)

    @error_boundary(default_return=jnp.array([jnp.nan]))
    def gradient(self, x: Array, validate_inputs: bool = None) -> Array:
        """Compute gradients with enhanced error handling.
        
        Args:
            x: Input points for gradient computation
            validate_inputs: Whether to validate inputs
            
        Returns:
            Gradient vectors
        """
        validate_inputs = validate_inputs if validate_inputs is not None else self.enable_validation

        try:
            if not self.is_fitted:
                raise ModelTrainingError("Surrogate must be trained before gradient computation")

            # Validate inputs
            if validate_inputs:
                x = validate_array_input(x, "gradient input", finite_values=True)

            # Compute gradients
            gradients = self.surrogate.gradient(x)

            # Validate outputs
            if self.enable_validation:
                check_numerical_stability(gradients, "gradients")

            return gradients

        except Exception as e:
            self.logger.error(f"Gradient computation failed: {e}")
            return jnp.full(x.shape, jnp.nan)

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        if not self.enable_monitoring:
            return {"monitoring_disabled": True}

        metrics = dict(self.metrics)

        # Add performance statistics
        if self.performance_history["training_times"]:
            metrics["training_time_stats"] = {
                "mean": float(jnp.mean(jnp.array(self.performance_history["training_times"]))),
                "std": float(jnp.std(jnp.array(self.performance_history["training_times"]))),
                "count": len(self.performance_history["training_times"])
            }

        if self.performance_history["optimization_times"]:
            metrics["optimization_time_stats"] = {
                "mean": float(jnp.mean(jnp.array(self.performance_history["optimization_times"]))),
                "std": float(jnp.std(jnp.array(self.performance_history["optimization_times"]))),
                "count": len(self.performance_history["optimization_times"])
            }

        if self.performance_history["prediction_times"]:
            pred_times = jnp.array(self.performance_history["prediction_times"])
            metrics["prediction_time_stats"] = {
                "mean": float(jnp.mean(pred_times)),
                "std": float(jnp.std(pred_times)),
                "count": len(pred_times)
            }

        # Add error and warning counts
        metrics["error_count"] = len(self.performance_history["errors"])
        metrics["warning_count"] = len(self.performance_history["warnings"])

        # Add configuration info
        metrics["configuration"] = {
            "surrogate_type": self.surrogate_type,
            "optimizer_type": self.optimizer_type,
            "enable_monitoring": self.enable_monitoring,
            "enable_validation": self.enable_validation,
            "max_retries": self.max_retries,
        }

        return metrics

    def reset_performance_history(self) -> None:
        """Reset performance tracking history."""
        self.performance_history = {
            "training_times": [],
            "optimization_times": [],
            "prediction_times": [],
            "errors": [],
            "warnings": [],
        }
        self.logger.info("Performance history reset")

    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check.
        
        Returns:
            Dictionary with health check results
        """
        health = {
            "status": "healthy",
            "issues": [],
            "warnings": [],
            "timestamp": time.time()
        }

        # Check if model is fitted
        if not self.is_fitted:
            health["warnings"].append("Model is not fitted")

        # Check for recent errors
        if self.enable_monitoring:
            recent_errors = self.performance_history["errors"][-5:]  # Last 5 errors
            if recent_errors:
                health["issues"].append(f"Recent errors: {len(recent_errors)}")
                health["status"] = "degraded"

        # Check training data quality
        if self.training_data is not None:
            if self.training_data.n_samples < 10:
                health["warnings"].append("Very few training samples")

            if jnp.var(self.training_data.y) < 1e-10:
                health["warnings"].append("Training data has very low variance")

        # Check surrogate model health
        if self.surrogate is not None:
            try:
                # Test prediction on a simple point
                if self.training_data is not None:
                    test_point = jnp.mean(self.training_data.X, axis=0)
                    pred = self.surrogate.predict(test_point)
                    if not jnp.isfinite(pred):
                        health["issues"].append("Surrogate produces non-finite predictions")
                        health["status"] = "unhealthy"
            except Exception as e:
                health["issues"].append(f"Surrogate prediction test failed: {e}")
                health["status"] = "unhealthy"

        # Set overall status based on issues
        if health["issues"]:
            health["status"] = "unhealthy" if any("unhealthy" in str(issue) for issue in health["issues"]) else "degraded"

        return health
