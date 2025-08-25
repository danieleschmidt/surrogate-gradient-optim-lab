"""Main SurrogateOptimizer class providing the primary interface."""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from jax import Array
import jax.numpy as jnp

from .data.collector import DataCollector, collect_data
from .models.base import Dataset, Surrogate
from .models.gaussian_process import GPSurrogate
from .models.neural import NeuralSurrogate
from .models.random_forest import RandomForestSurrogate
from .optimizers.base import OptimizationResult
from .optimizers.gradient_descent import GradientDescentOptimizer
from .optimizers.multi_start import MultiStartOptimizer
from .optimizers.trust_region import TrustRegionOptimizer


class SurrogateOptimizer:
    """Main interface for surrogate gradient optimization.

    Provides a high-level interface for the complete surrogate optimization
    workflow: data collection, surrogate training, and optimization.
    """

    def __init__(
        self,
        surrogate_type: str = "neural_network",
        surrogate_params: Optional[Dict[str, Any]] = None,
        optimizer_type: str = "gradient_descent",
        optimizer_params: Optional[Dict[str, Any]] = None,
    ):
        """Initialize SurrogateOptimizer.

        Args:
            surrogate_type: Type of surrogate model ('neural_network', 'gp', 'random_forest')
            surrogate_params: Parameters for surrogate model
            optimizer_type: Type of optimizer ('gradient_descent', 'trust_region', 'multi_start')
            optimizer_params: Parameters for optimizer
        """
        self.surrogate_type = surrogate_type
        self.surrogate_params = surrogate_params or {}
        self.optimizer_type = optimizer_type
        self.optimizer_params = optimizer_params or {}

        # Initialize surrogate model
        self.surrogate = self._create_surrogate()

        # Initialize optimizer
        self.optimizer = self._create_optimizer()

        # State
        self.is_fitted = False
        self.training_data = None

        # Generation 2: Robustness features
        from .health.system_monitor import system_monitor
        from .monitoring.enhanced_logging import enhanced_logger

        # Start system monitoring
        system_monitor.start_monitoring()
        enhanced_logger.info(
            f"SurrogateOptimizer initialized with {surrogate_type} surrogate and {optimizer_type} optimizer"
        )

    def _create_surrogate(self) -> Surrogate:
        """Create surrogate model based on configuration."""
        if self.surrogate_type in ["neural_network", "nn"]:
            return NeuralSurrogate(**self.surrogate_params)
        if self.surrogate_type in ["gaussian_process", "gp"]:
            return GPSurrogate(**self.surrogate_params)
        if self.surrogate_type in ["random_forest", "rf"]:
            return RandomForestSurrogate(**self.surrogate_params)
        raise ValueError(f"Unknown surrogate type: {self.surrogate_type}")

    def _create_optimizer(self):
        """Create optimizer based on configuration."""
        if self.optimizer_type == "gradient_descent":
            return GradientDescentOptimizer(**self.optimizer_params)
        if self.optimizer_type == "trust_region":
            return TrustRegionOptimizer(**self.optimizer_params)
        if self.optimizer_type == "multi_start":
            return MultiStartOptimizer(**self.optimizer_params)
        raise ValueError(f"Unknown optimizer type: {self.optimizer_type}")

    def _validate_training_data(self, data: Dataset) -> None:
        """Enhanced validation of training data."""
        from .validation.input_validation import ValidationError

        # Check data quality
        if jnp.any(jnp.isnan(data.X)) or jnp.any(jnp.isnan(data.y)):
            raise ValidationError("Training data contains NaN values")

        if jnp.any(jnp.isinf(data.X)) or jnp.any(jnp.isinf(data.y)):
            raise ValidationError("Training data contains infinite values")

        # Check data distribution
        if data.n_samples < 10:
            import warnings

            warnings.warn(
                "Training with very few samples may lead to poor performance",
                UserWarning,
            )

        # Check for constant function values
        if jnp.std(data.y) < 1e-12:
            import warnings

            warnings.warn(
                "Function values have very low variance - may be constant", UserWarning
            )

    def fit_surrogate(
        self, data: Union[Dataset, Dict[str, Array]]
    ) -> "SurrogateOptimizer":
        """Train the surrogate model on the given data.

        Args:
            data: Training data as Dataset or dict with 'X' and 'y' keys

        Returns:
            Self for method chaining
        """
        # Convert dict to Dataset if necessary
        if isinstance(data, dict):
            data = Dataset(
                X=data["X"],
                y=data["y"],
                gradients=data.get("gradients"),
                metadata=data.get("metadata", {}),
            )

        # Enhanced data validation
        try:
            if data.n_samples == 0:
                from .core.error_handling import DataValidationError

                raise DataValidationError("Cannot train on empty dataset")

            self._validate_training_data(data)
        except Exception as e:
            from .core.error_handling import DataValidationError

            if not isinstance(e, DataValidationError):
                raise DataValidationError(f"Data validation failed: {e!s}") from e
            raise

        from .monitoring.enhanced_logging import enhanced_logger

        enhanced_logger.info(
            f"Training {self.surrogate_type} surrogate on {data.n_samples} samples..."
        )

        # Train surrogate with error handling and circuit breaker
        try:
            from .robustness.circuit_breaker import training_circuit_breaker

            training_circuit_breaker.call(self.surrogate.fit, data)
            self.training_data = data
            self.is_fitted = True
            enhanced_logger.info("Surrogate training complete.")
            return self
        except Exception as e:
            from .core.error_handling import ModelTrainingError

            raise ModelTrainingError(f"Failed to train surrogate: {e!s}") from e

    def optimize(
        self,
        initial_point: Array,
        bounds: Optional[List[Tuple[float, float]]] = None,
        method: str = "L-BFGS-B",
        max_iterations: Optional[int] = None,
        num_steps: Optional[int] = None,
        **kwargs,
    ) -> OptimizationResult:
        """Optimize using the trained surrogate.

        Args:
            initial_point: Starting point for optimization
            bounds: Optional bounds for each dimension
            method: Optimization method (currently ignored, uses configured optimizer)
            num_steps: Maximum number of optimization steps
            **kwargs: Additional optimizer arguments

        Returns:
            Optimization result
        """
        if not self.is_fitted:
            raise ValueError("Surrogate must be trained before optimization")

        print(f"Starting optimization from point {initial_point}")

        # Update optimizer parameters if provided
        iterations = max_iterations or num_steps or 100
        if hasattr(self.optimizer, "max_iterations"):
            self.optimizer.max_iterations = iterations

        # Run optimization
        result = self.optimizer.optimize(
            surrogate=self.surrogate, x0=initial_point, bounds=bounds, **kwargs
        )

        print(
            f"Optimization complete. Found optimum at {result.x} with value {result.fun:.6f}"
        )

        return result

    def predict(self, x: Array) -> Array:
        """Predict function values using the trained surrogate.

        Args:
            x: Input points for prediction

        Returns:
            Predicted function values
        """
        if not self.is_fitted:
            raise ValueError("Surrogate must be trained before prediction")

        return self.surrogate.predict(x)

    def gradient(self, x: Array) -> Array:
        """Compute gradients using the trained surrogate.

        Args:
            x: Input points for gradient computation

        Returns:
            Gradient vectors
        """
        if not self.is_fitted:
            raise ValueError("Surrogate must be trained before gradient computation")

        return self.surrogate.gradient(x)

    def uncertainty(self, x: Array) -> Array:
        """Estimate prediction uncertainty.

        Args:
            x: Input points for uncertainty estimation

        Returns:
            Uncertainty estimates
        """
        if not self.is_fitted:
            raise ValueError("Surrogate must be trained before uncertainty computation")

        return self.surrogate.uncertainty(x)

    def validate(
        self,
        test_function: Callable[[Array], float],
        test_points: Optional[Array] = None,
        n_test_points: int = 100,
        metrics: List[str] = ["mse", "gradient_error"],
    ) -> Dict[str, float]:
        """Validate surrogate against true function.

        Args:
            test_function: True function for validation
            test_points: Optional test points (random if None)
            n_test_points: Number of test points if test_points is None
            metrics: Validation metrics to compute

        Returns:
            Dictionary of validation metrics
        """
        if not self.is_fitted:
            raise ValueError("Surrogate must be trained before validation")

        # Generate test points if not provided
        if test_points is None:
            if self.training_data is None:
                raise ValueError("Need training data bounds to generate test points")

            # Use training data bounds to generate test points
            bounds = self.training_data.metadata.get("bounds")
            if bounds is None:
                # Generate test points in [-2, 2] range
                bounds = [(-2, 2)] * self.training_data.n_dims

            collector = DataCollector(test_function, bounds)
            test_points = collector._generate_samples(n_test_points, "random")

        # Evaluate true function
        true_values = jnp.array([test_function(x) for x in test_points])

        # Get surrogate predictions
        pred_values = self.predict(test_points)

        # Compute metrics
        results = {}

        if "mse" in metrics:
            mse = float(jnp.mean((pred_values - true_values) ** 2))
            results["mse"] = mse

        if "mae" in metrics:
            mae = float(jnp.mean(jnp.abs(pred_values - true_values)))
            results["mae"] = mae

        if "r2" in metrics:
            ss_res = jnp.sum((true_values - pred_values) ** 2)
            ss_tot = jnp.sum((true_values - jnp.mean(true_values)) ** 2)
            r2 = float(1 - (ss_res / ss_tot))
            results["r2"] = r2

        if "gradient_error" in metrics:
            # Compare gradients if possible
            try:
                true_grads = []
                pred_grads = []

                # Use finite differences for true gradients
                eps = 1e-6
                for x in test_points[
                    : min(50, len(test_points))
                ]:  # Limit for efficiency
                    # Finite difference gradient
                    true_grad = jnp.zeros(len(x))
                    for i in range(len(x)):
                        x_plus = x.at[i].add(eps)
                        x_minus = x.at[i].add(-eps)
                        true_grad = true_grad.at[i].set(
                            (test_function(x_plus) - test_function(x_minus)) / (2 * eps)
                        )

                    true_grads.append(true_grad)
                    pred_grads.append(self.gradient(x))

                true_grads = jnp.stack(true_grads)
                pred_grads = jnp.stack(pred_grads)

                grad_error = float(
                    jnp.mean(jnp.linalg.norm(pred_grads - true_grads, axis=1))
                )
                results["gradient_error"] = grad_error

            except Exception as e:
                print(f"Warning: Could not compute gradient error: {e}")

        return results

    def get_training_info(self) -> Dict[str, Any]:
        """Get information about the training process.

        Returns:
            Dictionary with training information
        """
        if not self.is_fitted:
            return {"is_fitted": False}

        info = {
            "is_fitted": True,
            "surrogate_type": self.surrogate_type,
            "optimizer_type": self.optimizer_type,
            "n_training_samples": (
                self.training_data.n_samples if self.training_data else 0
            ),
            "input_dimension": self.training_data.n_dims if self.training_data else 0,
            "has_gradients": (
                self.training_data.gradients is not None
                if self.training_data
                else False
            ),
        }

        return info

    def optimize_parallel(
        self,
        initial_points: List[Array],
        bounds: Optional[List[Tuple[float, float]]] = None,
        **kwargs,
    ) -> List[OptimizationResult]:
        """Generation 3: Parallel optimization from multiple starting points.

        Args:
            initial_points: List of starting points for optimization
            bounds: Optimization bounds
            **kwargs: Additional optimizer arguments

        Returns:
            List of optimization results
        """
        if not self.is_fitted:
            raise ValueError("Surrogate must be trained before optimization")

        from .monitoring.enhanced_logging import enhanced_logger
        from .scalability.auto_scaling import adaptive_optimizer

        enhanced_logger.info(
            f"Starting parallel optimization with {len(initial_points)} points"
        )

        def single_optimize(x0: Array) -> OptimizationResult:
            """Optimize from single starting point."""
            return self.optimize(x0, bounds=bounds, **kwargs)

        # Use adaptive parallel execution
        results = adaptive_optimizer.auto_scaler.map_parallel(
            single_optimize, initial_points
        )

        enhanced_logger.info("Parallel optimization complete")
        return results

    def predict_batch(self, X: Array, batch_size: Optional[int] = None) -> Array:
        """Generation 3: Batch prediction with adaptive sizing.

        Args:
            X: Input points for prediction
            batch_size: Batch size (auto-determined if None)

        Returns:
            Predicted values
        """
        if not self.is_fitted:
            raise ValueError("Surrogate must be trained before prediction")

        from .health.system_monitor import system_monitor

        # Adaptive batch sizing
        if batch_size is None:
            health_report = system_monitor.get_health_report()
            current_load = health_report.get("current", {}).get("cpu_usage", 0.0)

            base_size = 100
            if current_load > 0.8:
                batch_size = max(10, base_size // 4)
            elif current_load < 0.3:
                batch_size = base_size * 2
            else:
                batch_size = base_size

        # Process in batches
        results = []
        for i in range(0, len(X), batch_size):
            batch = X[i : i + batch_size]
            batch_results = self.surrogate.predict(batch)
            results.extend(batch_results)

        return jnp.array(results)


# Convenience functions for quick usage
def create_surrogate_optimizer(
    surrogate_type: str = "neural_network", **kwargs
) -> SurrogateOptimizer:
    """Create a SurrogateOptimizer with default settings.

    Args:
        surrogate_type: Type of surrogate model
        **kwargs: Additional parameters

    Returns:
        Configured SurrogateOptimizer
    """
    return SurrogateOptimizer(surrogate_type=surrogate_type, **kwargs)


def quick_optimize(
    function: Callable[[Array], float],
    bounds: List[Tuple[float, float]],
    n_samples: int = 100,
    initial_point: Optional[Array] = None,
    surrogate_type: str = "neural_network",
    verbose: bool = True,
) -> OptimizationResult:
    """Quick optimization workflow with default settings.

    Args:
        function: Black-box function to optimize
        bounds: Bounds for each input dimension
        n_samples: Number of training samples
        initial_point: Starting point (random if None)
        surrogate_type: Type of surrogate model
        verbose: Whether to print progress

    Returns:
        Optimization result
    """
    # Collect training data
    if verbose:
        print("Collecting training data...")

    data = collect_data(
        function=function,
        n_samples=n_samples,
        bounds=bounds,
        sampling="sobol",
        verbose=verbose,
    )

    # Create and train optimizer
    if verbose:
        print("Creating surrogate optimizer...")

    optimizer = SurrogateOptimizer(surrogate_type=surrogate_type)
    optimizer.fit_surrogate(data)

    # Generate initial point if not provided
    if initial_point is None:
        # Use center of bounds as initial point
        initial_point = jnp.array([(lower + upper) / 2 for lower, upper in bounds])

    # Optimize
    if verbose:
        print("Starting optimization...")

    result = optimizer.optimize(initial_point=initial_point, bounds=bounds)

    return result
