"""Robustness and reliability enhancements for surrogate optimization."""

import logging
import time
import warnings
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from .models.base import Surrogate, Dataset


@dataclass
class ValidationResult:
    """Result from model validation."""
    is_valid: bool
    warnings: List[str]
    errors: List[str]
    metrics: Dict[str, float]
    recommendations: List[str]


class RobustSurrogate:
    """Wrapper that adds robustness and validation to any surrogate model."""
    
    def __init__(
        self,
        base_surrogate: Surrogate,
        validation_split: float = 0.2,
        min_training_samples: int = 10,
        max_condition_number: float = 1e12,
        gradient_check_tolerance: float = 1e-3,
        enable_warnings: bool = True,
    ):
        """Initialize robust surrogate wrapper.
        
        Args:
            base_surrogate: Base surrogate model to wrap
            validation_split: Fraction of data for validation
            min_training_samples: Minimum samples required for training
            max_condition_number: Maximum acceptable condition number
            gradient_check_tolerance: Tolerance for gradient validation
            enable_warnings: Whether to show warnings
        """
        self.base_surrogate = base_surrogate
        self.validation_split = validation_split
        self.min_training_samples = min_training_samples
        self.max_condition_number = max_condition_number
        self.gradient_check_tolerance = gradient_check_tolerance
        self.enable_warnings = enable_warnings
        
        # Validation state
        self.is_fitted = False
        self.validation_result = None
        self.training_metrics = {}
        
        # Setup logging
        self.logger = logging.getLogger(f"RobustSurrogate.{base_surrogate.__class__.__name__}")
    
    def _validate_dataset(self, dataset: Dataset) -> ValidationResult:
        """Validate training dataset quality."""
        warnings_list = []
        errors = []
        metrics = {}
        recommendations = []
        
        # Check minimum samples
        if dataset.n_samples < self.min_training_samples:
            errors.append(f"Insufficient training samples: {dataset.n_samples} < {self.min_training_samples}")
        
        # Check for NaN or infinite values
        if not jnp.isfinite(dataset.X).all():
            errors.append("Training inputs contain NaN or infinite values")
        
        if not jnp.isfinite(dataset.y).all():
            errors.append("Training outputs contain NaN or infinite values")
        
        # Check input diversity (condition number)
        try:
            cov_matrix = jnp.cov(dataset.X.T)
            condition_number = float(jnp.linalg.cond(cov_matrix))
            metrics["condition_number"] = condition_number
            
            if condition_number > self.max_condition_number:
                warnings_list.append(f"High condition number: {condition_number:.2e} (may cause numerical instability)")
                recommendations.append("Consider adding regularization or collecting more diverse data")
        except Exception as e:
            warnings_list.append(f"Could not compute condition number: {e}")
        
        # Check output range and distribution
        y_range = float(jnp.max(dataset.y) - jnp.min(dataset.y))
        y_std = float(jnp.std(dataset.y))
        metrics["output_range"] = y_range
        metrics["output_std"] = y_std
        
        if y_range == 0:
            warnings_list.append("All training outputs are identical (no variation to learn)")
        elif y_std < 1e-10:
            warnings_list.append("Very small output variation - may be difficult to learn")
        
        # Check for duplicate points
        unique_points = len(jnp.unique(dataset.X, axis=0))
        duplicate_fraction = 1 - (unique_points / dataset.n_samples)
        metrics["duplicate_fraction"] = duplicate_fraction
        
        if duplicate_fraction > 0.1:
            warnings_list.append(f"High fraction of duplicate input points: {duplicate_fraction:.1%}")
            recommendations.append("Remove duplicate points or add input noise")
        
        # Check input bounds and scaling
        input_ranges = jnp.max(dataset.X, axis=0) - jnp.min(dataset.X, axis=0)
        max_range = float(jnp.max(input_ranges))
        min_range = float(jnp.min(input_ranges[input_ranges > 0]))
        
        if max_range / min_range > 100:
            warnings_list.append("Large differences in input scales detected")
            recommendations.append("Consider normalizing input features")
        
        # Gradient validation if available
        if dataset.gradients is not None:
            if not jnp.isfinite(dataset.gradients).all():
                errors.append("Training gradients contain NaN or infinite values")
            else:
                grad_norms = jnp.linalg.norm(dataset.gradients, axis=1)
                metrics["max_gradient_norm"] = float(jnp.max(grad_norms))
                metrics["mean_gradient_norm"] = float(jnp.mean(grad_norms))
                
                if jnp.any(grad_norms > 1e6):
                    warnings_list.append("Very large gradient norms detected")
                    recommendations.append("Consider gradient clipping or data preprocessing")
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            warnings=warnings_list,
            errors=errors,
            metrics=metrics,
            recommendations=recommendations
        )
    
    def _validate_predictions(self, x: Array, predictions: Array) -> bool:
        """Validate model predictions."""
        if not jnp.isfinite(predictions).all():
            self.logger.error("Model produced NaN or infinite predictions")
            return False
        
        # Check for extreme values
        pred_range = jnp.max(predictions) - jnp.min(predictions)
        if pred_range > 1e10:
            self.logger.warning(f"Very large prediction range: {pred_range:.2e}")
        
        return True
    
    def _validate_gradients(self, x: Array, gradients: Array) -> bool:
        """Validate model gradients."""
        if not jnp.isfinite(gradients).all():
            self.logger.error("Model produced NaN or infinite gradients")
            return False
        
        # Check gradient magnitudes
        grad_norms = jnp.linalg.norm(gradients, axis=-1)
        max_grad_norm = jnp.max(grad_norms)
        
        if max_grad_norm > 1e8:
            self.logger.warning(f"Very large gradient norm: {max_grad_norm:.2e}")
        
        return True
    
    def fit(self, dataset: Dataset) -> "RobustSurrogate":
        """Fit surrogate with robustness validation."""
        # Validate dataset
        self.validation_result = self._validate_dataset(dataset)
        
        # Log warnings and errors
        for warning in self.validation_result.warnings:
            if self.enable_warnings:
                warnings.warn(f"Dataset validation warning: {warning}")
                self.logger.warning(warning)
        
        for error in self.validation_result.errors:
            self.logger.error(f"Dataset validation error: {error}")
        
        # Log recommendations
        for rec in self.validation_result.recommendations:
            self.logger.info(f"Recommendation: {rec}")
        
        if not self.validation_result.is_valid:
            raise ValueError(f"Dataset validation failed: {self.validation_result.errors}")
        
        # Train base model with error handling
        try:
            start_time = time.time()
            self.base_surrogate.fit(dataset)
            training_time = time.time() - start_time
            
            self.training_metrics = {
                "training_time": training_time,
                "n_samples": dataset.n_samples,
                "n_dims": dataset.n_dims,
                **self.validation_result.metrics
            }
            
            self.is_fitted = True
            self.logger.info(f"Model training completed in {training_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            raise
        
        return self
    
    def predict(self, x: Array) -> Array:
        """Robust prediction with validation."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Input validation
        if not jnp.isfinite(x).all():
            raise ValueError("Input contains NaN or infinite values")
        
        try:
            predictions = self.base_surrogate.predict(x)
            
            # Validate predictions
            if not self._validate_predictions(x, predictions):
                # Fallback to conservative prediction if validation fails
                self.logger.warning("Prediction validation failed, using fallback")
                predictions = jnp.zeros_like(predictions)
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            # Return zero predictions as safe fallback
            if x.ndim == 1:
                return jnp.array(0.0)
            else:
                return jnp.zeros(x.shape[0])
    
    def gradient(self, x: Array) -> Array:
        """Robust gradient computation with validation."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before gradient computation")
        
        # Input validation
        if not jnp.isfinite(x).all():
            raise ValueError("Input contains NaN or infinite values")
        
        try:
            gradients = self.base_surrogate.gradient(x)
            
            # Validate gradients
            if not self._validate_gradients(x, gradients):
                # Fallback to zero gradients if validation fails
                self.logger.warning("Gradient validation failed, using fallback")
                gradients = jnp.zeros_like(gradients)
            
            return gradients
            
        except Exception as e:
            self.logger.error(f"Gradient computation failed: {e}")
            # Return zero gradients as safe fallback
            if x.ndim == 1:
                return jnp.zeros_like(x)
            else:
                return jnp.zeros_like(x)
    
    def uncertainty(self, x: Array) -> Array:
        """Robust uncertainty estimation."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before uncertainty computation")
        
        try:
            return self.base_surrogate.uncertainty(x)
        except Exception as e:
            self.logger.warning(f"Uncertainty computation failed: {e}")
            # Return high uncertainty as safe fallback
            if x.ndim == 1:
                return jnp.array(1.0)
            else:
                return jnp.ones(x.shape[0])
    
    def get_validation_report(self) -> Dict[str, Any]:
        """Get detailed validation and training report."""
        if self.validation_result is None:
            return {"status": "not_validated"}
        
        return {
            "validation": {
                "is_valid": self.validation_result.is_valid,
                "warnings": self.validation_result.warnings,
                "errors": self.validation_result.errors,
                "recommendations": self.validation_result.recommendations,
            },
            "metrics": self.validation_result.metrics,
            "training": self.training_metrics,
            "model_type": self.base_surrogate.__class__.__name__,
        }


def robust_function_wrapper(
    func: Callable[[Array], float],
    max_retries: int = 3,
    timeout: float = 60.0,
    fallback_value: Optional[float] = None,
) -> Callable[[Array], float]:
    """Create a robust wrapper around a black-box function.
    
    Args:
        func: Function to wrap
        max_retries: Maximum number of retries on failure
        timeout: Timeout in seconds for each evaluation
        fallback_value: Value to return on failure (None to raise exception)
        
    Returns:
        Robust wrapped function
    """
    
    @wraps(func)
    def robust_func(x: Array) -> float:
        # Input validation
        if not jnp.isfinite(x).all():
            if fallback_value is not None:
                return fallback_value
            raise ValueError("Input contains NaN or infinite values")
        
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                # Set timeout (simplified - would need proper implementation)
                start_time = time.time()
                result = func(x)
                elapsed = time.time() - start_time
                
                if elapsed > timeout:
                    raise TimeoutError(f"Function evaluation timeout ({elapsed:.2f}s > {timeout}s)")
                
                # Validate result
                if not jnp.isfinite(result):
                    raise ValueError("Function returned NaN or infinite value")
                
                return float(result)
                
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    time.sleep(0.1 * (attempt + 1))  # Exponential backoff
                    continue
        
        # All retries failed
        if fallback_value is not None:
            logging.warning(f"Function evaluation failed after {max_retries} attempts: {last_exception}")
            return fallback_value
        else:
            raise last_exception
    
    return robust_func


class RobustOptimizer:
    """Robust optimization with automatic fallbacks and error recovery."""
    
    def __init__(self, base_optimizer, max_restarts: int = 3):
        """Initialize robust optimizer.
        
        Args:
            base_optimizer: Base optimizer to wrap
            max_restarts: Maximum number of restart attempts
        """
        self.base_optimizer = base_optimizer
        self.max_restarts = max_restarts
        self.logger = logging.getLogger("RobustOptimizer")
    
    def optimize(self, surrogate: Surrogate, x0: Array, bounds=None, **kwargs):
        """Robust optimization with automatic restarts."""
        best_result = None
        best_value = float('inf')
        
        for restart in range(self.max_restarts):
            try:
                # Add small perturbation for restarts
                if restart > 0:
                    perturbation = jnp.random.normal(0, 0.1, x0.shape)
                    start_point = x0 + perturbation
                    
                    # Ensure bounds are respected
                    if bounds is not None:
                        for i, (low, high) in enumerate(bounds):
                            start_point = start_point.at[i].set(
                                jnp.clip(start_point[i], low, high)
                            )
                else:
                    start_point = x0
                
                # Run optimization
                result = self.base_optimizer.optimize(
                    surrogate=surrogate,
                    x0=start_point,
                    bounds=bounds,
                    **kwargs
                )
                
                # Check if this is the best result so far
                if result.success and result.fun < best_value:
                    best_result = result
                    best_value = result.fun
                
                # If successful and good enough, return
                if result.success and result.fun < 1e-6:
                    return result
                    
            except Exception as e:
                self.logger.warning(f"Optimization attempt {restart + 1} failed: {e}")
                continue
        
        # Return best result found, or create failure result
        if best_result is not None:
            return best_result
        else:
            # Create failure result
            from .optimizers.base import OptimizationResult
            return OptimizationResult(
                x=x0,
                fun=float('inf'),
                success=False,
                message=f"All {self.max_restarts} optimization attempts failed",
                nit=0,
                nfev=0
            )


def add_robustness_to_optimizer(optimizer_class):
    """Decorator to add robustness to any optimizer class."""
    
    class RobustOptimizerWrapper(optimizer_class):
        def __init__(self, *args, **kwargs):
            # Extract robustness parameters
            self.max_restarts = kwargs.pop('max_restarts', 3)
            self.enable_validation = kwargs.pop('enable_validation', True)
            
            super().__init__(*args, **kwargs)
            self.logger = logging.getLogger(f"Robust{optimizer_class.__name__}")
        
        def optimize(self, surrogate, x0, bounds=None, **kwargs):
            # Wrap surrogate if validation enabled
            if self.enable_validation and not isinstance(surrogate, RobustSurrogate):
                surrogate = RobustSurrogate(surrogate)
            
            # Use robust optimization
            robust_opt = RobustOptimizer(self, self.max_restarts)
            return robust_opt.optimize(surrogate, x0, bounds, **kwargs)
    
    return RobustOptimizerWrapper


# Health check utilities
def health_check_surrogate(surrogate: Surrogate, test_points: Array) -> Dict[str, Any]:
    """Perform comprehensive health check on a surrogate model."""
    results = {
        "overall_health": "unknown",
        "issues": [],
        "warnings": [],
        "metrics": {}
    }
    
    try:
        # Test basic prediction
        predictions = surrogate.predict(test_points)
        results["metrics"]["prediction_success"] = True
        
        if not jnp.isfinite(predictions).all():
            results["issues"].append("Predictions contain NaN or infinite values")
        
        # Test gradient computation
        try:
            gradients = surrogate.gradient(test_points)
            results["metrics"]["gradient_success"] = True
            
            if not jnp.isfinite(gradients).all():
                results["issues"].append("Gradients contain NaN or infinite values")
                
            # Check gradient magnitudes
            grad_norms = jnp.linalg.norm(gradients, axis=-1)
            max_grad = float(jnp.max(grad_norms))
            results["metrics"]["max_gradient_norm"] = max_grad
            
            if max_grad > 1e6:
                results["warnings"].append(f"Very large gradient norms: {max_grad:.2e}")
                
        except Exception as e:
            results["issues"].append(f"Gradient computation failed: {e}")
            results["metrics"]["gradient_success"] = False
        
        # Test uncertainty estimation
        try:
            uncertainties = surrogate.uncertainty(test_points)
            results["metrics"]["uncertainty_success"] = True
            
            if not jnp.isfinite(uncertainties).all():
                results["warnings"].append("Uncertainties contain NaN or infinite values")
                
        except Exception as e:
            results["warnings"].append(f"Uncertainty estimation failed: {e}")
            results["metrics"]["uncertainty_success"] = False
        
        # Overall health assessment
        if len(results["issues"]) == 0:
            if len(results["warnings"]) == 0:
                results["overall_health"] = "excellent"
            elif len(results["warnings"]) <= 2:
                results["overall_health"] = "good"
            else:
                results["overall_health"] = "fair"
        else:
            results["overall_health"] = "poor"
            
    except Exception as e:
        results["overall_health"] = "critical"
        results["issues"].append(f"Critical failure: {e}")
    
    return results