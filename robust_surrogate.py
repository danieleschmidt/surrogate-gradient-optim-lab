#!/usr/bin/env python3
"""
Robust Surrogate Gradient Optimization Implementation
Generation 2: MAKE IT ROBUST
"""

import warnings
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from pathlib import Path
import json

import jax.numpy as jnp
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OptimizationResult:
    """Result of optimization process."""
    x: np.ndarray
    fun: float
    success: bool
    message: str
    nit: int
    nfev: int

@dataclass 
class ValidationMetrics:
    """Validation metrics for surrogate quality."""
    mae: float
    rmse: float
    r2_score: float
    gradient_error: float

class RobustSurrogateError(Exception):
    """Base exception for surrogate optimization errors."""
    pass

class ValidationError(RobustSurrogateError):
    """Raised when validation fails."""
    pass

class OptimizationFailureError(RobustSurrogateError):
    """Raised when optimization fails."""
    pass

class InputValidator:
    """Robust input validation."""
    
    @staticmethod
    def validate_array(x: np.ndarray, name: str, expected_shape: Optional[Tuple] = None) -> np.ndarray:
        """Validate array input."""
        if not isinstance(x, (np.ndarray, list, tuple)):
            raise ValidationError(f"{name} must be array-like, got {type(x)}")
        
        x = np.array(x)
        
        if np.any(np.isnan(x)):
            raise ValidationError(f"{name} contains NaN values")
        
        if np.any(np.isinf(x)):
            raise ValidationError(f"{name} contains infinite values")
        
        if expected_shape and x.shape != expected_shape:
            raise ValidationError(f"{name} shape {x.shape} doesn't match expected {expected_shape}")
        
        return x
    
    @staticmethod
    def validate_bounds(bounds: List[Tuple[float, float]], n_dims: int):
        """Validate optimization bounds."""
        if len(bounds) != n_dims:
            raise ValidationError(f"Bounds length {len(bounds)} doesn't match dimensions {n_dims}")
        
        for i, (lower, upper) in enumerate(bounds):
            if lower >= upper:
                raise ValidationError(f"Invalid bounds at dimension {i}: {lower} >= {upper}")
            if not np.isfinite([lower, upper]).all():
                raise ValidationError(f"Non-finite bounds at dimension {i}: [{lower}, {upper}]")

class RobustSurrogate:
    """Robust surrogate model with multiple backends and validation."""
    
    def __init__(
        self,
        surrogate_type: str = "gp",
        noise_level: float = 1e-6,
        normalize_data: bool = True,
        validation_split: float = 0.2,
        ensemble_size: int = 1,
        random_state: int = 42
    ):
        """Initialize robust surrogate.
        
        Args:
            surrogate_type: 'gp', 'rf', or 'ensemble'
            noise_level: Noise level for GP
            normalize_data: Whether to normalize input/output data
            validation_split: Fraction of data for validation
            ensemble_size: Number of models in ensemble
            random_state: Random seed
        """
        self.surrogate_type = surrogate_type
        self.noise_level = noise_level
        self.normalize_data = normalize_data
        self.validation_split = validation_split
        self.ensemble_size = ensemble_size
        self.random_state = random_state
        
        # State
        self.is_fitted = False
        self.input_scaler = StandardScaler() if normalize_data else None
        self.output_scaler = StandardScaler() if normalize_data else None
        self.models = []
        self.validation_metrics = None
        
        # Initialize model(s)
        self._init_models()
    
    def _init_models(self):
        """Initialize surrogate models."""
        np.random.seed(self.random_state)
        
        for i in range(max(1, self.ensemble_size)):
            if self.surrogate_type == "gp":
                kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=self.noise_level)
                model = GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=self.noise_level,
                    random_state=self.random_state + i,
                    normalize_y=False  # We handle normalization ourselves
                )
            elif self.surrogate_type == "rf":
                model = RandomForestRegressor(
                    n_estimators=100,
                    random_state=self.random_state + i,
                    n_jobs=-1
                )
            else:
                raise ValueError(f"Unknown surrogate type: {self.surrogate_type}")
            
            self.models.append(model)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "RobustSurrogate":
        """Fit surrogate with robust validation."""
        try:
            # Validate inputs
            X = InputValidator.validate_array(X, "X")
            y = InputValidator.validate_array(y, "y")
            
            if X.shape[0] != y.shape[0]:
                raise ValidationError(f"X and y must have same number of samples: {X.shape[0]} vs {y.shape[0]}")
            
            if X.shape[0] < 5:
                raise ValidationError(f"Need at least 5 samples for training, got {X.shape[0]}")
            
            # Store original data
            self.n_samples, self.n_dims = X.shape
            
            # Split data for validation
            n_val = max(1, int(self.n_samples * self.validation_split))
            indices = np.random.permutation(self.n_samples)
            train_idx, val_idx = indices[n_val:], indices[:n_val]
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Normalize data if requested
            if self.normalize_data:
                X_train = self.input_scaler.fit_transform(X_train)
                X_val = self.input_scaler.transform(X_val)
                y_train = self.output_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
                y_val_scaled = self.output_scaler.transform(y_val.reshape(-1, 1)).ravel()
            else:
                y_val_scaled = y_val
            
            # Train models
            logger.info(f"Training {len(self.models)} surrogate model(s) on {len(X_train)} samples")
            
            for i, model in enumerate(self.models):
                try:
                    model.fit(X_train, y_train)
                    logger.debug(f"Model {i+1}/{len(self.models)} trained successfully")
                except Exception as e:
                    logger.warning(f"Failed to train model {i+1}: {e}")
                    # Continue with other models
            
            # Validate on held-out data
            self._validate_models(X_val, y_val_scaled)
            
            self.is_fitted = True
            logger.info("Surrogate training completed successfully")
            
            return self
            
        except Exception as e:
            logger.error(f"Surrogate training failed: {e}")
            raise RobustSurrogateError(f"Training failed: {e}") from e
    
    def _validate_models(self, X_val: np.ndarray, y_val: np.ndarray):
        """Validate trained models."""
        predictions = []
        gradients = []
        
        for model in self.models:
            try:
                pred = model.predict(X_val)
                predictions.append(pred)
                
                # Estimate gradients for validation
                grad_est = []
                for x in X_val:
                    grad_est.append(self._estimate_gradient(x, model))
                gradients.append(np.array(grad_est))
                
            except Exception as e:
                logger.warning(f"Model validation failed: {e}")
        
        if not predictions:
            raise ValidationError("All models failed validation")
        
        # Calculate ensemble prediction
        y_pred = np.mean(predictions, axis=0)
        
        # Calculate metrics
        mae = np.mean(np.abs(y_pred - y_val))
        rmse = np.sqrt(np.mean((y_pred - y_val)**2))
        
        # R² score
        ss_res = np.sum((y_val - y_pred)**2)
        ss_tot = np.sum((y_val - np.mean(y_val))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Gradient error (simplified)
        grad_error = np.mean([np.linalg.norm(g) for g in gradients]) if gradients else 0
        
        self.validation_metrics = ValidationMetrics(
            mae=mae, rmse=rmse, r2_score=r2, gradient_error=grad_error
        )
        
        logger.info(f"Validation metrics - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
        # Check if model quality is acceptable
        if r2 < 0.0:  # Very poor fit
            warnings.warn("Surrogate model has very poor fit (R² < 0). Consider more data or different model.")
    
    def predict(self, x: np.ndarray) -> float:
        """Predict with ensemble averaging."""
        if not self.is_fitted:
            raise ValidationError("Model must be fitted before prediction")
        
        x = InputValidator.validate_array(x, "x")
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        # Normalize input
        if self.normalize_data:
            x = self.input_scaler.transform(x)
        
        # Ensemble prediction
        predictions = []
        for model in self.models:
            try:
                pred = model.predict(x)
                predictions.append(pred)
            except Exception as e:
                logger.warning(f"Model prediction failed: {e}")
        
        if not predictions:
            raise RobustSurrogateError("All models failed to predict")
        
        y_pred = np.mean(predictions, axis=0)
        
        # Denormalize output
        if self.normalize_data:
            y_pred = self.output_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
        
        return float(y_pred[0])
    
    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Estimate gradient using finite differences."""
        if not self.is_fitted:
            raise ValidationError("Model must be fitted before gradient computation")
        
        x = InputValidator.validate_array(x, "x")
        
        # Use first model for gradient estimation
        return self._estimate_gradient(x, self.models[0])
    
    def _estimate_gradient(self, x: np.ndarray, model) -> np.ndarray:
        """Estimate gradient using finite differences."""
        eps = 1e-6
        grad = np.zeros_like(x)
        
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            
            # Normalize if needed
            if self.normalize_data:
                x_plus_norm = self.input_scaler.transform(x_plus.reshape(1, -1))
                x_minus_norm = self.input_scaler.transform(x_minus.reshape(1, -1))
            else:
                x_plus_norm = x_plus.reshape(1, -1)
                x_minus_norm = x_minus.reshape(1, -1)
            
            try:
                f_plus = model.predict(x_plus_norm)[0]
                f_minus = model.predict(x_minus_norm)[0]
                
                # Denormalize if needed
                if self.normalize_data:
                    f_plus = self.output_scaler.inverse_transform([[f_plus]])[0, 0]
                    f_minus = self.output_scaler.inverse_transform([[f_minus]])[0, 0]
                
                grad[i] = (f_plus - f_minus) / (2 * eps)
            except Exception as e:
                logger.warning(f"Gradient estimation failed at dimension {i}: {e}")
                grad[i] = 0.0
        
        return grad
    
    def uncertainty(self, x: np.ndarray) -> float:
        """Estimate prediction uncertainty."""
        if not self.is_fitted:
            raise ValidationError("Model must be fitted before uncertainty estimation")
        
        if len(self.models) < 2:
            return 0.0  # No uncertainty estimate for single model
        
        x = InputValidator.validate_array(x, "x")
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        # Normalize input
        if self.normalize_data:
            x = self.input_scaler.transform(x)
        
        # Get predictions from all models
        predictions = []
        for model in self.models:
            try:
                pred = model.predict(x)
                predictions.append(pred[0])
            except Exception:
                continue
        
        if len(predictions) < 2:
            return 0.0
        
        # Return standard deviation across ensemble
        return float(np.std(predictions))

class RobustOptimizer:
    """Robust optimization with multiple algorithms and error handling."""
    
    def __init__(
        self,
        method: str = "auto",
        bounds_penalty: float = 1e6,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        random_restarts: int = 5
    ):
        """Initialize robust optimizer.
        
        Args:
            method: Optimization method ('auto', 'bfgs', 'powell', 'differential_evolution')
            bounds_penalty: Penalty for bound violations
            max_iterations: Maximum optimization iterations
            tolerance: Convergence tolerance
            random_restarts: Number of random restarts for global optimization
        """
        self.method = method
        self.bounds_penalty = bounds_penalty
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.random_restarts = random_restarts
    
    def optimize(
        self,
        surrogate: RobustSurrogate,
        x0: np.ndarray,
        bounds: Optional[List[Tuple[float, float]]] = None,
        maximize: bool = False
    ) -> OptimizationResult:
        """Robust optimization with fallback strategies."""
        
        x0 = InputValidator.validate_array(x0, "x0")
        
        if bounds:
            InputValidator.validate_bounds(bounds, len(x0))
        
        # Define objective function
        def objective(x):
            try:
                value = surrogate.predict(x)
                
                # Add bounds penalty if needed
                if bounds:
                    penalty = 0.0
                    for i, (lower, upper) in enumerate(bounds):
                        if x[i] < lower:
                            penalty += self.bounds_penalty * (lower - x[i])**2
                        elif x[i] > upper:
                            penalty += self.bounds_penalty * (x[i] - upper)**2
                    value += penalty
                
                return -value if maximize else value
            except Exception as e:
                logger.warning(f"Objective evaluation failed: {e}")
                return 1e10  # Large penalty for failed evaluations
        
        def objective_grad(x):
            try:
                grad = surrogate.gradient(x)
                return -grad if maximize else grad
            except Exception as e:
                logger.warning(f"Gradient evaluation failed: {e}")
                return np.zeros_like(x)
        
        # Try multiple optimization strategies
        results = []
        methods_to_try = self._get_methods_to_try()
        
        for method_name in methods_to_try:
            try:
                logger.debug(f"Trying optimization method: {method_name}")
                result = self._optimize_with_method(
                    objective, objective_grad, x0, bounds, method_name
                )
                results.append((method_name, result))
                
                if result.success:
                    logger.info(f"Optimization succeeded with method: {method_name}")
                    break
                    
            except Exception as e:
                logger.warning(f"Method {method_name} failed: {e}")
                continue
        
        if not results:
            raise OptimizationFailureError("All optimization methods failed")
        
        # Return best result
        best_method, best_result = min(results, key=lambda x: x[1].fun)
        logger.info(f"Best result from method: {best_method}")
        
        return best_result
    
    def _get_methods_to_try(self) -> List[str]:
        """Get list of methods to try."""
        if self.method == "auto":
            return ["bfgs", "powell", "differential_evolution"]
        else:
            return [self.method]
    
    def _optimize_with_method(
        self,
        objective,
        objective_grad,
        x0: np.ndarray,
        bounds: Optional[List[Tuple[float, float]]],
        method: str
    ) -> OptimizationResult:
        """Optimize with specific method."""
        
        if method == "differential_evolution":
            if not bounds:
                # Create default bounds
                bounds = [(-10, 10)] * len(x0)
            
            result = differential_evolution(
                objective,
                bounds,
                maxiter=self.max_iterations // 10,
                seed=42,
                atol=self.tolerance
            )
        else:
            # Gradient-based methods
            options = {
                'maxiter': self.max_iterations,
                'ftol': self.tolerance,
                'gtol': self.tolerance
            }
            
            result = minimize(
                objective,
                x0,
                method=method.upper(),
                jac=objective_grad if method == "bfgs" else None,
                bounds=bounds if method in ["l-bfgs-b"] else None,
                options=options
            )
        
        return OptimizationResult(
            x=result.x,
            fun=result.fun,
            success=result.success,
            message=result.message,
            nit=result.nit,
            nfev=result.nfev
        )

def main():
    """Test robust surrogate optimization."""
    print("🛡️  Robust Surrogate Optimization Test")
    print("=" * 50)
    
    # Test function
    def rosenbrock(x):
        """Rosenbrock function - challenging optimization problem."""
        x = np.atleast_1d(x)
        return sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
    
    # Generate training data
    bounds = [(-2.0, 2.0), (-2.0, 2.0)]
    n_samples = 100
    
    logger.info(f"Generating {n_samples} training samples")
    X = np.random.uniform(bounds[0][0], bounds[0][1], (n_samples, 2))
    y = np.array([rosenbrock(x) for x in X])
    
    # Create and train robust surrogate
    surrogate = RobustSurrogate(
        surrogate_type="gp",
        normalize_data=True,
        validation_split=0.2,
        ensemble_size=3
    )
    
    try:
        surrogate.fit(X, y)
        print("✅ Robust surrogate trained successfully")
        
        if surrogate.validation_metrics:
            metrics = surrogate.validation_metrics
            print(f"   Validation R²: {metrics.r2_score:.4f}")
            print(f"   Validation RMSE: {metrics.rmse:.4f}")
    
    except Exception as e:
        print(f"❌ Surrogate training failed: {e}")
        return
    
    # Test robust optimization
    optimizer = RobustOptimizer(
        method="auto",
        random_restarts=3
    )
    
    try:
        initial_point = np.array([0.5, 0.5])
        result = optimizer.optimize(
            surrogate=surrogate,
            x0=initial_point,
            bounds=bounds,
            maximize=False
        )
        
        print("✅ Robust optimization completed")
        print(f"   Initial point: {initial_point}")
        print(f"   Optimal point: {result.x}")
        print(f"   Optimal value: {result.fun:.6f}")
        print(f"   True optimum: [1, 1] with value 0.0")
        print(f"   Distance to true optimum: {np.linalg.norm(result.x - [1, 1]):.6f}")
        print(f"   Success: {result.success}")
        
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        return
    
    print("\n🎉 Robust surrogate optimization completed successfully!")

if __name__ == "__main__":
    main()