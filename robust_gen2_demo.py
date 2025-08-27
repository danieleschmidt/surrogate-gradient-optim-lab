#!/usr/bin/env python3
"""
Generation 2: Robust Surrogate Optimization
Enhanced with comprehensive error handling, validation, logging, and monitoring
"""

import numpy as np
import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from scipy.optimize import minimize
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('surrogate_optim.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class OptimizationResult:
    """Structured optimization result with metadata."""
    x_optimal: np.ndarray
    f_optimal: float
    n_iterations: int
    convergence_time: float
    surrogate_error: float
    success: bool
    message: str
    metadata: Dict[str, Any]

@dataclass 
class ValidationMetrics:
    """Validation metrics for surrogate quality."""
    cross_val_score: float
    gradient_consistency: float
    prediction_error: float
    r2_score: float
    is_valid: bool

class InputValidator:
    """Comprehensive input validation."""
    
    @staticmethod
    def validate_bounds(bounds: List[Tuple[float, float]]) -> None:
        if not bounds:
            raise ValueError("Bounds cannot be empty")
        
        for i, (low, high) in enumerate(bounds):
            if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
                raise TypeError(f"Bounds[{i}] must be numeric")
            if low >= high:
                raise ValueError(f"Bounds[{i}]: lower bound {low} >= upper bound {high}")
            if abs(high - low) < 1e-10:
                raise ValueError(f"Bounds[{i}] too narrow: {high - low}")
                
    @staticmethod
    def validate_data(X: np.ndarray, y: np.ndarray) -> None:
        if X.size == 0 or y.size == 0:
            raise ValueError("Empty data arrays")
            
        if len(X) != len(y):
            raise ValueError(f"Mismatched array sizes: X={len(X)}, y={len(y)}")
            
        if len(X) < 10:
            raise ValueError(f"Insufficient data: {len(X)} samples < 10 minimum")
            
        # Check for NaN/Inf values
        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            raise ValueError("Data contains NaN values")
        if np.any(np.isinf(X)) or np.any(np.isinf(y)):
            raise ValueError("Data contains infinite values")
            
        # Check data distribution
        if np.std(y) < 1e-10:
            raise ValueError("Target values have zero variance")

class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""
    
    def __init__(self, failure_threshold: int = 3, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
    def call(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time < self.recovery_timeout:
                raise RuntimeError("Circuit breaker is OPEN")
            else:
                self.state = "HALF_OPEN"
                
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
            
    def _on_success(self):
        self.failure_count = 0
        self.state = "CLOSED"
        
    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"

class RobustSurrogateOptimizer:
    """Generation 2 - Robust surrogate optimizer with comprehensive error handling."""
    
    def __init__(self, 
                 surrogate_type: str = "neural_network",
                 validation_enabled: bool = True,
                 circuit_breaker_enabled: bool = True,
                 logging_level: str = "INFO"):
        
        self.surrogate_type = surrogate_type
        self.validation_enabled = validation_enabled
        self.circuit_breaker_enabled = circuit_breaker_enabled
        
        # Initialize components
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.circuit_breaker = CircuitBreaker() if circuit_breaker_enabled else None
        
        # Setup logging
        self.logger = logging.getLogger(f"RobustOptimizer_{id(self)}")
        self.logger.setLevel(getattr(logging, logging_level.upper()))
        
        # Performance monitoring
        self.metrics = {
            "fit_time": 0,
            "predict_calls": 0,
            "optimization_time": 0,
            "errors": []
        }
        
        self.logger.info(f"Initialized robust optimizer with {surrogate_type} surrogate")
        
    def collect_data(self, func, bounds: List[Tuple[float, float]], 
                    n_samples: int = 200, sampling: str = "sobol") -> Tuple[np.ndarray, np.ndarray]:
        """Robust data collection with validation and multiple sampling strategies."""
        
        InputValidator.validate_bounds(bounds)
        
        if n_samples < 20:
            self.logger.warning(f"Low sample count: {n_samples}, increasing to 50")
            n_samples = 50
            
        dim = len(bounds)
        self.logger.info(f"Collecting {n_samples} samples in {dim}D space using {sampling} sampling")
        
        try:
            if sampling == "sobol":
                # Quasi-random Sobol sequence (better space filling)
                from scipy.stats import qmc
                sampler = qmc.Sobol(d=dim, scramble=True)
                samples = sampler.random(n_samples)
                # Scale to bounds
                X = np.array([
                    [bounds[j][0] + s[j] * (bounds[j][1] - bounds[j][0]) for j in range(dim)]
                    for s in samples
                ])
            else:
                # Random sampling fallback
                X = np.random.uniform(
                    low=[b[0] for b in bounds],
                    high=[b[1] for b in bounds], 
                    size=(n_samples, dim)
                )
                
            # Evaluate function with error handling
            y = []
            failed_evaluations = 0
            
            for i, x in enumerate(X):
                try:
                    if self.circuit_breaker:
                        value = self.circuit_breaker.call(func, x)
                    else:
                        value = func(x)
                    y.append(value)
                except Exception as e:
                    failed_evaluations += 1
                    self.logger.warning(f"Function evaluation failed at x={x}: {e}")
                    # Use interpolated value or median
                    if y:
                        y.append(np.median(y))
                    else:
                        y.append(0.0)
                        
            y = np.array(y)
            
            if failed_evaluations > n_samples * 0.1:
                self.logger.error(f"High failure rate: {failed_evaluations}/{n_samples} evaluations failed")
                
            # Validate collected data
            InputValidator.validate_data(X, y)
            
            self.logger.info(f"Successfully collected data: range [{y.min():.3f}, {y.max():.3f}]")
            return X, y
            
        except Exception as e:
            self.logger.error(f"Data collection failed: {e}")
            self.metrics["errors"].append(f"Data collection: {str(e)}")
            raise
            
    def validate_surrogate(self, X: np.ndarray, y: np.ndarray) -> ValidationMetrics:
        """Comprehensive surrogate validation."""
        
        try:
            # Cross-validation score
            cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='r2')
            cv_score = np.mean(cv_scores)
            
            # Gradient consistency check
            test_points = X[:min(10, len(X))]
            gradient_errors = []
            
            for x in test_points:
                grad1 = self._finite_diff_gradient(x, epsilon=1e-6)
                grad2 = self._finite_diff_gradient(x, epsilon=1e-7)
                error = np.linalg.norm(grad1 - grad2) / (np.linalg.norm(grad1) + 1e-10)
                gradient_errors.append(error)
                
            gradient_consistency = 1.0 - np.mean(gradient_errors)
            
            # Prediction error
            y_pred = self.model.predict(X)
            prediction_error = np.mean(np.abs(y - y_pred))
            r2 = cv_score
            
            # Overall validation
            is_valid = (cv_score > 0.5 and 
                       gradient_consistency > 0.8 and 
                       prediction_error < np.std(y))
                       
            metrics = ValidationMetrics(
                cross_val_score=cv_score,
                gradient_consistency=gradient_consistency,
                prediction_error=prediction_error,
                r2_score=r2,
                is_valid=is_valid
            )
            
            self.logger.info(f"Validation metrics: CV={cv_score:.3f}, "
                           f"Grad={gradient_consistency:.3f}, Valid={is_valid}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return ValidationMetrics(0, 0, float('inf'), 0, False)
        
    def fit_surrogate(self, X: np.ndarray, y: np.ndarray) -> Any:
        """Train surrogate with robust error handling and validation."""
        
        start_time = time.time()
        
        try:
            # Input validation
            InputValidator.validate_data(X, y)
            
            # Data preprocessing
            X_scaled = self.scaler.fit_transform(X)
            
            # Model initialization with error handling
            if self.surrogate_type == "neural_network":
                self.model = MLPRegressor(
                    hidden_layer_sizes=(128, 64, 32),
                    max_iter=2000,
                    alpha=0.001,
                    early_stopping=True,
                    validation_fraction=0.2,
                    n_iter_no_change=50,
                    random_state=42
                )
            elif self.surrogate_type == "gaussian_process":
                # Adaptive kernel selection
                kernel = RBF(length_scale=1.0) 
                self.model = GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=1e-6,
                    n_restarts_optimizer=5,
                    random_state=42
                )
            else:
                raise ValueError(f"Unknown surrogate type: {self.surrogate_type}")
                
            # Fit model
            self.logger.info(f"Training {self.surrogate_type} with {len(X)} samples...")
            self.model.fit(X_scaled, y)
            self.is_fitted = True
            
            # Validation
            if self.validation_enabled:
                validation_metrics = self.validate_surrogate(X_scaled, y)
                if not validation_metrics.is_valid:
                    self.logger.warning("Surrogate validation failed - proceeding with caution")
                    
            fit_time = time.time() - start_time
            self.metrics["fit_time"] = fit_time
            
            self.logger.info(f"Surrogate training completed in {fit_time:.2f}s")
            return self.model
            
        except Exception as e:
            self.logger.error(f"Surrogate training failed: {e}")
            self.metrics["errors"].append(f"Training: {str(e)}")
            raise
            
    def predict(self, x: np.ndarray) -> float:
        """Robust prediction with error handling."""
        
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
            
        try:
            x = np.atleast_2d(x)
            x_scaled = self.scaler.transform(x)
            
            if self.circuit_breaker:
                prediction = self.circuit_breaker.call(self.model.predict, x_scaled)[0]
            else:
                prediction = self.model.predict(x_scaled)[0]
                
            self.metrics["predict_calls"] += 1
            return float(prediction)
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            self.metrics["errors"].append(f"Prediction: {str(e)}")
            raise
            
    def _finite_diff_gradient(self, x: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
        """Robust numerical gradient with adaptive epsilon."""
        
        grad = np.zeros_like(x)
        f0 = self.predict(x)
        
        # Adaptive epsilon based on function scale
        adaptive_eps = max(epsilon, abs(f0) * 1e-8)
        
        for i in range(len(x)):
            try:
                x_plus = x.copy()
                x_plus[i] += adaptive_eps
                f_plus = self.predict(x_plus)
                grad[i] = (f_plus - f0) / adaptive_eps
            except Exception as e:
                self.logger.warning(f"Gradient computation failed for dimension {i}: {e}")
                grad[i] = 0.0
                
        return grad
        
    def optimize(self, initial_point: np.ndarray, bounds: Optional[List[Tuple[float, float]]] = None,
                method: str = "L-BFGS-B", max_iterations: int = 1000) -> OptimizationResult:
        """Robust optimization with comprehensive monitoring."""
        
        start_time = time.time()
        
        try:
            if not self.is_fitted:
                raise RuntimeError("Model must be fitted first")
                
            # Input validation
            if bounds:
                InputValidator.validate_bounds(bounds)
                
            initial_point = np.asarray(initial_point)
            
            def objective(x):
                return -self.predict(x)  # Maximize by minimizing negative
                
            def grad_objective(x):
                return -self._finite_diff_gradient(x)
            
            self.logger.info(f"Starting optimization with {method} from {initial_point}")
            
            # Optimization with multiple restart strategy
            best_result = None
            best_value = float('inf')
            
            for restart in range(3):  # Multiple restarts for robustness
                try:
                    if restart > 0:
                        # Perturb initial point for restart
                        perturb = np.random.normal(0, 0.1, size=initial_point.shape)
                        start_point = initial_point + perturb
                    else:
                        start_point = initial_point
                        
                    result = minimize(
                        objective,
                        start_point,
                        method=method,
                        jac=grad_objective,
                        bounds=bounds,
                        options={'maxiter': max_iterations}
                    )
                    
                    if result.fun < best_value:
                        best_result = result
                        best_value = result.fun
                        
                except Exception as e:
                    self.logger.warning(f"Optimization restart {restart} failed: {e}")
                    continue
                    
            if best_result is None:
                raise RuntimeError("All optimization restarts failed")
                
            # Validate result
            optimal_x = best_result.x
            optimal_value = -best_result.fun
            
            optimization_time = time.time() - start_time
            self.metrics["optimization_time"] = optimization_time
            
            # Create comprehensive result
            result = OptimizationResult(
                x_optimal=optimal_x,
                f_optimal=optimal_value,
                n_iterations=best_result.nit,
                convergence_time=optimization_time,
                surrogate_error=0.0,  # Would need true function to compute
                success=best_result.success,
                message=best_result.message,
                metadata={
                    "method": method,
                    "n_restarts": 3,
                    "function_evaluations": best_result.nfev,
                    "gradient_evaluations": best_result.njev if hasattr(best_result, 'njev') else 0
                }
            )
            
            self.logger.info(f"Optimization completed: x*={optimal_x}, f*={optimal_value:.3f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            self.metrics["errors"].append(f"Optimization: {str(e)}")
            raise
            
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        return {
            "surrogate_type": self.surrogate_type,
            "is_fitted": self.is_fitted,
            "metrics": self.metrics.copy(),
            "validation_enabled": self.validation_enabled,
            "circuit_breaker_enabled": self.circuit_breaker_enabled,
            "circuit_breaker_state": self.circuit_breaker.state if self.circuit_breaker else None
        }
        
    def save_state(self, filepath: str) -> None:
        """Save optimizer state for reproducibility."""
        state = {
            "config": {
                "surrogate_type": self.surrogate_type,
                "validation_enabled": self.validation_enabled,
                "circuit_breaker_enabled": self.circuit_breaker_enabled
            },
            "is_fitted": self.is_fitted,
            "metrics": self.metrics
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
            
        self.logger.info(f"State saved to {filepath}")

def demo_robust_optimization():
    """Demonstrate robust surrogate optimization."""
    print("🛡️ Generation 2: Robust Surrogate Optimization Demo")
    print("="*55)
    
    # Define a more challenging test function with noise
    def noisy_himmelblau(x):
        base = -((x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2)
        noise = np.random.normal(0, 0.1)  # Add small amount of noise
        return base + noise
    
    bounds = [(-6, 6), (-6, 6)]
    
    # Test robust optimization
    for surrogate_type in ["neural_network", "gaussian_process"]:
        print(f"\n🧠 Testing Robust {surrogate_type.replace('_', ' ').title()} Surrogate")
        print("-" * 40)
        
        # Initialize robust optimizer
        optimizer = RobustSurrogateOptimizer(
            surrogate_type=surrogate_type,
            validation_enabled=True,
            circuit_breaker_enabled=True,
            logging_level="INFO"
        )
        
        try:
            # Collect data with advanced sampling
            print("📊 Collecting training data with Sobol sampling...")
            X, y = optimizer.collect_data(noisy_himmelblau, bounds, n_samples=300, sampling="sobol")
            print(f"   ✅ Collected {len(X)} samples successfully")
            
            # Train surrogate with validation
            print("🎓 Training surrogate with validation...")
            optimizer.fit_surrogate(X, y)
            print(f"   ✅ Training completed")
            
            # Optimize with multiple strategies
            print("⚡ Robust optimization with multiple restarts...")
            initial_point = np.array([2.5, 2.5])
            result = optimizer.optimize(initial_point, bounds, max_iterations=500)
            
            print(f"   🎯 Optimization Results:")
            print(f"      Optimum: [{result.x_optimal[0]:.3f}, {result.x_optimal[1]:.3f}]")
            print(f"      Value: {result.f_optimal:.3f}")
            print(f"      Iterations: {result.n_iterations}")
            print(f"      Time: {result.convergence_time:.2f}s")
            print(f"      Success: {result.success}")
            
            # Performance metrics
            metrics = optimizer.get_performance_metrics()
            print(f"   📈 Performance Metrics:")
            print(f"      Fit time: {metrics['metrics']['fit_time']:.2f}s")
            print(f"      Predict calls: {metrics['metrics']['predict_calls']}")
            print(f"      Errors: {len(metrics['metrics']['errors'])}")
            
            # Save state
            state_file = f"robust_optimizer_{surrogate_type}.json"
            optimizer.save_state(state_file)
            print(f"   💾 State saved to {state_file}")
            
        except Exception as e:
            print(f"   ❌ ERROR: {e}")

if __name__ == "__main__":
    demo_robust_optimization()