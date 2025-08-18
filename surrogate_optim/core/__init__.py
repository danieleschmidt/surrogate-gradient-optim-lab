"""Core optimization components with enhanced robustness."""

# Re-export SurrogateOptimizer from the parent core.py module
import sys
from pathlib import Path

# Add parent directory to path to access core.py
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

try:
    # Import all required dependencies first
    from ..data.collector import DataCollector, collect_data
    from ..models.base import Dataset, Surrogate
    from ..models.gaussian_process import GPSurrogate
    from ..models.neural import NeuralSurrogate
    from ..models.random_forest import RandomForestSurrogate
    from ..optimizers.base import OptimizationResult
    from ..optimizers.gradient_descent import GradientDescentOptimizer
    from ..optimizers.trust_region import TrustRegionOptimizer
    from ..optimizers.multi_start import MultiStartOptimizer
    
    # Now create a SurrogateOptimizer class here
    from typing import Any, Callable, Dict, List, Optional, Tuple, Union
    import jax.numpy as jnp
    from jax import Array
    
    class SurrogateOptimizer:
        """Main interface for surrogate gradient optimization."""
        
        def __init__(
            self,
            surrogate_type: str = "neural_network",
            surrogate_params: Optional[Dict[str, Any]] = None,
            optimizer_type: str = "gradient_descent",
            optimizer_params: Optional[Dict[str, Any]] = None,
        ):
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
        
        def _create_surrogate(self) -> Surrogate:
            """Create surrogate model based on configuration."""
            if self.surrogate_type in ["neural_network", "nn"]:
                return NeuralSurrogate(**self.surrogate_params)
            elif self.surrogate_type in ["gaussian_process", "gp"]:
                return GPSurrogate(**self.surrogate_params)
            elif self.surrogate_type in ["random_forest", "rf"]:
                return RandomForestSurrogate(**self.surrogate_params)
            else:
                raise ValueError(f"Unknown surrogate type: {self.surrogate_type}")
        
        def _create_optimizer(self):
            """Create optimizer based on configuration."""
            if self.optimizer_type == "gradient_descent":
                return GradientDescentOptimizer(**self.optimizer_params)
            elif self.optimizer_type == "trust_region":
                return TrustRegionOptimizer(**self.optimizer_params)
            elif self.optimizer_type == "multi_start":
                return MultiStartOptimizer(**self.optimizer_params)
            else:
                raise ValueError(f"Unknown optimizer type: {self.optimizer_type}")
        
        def fit_surrogate(self, data):
            """Train the surrogate model on the given data."""
            if isinstance(data, dict):
                from ..models.base import Dataset
                data = Dataset(X=data["X"], y=data["y"])
            
            self.surrogate.fit(data)
            self.is_fitted = True
            self.training_data = data
            return self
        
        def optimize(self, initial_point: Array, bounds=None):
            """Optimize using the trained surrogate."""
            if not self.is_fitted:
                raise ValueError("Surrogate model must be fitted before optimization")
            
            # Simple gradient descent optimization
            result = self.optimizer.optimize(
                surrogate=self.surrogate,
                x0=initial_point,
                bounds=bounds
            )
            return result.x if hasattr(result, 'x') else result

except ImportError as e:
    print(f"Warning: Could not import SurrogateOptimizer dependencies: {e}")
    SurrogateOptimizer = None

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
    "SurrogateOptimizer",
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