"""Base classes for surrogate models and data structures."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import jax.numpy as jnp
from jax import Array


@dataclass
class Dataset:
    """Dataset for surrogate model training and evaluation.
    
    Attributes:
        X: Input points with shape [n_samples, n_dims]
        y: Function values with shape [n_samples]
        gradients: Optional gradient vectors with shape [n_samples, n_dims]
        metadata: Additional metadata for the dataset
    """
    X: Array
    y: Array
    gradients: Optional[Array] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate dataset consistency."""
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError(f"Inconsistent sample counts: X={self.X.shape[0]}, y={self.y.shape[0]}")
        
        if self.gradients is not None:
            expected_shape = (self.X.shape[0], self.X.shape[1])
            if self.gradients.shape != expected_shape:
                raise ValueError(f"Invalid gradient shape: {self.gradients.shape}, expected {expected_shape}")
    
    @property
    def n_samples(self) -> int:
        """Number of samples in the dataset."""
        return self.X.shape[0]
    
    @property
    def n_dims(self) -> int:
        """Number of input dimensions."""
        return self.X.shape[1]


class Surrogate(ABC):
    """Abstract base class for surrogate models.
    
    All surrogate models must implement prediction and gradient computation.
    """
    
    @abstractmethod
    def fit(self, dataset: Dataset) -> "Surrogate":
        """Train the surrogate model on the given dataset.
        
        Args:
            dataset: Training data with inputs, outputs, and optional gradients
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(self, x: Array) -> Array:
        """Predict function values at given points.
        
        Args:
            x: Input points with shape [n_points, n_dims] or [n_dims]
            
        Returns:
            Predicted function values with shape [n_points] or scalar
        """
        pass
    
    @abstractmethod
    def gradient(self, x: Array) -> Array:
        """Compute gradients at given points.
        
        Args:
            x: Input points with shape [n_points, n_dims] or [n_dims]
            
        Returns:
            Gradient vectors with shape [n_points, n_dims] or [n_dims]
        """
        pass
    
    def uncertainty(self, x: Array) -> Array:
        """Estimate prediction uncertainty at given points.
        
        Default implementation returns zeros. Override for models with
        uncertainty quantification capabilities.
        
        Args:
            x: Input points with shape [n_points, n_dims] or [n_dims]
            
        Returns:
            Uncertainty estimates with shape [n_points] or scalar
        """
        # Ensure x is 2D for consistent handling
        if x.ndim == 1:
            x = x[None, :]
        return jnp.zeros(x.shape[0])
    
    def predict_with_uncertainty(self, x: Array) -> tuple[Array, Array]:
        """Predict function values with uncertainty estimates.
        
        Args:
            x: Input points with shape [n_points, n_dims] or [n_dims]
            
        Returns:
            Tuple of (predictions, uncertainties)
        """
        predictions = self.predict(x)
        uncertainties = self.uncertainty(x)
        return predictions, uncertainties