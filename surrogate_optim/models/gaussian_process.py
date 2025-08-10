"""Gaussian process surrogate model using scikit-learn with JAX integration."""

from typing import Any, Dict, List, Optional, Union, Tuple
import warnings

import numpy as np
import jax.numpy as jnp
from jax import Array, grad, vmap, jit
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel

from .base import Surrogate, Dataset


class GPSurrogate(Surrogate):
    """Gaussian Process surrogate model with analytical gradients.
    
    Uses scikit-learn's GaussianProcessRegressor with JAX integration
    for efficient gradient computation.
    """
    
    def __init__(
        self,
        kernel: str = "rbf",
        length_scale: float = 1.0,
        nu: float = 1.5,
        noise_level: float = 1e-5,
        n_restarts_optimizer: int = 5,
        normalize_y: bool = True,
        alpha: float = 1e-10,
    ):
        """Initialize GP surrogate.
        
        Args:
            kernel: Kernel type ('rbf', 'matern', 'auto')
            length_scale: Initial length scale for kernel
            nu: Nu parameter for Matern kernel
            noise_level: Noise level for WhiteKernel
            n_restarts_optimizer: Number of optimizer restarts
            normalize_y: Whether to normalize targets
            alpha: Value for diagonal regularization
        """
        self.kernel_type = kernel
        self.length_scale = length_scale
        self.nu = nu
        self.noise_level = noise_level
        self.n_restarts_optimizer = n_restarts_optimizer
        self.normalize_y = normalize_y
        self.alpha = alpha
        
        # Create kernel
        self.kernel = self._create_kernel()
        
        # Initialize GP
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=alpha,
            n_restarts_optimizer=n_restarts_optimizer,
            normalize_y=normalize_y,
            copy_X_train=True
        )
        
        # JAX-compiled functions for gradients
        self._jax_predict_fn = None
        self._jax_gradient_fn = None
        
        # Training data storage
        self.X_train = None
        self.y_train = None
    
    def _create_kernel(self):
        """Create kernel based on specification."""
        if self.kernel_type in ["rbf", "gaussian"]:
            base_kernel = RBF(length_scale=self.length_scale)
        elif self.kernel_type == "matern":
            if self.nu not in [0.5, 1.5, 2.5]:
                warnings.warn(
                    f"Matern kernel with nu={self.nu} requires specialized implementation. "
                    f"Currently supported: nu=1.5, nu=2.5"
                )
            base_kernel = Matern(length_scale=self.length_scale, nu=self.nu)
        elif self.kernel_type == "auto":
            # Combination kernel for flexibility
            rbf = RBF(length_scale=self.length_scale)
            matern = Matern(length_scale=self.length_scale, nu=1.5)
            base_kernel = rbf + matern
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
        
        # Add noise kernel
        if self.noise_level > 0:
            noise_kernel = WhiteKernel(noise_level=self.noise_level)
            return base_kernel + noise_kernel
        else:
            return base_kernel
    
    def fit(self, dataset: Dataset) -> "GPSurrogate":
        """Fit GP to training data."""
        # Store training data
        self.X_train = dataset.X
        self.y_train = dataset.y
        
        # Convert JAX arrays to numpy for sklearn
        X_np = np.asarray(dataset.X)
        y_np = np.asarray(dataset.y)
        
        # Fit GP
        self.gp.fit(X_np, y_np)
        
        # Setup JAX functions for gradients
        self._setup_jax_functions()
        
        return self
    
    def _setup_jax_functions(self):
        """Setup JAX-compiled functions for efficient gradient computation."""
        try:
            # Get inverse of kernel matrix for predictions
            K_inv_y = self.gp.alpha_
            X_train_jax = jnp.array(self.X_train)
            
            # Get kernel hyperparameters
            kernel_params = self.gp.kernel_.get_params()
            
            def jax_rbf_kernel(x1, x2, length_scale=1.0):
                """JAX implementation of RBF kernel."""
                diff = x1 - x2
                return jnp.exp(-0.5 * jnp.sum(diff**2) / (length_scale**2))
            
            def jax_predict(x):
                """JAX-based prediction function."""
                # Compute kernel vector between x and training points
                k_vec = vmap(lambda x_train: jax_rbf_kernel(x, x_train, self.gp.kernel_.k1.length_scale))(X_train_jax)
                
                # Prediction is k^T @ K^(-1) @ y
                prediction = jnp.dot(k_vec, K_inv_y)
                
                # Add mean if normalize_y was used
                if self.normalize_y:
                    prediction += self.gp._y_train_mean
                
                return prediction
            
            # Compile functions
            self._jax_predict_fn = jit(jax_predict)
            self._jax_gradient_fn = jit(grad(jax_predict))
            
        except Exception as e:
            warnings.warn(f"Failed to setup JAX functions: {e}. Using fallback methods.")
            self._jax_predict_fn = None
            self._jax_gradient_fn = None
    
    def predict(self, x: Array) -> Array:
        """Predict function values."""
        if self.gp is None:
            raise ValueError("Model must be trained before prediction")
        
        # Convert JAX array to numpy for sklearn
        x_np = jnp.asarray(x)
        
        if x_np.ndim == 1:
            x_np = x_np[None, :]
            single_point = True
        else:
            single_point = False
        
        # Use sklearn GP for prediction
        predictions, _ = self.gp.predict(x_np, return_std=True)
        
        if single_point:
            return jnp.array(predictions[0])
        return jnp.array(predictions)
    
    def gradient(self, x: Array) -> Array:
        """Compute analytical gradients of the GP mean function."""
        if self.gp is None:
            raise ValueError("Model must be trained before gradient computation")
        
        if self._jax_gradient_fn is None:
            # Fallback to finite differences if JAX setup failed
            return self._finite_difference_gradient(x)
        
        # Use JAX for exact gradients
        if x.ndim == 1:
            return self._jax_gradient_fn(x)
        else:
            return vmap(self._jax_gradient_fn)(x)
    
    def _finite_difference_gradient(self, x: Array, eps: float = 1e-6) -> Array:
        """Compute gradients using finite differences as fallback."""
        if x.ndim == 1:
            x = x[None, :]
            single_point = True
        else:
            single_point = False
        
        gradients = jnp.zeros_like(x)
        
        for i in range(x.shape[1]):
            x_plus = x.at[:, i].add(eps)
            x_minus = x.at[:, i].add(-eps)
            
            y_plus = self.predict(x_plus)
            y_minus = self.predict(x_minus)
            
            gradients = gradients.at[:, i].set((y_plus - y_minus) / (2 * eps))
        
        if single_point:
            return gradients[0]
        return gradients
    
    def uncertainty(self, x: Array) -> Array:
        """Compute prediction uncertainty (standard deviation)."""
        if self.gp is None:
            raise ValueError("Model must be trained before uncertainty computation")
        
        # Convert JAX array to numpy for sklearn
        x_np = jnp.asarray(x)
        
        if x_np.ndim == 1:
            x_np = x_np[None, :]
            single_point = True
        else:
            single_point = False
        
        # Use sklearn GP for uncertainty
        _, std = self.gp.predict(x_np, return_std=True)
        
        if single_point:
            return jnp.array(std[0])
        return jnp.array(std)
    
    def predict_with_uncertainty(self, x: Array) -> Tuple[Array, Array]:
        """Predict with uncertainty in a single call for efficiency."""
        if self.gp is None:
            raise ValueError("Model must be trained before prediction")
        
        # Convert JAX array to numpy for sklearn
        x_np = jnp.asarray(x)
        
        if x_np.ndim == 1:
            x_np = x_np[None, :]
            single_point = True
        else:
            single_point = False
        
        # Use sklearn GP for prediction with uncertainty
        predictions, std = self.gp.predict(x_np, return_std=True)
        
        if single_point:
            return jnp.array(predictions[0]), jnp.array(std[0])
        return jnp.array(predictions), jnp.array(std)