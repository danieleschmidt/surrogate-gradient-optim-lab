"""Gaussian Process surrogate model implementation."""

from typing import Optional

import jax
import jax.numpy as jnp
from jax import Array, grad, jit, vmap
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel

from .base import Dataset, Surrogate


class GPSurrogate(Surrogate):
    """Gaussian Process surrogate model with analytical gradients.
    
    Combines scikit-learn's GP implementation with JAX for gradient computation.
    Provides both predictions and uncertainty quantification.
    """
    
    def __init__(
        self,
        kernel: str = "rbf",
        length_scale: float = 1.0,
        noise_level: float = 1e-5,
        alpha: float = 1e-10,
        normalize_y: bool = True,
        n_restarts_optimizer: int = 10,
    ):
        """Initialize Gaussian Process surrogate.
        
        Args:
            kernel: Kernel type ('rbf', 'matern32', 'matern52', 'auto')
            length_scale: Initial length scale for the kernel
            noise_level: Noise level for observations
            alpha: Value added to diagonal for numerical stability
            normalize_y: Whether to normalize target values
            n_restarts_optimizer: Number of restarts for hyperparameter optimization
        """
        self.kernel_name = kernel
        self.length_scale = length_scale
        self.noise_level = noise_level
        self.alpha = alpha
        self.normalize_y = normalize_y
        self.n_restarts_optimizer = n_restarts_optimizer
        
        # Model components
        self.gp = None
        self.X_train = None
        self.y_train = None
        self.input_dim = None
        
        # Compiled JAX functions
        self._jax_predict_fn = None
        self._jax_gradient_fn = None
    
    def _create_kernel(self, input_dim: int):
        """Create kernel based on configuration."""
        if self.kernel_name == "rbf":
            base_kernel = RBF(length_scale=self.length_scale)
        elif self.kernel_name == "matern32":
            base_kernel = Matern(length_scale=self.length_scale, nu=1.5)
        elif self.kernel_name == "matern52":
            base_kernel = Matern(length_scale=self.length_scale, nu=2.5)
        elif self.kernel_name == "auto":
            # Automatic kernel selection - use RBF + Matern combination
            base_kernel = RBF(length_scale=self.length_scale) + Matern(length_scale=self.length_scale, nu=2.5)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel_name}")
        
        # Add white noise kernel for numerical stability
        if self.noise_level > 0:
            kernel = base_kernel + WhiteKernel(noise_level=self.noise_level)
        else:
            kernel = base_kernel
        
        return kernel
    
    def fit(self, dataset: Dataset) -> "GPSurrogate":
        """Train the Gaussian Process model."""
        self.input_dim = dataset.n_dims
        self.X_train = dataset.X
        self.y_train = dataset.y
        
        # Create and configure GP
        kernel = self._create_kernel(self.input_dim)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.alpha,
            normalize_y=self.normalize_y,
            n_restarts_optimizer=self.n_restarts_optimizer,
            random_state=42,
        )
        
        # Fit the GP
        print("Training Gaussian Process...")
        self.gp.fit(self.X_train, self.y_train)
        print(f"GP trained. Final kernel: {self.gp.kernel_}")
        
        # Create JAX-based prediction function for gradients
        self._setup_jax_functions()
        
        return self
    
    def _setup_jax_functions(self):
        """Setup JAX functions for efficient gradient computation."""
        # Convert GP parameters to JAX arrays for gradient computation
        K_inv_y = jnp.array(self.gp.alpha_)  # This is K^(-1) @ y from sklearn
        X_train_jax = jnp.array(self.X_train)
        
        # Get kernel hyperparameters
        kernel_params = self.gp.kernel_.get_params()
        
        def jax_rbf_kernel(x1, x2, length_scale=1.0):
            \"\"\"JAX implementation of RBF kernel.\"\"\"\n            diff = x1 - x2\n            return jnp.exp(-0.5 * jnp.sum(diff**2) / (length_scale**2))\n        \n        def jax_predict(x):\n            \"\"\"JAX-based prediction function.\"\"\"\n            # Compute kernel vector between x and training points\n            k_vec = vmap(lambda x_train: jax_rbf_kernel(x, x_train, self.gp.kernel_.k1.length_scale))(X_train_jax)\n            \n            # Prediction is k^T @ K^(-1) @ y\n            prediction = jnp.dot(k_vec, K_inv_y)\n            \n            # Add mean if normalize_y was used\n            if self.normalize_y:\n                prediction += self.gp._y_train_mean\n            \n            return prediction\n        \n        # Compile functions\n        self._jax_predict_fn = jit(jax_predict)\n        self._jax_gradient_fn = jit(grad(jax_predict))\n    \n    def predict(self, x: Array) -> Array:\n        \"\"\"Predict function values.\"\"\"\n        if self.gp is None:\n            raise ValueError(\"Model must be trained before prediction\")\n        \n        # Convert JAX array to numpy for sklearn\n        x_np = jnp.asarray(x)\n        \n        if x_np.ndim == 1:\n            x_np = x_np[None, :]\n            single_point = True\n        else:\n            single_point = False\n        \n        # Use sklearn GP for prediction\n        predictions, _ = self.gp.predict(x_np, return_std=True)\n        \n        if single_point:\n            return jnp.array(predictions[0])\n        return jnp.array(predictions)\n    \n    def gradient(self, x: Array) -> Array:\n        \"\"\"Compute analytical gradients of the GP mean function.\"\"\"\n        if self.gp is None:\n            raise ValueError(\"Model must be trained before gradient computation\")\n        \n        if self._jax_gradient_fn is None:\n            # Fallback to finite differences if JAX setup failed\n            return self._finite_difference_gradient(x)\n        \n        # Use JAX for exact gradients\n        if x.ndim == 1:\n            return self._jax_gradient_fn(x)\n        else:\n            return vmap(self._jax_gradient_fn)(x)\n    \n    def _finite_difference_gradient(self, x: Array, eps: float = 1e-6) -> Array:\n        \"\"\"Compute gradients using finite differences as fallback.\"\"\"\n        if x.ndim == 1:\n            x = x[None, :]\n            single_point = True\n        else:\n            single_point = False\n        \n        gradients = jnp.zeros_like(x)\n        \n        for i in range(x.shape[1]):\n            x_plus = x.at[:, i].add(eps)\n            x_minus = x.at[:, i].add(-eps)\n            \n            y_plus = self.predict(x_plus)\n            y_minus = self.predict(x_minus)\n            \n            gradients = gradients.at[:, i].set((y_plus - y_minus) / (2 * eps))\n        \n        if single_point:\n            return gradients[0]\n        return gradients\n    \n    def uncertainty(self, x: Array) -> Array:\n        \"\"\"Compute prediction uncertainty (standard deviation).\"\"\"\n        if self.gp is None:\n            raise ValueError(\"Model must be trained before uncertainty computation\")\n        \n        # Convert JAX array to numpy for sklearn\n        x_np = jnp.asarray(x)\n        \n        if x_np.ndim == 1:\n            x_np = x_np[None, :]\n            single_point = True\n        else:\n            single_point = False\n        \n        # Use sklearn GP for uncertainty\n        _, std = self.gp.predict(x_np, return_std=True)\n        \n        if single_point:\n            return jnp.array(std[0])\n        return jnp.array(std)\n    \n    def predict_with_uncertainty(self, x: Array) -> tuple[Array, Array]:\n        \"\"\"Predict with uncertainty in a single call for efficiency.\"\"\"\n        if self.gp is None:\n            raise ValueError(\"Model must be trained before prediction\")\n        \n        # Convert JAX array to numpy for sklearn\n        x_np = jnp.asarray(x)\n        \n        if x_np.ndim == 1:\n            x_np = x_np[None, :]\n            single_point = True\n        else:\n            single_point = False\n        \n        # Use sklearn GP for prediction with uncertainty\n        predictions, std = self.gp.predict(x_np, return_std=True)\n        \n        if single_point:\n            return jnp.array(predictions[0]), jnp.array(std[0])\n        return jnp.array(predictions), jnp.array(std)