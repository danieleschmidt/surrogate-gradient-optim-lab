"""Surrogate model implementations for function approximation and gradient computation."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, vmap
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.preprocessing import StandardScaler


Array = Union[np.ndarray, jnp.ndarray]


class Surrogate(ABC):
    """Abstract base class for surrogate models."""

    @abstractmethod
    def fit(self, X: Array, y: Array) -> None:
        """Fit the surrogate model to training data.
        
        Args:
            X: Input points [n_samples, n_dims]
            y: Function values [n_samples]
        """
        pass

    @abstractmethod
    def predict(self, x: Array) -> float:
        """Predict function value at a point.
        
        Args:
            x: Input point [n_dims]
            
        Returns:
            Predicted function value
        """
        pass

    @abstractmethod
    def gradient(self, x: Array) -> Array:
        """Compute gradient at a point.
        
        Args:
            x: Input point [n_dims]
            
        Returns:
            Gradient vector [n_dims]
        """
        pass

    def uncertainty(self, x: Array) -> float:
        """Compute prediction uncertainty at a point.
        
        Args:
            x: Input point [n_dims]
            
        Returns:
            Uncertainty estimate (standard deviation)
        """
        return 0.0

    def predict_batch(self, X: Array) -> Array:
        """Predict function values for multiple points.
        
        Args:
            X: Input points [n_samples, n_dims]
            
        Returns:
            Predicted function values [n_samples]
        """
        return vmap(self.predict)(X)

    def gradient_batch(self, X: Array) -> Array:
        """Compute gradients for multiple points.
        
        Args:
            X: Input points [n_samples, n_dims]
            
        Returns:
            Gradient vectors [n_samples, n_dims]
        """
        return vmap(self.gradient)(X)


class NeuralSurrogate(Surrogate):
    """Neural network surrogate model with automatic differentiation."""

    def __init__(
        self,
        hidden_dims: List[int] = [64, 64],
        activation: str = "relu",
        learning_rate: float = 1e-3,
        epochs: int = 1000,
        batch_size: int = 32,
        ensemble_size: int = 1,
    ):
        """Initialize neural network surrogate.
        
        Args:
            hidden_dims: Hidden layer dimensions
            activation: Activation function ("relu", "tanh", "gelu")
            learning_rate: Training learning rate
            epochs: Number of training epochs
            batch_size: Training batch size
            ensemble_size: Number of ensemble members for uncertainty
        """
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.ensemble_size = ensemble_size
        
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.params = None
        self.input_dim = None

    def _activation_fn(self, x: Array) -> Array:
        """Apply activation function."""
        if self.activation == "relu":
            return jnp.maximum(0, x)
        elif self.activation == "tanh":
            return jnp.tanh(x)
        elif self.activation == "gelu":
            return 0.5 * x * (1 + jnp.tanh(jnp.sqrt(2 / jnp.pi) * (x + 0.044715 * x**3)))
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def _init_params(self, input_dim: int) -> Dict:
        """Initialize network parameters."""
        key = jax.random.PRNGKey(42)
        params = {}
        
        layers = [input_dim] + self.hidden_dims + [1]
        for i in range(len(layers) - 1):
            key, subkey = jax.random.split(key)
            w_init = jax.random.normal(subkey, (layers[i], layers[i+1])) * jnp.sqrt(2.0 / layers[i])
            b_init = jnp.zeros(layers[i+1])
            params[f'w{i}'] = w_init
            params[f'b{i}'] = b_init
            
        return params

    def _forward(self, params: Dict, x: Array) -> float:
        """Forward pass through network."""
        h = x
        num_layers = len(self.hidden_dims) + 1
        
        for i in range(num_layers - 1):
            h = h @ params[f'w{i}'] + params[f'b{i}']
            h = self._activation_fn(h)
            
        # Output layer (no activation)
        output = h @ params[f'w{num_layers-1}'] + params[f'b{num_layers-1}']
        return output.squeeze()

    def _loss_fn(self, params: Dict, X: Array, y: Array) -> float:
        """Compute loss function."""
        predictions = vmap(lambda x: self._forward(params, x))(X)
        return jnp.mean((predictions - y) ** 2)

    def fit(self, X: Array, y: Array) -> None:
        """Fit neural network to training data."""
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Normalize inputs and outputs
        X_norm = self.scaler_X.fit_transform(X)
        y_norm = self.scaler_y.fit_transform(y.reshape(-1, 1)).squeeze()
        
        self.input_dim = X.shape[1]
        self.params = self._init_params(self.input_dim)
        
        # Convert to JAX arrays
        X_jax = jnp.array(X_norm)
        y_jax = jnp.array(y_norm)
        
        # Training loop with Adam optimizer
        import optax
        optimizer = optax.adam(self.learning_rate)
        opt_state = optimizer.init(self.params)
        
        @jit
        def update_step(params, opt_state, X_batch, y_batch):
            loss, grads = jax.value_and_grad(self._loss_fn)(params, X_batch, y_batch)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss
        
        # Training epochs
        for epoch in range(self.epochs):
            # Simple training without batching for now
            self.params, opt_state, loss = update_step(
                self.params, opt_state, X_jax, y_jax
            )
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")

    def predict(self, x: Array) -> float:
        """Predict function value at a point."""
        if self.params is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        x = np.asarray(x)
        x_norm = self.scaler_X.transform(x.reshape(1, -1)).squeeze()
        x_jax = jnp.array(x_norm)
        
        pred_norm = self._forward(self.params, x_jax)
        pred = self.scaler_y.inverse_transform(pred_norm.reshape(1, -1)).squeeze()
        return float(pred)

    def gradient(self, x: Array) -> Array:
        """Compute gradient using automatic differentiation."""
        if self.params is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        x = np.asarray(x)
        x_norm = self.scaler_X.transform(x.reshape(1, -1)).squeeze()
        x_jax = jnp.array(x_norm)
        
        # Gradient of normalized prediction w.r.t. normalized input
        grad_fn = grad(lambda x: self._forward(self.params, x))
        grad_norm = grad_fn(x_jax)
        
        # Transform gradient back to original scale
        grad_original = grad_norm * (self.scaler_y.scale_ / self.scaler_X.scale_)
        return np.array(grad_original)


class GPSurrogate(Surrogate):
    """Gaussian Process surrogate model with analytical gradients."""

    def __init__(
        self,
        kernel: str = "rbf",
        length_scale: float = 1.0,
        nu: float = 1.5,
        noise_level: float = 1e-2,
        normalize_y: bool = True,
    ):
        """Initialize Gaussian Process surrogate.
        
        Args:
            kernel: Kernel type ("rbf", "matern")
            length_scale: Kernel length scale parameter
            nu: Matern kernel nu parameter
            noise_level: Noise level for numerical stability
            normalize_y: Whether to normalize target values
        """
        self.kernel_type = kernel
        self.length_scale = length_scale
        self.nu = nu
        self.noise_level = noise_level
        self.normalize_y = normalize_y
        
        # Initialize kernel
        if kernel == "rbf":
            kernel_obj = RBF(length_scale=length_scale)
        elif kernel == "matern":
            kernel_obj = Matern(length_scale=length_scale, nu=nu)
        else:
            raise ValueError(f"Unknown kernel: {kernel}")
            
        self.gp = GaussianProcessRegressor(
            kernel=kernel_obj,
            alpha=noise_level,
            normalize_y=normalize_y,
            random_state=42
        )
        
        self.X_train = None
        self.y_train = None

    def fit(self, X: Array, y: Array) -> None:
        """Fit Gaussian Process to training data."""
        X = np.asarray(X)
        y = np.asarray(y)
        
        self.X_train = X
        self.y_train = y
        self.gp.fit(X, y)

    def predict(self, x: Array) -> float:
        """Predict function value at a point."""
        if self.X_train is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        x = np.asarray(x).reshape(1, -1)
        pred, _ = self.gp.predict(x, return_std=True)
        return float(pred[0])

    def uncertainty(self, x: Array) -> float:
        """Compute prediction uncertainty."""
        if self.X_train is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        x = np.asarray(x).reshape(1, -1)
        _, std = self.gp.predict(x, return_std=True)
        return float(std[0])

    def gradient(self, x: Array) -> Array:
        """Compute gradient using GP predictive mean."""
        if self.X_train is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        x = np.asarray(x)
        eps = 1e-6
        grad = np.zeros_like(x)
        
        # Finite difference approximation of GP mean
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            
            y_plus = self.predict(x_plus)
            y_minus = self.predict(x_minus)
            
            grad[i] = (y_plus - y_minus) / (2 * eps)
            
        return grad


class RandomForestSurrogate(Surrogate):
    """Random Forest surrogate with smoothed gradients."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        smooth_predictions: bool = True,
        random_state: int = 42,
    ):
        """Initialize Random Forest surrogate.
        
        Args:
            n_estimators: Number of trees
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples to split
            min_samples_leaf: Minimum samples in leaf
            smooth_predictions: Enable prediction smoothing for gradients
            random_state: Random seed
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.smooth_predictions = smooth_predictions
        self.random_state = random_state
        
        self.rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
        
        self.scaler = StandardScaler()
        self.X_train = None
        self.y_train = None

    def fit(self, X: Array, y: Array) -> None:
        """Fit Random Forest to training data."""
        X = np.asarray(X)
        y = np.asarray(y)
        
        self.X_train = self.scaler.fit_transform(X)
        self.y_train = y
        self.rf.fit(self.X_train, y)

    def predict(self, x: Array) -> float:
        """Predict function value at a point."""
        if self.X_train is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        x = np.asarray(x).reshape(1, -1)
        x_scaled = self.scaler.transform(x)
        pred = self.rf.predict(x_scaled)
        return float(pred[0])

    def gradient(self, x: Array) -> Array:
        """Compute gradient using finite differences."""
        if self.X_train is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        x = np.asarray(x)
        eps = 1e-6
        grad = np.zeros_like(x)
        
        # Finite difference approximation
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            
            y_plus = self.predict(x_plus)
            y_minus = self.predict(x_minus)
            
            grad[i] = (y_plus - y_minus) / (2 * eps)
            
        return grad

    def smooth_gradient(self, x: Array, bandwidth: float = 0.1) -> Array:
        """Compute smoothed gradient using local averaging."""
        if not self.smooth_predictions:
            return self.gradient(x)
            
        # For now, return standard gradient - can implement local smoothing later
        return self.gradient(x)


class HybridSurrogate(Surrogate):
    """Ensemble of multiple surrogate models."""

    def __init__(
        self,
        models: List[Tuple[str, Surrogate]],
        aggregation: str = "weighted_average",
        weight_optimization: str = "uniform",
    ):
        """Initialize hybrid surrogate model.
        
        Args:
            models: List of (name, model) tuples
            aggregation: How to combine predictions ("weighted_average", "stacking")
            weight_optimization: How to optimize weights ("uniform", "cv")
        """
        self.models = models
        self.aggregation = aggregation
        self.weight_optimization = weight_optimization
        self.weights = None
        self.fitted = False

    def fit(self, X: Array, y: Array) -> None:
        """Fit all surrogate models."""
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Fit each model
        for name, model in self.models:
            print(f"Fitting {name}...")
            model.fit(X, y)
            
        # Initialize uniform weights
        n_models = len(self.models)
        self.weights = np.ones(n_models) / n_models
        
        # Could implement cross-validation weight optimization here
        self.fitted = True

    def predict(self, x: Array) -> float:
        """Predict using weighted ensemble."""
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
            
        predictions = []
        for _, model in self.models:
            pred = model.predict(x)
            predictions.append(pred)
            
        predictions = np.array(predictions)
        return float(np.sum(self.weights * predictions))

    def gradient(self, x: Array) -> Array:
        """Compute weighted ensemble gradient."""
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
            
        gradients = []
        for _, model in self.models:
            grad = model.gradient(x)
            gradients.append(grad)
            
        gradients = np.array(gradients)
        weighted_grad = np.sum(self.weights.reshape(-1, 1) * gradients, axis=0)
        return weighted_grad

    def uncertainty(self, x: Array) -> float:
        """Compute ensemble uncertainty."""
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
            
        predictions = []
        for _, model in self.models:
            pred = model.predict(x)
            predictions.append(pred)
            
        predictions = np.array(predictions)
        return float(np.std(predictions))