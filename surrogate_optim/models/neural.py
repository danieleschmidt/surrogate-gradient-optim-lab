"""Neural network surrogate model implementation."""

from typing import Callable, List, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array, grad, jit, random, vmap

from .base import Dataset, Surrogate


class NeuralSurrogate(Surrogate):
    """Neural network surrogate model with automatic differentiation.
    
    Uses JAX for automatic differentiation to compute exact gradients
    of the learned surrogate function.
    """
    
    def __init__(
        self,
        hidden_dims: List[int] = [64, 64],
        activation: str = "relu",
        learning_rate: float = 0.001,
        n_epochs: int = 1000,
        batch_size: int = 32,
        dropout_rate: float = 0.1,
        ensemble_size: int = 1,
        random_seed: int = 42,
    ):
        """Initialize neural surrogate model.
        
        Args:
            hidden_dims: Hidden layer dimensions
            activation: Activation function ('relu', 'tanh', 'gelu')
            learning_rate: Learning rate for Adam optimizer
            n_epochs: Number of training epochs
            batch_size: Batch size for training
            dropout_rate: Dropout rate for regularization
            ensemble_size: Number of models in ensemble for uncertainty
            random_seed: Random seed for reproducibility
        """
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.ensemble_size = ensemble_size
        self.random_seed = random_seed
        
        # Model parameters (set during training)
        self.params = None
        self.input_dim = None
        self.output_dim = 1
        
        # Set activation function
        self._activation_fn = self._get_activation_fn(activation)
        
        # Compiled functions for efficiency
        self._predict_fn = None
        self._gradient_fn = None
    
    def _get_activation_fn(self, activation: str) -> Callable:
        """Get activation function by name."""
        activations = {
            "relu": jax.nn.relu,
            "tanh": jax.nn.tanh,
            "gelu": jax.nn.gelu,
            "sigmoid": jax.nn.sigmoid,
            "swish": jax.nn.swish,
        }
        if activation not in activations:
            raise ValueError(f"Unknown activation: {activation}. Choose from {list(activations.keys())}")
        return activations[activation]
    
    def _init_network(self, key: Array, input_dim: int) -> dict:
        """Initialize network parameters."""
        dims = [input_dim] + self.hidden_dims + [self.output_dim]
        params = {}
        
        for i in range(len(dims) - 1):
            key, subkey = random.split(key)
            # Xavier initialization
            scale = jnp.sqrt(2.0 / (dims[i] + dims[i + 1]))
            params[f'W{i}'] = random.normal(subkey, (dims[i], dims[i + 1])) * scale
            params[f'b{i}'] = jnp.zeros(dims[i + 1])
        
        return params
    
    def _forward(self, params: dict, x: Array, training: bool = False) -> Array:
        """Forward pass through the network."""
        # Ensure x is 2D
        if x.ndim == 1:
            x = x[None, :]
        
        h = x
        n_layers = len(self.hidden_dims) + 1
        
        for i in range(n_layers - 1):
            h = h @ params[f'W{i}'] + params[f'b{i}']
            h = self._activation_fn(h)
            
            # Apply dropout during training (simplified - not using proper JAX dropout)
            if training and self.dropout_rate > 0:
                h = h * (1 - self.dropout_rate)
        
        # Output layer (no activation)
        output = h @ params[f'W{n_layers-1}'] + params[f'b{n_layers-1}']
        
        # Return scalar for single input, otherwise return vector
        if output.shape[0] == 1:
            return output[0, 0]
        return output.squeeze(-1)
    
    def _loss_fn(self, params: dict, batch_x: Array, batch_y: Array) -> float:
        """Compute loss for a batch."""
        predictions = vmap(lambda x: self._forward(params, x, training=True))(batch_x)
        mse_loss = jnp.mean((predictions - batch_y) ** 2)
        
        # L2 regularization
        l2_reg = 0.001 * sum(jnp.sum(params[key] ** 2) for key in params if key.startswith('W'))
        
        return mse_loss + l2_reg
    
    def fit(self, dataset: Dataset) -> "NeuralSurrogate":
        """Train the neural network surrogate."""
        self.input_dim = dataset.n_dims
        key = random.PRNGKey(self.random_seed)
        
        # Initialize parameters
        self.params = self._init_network(key, self.input_dim)
        
        # Create optimizer state (simple momentum)
        optimizer_state = {key: jnp.zeros_like(param) for key, param in self.params.items()}
        momentum = 0.9
        
        # Training loop
        n_batches = max(1, dataset.n_samples // self.batch_size)
        
        for epoch in range(self.n_epochs):
            key, subkey = random.split(key)
            
            # Shuffle data
            perm = random.permutation(subkey, dataset.n_samples)
            X_shuffled = dataset.X[perm]
            y_shuffled = dataset.y[perm]
            
            epoch_loss = 0.0
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, dataset.n_samples)
                
                batch_x = X_shuffled[start_idx:end_idx]
                batch_y = y_shuffled[start_idx:end_idx]
                
                # Compute gradients
                loss_value, grads = jax.value_and_grad(self._loss_fn)(self.params, batch_x, batch_y)
                epoch_loss += loss_value
                
                # Update parameters with momentum
                for key in self.params:
                    optimizer_state[key] = momentum * optimizer_state[key] - self.learning_rate * grads[key]
                    self.params[key] += optimizer_state[key]
            
            # Print progress occasionally
            if epoch % (self.n_epochs // 10) == 0 or epoch == self.n_epochs - 1:
                avg_loss = epoch_loss / n_batches
                print(f"Epoch {epoch:4d}/{self.n_epochs}: Loss = {avg_loss:.6f}")
        
        # Compile prediction and gradient functions for efficiency
        self._predict_fn = jit(lambda x: self._forward(self.params, x))
        self._gradient_fn = jit(grad(lambda x: self._forward(self.params, x)))
        
        return self
    
    def predict(self, x: Array) -> Array:
        """Predict function values."""
        if self.params is None:
            raise ValueError("Model must be trained before prediction")
        
        if self._predict_fn is None:
            self._predict_fn = jit(lambda x: self._forward(self.params, x))
        
        # Handle both single points and batches
        if x.ndim == 1:
            return self._predict_fn(x)
        else:
            return vmap(self._predict_fn)(x)
    
    def gradient(self, x: Array) -> Array:
        """Compute gradients using automatic differentiation."""
        if self.params is None:
            raise ValueError("Model must be trained before gradient computation")
        
        if self._gradient_fn is None:
            self._gradient_fn = jit(grad(lambda x: self._forward(self.params, x)))
        
        # Handle both single points and batches
        if x.ndim == 1:
            return self._gradient_fn(x)
        else:
            return vmap(self._gradient_fn)(x)
    
    def uncertainty(self, x: Array) -> Array:
        """Estimate uncertainty using ensemble variance (simplified)."""
        if self.ensemble_size == 1:
            # For single model, return zero uncertainty
            if x.ndim == 1:
                return jnp.array(0.0)
            return jnp.zeros(x.shape[0])
        
        # For ensemble models, this would compute variance across ensemble
        # For now, return placeholder implementation
        if x.ndim == 1:
            return jnp.array(0.1)  # Placeholder uncertainty
        return jnp.full(x.shape[0], 0.1)  # Placeholder uncertainty