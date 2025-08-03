"""Random Forest surrogate model with gradient estimation."""

from typing import Optional

import jax.numpy as jnp
from jax import Array
from sklearn.ensemble import RandomForestRegressor

from .base import Dataset, Surrogate


class RandomForestSurrogate(Surrogate):
    """Random Forest surrogate model with smoothed gradient estimation.
    
    Uses Random Forest for function approximation and finite differences
    with smoothing for gradient computation.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int = 42,
        smooth_predictions: bool = True,
        smoothing_bandwidth: float = 0.1,
        n_jobs: int = -1,
    ):
        """Initialize Random Forest surrogate.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees (None for unlimited)
            min_samples_split: Minimum samples required to split an internal node
            min_samples_leaf: Minimum samples required to be at a leaf node
            random_state: Random state for reproducibility
            smooth_predictions: Whether to enable gradient computation via smoothing
            smoothing_bandwidth: Bandwidth for kernel smoothing
            n_jobs: Number of parallel jobs (-1 for all processors)
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.smooth_predictions = smooth_predictions
        self.smoothing_bandwidth = smoothing_bandwidth
        self.n_jobs = n_jobs
        
        # Model components
        self.rf = None
        self.X_train = None
        self.y_train = None
        self.input_dim = None
    
    def fit(self, dataset: Dataset) -> "RandomForestSurrogate":
        """Train the Random Forest model."""
        self.input_dim = dataset.n_dims
        self.X_train = dataset.X
        self.y_train = dataset.y
        
        # Create and train Random Forest
        self.rf = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        
        print("Training Random Forest...")
        self.rf.fit(self.X_train, self.y_train)
        
        # Compute feature importance for analysis
        feature_importance = self.rf.feature_importances_
        print(f"Feature importance: {feature_importance}")
        
        return self
    
    def predict(self, x: Array) -> Array:
        """Predict function values."""
        if self.rf is None:
            raise ValueError("Model must be trained before prediction")
        
        # Convert JAX array to numpy for sklearn
        x_np = jnp.asarray(x)
        
        if x_np.ndim == 1:
            x_np = x_np[None, :]
            single_point = True
        else:
            single_point = False
        
        # Use Random Forest for prediction
        predictions = self.rf.predict(x_np)
        
        if single_point:
            return jnp.array(predictions[0])
        return jnp.array(predictions)
    
    def gradient(self, x: Array) -> Array:
        """Compute gradients using finite differences with optional smoothing."""
        if self.rf is None:
            raise ValueError("Model must be trained before gradient computation")
        
        if self.smooth_predictions:
            return self._smooth_gradient(x)
        else:
            return self._finite_difference_gradient(x)
    
    def _finite_difference_gradient(self, x: Array, eps: float = 1e-6) -> Array:
        """Compute gradients using standard finite differences."""
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
    
    def _smooth_gradient(self, x: Array) -> Array:
        """Compute gradients using kernel-smoothed finite differences.
        
        This method uses a local weighted regression approach to smooth
        the Random Forest predictions before computing gradients.
        """
        if x.ndim == 1:
            x = x[None, :]
            single_point = True
        else:
            single_point = False
        
        gradients = jnp.zeros_like(x)
        eps = 1e-5
        
        for i in range(x.shape[1]):
            # Create perturbation points around x
            x_plus = x.at[:, i].add(eps)
            x_minus = x.at[:, i].add(-eps)
            
            # Get predictions with local smoothing
            y_plus = self._local_weighted_prediction(x_plus)
            y_minus = self._local_weighted_prediction(x_minus)
            
            gradients = gradients.at[:, i].set((y_plus - y_minus) / (2 * eps))
        
        if single_point:
            return gradients[0]
        return gradients
    
    def _local_weighted_prediction(self, x: Array) -> Array:
        """Make predictions using local weighted regression for smoothing."""
        if x.ndim == 1:
            x = x[None, :]
        
        predictions = jnp.zeros(x.shape[0])
        
        for i, query_point in enumerate(x):
            # Compute distances to training points
            distances = jnp.linalg.norm(self.X_train - query_point, axis=1)
            
            # Compute weights using Gaussian kernel
            weights = jnp.exp(-0.5 * (distances / self.smoothing_bandwidth) ** 2)
            weights = weights / jnp.sum(weights)  # Normalize weights
            
            # Weighted prediction
            weighted_prediction = jnp.sum(weights * self.y_train)
            predictions = predictions.at[i].set(weighted_prediction)
        
        if predictions.shape[0] == 1:
            return predictions[0]
        return predictions
    
    def uncertainty(self, x: Array) -> Array:
        """Estimate uncertainty using prediction variance across trees."""
        if self.rf is None:
            raise ValueError("Model must be trained before uncertainty computation")
        
        # Convert JAX array to numpy for sklearn
        x_np = jnp.asarray(x)
        
        if x_np.ndim == 1:
            x_np = x_np[None, :]
            single_point = True
        else:
            single_point = False
        
        # Get predictions from individual trees
        tree_predictions = jnp.array([
            tree.predict(x_np) for tree in self.rf.estimators_
        ])
        
        # Compute variance across trees as uncertainty measure
        uncertainty = jnp.std(tree_predictions, axis=0)
        
        if single_point:
            return uncertainty[0]
        return uncertainty
    
    def feature_importance(self) -> Array:
        """Get feature importance from the trained model."""
        if self.rf is None:
            raise ValueError("Model must be trained before accessing feature importance")
        
        return jnp.array(self.rf.feature_importances_)
    
    def predict_with_uncertainty(self, x: Array) -> tuple[Array, Array]:
        """Predict with uncertainty in a single call for efficiency."""
        if self.rf is None:
            raise ValueError("Model must be trained before prediction")
        
        # Convert JAX array to numpy for sklearn
        x_np = jnp.asarray(x)
        
        if x_np.ndim == 1:
            x_np = x_np[None, :]
            single_point = True
        else:
            single_point = False
        
        # Get mean prediction
        predictions = jnp.array(self.rf.predict(x_np))
        
        # Get predictions from individual trees for uncertainty
        tree_predictions = jnp.array([
            tree.predict(x_np) for tree in self.rf.estimators_
        ])
        
        # Compute uncertainty as standard deviation across trees
        uncertainty = jnp.std(tree_predictions, axis=0)
        
        if single_point:
            return predictions[0], uncertainty[0]
        return predictions, uncertainty