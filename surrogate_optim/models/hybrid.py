"""Hybrid surrogate model combining multiple model types."""

from typing import Dict, List, Tuple

import jax.numpy as jnp
from jax import Array

from .base import Dataset, Surrogate


class HybridSurrogate(Surrogate):
    """Hybrid surrogate model that combines multiple surrogate types.
    
    Aggregates predictions and gradients from multiple surrogate models
    using weighted averaging, stacking, or voting strategies.
    """
    
    def __init__(
        self,
        models: List[Tuple[str, Surrogate]],
        aggregation: str = "weighted_average",
        weight_optimization: str = "equal",
        validation_split: float = 0.2,
    ):
        """Initialize hybrid surrogate model.
        
        Args:
            models: List of (name, surrogate_model) tuples
            aggregation: Aggregation method ('weighted_average', 'stacking', 'voting')
            weight_optimization: How to optimize weights ('equal', 'cv', 'performance')
            validation_split: Fraction of data for validation-based weight optimization
        """
        self.models = models
        self.aggregation = aggregation
        self.weight_optimization = weight_optimization
        self.validation_split = validation_split
        
        # Model state
        self.weights = None
        self.is_fitted = False
        self.input_dim = None
        
        # Validation for model types
        if len(models) == 0:
            raise ValueError("At least one model must be provided")
        
        for name, model in models:
            if not isinstance(model, Surrogate):
                raise ValueError(f"Model {name} must be a Surrogate instance")
    
    def fit(self, dataset: Dataset) -> "HybridSurrogate":
        """Train all component models and optimize weights."""
        self.input_dim = dataset.n_dims
        
        print(f"Training hybrid model with {len(self.models)} components...")
        
        # Split data for validation if needed
        if self.weight_optimization in ["cv", "performance"]:
            split_idx = int(dataset.n_samples * (1 - self.validation_split))
            train_dataset = Dataset(
                X=dataset.X[:split_idx],
                y=dataset.y[:split_idx],
                gradients=dataset.gradients[:split_idx] if dataset.gradients is not None else None,
                metadata=dataset.metadata
            )
            val_dataset = Dataset(
                X=dataset.X[split_idx:],
                y=dataset.y[split_idx:],
                gradients=dataset.gradients[split_idx:] if dataset.gradients is not None else None,
                metadata=dataset.metadata
            )
        else:
            train_dataset = dataset
            val_dataset = None
        
        # Train each component model
        for name, model in self.models:
            print(f"  Training {name}...")
            model.fit(train_dataset)
        
        # Optimize weights
        self.weights = self._optimize_weights(val_dataset if val_dataset else train_dataset)
        self.is_fitted = True
        
        print(f"Hybrid model trained. Weights: {dict(zip([name for name, _ in self.models], self.weights))}")
        
        return self
    
    def _optimize_weights(self, validation_dataset: Dataset) -> Array:
        """Optimize model weights based on validation performance."""
        n_models = len(self.models)
        
        if self.weight_optimization == "equal":
            return jnp.ones(n_models) / n_models
        
        elif self.weight_optimization == "performance":
            # Weight based on individual model performance
            errors = []
            
            for name, model in self.models:
                predictions = model.predict(validation_dataset.X)
                mse = jnp.mean((predictions - validation_dataset.y) ** 2)
                errors.append(mse)
            
            # Convert errors to weights (inverse of error)
            errors = jnp.array(errors)
            weights = 1.0 / (errors + 1e-8)  # Add small epsilon to avoid division by zero
            weights = weights / jnp.sum(weights)  # Normalize
            
            return weights
        
        elif self.weight_optimization == "cv":
            # Cross-validation based weight optimization (simplified)
            from sklearn.model_selection import cross_val_score
            
            # For simplicity, use performance-based weighting
            # In a full implementation, this would use proper CV
            return self._optimize_weights_cv(validation_dataset)
        
        else:
            raise ValueError(f"Unknown weight optimization method: {self.weight_optimization}")
    
    def _optimize_weights_cv(self, validation_dataset: Dataset) -> Array:
        """Optimize weights using cross-validation (simplified implementation)."""
        # Simplified CV - just use validation performance for now
        return self._optimize_weights_performance(validation_dataset)
    
    def _optimize_weights_performance(self, validation_dataset: Dataset) -> Array:
        """Optimize weights based on validation performance."""
        n_models = len(self.models)
        errors = []
        
        for name, model in self.models:
            try:
                predictions = model.predict(validation_dataset.X)
                mse = jnp.mean((predictions - validation_dataset.y) ** 2)
                errors.append(float(mse))
            except Exception as e:
                print(f"Error evaluating model {name}: {e}")
                errors.append(float('inf'))  # Give infinite error to failed models
        
        # Convert errors to weights (inverse of error)
        errors = jnp.array(errors)
        
        # Handle infinite errors
        finite_mask = jnp.isfinite(errors)
        if not jnp.any(finite_mask):
            # All models failed, use equal weights
            return jnp.ones(n_models) / n_models
        
        # Set infinite errors to max finite error
        max_finite_error = jnp.max(errors[finite_mask])
        errors = jnp.where(finite_mask, errors, max_finite_error * 10)
        
        # Compute inverse weights
        weights = 1.0 / (errors + 1e-8)
        weights = weights / jnp.sum(weights)
        
        return weights
    
    def predict(self, x: Array) -> Array:
        """Predict using weighted combination of all models."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        
        if self.aggregation == "weighted_average":
            return self._weighted_average_predict(x)
        elif self.aggregation == "stacking":
            return self._stacking_predict(x)
        elif self.aggregation == "voting":
            return self._voting_predict(x)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")
    
    def _weighted_average_predict(self, x: Array) -> Array:
        """Predict using weighted average of all models."""
        predictions = []
        
        for (name, model), weight in zip(self.models, self.weights):
            try:
                pred = model.predict(x)
                predictions.append(weight * pred)
            except Exception as e:
                print(f"Warning: Model {name} failed during prediction: {e}")
                # Skip failed models
                continue
        
        if not predictions:
            raise RuntimeError("All models failed during prediction")
        
        return jnp.sum(jnp.stack(predictions), axis=0)
    
    def _stacking_predict(self, x: Array) -> Array:
        """Predict using stacking (meta-learning) approach."""
        # Simplified stacking - for now just use weighted average
        # In full implementation, this would train a meta-model
        return self._weighted_average_predict(x)
    
    def _voting_predict(self, x: Array) -> Array:
        """Predict using voting (median) of all models."""
        predictions = []
        
        for name, model in self.models:
            try:
                pred = model.predict(x)
                predictions.append(pred)
            except Exception as e:
                print(f"Warning: Model {name} failed during prediction: {e}")
                continue
        
        if not predictions:
            raise RuntimeError("All models failed during prediction")
        
        # Use median as robust voting strategy
        return jnp.median(jnp.stack(predictions), axis=0)
    
    def gradient(self, x: Array) -> Array:
        """Compute gradients using weighted combination of all models."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before gradient computation")
        
        if self.aggregation == "weighted_average":
            return self._weighted_average_gradient(x)
        elif self.aggregation == "stacking":
            return self._stacking_gradient(x)
        elif self.aggregation == "voting":
            return self._voting_gradient(x)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")
    
    def _weighted_average_gradient(self, x: Array) -> Array:
        """Compute gradients using weighted average of all models."""
        gradients = []
        
        for (name, model), weight in zip(self.models, self.weights):
            try:
                grad = model.gradient(x)
                gradients.append(weight * grad)
            except Exception as e:
                print(f"Warning: Model {name} failed during gradient computation: {e}")
                continue
        
        if not gradients:
            raise RuntimeError("All models failed during gradient computation")
        
        return jnp.sum(jnp.stack(gradients), axis=0)
    
    def _stacking_gradient(self, x: Array) -> Array:
        """Compute gradients using stacking approach."""
        # Simplified - use weighted average for now
        return self._weighted_average_gradient(x)
    
    def _voting_gradient(self, x: Array) -> Array:
        """Compute gradients using voting (median) of all models."""
        gradients = []
        
        for name, model in self.models:
            try:
                grad = model.gradient(x)
                gradients.append(grad)
            except Exception as e:
                print(f"Warning: Model {name} failed during gradient computation: {e}")
                continue
        
        if not gradients:
            raise RuntimeError("All models failed during gradient computation")
        
        # Use median as robust voting strategy
        return jnp.median(jnp.stack(gradients), axis=0)
    
    def uncertainty(self, x: Array) -> Array:
        """Estimate uncertainty by combining uncertainties from all models."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before uncertainty computation")
        
        uncertainties = []
        predictions = []
        
        for name, model in self.models:
            try:
                pred = model.predict(x)
                unc = model.uncertainty(x)
                predictions.append(pred)
                uncertainties.append(unc)
            except Exception as e:
                print(f"Warning: Model {name} failed during uncertainty computation: {e}")
                continue
        
        if not uncertainties:
            # Fallback to zero uncertainty if all models fail
            if x.ndim == 1:
                return jnp.array(0.0)
            return jnp.zeros(x.shape[0])
        
        # Combine uncertainties: sqrt of weighted variance + variance of predictions
        predictions = jnp.stack(predictions)
        uncertainties = jnp.stack(uncertainties)
        
        # Epistemic uncertainty from prediction variance
        pred_variance = jnp.var(predictions, axis=0)
        
        # Aleatoric uncertainty from individual model uncertainties
        weighted_uncertainty = jnp.sqrt(jnp.mean(uncertainties ** 2, axis=0))
        
        # Total uncertainty combines both sources
        total_uncertainty = jnp.sqrt(pred_variance + weighted_uncertainty ** 2)
        
        return total_uncertainty
    
    def get_model_weights(self) -> Dict[str, float]:
        """Get the optimized weights for each model."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before accessing weights")
        
        return dict(zip([name for name, _ in self.models], self.weights))
    
    def get_individual_predictions(self, x: Array) -> Dict[str, Array]:
        """Get predictions from each individual model."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        
        predictions = {}
        for name, model in self.models:
            try:
                predictions[name] = model.predict(x)
            except Exception as e:
                print(f"Warning: Model {name} failed: {e}")
                predictions[name] = None
        
        return predictions