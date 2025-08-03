"""Data preprocessing utilities for surrogate models."""

from typing import Optional, Tuple

import jax.numpy as jnp
from jax import Array

from ..models.base import Dataset


def normalize_data(
    X: Array,
    bounds: Optional[Tuple[Array, Array]] = None,
    feature_range: Tuple[float, float] = (0, 1),
) -> Tuple[Array, Tuple[Array, Array]]:
    """Normalize data to specified feature range.
    
    Args:
        X: Input data to normalize
        bounds: Optional (min, max) bounds, computed from data if None
        feature_range: Target range for normalization
        
    Returns:
        Tuple of (normalized_data, (min_bounds, max_bounds))
    """
    if bounds is None:
        x_min = jnp.min(X, axis=0)
        x_max = jnp.max(X, axis=0)
    else:
        x_min, x_max = bounds
    
    # Avoid division by zero
    x_range = x_max - x_min
    x_range = jnp.where(x_range == 0, 1.0, x_range)
    
    # Normalize to [0, 1]
    X_unit = (X - x_min) / x_range
    
    # Scale to target range
    target_min, target_max = feature_range
    X_normalized = target_min + (target_max - target_min) * X_unit
    
    return X_normalized, (x_min, x_max)


def denormalize_data(
    X_normalized: Array,
    bounds: Tuple[Array, Array],
    feature_range: Tuple[float, float] = (0, 1),
) -> Array:
    """Denormalize data back to original scale.
    
    Args:
        X_normalized: Normalized data
        bounds: (min_bounds, max_bounds) from normalization
        feature_range: Feature range used in normalization
        
    Returns:
        Denormalized data
    """
    x_min, x_max = bounds
    target_min, target_max = feature_range
    
    # Scale back to [0, 1]
    X_unit = (X_normalized - target_min) / (target_max - target_min)
    
    # Scale back to original range
    X_original = x_min + (x_max - x_min) * X_unit
    
    return X_original


def standardize_data(
    X: Array,
    stats: Optional[Tuple[Array, Array]] = None,
) -> Tuple[Array, Tuple[Array, Array]]:
    """Standardize data to zero mean and unit variance.
    
    Args:
        X: Input data to standardize
        stats: Optional (mean, std) statistics, computed from data if None
        
    Returns:
        Tuple of (standardized_data, (mean, std))
    """
    if stats is None:
        mean = jnp.mean(X, axis=0)
        std = jnp.std(X, axis=0)
    else:
        mean, std = stats
    
    # Avoid division by zero
    std = jnp.where(std == 0, 1.0, std)
    
    X_standardized = (X - mean) / std
    
    return X_standardized, (mean, std)


def destandardize_data(
    X_standardized: Array,
    stats: Tuple[Array, Array],
) -> Array:
    """Destandardize data back to original scale.
    
    Args:
        X_standardized: Standardized data
        stats: (mean, std) from standardization
        
    Returns:
        Destandardized data
    """
    mean, std = stats
    X_original = X_standardized * std + mean
    return X_original


def preprocess_dataset(
    dataset: Dataset,
    normalize_inputs: bool = True,
    standardize_outputs: bool = True,
    feature_range: Tuple[float, float] = (0, 1),
) -> Tuple[Dataset, dict]:
    """Preprocess entire dataset with inputs and outputs.
    
    Args:
        dataset: Dataset to preprocess
        normalize_inputs: Whether to normalize input features
        standardize_outputs: Whether to standardize output values
        feature_range: Target range for input normalization
        
    Returns:
        Tuple of (preprocessed_dataset, preprocessing_info)
    """
    preprocessing_info = {}
    
    # Process inputs
    X_processed = dataset.X
    if normalize_inputs:
        X_processed, input_bounds = normalize_data(dataset.X, feature_range=feature_range)
        preprocessing_info['input_normalization'] = {
            'bounds': input_bounds,
            'feature_range': feature_range,
        }
    
    # Process outputs
    y_processed = dataset.y
    if standardize_outputs:
        y_processed, output_stats = standardize_data(dataset.y.reshape(-1, 1))
        y_processed = y_processed.flatten()
        preprocessing_info['output_standardization'] = {
            'stats': output_stats,
        }
    
    # Process gradients if available
    gradients_processed = dataset.gradients
    if dataset.gradients is not None and normalize_inputs:
        # Gradients need to be adjusted for input normalization
        # grad_normalized = grad_original * (x_max - x_min) / (target_max - target_min)
        x_min, x_max = input_bounds
        target_min, target_max = feature_range
        
        scale_factor = (x_max - x_min) / (target_max - target_min)
        gradients_processed = dataset.gradients * scale_factor[None, :]
        
        if standardize_outputs:
            # Also adjust for output standardization
            _, output_std = output_stats
            gradients_processed = gradients_processed * output_std
    
    # Create preprocessed dataset
    preprocessed_dataset = Dataset(
        X=X_processed,
        y=y_processed,
        gradients=gradients_processed,
        metadata={
            **dataset.metadata,
            'preprocessing_applied': True,
            'normalize_inputs': normalize_inputs,
            'standardize_outputs': standardize_outputs,
        }
    )
    
    return preprocessed_dataset, preprocessing_info


def reverse_preprocess_prediction(
    prediction: Array,
    preprocessing_info: dict,
) -> Array:
    """Reverse preprocessing for predictions.
    
    Args:
        prediction: Preprocessed prediction
        preprocessing_info: Preprocessing information from preprocess_dataset
        
    Returns:
        Original scale prediction
    """
    if 'output_standardization' in preprocessing_info:
        stats = preprocessing_info['output_standardization']['stats']
        prediction = destandardize_data(prediction.reshape(-1, 1), stats).flatten()
    
    return prediction


def reverse_preprocess_gradient(
    gradient: Array,
    preprocessing_info: dict,
) -> Array:
    """Reverse preprocessing for gradients.
    
    Args:
        gradient: Preprocessed gradient
        preprocessing_info: Preprocessing information from preprocess_dataset
        
    Returns:
        Original scale gradient
    """
    result = gradient
    
    # Reverse output standardization effect
    if 'output_standardization' in preprocessing_info:
        _, output_std = preprocessing_info['output_standardization']['stats']
        result = result / output_std
    
    # Reverse input normalization effect
    if 'input_normalization' in preprocessing_info:
        x_min, x_max = preprocessing_info['input_normalization']['bounds']
        target_min, target_max = preprocessing_info['input_normalization']['feature_range']
        
        scale_factor = (x_max - x_min) / (target_max - target_min)
        result = result / scale_factor
    
    return result


def remove_outliers(
    dataset: Dataset,
    method: str = "iqr",
    threshold: float = 1.5,
) -> Dataset:
    """Remove outliers from dataset.
    
    Args:
        dataset: Input dataset
        method: Outlier detection method ('iqr', 'zscore')
        threshold: Threshold for outlier detection
        
    Returns:
        Dataset with outliers removed
    """
    if method == "iqr":
        # Interquartile range method
        q1 = jnp.percentile(dataset.y, 25)
        q3 = jnp.percentile(dataset.y, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        
        mask = (dataset.y >= lower_bound) & (dataset.y <= upper_bound)
        
    elif method == "zscore":
        # Z-score method
        mean = jnp.mean(dataset.y)
        std = jnp.std(dataset.y)
        
        z_scores = jnp.abs((dataset.y - mean) / std)
        mask = z_scores <= threshold
        
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")
    
    # Filter dataset
    filtered_dataset = Dataset(
        X=dataset.X[mask],
        y=dataset.y[mask],
        gradients=dataset.gradients[mask] if dataset.gradients is not None else None,
        metadata={
            **dataset.metadata,
            'outliers_removed': True,
            'outlier_method': method,
            'outlier_threshold': threshold,
            'original_samples': dataset.n_samples,
            'removed_samples': int(jnp.sum(~mask)),
        }
    )
    
    return filtered_dataset