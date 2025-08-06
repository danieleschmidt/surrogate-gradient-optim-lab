"""Model validation and performance assessment."""

import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax.numpy as jnp
from jax import Array

from ..models.base import Dataset, Surrogate
from .input_validation import ValidationError, ValidationWarning


def validate_surrogate_performance(
    surrogate: Surrogate,
    test_dataset: Dataset,
    performance_thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Validate surrogate model performance on test data.
    
    Args:
        surrogate: Trained surrogate model
        test_dataset: Test dataset for validation
        performance_thresholds: Optional performance thresholds
        
    Returns:
        Performance metrics and validation results
        
    Raises:
        ValidationError: If surrogate performance is unacceptable
    """
    if performance_thresholds is None:
        performance_thresholds = {
            "min_r2": 0.7,
            "max_mse": None,  # Will be set relative to data variance
            "max_gradient_error": 1.0,
        }
    
    # Get predictions
    try:
        predictions = surrogate.predict(test_dataset.X)
    except Exception as e:
        raise ValidationError(f"Surrogate prediction failed: {e}")
    
    if not jnp.all(jnp.isfinite(predictions)):
        nan_count = jnp.sum(~jnp.isfinite(predictions))
        raise ValidationError(f"Surrogate produced {nan_count} non-finite predictions")
    
    # Compute basic metrics
    y_true = test_dataset.y
    y_pred = predictions
    
    # Mean Squared Error
    mse = float(jnp.mean((y_true - y_pred) ** 2))
    
    # Mean Absolute Error  
    mae = float(jnp.mean(jnp.abs(y_true - y_pred)))
    
    # R² Score
    ss_res = jnp.sum((y_true - y_pred) ** 2)
    ss_tot = jnp.sum((y_true - jnp.mean(y_true)) ** 2)
    r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 1e-12 else 0.0
    
    # Root Mean Squared Error
    rmse = float(jnp.sqrt(mse))
    
    # Normalized metrics
    y_std = float(jnp.std(y_true))
    normalized_rmse = rmse / y_std if y_std > 1e-12 else float('inf')
    
    metrics = {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "normalized_rmse": normalized_rmse,
    }
    
    # Gradient validation if gradients available
    if test_dataset.gradients is not None:
        try:
            pred_gradients = surrogate.gradient(test_dataset.X)
            
            if not jnp.all(jnp.isfinite(pred_gradients)):
                warnings.warn(
                    "Surrogate produced non-finite gradients",
                    ValidationWarning
                )
                gradient_error = float('inf')
            else:
                gradient_error = float(jnp.mean(jnp.linalg.norm(
                    pred_gradients - test_dataset.gradients, axis=1
                )))
            
            metrics["gradient_error"] = gradient_error
            
        except Exception as e:
            warnings.warn(f"Gradient validation failed: {e}", ValidationWarning)
            metrics["gradient_error"] = float('inf')
    
    # Uncertainty calibration if available
    try:
        uncertainties = surrogate.uncertainty(test_dataset.X)
        
        if jnp.all(jnp.isfinite(uncertainties)):
            # Check if uncertainties correlate with errors
            errors = jnp.abs(y_true - y_pred)
            correlation = float(jnp.corrcoef(uncertainties, errors)[0, 1])
            metrics["uncertainty_correlation"] = correlation
            
            # Coverage analysis (simplified)
            std_errors = errors / jnp.std(errors)
            coverage_68 = float(jnp.mean(std_errors <= 1.0))  # ~68% should be within 1 std
            coverage_95 = float(jnp.mean(std_errors <= 2.0))  # ~95% should be within 2 std
            
            metrics["coverage_68"] = coverage_68
            metrics["coverage_95"] = coverage_95
        
    except Exception as e:
        warnings.warn(f"Uncertainty validation failed: {e}", ValidationWarning)
    
    # Validate against thresholds
    validation_results = {
        "passed": True,
        "warnings": [],
        "errors": [],
    }
    
    # R² threshold
    if "min_r2" in performance_thresholds:
        min_r2 = performance_thresholds["min_r2"]
        if r2 < min_r2:
            validation_results["errors"].append(
                f"R² score {r2:.4f} is below threshold {min_r2}"
            )
            validation_results["passed"] = False
    
    # MSE threshold (relative to data variance)
    if performance_thresholds.get("max_mse") is None:
        max_mse = y_std ** 2 * 0.5  # MSE should be < 50% of data variance
    else:
        max_mse = performance_thresholds["max_mse"]
    
    if mse > max_mse:
        validation_results["warnings"].append(
            f"MSE {mse:.6f} is high (threshold: {max_mse:.6f})"
        )
    
    # Gradient error threshold
    if "gradient_error" in metrics and "max_gradient_error" in performance_thresholds:
        max_grad_error = performance_thresholds["max_gradient_error"]
        if metrics["gradient_error"] > max_grad_error:
            validation_results["warnings"].append(
                f"Gradient error {metrics['gradient_error']:.4f} "
                f"exceeds threshold {max_grad_error}"
            )
    
    # Additional checks
    if normalized_rmse > 1.0:
        validation_results["warnings"].append(
            f"Normalized RMSE {normalized_rmse:.4f} > 1.0 indicates poor fit"
        )
    
    if r2 < 0:
        validation_results["errors"].append(
            f"Negative R² score {r2:.4f} indicates very poor fit"
        )
        validation_results["passed"] = False
    
    # Issue warnings and errors
    for warning in validation_results["warnings"]:
        warnings.warn(warning, ValidationWarning)
    
    if not validation_results["passed"]:
        error_msg = "Surrogate validation failed: " + "; ".join(validation_results["errors"])
        raise ValidationError(error_msg)
    
    return {
        "metrics": metrics,
        "validation": validation_results,
        "thresholds": performance_thresholds,
    }


def cross_validate_surrogate(
    surrogate_class: type,
    surrogate_params: Dict[str, Any],
    dataset: Dataset,
    cv_folds: int = 5,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Perform cross-validation on surrogate model.
    
    Args:
        surrogate_class: Surrogate model class
        surrogate_params: Parameters for surrogate model
        dataset: Full dataset for cross-validation
        cv_folds: Number of cross-validation folds
        random_seed: Random seed for fold splitting
        
    Returns:
        Cross-validation results
    """
    from jax import random
    
    if cv_folds < 2:
        raise ValidationError("cv_folds must be at least 2")
    
    if dataset.n_samples < cv_folds:
        raise ValidationError(f"Dataset has {dataset.n_samples} samples but {cv_folds} folds requested")
    
    # Create fold indices
    key = random.PRNGKey(random_seed)
    indices = random.permutation(key, dataset.n_samples)
    fold_size = dataset.n_samples // cv_folds
    
    fold_results = []
    
    for fold in range(cv_folds):
        # Create train/test split for this fold
        start_idx = fold * fold_size
        end_idx = (fold + 1) * fold_size if fold < cv_folds - 1 else dataset.n_samples
        
        test_indices = indices[start_idx:end_idx]
        train_indices = jnp.concatenate([indices[:start_idx], indices[end_idx:]])
        
        # Create train and test datasets
        train_dataset = Dataset(
            X=dataset.X[train_indices],
            y=dataset.y[train_indices],
            gradients=dataset.gradients[train_indices] if dataset.gradients is not None else None,
            metadata=dataset.metadata
        )
        
        test_dataset = Dataset(
            X=dataset.X[test_indices],
            y=dataset.y[test_indices],
            gradients=dataset.gradients[test_indices] if dataset.gradients is not None else None,
            metadata=dataset.metadata
        )
        
        try:
            # Train surrogate on fold
            surrogate = surrogate_class(**surrogate_params)
            surrogate.fit(train_dataset)
            
            # Evaluate on test set
            predictions = surrogate.predict(test_dataset.X)
            
            # Compute metrics
            mse = float(jnp.mean((test_dataset.y - predictions) ** 2))
            mae = float(jnp.mean(jnp.abs(test_dataset.y - predictions)))
            
            ss_res = jnp.sum((test_dataset.y - predictions) ** 2)
            ss_tot = jnp.sum((test_dataset.y - jnp.mean(test_dataset.y)) ** 2)
            r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 1e-12 else 0.0
            
            fold_results.append({
                "fold": fold,
                "mse": mse,
                "mae": mae, 
                "r2": r2,
                "n_train": len(train_indices),
                "n_test": len(test_indices),
            })
            
        except Exception as e:
            warnings.warn(f"Cross-validation fold {fold} failed: {e}", ValidationWarning)
            fold_results.append({
                "fold": fold,
                "mse": float('inf'),
                "mae": float('inf'),
                "r2": -float('inf'),
                "n_train": len(train_indices),
                "n_test": len(test_indices),
                "error": str(e),
            })
    
    # Aggregate results
    successful_folds = [r for r in fold_results if "error" not in r]
    
    if len(successful_folds) == 0:
        raise ValidationError("All cross-validation folds failed")
    
    cv_results = {
        "fold_results": fold_results,
        "n_successful_folds": len(successful_folds),
        "mean_mse": float(jnp.mean([r["mse"] for r in successful_folds])),
        "std_mse": float(jnp.std([r["mse"] for r in successful_folds])),
        "mean_mae": float(jnp.mean([r["mae"] for r in successful_folds])),
        "std_mae": float(jnp.std([r["mae"] for r in successful_folds])),
        "mean_r2": float(jnp.mean([r["r2"] for r in successful_folds])),
        "std_r2": float(jnp.std([r["r2"] for r in successful_folds])),
    }
    
    return cv_results


def validate_model_robustness(
    surrogate: Surrogate,
    test_points: Array,
    perturbation_scale: float = 0.01,
    n_perturbations: int = 10,
) -> Dict[str, Any]:
    """Test surrogate model robustness to input perturbations.
    
    Args:
        surrogate: Trained surrogate model
        test_points: Points to test robustness around
        perturbation_scale: Scale of random perturbations
        n_perturbations: Number of perturbations per test point
        
    Returns:
        Robustness analysis results
    """
    from jax import random
    
    if test_points.ndim == 1:
        test_points = test_points[None, :]
    
    key = random.PRNGKey(42)
    robustness_results = []
    
    for i, point in enumerate(test_points):
        # Get baseline prediction
        try:
            baseline_pred = surrogate.predict(point)
            baseline_grad = surrogate.gradient(point)
        except Exception as e:
            warnings.warn(f"Failed to evaluate surrogate at test point {i}: {e}", ValidationWarning)
            continue
        
        # Generate perturbations
        key, subkey = random.split(key)
        perturbations = random.normal(subkey, (n_perturbations, len(point))) * perturbation_scale
        perturbed_points = point[None, :] + perturbations
        
        # Evaluate at perturbed points
        pred_variations = []
        grad_variations = []
        
        for perturbed_point in perturbed_points:
            try:
                pred = surrogate.predict(perturbed_point)
                grad = surrogate.gradient(perturbed_point)
                
                pred_variations.append(float(jnp.abs(pred - baseline_pred)))
                grad_variations.append(float(jnp.linalg.norm(grad - baseline_grad)))
                
            except Exception:
                # Skip failed evaluations
                continue
        
        if pred_variations:
            result = {
                "point_index": i,
                "baseline_pred": float(baseline_pred),
                "mean_pred_variation": float(jnp.mean(jnp.array(pred_variations))),
                "max_pred_variation": float(jnp.max(jnp.array(pred_variations))),
                "mean_grad_variation": float(jnp.mean(jnp.array(grad_variations))),
                "max_grad_variation": float(jnp.max(jnp.array(grad_variations))),
                "n_successful_perturbations": len(pred_variations),
            }
            robustness_results.append(result)
    
    if not robustness_results:
        raise ValidationError("Robustness testing failed for all test points")
    
    # Aggregate results
    overall_results = {
        "point_results": robustness_results,
        "mean_pred_sensitivity": float(jnp.mean([r["mean_pred_variation"] for r in robustness_results])),
        "max_pred_sensitivity": float(jnp.max([r["max_pred_variation"] for r in robustness_results])),
        "mean_grad_sensitivity": float(jnp.mean([r["mean_grad_variation"] for r in robustness_results])),
        "max_grad_sensitivity": float(jnp.max([r["max_grad_variation"] for r in robustness_results])),
        "perturbation_scale": perturbation_scale,
        "n_test_points": len(robustness_results),
    }
    
    # Check for concerning sensitivity levels
    if overall_results["max_pred_sensitivity"] > 10 * perturbation_scale:
        warnings.warn(
            f"High prediction sensitivity detected: max variation "
            f"{overall_results['max_pred_sensitivity']:.6f} for perturbation scale {perturbation_scale}",
            ValidationWarning
        )
    
    return overall_results