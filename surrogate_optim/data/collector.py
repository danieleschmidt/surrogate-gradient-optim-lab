"""Data collection utilities for surrogate model training."""

from typing import Callable, List, Optional, Tuple, Union

import jax.numpy as jnp
from jax import Array, random

from ..models.base import Dataset
from .samplers import LatinHypercubeSampler, RandomSampler, SamplingStrategy, SobolSampler


class DataCollector:
    """Collects data for surrogate model training.
    
    Supports various sampling strategies for efficient data collection
    from black-box functions.
    """
    
    def __init__(
        self,
        function: Callable[[Array], float],
        bounds: List[Tuple[float, float]],
        random_seed: int = 42,
    ):
        """Initialize data collector.
        
        Args:
            function: Black-box function to sample
            bounds: Bounds for each input dimension [(min, max), ...]
            random_seed: Random seed for reproducibility
        """
        self.function = function
        self.bounds = bounds
        self.n_dims = len(bounds)
        self.random_seed = random_seed
        
        # Validate bounds
        for i, (lower, upper) in enumerate(bounds):
            if lower >= upper:
                raise ValueError(f"Invalid bounds at dimension {i}: {lower} >= {upper}")
    
    def collect(
        self,
        n_samples: int,
        sampling: Union[str, SamplingStrategy] = "sobol",
        estimate_gradients: bool = False,
        gradient_eps: float = 1e-6,
        batch_size: Optional[int] = None,
        verbose: bool = True,
    ) -> Dataset:
        """Collect data from the black-box function.
        
        Args:
            n_samples: Number of samples to collect
            sampling: Sampling strategy ('random', 'sobol', 'lhs', or SamplingStrategy)
            estimate_gradients: Whether to estimate gradients via finite differences
            gradient_eps: Epsilon for finite difference gradient estimation
            batch_size: Batch size for function evaluations (None for all at once)
            verbose: Whether to print progress
            
        Returns:
            Dataset with collected samples
        """
        if verbose:
            print(f"Collecting {n_samples} samples using {sampling} sampling...")
        
        # Generate sample points
        X = self._generate_samples(n_samples, sampling)
        
        # Evaluate function at sample points
        y = self._evaluate_function(X, batch_size, verbose)
        
        # Estimate gradients if requested
        gradients = None
        if estimate_gradients:
            if verbose:
                print("Estimating gradients via finite differences...")
            gradients = self._estimate_gradients(X, gradient_eps, batch_size, verbose)
        
        # Create dataset
        dataset = Dataset(
            X=X,
            y=y,
            gradients=gradients,
            metadata={
                "sampling_method": str(sampling),
                "n_samples": n_samples,
                "bounds": self.bounds,
                "has_gradients": estimate_gradients,
            }
        )
        
        if verbose:
            print(f"Data collection complete. Dataset: {dataset.n_samples} samples, {dataset.n_dims} dims")
        
        return dataset
    
    def _generate_samples(self, n_samples: int, sampling: Union[str, SamplingStrategy]) -> Array:
        """Generate sample points using specified sampling strategy."""
        if isinstance(sampling, str):
            # Create sampler from string
            if sampling == "random":
                sampler = RandomSampler(self.random_seed)
            elif sampling == "sobol":
                sampler = SobolSampler(self.random_seed)
            elif sampling in ["lhs", "latin_hypercube"]:
                sampler = LatinHypercubeSampler(self.random_seed)
            else:
                raise ValueError(f"Unknown sampling strategy: {sampling}")
        else:
            sampler = sampling
        
        # Generate samples in unit hypercube
        X_unit = sampler.sample(n_samples, self.n_dims)
        
        # Transform to actual bounds
        X = self._transform_to_bounds(X_unit)
        
        return X
    
    def _transform_to_bounds(self, X_unit: Array) -> Array:
        """Transform samples from unit hypercube to actual bounds."""
        X = jnp.zeros_like(X_unit)
        
        for i, (lower, upper) in enumerate(self.bounds):
            X = X.at[:, i].set(lower + (upper - lower) * X_unit[:, i])
        
        return X
    
    def _evaluate_function(self, X: Array, batch_size: Optional[int], verbose: bool) -> Array:
        """Evaluate function at all sample points."""
        n_samples = X.shape[0]
        y = jnp.zeros(n_samples)
        
        if batch_size is None:
            batch_size = n_samples
        
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            
            # Evaluate batch
            for i in range(start_idx, end_idx):
                try:
                    y_val = self.function(X[i])
                    y = y.at[i].set(float(y_val))
                except Exception as e:
                    print(f"Warning: Function evaluation failed at point {i}: {e}")
                    y = y.at[i].set(jnp.nan)
            
            if verbose and n_batches > 1:
                print(f"  Batch {batch_idx + 1}/{n_batches} complete")
        
        # Check for failed evaluations
        n_failed = jnp.sum(jnp.isnan(y))
        if n_failed > 0:
            print(f"Warning: {n_failed} function evaluations failed")
        
        return y
    
    def _estimate_gradients(
        self,
        X: Array,
        eps: float,
        batch_size: Optional[int],
        verbose: bool
    ) -> Array:
        """Estimate gradients using finite differences."""
        n_samples, n_dims = X.shape
        gradients = jnp.zeros((n_samples, n_dims))
        
        if batch_size is None:
            batch_size = n_samples
        
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            
            # Estimate gradients for batch
            for i in range(start_idx, end_idx):
                x = X[i]
                grad = jnp.zeros(n_dims)
                
                for j in range(n_dims):
                    # Forward and backward perturbations
                    x_plus = x.at[j].add(eps)
                    x_minus = x.at[j].add(-eps)
                    
                    # Ensure perturbations stay within bounds
                    lower, upper = self.bounds[j]
                    x_plus = x_plus.at[j].set(jnp.clip(x_plus[j], lower, upper))
                    x_minus = x_minus.at[j].set(jnp.clip(x_minus[j], lower, upper))
                    
                    try:
                        f_plus = self.function(x_plus)
                        f_minus = self.function(x_minus)
                        
                        # Central difference
                        grad_j = (f_plus - f_minus) / (2 * eps)
                        grad = grad.at[j].set(grad_j)
                    except Exception as e:
                        print(f"Warning: Gradient estimation failed at point {i}, dim {j}: {e}")
                        grad = grad.at[j].set(jnp.nan)
                
                gradients = gradients.at[i].set(grad)
            
            if verbose and n_batches > 1:
                print(f"  Gradient batch {batch_idx + 1}/{n_batches} complete")
        
        return gradients
    
    def collect_adaptive(
        self,
        initial_samples: int,
        acquisition_function: str = "uncertainty",
        n_iterations: int = 10,
        batch_size: int = 10,
        surrogate_type: str = "gp",
    ) -> Dataset:
        """Collect data using adaptive sampling (active learning).
        
        Args:
            initial_samples: Number of initial random samples
            acquisition_function: Acquisition function for adaptive sampling
            n_iterations: Number of adaptive sampling iterations
            batch_size: Number of points to add each iteration
            surrogate_type: Type of surrogate for acquisition function
            
        Returns:
            Dataset with adaptively collected samples
        """
        # Collect initial data
        dataset = self.collect(initial_samples, sampling="sobol", verbose=True)
        
        # Adaptive sampling iterations
        for iteration in range(n_iterations):
            print(f"Adaptive sampling iteration {iteration + 1}/{n_iterations}")
            
            # Train surrogate model for acquisition function
            if surrogate_type == "gp":
                from ..models import GPSurrogate
                surrogate = GPSurrogate()
            else:
                raise ValueError(f"Unsupported surrogate type for adaptive sampling: {surrogate_type}")
            
            surrogate.fit(dataset)
            
            # Generate candidate points
            candidates = self._generate_samples(1000, "random")
            
            # Compute acquisition function
            acquisition_values = self._compute_acquisition(
                candidates, surrogate, acquisition_function
            )
            
            # Select best points
            best_indices = jnp.argsort(acquisition_values)[-batch_size:]
            new_X = candidates[best_indices]
            
            # Evaluate function at new points
            new_y = self._evaluate_function(new_X, None, False)
            
            # Add to dataset
            dataset = Dataset(
                X=jnp.vstack([dataset.X, new_X]),
                y=jnp.concatenate([dataset.y, new_y]),
                gradients=None,
                metadata=dataset.metadata
            )
        
        print(f"Adaptive sampling complete. Final dataset: {dataset.n_samples} samples")
        return dataset
    
    def _compute_acquisition(self, X: Array, surrogate, function_name: str) -> Array:
        """Compute acquisition function values."""
        if function_name == "uncertainty":
            return surrogate.uncertainty(X)
        elif function_name == "expected_improvement":
            # Simplified EI - would need current best value in practice
            mean, std = surrogate.predict_with_uncertainty(X)
            return std  # Simplified - just use uncertainty
        else:
            raise ValueError(f"Unknown acquisition function: {function_name}")


def collect_data(
    function: Callable[[Array], float],
    n_samples: int,
    bounds: List[Tuple[float, float]],
    sampling: str = "sobol",
    estimate_gradients: bool = False,
    random_seed: int = 42,
    verbose: bool = True,
) -> Dataset:
    """Convenience function for data collection.
    
    Args:
        function: Black-box function to sample
        n_samples: Number of samples to collect
        bounds: Bounds for each input dimension
        sampling: Sampling strategy
        estimate_gradients: Whether to estimate gradients
        random_seed: Random seed
        verbose: Whether to print progress
        
    Returns:
        Dataset with collected samples
    """
    collector = DataCollector(function, bounds, random_seed)
    return collector.collect(
        n_samples=n_samples,
        sampling=sampling,
        estimate_gradients=estimate_gradients,
        verbose=verbose,
    )