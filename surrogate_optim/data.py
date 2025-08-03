"""Data collection and management for surrogate optimization."""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import jax.numpy as jnp
from scipy.spatial.distance import cdist
from scipy.stats import qmc

Array = Union[np.ndarray, jnp.ndarray]


@dataclass
class Dataset:
    """Dataset container for surrogate optimization."""
    X: Array  # Input points [n_samples, n_dims]
    y: Array  # Function values [n_samples]
    gradients: Optional[Array] = None  # Gradient vectors [n_samples, n_dims]
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate dataset after initialization."""
        self.X = np.asarray(self.X)
        self.y = np.asarray(self.y)
        
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError("X and y must have same number of samples")
            
        if self.gradients is not None:
            self.gradients = np.asarray(self.gradients)
            if self.gradients.shape != self.X.shape:
                raise ValueError("Gradients must have same shape as X")
    
    @property
    def n_samples(self) -> int:
        """Number of samples in dataset."""
        return self.X.shape[0]
    
    @property
    def n_dims(self) -> int:
        """Number of dimensions."""
        return self.X.shape[1]
    
    def add_samples(self, X_new: Array, y_new: Array, gradients_new: Optional[Array] = None):
        """Add new samples to dataset."""
        X_new = np.asarray(X_new)
        y_new = np.asarray(y_new)
        
        self.X = np.vstack([self.X, X_new])
        self.y = np.concatenate([self.y, y_new])
        
        if gradients_new is not None and self.gradients is not None:
            gradients_new = np.asarray(gradients_new)
            self.gradients = np.vstack([self.gradients, gradients_new])
    
    def get_bounds(self) -> List[Tuple[float, float]]:
        """Get bounds of the dataset."""
        bounds = []
        for i in range(self.n_dims):
            bounds.append((float(self.X[:, i].min()), float(self.X[:, i].max())))
        return bounds


class DataCollector:
    """Data collection strategies for surrogate optimization."""
    
    def __init__(self, function: Callable[[Array], float]):
        """Initialize data collector.
        
        Args:
            function: Black-box function to evaluate
        """
        self.function = function
        self.evaluations = 0
    
    def collect_uniform(
        self,
        bounds: List[Tuple[float, float]],
        n_samples: int,
        random_state: int = 42,
    ) -> Dataset:
        """Collect data using uniform random sampling.
        
        Args:
            bounds: Variable bounds
            n_samples: Number of samples to collect
            random_state: Random seed
            
        Returns:
            Dataset with collected samples
        """
        np.random.seed(random_state)
        bounds = np.array(bounds)
        n_dims = len(bounds)
        
        # Generate random samples
        X = np.random.uniform(
            bounds[:, 0], bounds[:, 1], (n_samples, n_dims)
        )
        
        # Evaluate function
        y = np.array([self.function(x) for x in X])
        self.evaluations += n_samples
        
        return Dataset(X=X, y=y, metadata={'sampling': 'uniform'})
    
    def collect_sobol(
        self,
        bounds: List[Tuple[float, float]],
        n_samples: int,
        random_state: int = 42,
    ) -> Dataset:
        """Collect data using Sobol quasi-random sampling.
        
        Args:
            bounds: Variable bounds
            n_samples: Number of samples to collect
            random_state: Random seed
            
        Returns:
            Dataset with collected samples
        """
        bounds = np.array(bounds)
        n_dims = len(bounds)
        
        # Generate Sobol samples
        sampler = qmc.Sobol(d=n_dims, seed=random_state)
        samples = sampler.random(n=n_samples)
        
        # Scale to bounds
        X = qmc.scale(samples, bounds[:, 0], bounds[:, 1])
        
        # Evaluate function
        y = np.array([self.function(x) for x in X])
        self.evaluations += n_samples
        
        return Dataset(X=X, y=y, metadata={'sampling': 'sobol'})
    
    def collect_latin_hypercube(
        self,
        bounds: List[Tuple[float, float]],
        n_samples: int,
        random_state: int = 42,
    ) -> Dataset:
        """Collect data using Latin Hypercube sampling.
        
        Args:
            bounds: Variable bounds
            n_samples: Number of samples to collect
            random_state: Random seed
            
        Returns:
            Dataset with collected samples
        """
        bounds = np.array(bounds)
        n_dims = len(bounds)
        
        # Generate Latin Hypercube samples
        sampler = qmc.LatinHypercube(d=n_dims, seed=random_state)
        samples = sampler.random(n=n_samples)
        
        # Scale to bounds
        X = qmc.scale(samples, bounds[:, 0], bounds[:, 1])
        
        # Evaluate function
        y = np.array([self.function(x) for x in X])
        self.evaluations += n_samples
        
        return Dataset(X=X, y=y, metadata={'sampling': 'latin_hypercube'})
    
    def collect_adaptive(
        self,
        bounds: List[Tuple[float, float]],
        initial_samples: int,
        acquisition_function: str = "expected_improvement",
        batch_size: int = 10,
        n_iterations: int = 20,
        random_state: int = 42,
    ) -> Dataset:
        """Collect data using adaptive sampling.
        
        Args:
            bounds: Variable bounds
            initial_samples: Number of initial random samples
            acquisition_function: Acquisition function for adaptive sampling
            batch_size: Number of samples per iteration
            n_iterations: Number of adaptive iterations
            random_state: Random seed
            
        Returns:
            Dataset with adaptively collected samples
        """
        # Start with initial random samples
        dataset = self.collect_sobol(bounds, initial_samples, random_state)
        
        bounds = np.array(bounds)
        n_dims = len(bounds)
        
        for iteration in range(n_iterations):
            # Fit a simple surrogate for acquisition
            from .models import GPSurrogate
            surrogate = GPSurrogate()
            surrogate.fit(dataset.X, dataset.y)
            
            # Generate candidate points
            n_candidates = 1000
            candidates = np.random.uniform(
                bounds[:, 0], bounds[:, 1], (n_candidates, n_dims)
            )
            
            # Compute acquisition function
            acquisition_values = self._compute_acquisition(
                candidates, surrogate, acquisition_function
            )
            
            # Select best candidates
            best_indices = np.argsort(acquisition_values)[-batch_size:]
            X_new = candidates[best_indices]
            
            # Evaluate new points
            y_new = np.array([self.function(x) for x in X_new])
            self.evaluations += batch_size
            
            # Add to dataset
            dataset.add_samples(X_new, y_new)
            
        dataset.metadata['sampling'] = 'adaptive'
        return dataset
    
    def _compute_acquisition(
        self,
        candidates: Array,
        surrogate,
        acquisition_function: str,
    ) -> Array:
        """Compute acquisition function values."""
        if acquisition_function == "expected_improvement":
            return self._expected_improvement(candidates, surrogate)
        elif acquisition_function == "upper_confidence_bound":
            return self._upper_confidence_bound(candidates, surrogate)
        elif acquisition_function == "probability_improvement":
            return self._probability_improvement(candidates, surrogate)
        else:
            raise ValueError(f"Unknown acquisition function: {acquisition_function}")
    
    def _expected_improvement(self, candidates: Array, surrogate) -> Array:
        """Compute Expected Improvement acquisition function."""
        ei_values = []
        current_best = np.max([surrogate.predict(x) for x in candidates[:10]])  # Approximate
        
        for x in candidates:
            mean = surrogate.predict(x)
            std = surrogate.uncertainty(x)
            
            if std == 0:
                ei = 0
            else:
                z = (mean - current_best) / std
                ei = (mean - current_best) * self._normal_cdf(z) + std * self._normal_pdf(z)
                
            ei_values.append(ei)
            
        return np.array(ei_values)
    
    def _upper_confidence_bound(self, candidates: Array, surrogate, beta: float = 2.0) -> Array:
        """Compute Upper Confidence Bound acquisition function."""
        ucb_values = []
        for x in candidates:
            mean = surrogate.predict(x)
            std = surrogate.uncertainty(x)
            ucb = mean + beta * std
            ucb_values.append(ucb)
        return np.array(ucb_values)
    
    def _probability_improvement(self, candidates: Array, surrogate) -> Array:
        """Compute Probability of Improvement acquisition function."""
        pi_values = []
        current_best = np.max([surrogate.predict(x) for x in candidates[:10]])  # Approximate
        
        for x in candidates:
            mean = surrogate.predict(x)
            std = surrogate.uncertainty(x)
            
            if std == 0:
                pi = 0
            else:
                z = (mean - current_best) / std
                pi = self._normal_cdf(z)
                
            pi_values.append(pi)
            
        return np.array(pi_values)
    
    def _normal_cdf(self, x: float) -> float:
        """Compute standard normal CDF."""
        return 0.5 * (1 + np.erf(x / np.sqrt(2)))
    
    def _normal_pdf(self, x: float) -> float:
        """Compute standard normal PDF."""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    
    def estimate_gradients(
        self,
        data: Dataset,
        method: str = "finite_differences",
        epsilon: float = 1e-3,
    ) -> Dataset:
        """Estimate gradients from function evaluations.
        
        Args:
            data: Dataset with function evaluations
            method: Gradient estimation method
            epsilon: Finite difference step size
            
        Returns:
            Dataset with estimated gradients
        """
        if method == "finite_differences":
            gradients = self._finite_difference_gradients(data.X, epsilon)
        else:
            raise ValueError(f"Unknown gradient estimation method: {method}")
            
        data.gradients = gradients
        return data
    
    def _finite_difference_gradients(self, X: Array, epsilon: float) -> Array:
        """Estimate gradients using finite differences."""
        n_samples, n_dims = X.shape
        gradients = np.zeros_like(X)
        
        for i in range(n_samples):
            x = X[i]
            grad = np.zeros(n_dims)
            
            for j in range(n_dims):
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[j] += epsilon
                x_minus[j] -= epsilon
                
                y_plus = self.function(x_plus)
                y_minus = self.function(x_minus)
                
                grad[j] = (y_plus - y_minus) / (2 * epsilon)
                self.evaluations += 2
                
            gradients[i] = grad
            
        return gradients


class ActiveLearner:
    """Active learning for efficient surrogate training."""
    
    def __init__(
        self,
        function: Callable[[Array], float],
        initial_data: Dataset,
        surrogate_type: str = "gp",
    ):
        """Initialize active learner.
        
        Args:
            function: Black-box function to evaluate
            initial_data: Initial dataset
            surrogate_type: Type of surrogate model
        """
        self.function = function
        self.data = initial_data
        self.surrogate_type = surrogate_type
        self.surrogate = None
        self.evaluations = 0
    
    def learn_iteratively(
        self,
        n_iterations: int,
        batch_size: int = 1,
        acquisition_function: str = "expected_improvement",
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> Dataset:
        """Perform iterative active learning.
        
        Args:
            n_iterations: Number of learning iterations
            batch_size: Number of samples per iteration
            acquisition_function: Acquisition function
            bounds: Search bounds (defaults to data bounds)
            
        Returns:
            Updated dataset with new samples
        """
        if bounds is None:
            bounds = self.data.get_bounds()
            # Expand bounds slightly
            bounds = [(b[0] - 0.1 * (b[1] - b[0]), b[1] + 0.1 * (b[1] - b[0])) 
                     for b in bounds]
        
        bounds = np.array(bounds)
        n_dims = len(bounds)
        
        for iteration in range(n_iterations):
            # Fit surrogate to current data
            self._fit_surrogate()
            
            # Generate candidates
            n_candidates = 1000
            candidates = np.random.uniform(
                bounds[:, 0], bounds[:, 1], (n_candidates, n_dims)
            )
            
            # Compute acquisition values
            acquisition_values = self._compute_acquisition_values(
                candidates, acquisition_function
            )
            
            # Select best candidates
            best_indices = np.argsort(acquisition_values)[-batch_size:]
            X_new = candidates[best_indices]
            
            # Evaluate new points
            y_new = np.array([self.function(x) for x in X_new])
            self.evaluations += batch_size
            
            # Add to dataset
            self.data.add_samples(X_new, y_new)
            
            print(f"Iteration {iteration + 1}: Added {batch_size} samples, "
                  f"total evaluations: {self.evaluations}")
            
        return self.data
    
    def _fit_surrogate(self):
        """Fit surrogate model to current data."""
        from .models import NeuralSurrogate, GPSurrogate, RandomForestSurrogate
        
        if self.surrogate_type == "neural_network":
            self.surrogate = NeuralSurrogate(epochs=500)
        elif self.surrogate_type == "gp":
            self.surrogate = GPSurrogate()
        elif self.surrogate_type == "random_forest":
            self.surrogate = RandomForestSurrogate()
        else:
            raise ValueError(f"Unknown surrogate type: {self.surrogate_type}")
            
        self.surrogate.fit(self.data.X, self.data.y)
    
    def _compute_acquisition_values(
        self,
        candidates: Array,
        acquisition_function: str,
    ) -> Array:
        """Compute acquisition function values."""
        if acquisition_function == "expected_improvement":
            return self._expected_improvement(candidates)
        elif acquisition_function == "upper_confidence_bound":
            return self._upper_confidence_bound(candidates)
        elif acquisition_function == "entropy_search":
            return self._entropy_search(candidates)
        else:
            raise ValueError(f"Unknown acquisition function: {acquisition_function}")
    
    def _expected_improvement(self, candidates: Array) -> Array:
        """Expected improvement acquisition function."""
        current_best = np.max(self.data.y)
        ei_values = []
        
        for x in candidates:
            mean = self.surrogate.predict(x)
            std = self.surrogate.uncertainty(x)
            
            if std == 0:
                ei = 0
            else:
                z = (mean - current_best) / std
                ei = (mean - current_best) * self._normal_cdf(z) + std * self._normal_pdf(z)
                
            ei_values.append(max(0, ei))
            
        return np.array(ei_values)
    
    def _upper_confidence_bound(self, candidates: Array, beta: float = 2.0) -> Array:
        """Upper confidence bound acquisition function."""
        ucb_values = []
        for x in candidates:
            mean = self.surrogate.predict(x)
            std = self.surrogate.uncertainty(x)
            ucb = mean + beta * std
            ucb_values.append(ucb)
        return np.array(ucb_values)
    
    def _entropy_search(self, candidates: Array) -> Array:
        """Simplified entropy search acquisition function."""
        # Simplified version - just use uncertainty
        return np.array([self.surrogate.uncertainty(x) for x in candidates])
    
    def _normal_cdf(self, x: float) -> float:
        """Standard normal CDF."""
        return 0.5 * (1 + np.erf(x / np.sqrt(2)))
    
    def _normal_pdf(self, x: float) -> float:
        """Standard normal PDF."""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)


def collect_data(
    function: Callable[[Array], float],
    n_samples: int,
    bounds: List[Tuple[float, float]],
    sampling: str = "sobol",
    **kwargs
) -> Dataset:
    """Convenience function for data collection.
    
    Args:
        function: Black-box function to evaluate
        n_samples: Number of samples to collect
        bounds: Variable bounds
        sampling: Sampling strategy
        **kwargs: Additional arguments for sampling
        
    Returns:
        Dataset with collected samples
    """
    collector = DataCollector(function)
    
    if sampling == "uniform":
        return collector.collect_uniform(bounds, n_samples, **kwargs)
    elif sampling == "sobol":
        return collector.collect_sobol(bounds, n_samples, **kwargs)
    elif sampling == "latin_hypercube":
        return collector.collect_latin_hypercube(bounds, n_samples, **kwargs)
    elif sampling == "adaptive":
        return collector.collect_adaptive(bounds, n_samples // 5, **kwargs)
    else:
        raise ValueError(f"Unknown sampling strategy: {sampling}")