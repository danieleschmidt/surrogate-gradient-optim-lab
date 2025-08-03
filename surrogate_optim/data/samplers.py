"""Sampling strategies for data collection."""

from abc import ABC, abstractmethod
from typing import Optional

import jax.numpy as jnp
from jax import Array, random


class SamplingStrategy(ABC):
    """Abstract base class for sampling strategies."""
    
    def __init__(self, random_seed: int = 42):
        """Initialize sampling strategy.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        self.key = random.PRNGKey(random_seed)
    
    @abstractmethod
    def sample(self, n_samples: int, n_dims: int) -> Array:
        """Generate samples in the unit hypercube [0, 1]^n_dims.
        
        Args:
            n_samples: Number of samples to generate
            n_dims: Number of dimensions
            
        Returns:
            Array of shape [n_samples, n_dims] with values in [0, 1]
        """
        pass


class RandomSampler(SamplingStrategy):
    """Random uniform sampling in the unit hypercube."""
    
    def sample(self, n_samples: int, n_dims: int) -> Array:
        """Generate random uniform samples."""
        self.key, subkey = random.split(self.key)
        return random.uniform(subkey, shape=(n_samples, n_dims))


class SobolSampler(SamplingStrategy):
    """Sobol quasi-random sampling for better space coverage."""
    
    def sample(self, n_samples: int, n_dims: int) -> Array:
        """Generate Sobol sequence samples.
        
        Note: This is a simplified Sobol implementation.
        For production use, consider using scipy.stats.qmc.Sobol.
        """
        try:
            from scipy.stats import qmc
            
            # Use scipy's Sobol sampler if available
            sampler = qmc.Sobol(d=n_dims, seed=self.random_seed)
            samples = sampler.random(n_samples)
            return jnp.array(samples)
            
        except ImportError:
            # Fallback to random sampling if scipy not available
            print("Warning: scipy not available, falling back to random sampling")
            return RandomSampler(self.random_seed).sample(n_samples, n_dims)


class LatinHypercubeSampler(SamplingStrategy):
    """Latin Hypercube Sampling for stratified sampling."""
    
    def sample(self, n_samples: int, n_dims: int) -> Array:
        """Generate Latin Hypercube samples."""
        try:
            from scipy.stats import qmc
            
            # Use scipy's LHS sampler if available
            sampler = qmc.LatinHypercube(d=n_dims, seed=self.random_seed)
            samples = sampler.random(n_samples)
            return jnp.array(samples)
            
        except ImportError:
            # Fallback implementation of Latin Hypercube Sampling
            return self._fallback_lhs(n_samples, n_dims)
    
    def _fallback_lhs(self, n_samples: int, n_dims: int) -> Array:
        """Fallback Latin Hypercube implementation."""
        self.key, subkey = random.split(self.key)
        
        # Create grid
        samples = jnp.zeros((n_samples, n_dims))
        
        for dim in range(n_dims):
            # Create stratified intervals
            intervals = jnp.linspace(0, 1, n_samples + 1)
            
            # Random permutation of intervals
            perm_key, subkey = random.split(subkey)
            perm = random.permutation(perm_key, n_samples)
            
            # Random points within each interval
            rand_key, subkey = random.split(subkey)
            rand_vals = random.uniform(rand_key, shape=(n_samples,))
            
            # Combine to get LHS samples
            interval_width = 1.0 / n_samples
            lhs_samples = intervals[perm] + interval_width * rand_vals
            
            samples = samples.at[:, dim].set(lhs_samples)
        
        return samples


class GridSampler(SamplingStrategy):
    """Grid sampling for systematic coverage."""
    
    def sample(self, n_samples: int, n_dims: int) -> Array:
        """Generate grid samples.
        
        Note: Actual number of samples may differ from n_samples
        to maintain grid structure.
        """
        # Calculate points per dimension for approximately n_samples total
        points_per_dim = max(1, int(n_samples ** (1.0 / n_dims)))
        
        # Create grid coordinates
        coords = [jnp.linspace(0, 1, points_per_dim) for _ in range(n_dims)]
        
        # Create meshgrid
        mesh = jnp.meshgrid(*coords, indexing='ij')
        
        # Flatten to get sample points
        samples = jnp.stack([grid.flatten() for grid in mesh], axis=1)
        
        # If we have too many samples, subsample randomly
        if len(samples) > n_samples:
            self.key, subkey = random.split(self.key)
            indices = random.choice(subkey, len(samples), shape=(n_samples,), replace=False)
            samples = samples[indices]
        
        return samples


class HaltonSampler(SamplingStrategy):
    """Halton sequence sampling for quasi-random coverage."""
    
    def sample(self, n_samples: int, n_dims: int) -> Array:
        """Generate Halton sequence samples."""
        # Generate prime numbers for Halton sequence
        primes = self._get_primes(n_dims)
        
        samples = jnp.zeros((n_samples, n_dims))
        
        for dim in range(n_dims):
            base = primes[dim]
            halton_seq = jnp.array([self._halton_number(i + 1, base) for i in range(n_samples)])
            samples = samples.at[:, dim].set(halton_seq)
        
        return samples
    
    def _get_primes(self, n: int) -> list:
        """Get first n prime numbers."""
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
        if n <= len(primes):
            return primes[:n]
        
        # Generate more primes if needed (simple sieve)
        candidate = primes[-1] + 2
        while len(primes) < n:
            is_prime = True
            for p in primes:
                if p * p > candidate:
                    break
                if candidate % p == 0:
                    is_prime = False
                    break
            if is_prime:
                primes.append(candidate)
            candidate += 2
        
        return primes[:n]
    
    def _halton_number(self, index: int, base: int) -> float:
        """Generate single Halton number."""
        result = 0.0
        f = 1.0
        i = index
        
        while i > 0:
            f = f / base
            result += f * (i % base)
            i = i // base
        
        return result