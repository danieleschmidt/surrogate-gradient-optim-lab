"""Multi-start global optimization for surrogate models."""

from typing import Any, Dict, List, Optional, Tuple

import jax.numpy as jnp
from jax import Array, random

from ..models.base import Surrogate
from .base import BaseOptimizer, OptimizationResult
from .gradient_descent import GradientDescentOptimizer


class MultiStartOptimizer(BaseOptimizer):
    """Multi-start global optimizer using surrogate models.
    
    Performs optimization from multiple starting points to find global optimum.
    """
    
    def __init__(
        self,
        n_starts: int = 10,
        start_method: str = "random",
        local_optimizer: str = "gradient_descent",
        local_optimizer_params: Optional[Dict[str, Any]] = None,
        parallel: bool = False,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        verbose: bool = False,
        random_seed: int = 42,
    ):
        """Initialize multi-start optimizer.
        
        Args:
            n_starts: Number of starting points
            start_method: Method for generating starts ('random', 'sobol', 'grid')
            local_optimizer: Local optimizer to use ('gradient_descent')
            local_optimizer_params: Parameters for local optimizer
            parallel: Whether to run local optimizations in parallel (not implemented)
            max_iterations: Maximum iterations per local optimization
            tolerance: Convergence tolerance
            verbose: Whether to print progress
            random_seed: Random seed for reproducibility
        """
        super().__init__(max_iterations, tolerance, verbose)
        
        self.n_starts = n_starts
        self.start_method = start_method
        self.local_optimizer = local_optimizer
        self.local_optimizer_params = local_optimizer_params or {}
        self.parallel = parallel
        self.random_seed = random_seed
        
        # Results storage
        self.local_results = []
        self.starting_points = []
    
    def optimize(
        self,
        surrogate: Surrogate,
        x0: Array,
        bounds: Optional[List[Tuple[float, float]]] = None,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> OptimizationResult:
        """Perform multi-start global optimization.
        
        Args:
            surrogate: Trained surrogate model
            x0: Initial point (used as one of the starting points)
            bounds: Bounds for each dimension (required for global optimization)
            constraints: Optional constraints
            
        Returns:
            Best optimization result found
        """
        if bounds is None:
            raise ValueError("Bounds are required for multi-start global optimization")
        
        # Validate inputs
        self._validate_inputs(surrogate, x0, bounds)
        self._reset_state()
        
        # Generate starting points
        starting_points = self._generate_starting_points(x0, bounds)
        self.starting_points = starting_points
        
        if self.verbose:
            print(f"Starting multi-start optimization with {len(starting_points)} starts")
        
        # Perform local optimization from each starting point
        local_results = []
        
        for i, start_point in enumerate(starting_points):
            if self.verbose:
                print(f"  Start {i+1}/{len(starting_points)}: {start_point}")
            
            # Create local optimizer
            local_opt = self._create_local_optimizer()
            
            try:
                # Run local optimization
                result = local_opt.optimize(
                    surrogate=surrogate,
                    x0=start_point,
                    bounds=bounds,
                    constraints=constraints
                )
                
                local_results.append(result)
                
                if self.verbose:
                    status = "✓" if result.success else "✗"
                    print(f"    {status} f = {result.fun:.6f}, x = {result.x}")
                
            except Exception as e:
                if self.verbose:
                    print(f"    ✗ Failed: {e}")
                # Create dummy result for failed optimization
                dummy_result = OptimizationResult(
                    x=start_point,
                    fun=float('inf'),
                    success=False,
                    message=f"Optimization failed: {e}",
                    nit=0,
                    nfev=0
                )
                local_results.append(dummy_result)
        
        self.local_results = local_results
        
        # Find best result
        successful_results = [r for r in local_results if r.success and jnp.isfinite(r.fun)]
        
        if not successful_results:
            # No successful optimizations
            best_result = OptimizationResult(
                x=x0,
                fun=float('inf'),
                success=False,
                message="All local optimizations failed",
                nit=0,
                nfev=sum(r.nfev for r in local_results),
                metadata={
                    "n_starts": self.n_starts,
                    "n_successful": 0,
                    "local_results": local_results
                }
            )
        else:
            # Find globally best result
            best_idx = jnp.argmin(jnp.array([r.fun for r in successful_results]))
            best_result = successful_results[best_idx]
            
            # Update metadata
            best_result.metadata = best_result.metadata or {}
            best_result.metadata.update({
                "n_starts": self.n_starts,
                "n_successful": len(successful_results),
                "local_results": local_results,
                "global_best_idx": best_idx,
                "starting_points": starting_points,
            })
            
            # Update total function evaluations
            best_result.nfev = sum(r.nfev for r in local_results)
            
            best_result.message = f"Global optimum found from {len(successful_results)}/{self.n_starts} successful starts"
        
        if self.verbose:
            n_success = len(successful_results)
            print(f"Multi-start complete: {n_success}/{self.n_starts} successful")
            if n_success > 0:
                print(f"Global optimum: f = {best_result.fun:.6f} at x = {best_result.x}")
        
        return best_result
    
    def _generate_starting_points(
        self,
        x0: Array,
        bounds: List[Tuple[float, float]]
    ) -> List[Array]:
        """Generate starting points for multi-start optimization."""
        starting_points = [x0]  # Include provided initial point
        
        n_dims = len(bounds)
        n_additional = self.n_starts - 1
        
        if n_additional <= 0:
            return starting_points
        
        key = random.PRNGKey(self.random_seed)
        
        if self.start_method == "random":
            # Random uniform sampling within bounds
            key, subkey = random.split(key)
            uniform_samples = random.uniform(subkey, shape=(n_additional, n_dims))
            
            # Transform to bounds
            for i, (lower, upper) in enumerate(bounds):
                uniform_samples = uniform_samples.at[:, i].set(
                    lower + (upper - lower) * uniform_samples[:, i]
                )
            
            for sample in uniform_samples:
                starting_points.append(sample)
        
        elif self.start_method == "sobol":
            # Sobol sequence sampling
            try:
                from scipy.stats import qmc
                sampler = qmc.Sobol(d=n_dims, seed=self.random_seed)
                sobol_samples = sampler.random(n_additional)
                
                # Transform to bounds
                for i, (lower, upper) in enumerate(bounds):
                    sobol_samples[:, i] = lower + (upper - lower) * sobol_samples[:, i]
                
                for sample in sobol_samples:
                    starting_points.append(jnp.array(sample))
                    
            except ImportError:
                # Fallback to random if scipy not available
                return self._generate_starting_points_random(x0, bounds)
        
        elif self.start_method == "grid":
            # Grid-based starting points
            points_per_dim = max(1, int(n_additional ** (1.0 / n_dims)))
            
            # Create grid coordinates
            coords = []
            for lower, upper in bounds:
                coords.append(jnp.linspace(lower, upper, points_per_dim))
            
            # Create meshgrid and sample points
            if n_dims == 1:
                grid_points = coords[0][:n_additional]
                for point in grid_points:
                    starting_points.append(jnp.array([point]))
            elif n_dims == 2:
                X, Y = jnp.meshgrid(coords[0], coords[1])
                grid_points = jnp.stack([X.flatten(), Y.flatten()], axis=1)
                
                # Subsample if we have too many points
                if len(grid_points) > n_additional:
                    key, subkey = random.split(key)
                    indices = random.choice(subkey, len(grid_points), shape=(n_additional,), replace=False)
                    grid_points = grid_points[indices]
                
                for point in grid_points:
                    starting_points.append(point)
            else:
                # For higher dimensions, fall back to random
                if self.verbose:
                    print(f"Grid method not efficient for {n_dims}D, using random")
                return self._generate_starting_points_random(x0, bounds)
        
        else:
            raise ValueError(f"Unknown start method: {self.start_method}")
        
        return starting_points
    
    def _generate_starting_points_random(
        self,
        x0: Array,
        bounds: List[Tuple[float, float]]
    ) -> List[Array]:
        """Generate random starting points (fallback method)."""
        starting_points = [x0]
        
        key = random.PRNGKey(self.random_seed)
        key, subkey = random.split(key)
        
        n_dims = len(bounds)
        n_additional = self.n_starts - 1
        
        uniform_samples = random.uniform(subkey, shape=(n_additional, n_dims))
        
        # Transform to bounds
        for i, (lower, upper) in enumerate(bounds):
            uniform_samples = uniform_samples.at[:, i].set(
                lower + (upper - lower) * uniform_samples[:, i]
            )
        
        for sample in uniform_samples:
            starting_points.append(sample)
        
        return starting_points
    
    def _create_local_optimizer(self) -> BaseOptimizer:
        """Create local optimizer for individual starts."""
        if self.local_optimizer == "gradient_descent":
            params = {
                "max_iterations": self.max_iterations,
                "tolerance": self.tolerance,
                "verbose": False,  # Suppress local optimizer output
            }
            params.update(self.local_optimizer_params)
            return GradientDescentOptimizer(**params)
        else:
            raise ValueError(f"Unknown local optimizer: {self.local_optimizer}")
    
    def get_all_results(self) -> List[OptimizationResult]:
        """Get results from all local optimizations."""
        return self.local_results.copy()
    
    def get_starting_points(self) -> List[Array]:
        """Get all starting points used."""
        return self.starting_points.copy()
    
    def analyze_convergence(self) -> Dict[str, Any]:
        """Analyze convergence statistics across all starts."""
        if not self.local_results:
            return {}
        
        successful = [r for r in self.local_results if r.success]
        
        analysis = {
            "n_starts": len(self.local_results),
            "n_successful": len(successful),
            "success_rate": len(successful) / len(self.local_results),
        }
        
        if successful:
            function_values = [r.fun for r in successful]
            iterations = [r.nit for r in successful]
            
            analysis.update({
                "best_value": float(jnp.min(jnp.array(function_values))),
                "worst_value": float(jnp.max(jnp.array(function_values))),
                "mean_value": float(jnp.mean(jnp.array(function_values))),
                "std_value": float(jnp.std(jnp.array(function_values))),
                "mean_iterations": float(jnp.mean(jnp.array(iterations))),
                "total_evaluations": sum(r.nfev for r in self.local_results),
            })
        
        return analysis