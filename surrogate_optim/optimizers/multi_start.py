"""Multi-start global optimization using surrogate models."""

import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import jax.numpy as jnp
from jax import Array, random
import numpy as np

from .base import BaseOptimizer, OptimizationResult
from .gradient_descent import GradientDescentOptimizer


class MultiStartOptimizer(BaseOptimizer):
    """Multi-start global optimization using surrogate models.
    
    Performs optimization from multiple starting points to find the global
    optimum. Supports various initialization strategies and parallel execution.
    """
    
    def __init__(
        self,
        surrogate=None,
        local_optimizer: Optional[BaseOptimizer] = None,
        n_starts: int = 10,
        start_method: str = "sobol",
        clustering_threshold: float = 0.1,
        parallel: bool = True,
        n_jobs: int = -1,
        seed: int = 42,
        name: str = "multi_start"
    ):
        """Initialize multi-start optimizer.
        
        Args:
            surrogate: Surrogate model to optimize.
            local_optimizer: Local optimizer to use from each start.
            n_starts: Number of starting points.
            start_method: Method for generating starts ("random", "sobol", "grid").
            clustering_threshold: Distance threshold for clustering results.
            parallel: Whether to run optimizations in parallel.
            n_jobs: Number of parallel jobs (-1 for all cores).
            seed: Random seed for reproducible starts.
            name: Optimizer name.
        """
        super().__init__(surrogate, name)
        self.local_optimizer = local_optimizer
        self.n_starts = n_starts
        self.start_method = start_method
        self.clustering_threshold = clustering_threshold
        self.parallel = parallel
        self.n_jobs = n_jobs
        self.seed = seed
        
        # Set default local optimizer
        if self.local_optimizer is None:
            self.local_optimizer = GradientDescentOptimizer(
                method="adam", 
                learning_rate=0.01
            )
    
    def optimize(
        self,
        bounds: List[Tuple[float, float]],
        constraints: Optional[List[Dict[str, Any]]] = None,
        max_iterations_per_start: int = 500,
        tolerance: float = 1e-6,
        verbose: bool = False,
        **kwargs
    ) -> 'GlobalOptimizationResult':
        """Perform multi-start global optimization.
        
        Args:
            bounds: Bounds for each dimension as (min, max) tuples.
            constraints: Constraints (passed to local optimizer).
            max_iterations_per_start: Maximum iterations per local optimization.
            tolerance: Convergence tolerance for local optimizations.
            verbose: Whether to print progress.
            **kwargs: Additional parameters for local optimizer.
            
        Returns:
            Global optimization results.
        """
        start_time = time.time()
        
        self._check_surrogate()
        
        if len(bounds) == 0:
            raise ValueError("Bounds must be provided for multi-start optimization")
        
        # Generate starting points
        starting_points = self._generate_starting_points(bounds)
        
        if verbose:
            print(f"Generated {len(starting_points)} starting points using {self.start_method}")
        
        # Set up local optimizer
        self.local_optimizer.set_surrogate(self.surrogate)
        
        # Run local optimizations
        if self.parallel and len(starting_points) > 1:
            results = self._optimize_parallel(
                starting_points, bounds, constraints, 
                max_iterations_per_start, tolerance, verbose, **kwargs
            )
        else:
            results = self._optimize_sequential(
                starting_points, bounds, constraints,
                max_iterations_per_start, tolerance, verbose, **kwargs
            )
        
        # Process results
        valid_results = [r for r in results if r is not None]
        
        if not valid_results:
            raise RuntimeError("All local optimizations failed")
        
        # Find best result
        best_result = min(valid_results, key=lambda r: r.f_opt)
        
        # Cluster results to find unique local optima
        clustered_results = self._cluster_results(valid_results)
        
        optimization_time = time.time() - start_time
        
        return GlobalOptimizationResult(
            best_result=best_result,
            all_results=valid_results,
            clustered_results=clustered_results,
            starting_points=starting_points,
            n_starts=len(starting_points),
            n_successful=len(valid_results),
            optimization_time=optimization_time,
            metadata={
                "start_method": self.start_method,
                "local_optimizer": self.local_optimizer.name,
                "clustering_threshold": self.clustering_threshold,
            }
        )
    
    def _generate_starting_points(self, bounds: List[Tuple[float, float]]) -> List[Array]:
        """Generate starting points within bounds.
        
        Args:
            bounds: Bounds for each dimension.
            
        Returns:
            List of starting points.
        """
        dim = len(bounds)
        lower = jnp.array([b[0] for b in bounds])
        upper = jnp.array([b[1] for b in bounds])
        
        if self.start_method == "random":
            key = random.PRNGKey(self.seed)
            uniform_samples = random.uniform(key, (self.n_starts, dim))
            points = lower + uniform_samples * (upper - lower)
            
        elif self.start_method == "sobol":
            # Use scipy's Sobol sequence if available
            try:
                from scipy.stats import qmc
                sampler = qmc.Sobol(d=dim, seed=self.seed)
                uniform_samples = sampler.random(self.n_starts)
                points = lower + uniform_samples * (upper - lower)
                points = jnp.array(points)
            except ImportError:
                # Fallback to random sampling
                print("Warning: scipy not available, using random sampling instead of Sobol")
                key = random.PRNGKey(self.seed)
                uniform_samples = random.uniform(key, (self.n_starts, dim))
                points = lower + uniform_samples * (upper - lower)
        
        elif self.start_method == "grid":
            # Generate grid points
            n_per_dim = int(self.n_starts ** (1.0 / dim)) + 1
            grid_1d = [jnp.linspace(bounds[i][0], bounds[i][1], n_per_dim) for i in range(dim)]
            
            # Create meshgrid
            grids = jnp.meshgrid(*grid_1d, indexing='ij')
            points = jnp.stack([g.ravel() for g in grids], axis=1)
            
            # Subsample if we have too many points
            if len(points) > self.n_starts:
                key = random.PRNGKey(self.seed)
                indices = random.choice(key, len(points), (self.n_starts,), replace=False)
                points = points[indices]
        
        else:
            raise ValueError(f"Unknown start method: {self.start_method}")
        
        return [points[i] for i in range(len(points))]
    
    def _optimize_sequential(
        self,
        starting_points: List[Array],
        bounds: List[Tuple[float, float]],
        constraints: Optional[List[Dict[str, Any]]],
        max_iterations: int,
        tolerance: float,
        verbose: bool,
        **kwargs
    ) -> List[Optional[OptimizationResult]]:
        """Run local optimizations sequentially.
        
        Args:
            starting_points: List of starting points.
            bounds: Optimization bounds.
            constraints: Constraints.
            max_iterations: Max iterations per optimization.
            tolerance: Convergence tolerance.
            verbose: Verbosity flag.
            **kwargs: Additional parameters.
            
        Returns:
            List of optimization results.
        """
        results = []
        
        for i, x0 in enumerate(starting_points):
            if verbose:
                print(f"Starting local optimization {i+1}/{len(starting_points)}")
            
            try:
                result = self.local_optimizer.optimize(
                    x0=x0,
                    bounds=bounds,
                    constraints=constraints,
                    max_iterations=max_iterations,
                    tolerance=tolerance,
                    verbose=False,  # Suppress local verbosity
                    **kwargs
                )
                results.append(result)
            except Exception as e:
                if verbose:
                    print(f"Local optimization {i+1} failed: {e}")
                results.append(None)
        
        return results
    
    def _optimize_parallel(
        self,
        starting_points: List[Array],
        bounds: List[Tuple[float, float]],
        constraints: Optional[List[Dict[str, Any]]],
        max_iterations: int,
        tolerance: float,
        verbose: bool,
        **kwargs
    ) -> List[Optional[OptimizationResult]]:
        """Run local optimizations in parallel.
        
        Args:
            starting_points: List of starting points.
            bounds: Optimization bounds.
            constraints: Constraints.
            max_iterations: Max iterations per optimization.
            tolerance: Convergence tolerance.
            verbose: Verbosity flag.
            **kwargs: Additional parameters.
            
        Returns:
            List of optimization results.
        """
        def optimize_single(x0):
            """Single optimization task."""
            try:
                # Create a new optimizer instance for thread safety
                local_opt = type(self.local_optimizer)(
                    surrogate=self.surrogate,
                    **self.local_optimizer.__dict__
                )
                
                return local_opt.optimize(
                    x0=x0,
                    bounds=bounds,
                    constraints=constraints,
                    max_iterations=max_iterations,
                    tolerance=tolerance,
                    verbose=False,
                    **kwargs
                )
            except Exception as e:
                if verbose:
                    print(f"Local optimization failed: {e}")
                return None
        
        # Use ThreadPoolExecutor for JAX compatibility
        max_workers = self.n_jobs if self.n_jobs > 0 else None
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(optimize_single, starting_points))
        
        return results
    
    def _cluster_results(
        self, 
        results: List[OptimizationResult]
    ) -> List[OptimizationResult]:
        """Cluster optimization results to find unique local optima.
        
        Args:
            results: List of optimization results.
            
        Returns:
            List of unique local optima.
        """
        if len(results) <= 1:
            return results
        
        # Extract optimal points
        points = jnp.stack([r.x_opt for r in results])
        
        # Simple distance-based clustering
        clustered = []
        used = set()
        
        for i, result in enumerate(results):
            if i in used:
                continue
            
            # Start new cluster with this point
            cluster_points = [i]
            used.add(i)
            
            # Find nearby points
            for j in range(i + 1, len(results)):
                if j in used:
                    continue
                
                distance = jnp.linalg.norm(points[i] - points[j])
                if distance < self.clustering_threshold:
                    cluster_points.append(j)
                    used.add(j)
            
            # Take best result from cluster
            cluster_results = [results[idx] for idx in cluster_points]
            best_in_cluster = min(cluster_results, key=lambda r: r.f_opt)
            clustered.append(best_in_cluster)
        
        # Sort by function value
        clustered.sort(key=lambda r: r.f_opt)
        
        return clustered


class GlobalOptimizationResult:
    """Results from global multi-start optimization."""
    
    def __init__(
        self,
        best_result: OptimizationResult,
        all_results: List[OptimizationResult],
        clustered_results: List[OptimizationResult],
        starting_points: List[Array],
        n_starts: int,
        n_successful: int,
        optimization_time: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize global optimization result.
        
        Args:
            best_result: Best optimization result found.
            all_results: All successful optimization results.
            clustered_results: Unique local optima after clustering.
            starting_points: Starting points used.
            n_starts: Total number of starts.
            n_successful: Number of successful optimizations.
            optimization_time: Total optimization time.
            metadata: Additional metadata.
        """
        self.best_result = best_result
        self.all_results = all_results
        self.clustered_results = clustered_results
        self.starting_points = starting_points
        self.n_starts = n_starts
        self.n_successful = n_successful
        self.optimization_time = optimization_time
        self.metadata = metadata or {}
    
    @property
    def best_point(self) -> Array:
        """Best point found."""
        return self.best_result.x_opt
    
    @property
    def best_value(self) -> float:
        """Best function value found."""
        return self.best_result.f_opt
    
    @property
    def success_rate(self) -> float:
        """Success rate of local optimizations."""
        return self.n_successful / self.n_starts
    
    @property
    def n_unique_optima(self) -> int:
        """Number of unique local optima found."""
        return len(self.clustered_results)
    
    def summary(self) -> str:
        """Generate summary string."""
        return (
            f"Global Optimization Summary:\n"
            f"Best value: {self.best_value:.6f}\n"
            f"Best point: {self.best_point}\n"
            f"Success rate: {self.success_rate:.1%} ({self.n_successful}/{self.n_starts})\n"
            f"Unique optima: {self.n_unique_optima}\n"
            f"Total time: {self.optimization_time:.2f}s"
        )