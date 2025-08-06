"""Parallel processing utilities for surrogate optimization."""

import concurrent.futures
import multiprocessing as mp
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax.numpy as jnp
from jax import Array, vmap

from ..data.collector import DataCollector
from ..monitoring.logging import get_logger
from ..optimizers.base import BaseOptimizer, OptimizationResult


class ParallelDataCollector:
    """Parallel data collector for faster function evaluations."""
    
    def __init__(
        self,
        function: Callable[[Array], float],
        bounds: List[Tuple[float, float]],
        n_workers: Optional[int] = None,
        batch_size: Optional[int] = None,
        timeout: Optional[float] = None,
    ):
        """Initialize parallel data collector.
        
        Args:
            function: Black-box function to sample
            bounds: Bounds for each input dimension
            n_workers: Number of parallel workers (None for auto)
            batch_size: Batch size for parallel evaluation
            timeout: Timeout for individual function evaluations
        """
        self.function = function
        self.bounds = bounds
        self.n_workers = n_workers or min(mp.cpu_count(), 8)
        self.batch_size = batch_size or max(1, 100 // self.n_workers)
        self.timeout = timeout
        self.logger = get_logger()
        
        # Base collector for generating samples
        self.base_collector = DataCollector(function, bounds)
    
    def collect_parallel(
        self,
        n_samples: int,
        sampling: str = "sobol",
        estimate_gradients: bool = False,
        verbose: bool = True,
    ):
        """Collect data using parallel function evaluations.
        
        Args:
            n_samples: Number of samples to collect
            sampling: Sampling strategy
            estimate_gradients: Whether to estimate gradients
            verbose: Whether to print progress
            
        Returns:
            Dataset with collected samples
        """
        if verbose:
            self.logger.info(f"Starting parallel data collection: {n_samples} samples, {self.n_workers} workers")
        
        start_time = time.time()
        
        # Generate all sample points
        X = self.base_collector._generate_samples(n_samples, sampling)
        
        # Evaluate in parallel
        y = self._evaluate_parallel(X, verbose)
        
        # Estimate gradients if requested
        gradients = None
        if estimate_gradients:
            if verbose:
                self.logger.info("Estimating gradients in parallel...")
            gradients = self._estimate_gradients_parallel(X, verbose)
        
        # Create dataset
        from ..models.base import Dataset
        dataset = Dataset(
            X=X,
            y=y,
            gradients=gradients,
            metadata={
                "sampling_method": sampling,
                "n_samples": n_samples,
                "bounds": self.bounds,
                "parallel_workers": self.n_workers,
                "collection_time": time.time() - start_time,
            }
        )
        
        if verbose:
            duration = time.time() - start_time
            self.logger.info(f"Parallel data collection completed in {duration:.2f}s")
        
        return dataset
    
    def _evaluate_parallel(self, X: Array, verbose: bool) -> Array:
        """Evaluate function on all points in parallel."""
        n_samples = X.shape[0]
        y = jnp.zeros(n_samples)
        
        # Create batches
        batches = []
        for i in range(0, n_samples, self.batch_size):
            end_idx = min(i + self.batch_size, n_samples)
            batches.append((i, end_idx, X[i:end_idx]))
        
        # Process batches in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(self._evaluate_batch, batch_X, batch_idx): (start_idx, end_idx)
                for batch_idx, (start_idx, end_idx, batch_X) in enumerate(batches)
            }
            
            # Collect results
            completed = 0
            for future in concurrent.futures.as_completed(future_to_batch, timeout=self.timeout):
                start_idx, end_idx = future_to_batch[future]
                
                try:
                    batch_results = future.result()
                    y = y.at[start_idx:end_idx].set(batch_results)
                    completed += 1
                    
                    if verbose and len(batches) > 1:
                        progress = completed / len(batches) * 100
                        self.logger.info(f"  Evaluation progress: {progress:.1f}%")
                        
                except Exception as e:
                    self.logger.warning(f"Batch evaluation failed: {e}")
                    # Fill with NaN for failed evaluations
                    y = y.at[start_idx:end_idx].set(jnp.nan)
        
        return y
    
    def _evaluate_batch(self, batch_X: Array, batch_idx: int) -> Array:
        """Evaluate function on a batch of points."""
        batch_results = []
        
        for x in batch_X:
            try:
                result = self.function(x)
                batch_results.append(float(result))
            except Exception:
                batch_results.append(jnp.nan)
        
        return jnp.array(batch_results)
    
    def _estimate_gradients_parallel(self, X: Array, verbose: bool) -> Array:
        """Estimate gradients in parallel using finite differences."""
        n_samples, n_dims = X.shape
        gradients = jnp.zeros((n_samples, n_dims))
        
        # Create gradient estimation tasks
        tasks = []
        for i in range(n_samples):
            tasks.append((i, X[i]))
        
        # Process in parallel
        batch_size = max(1, len(tasks) // self.n_workers)
        batches = [tasks[i:i+batch_size] for i in range(0, len(tasks), batch_size)]
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            future_to_indices = {}
            
            for batch in batches:
                future = executor.submit(self._estimate_gradient_batch, batch)
                indices = [task[0] for task in batch]
                future_to_indices[future] = indices
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_indices, timeout=self.timeout):
                indices = future_to_indices[future]
                
                try:
                    batch_gradients = future.result()
                    for idx, grad in zip(indices, batch_gradients):
                        gradients = gradients.at[idx].set(grad)
                except Exception as e:
                    self.logger.warning(f"Gradient batch failed: {e}")
                    # Fill with NaN for failed gradients
                    for idx in indices:
                        gradients = gradients.at[idx].set(jnp.full(n_dims, jnp.nan))
        
        return gradients
    
    def _estimate_gradient_batch(self, tasks: List[Tuple[int, Array]]) -> List[Array]:
        """Estimate gradients for a batch of points."""
        batch_gradients = []
        eps = 1e-6
        
        for _, x in tasks:
            try:
                grad = jnp.zeros(len(x))
                
                for j in range(len(x)):
                    x_plus = x.at[j].add(eps)
                    x_minus = x.at[j].add(-eps)
                    
                    # Ensure within bounds
                    lower, upper = self.bounds[j]
                    x_plus = x_plus.at[j].set(jnp.clip(x_plus[j], lower, upper))
                    x_minus = x_minus.at[j].set(jnp.clip(x_minus[j], lower, upper))
                    
                    f_plus = self.function(x_plus)
                    f_minus = self.function(x_minus)
                    
                    grad_j = (f_plus - f_minus) / (2 * eps)
                    grad = grad.at[j].set(grad_j)
                
                batch_gradients.append(grad)
                
            except Exception:
                batch_gradients.append(jnp.full(len(x), jnp.nan))
        
        return batch_gradients


class ParallelOptimizer:
    """Parallel optimizer for running multiple optimization strategies."""
    
    def __init__(
        self,
        optimizers: List[BaseOptimizer],
        n_workers: Optional[int] = None,
        timeout: Optional[float] = None,
    ):
        """Initialize parallel optimizer.
        
        Args:
            optimizers: List of optimizers to run in parallel
            n_workers: Number of parallel workers
            timeout: Timeout for individual optimizations
        """
        self.optimizers = optimizers
        self.n_workers = n_workers or min(len(optimizers), mp.cpu_count())
        self.timeout = timeout
        self.logger = get_logger()
    
    def optimize_parallel(
        self,
        surrogate,
        x0_list: List[Array],
        bounds: Optional[List[Tuple[float, float]]] = None,
        verbose: bool = True,
    ) -> List[OptimizationResult]:
        """Run multiple optimizations in parallel.
        
        Args:
            surrogate: Surrogate model to optimize
            x0_list: List of initial points (one per optimizer)
            bounds: Optional bounds
            verbose: Whether to print progress
            
        Returns:
            List of optimization results
        """
        if len(x0_list) != len(self.optimizers):
            raise ValueError(f"Need {len(self.optimizers)} initial points, got {len(x0_list)}")
        
        if verbose:
            self.logger.info(f"Starting {len(self.optimizers)} parallel optimizations")
        
        start_time = time.time()
        results = []
        
        # Create optimization tasks
        tasks = []
        for i, (optimizer, x0) in enumerate(zip(self.optimizers, x0_list)):
            tasks.append((i, optimizer, surrogate, x0, bounds))
        
        # Run in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            future_to_task = {
                executor.submit(self._run_optimization, task): i
                for i, task in enumerate(tasks)
            }
            
            # Collect results in order
            results = [None] * len(tasks)
            
            for future in concurrent.futures.as_completed(future_to_task, timeout=self.timeout):
                task_idx = future_to_task[future]
                
                try:
                    result = future.result()
                    results[task_idx] = result
                    
                    if verbose:
                        completed = sum(1 for r in results if r is not None)
                        progress = completed / len(tasks) * 100
                        self.logger.info(f"  Optimization progress: {progress:.1f}%")
                        
                except Exception as e:
                    self.logger.warning(f"Optimization {task_idx} failed: {e}")
                    # Create failure result
                    results[task_idx] = OptimizationResult(
                        x=x0_list[task_idx],
                        fun=float('inf'),
                        success=False,
                        message=f"Parallel execution failed: {e}",
                        nit=0,
                        nfev=0,
                    )
        
        duration = time.time() - start_time
        successful = sum(1 for r in results if r and r.success)
        
        if verbose:
            self.logger.info(f"Parallel optimization completed: {successful}/{len(results)} successful in {duration:.2f}s")
        
        return results
    
    def _run_optimization(self, task: Tuple) -> OptimizationResult:
        """Run single optimization task."""
        task_idx, optimizer, surrogate, x0, bounds = task
        
        try:
            return optimizer.optimize(surrogate, x0, bounds)
        except Exception as e:
            return OptimizationResult(
                x=x0,
                fun=float('inf'),
                success=False,
                message=f"Optimization failed: {e}",
                nit=0,
                nfev=0,
            )


class VectorizedSurrogate:
    """Vectorized wrapper for efficient batch operations."""
    
    def __init__(self, surrogate):
        """Initialize vectorized surrogate.
        
        Args:
            surrogate: Base surrogate model
        """
        self.surrogate = surrogate
        
        # Create vectorized versions of predict and gradient
        self._vectorized_predict = None
        self._vectorized_gradient = None
        self._setup_vectorized_functions()
    
    def _setup_vectorized_functions(self):
        """Setup vectorized functions using JAX vmap."""
        try:
            # Try to create vectorized predict
            self._vectorized_predict = vmap(self.surrogate.predict)
        except:
            # Fallback to manual vectorization
            self._vectorized_predict = self._manual_vectorized_predict
        
        try:
            # Try to create vectorized gradient
            self._vectorized_gradient = vmap(self.surrogate.gradient)
        except:
            # Fallback to manual vectorization
            self._vectorized_gradient = self._manual_vectorized_gradient
    
    def _manual_vectorized_predict(self, X: Array) -> Array:
        """Manual vectorization for predict."""
        predictions = []
        for x in X:
            predictions.append(self.surrogate.predict(x))
        return jnp.stack(predictions)
    
    def _manual_vectorized_gradient(self, X: Array) -> Array:
        """Manual vectorization for gradient."""
        gradients = []
        for x in X:
            gradients.append(self.surrogate.gradient(x))
        return jnp.stack(gradients)
    
    def predict(self, x: Array) -> Array:
        """Predict with automatic vectorization."""
        if x.ndim == 1:
            return self.surrogate.predict(x)
        else:
            return self._vectorized_predict(x)
    
    def gradient(self, x: Array) -> Array:
        """Compute gradient with automatic vectorization."""
        if x.ndim == 1:
            return self.surrogate.gradient(x)
        else:
            return self._vectorized_gradient(x)
    
    def uncertainty(self, x: Array) -> Array:
        """Compute uncertainty (manual vectorization)."""
        if x.ndim == 1:
            return self.surrogate.uncertainty(x)
        else:
            uncertainties = []
            for point in x:
                uncertainties.append(self.surrogate.uncertainty(point))
            return jnp.stack(uncertainties)
    
    def __getattr__(self, name):
        """Delegate other attributes to base surrogate."""
        return getattr(self.surrogate, name)


class BatchOptimizer:
    """Optimizer that processes multiple points in batches."""
    
    def __init__(
        self,
        base_optimizer: BaseOptimizer,
        batch_size: int = 32,
        enable_vectorization: bool = True,
    ):
        """Initialize batch optimizer.
        
        Args:
            base_optimizer: Base optimizer to wrap
            batch_size: Batch size for operations
            enable_vectorization: Whether to use vectorized operations
        """
        self.base_optimizer = base_optimizer
        self.batch_size = batch_size
        self.enable_vectorization = enable_vectorization
        self.logger = get_logger()
    
    def optimize(self, surrogate, x0: Array, bounds=None, **kwargs) -> OptimizationResult:
        """Optimize using batched operations where possible."""
        # Wrap surrogate for vectorization if enabled
        if self.enable_vectorization:
            surrogate = VectorizedSurrogate(surrogate)
        
        # Use base optimizer
        return self.base_optimizer.optimize(surrogate, x0, bounds, **kwargs)
    
    def optimize_multiple(
        self,
        surrogate,
        x0_list: List[Array],
        bounds=None,
        **kwargs
    ) -> List[OptimizationResult]:
        """Optimize multiple starting points efficiently."""
        if self.enable_vectorization:
            surrogate = VectorizedSurrogate(surrogate)
        
        results = []
        
        # Process in batches
        for i in range(0, len(x0_list), self.batch_size):
            batch_x0 = x0_list[i:i + self.batch_size]
            batch_results = []
            
            for x0 in batch_x0:
                result = self.base_optimizer.optimize(surrogate, x0, bounds, **kwargs)
                batch_results.append(result)
            
            results.extend(batch_results)
            
            if len(x0_list) > self.batch_size:
                progress = min(i + self.batch_size, len(x0_list)) / len(x0_list) * 100
                self.logger.debug(f"Batch optimization progress: {progress:.1f}%")
        
        return results