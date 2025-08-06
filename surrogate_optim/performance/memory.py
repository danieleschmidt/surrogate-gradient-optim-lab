"""Memory optimization utilities."""

import gc
import psutil
import sys
import time
import warnings
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

import jax.numpy as jnp
from jax import Array

from ..monitoring.logging import get_logger


class MemoryMonitor:
    """Monitor memory usage during operations."""
    
    def __init__(self):
        """Initialize memory monitor."""
        self.logger = get_logger()
        self.process = psutil.Process()
        self.initial_memory = None
        self.peak_memory = None
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage information.
        
        Returns:
            Dictionary with memory usage statistics in MB
        """
        memory_info = self.process.memory_info()
        
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size
            "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            "percent": self.process.memory_percent(),
            "available_mb": psutil.virtual_memory().available / 1024 / 1024,
        }
    
    def start_monitoring(self):
        """Start memory monitoring."""
        self.initial_memory = self.get_memory_usage()
        self.peak_memory = self.initial_memory.copy()
    
    def update_peak(self):
        """Update peak memory usage."""
        current = self.get_memory_usage()
        if current["rss_mb"] > self.peak_memory["rss_mb"]:
            self.peak_memory = current
    
    def get_summary(self) -> Dict[str, Any]:
        """Get memory usage summary.
        
        Returns:
            Summary of memory usage during monitoring
        """
        if self.initial_memory is None:
            return {"error": "Monitoring not started"}
        
        current = self.get_memory_usage()
        
        return {
            "initial_mb": self.initial_memory["rss_mb"],
            "current_mb": current["rss_mb"],
            "peak_mb": self.peak_memory["rss_mb"],
            "increase_mb": current["rss_mb"] - self.initial_memory["rss_mb"],
            "peak_increase_mb": self.peak_memory["rss_mb"] - self.initial_memory["rss_mb"],
            "available_mb": current["available_mb"],
        }


@contextmanager
def memory_monitor(operation_name: str = "operation") -> Generator[MemoryMonitor, None, None]:
    """Context manager for monitoring memory usage.
    
    Args:
        operation_name: Name of the operation being monitored
        
    Yields:
        MemoryMonitor instance
    """
    monitor = MemoryMonitor()
    monitor.start_monitoring()
    
    try:
        yield monitor
    finally:
        summary = monitor.get_summary()
        logger = get_logger()
        
        logger.info(
            f"Memory usage for {operation_name}: "
            f"peak={summary['peak_mb']:.1f}MB, "
            f"increase={summary['increase_mb']:.1f}MB"
        )
        
        # Warn if memory usage is high
        if summary["peak_mb"] > 1000:  # > 1GB
            warnings.warn(
                f"High memory usage detected: {summary['peak_mb']:.1f}MB",
                UserWarning
            )


class MemoryOptimizer:
    """Utilities for optimizing memory usage."""
    
    @staticmethod
    def optimize_array_memory(arrays: list, copy_threshold_mb: float = 100.0) -> list:
        """Optimize memory usage of array operations.
        
        Args:
            arrays: List of arrays to optimize
            copy_threshold_mb: Threshold for making copies vs. in-place operations
            
        Returns:
            Optimized arrays
        """
        optimized = []
        
        for arr in arrays:
            if isinstance(arr, jnp.ndarray):
                # Estimate memory usage
                size_mb = arr.nbytes / 1024 / 1024
                
                # Convert to smaller dtype if possible
                if size_mb > copy_threshold_mb:
                    if arr.dtype == jnp.float64:
                        # Try to use float32 instead
                        arr_f32 = arr.astype(jnp.float32)
                        if jnp.allclose(arr, arr_f32, rtol=1e-6):
                            arr = arr_f32
                
                optimized.append(arr)
            else:
                optimized.append(arr)
        
        return optimized
    
    @staticmethod
    def cleanup_memory():
        """Force garbage collection and memory cleanup."""
        gc.collect()
        
        # JAX-specific cleanup if available
        try:
            import jax
            jax.clear_backends()
        except:
            pass
    
    @staticmethod
    def estimate_memory_requirement(
        n_samples: int,
        n_dims: int,
        dtype=jnp.float32,
        safety_factor: float = 2.0,
    ) -> float:
        """Estimate memory requirements for dataset operations.
        
        Args:
            n_samples: Number of samples
            n_dims: Number of dimensions
            dtype: Data type
            safety_factor: Safety factor for estimation
            
        Returns:
            Estimated memory requirement in MB
        """
        # Size of one array element
        element_size = jnp.dtype(dtype).itemsize
        
        # Estimate arrays needed:
        # X: n_samples x n_dims
        # y: n_samples
        # gradients: n_samples x n_dims (optional)
        # Various intermediate arrays
        
        X_size = n_samples * n_dims * element_size
        y_size = n_samples * element_size
        grad_size = n_samples * n_dims * element_size
        
        total_bytes = (X_size + y_size + grad_size) * safety_factor
        
        return total_bytes / 1024 / 1024  # Convert to MB
    
    @staticmethod
    def check_memory_availability(required_mb: float) -> bool:
        """Check if sufficient memory is available.
        
        Args:
            required_mb: Required memory in MB
            
        Returns:
            True if sufficient memory is available
        """
        available = psutil.virtual_memory().available / 1024 / 1024
        return available > required_mb * 1.2  # 20% buffer


class BatchProcessor:
    """Process large datasets in memory-efficient batches."""
    
    def __init__(
        self,
        batch_size: Optional[int] = None,
        max_memory_mb: float = 1000.0,
        adaptive_batching: bool = True,
    ):
        """Initialize batch processor.
        
        Args:
            batch_size: Fixed batch size (None for adaptive)
            max_memory_mb: Maximum memory to use for batching
            adaptive_batching: Whether to adapt batch size based on memory
        """
        self.batch_size = batch_size
        self.max_memory_mb = max_memory_mb
        self.adaptive_batching = adaptive_batching
        self.logger = get_logger()
        self.memory_monitor = MemoryMonitor()
    
    def calculate_batch_size(
        self,
        total_samples: int,
        sample_dims: int,
        dtype=jnp.float32,
    ) -> int:
        """Calculate optimal batch size based on memory constraints.
        
        Args:
            total_samples: Total number of samples
            sample_dims: Dimensions per sample
            dtype: Data type
            
        Returns:
            Optimal batch size
        """
        if self.batch_size is not None and not self.adaptive_batching:
            return min(self.batch_size, total_samples)
        
        # Estimate memory per sample
        element_size = jnp.dtype(dtype).itemsize
        memory_per_sample = sample_dims * element_size * 3  # X, y, intermediate
        memory_per_sample_mb = memory_per_sample / 1024 / 1024
        
        # Calculate batch size that fits in memory limit
        max_batch_size = int(self.max_memory_mb / memory_per_sample_mb)
        max_batch_size = max(1, max_batch_size)  # At least 1
        
        # Use smaller of calculated size and total samples
        optimal_batch_size = min(max_batch_size, total_samples)
        
        if self.batch_size is not None:
            optimal_batch_size = min(self.batch_size, optimal_batch_size)
        
        self.logger.debug(f"Calculated batch size: {optimal_batch_size}")
        
        return optimal_batch_size
    
    def process_batches(
        self,
        data: Array,
        processor_func: callable,
        batch_size: Optional[int] = None,
        progress_callback: Optional[callable] = None,
    ) -> Array:
        """Process data in batches.
        
        Args:
            data: Input data array
            processor_func: Function to apply to each batch
            batch_size: Override batch size
            progress_callback: Optional progress callback
            
        Returns:
            Processed results
        """
        n_samples = data.shape[0]
        
        if batch_size is None:
            batch_size = self.calculate_batch_size(
                n_samples, data.shape[1] if data.ndim > 1 else 1
            )
        
        # Process batches
        results = []
        
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch = data[i:end_idx]
            
            # Monitor memory during processing
            self.memory_monitor.update_peak()
            
            # Process batch
            batch_result = processor_func(batch)
            results.append(batch_result)
            
            # Progress callback
            if progress_callback:
                progress = (i + batch_size) / n_samples
                progress_callback(min(progress, 1.0))
            
            # Adaptive batch size adjustment
            if self.adaptive_batching and i > 0:
                memory_usage = self.memory_monitor.get_memory_usage()
                if memory_usage["rss_mb"] > self.max_memory_mb * 0.8:
                    # Reduce batch size if memory usage is high
                    batch_size = max(1, int(batch_size * 0.8))
                    self.logger.debug(f"Reduced batch size to {batch_size}")
        
        # Combine results
        if isinstance(results[0], jnp.ndarray):
            return jnp.concatenate(results, axis=0)
        else:
            return results
    
    def process_dataset_batches(
        self,
        dataset,
        processor_func: callable,
        include_gradients: bool = False,
    ):
        """Process entire dataset in batches.
        
        Args:
            dataset: Dataset to process
            processor_func: Function to apply
            include_gradients: Whether to include gradients in processing
            
        Returns:
            Processed dataset
        """
        from ..models.base import Dataset
        
        # Calculate batch size
        batch_size = self.calculate_batch_size(
            dataset.n_samples, dataset.n_dims
        )
        
        self.logger.info(f"Processing dataset in batches of {batch_size}")
        
        # Process X and y in batches
        processed_X = self.process_batches(dataset.X, processor_func, batch_size)
        
        # For y, we might need a different processor
        if hasattr(processor_func, 'process_targets'):
            processed_y = self.process_batches(
                dataset.y, processor_func.process_targets, batch_size
            )
        else:
            processed_y = dataset.y
        
        # Handle gradients if present and requested
        processed_gradients = None
        if include_gradients and dataset.gradients is not None:
            if hasattr(processor_func, 'process_gradients'):
                processed_gradients = self.process_batches(
                    dataset.gradients, processor_func.process_gradients, batch_size
                )
            else:
                processed_gradients = dataset.gradients
        
        return Dataset(
            X=processed_X,
            y=processed_y,
            gradients=processed_gradients,
            metadata=dataset.metadata,
        )


class MemoryEfficientSurrogate:
    """Memory-efficient wrapper for surrogate models."""
    
    def __init__(
        self,
        surrogate,
        batch_size: Optional[int] = None,
        enable_checkpointing: bool = True,
        checkpoint_frequency: int = 100,
    ):
        """Initialize memory-efficient surrogate.
        
        Args:
            surrogate: Base surrogate model
            batch_size: Batch size for operations
            enable_checkpointing: Whether to enable gradient checkpointing
            checkpoint_frequency: Frequency of checkpointing operations
        """
        self.surrogate = surrogate
        self.batch_processor = BatchProcessor(batch_size)
        self.enable_checkpointing = enable_checkpointing
        self.checkpoint_frequency = checkpoint_frequency
        self.logger = get_logger()
    
    def fit(self, dataset):
        """Fit surrogate with memory optimization."""
        with memory_monitor("surrogate_training") as monitor:
            # Check if dataset is too large for memory
            estimated_memory = MemoryOptimizer.estimate_memory_requirement(
                dataset.n_samples, dataset.n_dims
            )
            
            if estimated_memory > 500:  # > 500MB
                self.logger.info(f"Large dataset detected ({estimated_memory:.1f}MB), using batch training")
                return self._fit_batched(dataset)
            else:
                return self.surrogate.fit(dataset)
    
    def _fit_batched(self, dataset):
        """Fit surrogate using batched training."""
        # For neural networks, we can use batch training
        if hasattr(self.surrogate, '_fit_batched'):
            return self.surrogate._fit_batched(dataset, self.batch_processor)
        else:
            # For other models, fit on smaller subsample
            self.logger.warning("Surrogate doesn't support batch training, using subsample")
            
            # Create subsample
            max_samples = min(1000, dataset.n_samples)
            indices = jnp.linspace(0, dataset.n_samples - 1, max_samples, dtype=int)
            
            from ..models.base import Dataset
            subsample = Dataset(
                X=dataset.X[indices],
                y=dataset.y[indices],
                gradients=dataset.gradients[indices] if dataset.gradients is not None else None,
                metadata=dataset.metadata,
            )
            
            return self.surrogate.fit(subsample)
    
    def predict(self, x: Array) -> Array:
        """Predict with memory optimization."""
        if x.ndim == 1:
            return self.surrogate.predict(x)
        
        # For large batch predictions, use batch processing
        if x.shape[0] > 100:
            return self.batch_processor.process_batches(
                x, lambda batch: self.surrogate.predict(batch)
            )
        else:
            return self.surrogate.predict(x)
    
    def gradient(self, x: Array) -> Array:
        """Compute gradient with memory optimization."""
        if x.ndim == 1:
            return self.surrogate.gradient(x)
        
        # For large batch gradients, use batch processing
        if x.shape[0] > 50:  # Gradients are more memory-intensive
            return self.batch_processor.process_batches(
                x, lambda batch: self.surrogate.gradient(batch)
            )
        else:
            return self.surrogate.gradient(x)
    
    def __getattr__(self, name):
        """Delegate other attributes to base surrogate."""
        return getattr(self.surrogate, name)