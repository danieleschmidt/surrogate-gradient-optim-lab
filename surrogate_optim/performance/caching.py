"""Caching utilities for performance optimization."""

import functools
import hashlib
import pickle
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import jax.numpy as jnp
from jax import Array

from ..monitoring.logging import get_logger


class LRUCache:
    """Least Recently Used cache with size limits."""
    
    def __init__(self, maxsize: int = 128):
        """Initialize LRU cache.
        
        Args:
            maxsize: Maximum number of items to cache
        """
        self.maxsize = maxsize
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Any:
        """Get item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        else:
            self.misses += 1
            return None
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        if key in self.cache:
            # Update existing
            self.cache.move_to_end(key)
        else:
            # Add new
            if len(self.cache) >= self.maxsize:
                # Remove oldest
                self.cache.popitem(last=False)
        
        self.cache[key] = value
    
    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "size": len(self.cache),
            "maxsize": self.maxsize,
        }


class PersistentCache:
    """Persistent cache that saves to disk."""
    
    def __init__(
        self,
        cache_dir: Union[str, Path],
        max_age_seconds: Optional[float] = None,
        compression: bool = True,
    ):
        """Initialize persistent cache.
        
        Args:
            cache_dir: Directory to store cache files
            max_age_seconds: Maximum age of cached items in seconds
            compression: Whether to compress cached data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_age = max_age_seconds
        self.compression = compression
        self.logger = get_logger()
    
    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for key."""
        safe_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{safe_key}.cache"
    
    def get(self, key: str) -> Any:
        """Get item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            return None
        
        # Check age
        if self.max_age is not None:
            age = time.time() - cache_path.stat().st_mtime
            if age > self.max_age:
                cache_path.unlink()  # Remove expired
                return None
        
        try:
            with open(cache_path, "rb") as f:
                if self.compression:
                    import gzip
                    data = gzip.decompress(f.read())
                    return pickle.loads(data)
                else:
                    return pickle.load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load cache for key {key}: {e}")
            cache_path.unlink()  # Remove corrupted
            return None
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        cache_path = self._get_cache_path(key)
        
        try:
            with open(cache_path, "wb") as f:
                if self.compression:
                    import gzip
                    data = gzip.compress(pickle.dumps(value))
                    f.write(data)
                else:
                    pickle.dump(value, f)
        except Exception as e:
            self.logger.warning(f"Failed to cache key {key}: {e}")
    
    def clear(self) -> None:
        """Clear the cache."""
        for cache_file in self.cache_dir.glob("*.cache"):
            try:
                cache_file.unlink()
            except Exception as e:
                self.logger.warning(f"Failed to remove cache file {cache_file}: {e}")
    
    def cleanup_expired(self) -> int:
        """Remove expired cache files.
        
        Returns:
            Number of files removed
        """
        if self.max_age is None:
            return 0
        
        removed_count = 0
        current_time = time.time()
        
        for cache_file in self.cache_dir.glob("*.cache"):
            try:
                age = current_time - cache_file.stat().st_mtime
                if age > self.max_age:
                    cache_file.unlink()
                    removed_count += 1
            except Exception as e:
                self.logger.warning(f"Failed to check/remove cache file {cache_file}: {e}")
        
        return removed_count


class FunctionCache:
    """Decorator for caching function results."""
    
    def __init__(
        self,
        cache_type: str = "memory",
        maxsize: int = 128,
        cache_dir: Optional[str] = None,
        max_age_seconds: Optional[float] = None,
        key_func: Optional[Callable] = None,
    ):
        """Initialize function cache.
        
        Args:
            cache_type: Type of cache ("memory" or "disk")
            maxsize: Maximum cache size for memory cache
            cache_dir: Cache directory for disk cache
            max_age_seconds: Maximum age of cached items
            key_func: Custom key generation function
        """
        self.cache_type = cache_type
        self.key_func = key_func or self._default_key_func
        
        if cache_type == "memory":
            self.cache = LRUCache(maxsize)
        elif cache_type == "disk":
            if cache_dir is None:
                cache_dir = ".surrogate_cache"
            self.cache = PersistentCache(cache_dir, max_age_seconds)
        else:
            raise ValueError(f"Unknown cache type: {cache_type}")
    
    def _default_key_func(self, func: Callable, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function and arguments."""
        # Convert JAX arrays to strings for hashing
        processed_args = []
        for arg in args:
            if isinstance(arg, jnp.ndarray):
                processed_args.append(f"array_{hash(arg.tobytes())}")
            else:
                processed_args.append(str(arg))
        
        processed_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, jnp.ndarray):
                processed_kwargs[k] = f"array_{hash(v.tobytes())}"
            else:
                processed_kwargs[k] = str(v)
        
        key_parts = [func.__name__] + processed_args + [str(processed_kwargs)]
        key_str = "_".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def __call__(self, func: Callable) -> Callable:
        """Decorate function with caching."""
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = self.key_func(func, args, kwargs)
            
            # Try to get from cache
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache result
            self.cache.put(cache_key, result)
            
            return result
        
        # Add cache management methods
        wrapper.cache_clear = self.cache.clear
        wrapper.cache_stats = getattr(self.cache, "stats", lambda: {})
        
        return wrapper


def cached_function(
    cache_type: str = "memory",
    maxsize: int = 128,
    cache_dir: Optional[str] = None,
    max_age_seconds: Optional[float] = None,
) -> Callable:
    """Decorator for caching function results.
    
    Args:
        cache_type: Type of cache ("memory" or "disk")
        maxsize: Maximum cache size for memory cache
        cache_dir: Cache directory for disk cache
        max_age_seconds: Maximum age of cached items
        
    Returns:
        Function decorator
    """
    return FunctionCache(cache_type, maxsize, cache_dir, max_age_seconds)


class SurrogateCache:
    """Specialized cache for surrogate model predictions."""
    
    def __init__(
        self,
        tolerance: float = 1e-8,
        maxsize: int = 1000,
        enable_gradient_cache: bool = True,
    ):
        """Initialize surrogate cache.
        
        Args:
            tolerance: Tolerance for considering points "close enough"
            maxsize: Maximum number of cached predictions
            enable_gradient_cache: Whether to cache gradients too
        """
        self.tolerance = tolerance
        self.enable_gradient_cache = enable_gradient_cache
        
        self.prediction_cache = LRUCache(maxsize)
        self.gradient_cache = LRUCache(maxsize) if enable_gradient_cache else None
        
        # Spatial index for fast nearest neighbor lookup
        self.cached_points = []
        self.cached_keys = []
    
    def _find_nearest_cached_point(self, x: Array) -> Optional[str]:
        """Find nearest cached point within tolerance.
        
        Args:
            x: Query point
            
        Returns:
            Cache key of nearest point or None
        """
        if not self.cached_points:
            return None
        
        # Compute distances to all cached points
        distances = [jnp.linalg.norm(x - point) for point in self.cached_points]
        min_distance = min(distances)
        
        if min_distance <= self.tolerance:
            min_idx = distances.index(min_distance)
            return self.cached_keys[min_idx]
        
        return None
    
    def get_prediction(self, x: Array) -> Optional[float]:
        """Get cached prediction for point.
        
        Args:
            x: Input point
            
        Returns:
            Cached prediction or None
        """
        cache_key = self._find_nearest_cached_point(x)
        if cache_key is not None:
            return self.prediction_cache.get(cache_key)
        return None
    
    def get_gradient(self, x: Array) -> Optional[Array]:
        """Get cached gradient for point.
        
        Args:
            x: Input point
            
        Returns:
            Cached gradient or None
        """
        if self.gradient_cache is None:
            return None
        
        cache_key = self._find_nearest_cached_point(x)
        if cache_key is not None:
            return self.gradient_cache.get(cache_key)
        return None
    
    def put_prediction(self, x: Array, prediction: float) -> None:
        """Cache prediction for point.
        
        Args:
            x: Input point
            prediction: Function value
        """
        cache_key = f"pred_{len(self.cached_points)}"
        self.prediction_cache.put(cache_key, prediction)
        
        self.cached_points.append(x.copy())
        self.cached_keys.append(cache_key)
        
        # Limit spatial index size
        if len(self.cached_points) > self.prediction_cache.maxsize:
            self.cached_points.pop(0)
            self.cached_keys.pop(0)
    
    def put_gradient(self, x: Array, gradient: Array) -> None:
        """Cache gradient for point.
        
        Args:
            x: Input point
            gradient: Gradient vector
        """
        if self.gradient_cache is None:
            return
        
        # Find corresponding prediction cache key
        cache_key = self._find_nearest_cached_point(x)
        if cache_key is not None:
            grad_key = cache_key.replace("pred_", "grad_")
            self.gradient_cache.put(grad_key, gradient)
    
    def clear(self) -> None:
        """Clear all caches."""
        self.prediction_cache.clear()
        if self.gradient_cache:
            self.gradient_cache.clear()
        self.cached_points.clear()
        self.cached_keys.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pred_stats = self.prediction_cache.stats()
        stats = {"prediction_cache": pred_stats}
        
        if self.gradient_cache:
            grad_stats = self.gradient_cache.stats()
            stats["gradient_cache"] = grad_stats
        
        stats["spatial_index_size"] = len(self.cached_points)
        
        return stats


class CachedSurrogate:
    """Wrapper that adds caching to any surrogate model."""
    
    def __init__(
        self,
        surrogate,
        cache_tolerance: float = 1e-8,
        cache_size: int = 1000,
        enable_gradient_cache: bool = True,
    ):
        """Initialize cached surrogate.
        
        Args:
            surrogate: Base surrogate model
            cache_tolerance: Tolerance for cache hits
            cache_size: Maximum cache size
            enable_gradient_cache: Whether to cache gradients
        """
        self.surrogate = surrogate
        self.cache = SurrogateCache(cache_tolerance, cache_size, enable_gradient_cache)
        self.logger = get_logger()
    
    def fit(self, dataset):
        """Fit the underlying surrogate."""
        # Clear cache when retraining
        self.cache.clear()
        return self.surrogate.fit(dataset)
    
    def predict(self, x: Array) -> Array:
        """Predict with caching."""
        # Check cache first
        if x.ndim == 1:
            cached_pred = self.cache.get_prediction(x)
            if cached_pred is not None:
                return jnp.array(cached_pred)
            
            # Compute and cache
            prediction = self.surrogate.predict(x)
            self.cache.put_prediction(x, float(prediction))
            return prediction
        else:
            # Batch prediction - check each point
            predictions = []
            
            for point in x:
                cached_pred = self.cache.get_prediction(point)
                if cached_pred is not None:
                    predictions.append(cached_pred)
                else:
                    pred = self.surrogate.predict(point)
                    predictions.append(float(pred))
                    self.cache.put_prediction(point, float(pred))
            
            return jnp.array(predictions)
    
    def gradient(self, x: Array) -> Array:
        """Compute gradient with caching."""
        # Check cache first
        if x.ndim == 1:
            cached_grad = self.cache.get_gradient(x)
            if cached_grad is not None:
                return cached_grad
            
            # Compute and cache
            gradient = self.surrogate.gradient(x)
            self.cache.put_gradient(x, gradient)
            return gradient
        else:
            # Batch gradient computation
            gradients = []
            
            for point in x:
                cached_grad = self.cache.get_gradient(point)
                if cached_grad is not None:
                    gradients.append(cached_grad)
                else:
                    grad = self.surrogate.gradient(point)
                    gradients.append(grad)
                    self.cache.put_gradient(point, grad)
            
            return jnp.stack(gradients)
    
    def uncertainty(self, x: Array) -> Array:
        """Compute uncertainty (not cached due to complexity)."""
        return self.surrogate.uncertainty(x)
    
    def cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.stats()
    
    def clear_cache(self) -> None:
        """Clear all caches."""
        self.cache.clear()
    
    def __getattr__(self, name):
        """Delegate other attributes to underlying surrogate."""
        return getattr(self.surrogate, name)