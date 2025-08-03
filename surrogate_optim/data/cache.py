"""Caching system for expensive function evaluations."""

import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import jax.numpy as jnp
from jax import Array


class FunctionCache:
    """Cache for expensive function evaluations.
    
    Provides in-memory and disk-based caching for black-box function
    evaluations to avoid redundant computations.
    """
    
    def __init__(
        self,
        cache_dir: Union[str, Path] = "cache",
        max_memory_items: int = 1000,
        tolerance: float = 1e-10,
        enable_disk_cache: bool = True,
    ):
        """Initialize function cache.
        
        Args:
            cache_dir: Directory for disk cache
            max_memory_items: Maximum items in memory cache
            tolerance: Tolerance for considering points as equal
            enable_disk_cache: Whether to enable disk caching
        """
        self.cache_dir = Path(cache_dir)
        self.max_memory_items = max_memory_items
        self.tolerance = tolerance
        self.enable_disk_cache = enable_disk_cache
        
        # In-memory cache: hash -> (point, value, timestamp)
        self._memory_cache: Dict[str, Tuple[Array, float, float]] = {}
        
        # Create cache directory
        if self.enable_disk_cache:
            self.cache_dir.mkdir(exist_ok=True)
        
        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "disk_hits": 0,
            "disk_misses": 0,
        }
    
    def _hash_point(self, x: Array) -> str:
        """Generate hash for a point.
        
        Args:
            x: Input point
            
        Returns:
            Hash string
        """
        # Convert to bytes for hashing
        x_bytes = jnp.asarray(x).tobytes()
        return hashlib.md5(x_bytes).hexdigest()
    
    def _find_cached_point(self, x: Array) -> Optional[float]:
        """Find cached value for a point within tolerance.
        
        Args:
            x: Input point to lookup
            
        Returns:
            Cached value if found, None otherwise
        """
        # First check exact hash match
        point_hash = self._hash_point(x)
        if point_hash in self._memory_cache:
            _, value, _ = self._memory_cache[point_hash]
            self.stats["hits"] += 1
            return value
        
        # If tolerance > 0, check for nearby points
        if self.tolerance > 0:
            for cached_point, value, _ in self._memory_cache.values():
                if jnp.linalg.norm(x - cached_point) <= self.tolerance:
                    self.stats["hits"] += 1
                    return value
        
        # Check disk cache if enabled
        if self.enable_disk_cache:
            disk_value = self._disk_lookup(x)
            if disk_value is not None:
                # Add to memory cache
                self._add_to_memory_cache(x, disk_value)
                self.stats["disk_hits"] += 1
                return disk_value
            else:
                self.stats["disk_misses"] += 1
        
        self.stats["misses"] += 1
        return None
    
    def _add_to_memory_cache(self, x: Array, value: float):
        """Add point and value to memory cache.
        
        Args:
            x: Input point
            value: Function value
        """
        # Remove oldest items if cache is full
        if len(self._memory_cache) >= self.max_memory_items:
            # Remove oldest item
            oldest_hash = min(
                self._memory_cache.keys(),
                key=lambda h: self._memory_cache[h][2]  # timestamp
            )
            del self._memory_cache[oldest_hash]
        
        # Add new item
        point_hash = self._hash_point(x)
        self._memory_cache[point_hash] = (x.copy(), value, time.time())
    
    def _disk_lookup(self, x: Array) -> Optional[float]:
        """Look up value in disk cache.
        
        Args:
            x: Input point
            
        Returns:
            Cached value if found, None otherwise
        """
        if not self.enable_disk_cache:
            return None
        
        point_hash = self._hash_point(x)
        cache_file = self.cache_dir / f"{point_hash}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    cached_data = pickle.load(f)
                
                cached_point = cached_data["point"]
                cached_value = cached_data["value"]
                
                # Check tolerance
                if jnp.linalg.norm(x - cached_point) <= self.tolerance:
                    return cached_value
            except Exception:
                # Ignore corrupted cache files
                pass
        
        return None
    
    def _disk_store(self, x: Array, value: float):
        """Store value in disk cache.
        
        Args:
            x: Input point
            value: Function value
        """
        if not self.enable_disk_cache:
            return
        
        point_hash = self._hash_point(x)
        cache_file = self.cache_dir / f"{point_hash}.pkl"
        
        try:
            cache_data = {
                "point": x,
                "value": value,
                "timestamp": time.time(),
            }
            
            with open(cache_file, "wb") as f:
                pickle.dump(cache_data, f)
        except Exception:
            # Ignore disk write errors
            pass
    
    def get(self, x: Array) -> Optional[float]:
        """Get cached value for a point.
        
        Args:
            x: Input point
            
        Returns:
            Cached value if found, None otherwise
        """
        return self._find_cached_point(x)
    
    def put(self, x: Array, value: float):
        """Store value in cache.
        
        Args:
            x: Input point
            value: Function value
        """
        # Add to memory cache
        self._add_to_memory_cache(x, value)
        
        # Add to disk cache
        self._disk_store(x, value)
    
    def cached_function(self, func: Callable[[Array], float]) -> Callable[[Array], float]:
        """Create a cached version of a function.
        
        Args:
            func: Function to cache
            
        Returns:
            Cached function
        """
        def cached_func(x: Array) -> float:
            # Try to get from cache
            cached_value = self.get(x)
            if cached_value is not None:
                return cached_value
            
            # Compute and cache
            value = func(x)
            self.put(x, value)
            return value
        
        return cached_func
    
    def clear_memory(self):
        """Clear memory cache."""
        self._memory_cache.clear()
    
    def clear_disk(self):
        """Clear disk cache."""
        if self.enable_disk_cache:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
    
    def clear_all(self):
        """Clear both memory and disk caches."""
        self.clear_memory()
        self.clear_disk()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0
        
        disk_requests = self.stats["disk_hits"] + self.stats["disk_misses"]
        disk_hit_rate = self.stats["disk_hits"] / disk_requests if disk_requests > 0 else 0
        
        return {
            "memory_items": len(self._memory_cache),
            "max_memory_items": self.max_memory_items,
            "total_requests": total_requests,
            "memory_hits": self.stats["hits"],
            "memory_misses": self.stats["misses"],
            "memory_hit_rate": hit_rate,
            "disk_hits": self.stats["disk_hits"],
            "disk_misses": self.stats["disk_misses"],
            "disk_hit_rate": disk_hit_rate,
            "tolerance": self.tolerance,
            "disk_cache_enabled": self.enable_disk_cache,
        }
    
    def optimize_cache(self, max_age_hours: float = 24):
        """Optimize cache by removing old entries.
        
        Args:
            max_age_hours: Maximum age of cache entries in hours
        """
        cutoff_time = time.time() - (max_age_hours * 3600)
        
        # Clean memory cache
        old_hashes = [
            h for h, (_, _, timestamp) in self._memory_cache.items()
            if timestamp < cutoff_time
        ]
        for h in old_hashes:
            del self._memory_cache[h]
        
        # Clean disk cache
        if self.enable_disk_cache:
            for cache_file in self.cache_dir.glob("*.pkl"):
                if cache_file.stat().st_mtime < cutoff_time:
                    cache_file.unlink()


class CachedDataCollector:
    """Data collector with automatic caching of function evaluations."""
    
    def __init__(
        self,
        function: Callable[[Array], float],
        bounds: list,
        cache: Optional[FunctionCache] = None,
        cache_kwargs: Optional[Dict] = None,
    ):
        """Initialize cached data collector.
        
        Args:
            function: Black-box function to evaluate
            bounds: Bounds for input variables
            cache: Optional existing cache instance
            cache_kwargs: Arguments for creating new cache
        """
        self.original_function = function
        self.bounds = bounds
        
        if cache is None:
            cache_kwargs = cache_kwargs or {}
            self.cache = FunctionCache(**cache_kwargs)
        else:
            self.cache = cache
        
        # Create cached function
        self.cached_function = self.cache.cached_function(function)
    
    def __call__(self, x: Array) -> float:
        """Evaluate function with caching.
        
        Args:
            x: Input point
            
        Returns:
            Function value
        """
        return self.cached_function(x)
    
    def collect_data(
        self,
        n_samples: int,
        sampling: str = "sobol",
        **kwargs
    ):
        """Collect data using cached function evaluations.
        
        This method would integrate with the existing DataCollector
        but use the cached function for evaluations.
        """
        from .collector import DataCollector
        
        # Create data collector with cached function
        collector = DataCollector(self.cached_function, self.bounds)
        
        return collector.collect(n_samples, sampling, **kwargs)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()
    
    def clear_cache(self):
        """Clear function cache."""
        self.cache.clear_all()


# Global cache instance
_default_cache = None


def get_default_cache() -> FunctionCache:
    """Get default cache instance."""
    global _default_cache
    if _default_cache is None:
        _default_cache = FunctionCache()
    return _default_cache


def cached_function(func: Callable[[Array], float]) -> Callable[[Array], float]:
    """Decorator to add caching to a function.
    
    Args:
        func: Function to cache
        
    Returns:
        Cached function
    """
    return get_default_cache().cached_function(func)


def with_cache(
    function: Callable[[Array], float],
    cache_kwargs: Optional[Dict] = None,
) -> CachedDataCollector:
    """Create a cached data collector for a function.
    
    Args:
        function: Function to cache
        cache_kwargs: Optional cache configuration
        
    Returns:
        Cached data collector
    """
    cache_kwargs = cache_kwargs or {}
    cache = FunctionCache(**cache_kwargs)
    
    # Note: bounds would need to be provided separately
    # This is a utility function for when bounds are known
    return CachedDataCollector(function, [], cache)