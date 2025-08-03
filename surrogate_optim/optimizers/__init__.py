"""Optimization algorithms using surrogate gradients."""

from .base import BaseOptimizer, OptimizationResult
from .gradient_descent import GradientDescentOptimizer
from .trust_region import TrustRegionOptimizer
from .multi_start import MultiStartOptimizer
from .utils import optimize_with_surrogate

__all__ = [
    "BaseOptimizer",
    "OptimizationResult",
    "GradientDescentOptimizer", 
    "TrustRegionOptimizer",
    "MultiStartOptimizer",
    "optimize_with_surrogate",
]