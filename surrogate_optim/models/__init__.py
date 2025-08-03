"""Surrogate model implementations for gradient approximation."""

from .base import Surrogate, Dataset
from .neural import NeuralSurrogate
from .gaussian_process import GPSurrogate
from .random_forest import RandomForestSurrogate
from .hybrid import HybridSurrogate

__all__ = [
    "Surrogate",
    "Dataset",
    "NeuralSurrogate",
    "GPSurrogate",
    "RandomForestSurrogate",
    "HybridSurrogate",
]