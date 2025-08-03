"""Surrogate Gradient Optimization Lab.

A toolkit for offline black-box optimization using learned gradient surrogates.
"""

__version__ = "0.1.0"
__author__ = "Terragon Labs"
__email__ = "team@terragon-labs.com"

# Version information
VERSION = __version__

# Core imports
from .models import *
from .optimizers import *
from .data import *

# Main interface classes
from .core import SurrogateOptimizer

__all__ = [
    "__version__",
    "VERSION",
    # Core classes
    "SurrogateOptimizer",
    # Models
    "Surrogate",
    "Dataset", 
    "NeuralSurrogate",
    "GPSurrogate",
    "RandomForestSurrogate",
    "HybridSurrogate",
    # Optimizers
    "BaseOptimizer",
    "OptimizationResult",
    "GradientDescentOptimizer",
    # Data collection
    "DataCollector",
    "collect_data",
]