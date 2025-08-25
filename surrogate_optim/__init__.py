"""Surrogate Gradient Optimization Lab.

A toolkit for offline black-box optimization using learned gradient surrogates.
"""

__version__ = "0.1.0"
__author__ = "Terragon Labs"
__email__ = "team@terragon-labs.com"

# Version information
VERSION = __version__

# Core imports
# Main interface classes
from .core import SurrogateOptimizer as CoreSurrogateOptimizer
from .core import quick_optimize
from .data import *
from .models import *
from .optimizers import *

# Import the full version from core.py which has all methods
SurrogateOptimizer = CoreSurrogateOptimizer

__all__ = [
    "__version__",
    "VERSION",
    # Core classes
    "SurrogateOptimizer",
    "quick_optimize",
    "EnhancedSurrogateOptimizer",
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
