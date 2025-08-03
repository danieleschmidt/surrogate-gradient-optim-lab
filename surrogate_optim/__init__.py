"""Surrogate Gradient Optimization Lab.

A toolkit for offline black-box optimization using learned gradient surrogates.
"""

__version__ = "0.1.0"
__author__ = "Terragon Labs"
__email__ = "team@terragon-labs.com"

# Version information
VERSION = __version__

# Core imports
from .models import (
    Surrogate,
    NeuralSurrogate,
    GPSurrogate,
    RandomForestSurrogate,
    HybridSurrogate,
)

from .optimizers import (
    BaseOptimizer,
    GradientDescentOptimizer,
    TrustRegionOptimizer,
    MultiStartOptimizer,
    optimize_with_surrogate,
)

from .data import (
    Dataset,
    DataCollector,
    collect_data,
    ActiveLearner,
    GradientEstimator,
)

__all__ = [
    "__version__",
    "VERSION",
    # Models
    "Surrogate",
    "NeuralSurrogate", 
    "GPSurrogate",
    "RandomForestSurrogate",
    "HybridSurrogate",
    # Optimizers
    "BaseOptimizer",
    "GradientDescentOptimizer",
    "TrustRegionOptimizer", 
    "MultiStartOptimizer",
    "optimize_with_surrogate",
    # Data
    "Dataset",
    "DataCollector",
    "collect_data",
    "ActiveLearner",
    "GradientEstimator",
]