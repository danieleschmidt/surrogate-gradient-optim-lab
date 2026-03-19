"""
surrogate_gradient — learning gradient approximations for black-box functions.

Two surrogate methods:
  1. NeuralSurrogate   — small MLP trained via PyTorch; gradients via autograd.
  2. RFSurrogate       — Random-Forest surrogate; gradients via finite differences.

Both implement the same Optimizer interface:
  optimizer.run(black_box_fn, x0, bounds, n_init, n_iter) -> OptimizationResult

Plus a random-search baseline for comparison.
"""

from .neural import NeuralSurrogateOptimizer
from .rf import RFSurrogateOptimizer
from .random_search import RandomSearchOptimizer
from .result import OptimizationResult
from .benchmarks import rosenbrock, sphere, ackley, rastrigin

__all__ = [
    "NeuralSurrogateOptimizer",
    "RFSurrogateOptimizer",
    "RandomSearchOptimizer",
    "OptimizationResult",
    "rosenbrock",
    "sphere",
    "ackley",
    "rastrigin",
]
