"""Pure random-search baseline for comparison."""
from typing import Callable, List, Tuple
import numpy as np
from .result import OptimizationResult


class RandomSearchOptimizer:
    """Random search: sample uniformly in bounds, keep the best.

    This is the baseline that surrogate methods should beat — especially
    in low-evaluation regimes.
    """

    def __init__(self, seed: int = 0):
        self.seed = seed

    def run(
        self,
        fn: Callable[[np.ndarray], float],
        bounds: List[Tuple[float, float]],
        n_evals: int = 200,
    ) -> OptimizationResult:
        rng = np.random.default_rng(self.seed)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        x_best = None
        f_best = np.inf
        history: List[float] = []

        for _ in range(n_evals):
            x = rng.uniform(lo, hi)
            f = fn(x)
            if f < f_best:
                f_best = f
                x_best = x.copy()
            history.append(f_best)

        return OptimizationResult(
            x_best=x_best,
            f_best=f_best,
            history=history,
            n_evals=n_evals,
            method="RandomSearch",
        )
