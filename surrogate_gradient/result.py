"""Shared result dataclass."""
from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class OptimizationResult:
    """Result returned by every optimizer."""
    x_best: np.ndarray          # Best parameter vector found
    f_best: float               # Best function value (lower = better)
    history: List[float]        # Best-so-far at each iteration
    n_evals: int                # Total black-box evaluations used
    method: str = ""            # Name of the method

    def __repr__(self) -> str:
        return (
            f"OptimizationResult(method={self.method!r}, "
            f"f_best={self.f_best:.6f}, n_evals={self.n_evals})"
        )
