"""Sampling utilities — Latin Hypercube Sampling."""
from typing import List, Optional, Tuple
import numpy as np


def latin_hypercube(
    n: int,
    bounds: List[Tuple[float, float]],
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate n points via Latin Hypercube Sampling within bounds.

    Parameters
    ----------
    n : int
        Number of points.
    bounds : list of (lo, hi) pairs
    seed : int, optional

    Returns
    -------
    X : np.ndarray, shape (n, d)
    """
    rng = np.random.default_rng(seed)
    d = len(bounds)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    # Generate LHS: each dimension divided into n equal intervals
    X = np.zeros((n, d))
    for j in range(d):
        perm = rng.permutation(n)
        points = (perm + rng.uniform(size=n)) / n  # uniform within each stratum
        X[:, j] = lo[j] + points * (hi[j] - lo[j])
    return X
