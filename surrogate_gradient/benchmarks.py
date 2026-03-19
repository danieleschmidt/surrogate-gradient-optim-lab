"""Standard black-box benchmark functions.

All functions take a 1-D numpy array and return a scalar float.
They expose no gradients — callers only get function values.
"""
import numpy as np


def rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock banana function.  Global min = 0 at x = (1,...,1)."""
    x = np.asarray(x, dtype=float)
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


def sphere(x: np.ndarray) -> float:
    """Sphere function.  Global min = 0 at x = 0."""
    x = np.asarray(x, dtype=float)
    return float(np.dot(x, x))


def ackley(x: np.ndarray) -> float:
    """Ackley function (2-D standard form, generalised here to n-D).
    Global min ≈ 0 at x = 0.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    a, b, c = 20.0, 0.2, 2 * np.pi
    term1 = -a * np.exp(-b * np.sqrt(np.sum(x ** 2) / n))
    term2 = -np.exp(np.sum(np.cos(c * x)) / n)
    return float(term1 + term2 + a + np.e)


def rastrigin(x: np.ndarray) -> float:
    """Rastrigin function.  Global min = 0 at x = 0."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    return float(10 * n + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))
