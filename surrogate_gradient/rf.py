"""Random-Forest surrogate gradient optimizer.

Algorithm
---------
1. Collect initial batch via Latin-hypercube sampling.
2. Train a Random Forest regressor on (X, y).
3. Approximate surrogate gradient at x via finite differences:
       ∂f̂/∂xᵢ ≈ (f̂(x + ε·eᵢ) − f̂(x − ε·eᵢ)) / (2ε)
4. Take a gradient step: x ← x − η · ∇̂f(x).
5. Evaluate black-box, add to training set, retrain, repeat.

Random Forests have no analytic gradient (they are piecewise constant),
so finite-difference gradient approximation on the surrogate is the natural
approach.  Despite this, the forest prediction surface is smooth enough in
expectation that gradient descent makes useful progress.
"""
from typing import Callable, List, Optional, Tuple
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from .result import OptimizationResult
from .sampling import latin_hypercube


class RFSurrogateOptimizer:
    """Gradient optimizer via Random-Forest surrogate + finite differences.

    Parameters
    ----------
    n_estimators : int
        Number of trees in the forest.
    n_init : int
        Number of initial random evaluations (LHS).
    lr_optim : float
        Gradient step size.
    grad_steps : int
        Gradient steps per surrogate iteration.
    fd_eps : float
        Finite-difference perturbation size.
    seed : int
        Random seed.
    verbose : bool
        Print progress.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        n_init: int = 20,
        lr_optim: float = 0.05,
        grad_steps: int = 10,
        fd_eps: float = 1e-3,
        seed: int = 42,
        verbose: bool = False,
    ):
        self.n_estimators = n_estimators
        self.n_init = n_init
        self.lr_optim = lr_optim
        self.grad_steps = grad_steps
        self.fd_eps = fd_eps
        self.seed = seed
        self.verbose = verbose

    # ------------------------------------------------------------------

    def _fd_gradient(
        self,
        model: RandomForestRegressor,
        x: np.ndarray,
        lo: np.ndarray,
        hi: np.ndarray,
    ) -> np.ndarray:
        """Finite-difference gradient of RF surrogate at x."""
        dim = len(x)
        grad = np.zeros(dim)
        for i in range(dim):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] = min(x[i] + self.fd_eps, hi[i])
            x_minus[i] = max(x[i] - self.fd_eps, lo[i])
            f_plus = model.predict(x_plus[None])[0]
            f_minus = model.predict(x_minus[None])[0]
            grad[i] = (f_plus - f_minus) / (x_plus[i] - x_minus[i] + 1e-12)
        return grad

    # ------------------------------------------------------------------

    def run(
        self,
        fn: Callable[[np.ndarray], float],
        bounds: List[Tuple[float, float]],
        n_iter: int = 50,
    ) -> OptimizationResult:
        """Optimize fn over bounds."""
        np.random.seed(self.seed)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # ---- Phase 1: LHS init ----
        X_all = latin_hypercube(self.n_init, bounds, seed=self.seed)
        y_all = np.array([fn(x) for x in X_all])

        best_idx = np.argmin(y_all)
        x_best = X_all[best_idx].copy()
        f_best = float(y_all[best_idx])
        history: List[float] = [f_best] * self.n_init

        x_curr = x_best.copy()
        rng = np.random.default_rng(self.seed + 1000)

        # ---- Phase 2: surrogate-guided iterations ----
        for it in range(n_iter):
            # Refit RF
            model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=self.seed + it,
                n_jobs=-1,
            )
            model.fit(X_all, y_all)

            # Generate multiple candidates:
            # (a) gradient descent from x_best (exploitation)
            # (b) gradient descent from a new random start (exploration)
            candidates = []
            for x_start in [x_best, rng.uniform(lo, hi)]:
                x_opt = x_start.copy()
                for _ in range(self.grad_steps):
                    g = self._fd_gradient(model, x_opt, lo, hi)
                    x_opt = x_opt - self.lr_optim * g
                    x_opt = np.clip(x_opt, lo, hi)
                candidates.append(x_opt)

            # Pick the candidate with the lower surrogate prediction
            preds = [model.predict(c[None])[0] for c in candidates]
            x_opt = candidates[int(np.argmin(preds))]

            # Evaluate real black-box
            f_new = fn(x_opt)
            X_all = np.vstack([X_all, x_opt[None]])
            y_all = np.append(y_all, f_new)

            if f_new < f_best:
                f_best = f_new
                x_best = x_opt.copy()
                x_curr = x_opt.copy()

            history.append(f_best)
            if self.verbose:
                print(f"  [RF]     iter {it+1:3d} | f_new={f_new:.6f} | f_best={f_best:.6f}")

        return OptimizationResult(
            x_best=x_best,
            f_best=f_best,
            history=history,
            n_evals=self.n_init + n_iter,
            method="RFSurrogate",
        )
