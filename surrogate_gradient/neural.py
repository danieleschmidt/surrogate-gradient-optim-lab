"""Neural-network surrogate gradient optimizer.

Algorithm
---------
1. Collect an initial batch of (x, f(x)) pairs via Latin-hypercube sampling.
2. Train a small MLP to predict f(x).
3. Use autograd (torch.autograd) to compute ∂f̂/∂x (the surrogate gradient).
4. Take a gradient step on the current candidate: x ← x − η · ∇f̂(x).
5. Evaluate the black-box at the new point, add to training set.
6. Retrain surrogate (warm-started) and repeat.

Because the surrogate is differentiable, we can do gradient descent through
it even though the real black-box has no gradient API.
"""
from typing import Callable, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .result import OptimizationResult
from .sampling import latin_hypercube


class _MLP(nn.Module):
    """Small fully-connected network: input → [hidden] → 1 output."""

    def __init__(self, input_dim: int, hidden: List[int] = (64, 64)):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class NeuralSurrogateOptimizer:
    """Gradient optimizer via neural-network surrogate.

    Parameters
    ----------
    hidden : tuple of ints
        Hidden layer widths for the MLP surrogate.
    n_init : int
        Number of initial random evaluations (LHS).
    n_epochs : int
        Training epochs per surrogate refit.
    lr_surrogate : float
        Adam learning rate for the surrogate MLP.
    lr_optim : float
        Step size for gradient descent on x.
    grad_steps : int
        Number of gradient steps to take per surrogate iteration.
    seed : int
        Random seed.
    verbose : bool
        Print progress.
    """

    def __init__(
        self,
        hidden: Tuple[int, ...] = (64, 64),
        n_init: int = 20,
        n_epochs: int = 300,
        lr_surrogate: float = 1e-3,
        lr_optim: float = 0.05,
        grad_steps: int = 10,
        seed: int = 42,
        verbose: bool = False,
    ):
        self.hidden = hidden
        self.n_init = n_init
        self.n_epochs = n_epochs
        self.lr_surrogate = lr_surrogate
        self.lr_optim = lr_optim
        self.grad_steps = grad_steps
        self.seed = seed
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _train_surrogate(
        self,
        model: _MLP,
        X: np.ndarray,
        y: np.ndarray,
    ) -> None:
        """Fit the MLP on (X, y) in-place."""
        # Normalise targets for stable training
        y_mean, y_std = y.mean(), y.std() + 1e-8
        y_norm = (y - y_mean) / y_std

        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y_norm, dtype=torch.float32)

        optim_ = optim.Adam(model.parameters(), lr=self.lr_surrogate)
        loss_fn = nn.MSELoss()
        model.train()
        for _ in range(self.n_epochs):
            optim_.zero_grad()
            pred = model(Xt)
            loss = loss_fn(pred, yt)
            loss.backward()
            optim_.step()

        # Stash normalisation for gradient use
        model._y_mean = y_mean
        model._y_std = float(y_std)

    def _surrogate_gradient(
        self,
        model: _MLP,
        x: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """Return (f̂(x), ∇f̂(x)) using autograd through the MLP."""
        model.eval()
        xt = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        pred = model(xt.unsqueeze(0))  # shape (1,)
        val = pred[0]
        val.backward()
        grad = xt.grad.detach().numpy().copy()
        # Denormalize value (grad direction is scale-invariant)
        f_hat = float(val.item()) * model._y_std + model._y_mean
        return f_hat, grad

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        fn: Callable[[np.ndarray], float],
        bounds: List[Tuple[float, float]],
        n_iter: int = 50,
    ) -> OptimizationResult:
        """Optimize fn over bounds.

        Parameters
        ----------
        fn : callable
            Black-box function (no gradient API).
        bounds : list of (lo, hi) tuples
        n_iter : int
            Number of surrogate-guided iterations after init.
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = len(bounds)

        # ---- Phase 1: random initialisation (LHS) ----
        X_init = latin_hypercube(self.n_init, bounds, seed=self.seed)
        y_init = np.array([fn(x) for x in X_init])

        X_all = X_init.copy()
        y_all = y_init.copy()

        best_idx = np.argmin(y_all)
        x_best = X_all[best_idx].copy()
        f_best = float(y_all[best_idx])
        history: List[float] = [f_best] * self.n_init

        model = _MLP(dim, list(self.hidden))

        # ---- Phase 2: surrogate-guided iterations ----
        x_curr = x_best.copy()

        for it in range(n_iter):
            # Refit surrogate
            self._train_surrogate(model, X_all, y_all)

            # Gradient descent on x via surrogate
            x_opt = x_curr.copy()
            for _ in range(self.grad_steps):
                _, g = self._surrogate_gradient(model, x_opt)
                x_opt = x_opt - self.lr_optim * g
                x_opt = np.clip(x_opt, lo, hi)

            # Evaluate real black-box at proposed point
            f_new = fn(x_opt)
            X_all = np.vstack([X_all, x_opt[None]])
            y_all = np.append(y_all, f_new)

            if f_new < f_best:
                f_best = f_new
                x_best = x_opt.copy()
                x_curr = x_opt.copy()
            # else keep current best region

            history.append(f_best)
            if self.verbose:
                print(f"  [Neural] iter {it+1:3d} | f_new={f_new:.6f} | f_best={f_best:.6f}")

        return OptimizationResult(
            x_best=x_best,
            f_best=f_best,
            history=history,
            n_evals=self.n_init + n_iter,
            method="NeuralSurrogate",
        )
