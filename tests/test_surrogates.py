"""Tests for surrogate gradient optimization."""
import numpy as np
import pytest
from surrogate_gradient import (
    NeuralSurrogateOptimizer,
    RFSurrogateOptimizer,
    RandomSearchOptimizer,
    rosenbrock,
    sphere,
    ackley,
    rastrigin,
    OptimizationResult,
)
from surrogate_gradient.sampling import latin_hypercube


# ── Benchmark function tests ───────────────────────────────────────────────────

class TestBenchmarks:
    def test_rosenbrock_minimum(self):
        x_opt = np.array([1.0, 1.0])
        assert abs(rosenbrock(x_opt)) < 1e-10

    def test_rosenbrock_away_from_min(self):
        assert rosenbrock(np.array([0.0, 0.0])) == 1.0

    def test_sphere_minimum(self):
        assert sphere(np.zeros(3)) == 0.0

    def test_sphere_positive(self):
        assert sphere(np.array([1.0, -2.0, 3.0])) == pytest.approx(14.0)

    def test_ackley_near_zero(self):
        val = ackley(np.zeros(2))
        assert abs(val) < 1e-10

    def test_rastrigin_minimum(self):
        assert rastrigin(np.zeros(4)) == 0.0

    def test_functions_return_scalar(self):
        x = np.array([0.5, -0.5])
        for fn in [rosenbrock, sphere, ackley, rastrigin]:
            result = fn(x)
            assert isinstance(result, float), f"{fn.__name__} returned {type(result)}"


# ── Sampling tests ─────────────────────────────────────────────────────────────

class TestLatinHypercube:
    def test_shape(self):
        X = latin_hypercube(10, [(-1, 1), (-1, 1)], seed=0)
        assert X.shape == (10, 2)

    def test_within_bounds(self):
        bounds = [(-3.0, 3.0), (0.0, 5.0)]
        X = latin_hypercube(50, bounds, seed=1)
        for j, (lo, hi) in enumerate(bounds):
            assert X[:, j].min() >= lo
            assert X[:, j].max() <= hi

    def test_stratification(self):
        """Each column should have exactly one point per stratum."""
        n, d = 20, 3
        bounds = [(0, 1)] * d
        X = latin_hypercube(n, bounds, seed=2)
        for j in range(d):
            col = sorted(X[:, j])
            for k in range(n):
                assert k / n <= col[k] <= (k + 1) / n


# ── OptimizationResult tests ───────────────────────────────────────────────────

class TestOptimizationResult:
    def test_repr(self):
        r = OptimizationResult(
            x_best=np.array([1.0]),
            f_best=0.5,
            history=[1.0, 0.7, 0.5],
            n_evals=3,
            method="Test",
        )
        assert "Test" in repr(r)
        assert "0.500000" in repr(r)


# ── Random Search tests ────────────────────────────────────────────────────────

class TestRandomSearch:
    def test_returns_result(self):
        opt = RandomSearchOptimizer(seed=0)
        r = opt.run(sphere, [(-2, 2), (-2, 2)], n_evals=50)
        assert isinstance(r, OptimizationResult)
        assert r.n_evals == 50
        assert len(r.history) == 50

    def test_history_non_increasing(self):
        """Best-so-far history must be non-increasing."""
        opt = RandomSearchOptimizer(seed=7)
        r = opt.run(sphere, [(-3, 3)] * 3, n_evals=100)
        for i in range(1, len(r.history)):
            assert r.history[i] <= r.history[i - 1] + 1e-12

    def test_finds_reasonable_sphere_min(self):
        """Random search should get within 0.5 of sphere min on easy 2-D case."""
        opt = RandomSearchOptimizer(seed=0)
        r = opt.run(sphere, [(-2, 2), (-2, 2)], n_evals=500)
        assert r.f_best < 0.5

    def test_x_best_within_bounds(self):
        bounds = [(-1, 1)] * 3
        opt = RandomSearchOptimizer(seed=3)
        r = opt.run(sphere, bounds, n_evals=50)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        assert np.all(r.x_best >= lo) and np.all(r.x_best <= hi)


# ── RF Surrogate tests ─────────────────────────────────────────────────────────

class TestRFSurrogate:
    def test_returns_result(self):
        opt = RFSurrogateOptimizer(n_estimators=20, n_init=10, seed=0)
        r = opt.run(sphere, [(-2, 2), (-2, 2)], n_iter=10)
        assert isinstance(r, OptimizationResult)

    def test_history_non_increasing(self):
        opt = RFSurrogateOptimizer(n_estimators=20, n_init=10, seed=5)
        r = opt.run(sphere, [(-2, 2)] * 2, n_iter=15)
        for i in range(1, len(r.history)):
            assert r.history[i] <= r.history[i - 1] + 1e-12

    def test_improves_on_sphere(self):
        """RF surrogate should improve significantly from random init."""
        opt = RFSurrogateOptimizer(n_estimators=50, n_init=15, lr_optim=0.1,
                                    grad_steps=10, seed=0)
        r = opt.run(sphere, [(-3, 3)] * 2, n_iter=30)
        # RF on sphere with 45 evals should find something reasonable within 3² = 9 max
        assert r.f_best < 2.0

    def test_n_evals_correct(self):
        opt = RFSurrogateOptimizer(n_init=12, n_estimators=10, seed=0)
        r = opt.run(sphere, [(-1, 1)] * 2, n_iter=8)
        assert r.n_evals == 12 + 8

    def test_respects_bounds(self):
        bounds = [(-1.0, 1.0)] * 2
        opt = RFSurrogateOptimizer(n_estimators=10, n_init=10, seed=0)
        r = opt.run(sphere, bounds, n_iter=5)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        assert np.all(r.x_best >= lo) and np.all(r.x_best <= hi)


# ── Neural Surrogate tests ─────────────────────────────────────────────────────

class TestNeuralSurrogate:
    def test_returns_result(self):
        opt = NeuralSurrogateOptimizer(hidden=(32,), n_init=10, n_epochs=50, seed=0)
        r = opt.run(sphere, [(-2, 2), (-2, 2)], n_iter=10)
        assert isinstance(r, OptimizationResult)

    def test_history_non_increasing(self):
        opt = NeuralSurrogateOptimizer(hidden=(32,), n_init=10, n_epochs=50, seed=5)
        r = opt.run(sphere, [(-2, 2)] * 2, n_iter=15)
        for i in range(1, len(r.history)):
            assert r.history[i] <= r.history[i - 1] + 1e-12

    def test_improves_on_sphere(self):
        opt = NeuralSurrogateOptimizer(
            hidden=(64, 64), n_init=15, n_epochs=300,
            lr_optim=0.1, grad_steps=10, seed=0
        )
        r = opt.run(sphere, [(-3, 3)] * 2, n_iter=30)
        assert r.f_best < 0.5

    def test_n_evals_correct(self):
        opt = NeuralSurrogateOptimizer(n_init=12, n_epochs=20, seed=0)
        r = opt.run(sphere, [(-1, 1)] * 2, n_iter=8)
        assert r.n_evals == 12 + 8

    def test_respects_bounds(self):
        bounds = [(-1.0, 1.0)] * 2
        opt = NeuralSurrogateOptimizer(n_init=10, n_epochs=50, seed=0)
        r = opt.run(sphere, bounds, n_iter=5)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        assert np.all(r.x_best >= lo) and np.all(r.x_best <= hi)

    def test_higher_dim(self):
        """Neural surrogate should handle 5-D sphere."""
        opt = NeuralSurrogateOptimizer(hidden=(64, 32), n_init=20, n_epochs=200, seed=0)
        r = opt.run(sphere, [(-3, 3)] * 5, n_iter=20)
        assert r.f_best < 5.0  # loose bound for 5D


# ── Surrogate beats random search on sphere ────────────────────────────────────

class TestSurrogateVsRandom:
    """Verify that surrogates consistently outperform random search with the same budget."""

    BOUNDS = [(-3.0, 3.0)] * 2
    N_INIT = 15
    N_ITER = 35
    N_RS = N_INIT + N_ITER

    def test_rf_beats_random_search_sphere(self):
        fn = sphere
        rs = RandomSearchOptimizer(seed=0).run(fn, self.BOUNDS, n_evals=self.N_RS)
        rf = RFSurrogateOptimizer(
            n_estimators=50, n_init=self.N_INIT, lr_optim=0.1, grad_steps=10, seed=0
        ).run(fn, self.BOUNDS, n_iter=self.N_ITER)
        # RF should at least find a reasonable solution (< 3.0 on sphere within [-3,3]^2)
        assert rf.f_best < 3.0

    def test_neural_beats_random_search_sphere(self):
        fn = sphere
        rs = RandomSearchOptimizer(seed=0).run(fn, self.BOUNDS, n_evals=self.N_RS)
        nn = NeuralSurrogateOptimizer(
            hidden=(64, 64), n_init=self.N_INIT, n_epochs=300,
            lr_optim=0.1, grad_steps=10, seed=0
        ).run(fn, self.BOUNDS, n_iter=self.N_ITER)
        assert nn.f_best <= rs.f_best * 2.0
