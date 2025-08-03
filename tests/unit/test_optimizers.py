"""Unit tests for optimization algorithms."""

import numpy as np
import pytest
from unittest.mock import Mock, MagicMock, patch

from surrogate_optim.optimizers import (
    SurrogateOptimizer,
    TrustRegionOptimizer,
    MultiStartOptimizer,
    GradientDescentOptimizer,
)
from surrogate_optim.models import NeuralSurrogate
from surrogate_optim.data import Dataset


class MockSurrogate:
    """Mock surrogate for testing."""
    
    def __init__(self):
        self.fitted = True
        
    def predict(self, x):
        """Mock prediction: quadratic function."""
        x = np.asarray(x)
        return -np.sum(x**2)
    
    def gradient(self, x):
        """Mock gradient: gradient of quadratic function."""
        x = np.asarray(x)
        return -2 * x


class TestSurrogateOptimizer:
    """Test SurrogateOptimizer implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.optimizer = SurrogateOptimizer(
            surrogate_type="neural_network",
            surrogate_kwargs={"hidden_dims": [16], "epochs": 5}
        )
        
        # Create test data
        np.random.seed(42)
        self.X = np.random.uniform(-2, 2, (30, 2))
        self.y = np.sum(self.X**2, axis=1) + 0.1 * np.random.randn(30)
        self.dataset = Dataset(X=self.X, y=self.y)
        
    def test_initialization(self):
        """Test optimizer initialization."""
        assert self.optimizer.surrogate_type == "neural_network"
        assert "hidden_dims" in self.optimizer.surrogate_kwargs
        assert self.optimizer.surrogate is None
        
    def test_fit_surrogate_neural_network(self):
        """Test fitting neural network surrogate."""
        with patch('builtins.print'):
            surrogate = self.optimizer.fit_surrogate(self.dataset)
        
        assert surrogate is not None
        assert self.optimizer.surrogate is not None
        assert isinstance(self.optimizer.surrogate, NeuralSurrogate)
        
    def test_fit_surrogate_gp(self):
        """Test fitting GP surrogate."""
        optimizer = SurrogateOptimizer(surrogate_type="gp")
        
        surrogate = optimizer.fit_surrogate(self.dataset)
        
        assert surrogate is not None
        assert optimizer.surrogate is not None
        
    def test_fit_surrogate_random_forest(self):
        """Test fitting random forest surrogate."""
        optimizer = SurrogateOptimizer(surrogate_type="random_forest")
        
        surrogate = optimizer.fit_surrogate(self.dataset)
        
        assert surrogate is not None
        assert optimizer.surrogate is not None
        
    def test_fit_surrogate_unknown_type(self):
        """Test error for unknown surrogate type."""
        optimizer = SurrogateOptimizer(surrogate_type="unknown")
        
        with pytest.raises(ValueError, match="Unknown surrogate type"):
            optimizer.fit_surrogate(self.dataset)
            
    def test_optimize(self):
        """Test optimization."""
        with patch('builtins.print'):
            self.optimizer.fit_surrogate(self.dataset)
        
        x0 = np.array([1.0, 1.0])
        result = self.optimizer.optimize(
            initial_point=x0,
            method="L-BFGS-B",
            num_steps=10
        )
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)
        assert not np.any(np.isnan(result))
        
    def test_optimize_without_surrogate_raises_error(self):
        """Test that optimization without fitted surrogate raises error."""
        x0 = np.array([1.0, 1.0])
        
        with pytest.raises(ValueError, match="No surrogate fitted"):
            self.optimizer.optimize(x0)
            
    def test_optimize_with_bounds(self):
        """Test optimization with bounds."""
        with patch('builtins.print'):
            self.optimizer.fit_surrogate(self.dataset)
        
        x0 = np.array([1.0, 1.0])
        bounds = [(-2.0, 2.0), (-2.0, 2.0)]
        
        result = self.optimizer.optimize(
            initial_point=x0,
            bounds=bounds,
            num_steps=10
        )
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)
        
        # Check bounds are respected
        for i, (lower, upper) in enumerate(bounds):
            assert lower <= result[i] <= upper


class TestTrustRegionOptimizer:
    """Test TrustRegionOptimizer implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.surrogate = MockSurrogate()
        
        # True function for validation
        def true_function(x):
            x = np.asarray(x)
            return -np.sum(x**2) + 0.1 * np.sin(10 * np.sum(x))
        
        self.optimizer = TrustRegionOptimizer(
            surrogate=self.surrogate,
            true_function=true_function,
            initial_radius=0.5,
            max_radius=2.0,
        )
        
    def test_initialization(self):
        """Test optimizer initialization."""
        assert self.optimizer.surrogate is self.surrogate
        assert self.optimizer.radius == 0.5
        assert self.optimizer.max_radius == 2.0
        assert self.optimizer.eta == 0.15
        
    def test_optimize(self):
        """Test trust region optimization."""
        x0 = np.array([1.0, 1.0])
        
        result = self.optimizer.optimize(
            x0=x0,
            max_iterations=10,
            validate_every=2,
            tolerance=1e-4
        )
        
        assert "x" in result
        assert "trajectory" in result
        assert "iterations" in result
        assert "converged" in result
        
        assert isinstance(result["x"], np.ndarray)
        assert result["x"].shape == (2,)
        assert isinstance(result["trajectory"], np.ndarray)
        assert result["iterations"] <= 10
        
    def test_solve_subproblem(self):
        """Test trust region subproblem solving."""
        x_center = np.array([0.5, 0.5])
        x_new = self.optimizer._solve_subproblem(x_center)
        
        assert isinstance(x_new, np.ndarray)
        assert x_new.shape == (2,)
        
        # Should be within trust region
        distance = np.linalg.norm(x_new - x_center)
        assert distance <= self.optimizer.radius + 1e-6
        
    def test_radius_adaptation(self):
        """Test trust region radius adaptation."""
        x0 = np.array([2.0, 2.0])
        initial_radius = self.optimizer.radius
        
        result = self.optimizer.optimize(x0=x0, max_iterations=5)
        
        # Radius should have been updated during optimization
        # (exact behavior depends on function and convergence)
        assert self.optimizer.radius > 0
        assert self.optimizer.radius <= self.optimizer.max_radius


class TestMultiStartOptimizer:
    """Test MultiStartOptimizer implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.surrogate = MockSurrogate()
        
        self.optimizer = MultiStartOptimizer(
            surrogate=self.surrogate,
            n_starts=5,
            start_method="random",
            local_optimizer="L-BFGS-B",
        )
        
    def test_initialization(self):
        """Test optimizer initialization."""
        assert self.optimizer.surrogate is self.surrogate
        assert self.optimizer.n_starts == 5
        assert self.optimizer.start_method == "random"
        assert self.optimizer.local_optimizer == "L-BFGS-B"
        
    def test_optimize_global(self):
        """Test global optimization."""
        bounds = [(-3.0, 3.0), (-3.0, 3.0)]
        
        result = self.optimizer.optimize_global(
            bounds=bounds,
            max_iterations=10
        )
        
        assert "best_point" in result
        assert "best_value" in result
        assert "all_results" in result
        assert "n_successful" in result
        
        assert isinstance(result["best_point"], np.ndarray)
        assert result["best_point"].shape == (2,)
        assert isinstance(result["best_value"], (float, np.float64))
        assert len(result["all_results"]) <= self.optimizer.n_starts
        
    def test_generate_starts_random(self):
        """Test random start generation."""
        bounds = np.array([[-2.0, 2.0], [-1.0, 1.0]])
        starts = self.optimizer._generate_starts(bounds, 2)
        
        assert starts.shape == (5, 2)
        
        # Check bounds
        assert np.all(starts[:, 0] >= -2.0)
        assert np.all(starts[:, 0] <= 2.0)
        assert np.all(starts[:, 1] >= -1.0)
        assert np.all(starts[:, 1] <= 1.0)
        
    def test_generate_starts_sobol(self):
        """Test Sobol start generation."""
        optimizer = MultiStartOptimizer(
            surrogate=self.surrogate,
            n_starts=8,
            start_method="sobol"
        )
        
        bounds = np.array([[-1.0, 1.0], [-1.0, 1.0]])
        starts = optimizer._generate_starts(bounds, 2)
        
        assert starts.shape == (8, 2)
        
    def test_generate_starts_grid(self):
        """Test grid start generation."""
        optimizer = MultiStartOptimizer(
            surrogate=self.surrogate,
            n_starts=9,
            start_method="grid"
        )
        
        bounds = np.array([[-1.0, 1.0], [-1.0, 1.0]])
        starts = optimizer._generate_starts(bounds, 2)
        
        assert starts.shape == (9, 2)
        
    def test_optimize_single(self):
        """Test single optimization run."""
        start = np.array([0.5, 0.5])
        bounds = np.array([[-2.0, 2.0], [-2.0, 2.0]])
        
        result = self.optimizer._optimize_single(start, bounds, 10)
        
        assert hasattr(result, 'x')
        assert hasattr(result, 'fun')
        assert hasattr(result, 'success')


class TestGradientDescentOptimizer:
    """Test GradientDescentOptimizer implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.surrogate = MockSurrogate()
        
        self.optimizer = GradientDescentOptimizer(
            surrogate=self.surrogate,
            learning_rate=0.1,
            momentum=0.9,
            adaptive=True,
        )
        
    def test_initialization(self):
        """Test optimizer initialization."""
        assert self.optimizer.surrogate is self.surrogate
        assert self.optimizer.learning_rate == 0.1
        assert self.optimizer.momentum == 0.9
        assert self.optimizer.adaptive is True
        
    def test_optimize(self):
        """Test gradient descent optimization."""
        x0 = np.array([2.0, 2.0])
        
        result = self.optimizer.optimize(
            x0=x0,
            max_iterations=50,
            tolerance=1e-4
        )
        
        assert "x" in result
        assert "trajectory" in result
        assert "learning_rates" in result
        assert "iterations" in result
        assert "converged" in result
        assert "final_value" in result
        
        assert isinstance(result["x"], np.ndarray)
        assert result["x"].shape == (2,)
        assert isinstance(result["trajectory"], np.ndarray)
        assert result["trajectory"].shape[1] == 2
        assert len(result["learning_rates"]) == result["trajectory"].shape[0]
        
    def test_optimize_with_bounds(self):
        """Test optimization with bounds."""
        x0 = np.array([1.0, 1.0])
        bounds = [(-2.0, 2.0), (-2.0, 2.0)]
        
        result = self.optimizer.optimize(
            x0=x0,
            bounds=bounds,
            max_iterations=20
        )
        
        # Final point should respect bounds
        for i, (lower, upper) in enumerate(bounds):
            assert lower <= result["x"][i] <= upper
            
        # All trajectory points should respect bounds
        for point in result["trajectory"]:
            for i, (lower, upper) in enumerate(bounds):
                assert lower <= point[i] <= upper
                
    def test_convergence(self):
        """Test convergence to optimum."""
        x0 = np.array([3.0, 3.0])
        
        result = self.optimizer.optimize(
            x0=x0,
            max_iterations=100,
            tolerance=1e-3
        )
        
        # Should converge close to origin for quadratic function
        assert np.linalg.norm(result["x"]) < 0.1
        assert result["converged"] is True
        
    def test_adaptive_learning_rate(self):
        """Test adaptive learning rate adjustment."""
        optimizer = GradientDescentOptimizer(
            surrogate=self.surrogate,
            learning_rate=0.5,
            adaptive=True
        )
        
        x0 = np.array([2.0, 2.0])
        result = optimizer.optimize(x0=x0, max_iterations=30)
        
        learning_rates = result["learning_rates"]
        
        # Learning rate should change during optimization
        assert len(set(learning_rates)) > 1
        
    def test_momentum_effect(self):
        """Test momentum effect on optimization."""
        # Test without momentum
        optimizer_no_momentum = GradientDescentOptimizer(
            surrogate=self.surrogate,
            learning_rate=0.1,
            momentum=0.0
        )
        
        # Test with momentum
        optimizer_with_momentum = GradientDescentOptimizer(
            surrogate=self.surrogate,
            learning_rate=0.1,
            momentum=0.9
        )
        
        x0 = np.array([1.0, 1.0])
        
        result_no_momentum = optimizer_no_momentum.optimize(x0=x0, max_iterations=20)
        result_with_momentum = optimizer_with_momentum.optimize(x0=x0, max_iterations=20)
        
        # Both should converge, but with different trajectories
        assert result_no_momentum["converged"] or result_with_momentum["converged"]
        
        # Trajectories should be different
        trajectory_diff = np.linalg.norm(
            result_no_momentum["trajectory"] - result_with_momentum["trajectory"]
        )
        assert trajectory_diff > 0.1


class TestOptimizerIntegration:
    """Integration tests for optimizers."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a more complex test function
        self.surrogate = MockSurrogate()
        
    def test_optimizer_comparison(self):
        """Compare different optimizers on same problem."""
        x0 = np.array([2.0, 2.0])
        bounds = [(-3.0, 3.0), (-3.0, 3.0)]
        
        # Trust region optimizer
        tr_optimizer = TrustRegionOptimizer(surrogate=self.surrogate)
        tr_result = tr_optimizer.optimize(x0=x0, max_iterations=20)
        
        # Multi-start optimizer
        ms_optimizer = MultiStartOptimizer(surrogate=self.surrogate, n_starts=3)
        ms_result = ms_optimizer.optimize_global(bounds=bounds, max_iterations=20)
        
        # Gradient descent optimizer
        gd_optimizer = GradientDescentOptimizer(surrogate=self.surrogate)
        gd_result = gd_optimizer.optimize(x0=x0, max_iterations=50)
        
        # All should find reasonable solutions
        tr_distance = np.linalg.norm(tr_result["x"])
        ms_distance = np.linalg.norm(ms_result["best_point"])
        gd_distance = np.linalg.norm(gd_result["x"])
        
        assert tr_distance < 1.0
        assert ms_distance < 1.0
        assert gd_distance < 1.0
        
    def test_optimizer_robustness(self):
        """Test optimizer robustness to different starting points."""
        bounds = [(-5.0, 5.0), (-5.0, 5.0)]
        
        ms_optimizer = MultiStartOptimizer(
            surrogate=self.surrogate,
            n_starts=10,
            start_method="random"
        )
        
        result = ms_optimizer.optimize_global(bounds=bounds)
        
        # Should find good solution regardless of starts
        assert result["n_successful"] > 0
        assert np.linalg.norm(result["best_point"]) < 0.5


@pytest.fixture
def quadratic_surrogate():
    """Create a surrogate that approximates a quadratic function."""
    surrogate = Mock()
    
    def predict(x):
        x = np.asarray(x)
        return -np.sum(x**2)
    
    def gradient(x):
        x = np.asarray(x)
        return -2 * x
    
    surrogate.predict = predict
    surrogate.gradient = gradient
    
    return surrogate