"""Unit tests for surrogate models."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from surrogate_optim.models import (
    NeuralSurrogate,
    GPSurrogate,
    RandomForestSurrogate,
    HybridSurrogate,
)
from surrogate_optim.data import Dataset


class TestNeuralSurrogate:
    """Test NeuralSurrogate implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.surrogate = NeuralSurrogate(
            hidden_dims=[32, 16],
            epochs=10,  # Small for testing
            learning_rate=0.01,
        )
        
        # Create test data
        np.random.seed(42)
        self.X = np.random.uniform(-2, 2, (50, 2))
        self.y = np.sum(self.X**2, axis=1) + 0.1 * np.random.randn(50)
        
    def test_initialization(self):
        """Test surrogate initialization."""
        assert self.surrogate.hidden_dims == [32, 16]
        assert self.surrogate.epochs == 10
        assert self.surrogate.learning_rate == 0.01
        assert self.surrogate.params is None
        assert self.surrogate.input_dim is None
        
    def test_fit(self):
        """Test model fitting."""
        self.surrogate.fit(self.X, self.y)
        
        assert self.surrogate.params is not None
        assert self.surrogate.input_dim == 2
        assert self.surrogate.scaler_X is not None
        assert self.surrogate.scaler_y is not None
        
    def test_predict(self):
        """Test prediction."""
        self.surrogate.fit(self.X, self.y)
        
        x_test = np.array([1.0, -1.0])
        prediction = self.surrogate.predict(x_test)
        
        assert isinstance(prediction, (float, np.float64))
        assert not np.isnan(prediction)
        
    def test_predict_without_fit_raises_error(self):
        """Test that prediction without fitting raises error."""
        x_test = np.array([1.0, -1.0])
        
        with pytest.raises(ValueError, match="Model not fitted"):
            self.surrogate.predict(x_test)
            
    def test_gradient(self):
        """Test gradient computation."""
        self.surrogate.fit(self.X, self.y)
        
        x_test = np.array([1.0, -1.0])
        gradient = self.surrogate.gradient(x_test)
        
        assert isinstance(gradient, np.ndarray)
        assert gradient.shape == (2,)
        assert not np.any(np.isnan(gradient))
        
    def test_gradient_without_fit_raises_error(self):
        """Test that gradient without fitting raises error."""
        x_test = np.array([1.0, -1.0])
        
        with pytest.raises(ValueError, match="Model not fitted"):
            self.surrogate.gradient(x_test)
            
    def test_predict_batch(self):
        """Test batch prediction."""
        self.surrogate.fit(self.X, self.y)
        
        X_test = np.random.uniform(-1, 1, (10, 2))
        predictions = self.surrogate.predict_batch(X_test)
        
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (10,)
        assert not np.any(np.isnan(predictions))
        
    def test_gradient_batch(self):
        """Test batch gradient computation."""
        self.surrogate.fit(self.X, self.y)
        
        X_test = np.random.uniform(-1, 1, (10, 2))
        gradients = self.surrogate.gradient_batch(X_test)
        
        assert isinstance(gradients, np.ndarray)
        assert gradients.shape == (10, 2)
        assert not np.any(np.isnan(gradients))


class TestGPSurrogate:
    """Test GPSurrogate implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.surrogate = GPSurrogate(
            kernel="rbf",
            length_scale=1.0,
            noise_level=0.1,
        )
        
        # Create test data
        np.random.seed(42)
        self.X = np.random.uniform(-2, 2, (30, 2))
        self.y = np.sum(self.X**2, axis=1) + 0.1 * np.random.randn(30)
        
    def test_initialization(self):
        """Test surrogate initialization."""
        assert self.surrogate.kernel_type == "rbf"
        assert self.surrogate.length_scale == 1.0
        assert self.surrogate.noise_level == 0.1
        assert self.surrogate.X_train is None
        assert self.surrogate.y_train is None
        
    def test_fit(self):
        """Test model fitting."""
        self.surrogate.fit(self.X, self.y)
        
        assert self.surrogate.X_train is not None
        assert self.surrogate.y_train is not None
        assert hasattr(self.surrogate.gp, 'X_train_')
        
    def test_predict(self):
        """Test prediction."""
        self.surrogate.fit(self.X, self.y)
        
        x_test = np.array([1.0, -1.0])
        prediction = self.surrogate.predict(x_test)
        
        assert isinstance(prediction, (float, np.float64))
        assert not np.isnan(prediction)
        
    def test_uncertainty(self):
        """Test uncertainty quantification."""
        self.surrogate.fit(self.X, self.y)
        
        x_test = np.array([1.0, -1.0])
        uncertainty = self.surrogate.uncertainty(x_test)
        
        assert isinstance(uncertainty, (float, np.float64))
        assert uncertainty >= 0
        assert not np.isnan(uncertainty)
        
    def test_gradient(self):
        """Test gradient computation."""
        self.surrogate.fit(self.X, self.y)
        
        x_test = np.array([1.0, -1.0])
        gradient = self.surrogate.gradient(x_test)
        
        assert isinstance(gradient, np.ndarray)
        assert gradient.shape == (2,)
        assert not np.any(np.isnan(gradient))
        
    def test_matern_kernel(self):
        """Test Matern kernel initialization."""
        surrogate = GPSurrogate(kernel="matern", nu=2.5)
        surrogate.fit(self.X, self.y)
        
        x_test = np.array([0.5, 0.5])
        prediction = surrogate.predict(x_test)
        
        assert isinstance(prediction, (float, np.float64))
        assert not np.isnan(prediction)


class TestRandomForestSurrogate:
    """Test RandomForestSurrogate implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.surrogate = RandomForestSurrogate(
            n_estimators=20,  # Small for testing
            max_depth=5,
            random_state=42,
        )
        
        # Create test data
        np.random.seed(42)
        self.X = np.random.uniform(-2, 2, (100, 2))
        self.y = np.sum(self.X**2, axis=1) + 0.1 * np.random.randn(100)
        
    def test_initialization(self):
        """Test surrogate initialization."""
        assert self.surrogate.n_estimators == 20
        assert self.surrogate.max_depth == 5
        assert self.surrogate.random_state == 42
        assert self.surrogate.X_train is None
        
    def test_fit(self):
        """Test model fitting."""
        self.surrogate.fit(self.X, self.y)
        
        assert self.surrogate.X_train is not None
        assert self.surrogate.y_train is not None
        assert hasattr(self.surrogate.rf, 'estimators_')
        
    def test_predict(self):
        """Test prediction."""
        self.surrogate.fit(self.X, self.y)
        
        x_test = np.array([1.0, -1.0])
        prediction = self.surrogate.predict(x_test)
        
        assert isinstance(prediction, (float, np.float64))
        assert not np.isnan(prediction)
        
    def test_gradient(self):
        """Test gradient computation via finite differences."""
        self.surrogate.fit(self.X, self.y)
        
        x_test = np.array([1.0, -1.0])
        gradient = self.surrogate.gradient(x_test)
        
        assert isinstance(gradient, np.ndarray)
        assert gradient.shape == (2,)
        assert not np.any(np.isnan(gradient))
        
    def test_smooth_gradient(self):
        """Test smoothed gradient computation."""
        surrogate = RandomForestSurrogate(smooth_predictions=True)
        surrogate.fit(self.X, self.y)
        
        x_test = np.array([1.0, -1.0])
        gradient = surrogate.smooth_gradient(x_test, bandwidth=0.1)
        
        assert isinstance(gradient, np.ndarray)
        assert gradient.shape == (2,)
        assert not np.any(np.isnan(gradient))


class TestHybridSurrogate:
    """Test HybridSurrogate implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create component models
        models = [
            ("nn", NeuralSurrogate(hidden_dims=[16], epochs=5)),
            ("gp", GPSurrogate()),
            ("rf", RandomForestSurrogate(n_estimators=10)),
        ]
        
        self.surrogate = HybridSurrogate(
            models=models,
            aggregation="weighted_average",
        )
        
        # Create test data
        np.random.seed(42)
        self.X = np.random.uniform(-2, 2, (50, 2))
        self.y = np.sum(self.X**2, axis=1) + 0.1 * np.random.randn(50)
        
    def test_initialization(self):
        """Test surrogate initialization."""
        assert len(self.surrogate.models) == 3
        assert self.surrogate.aggregation == "weighted_average"
        assert self.surrogate.weights is None
        assert not self.surrogate.fitted
        
    def test_fit(self):
        """Test model fitting."""
        with patch('builtins.print'):  # Suppress print statements
            self.surrogate.fit(self.X, self.y)
        
        assert self.surrogate.fitted
        assert self.surrogate.weights is not None
        assert len(self.surrogate.weights) == 3
        assert np.isclose(np.sum(self.surrogate.weights), 1.0)
        
    def test_predict(self):
        """Test prediction."""
        with patch('builtins.print'):
            self.surrogate.fit(self.X, self.y)
        
        x_test = np.array([1.0, -1.0])
        prediction = self.surrogate.predict(x_test)
        
        assert isinstance(prediction, (float, np.float64))
        assert not np.isnan(prediction)
        
    def test_gradient(self):
        """Test gradient computation."""
        with patch('builtins.print'):
            self.surrogate.fit(self.X, self.y)
        
        x_test = np.array([1.0, -1.0])
        gradient = self.surrogate.gradient(x_test)
        
        assert isinstance(gradient, np.ndarray)
        assert gradient.shape == (2,)
        assert not np.any(np.isnan(gradient))
        
    def test_uncertainty(self):
        """Test uncertainty computation."""
        with patch('builtins.print'):
            self.surrogate.fit(self.X, self.y)
        
        x_test = np.array([1.0, -1.0])
        uncertainty = self.surrogate.uncertainty(x_test)
        
        assert isinstance(uncertainty, (float, np.float64))
        assert uncertainty >= 0
        assert not np.isnan(uncertainty)


class TestSurrogateIntegration:
    """Integration tests for surrogate models."""

    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.X = np.random.uniform(-3, 3, (100, 3))
        self.y = np.sum(self.X**2, axis=1) + 0.2 * np.random.randn(100)
        
    @pytest.mark.parametrize("surrogate_class,kwargs", [
        (NeuralSurrogate, {"hidden_dims": [16], "epochs": 5}),
        (GPSurrogate, {"kernel": "rbf"}),
        (RandomForestSurrogate, {"n_estimators": 10}),
    ])
    def test_surrogate_consistency(self, surrogate_class, kwargs):
        """Test that surrogates produce consistent predictions."""
        surrogate = surrogate_class(**kwargs)
        
        with patch('builtins.print'):
            surrogate.fit(self.X, self.y)
        
        x_test = np.array([1.0, 0.5, -0.5])
        
        # Multiple predictions should be identical
        pred1 = surrogate.predict(x_test)
        pred2 = surrogate.predict(x_test)
        
        assert np.isclose(pred1, pred2, rtol=1e-10)
        
        # Gradient should be consistent
        grad1 = surrogate.gradient(x_test)
        grad2 = surrogate.gradient(x_test)
        
        assert np.allclose(grad1, grad2, rtol=1e-10)
        
    def test_surrogate_reasonable_predictions(self):
        """Test that surrogates make reasonable predictions."""
        surrogate = NeuralSurrogate(hidden_dims=[32], epochs=50)
        
        with patch('builtins.print'):
            surrogate.fit(self.X, self.y)
        
        # Test on training data
        predictions = surrogate.predict_batch(self.X[:10])
        
        # Predictions should be in reasonable range
        assert np.all(predictions >= -50)
        assert np.all(predictions <= 50)
        
        # Should approximate training targets reasonably well
        mse = np.mean((predictions - self.y[:10])**2)
        assert mse < 10.0  # Reasonable MSE threshold


@pytest.fixture
def simple_dataset():
    """Create a simple test dataset."""
    np.random.seed(123)
    X = np.random.uniform(-1, 1, (20, 2))
    y = X[:, 0]**2 + X[:, 1]**2 + 0.1 * np.random.randn(20)
    return Dataset(X=X, y=y)