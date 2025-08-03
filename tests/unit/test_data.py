"""Unit tests for data collection and management."""

import numpy as np
import pytest
from unittest.mock import Mock, patch

from surrogate_optim.data import (
    Dataset,
    DataCollector,
    ActiveLearner,
    collect_data,
)


class TestDataset:
    """Test Dataset implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.X = np.random.uniform(-2, 2, (20, 3))
        self.y = np.sum(self.X**2, axis=1) + 0.1 * np.random.randn(20)
        self.gradients = 2 * self.X + 0.05 * np.random.randn(20, 3)
        
    def test_initialization_basic(self):
        """Test basic dataset initialization."""
        dataset = Dataset(X=self.X, y=self.y)
        
        assert dataset.n_samples == 20
        assert dataset.n_dims == 3
        assert np.array_equal(dataset.X, self.X)
        assert np.array_equal(dataset.y, self.y)
        assert dataset.gradients is None
        
    def test_initialization_with_gradients(self):
        """Test dataset initialization with gradients."""
        dataset = Dataset(X=self.X, y=self.y, gradients=self.gradients)
        
        assert dataset.n_samples == 20
        assert dataset.n_dims == 3
        assert np.array_equal(dataset.gradients, self.gradients)
        
    def test_initialization_with_metadata(self):
        """Test dataset initialization with metadata."""
        metadata = {"sampling": "sobol", "function": "rosenbrock"}
        dataset = Dataset(X=self.X, y=self.y, metadata=metadata)
        
        assert dataset.metadata == metadata
        
    def test_mismatched_dimensions_raises_error(self):
        """Test that mismatched X and y dimensions raise error."""
        y_wrong = self.y[:-5]  # Remove 5 samples
        
        with pytest.raises(ValueError, match="X and y must have same number of samples"):
            Dataset(X=self.X, y=y_wrong)
            
    def test_mismatched_gradients_raises_error(self):
        """Test that mismatched gradients shape raises error."""
        gradients_wrong = self.gradients[:-5]  # Remove 5 samples
        
        with pytest.raises(ValueError, match="Gradients must have same shape as X"):
            Dataset(X=self.X, y=self.y, gradients=gradients_wrong)
            
    def test_add_samples(self):
        """Test adding new samples to dataset."""
        dataset = Dataset(X=self.X, y=self.y)
        
        X_new = np.random.uniform(-1, 1, (5, 3))
        y_new = np.sum(X_new**2, axis=1)
        
        dataset.add_samples(X_new, y_new)
        
        assert dataset.n_samples == 25
        assert dataset.X.shape == (25, 3)
        assert dataset.y.shape == (25,)
        
        # Check that new data was appended
        assert np.array_equal(dataset.X[-5:], X_new)
        assert np.array_equal(dataset.y[-5:], y_new)
        
    def test_add_samples_with_gradients(self):
        """Test adding samples with gradients."""
        dataset = Dataset(X=self.X, y=self.y, gradients=self.gradients)
        
        X_new = np.random.uniform(-1, 1, (3, 3))
        y_new = np.sum(X_new**2, axis=1)
        gradients_new = 2 * X_new
        
        dataset.add_samples(X_new, y_new, gradients_new)
        
        assert dataset.n_samples == 23
        assert dataset.gradients.shape == (23, 3)
        assert np.array_equal(dataset.gradients[-3:], gradients_new)
        
    def test_get_bounds(self):
        """Test getting dataset bounds."""
        dataset = Dataset(X=self.X, y=self.y)
        bounds = dataset.get_bounds()
        
        assert len(bounds) == 3  # 3 dimensions
        
        for i, (lower, upper) in enumerate(bounds):
            assert lower <= np.min(self.X[:, i])
            assert upper >= np.max(self.X[:, i])
            assert lower < upper


class TestDataCollector:
    """Test DataCollector implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        def test_function(x):
            """Simple quadratic test function."""
            x = np.asarray(x)
            return np.sum(x**2)
        
        self.function = test_function
        self.collector = DataCollector(self.function)
        self.bounds = [(-2.0, 2.0), (-2.0, 2.0)]
        
    def test_initialization(self):
        """Test collector initialization."""
        assert self.collector.function is self.function
        assert self.collector.evaluations == 0
        
    def test_collect_uniform(self):
        """Test uniform random sampling."""
        dataset = self.collector.collect_uniform(
            bounds=self.bounds,
            n_samples=50,
            random_state=42
        )
        
        assert isinstance(dataset, Dataset)
        assert dataset.n_samples == 50
        assert dataset.n_dims == 2
        assert self.collector.evaluations == 50
        assert dataset.metadata["sampling"] == "uniform"
        
        # Check bounds are respected
        bounds_array = np.array(self.bounds)
        assert np.all(dataset.X >= bounds_array[:, 0])
        assert np.all(dataset.X <= bounds_array[:, 1])
        
    def test_collect_sobol(self):
        """Test Sobol quasi-random sampling."""
        dataset = self.collector.collect_sobol(
            bounds=self.bounds,
            n_samples=64,  # Power of 2 for Sobol
            random_state=123
        )
        
        assert isinstance(dataset, Dataset)
        assert dataset.n_samples == 64
        assert dataset.n_dims == 2
        assert dataset.metadata["sampling"] == "sobol"
        
        # Check bounds are respected
        bounds_array = np.array(self.bounds)
        assert np.all(dataset.X >= bounds_array[:, 0])
        assert np.all(dataset.X <= bounds_array[:, 1])
        
        # Sobol sequences should have better space-filling properties
        # (simple test: check that points are well-distributed)
        distances = []
        for i in range(min(10, dataset.n_samples)):
            for j in range(i+1, min(10, dataset.n_samples)):
                dist = np.linalg.norm(dataset.X[i] - dataset.X[j])
                distances.append(dist)
        
        min_distance = np.min(distances)
        assert min_distance > 0  # No duplicate points
        
    def test_collect_latin_hypercube(self):
        """Test Latin Hypercube sampling."""
        dataset = self.collector.collect_latin_hypercube(
            bounds=self.bounds,
            n_samples=25,
            random_state=456
        )
        
        assert isinstance(dataset, Dataset)
        assert dataset.n_samples == 25
        assert dataset.n_dims == 2
        assert dataset.metadata["sampling"] == "latin_hypercube"
        
        # Check bounds are respected
        bounds_array = np.array(self.bounds)
        assert np.all(dataset.X >= bounds_array[:, 0])
        assert np.all(dataset.X <= bounds_array[:, 1])
        
    def test_collect_adaptive(self):
        """Test adaptive sampling."""
        dataset = self.collector.collect_adaptive(
            bounds=self.bounds,
            initial_samples=10,
            acquisition_function="expected_improvement",
            batch_size=5,
            n_iterations=3,
            random_state=789
        )
        
        assert isinstance(dataset, Dataset)
        # Should have initial + (batch_size * n_iterations) samples
        expected_samples = 10 + 5 * 3
        assert dataset.n_samples == expected_samples
        assert dataset.metadata["sampling"] == "adaptive"
        
        # Should have evaluated function many times
        assert self.collector.evaluations >= expected_samples
        
    def test_acquisition_functions(self):
        """Test different acquisition functions."""
        # Create a simple surrogate for testing
        surrogate = Mock()
        surrogate.predict.return_value = 1.0
        surrogate.uncertainty.return_value = 0.5
        
        candidates = np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0]])
        
        # Test Expected Improvement
        ei_values = self.collector._compute_acquisition(
            candidates, surrogate, "expected_improvement"
        )
        assert len(ei_values) == 3
        assert all(val >= 0 for val in ei_values)
        
        # Test Upper Confidence Bound
        ucb_values = self.collector._compute_acquisition(
            candidates, surrogate, "upper_confidence_bound"
        )
        assert len(ucb_values) == 3
        
        # Test Probability of Improvement
        pi_values = self.collector._compute_acquisition(
            candidates, surrogate, "probability_improvement"
        )
        assert len(pi_values) == 3
        assert all(0 <= val <= 1 for val in pi_values)
        
    def test_estimate_gradients(self):
        """Test gradient estimation."""
        dataset = self.collector.collect_uniform(
            bounds=self.bounds,
            n_samples=10,
            random_state=42
        )
        
        initial_evaluations = self.collector.evaluations
        
        dataset_with_grads = self.collector.estimate_gradients(
            dataset,
            method="finite_differences",
            epsilon=1e-4
        )
        
        assert dataset_with_grads.gradients is not None
        assert dataset_with_grads.gradients.shape == (10, 2)
        
        # Should have made additional function evaluations for gradients
        assert self.collector.evaluations > initial_evaluations
        
        # Gradients should be approximately correct for quadratic function
        # True gradient is 2*x
        for i in range(5):  # Check first 5 points
            x = dataset.X[i]
            true_grad = 2 * x
            estimated_grad = dataset_with_grads.gradients[i]
            
            # Should be close (within finite difference error)
            assert np.allclose(estimated_grad, true_grad, atol=1e-2)


class TestActiveLearner:
    """Test ActiveLearner implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        def test_function(x):
            """Test function with some complexity."""
            x = np.asarray(x)
            return np.sum(x**2) + 0.1 * np.sin(10 * np.sum(x))
        
        self.function = test_function
        
        # Create initial dataset
        np.random.seed(42)
        X_initial = np.random.uniform(-1, 1, (10, 2))
        y_initial = np.array([self.function(x) for x in X_initial])
        self.initial_data = Dataset(X=X_initial, y=y_initial)
        
        self.learner = ActiveLearner(
            function=self.function,
            initial_data=self.initial_data,
            surrogate_type="gp"  # Use GP for uncertainty
        )
        
    def test_initialization(self):
        """Test active learner initialization."""
        assert self.learner.function is self.function
        assert self.learner.data is self.initial_data
        assert self.learner.surrogate_type == "gp"
        assert self.learner.surrogate is None
        assert self.learner.evaluations == 0
        
    def test_learn_iteratively(self):
        """Test iterative active learning."""
        bounds = [(-2.0, 2.0), (-2.0, 2.0)]
        
        with patch('builtins.print'):  # Suppress print statements
            updated_data = self.learner.learn_iteratively(
                n_iterations=3,
                batch_size=2,
                acquisition_function="expected_improvement",
                bounds=bounds
            )
        
        # Should have added 3 * 2 = 6 new samples
        expected_samples = 10 + 3 * 2
        assert updated_data.n_samples == expected_samples
        assert self.learner.evaluations == 6
        
        # Data should be within bounds
        bounds_array = np.array(bounds)
        assert np.all(updated_data.X >= bounds_array[:, 0])
        assert np.all(updated_data.X <= bounds_array[:, 1])
        
    def test_fit_surrogate(self):
        """Test surrogate fitting."""
        with patch('builtins.print'):
            self.learner._fit_surrogate()
        
        assert self.learner.surrogate is not None
        
        # Test prediction
        x_test = np.array([0.5, 0.5])
        prediction = self.learner.surrogate.predict(x_test)
        assert isinstance(prediction, (float, np.float64))
        
    def test_different_surrogate_types(self):
        """Test different surrogate types."""
        surrogate_types = ["neural_network", "gp", "random_forest"]
        
        for surrogate_type in surrogate_types:
            learner = ActiveLearner(
                function=self.function,
                initial_data=self.initial_data,
                surrogate_type=surrogate_type
            )
            
            with patch('builtins.print'):
                learner._fit_surrogate()
            
            assert learner.surrogate is not None
            
    def test_different_acquisition_functions(self):
        """Test different acquisition functions."""
        bounds = [(-1.0, 1.0), (-1.0, 1.0)]
        acquisition_functions = [
            "expected_improvement",
            "upper_confidence_bound", 
            "entropy_search"
        ]
        
        for acq_func in acquisition_functions:
            learner = ActiveLearner(
                function=self.function,
                initial_data=self.initial_data,
                surrogate_type="gp"
            )
            
            with patch('builtins.print'):
                updated_data = learner.learn_iteratively(
                    n_iterations=1,
                    batch_size=1,
                    acquisition_function=acq_func,
                    bounds=bounds
                )
            
            # Should have added 1 new sample
            assert updated_data.n_samples == 11


class TestCollectDataFunction:
    """Test collect_data convenience function."""

    def setup_method(self):
        """Set up test fixtures."""
        def test_function(x):
            x = np.asarray(x)
            return np.sum(x**2)
        
        self.function = test_function
        self.bounds = [(-1.0, 1.0), (-1.0, 1.0)]
        
    def test_collect_data_uniform(self):
        """Test collect_data with uniform sampling."""
        dataset = collect_data(
            function=self.function,
            n_samples=30,
            bounds=self.bounds,
            sampling="uniform",
            random_state=123
        )
        
        assert isinstance(dataset, Dataset)
        assert dataset.n_samples == 30
        assert dataset.n_dims == 2
        
    def test_collect_data_sobol(self):
        """Test collect_data with Sobol sampling."""
        dataset = collect_data(
            function=self.function,
            n_samples=32,
            bounds=self.bounds,
            sampling="sobol",
            random_state=456
        )
        
        assert isinstance(dataset, Dataset)
        assert dataset.n_samples == 32
        assert dataset.n_dims == 2
        
    def test_collect_data_latin_hypercube(self):
        """Test collect_data with Latin hypercube sampling."""
        dataset = collect_data(
            function=self.function,
            n_samples=25,
            bounds=self.bounds,
            sampling="latin_hypercube",
            random_state=789
        )
        
        assert isinstance(dataset, Dataset)
        assert dataset.n_samples == 25
        assert dataset.n_dims == 2
        
    def test_collect_data_adaptive(self):
        """Test collect_data with adaptive sampling."""
        dataset = collect_data(
            function=self.function,
            n_samples=25,  # Will use 5 initial + 4*5 adaptive
            bounds=self.bounds,
            sampling="adaptive",
            batch_size=5,
            n_iterations=4,
            random_state=101112
        )
        
        assert isinstance(dataset, Dataset)
        assert dataset.n_samples == 25
        assert dataset.n_dims == 2
        
    def test_collect_data_unknown_sampling(self):
        """Test error for unknown sampling strategy."""
        with pytest.raises(ValueError, match="Unknown sampling strategy"):
            collect_data(
                function=self.function,
                n_samples=10,
                bounds=self.bounds,
                sampling="unknown"
            )


class TestDataIntegration:
    """Integration tests for data module."""

    def test_end_to_end_data_collection_and_learning(self):
        """Test end-to-end data collection and active learning."""
        def complex_function(x):
            """More complex test function."""
            x = np.asarray(x)
            return -(x[0]**2 + x[1]**2) + 0.5 * np.sin(5 * x[0]) * np.cos(5 * x[1])
        
        bounds = [(-2.0, 2.0), (-2.0, 2.0)]
        
        # Initial data collection
        initial_data = collect_data(
            function=complex_function,
            n_samples=20,
            bounds=bounds,
            sampling="sobol",
            random_state=42
        )
        
        # Active learning
        learner = ActiveLearner(
            function=complex_function,
            initial_data=initial_data,
            surrogate_type="gp"
        )
        
        with patch('builtins.print'):
            final_data = learner.learn_iteratively(
                n_iterations=5,
                batch_size=3,
                acquisition_function="expected_improvement",
                bounds=bounds
            )
        
        # Should have improved data distribution
        assert final_data.n_samples == 20 + 5 * 3
        assert final_data.n_dims == 2
        
        # Data should explore the function space reasonably
        y_range = np.max(final_data.y) - np.min(final_data.y)
        assert y_range > 1.0  # Should have explored different regions
        
    def test_data_quality_metrics(self):
        """Test data quality and distribution."""
        def rosenbrock(x):
            """Rosenbrock function."""
            x = np.asarray(x)
            return -np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
        
        bounds = [(-2.0, 2.0), (-2.0, 2.0)]
        
        # Compare different sampling strategies
        strategies = ["uniform", "sobol", "latin_hypercube"]
        datasets = {}
        
        for strategy in strategies:
            datasets[strategy] = collect_data(
                function=rosenbrock,
                n_samples=50,
                bounds=bounds,
                sampling=strategy,
                random_state=42
            )
        
        # All should have same size and bounds
        for strategy, dataset in datasets.items():
            assert dataset.n_samples == 50
            assert dataset.n_dims == 2
            
            bounds_array = np.array(bounds)
            assert np.all(dataset.X >= bounds_array[:, 0])
            assert np.all(dataset.X <= bounds_array[:, 1])
            
        # Check space-filling properties (simplified test)
        for strategy, dataset in datasets.items():
            # Compute minimum pairwise distance
            min_distances = []
            for i in range(min(20, dataset.n_samples)):
                distances = [
                    np.linalg.norm(dataset.X[i] - dataset.X[j])
                    for j in range(dataset.n_samples) if i != j
                ]
                min_distances.append(np.min(distances))
            
            avg_min_distance = np.mean(min_distances)
            assert avg_min_distance > 0.1  # Reasonable space-filling


@pytest.fixture
def simple_quadratic_function():
    """Simple quadratic function for testing."""
    def func(x):
        x = np.asarray(x)
        return np.sum(x**2)
    return func


@pytest.fixture
def sample_dataset():
    """Sample dataset for testing."""
    np.random.seed(42)
    X = np.random.uniform(-1, 1, (15, 2))
    y = np.sum(X**2, axis=1) + 0.1 * np.random.randn(15)
    return Dataset(X=X, y=y)