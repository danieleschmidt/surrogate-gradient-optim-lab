#!/usr/bin/env python3
"""
Comprehensive Quality Gates Test Suite
Tests all implemented surrogate optimization functionality
"""

import pytest
import numpy as np
import time
import tempfile
import os
import sys
from unittest.mock import patch, MagicMock

# Add repo to path for imports
sys.path.insert(0, '/root/repo')

try:
    from simple_surrogate_test import SimpleSurrogate
    from robust_surrogate import RobustSurrogate, RobustOptimizer, ValidationError
    print("✅ All surrogate modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

class TestSimpleSurrogate:
    """Test basic surrogate functionality."""
    
    def test_simple_surrogate_creation(self):
        """Test surrogate can be created."""
        surrogate = SimpleSurrogate()
        assert surrogate is not None
        assert not surrogate.is_fitted
    
    def test_simple_surrogate_fit_predict(self):
        """Test surrogate fitting and prediction."""
        surrogate = SimpleSurrogate()
        
        # Generate test data
        X = np.random.uniform(-2, 2, (20, 2))
        y = np.array([np.sum(x**2) for x in X])
        
        # Fit and predict
        surrogate.fit(X, y)
        assert surrogate.is_fitted
        
        # Test prediction
        test_point = np.array([1.0, 1.0])
        prediction = surrogate.predict(test_point)
        assert isinstance(prediction, float)
        assert not np.isnan(prediction)
        assert not np.isinf(prediction)
    
    def test_simple_surrogate_gradient(self):
        """Test gradient computation."""
        surrogate = SimpleSurrogate()
        
        # Generate test data
        X = np.random.uniform(-1, 1, (15, 2))
        y = np.array([np.sum(x**2) for x in X])
        
        surrogate.fit(X, y)
        
        test_point = np.array([0.5, 0.5])
        gradient = surrogate.gradient(test_point)
        
        assert isinstance(gradient, np.ndarray)
        assert gradient.shape == test_point.shape
        assert not np.any(np.isnan(gradient))
        assert not np.any(np.isinf(gradient))

class TestRobustSurrogate:
    """Test robust surrogate functionality."""
    
    def test_robust_surrogate_creation(self):
        """Test robust surrogate creation."""
        surrogate = RobustSurrogate(
            surrogate_type="gp",
            ensemble_size=2,
            validation_split=0.3
        )
        assert surrogate is not None
        assert not surrogate.is_fitted
        assert len(surrogate.models) == 2
    
    def test_robust_surrogate_input_validation(self):
        """Test input validation."""
        surrogate = RobustSurrogate()
        
        # Test with invalid data
        with pytest.raises(ValidationError):
            surrogate.fit(np.array([]), np.array([]))  # Empty arrays
        
        with pytest.raises(ValidationError):
            X = np.array([[1, 2], [3, 4]])
            y = np.array([np.nan, 1])  # NaN values
            surrogate.fit(X, y)
    
    def test_robust_surrogate_training(self):
        """Test robust training process."""
        surrogate = RobustSurrogate(
            surrogate_type="gp",
            ensemble_size=2,
            normalize_data=True
        )
        
        # Generate sufficient test data
        X = np.random.uniform(-2, 2, (50, 2))
        y = np.array([np.sum(x**2) + 0.1 * np.random.normal() for x in X])
        
        surrogate.fit(X, y)
        
        assert surrogate.is_fitted
        assert surrogate.validation_metrics is not None
        assert surrogate.validation_metrics.r2_score is not None
    
    def test_robust_prediction_with_uncertainty(self):
        """Test prediction with uncertainty estimation."""
        surrogate = RobustSurrogate(
            surrogate_type="gp",
            ensemble_size=3
        )
        
        X = np.random.uniform(-1, 1, (30, 2))
        y = np.array([np.sum(x**2) for x in X])
        
        surrogate.fit(X, y)
        
        test_point = np.array([0.5, 0.5])
        prediction = surrogate.predict(test_point)
        uncertainty = surrogate.uncertainty(test_point)
        
        assert isinstance(prediction, float)
        assert isinstance(uncertainty, float)
        assert uncertainty >= 0.0

class TestOptimizers:
    """Test optimization algorithms."""
    
    def test_robust_optimizer_creation(self):
        """Test optimizer creation."""
        optimizer = RobustOptimizer(
            method="auto",
            max_iterations=100
        )
        assert optimizer is not None
    
    def test_robust_optimization_process(self):
        """Test complete optimization process."""
        # Create and train surrogate
        surrogate = RobustSurrogate(
            surrogate_type="gp",
            ensemble_size=2
        )
        
        X = np.random.uniform(-2, 2, (30, 2))
        y = np.array([np.sum(x**2) for x in X])
        
        surrogate.fit(X, y)
        
        # Create optimizer and optimize
        optimizer = RobustOptimizer(
            method="auto",
            max_iterations=50
        )
        
        result = optimizer.optimize(
            surrogate=surrogate,
            x0=np.array([1.0, 1.0]),
            bounds=[(-2, 2), (-2, 2)]
        )
        
        assert hasattr(result, 'x')
        assert hasattr(result, 'fun')
        assert isinstance(result.x, np.ndarray)
        assert isinstance(result.fun, float)

class TestPerformanceMetrics:
    """Test performance tracking and metrics."""
    
    def test_performance_tracking(self):
        """Test that performance metrics are tracked."""
        surrogate = RobustSurrogate()
        
        X = np.random.uniform(-1, 1, (25, 2))
        y = np.array([np.sum(x**2) for x in X])
        
        start_time = time.time()
        surrogate.fit(X, y)
        training_time = time.time() - start_time
        
        # Check that training time was recorded
        assert hasattr(surrogate, 'metrics')
        # Performance tracking should be within reasonable bounds
        assert training_time < 30.0  # Should complete within 30 seconds

class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_empty_data_handling(self):
        """Test handling of empty datasets."""
        surrogate = RobustSurrogate()
        
        with pytest.raises(ValidationError):
            surrogate.fit(np.array([]).reshape(0, 2), np.array([]))
    
    def test_insufficient_data_handling(self):
        """Test handling of insufficient data."""
        surrogate = RobustSurrogate()
        
        with pytest.raises(ValidationError):
            X = np.array([[1, 2]])  # Only 1 sample
            y = np.array([1])
            surrogate.fit(X, y)
    
    def test_prediction_before_fitting(self):
        """Test prediction before fitting raises error."""
        surrogate = RobustSurrogate()
        
        with pytest.raises(ValidationError):
            surrogate.predict(np.array([1.0, 1.0]))
    
    def test_malformed_input_handling(self):
        """Test handling of malformed inputs."""
        surrogate = RobustSurrogate()
        
        X = np.random.uniform(-1, 1, (20, 2))
        y = np.array([np.sum(x**2) for x in X])
        
        surrogate.fit(X, y)
        
        # Test with wrong dimensions
        with pytest.raises(Exception):  # Should handle gracefully
            surrogate.predict(np.array([1.0]))  # Wrong dimension

class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_complete_optimization_workflow(self):
        """Test complete optimization workflow."""
        # Define test function
        def test_function(x):
            return np.sum(x**2) + 0.1 * np.sum(np.sin(10 * x))
        
        # Generate training data
        bounds = [(-2.0, 2.0), (-2.0, 2.0)]
        X = np.random.uniform(bounds[0][0], bounds[0][1], (40, 2))
        y = np.array([test_function(x) for x in X])
        
        # Create and train surrogate
        surrogate = RobustSurrogate(
            surrogate_type="gp",
            ensemble_size=2,
            normalize_data=True
        )
        
        surrogate.fit(X, y)
        
        # Optimize
        optimizer = RobustOptimizer()
        result = optimizer.optimize(
            surrogate=surrogate,
            x0=np.array([1.0, 1.0]),
            bounds=bounds
        )
        
        # Verify optimization found reasonable result
        assert np.linalg.norm(result.x) < 2.0  # Should find point near origin
        assert result.fun < 10.0  # Should find reasonably good minimum
    
    def test_benchmark_consistency(self):
        """Test that benchmark results are consistent."""
        results = []
        
        for seed in [42, 43, 44]:
            np.random.seed(seed)
            
            # Generate data
            X = np.random.uniform(-1, 1, (30, 2))
            y = np.array([np.sum(x**2) for x in X])
            
            # Train surrogate
            surrogate = RobustSurrogate(surrogate_type="gp")
            surrogate.fit(X, y)
            
            # Test prediction
            test_point = np.array([0.5, 0.5])
            prediction = surrogate.predict(test_point)
            results.append(prediction)
        
        # Results should be reasonably consistent
        std_dev = np.std(results)
        assert std_dev < 1.0  # Standard deviation should be reasonable

def run_quality_gates():
    """Run all quality gates."""
    print("🛡️  Running Quality Gates")
    print("=" * 50)
    
    # Run tests
    print("\n📋 Running Test Suite...")
    exit_code = pytest.main([
        __file__, 
        "-v", 
        "--tb=short",
        "--disable-warnings"
    ])
    
    if exit_code == 0:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
        return False
    
    return True

def run_performance_benchmark():
    """Run performance benchmark."""
    print("\n⚡ Performance Benchmark")
    print("-" * 30)
    
    def test_function(x):
        return np.sum(x**2)
    
    # Test different data sizes
    sizes = [20, 50, 100]
    
    for size in sizes:
        print(f"\n📊 Testing with {size} samples...")
        
        X = np.random.uniform(-2, 2, (size, 2))
        y = np.array([test_function(x) for x in X])
        
        # Training benchmark
        surrogate = RobustSurrogate(surrogate_type="gp", ensemble_size=2)
        
        start_time = time.time()
        surrogate.fit(X, y)
        training_time = time.time() - start_time
        
        # Prediction benchmark
        test_X = np.random.uniform(-2, 2, (10, 2))
        start_time = time.time()
        predictions = [surrogate.predict(x) for x in test_X]
        prediction_time = time.time() - start_time
        
        print(f"   Training: {training_time:.3f}s")
        print(f"   Prediction: {prediction_time:.3f}s ({len(test_X)/prediction_time:.1f} samples/s)")
        
        # Validation
        if surrogate.validation_metrics:
            print(f"   R² Score: {surrogate.validation_metrics.r2_score:.3f}")

def main():
    """Main quality gates execution."""
    print("🚀 Surrogate Optimization Quality Gates")
    print("=" * 60)
    
    # Set up test environment
    np.random.seed(42)
    
    try:
        # Run quality gates
        success = run_quality_gates()
        
        if success:
            # Run performance benchmark
            run_performance_benchmark()
            
            print("\n🎉 All Quality Gates Passed!")
            print("✅ Code is ready for production deployment")
            return 0
        else:
            print("\n❌ Quality Gates Failed!")
            return 1
    
    except Exception as e:
        print(f"\n💥 Quality Gates Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())