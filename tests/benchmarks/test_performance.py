"""Performance benchmark tests."""

import time
from typing import Dict, Any

import jax
import jax.numpy as jnp
import pytest


@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Performance benchmarks for key components."""
    
    def test_gradient_computation_speed(self, benchmark):
        """Benchmark gradient computation speed."""
        def complex_function(x):
            return jnp.sum(x**4 + jnp.sin(10 * x) + jnp.cos(x**2))
        
        grad_fn = jax.grad(complex_function)
        x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Benchmark the gradient computation
        result = benchmark(grad_fn, x)
        assert jnp.isfinite(result).all()
    
    def test_array_operations_speed(self, benchmark):
        """Benchmark large array operations."""
        key = jax.random.PRNGKey(42)
        n = 10000
        x = jax.random.normal(key, (n,))
        
        def array_ops(x):
            return jnp.sum(x**2) + jnp.mean(jnp.sin(x))
        
        result = benchmark(array_ops, x)
        assert jnp.isfinite(result)
    
    def test_matrix_multiplication_speed(self, benchmark):
        """Benchmark matrix multiplication."""
        key = jax.random.PRNGKey(42)
        A = jax.random.normal(key, (500, 500))
        B = jax.random.normal(key, (500, 500))
        
        def matmul(A, B):
            return A @ B
        
        result = benchmark(matmul, A, B)
        assert result.shape == (500, 500)
    
    @pytest.mark.slow
    def test_memory_usage(self):
        """Test memory usage patterns."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create large arrays
        key = jax.random.PRNGKey(42)
        large_arrays = []
        for i in range(10):
            arr = jax.random.normal(key, (1000, 1000))
            large_arrays.append(arr)
        
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = current_memory - initial_memory
        
        # Should not use excessive memory (threshold: 500MB)
        assert memory_increase < 500, f"Memory usage increased by {memory_increase:.1f}MB"
        
        # Clean up
        del large_arrays
    
    def test_compilation_time(self):
        """Test JIT compilation overhead."""
        def test_function(x):
            return jnp.sum(x**3 + jnp.sin(x))
        
        # Time JIT compilation
        x = jnp.array([1.0, 2.0, 3.0])
        
        start_time = time.time()
        jit_fn = jax.jit(test_function)
        # First call triggers compilation
        result1 = jit_fn(x)
        compile_time = time.time() - start_time
        
        # Time subsequent calls
        start_time = time.time()
        result2 = jit_fn(x)
        execution_time = time.time() - start_time
        
        assert jnp.allclose(result1, result2)
        assert execution_time < compile_time  # Compiled version should be faster
        assert compile_time < 1.0  # Compilation should complete quickly


@pytest.mark.benchmark
class TestScalabilityBenchmarks:
    """Test scalability with increasing problem sizes."""
    
    @pytest.mark.parametrize("n_dims", [2, 5, 10, 20, 50])
    def test_gradient_scaling(self, n_dims):
        """Test gradient computation scaling with dimensionality."""
        def test_function(x):
            return jnp.sum(x**2) + 0.1 * jnp.sum(x**4)
        
        grad_fn = jax.grad(test_function)
        x = jnp.ones(n_dims)
        
        start_time = time.time()
        gradient = grad_fn(x)
        execution_time = time.time() - start_time
        
        assert gradient.shape == (n_dims,)
        assert jnp.isfinite(gradient).all()
        
        # Execution time should scale reasonably
        expected_max_time = 0.1 * (n_dims / 10)  # Linear scaling assumption
        assert execution_time < expected_max_time
    
    @pytest.mark.parametrize("n_samples", [100, 500, 1000, 5000])
    def test_data_processing_scaling(self, n_samples):
        """Test data processing scaling with sample size."""
        key = jax.random.PRNGKey(42)
        X = jax.random.normal(key, (n_samples, 5))
        
        def process_data(X):
            # Simulate typical data preprocessing
            X_normalized = (X - jnp.mean(X, axis=0)) / jnp.std(X, axis=0)
            return jnp.sum(X_normalized**2)
        
        start_time = time.time()
        result = process_data(X)
        execution_time = time.time() - start_time
        
        assert jnp.isfinite(result)
        
        # Should scale reasonably with data size
        expected_max_time = 0.01 * (n_samples / 1000)
        assert execution_time < expected_max_time


@pytest.mark.benchmark
class TestNumericalAccuracyBenchmarks:
    """Test numerical accuracy under various conditions."""
    
    def test_gradient_accuracy(self):
        """Test gradient accuracy against analytical solutions."""
        def polynomial(x):
            return x[0]**3 + 2*x[1]**2 + 3*x[0]*x[1]
        
        def analytical_gradient(x):
            return jnp.array([
                3*x[0]**2 + 3*x[1],  # ∂f/∂x₀
                4*x[1] + 3*x[0]       # ∂f/∂x₁
            ])
        
        grad_fn = jax.grad(polynomial)
        
        # Test at multiple points
        test_points = [
            jnp.array([1.0, 2.0]),
            jnp.array([-1.0, 0.5]),
            jnp.array([0.0, 0.0]),
            jnp.array([10.0, -5.0])
        ]
        
        for x in test_points:
            computed_grad = grad_fn(x)
            analytical_grad = analytical_gradient(x)
            
            relative_error = jnp.abs(computed_grad - analytical_grad) / (jnp.abs(analytical_grad) + 1e-10)
            assert jnp.all(relative_error < 1e-10), f"High gradient error at {x}: {relative_error}"
    
    def test_numerical_stability_extreme_values(self):
        """Test numerical stability with extreme values."""
        def stable_function(x):
            # Function that should remain stable
            return jnp.log1p(jnp.exp(jnp.clip(x, -10, 10)))
        
        grad_fn = jax.grad(stable_function)
        
        # Test extreme values
        extreme_values = [1e-10, 1e10, -1e10, jnp.pi, -jnp.pi]
        
        for val in extreme_values:
            x = jnp.array([val])
            try:
                result = stable_function(x)
                gradient = grad_fn(x)
                
                assert jnp.isfinite(result), f"Function not finite at {val}"
                assert jnp.isfinite(gradient).all(), f"Gradient not finite at {val}"
                
            except Exception as e:
                pytest.fail(f"Function failed at extreme value {val}: {e}")
    
    def test_condition_number_stability(self):
        """Test behavior with ill-conditioned matrices."""
        # Create an ill-conditioned matrix
        key = jax.random.PRNGKey(42)
        A = jax.random.normal(key, (10, 10))
        
        # Make it ill-conditioned by adding a near-zero eigenvalue
        U, S, Vt = jnp.linalg.svd(A)
        S = S.at[-1].set(1e-12)  # Very small singular value
        A_ill = U @ jnp.diag(S) @ Vt
        
        # Test that we can handle this gracefully
        try:
            cond_number = jnp.linalg.cond(A_ill)
            assert cond_number > 1e10  # Verify it's ill-conditioned
            
            # Test pseudo-inverse for stability
            A_pinv = jnp.linalg.pinv(A_ill)
            assert jnp.isfinite(A_pinv).all()
            
        except Exception as e:
            pytest.fail(f"Failed to handle ill-conditioned matrix: {e}")


def test_benchmark_infrastructure():
    """Test that benchmark infrastructure is working."""
    def simple_function():
        return sum(range(1000))
    
    # Test that we can time functions
    start_time = time.time()
    result = simple_function()
    execution_time = time.time() - start_time
    
    assert result == 499500  # Expected sum
    assert execution_time >= 0  # Time should be non-negative
    assert execution_time < 1.0  # Should complete quickly


if __name__ == "__main__":
    # Allow running benchmarks directly
    pytest.main([__file__, "-v", "-m", "benchmark"])