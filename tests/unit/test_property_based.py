"""Property-based testing for surrogate optimization components."""

import jax
import jax.numpy as jnp
import pytest
from hypothesis import given, strategies as st, settings
from hypothesis.extra.numpy import arrays


# Custom strategies for JAX arrays
def jax_array_strategy(shape, min_value=-100.0, max_value=100.0, dtype=jnp.float32):
    """Generate JAX arrays with specified properties."""
    return arrays(
        dtype=dtype,
        shape=shape,
        elements=st.floats(min_value=min_value, max_value=max_value, allow_nan=False, allow_infinity=False)
    ).map(jnp.array)


class TestMathematicalProperties:
    """Test mathematical properties that should hold for all inputs."""
    
    @given(
        x=jax_array_strategy(shape=(5,), min_value=-10.0, max_value=10.0)
    )
    @settings(max_examples=100, deadline=None)
    def test_gradient_consistency(self, x):
        """Test that gradients are consistent across different ways of computing them."""
        def f(x):
            return jnp.sum(x**2)
        
        # Compute gradient using JAX
        grad_f = jax.grad(f)
        jax_gradient = grad_f(x)
        
        # Analytical gradient of sum(x^2) is 2*x
        analytical_gradient = 2 * x
        
        assert jnp.allclose(jax_gradient, analytical_gradient, atol=1e-6)
    
    @given(
        x=jax_array_strategy(shape=(3,), min_value=-5.0, max_value=5.0)
    )
    @settings(max_examples=50, deadline=None)
    def test_function_smoothness(self, x):
        """Test that functions are smooth (have finite gradients)."""
        def smooth_function(x):
            return jnp.sum(x**2) + jnp.sum(jnp.sin(x))
        
        grad_f = jax.grad(smooth_function)
        gradient = grad_f(x)
        
        # Gradient should be finite
        assert jnp.isfinite(gradient).all()
        
        # Gradient should not be too large
        assert jnp.linalg.norm(gradient) < 1000.0
    
    @given(
        scale=st.floats(min_value=0.1, max_value=10.0),
        x=jax_array_strategy(shape=(4,), min_value=-3.0, max_value=3.0)
    )
    @settings(max_examples=50, deadline=None)
    def test_scaling_properties(self, scale, x):
        """Test that scaling behaves correctly."""
        def f(x):
            return jnp.sum(x**2)
        
        # Scaling input should scale output predictably
        original_value = f(x)
        scaled_value = f(scale * x)
        
        # For quadratic function, scaling by α should scale output by α²
        expected_scaled = (scale**2) * original_value
        assert jnp.allclose(scaled_value, expected_scaled, rtol=1e-6)
    
    @given(
        x=jax_array_strategy(shape=(3,), min_value=-2.0, max_value=2.0),
        y=jax_array_strategy(shape=(3,), min_value=-2.0, max_value=2.0)
    )
    @settings(max_examples=50, deadline=None)
    def test_additivity_properties(self, x, y):
        """Test additivity properties for linear functions."""
        def linear_function(z):
            weights = jnp.array([1.0, 2.0, 3.0])
            return jnp.dot(weights, z)
        
        # For linear functions: f(x + y) = f(x) + f(y)
        f_x = linear_function(x)
        f_y = linear_function(y)
        f_sum = linear_function(x + y)
        
        assert jnp.allclose(f_sum, f_x + f_y, atol=1e-6)


class TestOptimizationProperties:
    """Test properties specific to optimization functions."""
    
    @given(
        x=jax_array_strategy(shape=(2,), min_value=-5.0, max_value=5.0)
    )
    @settings(max_examples=50, deadline=None)
    def test_convex_function_properties(self, x):
        """Test properties of convex functions."""
        def convex_function(x):
            return jnp.sum(x**2)  # Quadratic is convex
        
        # Gradient should point away from origin for positive definite quadratic
        if jnp.linalg.norm(x) > 1e-6:  # Avoid numerical issues at origin
            grad_f = jax.grad(convex_function)
            gradient = grad_f(x)
            
            # For f(x) = ||x||², gradient is 2x, which points away from origin
            direction_check = jnp.dot(x, gradient)
            assert direction_check >= 0  # Should be non-negative
    
    @given(
        x=jax_array_strategy(shape=(2,), min_value=-3.0, max_value=3.0)
    )
    @settings(max_examples=50, deadline=None)
    def test_benchmark_function_bounds(self, x):
        """Test that benchmark functions produce reasonable values."""
        def rosenbrock(x):
            return jnp.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
        
        value = rosenbrock(x)
        
        # Rosenbrock function should be non-negative
        assert value >= 0
        
        # Should be finite
        assert jnp.isfinite(value)
        
        # Should not be too large for reasonable inputs
        assert value < 1e6


class TestNumericalStability:
    """Test numerical stability properties."""
    
    @given(
        x=jax_array_strategy(shape=(4,), min_value=1e-8, max_value=1e8, dtype=jnp.float64)
    )
    @settings(max_examples=50, deadline=None)
    def test_numerical_precision(self, x):
        """Test numerical precision with extreme values."""
        # Test that basic operations remain stable
        y = jnp.log(jnp.abs(x) + 1e-10)  # Add small constant to avoid log(0)
        
        assert jnp.isfinite(y).all()
        assert not jnp.isnan(y).any()
    
    @given(
        matrix_size=st.integers(min_value=2, max_value=5),
        condition_number=st.floats(min_value=1.1, max_value=100.0)
    )
    @settings(max_examples=20, deadline=None)
    def test_matrix_conditioning(self, matrix_size, condition_number):
        """Test matrix operations with different condition numbers."""
        # Create a well-conditioned matrix
        key = jax.random.PRNGKey(42)
        A = jax.random.normal(key, (matrix_size, matrix_size))
        
        # Make it positive definite
        A = A @ A.T + jnp.eye(matrix_size) * 1e-6
        
        # Test that we can compute eigenvalues
        eigenvals = jnp.linalg.eigvals(A)
        
        assert jnp.isfinite(eigenvals).all()
        assert (eigenvals > 0).all()  # Should be positive definite
    
    @given(
        x=jax_array_strategy(shape=(3,), min_value=-100.0, max_value=100.0)
    )
    @settings(max_examples=50, deadline=None)
    def test_gradient_magnitude_bounds(self, x):
        """Test that gradients have reasonable magnitudes."""
        def polynomial_function(x):
            return jnp.sum(x**4 + x**2 + x)
        
        grad_f = jax.grad(polynomial_function)
        gradient = grad_f(x)
        
        # Gradient should be finite
        assert jnp.isfinite(gradient).all()
        
        # Gradient magnitude should be reasonable
        grad_norm = jnp.linalg.norm(gradient)
        assert grad_norm < 1e6  # Not too large
        
        # For non-zero input, gradient should generally be non-zero
        if jnp.linalg.norm(x) > 1e-3:
            assert grad_norm > 1e-10  # Not too small


class TestSymmetryProperties:
    """Test symmetry properties that functions should satisfy."""
    
    @given(
        x=jax_array_strategy(shape=(4,), min_value=-5.0, max_value=5.0)
    )
    @settings(max_examples=50, deadline=None)
    def test_even_function_symmetry(self, x):
        """Test symmetry properties of even functions."""
        def even_function(x):
            return jnp.sum(x**2) + jnp.sum(x**4)
        
        # Even function: f(-x) = f(x)
        f_x = even_function(x)
        f_neg_x = even_function(-x)
        
        assert jnp.allclose(f_x, f_neg_x, rtol=1e-6)
    
    @given(
        permutation_seed=st.integers(min_value=0, max_value=1000),
        x=jax_array_strategy(shape=(4,), min_value=-3.0, max_value=3.0)
    )
    @settings(max_examples=30, deadline=None)
    def test_permutation_invariance(self, permutation_seed, x):
        """Test functions that should be invariant to permutations."""
        def permutation_invariant_function(x):
            return jnp.sum(x**2)  # Sum is permutation invariant
        
        # Create a random permutation
        key = jax.random.PRNGKey(permutation_seed)
        perm = jax.random.permutation(key, jnp.arange(len(x)))
        x_permuted = x[perm]
        
        # Function value should be the same
        f_original = permutation_invariant_function(x)
        f_permuted = permutation_invariant_function(x_permuted)
        
        assert jnp.allclose(f_original, f_permuted, rtol=1e-6)


class TestInterpolationProperties:
    """Test properties relevant to surrogate model interpolation."""
    
    @given(
        alpha=st.floats(min_value=0.0, max_value=1.0),
        x1=jax_array_strategy(shape=(3,), min_value=-2.0, max_value=2.0),
        x2=jax_array_strategy(shape=(3,), min_value=-2.0, max_value=2.0)
    )
    @settings(max_examples=50, deadline=None)
    def test_convex_combination_bounds(self, alpha, x1, x2):
        """Test that convex combinations preserve boundedness."""
        def bounded_function(x):
            # Function bounded between 0 and some maximum
            return jnp.sum(jnp.tanh(x)**2)  # tanh² ∈ [0, 1]
        
        # Convex combination
        x_combined = alpha * x1 + (1 - alpha) * x2
        
        f1 = bounded_function(x1)
        f2 = bounded_function(x2)
        f_combined = bounded_function(x_combined)
        
        # All values should be in [0, len(x)]
        assert 0 <= f1 <= len(x1)
        assert 0 <= f2 <= len(x2)
        assert 0 <= f_combined <= len(x_combined)
    
    @given(
        x=jax_array_strategy(shape=(3,), min_value=-1.0, max_value=1.0),
        epsilon=st.floats(min_value=1e-6, max_value=1e-3)
    )
    @settings(max_examples=30, deadline=None)
    def test_lipschitz_continuity(self, x, epsilon):
        """Test Lipschitz continuity for smooth functions."""
        def lipschitz_function(x):
            return jnp.sum(jnp.tanh(x))  # tanh is Lipschitz continuous
        
        # Small perturbation
        key = jax.random.PRNGKey(123)
        perturbation = epsilon * jax.random.normal(key, x.shape)
        x_perturbed = x + perturbation
        
        f_x = lipschitz_function(x)
        f_x_pert = lipschitz_function(x_perturbed)
        
        # Function change should be bounded by Lipschitz constant times input change
        func_change = abs(f_x_pert - f_x)
        input_change = jnp.linalg.norm(perturbation)
        
        # tanh has Lipschitz constant 1, so for sum of tanh's, L ≤ len(x)
        lipschitz_bound = len(x) * input_change
        
        assert func_change <= lipschitz_bound + 1e-10  # Small numerical tolerance


if __name__ == "__main__":
    pytest.main([__file__])