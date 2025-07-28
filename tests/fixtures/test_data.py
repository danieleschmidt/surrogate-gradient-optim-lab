"""Test data generation utilities."""

from typing import Dict, List, Tuple, Optional, Callable
import jax
import jax.numpy as jnp
from jax import random


def generate_synthetic_data(
    function: Callable,
    n_samples: int,
    bounds: List[Tuple[float, float]],
    noise_level: float = 0.0,
    sampling_method: str = "random",
    random_seed: int = 42
) -> Dict[str, jnp.ndarray]:
    """Generate synthetic training data for testing.
    
    Args:
        function: Function to evaluate at sample points
        n_samples: Number of sample points to generate
        bounds: List of (min, max) bounds for each dimension
        noise_level: Standard deviation of Gaussian noise to add
        sampling_method: Method to use for sampling ("random", "sobol", "grid")
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary containing X (inputs), y (outputs), y_true (noiseless), noise
    """
    key = random.PRNGKey(random_seed)
    n_dims = len(bounds)
    
    # Generate sample points
    if sampling_method == "random":
        X = generate_random_samples(key, n_samples, bounds)
    elif sampling_method == "sobol":
        X = generate_sobol_samples(n_samples, bounds)
    elif sampling_method == "grid":
        X = generate_grid_samples(n_samples, bounds)
    else:
        raise ValueError(f"Unknown sampling method: {sampling_method}")
    
    # Evaluate function
    y_true = jnp.array([function(x) for x in X])
    
    # Add noise if requested
    if noise_level > 0:
        noise_key = random.split(key)[1]
        noise = noise_level * random.normal(noise_key, (n_samples,))
        y = y_true + noise
    else:
        noise = jnp.zeros(n_samples)
        y = y_true
    
    return {
        "X": X,
        "y": y,
        "y_true": y_true,
        "noise": noise,
        "bounds": bounds,
        "n_samples": n_samples,
        "noise_level": noise_level,
        "sampling_method": sampling_method
    }


def generate_random_samples(
    key: jax.Array,
    n_samples: int,
    bounds: List[Tuple[float, float]]
) -> jnp.ndarray:
    """Generate random samples within bounds."""
    n_dims = len(bounds)
    
    # Extract bounds
    lower_bounds = jnp.array([b[0] for b in bounds])
    upper_bounds = jnp.array([b[1] for b in bounds])
    
    # Generate uniform samples
    samples = random.uniform(
        key,
        (n_samples, n_dims),
        minval=lower_bounds,
        maxval=upper_bounds
    )
    
    return samples


def generate_sobol_samples(
    n_samples: int,
    bounds: List[Tuple[float, float]]
) -> jnp.ndarray:
    """Generate Sobol sequence samples (mock implementation)."""
    # This is a simplified mock - real implementation would use scipy.stats.qmc
    key = random.PRNGKey(0)  # Fixed seed for deterministic Sobol-like sequence
    n_dims = len(bounds)
    
    # Generate stratified samples as approximation
    samples_per_dim = int(jnp.ceil(n_samples ** (1.0 / n_dims)))
    
    # Create grid points
    grid_points = []
    for i in range(n_dims):
        low, high = bounds[i]
        points = jnp.linspace(low, high, samples_per_dim)
        grid_points.append(points)
    
    # Create mesh grid and flatten
    meshgrid = jnp.meshgrid(*grid_points, indexing='ij')
    grid_samples = jnp.stack([g.flatten() for g in meshgrid], axis=1)
    
    # Take first n_samples points and add some randomness
    indices = random.choice(key, len(grid_samples), (n_samples,), replace=False)
    samples = grid_samples[indices]
    
    return samples


def generate_grid_samples(
    n_samples: int,
    bounds: List[Tuple[float, float]]
) -> jnp.ndarray:
    """Generate regular grid samples."""
    n_dims = len(bounds)
    samples_per_dim = int(jnp.ceil(n_samples ** (1.0 / n_dims)))
    
    # Create grid points
    grid_points = []
    for i in range(n_dims):
        low, high = bounds[i]
        points = jnp.linspace(low, high, samples_per_dim)
        grid_points.append(points)
    
    # Create mesh grid and flatten
    meshgrid = jnp.meshgrid(*grid_points, indexing='ij')
    samples = jnp.stack([g.flatten() for g in meshgrid], axis=1)
    
    # Take first n_samples points
    return samples[:n_samples]


def generate_multifidelity_data(
    high_fidelity_function: Callable,
    low_fidelity_function: Callable,
    n_high: int,
    n_low: int,
    bounds: List[Tuple[float, float]],
    random_seed: int = 42
) -> Dict[str, jnp.ndarray]:
    """Generate multi-fidelity training data.
    
    Args:
        high_fidelity_function: Expensive, accurate function
        low_fidelity_function: Cheap, approximate function
        n_high: Number of high-fidelity samples
        n_low: Number of low-fidelity samples
        bounds: Input bounds
        random_seed: Random seed
    
    Returns:
        Dictionary with high and low fidelity data
    """
    key = random.PRNGKey(random_seed)
    key_high, key_low = random.split(key)
    
    # Generate high-fidelity data
    X_high = generate_random_samples(key_high, n_high, bounds)
    y_high = jnp.array([high_fidelity_function(x) for x in X_high])
    
    # Generate low-fidelity data (including high-fidelity points)
    X_low_extra = generate_random_samples(key_low, n_low - n_high, bounds)
    X_low = jnp.concatenate([X_high, X_low_extra], axis=0)
    y_low = jnp.array([low_fidelity_function(x) for x in X_low])
    
    return {
        "X_high": X_high,
        "y_high": y_high,
        "X_low": X_low,
        "y_low": y_low,
        "n_high": n_high,
        "n_low": n_low
    }


def generate_time_series_data(
    function: Callable,
    n_timesteps: int,
    n_dims: int,
    bounds: List[Tuple[float, float]],
    correlation: float = 0.8,
    random_seed: int = 42
) -> Dict[str, jnp.ndarray]:
    """Generate time series data for dynamic optimization testing.
    
    Args:
        function: Function to evaluate (may depend on time)
        n_timesteps: Number of time steps
        n_dims: Dimensionality of input space
        bounds: Input bounds
        correlation: Temporal correlation coefficient
        random_seed: Random seed
    
    Returns:
        Dictionary with time series data
    """
    key = random.PRNGKey(random_seed)
    
    # Generate correlated time series of inputs
    X_series = []
    x_current = random.uniform(key, (n_dims,), 
                              minval=jnp.array([b[0] for b in bounds]),
                              maxval=jnp.array([b[1] for b in bounds]))
    
    for t in range(n_timesteps):
        key = random.split(key)[0]
        
        # Add correlated noise
        noise = random.normal(key, (n_dims,)) * 0.1
        x_next = correlation * x_current + (1 - correlation) * noise
        
        # Clip to bounds
        for i, (low, high) in enumerate(bounds):
            x_next = x_next.at[i].set(jnp.clip(x_next[i], low, high))
        
        X_series.append(x_next)
        x_current = x_next
    
    X = jnp.stack(X_series)
    
    # Evaluate function at each time step (function may depend on time)
    y = jnp.array([function(X[t], t) if function.__code__.co_argcount > 1 else function(X[t]) 
                   for t in range(n_timesteps)])
    
    return {
        "X": X,
        "y": y,
        "timesteps": jnp.arange(n_timesteps),
        "correlation": correlation
    }


def generate_constrained_data(
    function: Callable,
    constraints: List[Callable],
    n_samples: int,
    bounds: List[Tuple[float, float]],
    max_attempts: int = 10000,
    random_seed: int = 42
) -> Dict[str, jnp.ndarray]:
    """Generate data satisfying constraints.
    
    Args:
        function: Objective function
        constraints: List of constraint functions (should return <= 0 for feasible)
        n_samples: Number of feasible samples to generate
        bounds: Input bounds
        max_attempts: Maximum attempts to find feasible points
        random_seed: Random seed
    
    Returns:
        Dictionary with feasible data points
    """
    key = random.PRNGKey(random_seed)
    n_dims = len(bounds)
    
    feasible_X = []
    feasible_y = []
    attempts = 0
    
    while len(feasible_X) < n_samples and attempts < max_attempts:
        key = random.split(key)[0]
        
        # Generate candidate point
        x_candidate = random.uniform(
            key, (n_dims,),
            minval=jnp.array([b[0] for b in bounds]),
            maxval=jnp.array([b[1] for b in bounds])
        )
        
        # Check constraints
        feasible = True
        for constraint in constraints:
            if constraint(x_candidate) > 0:  # Constraint violated
                feasible = False
                break
        
        if feasible:
            feasible_X.append(x_candidate)
            feasible_y.append(function(x_candidate))
        
        attempts += 1
    
    if len(feasible_X) < n_samples:
        raise ValueError(f"Could only find {len(feasible_X)} feasible points out of {n_samples} requested")
    
    return {
        "X": jnp.stack(feasible_X),
        "y": jnp.array(feasible_y),
        "n_feasible": len(feasible_X),
        "n_attempts": attempts
    }


def load_benchmark_data(dataset_name: str) -> Dict[str, jnp.ndarray]:
    """Load benchmark datasets for testing.
    
    Args:
        dataset_name: Name of the dataset to load
    
    Returns:
        Dictionary with loaded data
    """
    if dataset_name == "hartmann_6d":
        return generate_hartmann_6d_data()
    elif dataset_name == "branin":
        return generate_branin_data()
    elif dataset_name == "goldstein_price":
        return generate_goldstein_price_data()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def generate_hartmann_6d_data(n_samples: int = 500) -> Dict[str, jnp.ndarray]:
    """Generate Hartmann 6D benchmark data."""
    def hartmann_6d(x):
        alpha = jnp.array([1.0, 1.2, 3.0, 3.2])
        A = jnp.array([
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14]
        ])
        P = jnp.array([
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381]
        ]) * 1e-4
        
        outer_sum = 0
        for i in range(4):
            inner_sum = jnp.sum(A[i] * (x - P[i])**2)
            outer_sum += alpha[i] * jnp.exp(-inner_sum)
        
        return -outer_sum
    
    bounds = [(0, 1)] * 6
    return generate_synthetic_data(hartmann_6d, n_samples, bounds)


def generate_branin_data(n_samples: int = 200) -> Dict[str, jnp.ndarray]:
    """Generate Branin benchmark data."""
    def branin(x):
        x1, x2 = x[0], x[1]
        a = 1
        b = 5.1 / (4 * jnp.pi**2)
        c = 5 / jnp.pi
        r = 6
        s = 10
        t = 1 / (8 * jnp.pi)
        
        return a * (x2 - b*x1**2 + c*x1 - r)**2 + s*(1-t)*jnp.cos(x1) + s
    
    bounds = [(-5, 10), (0, 15)]
    return generate_synthetic_data(branin, n_samples, bounds)


def generate_goldstein_price_data(n_samples: int = 200) -> Dict[str, jnp.ndarray]:
    """Generate Goldstein-Price benchmark data."""
    def goldstein_price(x):
        x1, x2 = x[0], x[1]
        
        term1 = 1 + (x1 + x2 + 1)**2 * (19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2)
        term2 = 30 + (2*x1 - 3*x2)**2 * (18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2)
        
        return term1 * term2
    
    bounds = [(-2, 2), (-2, 2)]
    return generate_synthetic_data(goldstein_price, n_samples, bounds)