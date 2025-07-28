"""Benchmark optimization functions for testing."""

from typing import Dict, List, Tuple, Callable
import jax.numpy as jnp
from jax import Array


class BenchmarkFunction:
    """Base class for benchmark optimization functions."""
    
    def __init__(self, name: str, bounds: List[Tuple[float, float]], 
                 global_optimum: Array, optimal_value: float):
        self.name = name
        self.bounds = bounds
        self.global_optimum = global_optimum
        self.optimal_value = optimal_value
        self.n_dims = len(bounds)
    
    def __call__(self, x: Array) -> float:
        """Evaluate function at point x."""
        raise NotImplementedError
    
    def is_in_bounds(self, x: Array) -> bool:
        """Check if point is within bounds."""
        for i, (low, high) in enumerate(self.bounds):
            if x[i] < low or x[i] > high:
                return False
        return True


class Rosenbrock(BenchmarkFunction):
    """Rosenbrock function - classic optimization benchmark."""
    
    def __init__(self, n_dims: int = 2):
        bounds = [(-5.0, 5.0)] * n_dims
        global_optimum = jnp.ones(n_dims)
        optimal_value = 0.0
        super().__init__("Rosenbrock", bounds, global_optimum, optimal_value)
    
    def __call__(self, x: Array) -> float:
        """Evaluate Rosenbrock function."""
        return jnp.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


class Rastrigin(BenchmarkFunction):
    """Rastrigin function - multimodal optimization benchmark."""
    
    def __init__(self, n_dims: int = 2):
        bounds = [(-5.12, 5.12)] * n_dims
        global_optimum = jnp.zeros(n_dims)
        optimal_value = 0.0
        super().__init__("Rastrigin", bounds, global_optimum, optimal_value)
    
    def __call__(self, x: Array) -> float:
        """Evaluate Rastrigin function."""
        n = len(x)
        return 10 * n + jnp.sum(x**2 - 10 * jnp.cos(2 * jnp.pi * x))


class Ackley(BenchmarkFunction):
    """Ackley function - multimodal optimization benchmark."""
    
    def __init__(self, n_dims: int = 2):
        bounds = [(-32.768, 32.768)] * n_dims
        global_optimum = jnp.zeros(n_dims)
        optimal_value = 0.0
        super().__init__("Ackley", bounds, global_optimum, optimal_value)
    
    def __call__(self, x: Array) -> float:
        """Evaluate Ackley function."""
        n = len(x)
        sum_sq = jnp.sum(x**2)
        sum_cos = jnp.sum(jnp.cos(2 * jnp.pi * x))
        return (-20 * jnp.exp(-0.2 * jnp.sqrt(sum_sq / n)) - 
                jnp.exp(sum_cos / n) + 20 + jnp.e)


class Sphere(BenchmarkFunction):
    """Sphere function - simple quadratic benchmark."""
    
    def __init__(self, n_dims: int = 2):
        bounds = [(-5.0, 5.0)] * n_dims
        global_optimum = jnp.zeros(n_dims)
        optimal_value = 0.0
        super().__init__("Sphere", bounds, global_optimum, optimal_value)
    
    def __call__(self, x: Array) -> float:
        """Evaluate Sphere function."""
        return jnp.sum(x**2)


class Griewank(BenchmarkFunction):
    """Griewank function - multimodal optimization benchmark."""
    
    def __init__(self, n_dims: int = 2):
        bounds = [(-600.0, 600.0)] * n_dims
        global_optimum = jnp.zeros(n_dims)
        optimal_value = 0.0
        super().__init__("Griewank", bounds, global_optimum, optimal_value)
    
    def __call__(self, x: Array) -> float:
        """Evaluate Griewank function."""
        sum_term = jnp.sum(x**2) / 4000
        prod_term = jnp.prod(jnp.cos(x / jnp.sqrt(jnp.arange(1, len(x) + 1))))
        return sum_term - prod_term + 1


class Levy(BenchmarkFunction):
    """Levy function - multimodal optimization benchmark."""
    
    def __init__(self, n_dims: int = 2):
        bounds = [(-10.0, 10.0)] * n_dims
        global_optimum = jnp.ones(n_dims)
        optimal_value = 0.0
        super().__init__("Levy", bounds, global_optimum, optimal_value)
    
    def __call__(self, x: Array) -> float:
        """Evaluate Levy function."""
        w = 1 + (x - 1) / 4
        
        term1 = jnp.sin(jnp.pi * w[0])**2
        
        term2 = jnp.sum((w[:-1] - 1)**2 * (1 + 10 * jnp.sin(jnp.pi * w[:-1] + 1)**2))
        
        term3 = (w[-1] - 1)**2 * (1 + jnp.sin(2 * jnp.pi * w[-1])**2)
        
        return term1 + term2 + term3


class Schwefel(BenchmarkFunction):
    """Schwefel function - multimodal optimization benchmark."""
    
    def __init__(self, n_dims: int = 2):
        bounds = [(-500.0, 500.0)] * n_dims
        global_optimum = jnp.full(n_dims, 420.9687)
        optimal_value = 0.0
        super().__init__("Schwefel", bounds, global_optimum, optimal_value)
    
    def __call__(self, x: Array) -> float:
        """Evaluate Schwefel function."""
        n = len(x)
        return 418.9829 * n - jnp.sum(x * jnp.sin(jnp.sqrt(jnp.abs(x))))


class Himmelblau(BenchmarkFunction):
    """Himmelblau function - multiple global optima."""
    
    def __init__(self):
        bounds = [(-5.0, 5.0), (-5.0, 5.0)]
        # Has 4 global optima - we'll use one of them
        global_optimum = jnp.array([3.0, 2.0])
        optimal_value = 0.0
        super().__init__("Himmelblau", bounds, global_optimum, optimal_value)
    
    def __call__(self, x: Array) -> float:
        """Evaluate Himmelblau function."""
        return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2


class Beale(BenchmarkFunction):
    """Beale function - 2D optimization benchmark."""
    
    def __init__(self):
        bounds = [(-4.5, 4.5), (-4.5, 4.5)]
        global_optimum = jnp.array([3.0, 0.5])
        optimal_value = 0.0
        super().__init__("Beale", bounds, global_optimum, optimal_value)
    
    def __call__(self, x: Array) -> float:
        """Evaluate Beale function."""
        term1 = (1.5 - x[0] + x[0] * x[1])**2
        term2 = (2.25 - x[0] + x[0] * x[1]**2)**2
        term3 = (2.625 - x[0] + x[0] * x[1]**3)**2
        return term1 + term2 + term3


class Matyas(BenchmarkFunction):
    """Matyas function - simple 2D benchmark."""
    
    def __init__(self):
        bounds = [(-10.0, 10.0), (-10.0, 10.0)]
        global_optimum = jnp.array([0.0, 0.0])
        optimal_value = 0.0
        super().__init__("Matyas", bounds, global_optimum, optimal_value)
    
    def __call__(self, x: Array) -> float:
        """Evaluate Matyas function."""
        return 0.26 * (x[0]**2 + x[1]**2) - 0.48 * x[0] * x[1]


class StyblinskiTang(BenchmarkFunction):
    """Styblinski-Tang function - multimodal benchmark."""
    
    def __init__(self, n_dims: int = 2):
        bounds = [(-5.0, 5.0)] * n_dims
        global_optimum = jnp.full(n_dims, -2.903534)
        optimal_value = -39.16617 * n_dims  # Approximate
        super().__init__("StyblinskiTang", bounds, global_optimum, optimal_value)
    
    def __call__(self, x: Array) -> float:
        """Evaluate Styblinski-Tang function."""
        return jnp.sum(x**4 - 16*x**2 + 5*x) / 2


# High-dimensional benchmark functions
class RotatedEllipsoid(BenchmarkFunction):
    """Rotated ellipsoid function - tests algorithm's ability to handle ill-conditioning."""
    
    def __init__(self, n_dims: int = 10):
        bounds = [(-5.0, 5.0)] * n_dims
        global_optimum = jnp.zeros(n_dims)
        optimal_value = 0.0
        super().__init__("RotatedEllipsoid", bounds, global_optimum, optimal_value)
    
    def __call__(self, x: Array) -> float:
        """Evaluate rotated ellipsoid function."""
        return jnp.sum(jnp.cumsum(x)**2)


class SumOfSquares(BenchmarkFunction):
    """Sum of squares function - separable quadratic."""
    
    def __init__(self, n_dims: int = 10):
        bounds = [(-10.0, 10.0)] * n_dims
        global_optimum = jnp.zeros(n_dims)
        optimal_value = 0.0
        super().__init__("SumOfSquares", bounds, global_optimum, optimal_value)
    
    def __call__(self, x: Array) -> float:
        """Evaluate sum of squares function."""
        return jnp.sum(jnp.arange(1, len(x) + 1) * x**2)


class Powell(BenchmarkFunction):
    """Powell function - non-separable benchmark."""
    
    def __init__(self, n_dims: int = 4):
        if n_dims % 4 != 0:
            raise ValueError("Powell function requires dimensionality divisible by 4")
        bounds = [(-4.0, 5.0)] * n_dims
        global_optimum = jnp.zeros(n_dims)
        optimal_value = 0.0
        super().__init__("Powell", bounds, global_optimum, optimal_value)
    
    def __call__(self, x: Array) -> float:
        """Evaluate Powell function."""
        n = len(x)
        result = 0.0
        
        for i in range(0, n, 4):
            if i + 3 < n:
                term1 = (x[i] + 10 * x[i+1])**2
                term2 = 5 * (x[i+2] - x[i+3])**2
                term3 = (x[i+1] - 2 * x[i+2])**4
                term4 = 10 * (x[i] - x[i+3])**4
                result += term1 + term2 + term3 + term4
        
        return result


# Constrained benchmark functions
class G1Constraint(BenchmarkFunction):
    """G1 constrained optimization problem."""
    
    def __init__(self):
        bounds = [(0, 1)] * 13
        # Approximate global optimum
        global_optimum = jnp.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        optimal_value = -15.0  # Approximate
        super().__init__("G1", bounds, global_optimum, optimal_value)
    
    def __call__(self, x: Array) -> float:
        """Evaluate G1 objective function."""
        return 5 * jnp.sum(x[:4]) - 5 * jnp.sum(x[:4]**2) - jnp.sum(x[4:])
    
    def constraints(self, x: Array) -> Array:
        """Evaluate G1 constraints (should be <= 0)."""
        g = jnp.zeros(9)
        
        g = g.at[0].set(2*x[0] + 2*x[1] + x[9] + x[10] - 10)
        g = g.at[1].set(2*x[0] + 2*x[2] + x[9] + x[11] - 10)
        g = g.at[2].set(2*x[1] + 2*x[2] + x[10] + x[11] - 10)
        g = g.at[3].set(-8*x[0] + x[9])
        g = g.at[4].set(-8*x[1] + x[10])
        g = g.at[5].set(-8*x[2] + x[11])
        g = g.at[6].set(-2*x[3] - x[4] + x[9])
        g = g.at[7].set(-2*x[5] - x[6] + x[10])
        g = g.at[8].set(-2*x[7] - x[8] + x[11])
        
        return g


# Factory function to create benchmark functions
def create_benchmark_functions() -> Dict[str, BenchmarkFunction]:
    """Create dictionary of all benchmark functions."""
    return {
        "rosenbrock_2d": Rosenbrock(2),
        "rosenbrock_5d": Rosenbrock(5),
        "rosenbrock_10d": Rosenbrock(10),
        "rastrigin_2d": Rastrigin(2),
        "rastrigin_5d": Rastrigin(5),
        "rastrigin_10d": Rastrigin(10),
        "ackley_2d": Ackley(2),
        "ackley_5d": Ackley(5),
        "ackley_10d": Ackley(10),
        "sphere_2d": Sphere(2),
        "sphere_5d": Sphere(5),
        "sphere_10d": Sphere(10),
        "griewank_2d": Griewank(2),
        "griewank_10d": Griewank(10),
        "levy_2d": Levy(2),
        "levy_5d": Levy(5),
        "schwefel_2d": Schwefel(2),
        "schwefel_5d": Schwefel(5),
        "himmelblau": Himmelblau(),
        "beale": Beale(),
        "matyas": Matyas(),
        "styblinski_tang_2d": StyblinskiTang(2),
        "styblinski_tang_5d": StyblinskiTang(5),
        "rotated_ellipsoid_10d": RotatedEllipsoid(10),
        "sum_of_squares_10d": SumOfSquares(10),
        "powell_4d": Powell(4),
        "powell_8d": Powell(8),
        "g1_constraint": G1Constraint()
    }


# Convenience functions
benchmark_functions = create_benchmark_functions()


def get_benchmark_function(name: str) -> BenchmarkFunction:
    """Get benchmark function by name."""
    if name not in benchmark_functions:
        raise ValueError(f"Unknown benchmark function: {name}. "
                        f"Available: {list(benchmark_functions.keys())}")
    return benchmark_functions[name]


def get_2d_functions() -> Dict[str, BenchmarkFunction]:
    """Get all 2D benchmark functions for visualization."""
    return {name: func for name, func in benchmark_functions.items() 
            if func.n_dims == 2}


def get_multimodal_functions() -> Dict[str, BenchmarkFunction]:
    """Get multimodal benchmark functions."""
    multimodal_names = [
        "rastrigin_2d", "rastrigin_5d", "rastrigin_10d",
        "ackley_2d", "ackley_5d", "ackley_10d",
        "griewank_2d", "griewank_10d",
        "levy_2d", "levy_5d",
        "schwefel_2d", "schwefel_5d",
        "himmelblau", "styblinski_tang_2d", "styblinski_tang_5d"
    ]
    return {name: benchmark_functions[name] for name in multimodal_names}


def get_high_dim_functions() -> Dict[str, BenchmarkFunction]:
    """Get high-dimensional benchmark functions."""
    return {name: func for name, func in benchmark_functions.items() 
            if func.n_dims >= 5}


def evaluate_on_grid(func: BenchmarkFunction, resolution: int = 50) -> Tuple[Array, Array, Array]:
    """Evaluate 2D function on a grid for visualization.
    
    Args:
        func: 2D benchmark function
        resolution: Grid resolution
    
    Returns:
        X, Y, Z arrays for contour plotting
    """
    if func.n_dims != 2:
        raise ValueError("Grid evaluation only supported for 2D functions")
    
    x_bounds = func.bounds[0]
    y_bounds = func.bounds[1]
    
    x = jnp.linspace(x_bounds[0], x_bounds[1], resolution)
    y = jnp.linspace(y_bounds[0], y_bounds[1], resolution)
    X, Y = jnp.meshgrid(x, y)
    
    Z = jnp.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            point = jnp.array([X[i, j], Y[i, j]])
            Z = Z.at[i, j].set(func(point))
    
    return X, Y, Z