"""Meta-Optimization Framework for optimization algorithm selection and tuning.

This module implements meta-learning techniques to automatically select
the best optimization algorithms and hyperparameters for different problem types.
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
from jax import Array
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score


@dataclass
class ProblemCharacteristics:
    """Characteristics that describe an optimization problem."""
    dimension: int
    sample_size: int
    noise_level: float
    multimodality: float  # Measure of how many local optima
    smoothness: float    # Measure of function smoothness
    condition_number: float  # Numerical conditioning
    symmetry: float      # Degree of symmetry
    separability: float  # Degree of variable separability
    problem_type: str    # Category like "continuous", "discrete", etc.


@dataclass
class AlgorithmPerformance:
    """Performance metrics for an algorithm on a specific problem."""
    algorithm_name: str
    problem_characteristics: ProblemCharacteristics
    final_objective_value: float
    convergence_iterations: int
    computation_time: float
    success: bool
    robustness_score: float
    memory_usage: float


@dataclass
class MetaLearningData:
    """Data point for meta-learning algorithm selection."""
    features: Array  # Problem characteristics as feature vector
    algorithm_id: int  # Algorithm identifier
    performance_score: float  # Normalized performance metric
    confidence: float  # Confidence in this measurement


class Algorithm(ABC):
    """Abstract base class for optimization algorithms."""

    @abstractmethod
    def optimize(
        self,
        objective_function: Callable[[Array], float],
        initial_point: Array,
        bounds: Optional[List[Tuple[float, float]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Perform optimization and return results."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Get algorithm name."""
        pass

    @property
    @abstractmethod
    def hyperparameters(self) -> Dict[str, Any]:
        """Get current hyperparameters."""
        pass

    @abstractmethod
    def set_hyperparameters(self, params: Dict[str, Any]) -> None:
        """Set algorithm hyperparameters."""
        pass


class GradientDescentAlgorithm(Algorithm):
    """Gradient descent optimization algorithm."""

    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations

    def optimize(
        self,
        objective_function: Callable[[Array], float],
        initial_point: Array,
        bounds: Optional[List[Tuple[float, float]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Perform gradient descent optimization."""
        grad_fn = jax.grad(objective_function)

        x = initial_point
        trajectory = [x]

        for i in range(self.max_iterations):
            try:
                gradient = grad_fn(x)
                x = x - self.learning_rate * gradient

                # Apply bounds if provided
                if bounds:
                    x = jnp.clip(x,
                               jnp.array([b[0] for b in bounds]),
                               jnp.array([b[1] for b in bounds]))

                trajectory.append(x)

                # Check convergence
                if jnp.linalg.norm(gradient) < 1e-6:
                    break

            except Exception as e:
                return {
                    "success": False,
                    "x": x,
                    "fun": float("inf"),
                    "nit": i,
                    "message": str(e)
                }

        final_value = objective_function(x)

        return {
            "success": True,
            "x": x,
            "fun": final_value,
            "nit": i + 1,
            "trajectory": trajectory,
            "message": "Optimization completed"
        }

    @property
    def name(self) -> str:
        return "gradient_descent"

    @property
    def hyperparameters(self) -> Dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "max_iterations": self.max_iterations
        }

    def set_hyperparameters(self, params: Dict[str, Any]) -> None:
        if "learning_rate" in params:
            self.learning_rate = params["learning_rate"]
        if "max_iterations" in params:
            self.max_iterations = params["max_iterations"]


class RandomSearchAlgorithm(Algorithm):
    """Random search optimization algorithm."""

    def __init__(self, max_evaluations: int = 1000, seed: int = 42):
        self.max_evaluations = max_evaluations
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def optimize(
        self,
        objective_function: Callable[[Array], float],
        initial_point: Array,
        bounds: Optional[List[Tuple[float, float]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Perform random search optimization."""
        best_x = initial_point
        best_value = objective_function(initial_point)

        if bounds is None:
            # Default bounds around initial point
            bounds = [(-10, 10)] * len(initial_point)

        for i in range(self.max_evaluations):
            # Generate random point
            x = jnp.array([
                self.rng.uniform(low, high)
                for low, high in bounds
            ])

            try:
                value = objective_function(x)

                if value < best_value:
                    best_x = x
                    best_value = value

            except Exception:
                continue

        return {
            "success": True,
            "x": best_x,
            "fun": best_value,
            "nit": self.max_evaluations,
            "message": "Random search completed"
        }

    @property
    def name(self) -> str:
        return "random_search"

    @property
    def hyperparameters(self) -> Dict[str, Any]:
        return {
            "max_evaluations": self.max_evaluations,
            "seed": self.seed
        }

    def set_hyperparameters(self, params: Dict[str, Any]) -> None:
        if "max_evaluations" in params:
            self.max_evaluations = params["max_evaluations"]
        if "seed" in params:
            self.seed = params["seed"]
            self.rng = np.random.RandomState(self.seed)


class ProblemAnalyzer:
    """Analyzes optimization problems to extract characteristics."""

    def __init__(self, sample_size: int = 100):
        """Initialize problem analyzer.
        
        Args:
            sample_size: Number of samples to use for analysis
        """
        self.sample_size = sample_size

    def analyze_problem(
        self,
        objective_function: Callable[[Array], float],
        dimension: int,
        bounds: Optional[List[Tuple[float, float]]] = None
    ) -> ProblemCharacteristics:
        """Analyze an optimization problem to extract characteristics.
        
        Args:
            objective_function: The objective function to analyze
            dimension: Problem dimension
            bounds: Variable bounds
            
        Returns:
            Problem characteristics
        """
        if bounds is None:
            bounds = [(-5, 5)] * dimension

        # Generate sample points
        samples = self._generate_sample_points(dimension, bounds, self.sample_size)

        # Evaluate function at sample points
        try:
            values = jnp.array([objective_function(x) for x in samples])
        except Exception:
            # Fallback for problematic functions
            values = jnp.full(len(samples), float("inf"))

        # Extract characteristics
        characteristics = ProblemCharacteristics(
            dimension=dimension,
            sample_size=len(samples),
            noise_level=self._estimate_noise_level(samples, values),
            multimodality=self._estimate_multimodality(values),
            smoothness=self._estimate_smoothness(samples, values),
            condition_number=self._estimate_condition_number(samples, values),
            symmetry=self._estimate_symmetry(samples, values),
            separability=self._estimate_separability(samples, values, dimension),
            problem_type="continuous"
        )

        return characteristics

    def _generate_sample_points(
        self,
        dimension: int,
        bounds: List[Tuple[float, float]],
        n_samples: int
    ) -> List[Array]:
        """Generate sample points for analysis."""
        points = []

        for _ in range(n_samples):
            point = jnp.array([
                np.random.uniform(low, high)
                for low, high in bounds
            ])
            points.append(point)

        return points

    def _estimate_noise_level(self, samples: List[Array], values: Array) -> float:
        """Estimate noise level in the function."""
        if len(values) < 10:
            return 0.1  # Default assumption

        # Use coefficient of variation as noise estimate
        mean_val = jnp.mean(values[jnp.isfinite(values)])
        std_val = jnp.std(values[jnp.isfinite(values)])

        if abs(mean_val) > 1e-8:
            noise = float(std_val / abs(mean_val))
        else:
            noise = float(std_val)

        return min(1.0, max(0.0, noise))

    def _estimate_multimodality(self, values: Array) -> float:
        """Estimate multimodality (number of local optima)."""
        if len(values) < 5:
            return 0.5  # Default assumption

        # Count local minima (simplified approach)
        finite_values = values[jnp.isfinite(values)]

        if len(finite_values) < 3:
            return 0.1

        # Use value distribution to estimate multimodality
        sorted_values = jnp.sort(finite_values)
        value_range = sorted_values[-1] - sorted_values[0]

        if value_range < 1e-8:
            return 0.1  # Essentially flat function

        # Look for multiple clusters in the values
        # Simple heuristic based on value distribution
        percentiles = jnp.percentile(sorted_values, [25, 50, 75])
        spread = (percentiles[2] - percentiles[0]) / value_range

        return min(1.0, spread)

    def _estimate_smoothness(self, samples: List[Array], values: Array) -> float:
        """Estimate function smoothness."""
        if len(samples) < 3:
            return 0.5

        finite_indices = jnp.isfinite(values)
        if jnp.sum(finite_indices) < 3:
            return 0.1  # Likely discontinuous

        finite_values = values[finite_indices]

        # Estimate smoothness based on value variation
        value_std = jnp.std(finite_values)
        value_mean = jnp.abs(jnp.mean(finite_values))

        if value_mean > 1e-8:
            smoothness = 1.0 / (1.0 + value_std / value_mean)
        else:
            smoothness = 1.0 / (1.0 + value_std)

        return float(jnp.clip(smoothness, 0.0, 1.0))

    def _estimate_condition_number(self, samples: List[Array], values: Array) -> float:
        """Estimate numerical condition number."""
        # Simplified approach using value distribution
        finite_values = values[jnp.isfinite(values)]

        if len(finite_values) < 2:
            return 1.0  # Default

        value_range = jnp.max(finite_values) - jnp.min(finite_values)
        value_std = jnp.std(finite_values)

        if value_std > 1e-8:
            condition = float(value_range / value_std)
        else:
            condition = 1.0

        # Normalize to [0, 1] scale
        return float(jnp.clip(condition / 100.0, 0.0, 1.0))

    def _estimate_symmetry(self, samples: List[Array], values: Array) -> float:
        """Estimate degree of symmetry."""
        # Simplified symmetry estimation
        finite_indices = jnp.isfinite(values)

        if jnp.sum(finite_indices) < 10:
            return 0.5  # Default assumption

        finite_values = values[finite_indices]

        # Check if value distribution is symmetric around median
        median_val = jnp.median(finite_values)

        # Measure symmetry of distribution
        lower_tail = finite_values[finite_values <= median_val]
        upper_tail = finite_values[finite_values >= median_val]

        if len(lower_tail) > 0 and len(upper_tail) > 0:
            lower_spread = jnp.std(lower_tail)
            upper_spread = jnp.std(upper_tail)

            if lower_spread + upper_spread > 1e-8:
                symmetry = 1.0 - abs(lower_spread - upper_spread) / (lower_spread + upper_spread)
            else:
                symmetry = 1.0
        else:
            symmetry = 0.5

        return float(jnp.clip(symmetry, 0.0, 1.0))

    def _estimate_separability(
        self,
        samples: List[Array],
        values: Array,
        dimension: int
    ) -> float:
        """Estimate degree of variable separability."""
        if dimension == 1:
            return 1.0  # Trivially separable

        if len(samples) < 20:
            return 0.5  # Default assumption

        # Simplified separability test
        # Check if function can be approximated as sum of univariate functions
        try:
            # Create design matrix for separability test
            X = jnp.array([jnp.array(x) for x in samples])
            finite_mask = jnp.isfinite(values)

            if jnp.sum(finite_mask) < 10:
                return 0.5

            X_finite = X[finite_mask]
            y_finite = values[finite_mask]

            # Fit additive model (sum of univariate functions)
            # This is a very simplified test
            separability_score = 0.0

            for dim in range(dimension):
                # Correlation between this dimension and function values
                dim_values = X_finite[:, dim]
                correlation = abs(jnp.corrcoef(dim_values, y_finite)[0, 1])

                if jnp.isfinite(correlation):
                    separability_score += correlation

            separability_score /= dimension

            return float(jnp.clip(separability_score, 0.0, 1.0))

        except Exception:
            return 0.5


class MetaOptimizationFramework:
    """Framework for meta-optimization and algorithm selection.
    
    This framework learns from optimization experiences to automatically
    select the best algorithms and hyperparameters for new problems.
    """

    def __init__(self):
        """Initialize meta-optimization framework."""
        # Algorithm registry
        self.algorithms: Dict[str, Algorithm] = {
            "gradient_descent": GradientDescentAlgorithm(),
            "random_search": RandomSearchAlgorithm(),
        }

        # Meta-learning data
        self.meta_data: List[MetaLearningData] = []
        self.performance_history: List[AlgorithmPerformance] = []

        # Problem analyzer
        self.problem_analyzer = ProblemAnalyzer()

        # Meta-models for algorithm selection and hyperparameter tuning
        self.algorithm_selector: Optional[RandomForestClassifier] = None
        self.performance_predictor: Optional[RandomForestRegressor] = None

        # Performance tracking
        self.selection_accuracy_history: List[float] = []
        self.prediction_error_history: List[float] = []

        # Logger
        self.logger = logging.getLogger(__name__)

    def register_algorithm(self, algorithm: Algorithm) -> None:
        """Register a new optimization algorithm.
        
        Args:
            algorithm: Algorithm instance to register
        """
        self.algorithms[algorithm.name] = algorithm
        self.logger.info(f"Registered algorithm: {algorithm.name}")

    def optimize_with_meta_learning(
        self,
        objective_function: Callable[[Array], float],
        initial_point: Array,
        bounds: Optional[List[Tuple[float, float]]] = None,
        max_algorithm_trials: int = 3,
        **kwargs
    ) -> Dict[str, Any]:
        """Optimize using meta-learning for algorithm selection.
        
        Args:
            objective_function: Function to optimize
            initial_point: Starting point
            bounds: Variable bounds
            max_algorithm_trials: Maximum algorithms to try
            **kwargs: Additional optimization arguments
            
        Returns:
            Optimization results with meta-learning information
        """
        start_time = time.time()

        # Analyze problem characteristics
        problem_chars = self.problem_analyzer.analyze_problem(
            objective_function, len(initial_point), bounds
        )

        self.logger.info(f"Problem analysis: dim={problem_chars.dimension}, "
                        f"multimodality={problem_chars.multimodality:.2f}, "
                        f"smoothness={problem_chars.smoothness:.2f}")

        # Select algorithms to try
        selected_algorithms = self._select_algorithms(
            problem_chars, max_algorithm_trials
        )

        # Try selected algorithms
        results = []
        best_result = None
        best_performance = float("inf")

        for algorithm_name in selected_algorithms:
            algorithm = self.algorithms[algorithm_name]

            # Optimize hyperparameters for this algorithm and problem
            optimized_params = self._optimize_hyperparameters(
                algorithm, objective_function, initial_point, bounds, problem_chars
            )

            # Set optimized hyperparameters
            if optimized_params:
                algorithm.set_hyperparameters(optimized_params)

            # Run optimization
            try:
                result = algorithm.optimize(
                    objective_function, initial_point, bounds, **kwargs
                )

                result["algorithm"] = algorithm_name
                result["hyperparameters"] = algorithm.hyperparameters
                result["problem_characteristics"] = problem_chars

                results.append(result)

                # Track best result
                if result.get("success", False) and result.get("fun", float("inf")) < best_performance:
                    best_result = result
                    best_performance = result["fun"]

                self.logger.info(f"{algorithm_name}: success={result.get('success')}, "
                               f"value={result.get('fun', 'N/A'):.6f}")

            except Exception as e:
                self.logger.warning(f"Algorithm {algorithm_name} failed: {e}")
                continue

        # Record performance for meta-learning
        self._record_performance(results, problem_chars)

        # Update meta-models if we have enough data
        if len(self.meta_data) >= 50:  # Minimum data for training
            self._update_meta_models()

        # Prepare final result
        if best_result is None:
            best_result = {
                "success": False,
                "message": "All algorithms failed",
                "x": initial_point,
                "fun": float("inf")
            }

        best_result["meta_optimization"] = {
            "algorithms_tried": selected_algorithms,
            "total_time": time.time() - start_time,
            "problem_characteristics": problem_chars,
            "all_results": results
        }

        return best_result

    def _select_algorithms(
        self,
        problem_chars: ProblemCharacteristics,
        max_trials: int
    ) -> List[str]:
        """Select algorithms to try based on problem characteristics.
        
        Args:
            problem_chars: Characteristics of the problem
            max_trials: Maximum number of algorithms to select
            
        Returns:
            List of algorithm names to try
        """
        if self.algorithm_selector is None or len(self.meta_data) < 20:
            # Use heuristic selection when not enough meta-data
            return self._heuristic_algorithm_selection(problem_chars, max_trials)

        # Use trained meta-model for selection
        features = self._problem_to_features(problem_chars).reshape(1, -1)

        try:
            # Get probabilities for each algorithm
            algorithm_names = list(self.algorithms.keys())
            probabilities = self.algorithm_selector.predict_proba(features)[0]

            # Sort algorithms by probability
            algorithm_probs = list(zip(algorithm_names, probabilities))
            algorithm_probs.sort(key=lambda x: x[1], reverse=True)

            # Select top algorithms
            selected = [name for name, _ in algorithm_probs[:max_trials]]

            self.logger.info(f"Meta-model selected algorithms: {selected}")

            return selected

        except Exception as e:
            self.logger.warning(f"Meta-model selection failed: {e}, using heuristic")
            return self._heuristic_algorithm_selection(problem_chars, max_trials)

    def _heuristic_algorithm_selection(
        self,
        problem_chars: ProblemCharacteristics,
        max_trials: int
    ) -> List[str]:
        """Heuristic algorithm selection based on problem characteristics.
        
        Args:
            problem_chars: Problem characteristics
            max_trials: Maximum algorithms to select
            
        Returns:
            List of algorithm names
        """
        # Simple heuristics for algorithm selection
        candidates = []

        # Gradient descent for smooth problems
        if problem_chars.smoothness > 0.6:
            candidates.append("gradient_descent")

        # Random search for multimodal problems
        if problem_chars.multimodality > 0.5:
            candidates.append("random_search")

        # Ensure we have at least some algorithms
        if not candidates:
            candidates = list(self.algorithms.keys())

        # Return up to max_trials algorithms
        return candidates[:max_trials]

    def _optimize_hyperparameters(
        self,
        algorithm: Algorithm,
        objective_function: Callable[[Array], float],
        initial_point: Array,
        bounds: Optional[List[Tuple[float, float]]],
        problem_chars: ProblemCharacteristics
    ) -> Optional[Dict[str, Any]]:
        """Optimize hyperparameters for an algorithm on a specific problem.
        
        Args:
            algorithm: Algorithm to optimize hyperparameters for
            objective_function: Objective function
            initial_point: Initial point for optimization
            bounds: Variable bounds
            problem_chars: Problem characteristics
            
        Returns:
            Optimized hyperparameters or None if optimization fails
        """
        if self.performance_predictor is None or len(self.performance_history) < 20:
            # Not enough data for hyperparameter optimization
            return None

        try:
            # Define hyperparameter search space for this algorithm
            param_bounds = self._get_hyperparameter_bounds(algorithm)

            if not param_bounds:
                return None

            # Objective function for hyperparameter optimization
            def hyperparam_objective(params_array: np.ndarray) -> float:
                # Convert array to parameter dictionary
                params_dict = self._array_to_hyperparams(params_array, algorithm, param_bounds)

                # Predict performance using meta-model
                features = self._create_meta_features(problem_chars, algorithm.name, params_dict)

                try:
                    predicted_performance = self.performance_predictor.predict([features])[0]
                    return predicted_performance  # Lower is better
                except Exception:
                    return float("inf")

            # Optimize hyperparameters
            bounds_array = [(bounds[0], bounds[1]) for _, bounds in param_bounds.items()]

            result = minimize(
                hyperparam_objective,
                x0=[(b[0] + b[1]) / 2 for b in bounds_array],  # Start at midpoint
                bounds=bounds_array,
                method="L-BFGS-B"
            )

            if result.success:
                optimized_params = self._array_to_hyperparams(
                    result.x, algorithm, param_bounds
                )

                self.logger.info(f"Optimized hyperparameters for {algorithm.name}: {optimized_params}")

                return optimized_params
            return None

        except Exception as e:
            self.logger.warning(f"Hyperparameter optimization failed for {algorithm.name}: {e}")
            return None

    def _get_hyperparameter_bounds(self, algorithm: Algorithm) -> Dict[str, Tuple[float, float]]:
        """Get hyperparameter bounds for an algorithm.
        
        Args:
            algorithm: Algorithm to get bounds for
            
        Returns:
            Dictionary of parameter bounds
        """
        if algorithm.name == "gradient_descent":
            return {
                "learning_rate": (1e-5, 1e-1),
                "max_iterations": (50, 2000)
            }
        if algorithm.name == "random_search":
            return {
                "max_evaluations": (100, 5000)
            }
        return {}

    def _array_to_hyperparams(
        self,
        params_array: np.ndarray,
        algorithm: Algorithm,
        param_bounds: Dict[str, Tuple[float, float]]
    ) -> Dict[str, Any]:
        """Convert parameter array to hyperparameter dictionary.
        
        Args:
            params_array: Parameter values as array
            algorithm: Algorithm instance
            param_bounds: Parameter bounds
            
        Returns:
            Hyperparameter dictionary
        """
        params_dict = {}

        for i, (param_name, (low, high)) in enumerate(param_bounds.items()):
            if param_name in ["max_iterations", "max_evaluations"]:
                # Integer parameters
                params_dict[param_name] = int(np.clip(params_array[i], low, high))
            else:
                # Float parameters
                params_dict[param_name] = np.clip(params_array[i], low, high)

        return params_dict

    def _problem_to_features(self, problem_chars: ProblemCharacteristics) -> Array:
        """Convert problem characteristics to feature vector.
        
        Args:
            problem_chars: Problem characteristics
            
        Returns:
            Feature vector
        """
        features = jnp.array([
            problem_chars.dimension,
            problem_chars.noise_level,
            problem_chars.multimodality,
            problem_chars.smoothness,
            problem_chars.condition_number,
            problem_chars.symmetry,
            problem_chars.separability,
        ])

        return features

    def _create_meta_features(
        self,
        problem_chars: ProblemCharacteristics,
        algorithm_name: str,
        hyperparams: Dict[str, Any]
    ) -> Array:
        """Create meta-features for performance prediction.
        
        Args:
            problem_chars: Problem characteristics
            algorithm_name: Name of algorithm
            hyperparams: Algorithm hyperparameters
            
        Returns:
            Meta-feature vector
        """
        problem_features = self._problem_to_features(problem_chars)

        # Algorithm encoding (simple one-hot)
        algorithm_features = jnp.zeros(len(self.algorithms))
        algorithm_names = list(self.algorithms.keys())
        if algorithm_name in algorithm_names:
            algorithm_features = algorithm_features.at[algorithm_names.index(algorithm_name)].set(1.0)

        # Hyperparameter features (normalized)
        hyperparam_features = []

        if algorithm_name == "gradient_descent":
            hyperparam_features.extend([
                np.log10(hyperparams.get("learning_rate", 0.01)) / 5.0,  # Normalize log scale
                hyperparams.get("max_iterations", 1000) / 2000.0
            ])
        elif algorithm_name == "random_search":
            hyperparam_features.extend([
                hyperparams.get("max_evaluations", 1000) / 5000.0,
                0.0  # Padding to keep consistent feature size
            ])
        else:
            hyperparam_features.extend([0.0, 0.0])  # Default padding

        hyperparam_features = jnp.array(hyperparam_features)

        # Combine all features
        meta_features = jnp.concatenate([problem_features, algorithm_features, hyperparam_features])

        return meta_features

    def _record_performance(
        self,
        results: List[Dict[str, Any]],
        problem_chars: ProblemCharacteristics
    ) -> None:
        """Record performance data for meta-learning.
        
        Args:
            results: Optimization results
            problem_chars: Problem characteristics
        """
        for result in results:
            if "algorithm" not in result:
                continue

            # Create performance record
            performance = AlgorithmPerformance(
                algorithm_name=result["algorithm"],
                problem_characteristics=problem_chars,
                final_objective_value=result.get("fun", float("inf")),
                convergence_iterations=result.get("nit", 0),
                computation_time=result.get("computation_time", 0.0),
                success=result.get("success", False),
                robustness_score=1.0 if result.get("success", False) else 0.0,
                memory_usage=0.0  # Could be measured in practice
            )

            self.performance_history.append(performance)

            # Create meta-learning data point
            features = self._create_meta_features(
                problem_chars,
                result["algorithm"],
                result.get("hyperparameters", {})
            )

            # Performance score (lower is better, so we negate for learning)
            if result.get("success", False):
                performance_score = -result.get("fun", 0.0)
            else:
                performance_score = -1000.0  # Penalty for failure

            # Algorithm ID for classification
            algorithm_names = list(self.algorithms.keys())
            algorithm_id = algorithm_names.index(result["algorithm"]) if result["algorithm"] in algorithm_names else 0

            meta_data_point = MetaLearningData(
                features=features,
                algorithm_id=algorithm_id,
                performance_score=performance_score,
                confidence=1.0 if result.get("success", False) else 0.5
            )

            self.meta_data.append(meta_data_point)

    def _update_meta_models(self) -> None:
        """Update meta-learning models with accumulated data."""
        if len(self.meta_data) < 20:
            return

        try:
            # Prepare training data
            X = jnp.stack([d.features for d in self.meta_data])
            y_algorithm = jnp.array([d.algorithm_id for d in self.meta_data])
            y_performance = jnp.array([d.performance_score for d in self.meta_data])

            # Train algorithm selector
            self.algorithm_selector = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10
            )

            self.algorithm_selector.fit(X, y_algorithm)

            # Evaluate selector accuracy
            cv_scores = cross_val_score(self.algorithm_selector, X, y_algorithm, cv=3)
            selector_accuracy = np.mean(cv_scores)
            self.selection_accuracy_history.append(selector_accuracy)

            # Train performance predictor
            self.performance_predictor = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                max_depth=10
            )

            self.performance_predictor.fit(X, y_performance)

            # Evaluate predictor accuracy
            cv_scores = cross_val_score(
                self.performance_predictor, X, y_performance,
                cv=3, scoring="neg_mean_squared_error"
            )
            predictor_mse = -np.mean(cv_scores)
            self.prediction_error_history.append(predictor_mse)

            self.logger.info(f"Updated meta-models: selector accuracy={selector_accuracy:.3f}, "
                           f"predictor MSE={predictor_mse:.3f}")

        except Exception as e:
            self.logger.error(f"Failed to update meta-models: {e}")

    def get_meta_learning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive meta-learning statistics.
        
        Returns:
            Dictionary containing meta-learning metrics
        """
        return {
            "total_optimization_experiences": len(self.performance_history),
            "meta_data_points": len(self.meta_data),
            "algorithms_registered": len(self.algorithms),
            "selector_trained": self.algorithm_selector is not None,
            "predictor_trained": self.performance_predictor is not None,
            "average_selector_accuracy": np.mean(self.selection_accuracy_history)
                                        if self.selection_accuracy_history else 0.0,
            "average_prediction_error": np.mean(self.prediction_error_history)
                                      if self.prediction_error_history else float("inf"),
            "recent_selector_accuracy": self.selection_accuracy_history[-1]
                                      if self.selection_accuracy_history else 0.0,
            "recent_prediction_error": self.prediction_error_history[-1]
                                     if self.prediction_error_history else float("inf"),
            "algorithm_success_rates": self._calculate_algorithm_success_rates(),
        }

    def _calculate_algorithm_success_rates(self) -> Dict[str, float]:
        """Calculate success rates for each algorithm.
        
        Returns:
            Dictionary of algorithm success rates
        """
        algorithm_stats = defaultdict(lambda: {"successes": 0, "total": 0})

        for performance in self.performance_history:
            algorithm_stats[performance.algorithm_name]["total"] += 1
            if performance.success:
                algorithm_stats[performance.algorithm_name]["successes"] += 1

        success_rates = {}
        for algorithm, stats in algorithm_stats.items():
            if stats["total"] > 0:
                success_rates[algorithm] = stats["successes"] / stats["total"]
            else:
                success_rates[algorithm] = 0.0

        return success_rates
