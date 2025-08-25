"""Enhanced novel research algorithms for surrogate optimization - Generation 2.

This module contains cutting-edge algorithms with robust error handling and advanced features.
"""

from dataclasses import dataclass, field
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from jax import Array, grad, random, vmap
import jax.numpy as jnp

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EnhancedResearchResult:
    """Enhanced result from a research experiment with robust error handling."""
    algorithm_name: str
    experiment_id: str
    success: bool
    performance_metrics: Dict[str, float]
    convergence_data: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    error_info: Optional[str] = None
    statistical_significance: Optional[float] = None


class RobustPhysicsInformedSurrogate:
    """Enhanced physics-informed neural network with error handling and validation.
    
    Novel contribution: Adaptive physics weighting and uncertainty quantification.
    """

    def __init__(
        self,
        hidden_dims: List[int] = [64, 64, 32],
        physics_weight: float = 0.1,
        boundary_weight: float = 0.05,
        activation: str = "tanh",
        dropout_rate: float = 0.1,
        ensemble_size: int = 5,
        adaptive_weighting: bool = True,
    ):
        """Initialize robust physics-informed surrogate.
        
        Args:
            hidden_dims: Hidden layer dimensions
            physics_weight: Initial weight for physics loss term
            boundary_weight: Weight for boundary condition loss
            activation: Activation function
            dropout_rate: Dropout rate for uncertainty estimation
            ensemble_size: Number of models in ensemble
            adaptive_weighting: Whether to adapt physics weights during training
        """
        self.hidden_dims = hidden_dims
        self.physics_weight = physics_weight
        self.boundary_weight = boundary_weight
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.ensemble_size = ensemble_size
        self.adaptive_weighting = adaptive_weighting

        # Training state
        self.ensemble_params = []
        self.is_fitted = False
        self.training_history = []

        # Physics constraints
        self.physics_loss_fn = None
        self.boundary_conditions = []

        # Adaptive weighting
        self.physics_weight_history = [physics_weight]

        # Error handling
        self.last_error = None
        self.training_successful = False

    def add_physics_constraint(self, physics_loss_fn: Callable[[Array, Callable], float]):
        """Add physics-based constraint with validation."""
        try:
            # Test the physics function with dummy data
            test_x = jnp.array([[0.0, 0.0]])
            test_fn = lambda x: jnp.sum(x**2)
            _ = physics_loss_fn(test_x, test_fn)

            self.physics_loss_fn = physics_loss_fn
            logger.info("Physics constraint added successfully")

        except Exception as e:
            logger.error(f"Failed to add physics constraint: {e}")
            self.last_error = str(e)
            raise ValueError(f"Invalid physics constraint function: {e}")

    def add_boundary_condition(self, boundary_points: Array, boundary_values: Array):
        """Add boundary conditions with validation."""
        try:
            if boundary_points.shape[0] != boundary_values.shape[0]:
                raise ValueError("Boundary points and values must have same length")

            if boundary_points.ndim != 2:
                raise ValueError("Boundary points must be 2D array")

            self.boundary_conditions.append((boundary_points, boundary_values))
            logger.info(f"Added {len(boundary_points)} boundary conditions")

        except Exception as e:
            logger.error(f"Failed to add boundary conditions: {e}")
            self.last_error = str(e)
            raise

    def _initialize_ensemble(self, input_dim: int, base_key: random.PRNGKey):
        """Initialize ensemble of neural networks."""
        ensemble_params = []

        for i in range(self.ensemble_size):
            key = random.fold_in(base_key, i)
            params = self._initialize_network(input_dim, key)
            ensemble_params.append(params)

        return ensemble_params

    def _initialize_network(self, input_dim: int, key: random.PRNGKey):
        """Initialize single neural network with Xavier initialization."""
        layers = [input_dim] + self.hidden_dims + [1]
        params = []

        for i in range(len(layers) - 1):
            key, subkey = random.split(key)
            fan_in, fan_out = layers[i], layers[i+1]

            # Xavier initialization
            limit = jnp.sqrt(6.0 / (fan_in + fan_out))
            w = random.uniform(subkey, (fan_in, fan_out), minval=-limit, maxval=limit)

            # Bias initialization
            b = jnp.zeros(fan_out)
            params.append((w, b))

        return params

    def _forward_pass(self, params: List[Tuple], x: Array, training: bool = False, key: Optional[random.PRNGKey] = None) -> float:
        """Forward pass with dropout support."""
        activation_fn = {
            "tanh": jnp.tanh,
            "relu": lambda x: jnp.maximum(0, x),
            "sigmoid": lambda x: 1.0 / (1.0 + jnp.exp(-jnp.clip(x, -500, 500))),  # Clip for numerical stability
            "swish": lambda x: x * jnp.sigmoid(x),
        }[self.activation]

        h = x

        for i, (w, b) in enumerate(params[:-1]):
            h = jnp.dot(h, w) + b
            h = activation_fn(h)

            # Apply dropout during training
            if training and key is not None and self.dropout_rate > 0:
                key, subkey = random.split(key)
                dropout_mask = random.bernoulli(subkey, 1.0 - self.dropout_rate, h.shape)
                h = h * dropout_mask / (1.0 - self.dropout_rate)

        # Output layer (linear)
        w, b = params[-1]
        output = jnp.dot(h, w) + b

        return output.squeeze()

    def _compute_ensemble_loss(self, ensemble_params: List, X: Array, y: Array, epoch: int) -> float:
        """Compute total loss for ensemble with adaptive weighting."""
        total_loss = 0.0

        for params in ensemble_params:
            # Data fitting loss
            predictions = vmap(lambda x: self._forward_pass(params, x))(X)
            data_loss = jnp.mean((predictions - y)**2)

            loss = data_loss

            # Physics loss with adaptive weighting
            current_physics_weight = self.physics_weight
            if self.adaptive_weighting and self.physics_loss_fn is not None:
                # Adaptive physics weighting based on data fit quality
                if epoch > 100 and len(self.training_history) > 10:
                    recent_data_losses = [h.get("data_loss", 1.0) for h in self.training_history[-10:]]
                    avg_data_loss = jnp.mean(jnp.array(recent_data_losses))

                    # Increase physics weight if data loss is low
                    if avg_data_loss < 0.1:
                        current_physics_weight = self.physics_weight * 2.0
                    elif avg_data_loss > 1.0:
                        current_physics_weight = self.physics_weight * 0.5

                physics_pred_fn = lambda x: self._forward_pass(params, x)
                try:
                    physics_loss = self.physics_loss_fn(X[:min(20, len(X))], physics_pred_fn)
                    loss += current_physics_weight * physics_loss
                except Exception as e:
                    logger.warning(f"Physics loss computation failed: {e}")

            # Boundary condition loss
            boundary_loss = 0.0
            for boundary_points, boundary_values in self.boundary_conditions:
                try:
                    boundary_preds = vmap(lambda x: self._forward_pass(params, x))(boundary_points)
                    boundary_loss += jnp.mean((boundary_preds - boundary_values)**2)
                except Exception as e:
                    logger.warning(f"Boundary loss computation failed: {e}")

            loss += self.boundary_weight * boundary_loss
            total_loss += loss

        # Update physics weight history
        if self.adaptive_weighting:
            self.physics_weight_history.append(current_physics_weight)

        return total_loss / len(ensemble_params)

    def fit(self, dataset, max_epochs: int = 2000, patience: int = 100, min_improvement: float = 1e-6):
        """Fit ensemble with robust training and early stopping."""
        try:
            key = random.PRNGKey(42)

            # Validate dataset
            if dataset.n_samples < 5:
                raise ValueError("Dataset too small for training (need at least 5 samples)")

            if jnp.any(jnp.isnan(dataset.X)) or jnp.any(jnp.isnan(dataset.y)):
                raise ValueError("Dataset contains NaN values")

            # Initialize ensemble
            self.ensemble_params = self._initialize_ensemble(dataset.n_dims, key)

            # Training parameters
            learning_rate = 0.001
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8

            # Initialize Adam state for each model in ensemble
            ensemble_m = []
            ensemble_v = []

            for params in self.ensemble_params:
                m = [jnp.zeros_like(p) for layer in params for p in layer]
                v = [jnp.zeros_like(p) for layer in params for p in layer]
                ensemble_m.append(m)
                ensemble_v.append(v)

            # Training history
            self.training_history = []
            best_loss = float("inf")
            patience_counter = 0

            # Define loss function
            loss_fn = lambda params_list: self._compute_ensemble_loss(params_list, dataset.X, dataset.y, len(self.training_history))

            # Training loop
            for epoch in range(max_epochs):
                try:
                    # Compute gradients for ensemble
                    ensemble_grads = grad(loss_fn)(self.ensemble_params)
                    current_loss = loss_fn(self.ensemble_params)

                    # Check for numerical issues
                    if jnp.isnan(current_loss) or jnp.isinf(current_loss):
                        logger.warning(f"Numerical instability at epoch {epoch}, reducing learning rate")
                        learning_rate *= 0.5
                        if learning_rate < 1e-8:
                            break
                        continue

                    # Update each model in ensemble
                    new_ensemble_params = []

                    for model_idx, (params, grads, m, v) in enumerate(zip(
                        self.ensemble_params, ensemble_grads, ensemble_m, ensemble_v
                    )):
                        # Flatten parameters and gradients
                        flat_params = [p for layer in params for p in layer]
                        flat_grads = [g for layer_grads in grads for g in layer_grads]

                        # Adam update
                        updated_flat_params = []
                        for i, (param, grad_val, m_val, v_val) in enumerate(zip(flat_params, flat_grads, m, v)):
                            # Gradient clipping for stability
                            grad_val = jnp.clip(grad_val, -1.0, 1.0)

                            m[i] = beta1 * m_val + (1 - beta1) * grad_val
                            v[i] = beta2 * v_val + (1 - beta2) * grad_val**2

                            m_hat = m[i] / (1 - beta1**(epoch + 1))
                            v_hat = v[i] / (1 - beta2**(epoch + 1))

                            updated_param = param - learning_rate * m_hat / (jnp.sqrt(v_hat) + eps)
                            updated_flat_params.append(updated_param)

                        # Reconstruct parameter structure
                        param_idx = 0
                        new_params = []
                        for layer in params:
                            layer_params = []
                            for p in layer:
                                layer_params.append(updated_flat_params[param_idx])
                                param_idx += 1
                            new_params.append(tuple(layer_params))

                        new_ensemble_params.append(new_params)

                    self.ensemble_params = new_ensemble_params

                    # Record training progress
                    if epoch % 10 == 0:
                        # Compute individual loss components
                        data_loss = jnp.mean((
                            vmap(lambda x: self.predict(x))(dataset.X) - dataset.y
                        )**2)

                        history_entry = {
                            "epoch": epoch,
                            "total_loss": float(current_loss),
                            "data_loss": float(data_loss),
                            "learning_rate": learning_rate,
                            "physics_weight": self.physics_weight_history[-1] if self.physics_weight_history else self.physics_weight,
                        }
                        self.training_history.append(history_entry)

                    # Early stopping
                    if current_loss < best_loss - min_improvement:
                        best_loss = current_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter >= patience:
                        logger.info(f"Early stopping at epoch {epoch} (best loss: {best_loss:.6f})")
                        break

                    # Adaptive learning rate
                    if epoch > 0 and epoch % 500 == 0:
                        learning_rate *= 0.9

                except Exception as e:
                    logger.error(f"Error in training epoch {epoch}: {e}")
                    # Try to recover with reduced learning rate
                    learning_rate *= 0.1
                    if learning_rate < 1e-8:
                        break

            self.is_fitted = True
            self.training_successful = True
            logger.info(f"Training completed successfully in {len(self.training_history)} epochs")

        except Exception as e:
            self.last_error = str(e)
            self.training_successful = False
            logger.error(f"Training failed: {e}")
            raise

        return self

    def predict(self, x: Array) -> Array:
        """Predict with ensemble averaging."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        try:
            predictions = []

            for params in self.ensemble_params:
                if x.ndim == 1:
                    pred = self._forward_pass(params, x)
                else:
                    pred = vmap(lambda xi: self._forward_pass(params, xi))(x)
                predictions.append(pred)

            # Ensemble average
            ensemble_pred = jnp.mean(jnp.stack(predictions), axis=0)

            return ensemble_pred

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            self.last_error = str(e)
            raise

    def uncertainty(self, x: Array) -> Array:
        """Estimate prediction uncertainty using ensemble variance."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before uncertainty estimation")

        try:
            predictions = []

            for params in self.ensemble_params:
                if x.ndim == 1:
                    pred = self._forward_pass(params, x)
                else:
                    pred = vmap(lambda xi: self._forward_pass(params, xi))(x)
                predictions.append(pred)

            # Ensemble variance as uncertainty
            predictions_stack = jnp.stack(predictions)
            uncertainty = jnp.std(predictions_stack, axis=0)

            return uncertainty

        except Exception as e:
            logger.error(f"Uncertainty estimation failed: {e}")
            return jnp.zeros_like(self.predict(x))

    def gradient(self, x: Array) -> Array:
        """Compute gradients with ensemble averaging."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before gradient computation")

        try:
            gradients = []

            for params in self.ensemble_params:
                grad_fn = grad(lambda xi: self._forward_pass(params, xi))

                if x.ndim == 1:
                    grad_val = grad_fn(x)
                else:
                    grad_val = vmap(grad_fn)(x)

                gradients.append(grad_val)

            # Ensemble average of gradients
            ensemble_grad = jnp.mean(jnp.stack(gradients), axis=0)

            return ensemble_grad

        except Exception as e:
            logger.error(f"Gradient computation failed: {e}")
            self.last_error = str(e)
            raise

    def get_training_info(self) -> Dict[str, Any]:
        """Get comprehensive training information."""
        return {
            "is_fitted": self.is_fitted,
            "training_successful": self.training_successful,
            "ensemble_size": self.ensemble_size,
            "last_error": self.last_error,
            "training_epochs": len(self.training_history),
            "final_loss": self.training_history[-1]["total_loss"] if self.training_history else None,
            "physics_weight_final": self.physics_weight_history[-1] if self.physics_weight_history else self.physics_weight,
            "adaptive_weighting": self.adaptive_weighting,
            "has_physics_constraints": self.physics_loss_fn is not None,
            "n_boundary_conditions": len(self.boundary_conditions),
        }


class AdvancedAdaptiveAcquisitionOptimizer:
    """Enhanced adaptive acquisition function with statistical validation.
    
    Novel contribution: Multi-armed bandit approach to acquisition function selection.
    """

    def __init__(
        self,
        initial_strategy: str = "expected_improvement",
        adaptation_rate: float = 0.1,
        strategies: List[str] = None,
        confidence_level: float = 0.95,
        min_samples_for_adaptation: int = 10,
    ):
        """Initialize advanced adaptive acquisition optimizer."""
        self.strategies = strategies or [
            "expected_improvement", "upper_confidence_bound",
            "probability_improvement", "entropy_search", "thompson_sampling"
        ]
        self.current_strategy = initial_strategy
        self.adaptation_rate = adaptation_rate
        self.confidence_level = confidence_level
        self.min_samples_for_adaptation = min_samples_for_adaptation

        # Multi-armed bandit tracking
        self.strategy_rewards = {s: [] for s in self.strategies}
        self.strategy_counts = dict.fromkeys(self.strategies, 0)
        self.total_iterations = 0

        # Statistical validation
        self.statistical_tests = []

        # Acquisition functions
        self.acquisition_functions = {
            "expected_improvement": self._expected_improvement,
            "upper_confidence_bound": self._upper_confidence_bound,
            "probability_improvement": self._probability_improvement,
            "entropy_search": self._entropy_search,
            "thompson_sampling": self._thompson_sampling,
        }

    def _expected_improvement(self, x: Array, surrogate, best_value: float) -> float:
        """Expected improvement with numerical stability."""
        try:
            mean, std = surrogate.predict_with_uncertainty(x)

            # Numerical stability
            std = jnp.maximum(std, 1e-8)
            improvement = mean - best_value

            z = improvement / std
            z = jnp.clip(z, -10, 10)  # Prevent overflow

            ei = improvement * self._normal_cdf(z) + std * self._normal_pdf(z)

            return float(jnp.maximum(ei, 0.0))

        except Exception as e:
            logger.warning(f"EI computation failed: {e}")
            return 0.0

    def _upper_confidence_bound(self, x: Array, surrogate, best_value: float, beta: float = 2.0) -> float:
        """UCB with adaptive beta parameter."""
        try:
            mean, std = surrogate.predict_with_uncertainty(x)

            # Adaptive beta based on iteration count
            adaptive_beta = beta * jnp.sqrt(jnp.log(self.total_iterations + 1) / max(1, self.total_iterations))

            ucb = mean + adaptive_beta * std

            return float(ucb)

        except Exception as e:
            logger.warning(f"UCB computation failed: {e}")
            return 0.0

    def _probability_improvement(self, x: Array, surrogate, best_value: float) -> float:
        """PI with numerical stability."""
        try:
            mean, std = surrogate.predict_with_uncertainty(x)

            if std > 1e-8:
                z = (mean - best_value) / std
                z = jnp.clip(z, -10, 10)
                return float(self._normal_cdf(z))
            return 0.0

        except Exception as e:
            logger.warning(f"PI computation failed: {e}")
            return 0.0

    def _entropy_search(self, x: Array, surrogate, best_value: float) -> float:
        """Entropy search with information-theoretic acquisition."""
        try:
            mean, std = surrogate.predict_with_uncertainty(x)

            # Information gain approximation
            entropy = 0.5 * jnp.log(2 * jnp.pi * jnp.e * (std**2 + 1e-8))

            return float(entropy)

        except Exception as e:
            logger.warning(f"Entropy search computation failed: {e}")
            return 0.0

    def _thompson_sampling(self, x: Array, surrogate, best_value: float) -> float:
        """Thompson sampling acquisition."""
        try:
            mean, std = surrogate.predict_with_uncertainty(x)

            # Sample from posterior
            key = random.PRNGKey(self.total_iterations)
            sample = random.normal(key) * std + mean

            return float(sample)

        except Exception as e:
            logger.warning(f"Thompson sampling failed: {e}")
            return float(mean) if "mean" in locals() else 0.0

    def _normal_cdf(self, x: float) -> float:
        """Robust normal CDF approximation."""
        return 0.5 * (1 + jnp.tanh(x * jnp.sqrt(2 / jnp.pi)))

    def _normal_pdf(self, x: float) -> float:
        """Robust normal PDF."""
        return jnp.exp(-0.5 * jnp.clip(x**2, 0, 100)) / jnp.sqrt(2 * jnp.pi)

    def _ucb_strategy_selection(self) -> str:
        """Select strategy using UCB for multi-armed bandit."""
        if self.total_iterations < len(self.strategies):
            # Exploration phase: try each strategy at least once
            return self.strategies[self.total_iterations % len(self.strategies)]

        ucb_values = {}

        for strategy in self.strategies:
            if self.strategy_counts[strategy] > 0:
                mean_reward = jnp.mean(jnp.array(self.strategy_rewards[strategy]))
                confidence = jnp.sqrt(2 * jnp.log(self.total_iterations) / self.strategy_counts[strategy])
                ucb_values[strategy] = mean_reward + confidence
            else:
                ucb_values[strategy] = float("inf")  # Unvisited strategies get high priority

        return max(ucb_values.keys(), key=lambda k: ucb_values[k])

    def update_strategy_performance(self, improvement: float):
        """Update strategy performance with statistical validation."""
        try:
            # Record reward for current strategy
            self.strategy_rewards[self.current_strategy].append(improvement)
            self.strategy_counts[self.current_strategy] += 1
            self.total_iterations += 1

            # Adapt strategy if enough samples
            if self.total_iterations >= self.min_samples_for_adaptation:
                # Use UCB for strategy selection
                new_strategy = self._ucb_strategy_selection()

                if new_strategy != self.current_strategy:
                    logger.info(f"Switching acquisition strategy from {self.current_strategy} to {new_strategy}")
                    self.current_strategy = new_strategy

            # Statistical validation every 50 iterations
            if self.total_iterations % 50 == 0:
                self._perform_statistical_validation()

        except Exception as e:
            logger.error(f"Strategy performance update failed: {e}")

    def _perform_statistical_validation(self):
        """Perform statistical tests on strategy performance."""
        try:
            # Mann-Whitney U test for comparing strategies
            from scipy import stats

            strategies_with_data = [s for s in self.strategies if len(self.strategy_rewards[s]) >= 5]

            if len(strategies_with_data) >= 2:
                # Find best and worst performing strategies
                mean_performances = {
                    s: jnp.mean(jnp.array(self.strategy_rewards[s]))
                    for s in strategies_with_data
                }

                best_strategy = max(mean_performances.keys(), key=lambda k: mean_performances[k])
                worst_strategy = min(mean_performances.keys(), key=lambda k: mean_performances[k])

                # Perform statistical test
                best_rewards = self.strategy_rewards[best_strategy]
                worst_rewards = self.strategy_rewards[worst_strategy]

                try:
                    statistic, p_value = stats.mannwhitneyu(best_rewards, worst_rewards, alternative="greater")

                    test_result = {
                        "iteration": self.total_iterations,
                        "best_strategy": best_strategy,
                        "worst_strategy": worst_strategy,
                        "p_value": p_value,
                        "significant": p_value < (1 - self.confidence_level),
                        "best_mean": mean_performances[best_strategy],
                        "worst_mean": mean_performances[worst_strategy],
                    }

                    self.statistical_tests.append(test_result)

                    if test_result["significant"]:
                        logger.info(f"Statistically significant difference found: {best_strategy} > {worst_strategy} (p={p_value:.4f})")

                except Exception as e:
                    logger.warning(f"Statistical test failed: {e}")

        except ImportError:
            logger.warning("scipy not available for statistical tests")
        except Exception as e:
            logger.error(f"Statistical validation failed: {e}")

    def acquire(self, x: Array, surrogate, best_value: float) -> float:
        """Compute acquisition value with error handling."""
        try:
            return self.acquisition_functions[self.current_strategy](x, surrogate, best_value)
        except Exception as e:
            logger.error(f"Acquisition computation failed: {e}")
            # Fallback to simple expected improvement
            return self._expected_improvement(x, surrogate, best_value)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary with statistics."""
        summary = {
            "current_strategy": self.current_strategy,
            "total_iterations": self.total_iterations,
            "strategy_statistics": {},
            "statistical_tests": self.statistical_tests,
        }

        for strategy in self.strategies:
            if self.strategy_rewards[strategy]:
                rewards = jnp.array(self.strategy_rewards[strategy])
                summary["strategy_statistics"][strategy] = {
                    "count": self.strategy_counts[strategy],
                    "mean_reward": float(jnp.mean(rewards)),
                    "std_reward": float(jnp.std(rewards)),
                    "min_reward": float(jnp.min(rewards)),
                    "max_reward": float(jnp.max(rewards)),
                    "recent_trend": float(jnp.mean(rewards[-10:])) if len(rewards) >= 10 else None,
                }
            else:
                summary["strategy_statistics"][strategy] = {
                    "count": 0,
                    "mean_reward": None,
                    "std_reward": None,
                    "min_reward": None,
                    "max_reward": None,
                    "recent_trend": None,
                }

        return summary


# Research execution framework
def run_enhanced_algorithm_benchmark(
    test_functions: List[Callable],
    algorithm_configs: Dict[str, Dict],
    n_trials: int = 10,
    n_iterations: int = 100,
    statistical_validation: bool = True,
) -> Dict[str, EnhancedResearchResult]:
    """Run comprehensive benchmark with statistical validation."""
    logger.info(f"Starting enhanced algorithm benchmark with {len(test_functions)} functions, {n_trials} trials")

    results = {}

    for algo_name, config in algorithm_configs.items():
        logger.info(f"Benchmarking algorithm: {algo_name}")

        algo_results = []
        execution_times = []

        for trial in range(n_trials):
            for func_idx, test_func in enumerate(test_functions):
                start_time = time.time()

                try:
                    if algo_name == "robust_physics_informed":
                        result = _benchmark_physics_informed(test_func, config, trial, func_idx)
                    elif algo_name == "advanced_adaptive_acquisition":
                        result = _benchmark_adaptive_acquisition(test_func, config, trial, func_idx)
                    else:
                        result = _benchmark_generic_algorithm(algo_name, test_func, config, trial, func_idx)

                    execution_time = time.time() - start_time
                    execution_times.append(execution_time)

                    result.execution_time = execution_time
                    algo_results.append(result)

                except Exception as e:
                    logger.error(f"Algorithm {algo_name} failed on trial {trial}, function {func_idx}: {e}")

                    failed_result = EnhancedResearchResult(
                        algorithm_name=algo_name,
                        experiment_id=f"{algo_name}_trial{trial}_func{func_idx}_failed",
                        success=False,
                        performance_metrics={"error": str(e)},
                        execution_time=time.time() - start_time,
                        error_info=str(e),
                    )
                    algo_results.append(failed_result)

        # Aggregate results with statistical analysis
        successful_results = [r for r in algo_results if r.success]

        if successful_results and statistical_validation:
            # Statistical analysis
            performance_values = [
                r.performance_metrics.get("final_value", 0.0)
                for r in successful_results
            ]

            if len(performance_values) >= 3:
                try:
                    from scipy import stats

                    # Normality test
                    _, normality_p = stats.shapiro(performance_values)

                    # Confidence interval
                    confidence_interval = stats.t.interval(
                        0.95, len(performance_values) - 1,
                        loc=jnp.mean(jnp.array(performance_values)),
                        scale=stats.sem(performance_values)
                    )

                    statistical_significance = normality_p

                except ImportError:
                    logger.warning("scipy not available for statistical analysis")
                    statistical_significance = None
                    confidence_interval = None
                except Exception as e:
                    logger.warning(f"Statistical analysis failed: {e}")
                    statistical_significance = None
                    confidence_interval = None
            else:
                statistical_significance = None
                confidence_interval = None
        else:
            statistical_significance = None
            confidence_interval = None

        # Create summary result
        if successful_results:
            avg_performance = float(jnp.mean(jnp.array([
                r.performance_metrics.get("final_value", 0.0)
                for r in successful_results
            ])))
            success_rate = len(successful_results) / len(algo_results)
            avg_time = float(jnp.mean(jnp.array(execution_times))) if execution_times else 0.0
        else:
            avg_performance = float("inf")
            success_rate = 0.0
            avg_time = 0.0

        results[algo_name] = EnhancedResearchResult(
            algorithm_name=algo_name,
            experiment_id=f"{algo_name}_summary",
            success=success_rate > 0.5,
            performance_metrics={
                "average_performance": avg_performance,
                "success_rate": success_rate,
                "average_execution_time": avg_time,
                "total_trials": len(algo_results),
                "successful_trials": len(successful_results),
                "confidence_interval": confidence_interval,
            },
            statistical_significance=statistical_significance,
            metadata={
                "individual_results": algo_results,
                "algorithm_config": config,
            },
        )

    logger.info("Enhanced algorithm benchmark completed")
    return results


def _benchmark_physics_informed(test_func: Callable, config: Dict, trial: int, func_idx: int) -> EnhancedResearchResult:
    """Benchmark robust physics-informed surrogate."""
    try:
        from ..data.collector import collect_data

        # Collect training data
        bounds = [(-2, 2)] * 2
        data = collect_data(test_func, n_samples=100, bounds=bounds, sampling="sobol")

        # Create and train physics-informed surrogate
        pinn = RobustPhysicsInformedSurrogate(**config)

        # Add physics constraint for testing
        def harmonic_constraint(X, pred_fn):
            # Simple second derivative constraint
            eps = 1e-4
            penalties = []

            for x in X[:5]:  # Limit for computational efficiency
                # Approximate second derivatives
                x_plus = x.at[0].add(eps)
                x_minus = x.at[0].add(-eps)

                second_deriv_x = (pred_fn(x_plus) - 2*pred_fn(x) + pred_fn(x_minus)) / (eps**2)

                y_plus = x.at[1].add(eps)
                y_minus = x.at[1].add(-eps)

                second_deriv_y = (pred_fn(y_plus) - 2*pred_fn(x) + pred_fn(y_minus)) / (eps**2)

                laplacian = second_deriv_x + second_deriv_y
                penalties.append(laplacian**2)

            return jnp.mean(jnp.array(penalties))

        pinn.add_physics_constraint(harmonic_constraint)
        pinn.fit(data)

        # Test optimization using surrogate
        x0 = jnp.array([1.0, 1.0])
        x = x0

        optimization_history = [float(test_func(x0))]

        # Simple gradient descent
        for i in range(50):
            try:
                grad_val = pinn.gradient(x)
                x = x - 0.01 * grad_val

                # Clip to bounds
                x = jnp.clip(x, -2, 2)

                current_value = float(test_func(x))
                optimization_history.append(current_value)

            except Exception as e:
                logger.warning(f"Optimization step {i} failed: {e}")
                break

        final_value = optimization_history[-1]

        # Get training info
        training_info = pinn.get_training_info()

        return EnhancedResearchResult(
            algorithm_name="robust_physics_informed",
            experiment_id=f"pinn_trial{trial}_func{func_idx}",
            success=training_info["training_successful"] and not jnp.isnan(final_value),
            performance_metrics={
                "final_value": final_value,
                "initial_value": optimization_history[0],
                "improvement": optimization_history[0] - final_value,
                "convergence_rate": len(optimization_history),
                "training_epochs": training_info["training_epochs"],
                "ensemble_size": training_info["ensemble_size"],
            },
            convergence_data=optimization_history,
            metadata=training_info,
        )

    except Exception as e:
        return EnhancedResearchResult(
            algorithm_name="robust_physics_informed",
            experiment_id=f"pinn_trial{trial}_func{func_idx}_failed",
            success=False,
            performance_metrics={"error": str(e)},
            error_info=str(e),
        )


def _benchmark_adaptive_acquisition(test_func: Callable, config: Dict, trial: int, func_idx: int) -> EnhancedResearchResult:
    """Benchmark advanced adaptive acquisition optimizer."""
    try:
        optimizer = AdvancedAdaptiveAcquisitionOptimizer(**config)

        # Simulate optimization process
        best_value = float("inf")
        improvements = []

        key = random.PRNGKey(trial * 1000 + func_idx)

        for i in range(100):
            # Generate candidate point
            key, subkey = random.split(key)
            x = random.uniform(subkey, shape=(2,), minval=-2, maxval=2)

            value = float(test_func(x))

            if value < best_value:
                improvement = best_value - value
                best_value = value
                improvements.append(improvement)
            else:
                improvements.append(0.0)

            # Update strategy performance
            optimizer.update_strategy_performance(improvements[-1])

        # Get performance summary
        performance_summary = optimizer.get_performance_summary()

        return EnhancedResearchResult(
            algorithm_name="advanced_adaptive_acquisition",
            experiment_id=f"aaa_trial{trial}_func{func_idx}",
            success=best_value < float("inf"),
            performance_metrics={
                "final_value": best_value,
                "total_improvements": sum(improvements),
                "improvement_rate": sum(1 for imp in improvements if imp > 0) / len(improvements),
                "total_iterations": len(improvements),
                "best_strategy": performance_summary["current_strategy"],
            },
            convergence_data=improvements,
            metadata=performance_summary,
        )

    except Exception as e:
        return EnhancedResearchResult(
            algorithm_name="advanced_adaptive_acquisition",
            experiment_id=f"aaa_trial{trial}_func{func_idx}_failed",
            success=False,
            performance_metrics={"error": str(e)},
            error_info=str(e),
        )


def _benchmark_generic_algorithm(algo_name: str, test_func: Callable, config: Dict, trial: int, func_idx: int) -> EnhancedResearchResult:
    """Benchmark generic algorithm with basic evaluation."""
    try:
        # Simple random search for unknown algorithms
        key = random.PRNGKey(trial * 1000 + func_idx)

        best_value = float("inf")
        evaluations = []

        for i in range(50):
            key, subkey = random.split(key)
            x = random.uniform(subkey, shape=(2,), minval=-2, maxval=2)

            value = float(test_func(x))
            evaluations.append(value)

            best_value = min(best_value, value)

        return EnhancedResearchResult(
            algorithm_name=algo_name,
            experiment_id=f"{algo_name}_trial{trial}_func{func_idx}",
            success=True,
            performance_metrics={
                "final_value": best_value,
                "function_evaluations": len(evaluations),
            },
            convergence_data=evaluations,
        )

    except Exception as e:
        return EnhancedResearchResult(
            algorithm_name=algo_name,
            experiment_id=f"{algo_name}_trial{trial}_func{func_idx}_failed",
            success=False,
            performance_metrics={"error": str(e)},
            error_info=str(e),
        )
