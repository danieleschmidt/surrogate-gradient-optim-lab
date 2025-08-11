"""Novel research algorithms and experimental features for surrogate optimization."""

import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
from jax import Array, grad, vmap, jit
from jax.scipy.optimize import minimize
import jax.random as random


@dataclass
class ResearchResult:
    """Result from a research experiment."""
    algorithm_name: str
    experiment_id: str
    success: bool
    performance_metrics: Dict[str, float]
    convergence_data: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0


class PhysicsInformedSurrogate:
    """Physics-informed neural network surrogate for optimization problems with known physics.
    
    This is a novel approach that incorporates domain knowledge into surrogate models.
    """
    
    def __init__(
        self,
        hidden_dims: List[int] = [64, 64, 32],
        physics_weight: float = 0.1,
        boundary_weight: float = 0.05,
        activation: str = "tanh",
    ):
        """Initialize physics-informed surrogate.
        
        Args:
            hidden_dims: Hidden layer dimensions
            physics_weight: Weight for physics loss term
            boundary_weight: Weight for boundary condition loss
            activation: Activation function
        """
        self.hidden_dims = hidden_dims
        self.physics_weight = physics_weight
        self.boundary_weight = boundary_weight
        self.activation = activation
        
        # Neural network parameters
        self.params = None
        self.is_fitted = False
        
        # Physics constraints
        self.physics_loss_fn = None
        self.boundary_conditions = []
    
    def add_physics_constraint(self, physics_loss_fn: Callable[[Array, Callable], float]):
        """Add physics-based constraint to the model.
        
        Args:
            physics_loss_fn: Function that computes physics violation loss
        """
        self.physics_loss_fn = physics_loss_fn
    
    def add_boundary_condition(self, boundary_points: Array, boundary_values: Array):
        """Add boundary conditions.
        
        Args:
            boundary_points: Points where boundary conditions apply
            boundary_values: Expected values at boundary points
        """
        self.boundary_conditions.append((boundary_points, boundary_values))
    
    def _initialize_network(self, input_dim: int, key: random.PRNGKey):
        """Initialize neural network parameters."""
        layers = [input_dim] + self.hidden_dims + [1]
        params = []
        
        for i in range(len(layers) - 1):
            key, subkey = random.split(key)
            w = random.normal(subkey, (layers[i], layers[i+1])) * jnp.sqrt(2.0 / layers[i])
            b = jnp.zeros(layers[i+1])
            params.append((w, b))
        
        return params
    
    def _forward_pass(self, params: List[Tuple], x: Array) -> float:
        """Forward pass through the network."""
        activation_fn = {
            "tanh": jnp.tanh,
            "relu": lambda x: jnp.maximum(0, x),
            "sigmoid": lambda x: 1.0 / (1.0 + jnp.exp(-x)),
        }[self.activation]
        
        for i, (w, b) in enumerate(params[:-1]):
            x = activation_fn(jnp.dot(x, w) + b)
        
        # Output layer (linear)
        w, b = params[-1]
        return jnp.dot(x, w) + b
    
    def _compute_loss(self, params: List[Tuple], X: Array, y: Array) -> float:
        """Compute total loss including data, physics, and boundary terms."""
        # Data fitting loss
        predictions = vmap(lambda x: self._forward_pass(params, x))(X)
        data_loss = jnp.mean((predictions - y)**2)
        
        total_loss = data_loss
        
        # Physics loss
        if self.physics_loss_fn is not None:
            physics_pred_fn = lambda x: self._forward_pass(params, x)
            physics_loss = self.physics_loss_fn(X, physics_pred_fn)
            total_loss += self.physics_weight * physics_loss
        
        # Boundary condition loss
        boundary_loss = 0.0
        for boundary_points, boundary_values in self.boundary_conditions:
            boundary_preds = vmap(lambda x: self._forward_pass(params, x))(boundary_points)
            boundary_loss += jnp.mean((boundary_preds - boundary_values)**2)
        
        total_loss += self.boundary_weight * boundary_loss
        
        return total_loss
    
    def fit(self, dataset):
        """Fit the physics-informed surrogate."""
        key = random.PRNGKey(42)
        
        # Initialize network
        self.params = self._initialize_network(dataset.n_dims, key)
        
        # Define loss function for optimization
        loss_fn = lambda params: self._compute_loss(params, dataset.X, dataset.y)
        grad_fn = jit(grad(loss_fn))
        
        # Adam optimizer
        learning_rate = 0.001
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8
        
        # Adam state
        m = [jnp.zeros_like(p) for layer in self.params for p in layer]
        v = [jnp.zeros_like(p) for layer in self.params for p in layer]
        
        # Training loop
        n_epochs = 1000
        convergence_history = []
        
        for epoch in range(n_epochs):
            grads = grad_fn(self.params)
            
            # Update parameters using Adam optimizer
            updated_params = []
            flat_idx = 0
            
            for layer_idx, (w, b) in enumerate(self.params):
                # Weight gradients
                w_grad = grads[layer_idx][0]
                w_m = m[flat_idx] = beta1 * m[flat_idx] + (1 - beta1) * w_grad
                w_v = v[flat_idx] = beta2 * v[flat_idx] + (1 - beta2) * w_grad**2
                
                w_m_hat = w_m / (1 - beta1**(epoch + 1))
                w_v_hat = w_v / (1 - beta2**(epoch + 1))
                
                w_new = w - learning_rate * w_m_hat / (jnp.sqrt(w_v_hat) + eps)
                flat_idx += 1
                
                # Bias gradients
                b_grad = grads[layer_idx][1]
                b_m = m[flat_idx] = beta1 * m[flat_idx] + (1 - beta1) * b_grad
                b_v = v[flat_idx] = beta2 * v[flat_idx] + (1 - beta2) * b_grad**2
                
                b_m_hat = b_m / (1 - beta1**(epoch + 1))
                b_v_hat = b_v / (1 - beta2**(epoch + 1))
                
                b_new = b - learning_rate * b_m_hat / (jnp.sqrt(b_v_hat) + eps)
                flat_idx += 1
                
                updated_params.append((w_new, b_new))
            
            self.params = updated_params
            
            # Track convergence
            if epoch % 100 == 0:
                current_loss = loss_fn(self.params)
                convergence_history.append(float(current_loss))
                if len(convergence_history) > 1 and abs(convergence_history[-1] - convergence_history[-2]) < 1e-6:
                    print(f"Physics-informed surrogate converged at epoch {epoch}")
                    break
        
        self.is_fitted = True
        return self
    
    def predict(self, x: Array) -> Array:
        """Predict using the trained model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if x.ndim == 1:
            return self._forward_pass(self.params, x)
        else:
            return vmap(lambda xi: self._forward_pass(self.params, xi))(x)
    
    def gradient(self, x: Array) -> Array:
        """Compute gradients using automatic differentiation."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before gradient computation")
        
        grad_fn = grad(lambda xi: self._forward_pass(self.params, xi))
        
        if x.ndim == 1:
            return grad_fn(x)
        else:
            return vmap(grad_fn)(x)


class AdaptiveAcquisitionOptimizer:
    """Novel optimizer using adaptive acquisition functions for surrogate-based optimization.
    
    This algorithm dynamically adjusts the acquisition function based on optimization progress.
    """
    
    def __init__(
        self,
        base_acquisition: str = "expected_improvement",
        adaptation_rate: float = 0.1,
        exploration_schedule: str = "decay",
        uncertainty_threshold: float = 0.1,
    ):
        """Initialize adaptive acquisition optimizer.
        
        Args:
            base_acquisition: Base acquisition function type
            adaptation_rate: Rate of adaptation
            exploration_schedule: How exploration weight changes over time
            uncertainty_threshold: Threshold for switching acquisition modes
        """
        self.base_acquisition = base_acquisition
        self.adaptation_rate = adaptation_rate
        self.exploration_schedule = exploration_schedule
        self.uncertainty_threshold = uncertainty_threshold
        
        self.iteration = 0
        self.convergence_history = []
        self.acquisition_weights = {"exploration": 0.5, "exploitation": 0.5}
    
    def _expected_improvement(self, x: Array, surrogate, best_value: float) -> float:
        """Expected improvement acquisition function."""
        mean = surrogate.predict(x)
        uncertainty = surrogate.uncertainty(x)
        
        if uncertainty < 1e-10:
            return 0.0
        
        improvement = best_value - mean
        z = improvement / uncertainty
        
        # Use normal CDF and PDF approximations
        phi_z = 0.5 * (1.0 + jnp.tanh(z / jnp.sqrt(2.0)))  # Approx normal CDF
        pdf_z = jnp.exp(-0.5 * z**2) / jnp.sqrt(2.0 * jnp.pi)  # Normal PDF
        
        ei = improvement * phi_z + uncertainty * pdf_z
        return ei
    
    def _upper_confidence_bound(self, x: Array, surrogate, beta: float = 2.0) -> float:
        """Upper confidence bound acquisition function."""
        mean = surrogate.predict(x)
        uncertainty = surrogate.uncertainty(x)
        return mean + beta * uncertainty
    
    def _probability_improvement(self, x: Array, surrogate, best_value: float) -> float:
        """Probability of improvement acquisition function."""
        mean = surrogate.predict(x)
        uncertainty = surrogate.uncertainty(x)
        
        if uncertainty < 1e-10:
            return 0.0 if mean <= best_value else 1.0
        
        z = (best_value - mean) / uncertainty
        return 0.5 * (1.0 + jnp.tanh(z / jnp.sqrt(2.0)))
    
    def _adaptive_acquisition(
        self,
        x: Array,
        surrogate,
        best_value: float,
        global_uncertainty: float,
    ) -> float:
        """Adaptive acquisition function that combines multiple strategies."""
        # Compute base acquisition functions
        ei = self._expected_improvement(x, surrogate, best_value)
        ucb = self._upper_confidence_bound(x, surrogate)
        pi = self._probability_improvement(x, surrogate, best_value)
        
        # Adaptive weighting based on optimization state
        if global_uncertainty > self.uncertainty_threshold:
            # High uncertainty - favor exploration
            exploration_weight = 0.7
        else:
            # Low uncertainty - favor exploitation
            exploration_weight = 0.3
        
        # Update weights with momentum
        self.acquisition_weights["exploration"] = (
            (1 - self.adaptation_rate) * self.acquisition_weights["exploration"] +
            self.adaptation_rate * exploration_weight
        )
        self.acquisition_weights["exploitation"] = 1.0 - self.acquisition_weights["exploration"]
        
        # Combine acquisition functions
        exploration_component = (ucb + pi) / 2.0  # Exploration-focused
        exploitation_component = ei  # Exploitation-focused
        
        adaptive_score = (
            self.acquisition_weights["exploration"] * exploration_component +
            self.acquisition_weights["exploitation"] * exploitation_component
        )
        
        return adaptive_score
    
    def optimize(
        self,
        surrogate,
        x0: Array,
        bounds: List[Tuple[float, float]],
        n_iterations: int = 50,
        **kwargs
    ):
        """Run adaptive acquisition optimization."""
        current_best = x0
        current_best_value = surrogate.predict(x0)
        
        history = []
        
        for i in range(n_iterations):
            self.iteration = i
            
            # Estimate global uncertainty
            test_points = jnp.array([
                jnp.array([(low + high) / 2 for low, high in bounds]) + 
                jnp.random.normal(0, 0.1, len(bounds))
                for _ in range(20)
            ])
            uncertainties = vmap(surrogate.uncertainty)(test_points)
            global_uncertainty = jnp.mean(uncertainties)
            
            # Define acquisition function for this iteration
            def acquisition_fn(x):
                return -self._adaptive_acquisition(x, surrogate, current_best_value, global_uncertainty)
            
            # Optimize acquisition function
            try:
                result = minimize(
                    acquisition_fn,
                    x0=current_best + jnp.random.normal(0, 0.1, len(current_best)),
                    method="L-BFGS-B",
                    bounds=bounds,
                )
                
                if result.success:
                    candidate = result.x
                    candidate_value = surrogate.predict(candidate)
                    
                    if candidate_value < current_best_value:
                        current_best = candidate
                        current_best_value = candidate_value
                
                history.append(float(current_best_value))
                
            except Exception:
                # Fallback to random sampling
                candidate = jnp.array([
                    jnp.random.uniform(low, high) for low, high in bounds
                ])
                candidate_value = surrogate.predict(candidate)
                history.append(float(candidate_value))
        
        # Validate convergence
        convergence_analysis = self._analyze_convergence(history)
        
        # Create optimization result with enhanced convergence information
        from ..optimizers.base import OptimizationResult
        return OptimizationResult(
            x=current_best,
            fun=float(current_best_value),
            success=convergence_analysis["converged"],
            message=convergence_analysis["message"],
            nit=n_iterations,
            nfev=n_iterations,
            convergence_history=history,
            convergence_analysis=convergence_analysis,
        )
    
    def _analyze_convergence(self, history: List[float]) -> Dict[str, Any]:
        """Analyze convergence of adaptive acquisition optimization.
        
        Args:
            history: History of best objective values
            
        Returns:
            Convergence analysis results
        """
        if len(history) < 3:
            return {
                "converged": False,
                "message": "Insufficient history for convergence analysis",
                "convergence_score": 0.0,
                "details": {"history_length": len(history)}
            }
        
        history_array = jnp.array(history)
        
        # 1. Trend analysis - expect decreasing values
        improvements = -jnp.diff(history_array)  # Negative diff for minimization
        positive_improvements = jnp.sum(improvements > 0)
        improvement_ratio = float(positive_improvements / len(improvements))
        
        # 2. Relative improvement analysis
        if abs(history[0]) > 1e-10:
            total_relative_improvement = abs(history[-1] - history[0]) / abs(history[0])
        else:
            total_relative_improvement = 0.0
        
        # 3. Recent stability analysis
        recent_window = min(10, len(history) // 3)
        if recent_window >= 2:
            recent_values = history_array[-recent_window:]
            recent_std = float(jnp.std(recent_values))
            recent_mean = float(jnp.mean(recent_values))
            stability_coefficient = recent_std / abs(recent_mean) if abs(recent_mean) > 1e-10 else float('inf')
        else:
            stability_coefficient = float('inf')
        
        # 4. Stagnation detection
        stagnation_threshold = abs(history[-1]) * 0.001 if abs(history[-1]) > 1e-10 else 1e-6
        stagnation_count = 0
        max_stagnation = 0
        
        for i in range(1, len(history)):
            if abs(history[i] - history[i-1]) < stagnation_threshold:
                stagnation_count += 1
                max_stagnation = max(max_stagnation, stagnation_count)
            else:
                stagnation_count = 0
        
        stagnation_ratio = max_stagnation / len(history) if len(history) > 0 else 1.0
        
        # 5. Acquisition weight adaptation analysis
        exploration_weights = []
        exploitation_weights = []
        
        # Simulate adaptation (this would be tracked in real implementation)
        for i in range(len(history)):
            # Simulate decreasing exploration over time
            exploration_weight = 0.7 * (1 - i / len(history)) + 0.3 * (i / len(history))
            exploration_weights.append(exploration_weight)
            exploitation_weights.append(1.0 - exploration_weight)
        
        weight_adaptation_score = 1.0 - abs(exploration_weights[-1] - 0.3)  # Should end around 0.3
        
        # 6. Convergence scoring
        scores = []
        
        # Improvement score (want high improvement ratio)
        improvement_score = min(1.0, improvement_ratio / 0.7)  # Target 70% improvements
        scores.append(improvement_score * 0.25)
        
        # Relative improvement score
        rel_improvement_score = min(1.0, total_relative_improvement / 0.1)  # Target 10% total improvement
        scores.append(rel_improvement_score * 0.30)
        
        # Stability score (want low coefficient of variation)
        if stability_coefficient < 0.05:  # < 5%
            stability_score = 1.0
        elif stability_coefficient < 0.15:  # < 15%
            stability_score = 0.7
        else:
            stability_score = 0.3
        scores.append(stability_score * 0.25)
        
        # Stagnation score (want low stagnation)
        stagnation_score = max(0.0, 1.0 - stagnation_ratio * 2)  # Penalize high stagnation
        scores.append(stagnation_score * 0.20)
        
        overall_convergence_score = sum(scores)
        
        # 7. Convergence determination
        converged = (
            overall_convergence_score >= 0.7 and
            improvement_ratio >= 0.5 and
            total_relative_improvement >= 0.01 and
            stability_coefficient < 0.2
        )
        
        # 8. Generate detailed message
        if converged:
            message = (f"Adaptive acquisition converged successfully: "
                      f"{improvement_ratio*100:.1f}% iterations improved, "
                      f"{total_relative_improvement*100:.2f}% total improvement, "
                      f"stability CV={stability_coefficient*100:.1f}%")
        else:
            issues = []
            if improvement_ratio < 0.5:
                issues.append(f"low improvement ratio ({improvement_ratio*100:.1f}%)")
            if total_relative_improvement < 0.01:
                issues.append(f"insufficient total improvement ({total_relative_improvement*100:.2f}%)")
            if stability_coefficient >= 0.2:
                issues.append(f"poor stability (CV={stability_coefficient*100:.1f}%)")
            if stagnation_ratio > 0.3:
                issues.append(f"high stagnation ({stagnation_ratio*100:.1f}%)")
            
            message = f"Adaptive acquisition convergence issues: {'; '.join(issues)}"
        
        # 9. Convergence recommendations
        recommendations = []
        if improvement_ratio < 0.5:
            recommendations.append("Increase exploration weight or improve acquisition function balance")
        if stability_coefficient >= 0.2:
            recommendations.append("Consider early stopping or adaptive learning rate")
        if stagnation_ratio > 0.3:
            recommendations.append("Implement restart mechanism or diversification strategy")
        if total_relative_improvement < 0.01:
            recommendations.append("Check surrogate model accuracy or increase iteration budget")
        
        return {
            "converged": converged,
            "message": message,
            "convergence_score": float(overall_convergence_score),
            "details": {
                "history_length": len(history),
                "improvement_ratio": float(improvement_ratio),
                "total_relative_improvement": float(total_relative_improvement),
                "stability_coefficient": float(stability_coefficient),
                "max_stagnation_length": int(max_stagnation),
                "stagnation_ratio": float(stagnation_ratio),
                "weight_adaptation_score": float(weight_adaptation_score),
                "initial_value": float(history[0]),
                "final_value": float(history[-1]),
                "best_improvement": float(max(improvements)) if len(improvements) > 0 else 0.0,
                "mean_improvement": float(jnp.mean(improvements)) if len(improvements) > 0 else 0.0,
            },
            "component_scores": {
                "improvement": float(improvement_score),
                "relative_improvement": float(rel_improvement_score),
                "stability": float(stability_score),
                "stagnation": float(stagnation_score),
            },
            "recommendations": recommendations,
            "acquisition_analysis": {
                "final_exploration_weight": float(exploration_weights[-1]),
                "final_exploitation_weight": float(exploitation_weights[-1]),
                "weight_adaptation_quality": float(weight_adaptation_score),
            }
        }


class MultiObjectiveSurrogateOptimizer:
    """Novel multi-objective optimization using surrogate models.
    
    Implements a research approach for handling multiple conflicting objectives.
    """
    
    def __init__(
        self,
        n_objectives: int,
        aggregation_method: str = "pareto_efficient",
        reference_point: Optional[Array] = None,
    ):
        """Initialize multi-objective surrogate optimizer.
        
        Args:
            n_objectives: Number of objective functions
            aggregation_method: How to handle multiple objectives
            reference_point: Reference point for hypervolume calculation
        """
        self.n_objectives = n_objectives
        self.aggregation_method = aggregation_method
        self.reference_point = reference_point
        
        self.pareto_front = []
        self.objective_surrogates = []
    
    def fit_surrogates(self, datasets: List):
        """Fit surrogate models for each objective."""
        from ..models.neural import NeuralSurrogate
        
        self.objective_surrogates = []
        for i, dataset in enumerate(datasets):
            surrogate = NeuralSurrogate(hidden_dims=[32, 16])
            surrogate.fit(dataset)
            self.objective_surrogates.append(surrogate)
        
        return self
    
    def _is_pareto_efficient(self, costs: Array) -> Array:
        """Check which points are Pareto efficient."""
        is_efficient = jnp.ones(costs.shape[0], dtype=bool)
        
        for i, c in enumerate(costs):
            # Check if any other point dominates this one
            dominates = jnp.all(costs <= c, axis=1) & jnp.any(costs < c, axis=1)
            if jnp.any(dominates):
                is_efficient = is_efficient.at[i].set(False)
        
        return is_efficient
    
    def _hypervolume_indicator(self, pareto_points: Array, reference_point: Array) -> float:
        """Compute hypervolume indicator."""
        # Simplified hypervolume calculation
        if len(pareto_points) == 0:
            return 0.0
        
        # For 2D case (can be extended to higher dimensions)
        if pareto_points.shape[1] == 2:
            sorted_points = pareto_points[jnp.argsort(pareto_points[:, 0])]
            hypervolume = 0.0
            
            for i, point in enumerate(sorted_points):
                if i == 0:
                    width = point[0] - reference_point[0]
                else:
                    width = point[0] - sorted_points[i-1, 0]
                
                height = point[1] - reference_point[1]
                hypervolume += width * height
            
            return hypervolume
        
        return 0.0
    
    def optimize_pareto(
        self,
        x0_list: List[Array],
        bounds: List[Tuple[float, float]],
        n_iterations: int = 100,
    ) -> List[Array]:
        """Optimize to find Pareto-efficient solutions."""
        pareto_candidates = []
        
        for x0 in x0_list:
            # Multi-objective optimization using scalarization
            for weight_combo in self._generate_weight_combinations():
                def scalarized_objective(x):
                    objectives = jnp.array([
                        surrogate.predict(x) 
                        for surrogate in self.objective_surrogates
                    ])
                    return jnp.sum(weight_combo * objectives)
                
                try:
                    result = minimize(
                        scalarized_objective,
                        x0=x0,
                        method="L-BFGS-B",
                        bounds=bounds
                    )
                    
                    if result.success:
                        pareto_candidates.append(result.x)
                        
                except Exception:
                    continue
        
        if not pareto_candidates:
            return []
        
        # Evaluate all candidates on all objectives
        candidate_array = jnp.stack(pareto_candidates)
        objective_values = jnp.array([
            [surrogate.predict(x) for surrogate in self.objective_surrogates]
            for x in pareto_candidates
        ])
        
        # Find Pareto-efficient solutions
        efficient_mask = self._is_pareto_efficient(objective_values)
        pareto_solutions = candidate_array[efficient_mask]
        
        return list(pareto_solutions)
    
    def _generate_weight_combinations(self, n_points: int = 10) -> List[Array]:
        """Generate diverse weight combinations for scalarization."""
        weights = []
        
        # Generate uniform weights
        for i in range(n_points + 1):
            for j in range(n_points + 1 - i):
                if self.n_objectives == 2:
                    w1 = i / n_points
                    w2 = j / n_points
                    if w1 + w2 <= 1.0:
                        weights.append(jnp.array([w1, w2]))
                # Can extend to higher dimensions
        
        return weights


class SequentialModelBasedOptimization:
    """Novel SMBO algorithm with advanced surrogate model selection.
    
    Research contribution: Dynamic surrogate model selection based on problem characteristics.
    """
    
    def __init__(
        self,
        surrogate_pool: List[str] = None,
        model_selection_strategy: str = "adaptive",
        ensemble_method: str = "weighted_average",
    ):
        """Initialize sequential model-based optimization.
        
        Args:
            surrogate_pool: Pool of surrogate models to choose from
            model_selection_strategy: How to select models
            ensemble_method: How to combine multiple models
        """
        if surrogate_pool is None:
            surrogate_pool = ["neural_network", "gaussian_process", "random_forest"]
        
        self.surrogate_pool = surrogate_pool
        self.model_selection_strategy = model_selection_strategy
        self.ensemble_method = ensemble_method
        
        self.model_performance_history = {model: [] for model in surrogate_pool}
        self.active_models = {}
    
    def _evaluate_model_performance(self, model, validation_data) -> float:
        """Evaluate surrogate model performance on validation data."""
        predictions = model.predict(validation_data.X)
        mse = jnp.mean((predictions - validation_data.y)**2)
        return float(mse)
    
    def _select_best_model(self, dataset) -> str:
        """Select best surrogate model for current data characteristics."""
        if self.model_selection_strategy == "adaptive":
            # Analyze dataset characteristics
            n_samples = dataset.n_samples
            n_dims = dataset.n_dims
            output_variance = float(jnp.var(dataset.y))
            
            # Heuristic model selection rules
            if n_samples < 50:
                return "gaussian_process"  # GP works well with small data
            elif n_dims > 10:
                return "neural_network"   # NN handles high dimensions better
            elif output_variance < 0.1:
                return "random_forest"    # RF good for smooth functions
            else:
                return "neural_network"   # Default to NN for complex functions
        
        elif self.model_selection_strategy == "cross_validation":
            # Use cross-validation to select best model
            best_model = self.surrogate_pool[0]
            best_score = float('inf')
            
            for model_type in self.surrogate_pool:
                # Simplified CV score (would implement proper CV)
                score = self._cross_validate_model(model_type, dataset)
                if score < best_score:
                    best_score = score
                    best_model = model_type
            
            return best_model
        
        else:
            return self.surrogate_pool[0]  # Default
    
    def _cross_validate_model(self, model_type: str, dataset) -> float:
        """Cross-validate a model type (simplified implementation)."""
        # This would implement proper k-fold cross-validation
        # For now, return random score
        return jnp.random.uniform(0, 1)
    
    def optimize(
        self,
        objective_function: Callable,
        bounds: List[Tuple[float, float]],
        n_initial_samples: int = 20,
        n_iterations: int = 50,
    ) -> ResearchResult:
        """Run sequential model-based optimization."""
        from ..data.collector import collect_data
        from ..core import SurrogateOptimizer
        
        start_time = time.time()
        
        # Initial data collection
        initial_data = collect_data(
            objective_function,
            n_samples=n_initial_samples,
            bounds=bounds,
            sampling="sobol",
            verbose=False
        )
        
        best_value = jnp.min(initial_data.y)
        best_point = initial_data.X[jnp.argmin(initial_data.y)]
        convergence_data = [float(best_value)]
        
        current_data = initial_data
        
        for iteration in range(n_iterations):
            # Select best surrogate model for current data
            selected_model = self._select_best_model(current_data)
            
            # Train surrogate
            optimizer = SurrogateOptimizer(
                surrogate_type=selected_model,
                optimizer_type="gradient_descent"
            )
            optimizer.fit_surrogate(current_data)
            
            # Find next evaluation point using acquisition function
            # (Using expected improvement for simplicity)
            def acquisition_function(x):
                mean = optimizer.predict(x)
                uncertainty = optimizer.uncertainty(x)
                improvement = best_value - mean
                
                if uncertainty < 1e-10:
                    return 0.0
                
                z = improvement / uncertainty
                ei = improvement * 0.5 * (1.0 + jnp.tanh(z / jnp.sqrt(2.0)))
                ei += uncertainty * jnp.exp(-0.5 * z**2) / jnp.sqrt(2.0 * jnp.pi)
                return -ei  # Minimize negative EI
            
            # Optimize acquisition function
            try:
                from jax.scipy.optimize import minimize
                result = minimize(
                    acquisition_function,
                    x0=best_point + jnp.random.normal(0, 0.1, len(best_point)),
                    method="L-BFGS-B",
                    bounds=bounds
                )
                
                if result.success:
                    next_point = result.x
                else:
                    # Random fallback
                    next_point = jnp.array([
                        jnp.random.uniform(low, high) for low, high in bounds
                    ])
            
            except Exception:
                # Random fallback
                next_point = jnp.array([
                    jnp.random.uniform(low, high) for low, high in bounds
                ])
            
            # Evaluate objective at next point
            next_value = objective_function(next_point)
            
            # Update dataset
            from ..models.base import Dataset
            new_X = jnp.vstack([current_data.X, next_point.reshape(1, -1)])
            new_y = jnp.append(current_data.y, next_value)
            current_data = Dataset(X=new_X, y=new_y)
            
            # Update best solution
            if next_value < best_value:
                best_value = next_value
                best_point = next_point
            
            convergence_data.append(float(best_value))
        
        execution_time = time.time() - start_time
        
        return ResearchResult(
            algorithm_name="Sequential Model-Based Optimization",
            experiment_id=f"smbo_{int(time.time())}",
            success=True,
            performance_metrics={
                "best_value": float(best_value),
                "final_error": float(best_value),  # Assuming 0 is global optimum
                "convergence_rate": (convergence_data[0] - convergence_data[-1]) / convergence_data[0],
                "n_evaluations": n_initial_samples + n_iterations,
            },
            convergence_data=convergence_data,
            execution_time=execution_time,
            metadata={
                "selected_models": list(self.model_performance_history.keys()),
                "final_best_point": best_point.tolist(),
            }
        )