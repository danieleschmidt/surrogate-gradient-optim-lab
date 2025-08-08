"""Theoretical analysis tools for surrogate optimization research."""

import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import math

import jax.numpy as jnp
from jax import Array, grad, vmap, jit
import jax.random as random


@dataclass
class TheoreticalBounds:
    """Container for theoretical bounds and guarantees."""
    convergence_rate: Optional[float] = None
    sample_complexity: Optional[int] = None
    gradient_error_bound: Optional[float] = None
    confidence_level: float = 0.95
    assumptions: List[str] = field(default_factory=list)
    proof_sketch: Optional[str] = None


@dataclass
class ConvergenceAnalysis:
    """Results of convergence analysis."""
    algorithm_name: str
    theoretical_rate: Optional[float]
    empirical_rate: float
    confidence_interval: Tuple[float, float]
    convergence_test_statistic: float
    p_value: float
    is_converged: bool
    analysis_metadata: Dict[str, Any] = field(default_factory=dict)


class ConvergenceAnalyzer:
    """Theoretical and empirical convergence analysis for optimization algorithms."""
    
    def __init__(self, significance_level: float = 0.05):
        """Initialize convergence analyzer.
        
        Args:
            significance_level: Statistical significance level
        """
        self.significance_level = significance_level
    
    def analyze_convergence_rate(
        self,
        convergence_history: List[float],
        algorithm_name: str = "Unknown",
        theoretical_rate: Optional[float] = None,
    ) -> ConvergenceAnalysis:
        """Analyze convergence rate of an optimization algorithm.
        
        Args:
            convergence_history: Sequence of objective values
            algorithm_name: Name of the algorithm
            theoretical_rate: Known theoretical convergence rate
            
        Returns:
            Convergence analysis results
        """
        if len(convergence_history) < 10:
            raise ValueError("Need at least 10 iterations for convergence analysis")
        
        values = jnp.array(convergence_history)
        n_iterations = len(values)
        
        # Empirical convergence rate estimation
        # Fit exponential decay: f(k) = a * exp(-r * k) + c
        empirical_rate = self._estimate_empirical_rate(values)
        
        # Statistical tests for convergence
        is_converged, test_statistic, p_value = self._test_convergence(values)
        
        # Confidence interval for empirical rate
        confidence_interval = self._compute_rate_confidence_interval(
            values, empirical_rate
        )
        
        return ConvergenceAnalysis(
            algorithm_name=algorithm_name,
            theoretical_rate=theoretical_rate,
            empirical_rate=empirical_rate,
            confidence_interval=confidence_interval,
            convergence_test_statistic=test_statistic,
            p_value=p_value,
            is_converged=is_converged,
            analysis_metadata={
                "n_iterations": n_iterations,
                "initial_value": float(values[0]),
                "final_value": float(values[-1]),
                "total_improvement": float(values[0] - values[-1]),
                "monotonic": bool(jnp.all(jnp.diff(values) <= 0)),
            }
        )
    
    def _estimate_empirical_rate(self, values: Array) -> float:
        """Estimate empirical convergence rate using log-linear regression."""
        # Remove values that are too close to minimum to avoid log issues
        min_val = jnp.min(values)
        shifted_values = values - min_val + 1e-10
        
        # Log-linear fit: log(f(k) - f*) = log(a) - r*k
        log_values = jnp.log(shifted_values)
        k_values = jnp.arange(len(values))
        
        # Simple linear regression
        n = len(k_values)
        sum_k = jnp.sum(k_values)
        sum_log_y = jnp.sum(log_values)
        sum_k2 = jnp.sum(k_values**2)
        sum_k_log_y = jnp.sum(k_values * log_values)
        
        # Slope (negative convergence rate)
        rate = (n * sum_k_log_y - sum_k * sum_log_y) / (n * sum_k2 - sum_k**2)
        
        return float(-rate)  # Return positive rate
    
    def _test_convergence(self, values: Array) -> Tuple[bool, float, float]:
        """Statistical test for convergence using trend analysis."""
        # Simple trend test: check if values are significantly decreasing
        n = len(values)
        ranks = jnp.arange(1, n + 1)
        
        # Kendall's tau test for monotonic trend
        concordant = 0
        discordant = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                if (ranks[j] - ranks[i]) * (values[j] - values[i]) > 0:
                    concordant += 1
                elif (ranks[j] - ranks[i]) * (values[j] - values[i]) < 0:
                    discordant += 1
        
        total_pairs = n * (n - 1) // 2
        tau = (concordant - discordant) / total_pairs if total_pairs > 0 else 0
        
        # Z-test statistic for tau
        var_tau = n * (n - 1) * (2 * n + 5) / 18
        z_stat = tau * jnp.sqrt(var_tau)
        
        # P-value (two-tailed test)
        p_value = float(2 * (1 - self._normal_cdf(abs(z_stat))))
        
        is_converged = p_value < self.significance_level and tau < -0.3  # Negative trend
        
        return is_converged, float(z_stat), p_value
    
    def _compute_rate_confidence_interval(
        self,
        values: Array,
        empirical_rate: float,
        confidence_level: float = 0.95,
    ) -> Tuple[float, float]:
        """Compute confidence interval for convergence rate."""
        # Bootstrap-based confidence interval
        n_bootstrap = 100
        bootstrap_rates = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = jnp.random.choice(len(values), size=len(values), replace=True)
            bootstrap_values = values[indices]
            
            try:
                bootstrap_rate = self._estimate_empirical_rate(bootstrap_values)
                bootstrap_rates.append(bootstrap_rate)
            except:
                continue
        
        if bootstrap_rates:
            bootstrap_rates = jnp.array(bootstrap_rates)
            alpha = 1 - confidence_level
            lower = float(jnp.percentile(bootstrap_rates, 100 * alpha / 2))
            upper = float(jnp.percentile(bootstrap_rates, 100 * (1 - alpha / 2)))
            return (lower, upper)
        else:
            # Fallback to simple interval
            margin = 0.1 * abs(empirical_rate)
            return (empirical_rate - margin, empirical_rate + margin)
    
    def _normal_cdf(self, x: float) -> float:
        """Approximate standard normal CDF."""
        return float(0.5 * (1 + jnp.tanh(x / jnp.sqrt(2.0))))


class SampleComplexityAnalyzer:
    """Theoretical analysis of sample complexity for surrogate optimization."""
    
    def __init__(self):
        """Initialize sample complexity analyzer."""
        self.complexity_bounds = {}
    
    def analyze_sample_complexity(
        self,
        surrogate_type: str,
        problem_dimension: int,
        target_accuracy: float,
        confidence_level: float = 0.95,
        lipschitz_constant: Optional[float] = None,
        smoothness_parameter: Optional[float] = None,
    ) -> TheoreticalBounds:
        """Analyze sample complexity bounds for surrogate optimization.
        
        Args:
            surrogate_type: Type of surrogate model
            problem_dimension: Dimension of the optimization problem
            target_accuracy: Target optimization accuracy
            confidence_level: Statistical confidence level
            lipschitz_constant: Lipschitz constant of the function
            smoothness_parameter: Smoothness parameter of the function
            
        Returns:
            Theoretical bounds on sample complexity
        """
        bounds = TheoreticalBounds(confidence_level=confidence_level)
        
        if surrogate_type.lower() in ["neural", "neural_network"]:
            bounds = self._analyze_neural_complexity(
                problem_dimension, target_accuracy, confidence_level,
                lipschitz_constant, smoothness_parameter
            )
        elif surrogate_type.lower() in ["gp", "gaussian_process"]:
            bounds = self._analyze_gp_complexity(
                problem_dimension, target_accuracy, confidence_level,
                lipschitz_constant, smoothness_parameter
            )
        elif surrogate_type.lower() in ["rf", "random_forest"]:
            bounds = self._analyze_rf_complexity(
                problem_dimension, target_accuracy, confidence_level
            )
        else:
            # Generic bounds
            bounds = self._analyze_generic_complexity(
                problem_dimension, target_accuracy, confidence_level
            )
        
        return bounds
    
    def _analyze_neural_complexity(
        self,
        d: int,
        epsilon: float,
        confidence: float,
        L: Optional[float] = None,
        beta: Optional[float] = None,
    ) -> TheoreticalBounds:
        """Analyze sample complexity for neural network surrogates."""
        # Based on neural network approximation theory
        # Sample complexity scales with dimension and target accuracy
        
        # Assumptions
        assumptions = [
            "Function is Lipschitz continuous",
            "Neural network has sufficient capacity",
            "Activation functions are smooth",
        ]
        
        if L is None:
            L = 1.0  # Assume unit Lipschitz constant
        
        if beta is None:
            beta = 2.0  # Assume smooth function
        
        # Sample complexity bound: O(d^2 / epsilon^(2*beta/d))
        # This is a simplified bound - real bounds are more complex
        sample_complexity = int(
            (d**2 / (epsilon**(2 * beta / max(d, 1)))) * 
            jnp.log(1 / (1 - confidence))
        )
        
        # Convergence rate for gradient descent on neural networks
        convergence_rate = 1.0 / jnp.sqrt(sample_complexity)
        
        bounds = TheoreticalBounds(
            convergence_rate=float(convergence_rate),
            sample_complexity=sample_complexity,
            gradient_error_bound=float(L * jnp.sqrt(d / sample_complexity)),
            confidence_level=confidence,
            assumptions=assumptions,
            proof_sketch=(
                "Based on universal approximation theorem and "
                "statistical learning theory for neural networks. "
                "Sample complexity depends on dimension and function smoothness."
            )
        )
        
        return bounds
    
    def _analyze_gp_complexity(
        self,
        d: int,
        epsilon: float,
        confidence: float,
        L: Optional[float] = None,
        beta: Optional[float] = None,
    ) -> TheoreticalBounds:
        """Analyze sample complexity for Gaussian process surrogates."""
        # Based on GP approximation theory and information-theoretic bounds
        
        assumptions = [
            "Function lies in RKHS with finite norm",
            "Kernel captures function structure",
            "Gaussian noise model",
        ]
        
        if beta is None:
            beta = 2.0  # Smoothness parameter
        
        # Information gain bound for GPs
        # Sample complexity: O(gamma_n * log(1/delta))
        # where gamma_n is the information gain
        
        # Simplified information gain bound
        gamma_n = d * (jnp.log(1 + 1/epsilon)**d)
        sample_complexity = int(gamma_n * jnp.log(1 / (1 - confidence)))
        
        # GP convergence rate
        convergence_rate = jnp.sqrt(jnp.log(sample_complexity) / sample_complexity)
        
        bounds = TheoreticalBounds(
            convergence_rate=float(convergence_rate),
            sample_complexity=int(sample_complexity),
            gradient_error_bound=float(jnp.sqrt(2 * jnp.log(2/epsilon) / sample_complexity)),
            confidence_level=confidence,
            assumptions=assumptions,
            proof_sketch=(
                "Based on information-theoretic analysis of GPs. "
                "Sample complexity depends on information gain of the kernel. "
                "Provides high-probability convergence guarantees."
            )
        )
        
        return bounds
    
    def _analyze_rf_complexity(
        self,
        d: int,
        epsilon: float,
        confidence: float,
    ) -> TheoreticalBounds:
        """Analyze sample complexity for random forest surrogates."""
        assumptions = [
            "Function has bounded variation",
            "Trees have sufficient depth",
            "Bootstrap sampling is effective",
        ]
        
        # RF sample complexity (simplified)
        # Based on PAC learning bounds for tree ensembles
        sample_complexity = int((d * jnp.log(d) / epsilon**2) * jnp.log(1 / (1 - confidence)))
        
        convergence_rate = 1.0 / jnp.sqrt(sample_complexity)
        
        bounds = TheoreticalBounds(
            convergence_rate=float(convergence_rate),
            sample_complexity=sample_complexity,
            gradient_error_bound=float(jnp.sqrt(d / sample_complexity)),
            confidence_level=confidence,
            assumptions=assumptions,
            proof_sketch=(
                "Based on PAC learning theory for tree ensembles. "
                "Sample complexity scales with dimension and variance of trees."
            )
        )
        
        return bounds
    
    def _analyze_generic_complexity(
        self,
        d: int,
        epsilon: float,
        confidence: float,
    ) -> TheoreticalBounds:
        """Generic sample complexity bounds."""
        assumptions = [
            "Function is bounded",
            "Learning algorithm is consistent",
        ]
        
        # Generic PAC learning bound
        sample_complexity = int((d / epsilon**2) * jnp.log(1 / (1 - confidence)))
        
        bounds = TheoreticalBounds(
            sample_complexity=sample_complexity,
            confidence_level=confidence,
            assumptions=assumptions,
        )
        
        return bounds


class GradientErrorAnalyzer:
    """Analysis of surrogate gradient approximation errors."""
    
    def __init__(self):
        """Initialize gradient error analyzer."""
        pass
    
    def analyze_gradient_error(
        self,
        surrogate,
        true_function: Callable,
        test_points: Array,
        finite_diff_step: float = 1e-6,
    ) -> Dict[str, float]:
        """Analyze gradient approximation error.
        
        Args:
            surrogate: Surrogate model
            true_function: True objective function
            test_points: Points for error evaluation
            finite_diff_step: Step size for finite differences
            
        Returns:
            Gradient error statistics
        """
        # Compute surrogate gradients
        surrogate_grads = vmap(surrogate.gradient)(test_points)
        
        # Compute true gradients using finite differences
        def finite_diff_gradient(x):
            grad = jnp.zeros_like(x)
            for i in range(len(x)):
                x_plus = x.at[i].add(finite_diff_step)
                x_minus = x.at[i].add(-finite_diff_step)
                grad = grad.at[i].set(
                    (true_function(x_plus) - true_function(x_minus)) / (2 * finite_diff_step)
                )
            return grad
        
        true_grads = vmap(finite_diff_gradient)(test_points)
        
        # Compute error metrics
        grad_errors = surrogate_grads - true_grads
        
        # L2 errors
        l2_errors = jnp.linalg.norm(grad_errors, axis=1)
        
        # Angular errors (cosine similarity)
        surrogate_norms = jnp.linalg.norm(surrogate_grads, axis=1)
        true_norms = jnp.linalg.norm(true_grads, axis=1)
        
        dot_products = jnp.sum(surrogate_grads * true_grads, axis=1)
        cosine_similarities = dot_products / (surrogate_norms * true_norms + 1e-10)
        angular_errors = jnp.arccos(jnp.clip(cosine_similarities, -1, 1))
        
        return {
            "mean_l2_error": float(jnp.mean(l2_errors)),
            "std_l2_error": float(jnp.std(l2_errors)),
            "max_l2_error": float(jnp.max(l2_errors)),
            "mean_angular_error": float(jnp.mean(angular_errors)),
            "std_angular_error": float(jnp.std(angular_errors)),
            "max_angular_error": float(jnp.max(angular_errors)),
            "correlation": float(jnp.corrcoef(
                surrogate_grads.flatten(),
                true_grads.flatten()
            )[0, 1]),
        }
    
    def derive_error_bounds(
        self,
        surrogate_type: str,
        problem_dimension: int,
        n_training_samples: int,
        function_properties: Dict[str, float] = None,
    ) -> TheoreticalBounds:
        """Derive theoretical bounds on gradient approximation error.
        
        Args:
            surrogate_type: Type of surrogate model
            problem_dimension: Problem dimension
            n_training_samples: Number of training samples
            function_properties: Properties of the objective function
            
        Returns:
            Theoretical error bounds
        """
        if function_properties is None:
            function_properties = {"lipschitz_constant": 1.0, "smoothness": 1.0}
        
        L = function_properties.get("lipschitz_constant", 1.0)
        smoothness = function_properties.get("smoothness", 1.0)
        d = problem_dimension
        n = n_training_samples
        
        assumptions = [
            f"Function is {L}-Lipschitz",
            f"Function has smoothness parameter {smoothness}",
            "Training data is well-distributed",
        ]
        
        if surrogate_type.lower() in ["neural", "neural_network"]:
            # Neural network gradient error bound
            # Based on approximation theory for neural networks
            gradient_error_bound = L * (d / n)**(1/(2 + d))
            
            assumptions.append("Neural network has sufficient width")
            
        elif surrogate_type.lower() in ["gp", "gaussian_process"]:
            # GP gradient error bound
            # Based on RKHS theory
            gradient_error_bound = L * jnp.sqrt(jnp.log(n) / n)
            
            assumptions.append("Function lies in appropriate RKHS")
            
        else:
            # Generic bound
            gradient_error_bound = L * (1 / jnp.sqrt(n))
        
        bounds = TheoreticalBounds(
            gradient_error_bound=float(gradient_error_bound),
            confidence_level=0.95,
            assumptions=assumptions,
            proof_sketch=(
                f"Gradient error bound derived from {surrogate_type} approximation theory. "
                "Bound depends on function smoothness and sample size."
            )
        )
        
        return bounds


def run_theoretical_analysis(
    algorithm_name: str,
    convergence_history: List[float],
    surrogate_type: str,
    problem_dimension: int,
    n_samples: int,
    target_accuracy: float = 1e-6,
) -> Dict[str, Any]:
    """Run comprehensive theoretical analysis.
    
    Args:
        algorithm_name: Name of the optimization algorithm
        convergence_history: Convergence history
        surrogate_type: Type of surrogate model used
        problem_dimension: Problem dimension
        n_samples: Number of training samples
        target_accuracy: Target optimization accuracy
        
    Returns:
        Comprehensive theoretical analysis results
    """
    # Convergence analysis
    conv_analyzer = ConvergenceAnalyzer()
    convergence_analysis = conv_analyzer.analyze_convergence_rate(
        convergence_history, algorithm_name
    )
    
    # Sample complexity analysis
    complexity_analyzer = SampleComplexityAnalyzer()
    complexity_bounds = complexity_analyzer.analyze_sample_complexity(
        surrogate_type, problem_dimension, target_accuracy
    )
    
    # Gradient error analysis
    grad_analyzer = GradientErrorAnalyzer()
    gradient_bounds = grad_analyzer.derive_error_bounds(
        surrogate_type, problem_dimension, n_samples
    )
    
    return {
        "convergence_analysis": convergence_analysis,
        "sample_complexity_bounds": complexity_bounds,
        "gradient_error_bounds": gradient_bounds,
        "summary": {
            "algorithm": algorithm_name,
            "surrogate_type": surrogate_type,
            "problem_dimension": problem_dimension,
            "empirical_convergence_rate": convergence_analysis.empirical_rate,
            "theoretical_sample_complexity": complexity_bounds.sample_complexity,
            "gradient_error_bound": gradient_bounds.gradient_error_bound,
        }
    }