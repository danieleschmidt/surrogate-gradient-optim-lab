"""Autonomous Tuning System for real-time optimization adaptation.

This system continuously monitors optimization performance and automatically
adjusts hyperparameters, algorithms, and strategies without human intervention.
"""

import time
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import logging
from concurrent.futures import ThreadPoolExecutor, Future

import jax
import jax.numpy as jnp
from jax import Array
import numpy as np
from scipy.optimize import differential_evolution
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


@dataclass
class TuningConfiguration:
    """Configuration for autonomous tuning system."""
    max_concurrent_tuning: int = 3
    tuning_frequency_seconds: int = 300  # 5 minutes
    performance_window_size: int = 100
    min_improvement_threshold: float = 0.02
    exploration_ratio: float = 0.2
    max_tuning_iterations: int = 50
    convergence_tolerance: float = 1e-4


@dataclass
class PerformanceMetric:
    """Single performance measurement."""
    timestamp: float
    optimization_time: float
    convergence_rate: float
    final_objective_value: float
    iterations_to_convergence: int
    resource_usage: Dict[str, float]
    configuration: Dict[str, Any]
    success: bool


@dataclass
class TuningResult:
    """Result of an autonomous tuning session."""
    original_config: Dict[str, Any]
    optimized_config: Dict[str, Any]
    performance_improvement: float
    confidence_score: float
    tuning_duration: float
    iterations_performed: int


class AutonomousTuningSystem:
    """System for autonomous hyperparameter and algorithm tuning.
    
    This system runs continuously in the background, monitoring optimization
    performance and automatically tuning parameters to improve efficiency.
    """
    
    def __init__(
        self,
        optimization_function: Callable,
        config: Optional[TuningConfiguration] = None
    ):
        """Initialize the autonomous tuning system.
        
        Args:
            optimization_function: Function that performs optimization given config
            config: Tuning system configuration
        """
        self.optimization_function = optimization_function
        self.config = config or TuningConfiguration()
        
        # Performance tracking
        self.performance_history = deque(maxlen=self.config.performance_window_size)
        self.current_configuration: Dict[str, Any] = {}
        self.baseline_performance: Optional[float] = None
        
        # Tuning state
        self.tuning_active = False
        self.tuning_results_history: List[TuningResult] = []
        self.parameter_space = self._define_parameter_space()
        self.gp_surrogate: Optional[GaussianProcessRegressor] = None
        
        # Threading
        self.tuning_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_concurrent_tuning)
        self.active_tuning_futures: List[Future] = []
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
        # Parameter bounds and types
        self.parameter_bounds = self._initialize_parameter_bounds()
        
    def _define_parameter_space(self) -> Dict[str, Dict[str, Any]]:
        """Define the parameter space for tuning."""
        return {
            "neural_network": {
                "learning_rate": {"type": "log_uniform", "bounds": [1e-5, 1e-1]},
                "batch_size": {"type": "choice", "values": [16, 32, 64, 128, 256]},
                "hidden_layers": {"type": "choice", "values": [[32], [64], [128], [64, 32], [128, 64], [128, 64, 32]]},
                "dropout_rate": {"type": "uniform", "bounds": [0.0, 0.5]},
                "epochs": {"type": "choice", "values": [50, 100, 200, 500]},
                "activation": {"type": "choice", "values": ["relu", "tanh", "gelu", "swish"]},
            },
            "gaussian_process": {
                "length_scale": {"type": "log_uniform", "bounds": [0.1, 10.0]},
                "noise_level": {"type": "log_uniform", "bounds": [1e-6, 1e-1]},
                "kernel": {"type": "choice", "values": ["rbf", "matern", "rational_quadratic"]},
                "alpha": {"type": "log_uniform", "bounds": [1e-10, 1e-3]},
            },
            "optimization": {
                "max_iterations": {"type": "choice", "values": [50, 100, 200, 500, 1000]},
                "tolerance": {"type": "log_uniform", "bounds": [1e-8, 1e-4]},
                "step_size": {"type": "log_uniform", "bounds": [1e-4, 1e-1]},
                "momentum": {"type": "uniform", "bounds": [0.0, 0.99]},
            },
            "acquisition": {
                "exploration_weight": {"type": "uniform", "bounds": [0.01, 10.0]},
                "acquisition_function": {"type": "choice", "values": ["ei", "pi", "ucb", "poi"]},
                "batch_size": {"type": "choice", "values": [1, 5, 10, 20]},
            }
        }
    
    def _initialize_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Initialize parameter bounds for optimization."""
        bounds = {}
        param_index = 0
        
        for category, params in self.parameter_space.items():
            for param_name, param_config in params.items():
                if param_config["type"] in ["uniform", "log_uniform"]:
                    bounds[f"{category}.{param_name}"] = tuple(param_config["bounds"])
                    param_index += 1
        
        return bounds
    
    def start_autonomous_tuning(self, initial_config: Dict[str, Any]) -> None:
        """Start the autonomous tuning system.
        
        Args:
            initial_config: Initial configuration to start with
        """
        self.current_configuration = initial_config.copy()
        self.tuning_active = True
        
        # Start background tuning thread
        self.tuning_thread = threading.Thread(target=self._tuning_loop, daemon=True)
        self.tuning_thread.start()
        
        self.logger.info("Autonomous tuning system started")
    
    def stop_autonomous_tuning(self) -> None:
        """Stop the autonomous tuning system."""
        self.tuning_active = False
        self.stop_event.set()
        
        if self.tuning_thread:
            self.tuning_thread.join(timeout=10)
        
        # Cancel active futures
        for future in self.active_tuning_futures:
            future.cancel()
        
        self.executor.shutdown(wait=False)
        
        self.logger.info("Autonomous tuning system stopped")
    
    def record_performance(self, performance_data: Dict[str, Any]) -> None:
        """Record performance data for analysis.
        
        Args:
            performance_data: Dictionary containing performance metrics
        """
        metric = PerformanceMetric(
            timestamp=time.time(),
            optimization_time=performance_data.get("optimization_time", 0.0),
            convergence_rate=performance_data.get("convergence_rate", 0.0),
            final_objective_value=performance_data.get("final_objective_value", float('inf')),
            iterations_to_convergence=performance_data.get("iterations_to_convergence", 0),
            resource_usage=performance_data.get("resource_usage", {}),
            configuration=performance_data.get("configuration", {}),
            success=performance_data.get("success", False)
        )
        
        self.performance_history.append(metric)
        
        # Update baseline if this is the first measurement
        if self.baseline_performance is None and metric.success:
            self.baseline_performance = metric.final_objective_value
    
    def _tuning_loop(self) -> None:
        """Main loop for autonomous tuning."""
        while self.tuning_active and not self.stop_event.is_set():
            try:
                # Wait for tuning frequency
                if self.stop_event.wait(self.config.tuning_frequency_seconds):
                    break
                
                # Check if we have enough data for tuning
                if len(self.performance_history) < 10:
                    self.logger.debug("Insufficient performance data for tuning")
                    continue
                
                # Analyze current performance
                recent_performance = self._analyze_recent_performance()
                
                # Decide if tuning is needed
                if self._should_trigger_tuning(recent_performance):
                    self.logger.info("Triggering autonomous tuning session")
                    
                    # Launch tuning in background
                    future = self.executor.submit(self._perform_tuning_session)
                    self.active_tuning_futures.append(future)
                    
                    # Clean up completed futures
                    self.active_tuning_futures = [f for f in self.active_tuning_futures if not f.done()]
                
            except Exception as e:
                self.logger.error(f"Error in tuning loop: {e}")
    
    def _analyze_recent_performance(self) -> Dict[str, float]:
        """Analyze recent performance metrics.
        
        Returns:
            Dictionary with performance analysis results
        """
        if not self.performance_history:
            return {}
        
        recent_metrics = list(self.performance_history)[-20:]  # Last 20 measurements
        successful_metrics = [m for m in recent_metrics if m.success]
        
        if not successful_metrics:
            return {"success_rate": 0.0}
        
        analysis = {
            "success_rate": len(successful_metrics) / len(recent_metrics),
            "avg_optimization_time": np.mean([m.optimization_time for m in successful_metrics]),
            "avg_convergence_rate": np.mean([m.convergence_rate for m in successful_metrics]),
            "avg_objective_value": np.mean([m.final_objective_value for m in successful_metrics]),
            "avg_iterations": np.mean([m.iterations_to_convergence for m in successful_metrics]),
            "performance_variance": np.var([m.final_objective_value for m in successful_metrics]),
        }
        
        return analysis
    
    def _should_trigger_tuning(self, performance_analysis: Dict[str, float]) -> bool:
        """Determine if tuning should be triggered.
        
        Args:
            performance_analysis: Recent performance analysis
            
        Returns:
            True if tuning should be triggered
        """
        if not performance_analysis:
            return False
        
        # Check success rate
        if performance_analysis.get("success_rate", 1.0) < 0.7:
            self.logger.info("Tuning triggered: Low success rate")
            return True
        
        # Check performance degradation
        if len(self.performance_history) >= 50:
            old_metrics = list(self.performance_history)[-50:-25]
            new_metrics = list(self.performance_history)[-25:]
            
            old_avg = np.mean([m.final_objective_value for m in old_metrics if m.success])
            new_avg = np.mean([m.final_objective_value for m in new_metrics if m.success])
            
            if len([m for m in old_metrics if m.success]) > 5 and len([m for m in new_metrics if m.success]) > 5:
                # Performance has degraded (assuming minimization)
                if new_avg > old_avg * (1 + self.config.min_improvement_threshold):
                    self.logger.info("Tuning triggered: Performance degradation detected")
                    return True
        
        # Check high variance (instability)
        if performance_analysis.get("performance_variance", 0) > 1.0:
            self.logger.info("Tuning triggered: High performance variance")
            return True
        
        # Periodic tuning (every 10th cycle)
        if len(self.tuning_results_history) % 10 == 0 and len(self.performance_history) >= 100:
            self.logger.info("Tuning triggered: Periodic optimization")
            return True
        
        return False
    
    def _perform_tuning_session(self) -> Optional[TuningResult]:
        """Perform a single tuning session.
        
        Returns:
            TuningResult if successful, None otherwise
        """
        start_time = time.time()
        original_config = self.current_configuration.copy()
        
        try:
            # Build surrogate model of performance
            self._build_performance_surrogate()
            
            # Optimize configuration using surrogate
            optimized_config = self._optimize_configuration()
            
            if optimized_config is None:
                return None
            
            # Evaluate the optimized configuration
            performance_improvement = self._evaluate_configuration_improvement(
                original_config, optimized_config
            )
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(optimized_config)
            
            # Create tuning result
            result = TuningResult(
                original_config=original_config,
                optimized_config=optimized_config,
                performance_improvement=performance_improvement,
                confidence_score=confidence_score,
                tuning_duration=time.time() - start_time,
                iterations_performed=self.config.max_tuning_iterations
            )
            
            # Update current configuration if improvement is significant
            if (performance_improvement > self.config.min_improvement_threshold and 
                confidence_score > 0.6):
                
                self.current_configuration = optimized_config.copy()
                self.logger.info(f"Configuration updated with {performance_improvement:.2%} improvement")
            
            self.tuning_results_history.append(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error during tuning session: {e}")
            return None
    
    def _build_performance_surrogate(self) -> None:
        """Build a surrogate model of configuration -> performance mapping."""
        if len(self.performance_history) < 20:
            return
        
        # Extract features and targets
        X, y = self._extract_training_data()
        
        if X.size == 0 or y.size == 0:
            return
        
        # Create and fit Gaussian Process
        kernel = Matern(length_scale=1.0, nu=2.5)
        self.gp_surrogate = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=10,
        )
        
        try:
            self.gp_surrogate.fit(X, y)
            self.logger.debug(f"Built performance surrogate with {len(X)} samples")
        except Exception as e:
            self.logger.warning(f"Failed to build performance surrogate: {e}")
            self.gp_surrogate = None
    
    def _extract_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Extract training data for surrogate model.
        
        Returns:
            Tuple of (features, targets)
        """
        successful_metrics = [m for m in self.performance_history if m.success]
        
        if len(successful_metrics) < 10:
            return np.array([]), np.array([])
        
        features = []
        targets = []
        
        for metric in successful_metrics:
            # Extract numerical features from configuration
            feature_vector = self._config_to_feature_vector(metric.configuration)
            
            if feature_vector is not None:
                features.append(feature_vector)
                targets.append(metric.final_objective_value)
        
        return np.array(features), np.array(targets)
    
    def _config_to_feature_vector(self, config: Dict[str, Any]) -> Optional[List[float]]:
        """Convert configuration to feature vector.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Feature vector or None if conversion fails
        """
        try:
            features = []
            
            # Neural network parameters
            nn_config = config.get("neural_network", {})
            features.extend([
                np.log10(nn_config.get("learning_rate", 0.001)),
                float(nn_config.get("batch_size", 32)),
                len(nn_config.get("hidden_layers", [64])),
                float(nn_config.get("dropout_rate", 0.1)),
                float(nn_config.get("epochs", 100)),
            ])
            
            # GP parameters
            gp_config = config.get("gaussian_process", {})
            features.extend([
                np.log10(gp_config.get("length_scale", 1.0)),
                np.log10(gp_config.get("noise_level", 0.01)),
                np.log10(gp_config.get("alpha", 1e-5)),
            ])
            
            # Optimization parameters
            opt_config = config.get("optimization", {})
            features.extend([
                float(opt_config.get("max_iterations", 100)),
                np.log10(opt_config.get("tolerance", 1e-6)),
                np.log10(opt_config.get("step_size", 0.01)),
                float(opt_config.get("momentum", 0.9)),
            ])
            
            return features
            
        except Exception as e:
            self.logger.warning(f"Failed to convert config to features: {e}")
            return None
    
    def _optimize_configuration(self) -> Optional[Dict[str, Any]]:
        """Optimize configuration using surrogate model.
        
        Returns:
            Optimized configuration or None if optimization fails
        """
        if self.gp_surrogate is None:
            return self._random_configuration_search()
        
        # Define objective function for optimization
        def objective(x: np.ndarray) -> float:
            config = self._feature_vector_to_config(x)
            if config is None:
                return float('inf')
            
            # Predict performance using surrogate
            try:
                prediction, uncertainty = self.gp_surrogate.predict([x], return_std=True)
                # Use lower confidence bound for robust optimization
                return prediction[0] - 0.5 * uncertainty[0]
            except Exception:
                return float('inf')
        
        # Extract bounds for continuous parameters
        bounds = []
        for category in ["neural_network", "gaussian_process", "optimization"]:
            category_params = self.parameter_space.get(category, {})
            for param_name, param_config in category_params.items():
                if param_config["type"] in ["uniform", "log_uniform"]:
                    bounds.append(tuple(param_config["bounds"]))
        
        if not bounds:
            return self._random_configuration_search()
        
        # Optimize using differential evolution
        try:
            result = differential_evolution(
                objective,
                bounds,
                maxiter=self.config.max_tuning_iterations,
                tol=self.config.convergence_tolerance,
                seed=int(time.time()) % 2**32,
                workers=1  # Single worker to avoid threading issues
            )
            
            if result.success:
                optimized_config = self._feature_vector_to_config(result.x)
                return optimized_config
            else:
                self.logger.warning("Optimization did not converge")
                return None
                
        except Exception as e:
            self.logger.error(f"Configuration optimization failed: {e}")
            return self._random_configuration_search()
    
    def _feature_vector_to_config(self, x: np.ndarray) -> Optional[Dict[str, Any]]:
        """Convert feature vector back to configuration.
        
        Args:
            x: Feature vector
            
        Returns:
            Configuration dictionary or None if conversion fails
        """
        try:
            config = {}
            idx = 0
            
            # Neural network parameters
            config["neural_network"] = {
                "learning_rate": 10 ** x[idx],
                "batch_size": int(x[idx + 1]),
                "hidden_layers": [64, 32],  # Default, would need more complex mapping
                "dropout_rate": x[idx + 2],
                "epochs": int(x[idx + 3]),
                "activation": "relu"  # Default
            }
            idx += 5
            
            # GP parameters
            config["gaussian_process"] = {
                "length_scale": 10 ** x[idx],
                "noise_level": 10 ** x[idx + 1],
                "alpha": 10 ** x[idx + 2],
                "kernel": "rbf"  # Default
            }
            idx += 3
            
            # Optimization parameters
            config["optimization"] = {
                "max_iterations": int(x[idx]),
                "tolerance": 10 ** x[idx + 1],
                "step_size": 10 ** x[idx + 2],
                "momentum": x[idx + 3]
            }
            
            return config
            
        except Exception as e:
            self.logger.warning(f"Failed to convert features to config: {e}")
            return None
    
    def _random_configuration_search(self) -> Dict[str, Any]:
        """Perform random search for configuration optimization.
        
        Returns:
            Randomly optimized configuration
        """
        best_config = self.current_configuration.copy()
        
        # Try a few random configurations
        for _ in range(10):
            config = self._generate_random_config()
            
            # Simple evaluation (could be improved)
            score = np.random.random()  # Placeholder
            
            # Update best if better (this is a simplified approach)
            if score > 0.7:  # Arbitrary threshold
                best_config = config
                break
        
        return best_config
    
    def _generate_random_config(self) -> Dict[str, Any]:
        """Generate a random configuration within parameter bounds.
        
        Returns:
            Random configuration dictionary
        """
        config = {}
        
        for category, params in self.parameter_space.items():
            config[category] = {}
            
            for param_name, param_config in params.items():
                if param_config["type"] == "uniform":
                    low, high = param_config["bounds"]
                    config[category][param_name] = np.random.uniform(low, high)
                
                elif param_config["type"] == "log_uniform":
                    low, high = param_config["bounds"]
                    config[category][param_name] = 10 ** np.random.uniform(np.log10(low), np.log10(high))
                
                elif param_config["type"] == "choice":
                    config[category][param_name] = np.random.choice(param_config["values"])
        
        return config
    
    def _evaluate_configuration_improvement(
        self,
        original_config: Dict[str, Any],
        optimized_config: Dict[str, Any]
    ) -> float:
        """Evaluate improvement of optimized configuration over original.
        
        Args:
            original_config: Original configuration
            optimized_config: Optimized configuration
            
        Returns:
            Estimated improvement ratio
        """
        if self.gp_surrogate is None:
            return 0.0
        
        try:
            original_features = self._config_to_feature_vector(original_config)
            optimized_features = self._config_to_feature_vector(optimized_config)
            
            if original_features is None or optimized_features is None:
                return 0.0
            
            original_pred = self.gp_surrogate.predict([original_features])[0]
            optimized_pred = self.gp_surrogate.predict([optimized_features])[0]
            
            # Calculate relative improvement (assuming minimization)
            if original_pred > 0:
                improvement = (original_pred - optimized_pred) / abs(original_pred)
            else:
                improvement = 0.0
            
            return max(0.0, improvement)  # Non-negative improvement
            
        except Exception as e:
            self.logger.warning(f"Failed to evaluate configuration improvement: {e}")
            return 0.0
    
    def _calculate_confidence_score(self, config: Dict[str, Any]) -> float:
        """Calculate confidence score for a configuration.
        
        Args:
            config: Configuration to evaluate
            
        Returns:
            Confidence score between 0 and 1
        """
        if self.gp_surrogate is None:
            return 0.5  # Medium confidence without surrogate
        
        try:
            features = self._config_to_feature_vector(config)
            if features is None:
                return 0.0
            
            _, uncertainty = self.gp_surrogate.predict([features], return_std=True)
            
            # Convert uncertainty to confidence (lower uncertainty = higher confidence)
            max_uncertainty = 2.0  # Arbitrary scaling
            confidence = max(0.0, 1.0 - uncertainty[0] / max_uncertainty)
            
            return min(1.0, confidence)
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate confidence score: {e}")
            return 0.5
    
    def get_current_configuration(self) -> Dict[str, Any]:
        """Get the current optimized configuration.
        
        Returns:
            Current configuration dictionary
        """
        return self.current_configuration.copy()
    
    def get_tuning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive tuning statistics.
        
        Returns:
            Dictionary containing tuning metrics and statistics
        """
        if not self.tuning_results_history:
            return {
                "total_tuning_sessions": 0,
                "average_improvement": 0.0,
                "average_confidence": 0.0,
                "total_tuning_time": 0.0,
                "tuning_active": self.tuning_active,
            }
        
        return {
            "total_tuning_sessions": len(self.tuning_results_history),
            "average_improvement": np.mean([r.performance_improvement for r in self.tuning_results_history]),
            "average_confidence": np.mean([r.confidence_score for r in self.tuning_results_history]),
            "total_tuning_time": sum(r.tuning_duration for r in self.tuning_results_history),
            "best_improvement": max(r.performance_improvement for r in self.tuning_results_history),
            "recent_improvements": [r.performance_improvement for r in self.tuning_results_history[-5:]],
            "tuning_active": self.tuning_active,
            "performance_samples": len(self.performance_history),
            "baseline_performance": self.baseline_performance,
        }