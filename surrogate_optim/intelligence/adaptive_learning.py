"""Adaptive Learning Engine for autonomous capability enhancement.

This module implements self-improving algorithms that learn from usage patterns
and automatically optimize performance without human intervention.
"""

from collections import defaultdict, deque
from dataclasses import dataclass, field
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from jax import Array
import numpy as np
from sklearn.ensemble import IsolationForest


@dataclass
class LearningMetrics:
    """Metrics for tracking learning performance."""
    accuracy_improvements: List[float] = field(default_factory=list)
    speed_improvements: List[float] = field(default_factory=list)
    resource_efficiency_gains: List[float] = field(default_factory=list)
    adaptation_timestamps: List[float] = field(default_factory=list)
    confidence_scores: List[float] = field(default_factory=list)


@dataclass
class UsagePattern:
    """Pattern extracted from system usage."""
    problem_type: str
    dimension_range: Tuple[int, int]
    sample_count_range: Tuple[int, int]
    optimization_difficulty: float
    success_rate: float
    avg_convergence_time: float
    resource_usage: Dict[str, float]


class AdaptiveLearningEngine:
    """Engine for autonomous learning and system adaptation.
    
    This engine continuously monitors system performance and automatically
    adapts algorithms, hyperparameters, and resource allocation based on
    observed usage patterns and outcomes.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        adaptation_threshold: float = 0.05,
        memory_window: int = 1000,
        min_samples_for_adaptation: int = 50,
        confidence_threshold: float = 0.8,
    ):
        """Initialize the adaptive learning engine.
        
        Args:
            learning_rate: Rate of adaptation for parameters
            adaptation_threshold: Minimum improvement required for adaptation
            memory_window: Number of recent experiences to remember
            min_samples_for_adaptation: Minimum samples before adapting
            confidence_threshold: Minimum confidence for making adaptations
        """
        self.learning_rate = learning_rate
        self.adaptation_threshold = adaptation_threshold
        self.memory_window = memory_window
        self.min_samples_for_adaptation = min_samples_for_adaptation
        self.confidence_threshold = confidence_threshold

        # Learning state
        self.experience_buffer = deque(maxlen=memory_window)
        self.performance_history = defaultdict(list)
        self.adaptation_history: List[Dict[str, Any]] = []
        self.usage_patterns: List[UsagePattern] = []
        self.metrics = LearningMetrics()

        # Algorithm configurations
        self.algorithm_configs = self._initialize_algorithm_configs()
        self.current_best_configs = self.algorithm_configs.copy()

        # Anomaly detection for identifying new problem types
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.anomaly_detector_fitted = False

        # Logger
        self.logger = logging.getLogger(__name__)

    def _initialize_algorithm_configs(self) -> Dict[str, Dict[str, Any]]:
        """Initialize default algorithm configurations."""
        return {
            "neural_surrogate": {
                "layers": [128, 64, 32],
                "learning_rate": 0.001,
                "batch_size": 32,
                "dropout_rate": 0.1,
                "epochs": 100,
                "activation": "relu",
            },
            "gp_surrogate": {
                "kernel": "rbf",
                "length_scale": 1.0,
                "noise_level": 0.1,
                "alpha": 1e-5,
            },
            "optimizer": {
                "max_iterations": 100,
                "tolerance": 1e-6,
                "learning_rate": 0.01,
                "momentum": 0.9,
            },
            "data_collection": {
                "sampling_strategy": "sobol",
                "batch_size": 10,
                "acquisition_function": "expected_improvement",
            }
        }

    def record_experience(
        self,
        problem_context: Dict[str, Any],
        algorithm_used: str,
        configuration: Dict[str, Any],
        performance_metrics: Dict[str, float],
        success: bool,
        execution_time: float,
        resource_usage: Dict[str, float],
    ) -> None:
        """Record a new experience for learning.
        
        Args:
            problem_context: Context about the optimization problem
            algorithm_used: Name of algorithm that was used
            configuration: Configuration parameters used
            performance_metrics: Performance metrics achieved
            success: Whether the optimization was successful
            execution_time: Time taken for execution
            resource_usage: Resources consumed (memory, CPU, etc.)
        """
        experience = {
            "timestamp": time.time(),
            "problem_context": problem_context,
            "algorithm": algorithm_used,
            "configuration": configuration,
            "performance": performance_metrics,
            "success": success,
            "execution_time": execution_time,
            "resource_usage": resource_usage,
        }

        self.experience_buffer.append(experience)

        # Update performance history
        context_key = self._generate_context_key(problem_context)
        self.performance_history[context_key].append({
            "performance": performance_metrics,
            "configuration": configuration,
            "success": success,
        })

        # Trigger adaptation if enough data
        if len(self.experience_buffer) >= self.min_samples_for_adaptation:
            self._attempt_adaptation()

    def _generate_context_key(self, context: Dict[str, Any]) -> str:
        """Generate a key for problem context categorization."""
        dimension = context.get("dimension", 0)
        problem_type = context.get("problem_type", "unknown")
        difficulty = context.get("difficulty", "medium")

        return f"{problem_type}_{dimension}d_{difficulty}"

    def _attempt_adaptation(self) -> None:
        """Attempt to adapt system configuration based on learned patterns."""
        # Analyze recent experiences
        patterns = self._extract_usage_patterns()

        if not patterns:
            return

        # Identify improvement opportunities
        adaptations = self._identify_adaptations(patterns)

        # Apply adaptations with confidence filtering
        for adaptation in adaptations:
            if adaptation["confidence"] >= self.confidence_threshold:
                self._apply_adaptation(adaptation)

    def _extract_usage_patterns(self) -> List[UsagePattern]:
        """Extract usage patterns from recent experiences."""
        if len(self.experience_buffer) < self.min_samples_for_adaptation:
            return []

        # Group experiences by problem characteristics
        pattern_groups = defaultdict(list)

        for exp in list(self.experience_buffer):
            context = exp["problem_context"]
            problem_type = context.get("problem_type", "unknown")
            dimension = context.get("dimension", 0)

            # Create pattern group key
            group_key = f"{problem_type}_{dimension // 10 * 10}d"  # Group by decade
            pattern_groups[group_key].append(exp)

        patterns = []

        for group_key, experiences in pattern_groups.items():
            if len(experiences) < 10:  # Need sufficient samples
                continue

            # Analyze pattern
            dimensions = [exp["problem_context"].get("dimension", 0) for exp in experiences]
            sample_counts = [exp["problem_context"].get("n_samples", 0) for exp in experiences]
            success_rates = [exp["success"] for exp in experiences]
            convergence_times = [exp["execution_time"] for exp in experiences]

            pattern = UsagePattern(
                problem_type=group_key.split("_")[0],
                dimension_range=(min(dimensions), max(dimensions)),
                sample_count_range=(min(sample_counts), max(sample_counts)),
                optimization_difficulty=1.0 - np.mean(success_rates),
                success_rate=np.mean(success_rates),
                avg_convergence_time=np.mean(convergence_times),
                resource_usage={
                    "avg_memory": np.mean([exp["resource_usage"].get("memory", 0) for exp in experiences]),
                    "avg_cpu": np.mean([exp["resource_usage"].get("cpu", 0) for exp in experiences])
                }
            )

            patterns.append(pattern)

        return patterns

    def _identify_adaptations(self, patterns: List[UsagePattern]) -> List[Dict[str, Any]]:
        """Identify potential adaptations based on usage patterns."""
        adaptations = []

        for pattern in patterns:
            # Analyze performance for this pattern type
            pattern_key = f"{pattern.problem_type}_{pattern.dimension_range[1]//10*10}d"

            if pattern_key not in self.performance_history:
                continue

            performances = self.performance_history[pattern_key]

            if len(performances) < self.min_samples_for_adaptation:
                continue

            # Find best performing configurations
            best_configs = sorted(
                performances,
                key=lambda x: x["performance"].get("optimization_score", 0),
                reverse=True
            )[:5]  # Top 5 configurations

            if not best_configs:
                continue

            # Extract common characteristics of successful configurations
            successful_configs = [p for p in performances if p["success"]]

            if len(successful_configs) < 5:
                continue

            # Identify configuration adaptations
            adaptation = self._analyze_configuration_patterns(
                successful_configs, pattern
            )

            if adaptation:
                adaptations.append(adaptation)

        return adaptations

    def _analyze_configuration_patterns(
        self,
        successful_configs: List[Dict[str, Any]],
        pattern: UsagePattern
    ) -> Optional[Dict[str, Any]]:
        """Analyze configuration patterns to identify adaptations."""
        # Extract configuration values
        config_values = defaultdict(list)

        for config_entry in successful_configs:
            config = config_entry["configuration"]
            for key, value in config.items():
                if isinstance(value, (int, float)):
                    config_values[key].append(value)

        # Calculate optimal ranges
        adaptations = {}
        confidence_scores = []

        for param, values in config_values.items():
            if len(values) < 3:
                continue

            # Calculate statistics
            mean_val = np.mean(values)
            std_val = np.std(values)

            # Current configuration value
            current_val = self.current_best_configs.get("neural_surrogate", {}).get(param)

            if current_val is None:
                continue

            # Calculate improvement potential
            if abs(mean_val - current_val) > 0.1 * abs(current_val):
                improvement_estimate = abs(mean_val - current_val) / abs(current_val)

                if improvement_estimate > self.adaptation_threshold:
                    adaptations[param] = mean_val
                    confidence_scores.append(min(0.9, len(values) / 20))  # Confidence based on sample size

        if not adaptations:
            return None

        overall_confidence = np.mean(confidence_scores) if confidence_scores else 0.0

        return {
            "type": "parameter_optimization",
            "pattern": pattern,
            "adaptations": adaptations,
            "confidence": overall_confidence,
            "expected_improvement": sum(abs(v - self.current_best_configs.get("neural_surrogate", {}).get(k, v))
                                     for k, v in adaptations.items()) / len(adaptations)
        }

    def _apply_adaptation(self, adaptation: Dict[str, Any]) -> None:
        """Apply an adaptation to the system configuration."""
        if adaptation["type"] == "parameter_optimization":
            # Update algorithm configurations
            for param, value in adaptation["adaptations"].items():
                if "neural_surrogate" in self.current_best_configs:
                    old_value = self.current_best_configs["neural_surrogate"].get(param)

                    # Gradual adaptation with learning rate
                    if old_value is not None:
                        new_value = old_value + self.learning_rate * (value - old_value)
                        self.current_best_configs["neural_surrogate"][param] = new_value
                    else:
                        self.current_best_configs["neural_surrogate"][param] = value

            # Record adaptation
            adaptation_record = {
                "timestamp": time.time(),
                "type": adaptation["type"],
                "changes": adaptation["adaptations"],
                "confidence": adaptation["confidence"],
                "expected_improvement": adaptation["expected_improvement"]
            }

            self.adaptation_history.append(adaptation_record)

            # Update metrics
            self.metrics.adaptation_timestamps.append(time.time())
            self.metrics.confidence_scores.append(adaptation["confidence"])

            self.logger.info(f"Applied adaptation: {adaptation['adaptations']} with confidence {adaptation['confidence']:.3f}")

    def get_optimized_configuration(
        self,
        problem_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get optimized configuration for a given problem context.
        
        Args:
            problem_context: Context about the optimization problem
            
        Returns:
            Optimized configuration dictionary
        """
        context_key = self._generate_context_key(problem_context)

        # Check if we have learned patterns for this context
        if context_key in self.performance_history:
            # Use learned configuration
            return self.current_best_configs.copy()
        # Use default configuration but mark for learning
        self.logger.info(f"New problem context detected: {context_key}")
        return self.algorithm_configs.copy()

    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics.
        
        Returns:
            Dictionary containing learning metrics and statistics
        """
        return {
            "total_experiences": len(self.experience_buffer),
            "adaptations_made": len(self.adaptation_history),
            "patterns_identified": len(self.usage_patterns),
            "avg_confidence": np.mean(self.metrics.confidence_scores) if self.metrics.confidence_scores else 0.0,
            "recent_accuracy_improvement": np.mean(self.metrics.accuracy_improvements[-10:]) if self.metrics.accuracy_improvements else 0.0,
            "recent_speed_improvement": np.mean(self.metrics.speed_improvements[-10:]) if self.metrics.speed_improvements else 0.0,
            "learning_effectiveness": len(self.adaptation_history) / max(1, len(self.experience_buffer)) * 100,
            "problem_contexts_seen": len(self.performance_history),
            "current_configurations": self.current_best_configs,
        }

    def detect_anomalous_problems(self, problem_features: Array) -> bool:
        """Detect if a problem is anomalous (new problem type).
        
        Args:
            problem_features: Features describing the problem
            
        Returns:
            True if problem appears to be anomalous/new
        """
        if not self.anomaly_detector_fitted:
            # Collect features from experience buffer for training
            if len(self.experience_buffer) >= 20:
                features = []
                for exp in list(self.experience_buffer):
                    ctx = exp["problem_context"]
                    feature_vector = [
                        ctx.get("dimension", 0),
                        ctx.get("n_samples", 0),
                        exp["execution_time"],
                        1.0 if exp["success"] else 0.0,
                        exp["performance"].get("optimization_score", 0.0)
                    ]
                    features.append(feature_vector)

                self.anomaly_detector.fit(features)
                self.anomaly_detector_fitted = True

        if self.anomaly_detector_fitted:
            # Reshape for single sample prediction
            features_reshaped = np.array(problem_features).reshape(1, -1)
            is_anomaly = self.anomaly_detector.predict(features_reshaped)[0] == -1
            return is_anomaly

        return False  # Can't detect anomalies without training data

    def export_learned_knowledge(self) -> Dict[str, Any]:
        """Export learned knowledge for persistence or transfer.
        
        Returns:
            Dictionary containing all learned knowledge
        """
        return {
            "version": "1.0",
            "timestamp": time.time(),
            "configurations": self.current_best_configs,
            "performance_history": dict(self.performance_history),
            "adaptation_history": self.adaptation_history,
            "usage_patterns": [
                {
                    "problem_type": p.problem_type,
                    "dimension_range": p.dimension_range,
                    "sample_count_range": p.sample_count_range,
                    "optimization_difficulty": p.optimization_difficulty,
                    "success_rate": p.success_rate,
                    "avg_convergence_time": p.avg_convergence_time,
                    "resource_usage": p.resource_usage
                }
                for p in self.usage_patterns
            ],
            "metrics": {
                "accuracy_improvements": self.metrics.accuracy_improvements,
                "speed_improvements": self.metrics.speed_improvements,
                "resource_efficiency_gains": self.metrics.resource_efficiency_gains,
                "adaptation_timestamps": self.metrics.adaptation_timestamps,
                "confidence_scores": self.metrics.confidence_scores,
            }
        }

    def import_learned_knowledge(self, knowledge: Dict[str, Any]) -> None:
        """Import previously learned knowledge.
        
        Args:
            knowledge: Dictionary containing learned knowledge
        """
        if knowledge.get("version") != "1.0":
            self.logger.warning("Incompatible knowledge version, skipping import")
            return

        self.current_best_configs = knowledge.get("configurations", self.algorithm_configs.copy())

        # Import performance history
        perf_hist = knowledge.get("performance_history", {})
        for key, value in perf_hist.items():
            self.performance_history[key] = value

        self.adaptation_history = knowledge.get("adaptation_history", [])

        # Import usage patterns
        pattern_data = knowledge.get("usage_patterns", [])
        self.usage_patterns = [
            UsagePattern(
                problem_type=p["problem_type"],
                dimension_range=tuple(p["dimension_range"]),
                sample_count_range=tuple(p["sample_count_range"]),
                optimization_difficulty=p["optimization_difficulty"],
                success_rate=p["success_rate"],
                avg_convergence_time=p["avg_convergence_time"],
                resource_usage=p["resource_usage"]
            )
            for p in pattern_data
        ]

        # Import metrics
        metrics_data = knowledge.get("metrics", {})
        self.metrics = LearningMetrics(
            accuracy_improvements=metrics_data.get("accuracy_improvements", []),
            speed_improvements=metrics_data.get("speed_improvements", []),
            resource_efficiency_gains=metrics_data.get("resource_efficiency_gains", []),
            adaptation_timestamps=metrics_data.get("adaptation_timestamps", []),
            confidence_scores=metrics_data.get("confidence_scores", []),
        )

        self.logger.info(f"Imported knowledge with {len(self.adaptation_history)} adaptations and {len(self.usage_patterns)} patterns")
