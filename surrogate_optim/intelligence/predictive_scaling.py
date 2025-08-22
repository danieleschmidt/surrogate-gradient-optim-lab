"""Predictive Scaling Engine for real-time performance optimization.

This module implements intelligent scaling and resource optimization based on
predictive models that anticipate workload demands and performance requirements.
"""

import time
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import logging
import json

import jax
import jax.numpy as jnp
from jax import Array
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score


class ScalingAction(Enum):
    """Types of scaling actions that can be taken."""
    SCALE_UP_CPU = "scale_up_cpu"
    SCALE_DOWN_CPU = "scale_down_cpu"
    SCALE_UP_MEMORY = "scale_up_memory"
    SCALE_DOWN_MEMORY = "scale_down_memory"
    SCALE_UP_GPU = "scale_up_gpu"
    SCALE_DOWN_GPU = "scale_down_gpu"
    SCALE_OUT_WORKERS = "scale_out_workers"
    SCALE_IN_WORKERS = "scale_in_workers"
    OPTIMIZE_BATCH_SIZE = "optimize_batch_size"
    ADJUST_PARALLELISM = "adjust_parallelism"
    NO_ACTION = "no_action"


@dataclass
class ResourceMetrics:
    """Current resource utilization metrics."""
    timestamp: float
    cpu_utilization: float  # 0.0 to 1.0
    memory_utilization: float  # 0.0 to 1.0
    gpu_utilization: float  # 0.0 to 1.0
    network_io: float  # bytes/second
    disk_io: float  # bytes/second
    active_workers: int
    queue_length: int
    response_time: float  # seconds
    throughput: float  # operations/second
    error_rate: float  # 0.0 to 1.0


@dataclass
class WorkloadPrediction:
    """Predicted workload characteristics."""
    timestamp: float
    predicted_cpu_demand: float
    predicted_memory_demand: float
    predicted_gpu_demand: float
    predicted_throughput: float
    predicted_response_time: float
    confidence: float  # 0.0 to 1.0
    time_horizon: float  # seconds into future


@dataclass
class ScalingDecision:
    """Decision to take scaling action."""
    timestamp: float
    action: ScalingAction
    current_state: Dict[str, Any]
    target_state: Dict[str, Any]
    expected_improvement: float
    confidence: float
    reasoning: str


class WorkloadPredictor:
    """Predicts future workload based on historical patterns."""
    
    def __init__(self, prediction_horizon: int = 300):  # 5 minutes default
        """Initialize workload predictor.
        
        Args:
            prediction_horizon: How far into the future to predict (seconds)
        """
        self.prediction_horizon = prediction_horizon
        
        # Prediction models
        self.cpu_predictor: Optional[RandomForestRegressor] = None
        self.memory_predictor: Optional[RandomForestRegressor] = None
        self.throughput_predictor: Optional[RandomForestRegressor] = None
        self.response_time_predictor: Optional[RandomForestRegressor] = None
        
        # Feature scaling
        self.scaler = StandardScaler()
        self.scaler_fitted = False
        
        # Training data
        self.historical_data: deque = deque(maxlen=1000)  # Keep last 1000 data points
        
        # Model performance tracking
        self.prediction_accuracy: Dict[str, List[float]] = defaultdict(list)
        self.model_last_trained = 0
        self.training_interval = 300  # Retrain every 5 minutes
        
        # Logger
        self.logger = logging.getLogger(__name__)
    
    def add_observation(self, metrics: ResourceMetrics) -> None:
        """Add new resource metrics observation.
        
        Args:
            metrics: Current resource metrics
        """
        self.historical_data.append(metrics)
        
        # Retrain models periodically
        if (time.time() - self.model_last_trained) > self.training_interval:
            if len(self.historical_data) >= 50:  # Minimum data for training
                self._train_models()
    
    def predict_workload(self, horizon_seconds: Optional[int] = None) -> WorkloadPrediction:
        """Predict future workload.
        
        Args:
            horizon_seconds: How far into future to predict (uses default if None)
            
        Returns:
            Workload prediction
        """
        if horizon_seconds is None:
            horizon_seconds = self.prediction_horizon
        
        if not self._models_trained():
            return self._fallback_prediction()
        
        try:
            # Prepare features from recent history
            features = self._extract_features()
            
            if features is None:
                return self._fallback_prediction()
            
            # Make predictions
            cpu_pred = self.cpu_predictor.predict([features])[0]
            memory_pred = self.memory_predictor.predict([features])[0]
            throughput_pred = self.throughput_predictor.predict([features])[0]
            response_time_pred = self.response_time_predictor.predict([features])[0]
            
            # Estimate confidence based on model performance
            confidence = self._estimate_prediction_confidence()
            
            return WorkloadPrediction(
                timestamp=time.time(),
                predicted_cpu_demand=max(0.0, min(1.0, cpu_pred)),
                predicted_memory_demand=max(0.0, min(1.0, memory_pred)),
                predicted_gpu_demand=max(0.0, min(1.0, cpu_pred * 0.8)),  # Simple GPU estimation
                predicted_throughput=max(0.0, throughput_pred),
                predicted_response_time=max(0.0, response_time_pred),
                confidence=confidence,
                time_horizon=horizon_seconds
            )
            
        except Exception as e:
            self.logger.warning(f"Prediction failed: {e}, using fallback")
            return self._fallback_prediction()
    
    def _train_models(self) -> None:
        """Train prediction models using historical data."""
        try:
            # Prepare training data
            X, y_cpu, y_memory, y_throughput, y_response_time = self._prepare_training_data()
            
            if len(X) < 10:  # Need minimum data
                return
            
            # Fit scaler
            if not self.scaler_fitted:
                self.scaler.fit(X)
                self.scaler_fitted = True
            
            X_scaled = self.scaler.transform(X)
            
            # Train individual models
            self.cpu_predictor = RandomForestRegressor(n_estimators=50, random_state=42)
            self.cpu_predictor.fit(X_scaled, y_cpu)
            
            self.memory_predictor = RandomForestRegressor(n_estimators=50, random_state=42)
            self.memory_predictor.fit(X_scaled, y_memory)
            
            self.throughput_predictor = RandomForestRegressor(n_estimators=50, random_state=42)
            self.throughput_predictor.fit(X_scaled, y_throughput)
            
            self.response_time_predictor = RandomForestRegressor(n_estimators=50, random_state=42)
            self.response_time_predictor.fit(X_scaled, y_response_time)
            
            self.model_last_trained = time.time()
            
            # Evaluate model performance
            self._evaluate_models(X_scaled, y_cpu, y_memory, y_throughput, y_response_time)
            
            self.logger.info("Workload prediction models retrained successfully")
            
        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
    
    def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training data from historical observations.
        
        Returns:
            Tuple of (features, cpu_targets, memory_targets, throughput_targets, response_time_targets)
        """
        if len(self.historical_data) < 20:
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        
        # Create time series features and targets
        data_list = list(self.historical_data)
        
        # Use sliding window approach
        window_size = 10  # Look at last 10 time points for prediction
        features = []
        targets_cpu = []
        targets_memory = []
        targets_throughput = []
        targets_response_time = []
        
        for i in range(window_size, len(data_list)):
            # Features: statistics from previous window_size observations
            window = data_list[i-window_size:i]
            
            # Extract features from window
            feature_vector = self._extract_window_features(window)
            features.append(feature_vector)
            
            # Target: next observation
            target = data_list[i]
            targets_cpu.append(target.cpu_utilization)
            targets_memory.append(target.memory_utilization)
            targets_throughput.append(target.throughput)
            targets_response_time.append(target.response_time)
        
        return (np.array(features), np.array(targets_cpu), np.array(targets_memory),
                np.array(targets_throughput), np.array(targets_response_time))
    
    def _extract_window_features(self, window: List[ResourceMetrics]) -> List[float]:
        """Extract features from a window of observations.
        
        Args:
            window: List of resource metrics
            
        Returns:
            Feature vector
        """
        if not window:
            return [0.0] * 15  # Return zero features if empty
        
        # Statistical features from the window
        cpu_values = [m.cpu_utilization for m in window]
        memory_values = [m.memory_utilization for m in window]
        gpu_values = [m.gpu_utilization for m in window]
        throughput_values = [m.throughput for m in window]
        response_time_values = [m.response_time for m in window]
        
        features = [
            # CPU statistics
            np.mean(cpu_values),
            np.std(cpu_values),
            np.max(cpu_values),
            
            # Memory statistics
            np.mean(memory_values),
            np.std(memory_values),
            np.max(memory_values),
            
            # GPU statistics
            np.mean(gpu_values),
            np.std(gpu_values),
            
            # Performance statistics
            np.mean(throughput_values),
            np.std(throughput_values),
            np.mean(response_time_values),
            np.std(response_time_values),
            
            # System state
            window[-1].active_workers,
            window[-1].queue_length,
            window[-1].error_rate,
        ]
        
        return features
    
    def _extract_features(self) -> Optional[np.ndarray]:
        """Extract features from recent historical data for prediction.
        
        Returns:
            Feature array or None if insufficient data
        """
        if len(self.historical_data) < 10:
            return None
        
        # Use last 10 observations as features
        recent_window = list(self.historical_data)[-10:]
        features = self._extract_window_features(recent_window)
        
        if not self.scaler_fitted:
            return None
        
        return self.scaler.transform([features])[0]
    
    def _models_trained(self) -> bool:
        """Check if prediction models are trained.
        
        Returns:
            True if all models are trained
        """
        return all([
            self.cpu_predictor is not None,
            self.memory_predictor is not None,
            self.throughput_predictor is not None,
            self.response_time_predictor is not None,
            self.scaler_fitted
        ])
    
    def _evaluate_models(
        self, 
        X: np.ndarray, 
        y_cpu: np.ndarray, 
        y_memory: np.ndarray, 
        y_throughput: np.ndarray,
        y_response_time: np.ndarray
    ) -> None:
        """Evaluate model performance and track accuracy.
        
        Args:
            X: Feature matrix
            y_cpu: CPU utilization targets
            y_memory: Memory utilization targets
            y_throughput: Throughput targets
            y_response_time: Response time targets
        """
        try:
            # CPU model evaluation
            cpu_pred = self.cpu_predictor.predict(X)
            cpu_mae = mean_absolute_error(y_cpu, cpu_pred)
            cpu_r2 = r2_score(y_cpu, cpu_pred)
            self.prediction_accuracy["cpu_mae"].append(cpu_mae)
            self.prediction_accuracy["cpu_r2"].append(cpu_r2)
            
            # Memory model evaluation
            memory_pred = self.memory_predictor.predict(X)
            memory_mae = mean_absolute_error(y_memory, memory_pred)
            memory_r2 = r2_score(y_memory, memory_pred)
            self.prediction_accuracy["memory_mae"].append(memory_mae)
            self.prediction_accuracy["memory_r2"].append(memory_r2)
            
            # Throughput model evaluation
            throughput_pred = self.throughput_predictor.predict(X)
            throughput_mae = mean_absolute_error(y_throughput, throughput_pred)
            throughput_r2 = r2_score(y_throughput, throughput_pred)
            self.prediction_accuracy["throughput_mae"].append(throughput_mae)
            self.prediction_accuracy["throughput_r2"].append(throughput_r2)
            
            # Response time model evaluation
            response_pred = self.response_time_predictor.predict(X)
            response_mae = mean_absolute_error(y_response_time, response_pred)
            response_r2 = r2_score(y_response_time, response_pred)
            self.prediction_accuracy["response_time_mae"].append(response_mae)
            self.prediction_accuracy["response_time_r2"].append(response_r2)
            
        except Exception as e:
            self.logger.warning(f"Model evaluation failed: {e}")
    
    def _estimate_prediction_confidence(self) -> float:
        """Estimate confidence in current predictions.
        
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not self.prediction_accuracy["cpu_r2"]:
            return 0.5  # Medium confidence if no history
        
        # Average R² scores across all models (last 10 evaluations)
        recent_r2_scores = []
        
        for metric in ["cpu_r2", "memory_r2", "throughput_r2", "response_time_r2"]:
            if self.prediction_accuracy[metric]:
                recent_scores = self.prediction_accuracy[metric][-10:]  # Last 10 evaluations
                recent_r2_scores.extend(recent_scores)
        
        if not recent_r2_scores:
            return 0.5
        
        # Confidence based on average R² score
        avg_r2 = np.mean(recent_r2_scores)
        confidence = max(0.0, min(1.0, avg_r2))  # Clamp to [0, 1]
        
        return confidence
    
    def _fallback_prediction(self) -> WorkloadPrediction:
        """Generate fallback prediction when models aren't available.
        
        Returns:
            Basic prediction based on recent trends
        """
        if len(self.historical_data) == 0:
            # No data available - return conservative prediction
            return WorkloadPrediction(
                timestamp=time.time(),
                predicted_cpu_demand=0.5,
                predicted_memory_demand=0.5,
                predicted_gpu_demand=0.3,
                predicted_throughput=100.0,
                predicted_response_time=1.0,
                confidence=0.2,
                time_horizon=self.prediction_horizon
            )
        
        # Use simple trend-based prediction
        recent_metrics = list(self.historical_data)[-5:]  # Last 5 observations
        
        avg_cpu = np.mean([m.cpu_utilization for m in recent_metrics])
        avg_memory = np.mean([m.memory_utilization for m in recent_metrics])
        avg_gpu = np.mean([m.gpu_utilization for m in recent_metrics])
        avg_throughput = np.mean([m.throughput for m in recent_metrics])
        avg_response_time = np.mean([m.response_time for m in recent_metrics])
        
        return WorkloadPrediction(
            timestamp=time.time(),
            predicted_cpu_demand=avg_cpu,
            predicted_memory_demand=avg_memory,
            predicted_gpu_demand=avg_gpu,
            predicted_throughput=avg_throughput,
            predicted_response_time=avg_response_time,
            confidence=0.4,  # Lower confidence for fallback
            time_horizon=self.prediction_horizon
        )


class ScalingDecisionEngine:
    """Makes intelligent scaling decisions based on predictions and current state."""
    
    def __init__(self):
        """Initialize scaling decision engine."""
        
        # Decision thresholds
        self.cpu_scale_up_threshold = 0.75
        self.cpu_scale_down_threshold = 0.25
        self.memory_scale_up_threshold = 0.80
        self.memory_scale_down_threshold = 0.30
        self.response_time_threshold = 2.0  # seconds
        self.throughput_decline_threshold = 0.8  # 80% of expected
        
        # Scaling constraints
        self.min_workers = 1
        self.max_workers = 100
        self.min_confidence_for_action = 0.6
        
        # Decision history
        self.decision_history: List[ScalingDecision] = []
        self.cooldown_period = 60  # seconds between scaling actions
        
        # Logger
        self.logger = logging.getLogger(__name__)
    
    def make_scaling_decision(
        self,
        current_metrics: ResourceMetrics,
        prediction: WorkloadPrediction
    ) -> ScalingDecision:
        """Make scaling decision based on current state and predictions.
        
        Args:
            current_metrics: Current resource utilization
            prediction: Future workload prediction
            
        Returns:
            Scaling decision
        """
        # Check if we're in cooldown period
        if self._in_cooldown_period():
            return self._no_action_decision(current_metrics, "In cooldown period")
        
        # Check prediction confidence
        if prediction.confidence < self.min_confidence_for_action:
            return self._no_action_decision(current_metrics, "Low prediction confidence")
        
        # Analyze resource needs
        cpu_action = self._analyze_cpu_scaling(current_metrics, prediction)
        memory_action = self._analyze_memory_scaling(current_metrics, prediction)
        performance_action = self._analyze_performance_scaling(current_metrics, prediction)
        
        # Choose highest priority action
        actions = [cpu_action, memory_action, performance_action]
        actions = [a for a in actions if a.action != ScalingAction.NO_ACTION]
        
        if not actions:
            return self._no_action_decision(current_metrics, "No scaling needed")
        
        # Sort by expected improvement (descending)
        actions.sort(key=lambda x: x.expected_improvement, reverse=True)
        
        best_action = actions[0]
        
        # Record decision
        self.decision_history.append(best_action)
        
        return best_action
    
    def _analyze_cpu_scaling(
        self,
        current_metrics: ResourceMetrics,
        prediction: WorkloadPrediction
    ) -> ScalingDecision:
        """Analyze CPU scaling needs.
        
        Args:
            current_metrics: Current metrics
            prediction: Workload prediction
            
        Returns:
            CPU scaling decision
        """
        current_cpu = current_metrics.cpu_utilization
        predicted_cpu = prediction.predicted_cpu_demand
        
        # Scale up if current or predicted CPU is high
        if current_cpu > self.cpu_scale_up_threshold or predicted_cpu > self.cpu_scale_up_threshold:
            target_workers = min(
                self.max_workers,
                current_metrics.active_workers + max(1, int((predicted_cpu - 0.5) * 4))
            )
            
            expected_improvement = (predicted_cpu - 0.5) * 100  # Percentage points
            
            return ScalingDecision(
                timestamp=time.time(),
                action=ScalingAction.SCALE_UP_CPU,
                current_state={"cpu_utilization": current_cpu, "workers": current_metrics.active_workers},
                target_state={"workers": target_workers},
                expected_improvement=expected_improvement,
                confidence=prediction.confidence,
                reasoning=f"CPU utilization high: current={current_cpu:.2f}, predicted={predicted_cpu:.2f}"
            )
        
        # Scale down if both current and predicted CPU are low
        elif (current_cpu < self.cpu_scale_down_threshold and 
              predicted_cpu < self.cpu_scale_down_threshold and
              current_metrics.active_workers > self.min_workers):
            
            target_workers = max(
                self.min_workers,
                current_metrics.active_workers - max(1, int((0.5 - predicted_cpu) * 2))
            )
            
            expected_improvement = (0.5 - predicted_cpu) * 50  # Cost savings
            
            return ScalingDecision(
                timestamp=time.time(),
                action=ScalingAction.SCALE_DOWN_CPU,
                current_state={"cpu_utilization": current_cpu, "workers": current_metrics.active_workers},
                target_state={"workers": target_workers},
                expected_improvement=expected_improvement,
                confidence=prediction.confidence,
                reasoning=f"CPU utilization low: current={current_cpu:.2f}, predicted={predicted_cpu:.2f}"
            )
        
        return self._no_action_decision(current_metrics, "CPU scaling not needed")
    
    def _analyze_memory_scaling(
        self,
        current_metrics: ResourceMetrics,
        prediction: WorkloadPrediction
    ) -> ScalingDecision:
        """Analyze memory scaling needs.
        
        Args:
            current_metrics: Current metrics
            prediction: Workload prediction
            
        Returns:
            Memory scaling decision
        """
        current_memory = current_metrics.memory_utilization
        predicted_memory = prediction.predicted_memory_demand
        
        # Scale up memory if current or predicted is high
        if current_memory > self.memory_scale_up_threshold or predicted_memory > self.memory_scale_up_threshold:
            expected_improvement = (max(current_memory, predicted_memory) - 0.6) * 100
            
            return ScalingDecision(
                timestamp=time.time(),
                action=ScalingAction.SCALE_UP_MEMORY,
                current_state={"memory_utilization": current_memory},
                target_state={"memory_increase": "20%"},
                expected_improvement=expected_improvement,
                confidence=prediction.confidence,
                reasoning=f"Memory utilization high: current={current_memory:.2f}, predicted={predicted_memory:.2f}"
            )
        
        # Scale down memory if both are low
        elif (current_memory < self.memory_scale_down_threshold and 
              predicted_memory < self.memory_scale_down_threshold):
            
            expected_improvement = (0.5 - min(current_memory, predicted_memory)) * 50
            
            return ScalingDecision(
                timestamp=time.time(),
                action=ScalingAction.SCALE_DOWN_MEMORY,
                current_state={"memory_utilization": current_memory},
                target_state={"memory_decrease": "10%"},
                expected_improvement=expected_improvement,
                confidence=prediction.confidence,
                reasoning=f"Memory utilization low: current={current_memory:.2f}, predicted={predicted_memory:.2f}"
            )
        
        return self._no_action_decision(current_metrics, "Memory scaling not needed")
    
    def _analyze_performance_scaling(
        self,
        current_metrics: ResourceMetrics,
        prediction: WorkloadPrediction
    ) -> ScalingDecision:
        """Analyze performance-based scaling needs.
        
        Args:
            current_metrics: Current metrics
            prediction: Workload prediction
            
        Returns:
            Performance scaling decision
        """
        current_response_time = current_metrics.response_time
        predicted_response_time = prediction.predicted_response_time
        current_throughput = current_metrics.throughput
        predicted_throughput = prediction.predicted_throughput
        
        # Scale out workers if response time is or will be too high
        if (current_response_time > self.response_time_threshold or 
            predicted_response_time > self.response_time_threshold):
            
            target_workers = min(
                self.max_workers,
                current_metrics.active_workers + max(1, int(predicted_response_time - 1))
            )
            
            expected_improvement = (max(current_response_time, predicted_response_time) - 1.0) * 100
            
            return ScalingDecision(
                timestamp=time.time(),
                action=ScalingAction.SCALE_OUT_WORKERS,
                current_state={"workers": current_metrics.active_workers, "response_time": current_response_time},
                target_state={"workers": target_workers},
                expected_improvement=expected_improvement,
                confidence=prediction.confidence,
                reasoning=f"Response time high: current={current_response_time:.2f}s, predicted={predicted_response_time:.2f}s"
            )
        
        # Scale in workers if throughput decline is expected and we have extra capacity
        elif (predicted_throughput < current_throughput * self.throughput_decline_threshold and
              current_metrics.active_workers > self.min_workers and
              current_response_time < self.response_time_threshold * 0.5):
            
            target_workers = max(
                self.min_workers,
                current_metrics.active_workers - 1
            )
            
            expected_improvement = (1.0 - predicted_throughput / current_throughput) * 50
            
            return ScalingDecision(
                timestamp=time.time(),
                action=ScalingAction.SCALE_IN_WORKERS,
                current_state={"workers": current_metrics.active_workers, "throughput": current_throughput},
                target_state={"workers": target_workers},
                expected_improvement=expected_improvement,
                confidence=prediction.confidence,
                reasoning=f"Throughput decline expected: current={current_throughput:.1f}, predicted={predicted_throughput:.1f}"
            )
        
        return self._no_action_decision(current_metrics, "Performance scaling not needed")
    
    def _in_cooldown_period(self) -> bool:
        """Check if we're in cooldown period after recent scaling action.
        
        Returns:
            True if in cooldown period
        """
        if not self.decision_history:
            return False
        
        last_action = self.decision_history[-1]
        if last_action.action == ScalingAction.NO_ACTION:
            return False
        
        time_since_last = time.time() - last_action.timestamp
        return time_since_last < self.cooldown_period
    
    def _no_action_decision(self, current_metrics: ResourceMetrics, reason: str) -> ScalingDecision:
        """Create a no-action decision.
        
        Args:
            current_metrics: Current metrics
            reason: Reason for no action
            
        Returns:
            No-action scaling decision
        """
        return ScalingDecision(
            timestamp=time.time(),
            action=ScalingAction.NO_ACTION,
            current_state={
                "cpu_utilization": current_metrics.cpu_utilization,
                "memory_utilization": current_metrics.memory_utilization,
                "workers": current_metrics.active_workers
            },
            target_state={},
            expected_improvement=0.0,
            confidence=1.0,
            reasoning=reason
        )


class PredictiveScalingEngine:
    """Main engine for predictive scaling and real-time performance optimization.
    
    This engine combines workload prediction with intelligent scaling decisions
    to optimize system performance proactively.
    """
    
    def __init__(
        self,
        monitoring_interval: int = 30,  # seconds
        prediction_horizon: int = 300   # seconds
    ):
        """Initialize predictive scaling engine.
        
        Args:
            monitoring_interval: How often to check metrics and make decisions
            prediction_horizon: How far ahead to predict workload
        """
        self.monitoring_interval = monitoring_interval
        self.prediction_horizon = prediction_horizon
        
        # Core components
        self.predictor = WorkloadPredictor(prediction_horizon)
        self.decision_engine = ScalingDecisionEngine()
        
        # System state
        self.current_metrics: Optional[ResourceMetrics] = None
        self.scaling_active = False
        
        # Background processing
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Performance tracking
        self.scaling_effectiveness: List[Dict[str, Any]] = []
        self.prediction_performance: List[Dict[str, Any]] = []
        
        # Callbacks for scaling actions
        self.scaling_callbacks: Dict[ScalingAction, Callable] = {}
        
        # Logger
        self.logger = logging.getLogger(__name__)
    
    def start_predictive_scaling(self) -> None:
        """Start the predictive scaling engine."""
        self.scaling_active = True
        
        # Start background monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("Predictive scaling engine started")
    
    def stop_predictive_scaling(self) -> None:
        """Stop the predictive scaling engine."""
        self.scaling_active = False
        self.stop_event.set()
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)
        
        self.logger.info("Predictive scaling engine stopped")
    
    def update_metrics(self, metrics: ResourceMetrics) -> None:
        """Update current resource metrics.
        
        Args:
            metrics: Latest resource metrics
        """
        self.current_metrics = metrics
        self.predictor.add_observation(metrics)
    
    def register_scaling_callback(
        self, 
        action: ScalingAction, 
        callback: Callable[[Dict[str, Any]], bool]
    ) -> None:
        """Register callback for scaling actions.
        
        Args:
            action: Scaling action type
            callback: Function to execute the scaling action
                     Should return True if successful, False otherwise
        """
        self.scaling_callbacks[action] = callback
        self.logger.info(f"Registered callback for {action.value}")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring and scaling loop."""
        while self.scaling_active and not self.stop_event.is_set():
            try:
                # Wait for monitoring interval
                if self.stop_event.wait(self.monitoring_interval):
                    break
                
                # Skip if no current metrics
                if self.current_metrics is None:
                    continue
                
                # Get workload prediction
                prediction = self.predictor.predict_workload()
                
                # Make scaling decision
                decision = self.decision_engine.make_scaling_decision(
                    self.current_metrics, prediction
                )
                
                # Execute scaling action if needed
                if decision.action != ScalingAction.NO_ACTION:
                    success = self._execute_scaling_action(decision)
                    
                    # Track effectiveness
                    self._track_scaling_effectiveness(decision, success, prediction)
                
                # Track prediction performance
                self._track_prediction_performance(prediction)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
    
    def _execute_scaling_action(self, decision: ScalingDecision) -> bool:
        """Execute a scaling action.
        
        Args:
            decision: Scaling decision to execute
            
        Returns:
            True if action was successful
        """
        if decision.action not in self.scaling_callbacks:
            self.logger.warning(f"No callback registered for {decision.action.value}")
            return False
        
        try:
            callback = self.scaling_callbacks[decision.action]
            success = callback(decision.target_state)
            
            if success:
                self.logger.info(f"Successfully executed {decision.action.value}: {decision.reasoning}")
            else:
                self.logger.warning(f"Failed to execute {decision.action.value}: {decision.reasoning}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error executing {decision.action.value}: {e}")
            return False
    
    def _track_scaling_effectiveness(
        self,
        decision: ScalingDecision,
        success: bool,
        prediction: WorkloadPrediction
    ) -> None:
        """Track effectiveness of scaling decisions.
        
        Args:
            decision: Scaling decision that was made
            success: Whether the action was successful
            prediction: Workload prediction that informed the decision
        """
        effectiveness_record = {
            "timestamp": time.time(),
            "action": decision.action.value,
            "success": success,
            "expected_improvement": decision.expected_improvement,
            "confidence": decision.confidence,
            "prediction_confidence": prediction.confidence,
            "reasoning": decision.reasoning,
        }
        
        self.scaling_effectiveness.append(effectiveness_record)
        
        # Keep only recent records
        if len(self.scaling_effectiveness) > 100:
            self.scaling_effectiveness = self.scaling_effectiveness[-100:]
    
    def _track_prediction_performance(self, prediction: WorkloadPrediction) -> None:
        """Track prediction performance over time.
        
        Args:
            prediction: Workload prediction to track
        """
        if self.current_metrics is None:
            return
        
        # Compare prediction with current reality (simplified)
        performance_record = {
            "timestamp": time.time(),
            "prediction_confidence": prediction.confidence,
            "predicted_cpu": prediction.predicted_cpu_demand,
            "actual_cpu": self.current_metrics.cpu_utilization,
            "predicted_memory": prediction.predicted_memory_demand,
            "actual_memory": self.current_metrics.memory_utilization,
            "predicted_throughput": prediction.predicted_throughput,
            "actual_throughput": self.current_metrics.throughput,
        }
        
        self.prediction_performance.append(performance_record)
        
        # Keep only recent records
        if len(self.prediction_performance) > 200:
            self.prediction_performance = self.prediction_performance[-200:]
    
    def get_scaling_statistics(self) -> Dict[str, Any]:
        """Get comprehensive scaling statistics.
        
        Returns:
            Dictionary containing scaling performance metrics
        """
        if not self.scaling_effectiveness:
            scaling_success_rate = 0.0
            avg_expected_improvement = 0.0
        else:
            scaling_success_rate = np.mean([r["success"] for r in self.scaling_effectiveness])
            avg_expected_improvement = np.mean([r["expected_improvement"] for r in self.scaling_effectiveness])
        
        # Calculate prediction accuracy
        if not self.prediction_performance:
            prediction_accuracy = {"cpu": 0.0, "memory": 0.0, "throughput": 0.0}
        else:
            recent_predictions = self.prediction_performance[-50:]  # Last 50 predictions
            
            cpu_errors = [abs(p["predicted_cpu"] - p["actual_cpu"]) for p in recent_predictions]
            memory_errors = [abs(p["predicted_memory"] - p["actual_memory"]) for p in recent_predictions]
            throughput_errors = [abs(p["predicted_throughput"] - p["actual_throughput"]) for p in recent_predictions 
                               if p["actual_throughput"] > 0]
            
            prediction_accuracy = {
                "cpu": 1.0 - np.mean(cpu_errors) if cpu_errors else 0.0,
                "memory": 1.0 - np.mean(memory_errors) if memory_errors else 0.0,
                "throughput": 1.0 - (np.mean(throughput_errors) / 100.0) if throughput_errors else 0.0
            }
        
        return {
            "scaling_active": self.scaling_active,
            "total_scaling_actions": len(self.scaling_effectiveness),
            "scaling_success_rate": scaling_success_rate,
            "average_expected_improvement": avg_expected_improvement,
            "prediction_accuracy": prediction_accuracy,
            "recent_actions": [r["action"] for r in self.scaling_effectiveness[-10:]],
            "predictor_statistics": {
                "historical_data_points": len(self.predictor.historical_data),
                "models_trained": self.predictor._models_trained(),
                "last_training": self.predictor.model_last_trained,
            },
            "decision_engine_statistics": {
                "decisions_made": len(self.decision_engine.decision_history),
                "in_cooldown": self.decision_engine._in_cooldown_period(),
                "cooldown_period": self.decision_engine.cooldown_period,
            }
        }
    
    def get_current_prediction(self) -> Optional[WorkloadPrediction]:
        """Get current workload prediction.
        
        Returns:
            Current workload prediction or None if not available
        """
        try:
            return self.predictor.predict_workload()
        except Exception as e:
            self.logger.warning(f"Failed to get current prediction: {e}")
            return None
    
    def export_scaling_history(self) -> Dict[str, Any]:
        """Export scaling history for analysis.
        
        Returns:
            Dictionary containing complete scaling history
        """
        return {
            "scaling_effectiveness": self.scaling_effectiveness,
            "prediction_performance": self.prediction_performance,
            "decision_history": [
                {
                    "timestamp": d.timestamp,
                    "action": d.action.value,
                    "expected_improvement": d.expected_improvement,
                    "confidence": d.confidence,
                    "reasoning": d.reasoning,
                }
                for d in self.decision_engine.decision_history
            ],
            "statistics": self.get_scaling_statistics(),
        }