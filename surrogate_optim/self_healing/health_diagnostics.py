"""Advanced health diagnostics and anomaly detection for pipeline monitoring."""

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, Dict, List, Optional

from loguru import logger
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest

from .pipeline_monitor import HealthMetric, HealthStatus, PipelineHealth


class AnomalyType(Enum):
    """Types of anomalies that can be detected."""
    STATISTICAL_OUTLIER = "statistical_outlier"
    TREND_ANOMALY = "trend_anomaly"
    SEASONAL_ANOMALY = "seasonal_anomaly"
    THRESHOLD_BREACH = "threshold_breach"
    CORRELATION_BREAK = "correlation_break"


@dataclass
class Anomaly:
    """Detected anomaly in pipeline metrics."""
    type: AnomalyType
    metric_name: str
    severity: float  # 0-1 scale
    description: str
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DiagnosticReport:
    """Comprehensive diagnostic report."""
    timestamp: float
    overall_health_score: float
    anomalies: List[Anomaly]
    trend_analysis: Dict[str, Dict[str, float]]
    correlation_matrix: Optional[np.ndarray] = None
    recommendations: List[str] = field(default_factory=list)


class HealthDiagnostics:
    """Advanced diagnostics system for pipeline health analysis."""

    def __init__(
        self,
        history_window: int = 100,
        anomaly_threshold: float = 0.7,
        enable_ml_detection: bool = True
    ):
        self.history_window = history_window
        self.anomaly_threshold = anomaly_threshold
        self.enable_ml_detection = enable_ml_detection

        self._health_buffer: List[PipelineHealth] = []
        self._anomaly_detectors: Dict[str, IsolationForest] = {}
        self._baseline_stats: Dict[str, Dict[str, float]] = {}

        # Initialize ML models if enabled
        if self.enable_ml_detection:
            self._initialize_ml_detectors()

    def _initialize_ml_detectors(self) -> None:
        """Initialize machine learning anomaly detectors."""
        # Use Isolation Forest for unsupervised anomaly detection
        default_detector = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )

        # Create detectors for common metrics
        metric_names = [
            "memory_usage", "cpu_usage", "error_rate",
            "latency_p95", "optimization_convergence", "model_accuracy"
        ]

        for metric_name in metric_names:
            self._anomaly_detectors[metric_name] = IsolationForest(
                contamination=0.05,  # 5% contamination expected
                random_state=42
            )

    def update_health_history(self, health: PipelineHealth) -> None:
        """Update health history buffer."""
        self._health_buffer.append(health)

        # Maintain rolling window
        if len(self._health_buffer) > self.history_window:
            self._health_buffer = self._health_buffer[-self.history_window:]

        # Update baseline statistics
        if len(self._health_buffer) >= 10:
            self._update_baseline_stats()

        # Train ML detectors if enough data
        if self.enable_ml_detection and len(self._health_buffer) >= 20:
            self._update_ml_detectors()

    def _update_baseline_stats(self) -> None:
        """Update baseline statistics from health history."""
        if len(self._health_buffer) < 10:
            return

        # Aggregate metrics across health snapshots
        metric_values = {}
        for health in self._health_buffer:
            for metric_name, metric in health.metrics.items():
                if metric_name not in metric_values:
                    metric_values[metric_name] = []
                metric_values[metric_name].append(metric.value)

        # Calculate baseline statistics
        self._baseline_stats = {}
        for metric_name, values in metric_values.items():
            if len(values) >= 5:
                self._baseline_stats[metric_name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "median": np.median(values),
                    "p25": np.percentile(values, 25),
                    "p75": np.percentile(values, 75),
                    "min": np.min(values),
                    "max": np.max(values)
                }

    def _update_ml_detectors(self) -> None:
        """Update machine learning anomaly detectors."""
        if not self.enable_ml_detection or len(self._health_buffer) < 20:
            return

        # Prepare data for each metric
        for metric_name, detector in self._anomaly_detectors.items():
            values = []
            for health in self._health_buffer:
                if metric_name in health.metrics:
                    values.append(health.metrics[metric_name].value)

            if len(values) >= 20:
                # Reshape for sklearn (needs 2D array)
                X = np.array(values).reshape(-1, 1)

                try:
                    detector.fit(X)
                except Exception as e:
                    logger.warning(f"Failed to train anomaly detector for {metric_name}: {e}")

    def analyze_current_health(self, health: PipelineHealth) -> DiagnosticReport:
        """Perform comprehensive analysis of current health."""
        start_time = time.time()

        # Update history first
        self.update_health_history(health)

        # Detect anomalies
        anomalies = self._detect_anomalies(health)

        # Perform trend analysis
        trend_analysis = self._analyze_trends()

        # Calculate correlation matrix
        correlation_matrix = self._calculate_correlations()

        # Generate recommendations
        recommendations = self._generate_recommendations(health, anomalies)

        # Calculate overall health score
        health_score = self._calculate_health_score(health, anomalies)

        duration = time.time() - start_time
        logger.debug(f"Health diagnostics completed in {duration:.3f}s")

        return DiagnosticReport(
            timestamp=time.time(),
            overall_health_score=health_score,
            anomalies=anomalies,
            trend_analysis=trend_analysis,
            correlation_matrix=correlation_matrix,
            recommendations=recommendations
        )

    def _detect_anomalies(self, health: PipelineHealth) -> List[Anomaly]:
        """Detect anomalies in current health metrics."""
        anomalies = []

        for metric_name, metric in health.metrics.items():
            # Statistical outlier detection
            if metric_name in self._baseline_stats:
                stats_anomalies = self._detect_statistical_anomalies(metric_name, metric)
                anomalies.extend(stats_anomalies)

            # ML-based anomaly detection
            if self.enable_ml_detection and metric_name in self._anomaly_detectors:
                ml_anomalies = self._detect_ml_anomalies(metric_name, metric)
                anomalies.extend(ml_anomalies)

            # Threshold breach detection
            threshold_anomalies = self._detect_threshold_anomalies(metric_name, metric)
            anomalies.extend(threshold_anomalies)

        # Trend anomaly detection
        trend_anomalies = self._detect_trend_anomalies()
        anomalies.extend(trend_anomalies)

        return anomalies

    def _detect_statistical_anomalies(self, metric_name: str, metric: HealthMetric) -> List[Anomaly]:
        """Detect statistical outliers using baseline statistics."""
        anomalies = []

        if metric_name not in self._baseline_stats:
            return anomalies

        stats = self._baseline_stats[metric_name]
        value = metric.value

        # Z-score based detection
        if stats["std"] > 0:
            z_score = abs(value - stats["mean"]) / stats["std"]
            if z_score > 3.0:  # 3-sigma rule
                severity = min(1.0, z_score / 5.0)  # Cap at 1.0
                anomalies.append(Anomaly(
                    type=AnomalyType.STATISTICAL_OUTLIER,
                    metric_name=metric_name,
                    severity=severity,
                    description=f"Z-score: {z_score:.2f} (>3σ threshold)",
                    timestamp=time.time(),
                    context={"z_score": z_score, "baseline_mean": stats["mean"], "baseline_std": stats["std"]}
                ))

        # IQR-based detection
        iqr = stats["p75"] - stats["p25"]
        if iqr > 0:
            lower_bound = stats["p25"] - 1.5 * iqr
            upper_bound = stats["p75"] + 1.5 * iqr

            if value < lower_bound or value > upper_bound:
                severity = 0.6
                anomalies.append(Anomaly(
                    type=AnomalyType.STATISTICAL_OUTLIER,
                    metric_name=metric_name,
                    severity=severity,
                    description=f"IQR outlier: {value:.3f} outside [{lower_bound:.3f}, {upper_bound:.3f}]",
                    timestamp=time.time(),
                    context={"iqr_bounds": [lower_bound, upper_bound]}
                ))

        return anomalies

    def _detect_ml_anomalies(self, metric_name: str, metric: HealthMetric) -> List[Anomaly]:
        """Detect anomalies using machine learning models."""
        anomalies = []

        if metric_name not in self._anomaly_detectors:
            return anomalies

        detector = self._anomaly_detectors[metric_name]

        try:
            # Check if detector is trained
            if not hasattr(detector, "decision_function"):
                return anomalies

            X = np.array([[metric.value]])
            anomaly_score = detector.decision_function(X)[0]
            is_anomaly = detector.predict(X)[0] == -1

            if is_anomaly:
                # Convert anomaly score to severity (0-1)
                severity = min(1.0, abs(anomaly_score) / 0.5)

                anomalies.append(Anomaly(
                    type=AnomalyType.STATISTICAL_OUTLIER,
                    metric_name=metric_name,
                    severity=severity,
                    description=f"ML anomaly detected (score: {anomaly_score:.3f})",
                    timestamp=time.time(),
                    context={"anomaly_score": anomaly_score}
                ))

        except Exception as e:
            logger.warning(f"ML anomaly detection failed for {metric_name}: {e}")

        return anomalies

    def _detect_threshold_anomalies(self, metric_name: str, metric: HealthMetric) -> List[Anomaly]:
        """Detect threshold breaches."""
        anomalies = []

        if metric.status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
            severity = 0.7 if metric.status == HealthStatus.WARNING else 0.9

            anomalies.append(Anomaly(
                type=AnomalyType.THRESHOLD_BREACH,
                metric_name=metric_name,
                severity=severity,
                description=f"Threshold breach: {metric.value:.3f} > {metric.threshold_warning:.3f}",
                timestamp=time.time(),
                context={"threshold_type": metric.status.value}
            ))

        return anomalies

    def _detect_trend_anomalies(self) -> List[Anomaly]:
        """Detect anomalous trends in metrics."""
        anomalies = []

        if len(self._health_buffer) < 10:
            return anomalies

        # Analyze trends for each metric
        for metric_name in set().union(*[h.metrics.keys() for h in self._health_buffer]):
            values = []
            timestamps = []

            for health in self._health_buffer[-10:]:  # Last 10 points
                if metric_name in health.metrics:
                    values.append(health.metrics[metric_name].value)
                    timestamps.append(health.timestamp)

            if len(values) >= 5:
                trend_anomaly = self._analyze_single_trend(metric_name, values, timestamps)
                if trend_anomaly:
                    anomalies.append(trend_anomaly)

        return anomalies

    def _analyze_single_trend(
        self,
        metric_name: str,
        values: List[float],
        timestamps: List[float]
    ) -> Optional[Anomaly]:
        """Analyze trend for a single metric."""
        if len(values) < 5:
            return None

        # Calculate trend using linear regression
        x = np.array(timestamps)
        y = np.array(values)

        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

            # Detect significant trends
            if abs(r_value) > 0.7 and p_value < 0.05:
                # Determine if trend is concerning
                if metric_name in ["error_rate", "latency_p95", "memory_usage", "cpu_usage"]:
                    # For these metrics, increasing trend is bad
                    if slope > 0:
                        severity = min(1.0, abs(r_value))
                        return Anomaly(
                            type=AnomalyType.TREND_ANOMALY,
                            metric_name=metric_name,
                            severity=severity,
                            description=f"Increasing trend detected (r={r_value:.3f})",
                            timestamp=time.time(),
                            context={"slope": slope, "r_value": r_value, "p_value": p_value}
                        )

        except Exception as e:
            logger.warning(f"Trend analysis failed for {metric_name}: {e}")

        return None

    def _analyze_trends(self) -> Dict[str, Dict[str, float]]:
        """Analyze trends across all metrics."""
        trends = {}

        if len(self._health_buffer) < 5:
            return trends

        # Analyze each metric
        for metric_name in set().union(*[h.metrics.keys() for h in self._health_buffer]):
            values = []
            for health in self._health_buffer:
                if metric_name in health.metrics:
                    values.append(health.metrics[metric_name].value)

            if len(values) >= 5:
                trends[metric_name] = self._calculate_trend_stats(values)

        return trends

    def _calculate_trend_stats(self, values: List[float]) -> Dict[str, float]:
        """Calculate trend statistics for a series of values."""
        x = np.arange(len(values))
        y = np.array(values)

        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

            return {
                "slope": slope,
                "correlation": r_value,
                "p_value": p_value,
                "trend_strength": abs(r_value),
                "recent_change": (values[-1] - values[0]) / values[0] if values[0] != 0 else 0.0
            }
        except Exception:
            return {
                "slope": 0.0,
                "correlation": 0.0,
                "p_value": 1.0,
                "trend_strength": 0.0,
                "recent_change": 0.0
            }

    def _calculate_correlations(self) -> Optional[np.ndarray]:
        """Calculate correlation matrix between metrics."""
        if len(self._health_buffer) < 10:
            return None

        # Collect metric values
        metric_data = {}
        for health in self._health_buffer:
            for metric_name, metric in health.metrics.items():
                if metric_name not in metric_data:
                    metric_data[metric_name] = []
                metric_data[metric_name].append(metric.value)

        # Ensure all metrics have same length
        min_length = min(len(values) for values in metric_data.values())
        for metric_name in metric_data:
            metric_data[metric_name] = metric_data[metric_name][-min_length:]

        if min_length < 5:
            return None

        try:
            # Create data matrix
            metric_names = list(metric_data.keys())
            data_matrix = np.array([metric_data[name] for name in metric_names])

            # Calculate correlation matrix
            correlation_matrix = np.corrcoef(data_matrix)
            return correlation_matrix

        except Exception as e:
            logger.warning(f"Correlation calculation failed: {e}")
            return None

    def _generate_recommendations(
        self,
        health: PipelineHealth,
        anomalies: List[Anomaly]
    ) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []

        # Recommendations based on anomalies
        for anomaly in anomalies:
            if anomaly.severity > 0.8:
                if anomaly.type == AnomalyType.THRESHOLD_BREACH:
                    recommendations.append(
                        f"URGENT: {anomaly.metric_name} breach - consider immediate intervention"
                    )
                elif anomaly.type == AnomalyType.TREND_ANOMALY:
                    recommendations.append(
                        f"Monitor {anomaly.metric_name} trend - preventive action may be needed"
                    )

        # Recommendations based on overall health
        if health.performance_score < 0.5:
            recommendations.append("Overall performance degraded - comprehensive health check recommended")

        # Resource-specific recommendations
        memory_metric = health.metrics.get("memory_usage")
        if memory_metric and memory_metric.value > 0.8:
            recommendations.append("High memory usage - consider garbage collection or memory optimization")

        cpu_metric = health.metrics.get("cpu_usage")
        if cpu_metric and cpu_metric.value > 0.8:
            recommendations.append("High CPU usage - consider workload distribution or resource scaling")

        return recommendations

    def _calculate_health_score(
        self,
        health: PipelineHealth,
        anomalies: List[Anomaly]
    ) -> float:
        """Calculate comprehensive health score (0-1)."""
        base_score = health.performance_score

        # Penalize for anomalies
        anomaly_penalty = 0.0
        for anomaly in anomalies:
            anomaly_penalty += anomaly.severity * 0.1

        # Cap penalty at 0.5
        anomaly_penalty = min(0.5, anomaly_penalty)

        final_score = max(0.0, base_score - anomaly_penalty)
        return final_score
