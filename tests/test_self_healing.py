"""Comprehensive tests for self-healing pipeline functionality."""

import pytest
import time
import threading
import numpy as np
import jax.numpy as jnp
from unittest.mock import Mock, patch, MagicMock

from surrogate_optim.self_healing import (
    PipelineMonitor,
    RecoveryEngine,
    HealthDiagnostics,
    SelfHealingOptimizer,
    RobustMonitor,
    MonitoringConfig,
    AdvancedErrorHandler,
    ScalableOptimizer,
    PerformanceOptimizer
)
from surrogate_optim.self_healing.pipeline_monitor import HealthStatus, HealthMetric, PipelineHealth
from surrogate_optim.self_healing.recovery_engine import RecoveryAction
from surrogate_optim.self_healing.robust_monitoring import MonitoringLevel
from surrogate_optim.self_healing.error_handling import RetryConfig, ErrorCategory
from surrogate_optim.self_healing.scalable_architecture import ScalingStrategy, WorkloadType
from surrogate_optim.self_healing.performance_optimization import OptimizationLevel


class TestPipelineMonitor:
    """Test pipeline monitoring functionality."""
    
    def test_monitor_initialization(self):
        """Test pipeline monitor initialization."""
        monitor = PipelineMonitor(check_interval=1.0, enable_auto_recovery=True)
        
        assert monitor.check_interval == 1.0
        assert monitor.enable_auto_recovery is True
        assert not monitor._monitoring
        assert len(monitor._health_history) == 0
        
    def test_metric_collector_registration(self):
        """Test custom metric collector registration."""
        monitor = PipelineMonitor()
        
        def custom_metric():
            return 0.5
            
        monitor.register_metric_collector("test_metric", custom_metric)
        
        assert "test_metric" in monitor._metric_collectors
        
    def test_health_metric_creation(self):
        """Test health metric creation and status determination."""
        metric = HealthMetric(
            name="test_metric",
            value=0.9,
            threshold_warning=0.8,
            threshold_critical=0.95
        )
        
        assert metric.status == HealthStatus.WARNING
        
        critical_metric = HealthMetric(
            name="critical_metric",
            value=0.97,
            threshold_warning=0.8,
            threshold_critical=0.95
        )
        
        assert critical_metric.status == HealthStatus.CRITICAL
        
    def test_monitoring_lifecycle(self):
        """Test monitoring start/stop lifecycle."""
        monitor = PipelineMonitor(check_interval=0.1)
        
        # Start monitoring
        monitor.start_monitoring()
        assert monitor._monitoring is True
        assert monitor._monitor_thread is not None
        
        # Let it run briefly
        time.sleep(0.2)
        
        # Stop monitoring
        monitor.stop_monitoring()
        assert monitor._monitoring is False
        
    @patch('psutil.virtual_memory')
    @patch('psutil.cpu_percent')
    def test_health_metrics_collection(self, mock_cpu, mock_memory):
        """Test health metrics collection."""
        # Mock system metrics
        mock_memory.return_value = Mock(percent=75.0)
        mock_cpu.return_value = 45.0
        
        monitor = PipelineMonitor(check_interval=0.1)
        monitor.start_monitoring()
        
        time.sleep(0.2)  # Let it collect some data
        
        current_health = monitor.get_current_health()
        assert current_health is not None
        assert "memory_usage" in current_health.metrics
        assert "cpu_usage" in current_health.metrics
        
        monitor.stop_monitoring()


class TestRecoveryEngine:
    """Test recovery engine functionality."""
    
    def test_recovery_engine_initialization(self):
        """Test recovery engine initialization."""
        engine = RecoveryEngine()
        
        assert len(engine._recovery_strategies) > 0
        assert RecoveryAction.MEMORY_CLEANUP in engine._recovery_strategies
        assert len(engine._recovery_history) == 0
        
    def test_memory_cleanup_recovery(self):
        """Test memory cleanup recovery action."""
        engine = RecoveryEngine()
        
        # Create mock health with high memory usage
        health = PipelineHealth(
            timestamp=time.time(),
            overall_status=HealthStatus.WARNING,
            metrics={
                "memory_usage": HealthMetric(
                    name="memory_usage",
                    value=0.9,
                    threshold_warning=0.8,
                    threshold_critical=0.95
                )
            }
        )
        
        result = engine._memory_cleanup(health)
        
        assert result.action == RecoveryAction.MEMORY_CLEANUP
        assert result.duration > 0
        
    def test_recovery_strategy_registration(self):
        """Test custom recovery strategy registration."""
        engine = RecoveryEngine()
        
        def custom_recovery(health):
            return Mock(action=RecoveryAction.PARAMETER_ADJUSTMENT, success=True)
            
        engine.register_recovery_strategy(RecoveryAction.PARAMETER_ADJUSTMENT, custom_recovery)
        
        assert RecoveryAction.PARAMETER_ADJUSTMENT in engine._recovery_strategies
        
    def test_checkpoint_functionality(self):
        """Test checkpoint save/restore functionality."""
        engine = RecoveryEngine()
        
        test_state = {"param1": 1.0, "param2": [1, 2, 3]}
        engine.save_checkpoint(test_state)
        
        assert engine._last_checkpoint is not None
        assert engine._last_checkpoint["state"] == test_state


class TestHealthDiagnostics:
    """Test health diagnostics functionality."""
    
    def test_diagnostics_initialization(self):
        """Test health diagnostics initialization."""
        diagnostics = HealthDiagnostics(
            history_window=50,
            anomaly_threshold=0.8,
            enable_ml_detection=True
        )
        
        assert diagnostics.history_window == 50
        assert diagnostics.anomaly_threshold == 0.8
        assert diagnostics.enable_ml_detection is True
        
    def test_anomaly_detection(self):
        """Test basic anomaly detection."""
        diagnostics = HealthDiagnostics(enable_ml_detection=False)
        
        # Create health with anomalous metrics
        health = PipelineHealth(
            timestamp=time.time(),
            overall_status=HealthStatus.CRITICAL,
            metrics={
                "error_rate": HealthMetric(
                    name="error_rate",
                    value=0.15,
                    threshold_warning=0.05,
                    threshold_critical=0.1
                )
            }
        )
        
        report = diagnostics.analyze_current_health(health)
        
        assert report.overall_health_score >= 0.0
        assert len(report.anomalies) > 0
        assert len(report.recommendations) > 0
        
    def test_trend_analysis(self):
        """Test trend analysis functionality."""
        diagnostics = HealthDiagnostics()
        
        # Add several health snapshots with increasing error rate
        for i in range(10):
            health = PipelineHealth(
                timestamp=time.time() + i,
                overall_status=HealthStatus.HEALTHY,
                metrics={
                    "error_rate": HealthMetric(
                        name="error_rate",
                        value=i * 0.01,  # Increasing trend
                        threshold_warning=0.05,
                        threshold_critical=0.1
                    )
                }
            )
            diagnostics.update_health_history(health)
            
        latest_health = PipelineHealth(
            timestamp=time.time() + 10,
            overall_status=HealthStatus.WARNING,
            metrics={
                "error_rate": HealthMetric(
                    name="error_rate",
                    value=0.12,
                    threshold_warning=0.05,
                    threshold_critical=0.1
                )
            }
        )
        
        report = diagnostics.analyze_current_health(latest_health)
        
        # Should detect trend anomaly
        trend_anomalies = [a for a in report.anomalies if "trend" in a.description.lower()]
        assert len(trend_anomalies) > 0 or report.trend_analysis  # Either direct detection or trend analysis


class TestSelfHealingOptimizer:
    """Test self-healing optimizer functionality."""
    
    def test_optimizer_initialization(self):
        """Test self-healing optimizer initialization."""
        optimizer = SelfHealingOptimizer(
            surrogate_type="neural_network",
            monitoring_interval=1.0,
            auto_recovery=True
        )
        
        assert optimizer.monitoring_interval == 1.0
        assert optimizer.auto_recovery is True
        assert optimizer.pipeline_monitor is not None
        assert optimizer.recovery_engine is not None
        
    def test_monitoring_lifecycle(self):
        """Test monitoring start/stop in optimizer."""
        optimizer = SelfHealingOptimizer(monitoring_interval=0.1)
        
        optimizer.start_monitoring()
        assert optimizer.pipeline_monitor._monitoring is True
        
        time.sleep(0.2)
        
        optimizer.stop_monitoring()
        assert optimizer.pipeline_monitor._monitoring is False
        
    def test_optimization_with_monitoring(self):
        """Test optimization with health monitoring."""
        optimizer = SelfHealingOptimizer(monitoring_interval=0.1, auto_recovery=False)
        
        # Simple objective function
        def objective(x):
            return np.sum(x**2)
            
        initial_point = jnp.array([1.0, 1.0])
        bounds = [(-2, 2), (-2, 2)]
        
        # Mock the base optimization method
        with patch.object(optimizer.__class__.__bases__[0], 'optimize') as mock_optimize:
            mock_optimize.return_value = {
                "x": np.array([0.1, 0.1]),
                "fun": 0.02,
                "success": True,
                "nfev": 50
            }
            
            result = optimizer.optimize(
                objective_function=objective,
                initial_point=initial_point,
                bounds=bounds,
                max_iterations=10
            )
            
            assert "optimization_health" in result
            assert result["optimization_health"] is not None


class TestRobustMonitoring:
    """Test robust monitoring functionality."""
    
    def test_robust_monitor_initialization(self):
        """Test robust monitor initialization."""
        config = MonitoringConfig(
            level=MonitoringLevel.STANDARD,
            check_interval=5.0,
            enable_predictive_monitoring=True
        )
        
        monitor = RobustMonitor(config)
        
        assert monitor.config.level == MonitoringLevel.STANDARD
        assert monitor.config.check_interval == 5.0
        assert len(monitor._circuit_breakers) > 0
        
    def test_monitored_operation_context(self):
        """Test monitored operation context manager."""
        monitor = RobustMonitor()
        
        # Test successful operation
        with monitor.monitored_operation("test_operation") as execute:
            result = execute(lambda: 42)
            assert result == 42
            
        # Test operation with error
        with pytest.raises(ValueError):
            with monitor.monitored_operation("failing_operation") as execute:
                execute(lambda: (_ for _ in ()).throw(ValueError("Test error")))
                
    def test_error_event_creation(self):
        """Test error event creation and classification."""
        monitor = RobustMonitor()
        
        error = ValueError("Test value error")
        event = monitor._create_error_event("test_operation", error)
        
        assert event.error_type == "ValueError"
        assert event.message == "Test value error"
        assert event.context["operation"] == "test_operation"
        assert event.severity is not None


class TestAdvancedErrorHandler:
    """Test advanced error handling functionality."""
    
    def test_error_handler_initialization(self):
        """Test error handler initialization."""
        handler = AdvancedErrorHandler()
        
        assert handler.classifier is not None
        assert len(handler._retry_configs) > 0
        
    def test_error_classification(self):
        """Test error classification."""
        handler = AdvancedErrorHandler()
        
        # Test different error types
        value_error = ValueError("Invalid value")
        assert handler.classifier.classify_error(value_error) == ErrorCategory.PERSISTENT
        
        connection_error = ConnectionError("Connection failed")
        assert handler.classifier.classify_error(connection_error) == ErrorCategory.TRANSIENT
        
        memory_error = MemoryError("Out of memory")
        assert handler.classifier.classify_error(memory_error) == ErrorCategory.FATAL
        
    def test_retry_decorator(self):
        """Test retry decorator functionality."""
        handler = AdvancedErrorHandler()
        
        # Counter to track function calls
        call_count = {"count": 0}
        
        @handler.with_retry("test_operation", RetryConfig(max_attempts=3, initial_delay=0.01))
        def failing_function():
            call_count["count"] += 1
            if call_count["count"] < 3:
                raise RuntimeError("Temporary failure")
            return "success"
            
        result = failing_function()
        
        assert result == "success"
        assert call_count["count"] == 3
        
    def test_circuit_breaker_decorator(self):
        """Test circuit breaker decorator."""
        handler = AdvancedErrorHandler()
        
        failure_count = {"count": 0}
        
        @handler.circuit_breaker("test_circuit", failure_threshold=2, recovery_timeout=0.1)
        def unreliable_function():
            failure_count["count"] += 1
            if failure_count["count"] <= 3:
                raise RuntimeError("Service unavailable")
            return "success"
            
        # First two failures should pass through
        with pytest.raises(RuntimeError):
            unreliable_function()
            
        with pytest.raises(RuntimeError):
            unreliable_function()
            
        # Third failure should trigger circuit breaker
        with pytest.raises(RuntimeError):
            unreliable_function()


class TestScalableArchitecture:
    """Test scalable architecture functionality."""
    
    def test_scalable_optimizer_initialization(self):
        """Test scalable optimizer initialization."""
        from surrogate_optim.self_healing.scalable_architecture import ScalingConfig
        
        config = ScalingConfig(
            strategy=ScalingStrategy.LOCAL_THREADING,
            max_workers=4
        )
        
        optimizer = ScalableOptimizer(config, enable_auto_scaling=False)
        
        assert optimizer.scaling_config.strategy == ScalingStrategy.LOCAL_THREADING
        assert optimizer.scaling_config.max_workers == 4
        assert optimizer.task_manager is not None
        
    def test_batch_optimization(self):
        """Test batch optimization functionality."""
        optimizer = ScalableOptimizer(enable_auto_scaling=False)
        
        # Create simple objective functions
        def objective1(x):
            return np.sum(x**2)
            
        def objective2(x):
            return np.sum((x - 1)**2)
            
        functions = [objective1, objective2]
        initial_points = [np.array([0.5, 0.5]), np.array([1.5, 1.5])]
        bounds = [[(-2, 2), (-2, 2)], [(-2, 2), (-2, 2)]]
        
        # Mock the task manager to avoid actual optimization
        with patch.object(optimizer.task_manager, 'submit_batch') as mock_submit, \
             patch.object(optimizer.task_manager, 'gather_results') as mock_gather:
            
            mock_submit.return_value = [Mock(), Mock()]
            mock_gather.return_value = [
                {"partition_id": "optimization_0", "status": "success", "result": {"optimization_results": [{"x": [0.0, 0.0], "fun": 0.0}]}},
                {"partition_id": "optimization_1", "status": "success", "result": {"optimization_results": [{"x": [1.0, 1.0], "fun": 0.0}]}}
            ]
            
            results = optimizer.optimize_batch(functions, initial_points, bounds)
            
            assert len(results) == 2
            assert all(r["status"] == "success" for r in results)
            
        # Cleanup
        optimizer.shutdown()


class TestPerformanceOptimization:
    """Test performance optimization functionality."""
    
    def test_performance_optimizer_initialization(self):
        """Test performance optimizer initialization."""
        from surrogate_optim.self_healing.performance_optimization import PerformanceConfig
        
        config = PerformanceConfig(
            optimization_level=OptimizationLevel.AGGRESSIVE,
            enable_jit=True,
            enable_gpu=False  # Disable for testing
        )
        
        optimizer = PerformanceOptimizer(config)
        
        assert optimizer.config.optimization_level == OptimizationLevel.AGGRESSIVE
        assert optimizer.config.enable_jit is True
        assert optimizer.cache is not None
        assert optimizer.jax_optimizer is not None
        
    def test_function_optimization(self):
        """Test function optimization."""
        optimizer = PerformanceOptimizer()
        
        def simple_function(x):
            return x**2
            
        optimized_func = optimizer.optimize_function(simple_function, "simple_function")
        
        # Test that optimized function works
        result = optimized_func(2.0)
        assert result == 4.0
        
        # Check that performance metrics were recorded
        assert "simple_function" in optimizer._optimization_metrics
        
    def test_caching_functionality(self):
        """Test caching functionality."""
        optimizer = PerformanceOptimizer()
        
        # Test cache operations
        optimizer.cache.put("test_key", "test_value")
        assert optimizer.cache.get("test_key") == "test_value"
        assert optimizer.cache.get("nonexistent_key") is None
        
        # Test cache statistics
        stats = optimizer.cache.get_stats()
        assert "hit_ratio" in stats
        assert "entries" in stats
        
    def test_performance_context_manager(self):
        """Test optimized performance context manager."""
        optimizer = PerformanceOptimizer()
        
        with optimizer.optimized_context("test_context"):
            # Simple computation
            result = np.sum(np.array([1, 2, 3, 4, 5]))
            assert result == 15


class TestIntegration:
    """Integration tests for self-healing system."""
    
    def test_full_self_healing_workflow(self):
        """Test complete self-healing workflow."""
        # Initialize self-healing optimizer
        optimizer = SelfHealingOptimizer(
            monitoring_interval=0.1,
            auto_recovery=True,
            health_diagnostics=True
        )
        
        # Start monitoring
        optimizer.start_monitoring()
        
        try:
            # Let monitoring run briefly
            time.sleep(0.2)
            
            # Check health
            health = optimizer.get_pipeline_health()
            assert health is not None
            
            # Check if system is healthy
            is_healthy = optimizer.is_healthy()
            assert isinstance(is_healthy, bool)
            
            # Get diagnostic report
            diagnostic_report = optimizer.get_diagnostic_report()
            # May be None if insufficient data
            
        finally:
            optimizer.stop_monitoring()
            
    def test_error_handling_integration(self):
        """Test error handling integration with monitoring."""
        from surrogate_optim.self_healing.error_handling import with_retry
        
        call_count = {"count": 0}
        
        @with_retry("integration_test")
        def potentially_failing_function():
            call_count["count"] += 1
            if call_count["count"] < 2:
                raise RuntimeError("Temporary failure")
            return "success"
            
        result = potentially_failing_function()
        assert result == "success"
        assert call_count["count"] == 2
        
    def test_performance_monitoring_integration(self):
        """Test performance monitoring integration."""
        from surrogate_optim.self_healing.performance_optimization import optimize_performance
        
        @optimize_performance("integration_function")
        def computation_function(x):
            return np.sum(x**2)
            
        result = computation_function(np.array([1, 2, 3]))
        assert result == 14  # 1 + 4 + 9
        
        # Check that performance was tracked
        from surrogate_optim.self_healing.performance_optimization import global_performance_optimizer
        assert "integration_function" in global_performance_optimizer._optimization_metrics


# Performance benchmarks
class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    def test_monitoring_overhead(self):
        """Test monitoring system overhead."""
        # Test without monitoring
        def simple_computation():
            return np.sum(np.random.randn(1000))
            
        # Measure baseline performance
        start_time = time.time()
        for _ in range(100):
            simple_computation()
        baseline_time = time.time() - start_time
        
        # Test with monitoring
        from surrogate_optim.self_healing.performance_optimization import optimize_performance
        
        @optimize_performance("benchmark_function")
        def monitored_computation():
            return np.sum(np.random.randn(1000))
            
        start_time = time.time()
        for _ in range(100):
            monitored_computation()
        monitored_time = time.time() - start_time
        
        # Overhead should be reasonable (less than 2x)
        overhead_ratio = monitored_time / baseline_time
        assert overhead_ratio < 2.0, f"Monitoring overhead too high: {overhead_ratio:.2f}x"
        
    def test_caching_performance(self):
        """Test caching performance benefits."""
        from surrogate_optim.self_healing.performance_optimization import PerformanceOptimizer, PerformanceConfig, OptimizationLevel
        
        config = PerformanceConfig(optimization_level=OptimizationLevel.AGGRESSIVE)
        optimizer = PerformanceOptimizer(config)
        
        # Expensive computation
        def expensive_function(x):
            time.sleep(0.01)  # Simulate expensive computation
            return np.sum(x**2)
            
        optimized_func = optimizer.optimize_function(expensive_function, "expensive_function")
        
        # First call (should be slow)
        start_time = time.time()
        result1 = optimized_func(np.array([1, 2, 3]))
        first_call_time = time.time() - start_time
        
        # Second call with same input (should be cached)
        start_time = time.time()
        result2 = optimized_func(np.array([1, 2, 3]))
        second_call_time = time.time() - start_time
        
        assert result1 == result2
        # Second call should be significantly faster due to caching
        assert second_call_time < first_call_time * 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])