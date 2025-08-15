"""Standalone tests for self-healing functionality without JAX dependencies."""

import sys
import os
import time
import threading
import tempfile
from pathlib import Path

# Add repo to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test the security validation module directly
def test_security_validation():
    """Test security validation module."""
    try:
        # Import without going through main surrogate_optim module
        from surrogate_optim.self_healing.security_validation import (
            SecurityManager, SecurityConfig, SecurityLevel, InputValidator
        )
        
        print("✅ Security validation imports successful")
        
        # Test configuration
        config = SecurityConfig(
            security_level=SecurityLevel.STANDARD,
            enable_input_validation=True,
            max_memory_mb=512
        )
        print("✅ SecurityConfig creation successful")
        
        # Test input validator
        validator = InputValidator(config)
        
        # Test safe inputs
        assert validator.validate_input("safe string", "test")
        assert validator.validate_input(42, "test")
        assert validator.validate_input([1, 2, 3], "test")
        assert validator.validate_input({"key": "value"}, "test")
        print("✅ Safe input validation passed")
        
        # Test dangerous inputs
        dangerous_code = "__import__('os').system('ls')"
        assert not validator.validate_input(dangerous_code, "test")
        print("✅ Dangerous input detection passed")
        
        # Test security manager
        manager = SecurityManager(config)
        status = manager.get_security_status()
        assert "security_level" in status
        assert status["security_level"] == "standard"
        print("✅ SecurityManager functionality passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Security validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_error_handling():
    """Test error handling module."""
    try:
        from surrogate_optim.self_healing.error_handling import (
            AdvancedErrorHandler, ErrorClassifier, RetryConfig, ErrorCategory
        )
        
        print("✅ Error handling imports successful")
        
        # Test error classifier
        classifier = ErrorClassifier()
        
        # Test error classification
        value_error = ValueError("test error")
        category = classifier.classify_error(value_error)
        assert category == ErrorCategory.PERSISTENT
        print("✅ Error classification passed")
        
        # Test retryability
        is_retryable = classifier.is_retryable(value_error)
        assert not is_retryable  # ValueError should not be retryable by default
        print("✅ Retryability check passed")
        
        # Test error handler
        handler = AdvancedErrorHandler()
        stats = handler.get_error_statistics()
        assert "total_errors" in stats
        print("✅ Error handler functionality passed")
        
        # Test retry configuration
        config = RetryConfig(max_attempts=3, initial_delay=0.1)
        handler.register_retry_config("test_operation", config)
        print("✅ Retry configuration passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_monitoring_components():
    """Test monitoring components without JAX."""
    try:
        # Mock numpy for basic functionality
        import numpy as np
        
        # Test pipeline monitor without JAX dependencies
        from surrogate_optim.self_healing.pipeline_monitor import (
            HealthStatus, HealthMetric, PipelineHealth
        )
        
        print("✅ Monitoring component imports successful")
        
        # Test health metric
        metric = HealthMetric(
            name="test_metric",
            value=0.75,
            threshold_warning=0.8,
            threshold_critical=0.9
        )
        
        assert metric.status == HealthStatus.HEALTHY
        print("✅ HealthMetric creation and status passed")
        
        # Test critical metric
        critical_metric = HealthMetric(
            name="critical_metric",
            value=0.95,
            threshold_warning=0.8,
            threshold_critical=0.9
        )
        
        assert critical_metric.status == HealthStatus.CRITICAL
        print("✅ Critical metric detection passed")
        
        # Test pipeline health
        health = PipelineHealth(
            timestamp=time.time(),
            overall_status=HealthStatus.HEALTHY,
            metrics={"test": metric}
        )
        
        assert health.overall_status == HealthStatus.HEALTHY
        assert "test" in health.metrics
        print("✅ PipelineHealth creation passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Monitoring components test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance_optimization():
    """Test performance optimization components."""
    try:
        from surrogate_optim.self_healing.performance_optimization import (
            PerformanceConfig, OptimizationLevel, AdvancedCache
        )
        
        print("✅ Performance optimization imports successful")
        
        # Test performance configuration
        config = PerformanceConfig(
            optimization_level=OptimizationLevel.BALANCED,
            enable_jit=False,  # Disable JAX features for testing
            enable_gpu=False,
            cache_size_mb=128
        )
        
        assert config.optimization_level == OptimizationLevel.BALANCED
        print("✅ PerformanceConfig creation passed")
        
        # Test advanced cache
        cache = AdvancedCache(max_size_mb=64)
        
        # Test cache operations
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        assert cache.get("nonexistent") is None
        print("✅ Cache operations passed")
        
        # Test cache statistics
        stats = cache.get_stats()
        assert "hit_ratio" in stats
        assert "entries" in stats
        assert stats["entries"] == 1
        print("✅ Cache statistics passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance optimization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_robust_monitoring():
    """Test robust monitoring configuration."""
    try:
        from surrogate_optim.self_healing.robust_monitoring import (
            MonitoringConfig, MonitoringLevel, ErrorSeverity
        )
        
        print("✅ Robust monitoring imports successful")
        
        # Test monitoring configuration
        config = MonitoringConfig(
            level=MonitoringLevel.STANDARD,
            check_interval=10.0,
            enable_predictive_monitoring=True
        )
        
        assert config.level == MonitoringLevel.STANDARD
        assert config.check_interval == 10.0
        print("✅ MonitoringConfig creation passed")
        
        # Test error severity enum
        assert ErrorSeverity.LOW.value == "low"
        assert ErrorSeverity.CRITICAL.value == "critical"
        print("✅ ErrorSeverity enum passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Robust monitoring test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_security_benchmark():
    """Run security validation benchmark."""
    try:
        from surrogate_optim.self_healing.security_validation import SecurityManager, SecurityConfig
        
        print("\n🔒 Running Security Benchmark...")
        
        manager = SecurityManager(SecurityConfig())
        
        # Benchmark input validation
        test_inputs = [
            "safe_string",
            42,
            [1, 2, 3, 4, 5],
            {"key": "value", "number": 123},
            "__import__('os')",  # Should be rejected
            "eval('malicious_code')",  # Should be rejected
        ]
        
        start_time = time.time()
        results = []
        
        for i, test_input in enumerate(test_inputs):
            is_valid = manager.input_validator.validate_input(test_input, f"benchmark_{i}")
            results.append(is_valid)
            
        end_time = time.time()
        
        # Expected results: first 4 should pass, last 2 should fail
        expected = [True, True, True, True, False, False]
        
        if results == expected:
            print(f"✅ Security validation benchmark passed in {end_time - start_time:.3f}s")
            print(f"   Processed {len(test_inputs)} inputs")
            print(f"   Rate: {len(test_inputs) / (end_time - start_time):.1f} validations/second")
            return True
        else:
            print(f"❌ Security validation benchmark failed")
            print(f"   Expected: {expected}")
            print(f"   Got: {results}")
            return False
            
    except Exception as e:
        print(f"❌ Security benchmark failed: {e}")
        return False


def run_performance_benchmark():
    """Run performance optimization benchmark."""
    try:
        from surrogate_optim.self_healing.performance_optimization import AdvancedCache
        
        print("\n⚡ Running Performance Benchmark...")
        
        cache = AdvancedCache(max_size_mb=64)
        
        # Benchmark cache operations
        num_operations = 1000
        
        # Write benchmark
        start_time = time.time()
        for i in range(num_operations):
            cache.put(f"key_{i}", f"value_{i}")
        write_time = time.time() - start_time
        
        # Read benchmark
        start_time = time.time()
        hits = 0
        for i in range(num_operations):
            if cache.get(f"key_{i}") is not None:
                hits += 1
        read_time = time.time() - start_time
        
        # Cache statistics
        stats = cache.get_stats()
        
        print(f"✅ Cache performance benchmark completed")
        print(f"   Write operations: {num_operations} in {write_time:.3f}s ({num_operations/write_time:.1f} ops/sec)")
        print(f"   Read operations: {num_operations} in {read_time:.3f}s ({num_operations/read_time:.1f} ops/sec)")
        print(f"   Cache hit ratio: {stats['hit_ratio']:.2%}")
        print(f"   Cache entries: {stats['entries']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance benchmark failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 Running Self-Healing Pipeline Tests (Standalone)")
    print("=" * 50)
    
    tests = [
        ("Security Validation", test_security_validation),
        ("Error Handling", test_error_handling),
        ("Monitoring Components", test_monitoring_components),
        ("Performance Optimization", test_performance_optimization),
        ("Robust Monitoring", test_robust_monitoring),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Testing {test_name}...")
        if test_func():
            passed += 1
            print(f"✅ {test_name} - PASSED")
        else:
            print(f"❌ {test_name} - FAILED")
    
    # Run benchmarks
    print("\n🏃 Running Benchmarks...")
    benchmark_passed = 0
    benchmarks = [
        run_security_benchmark,
        run_performance_benchmark,
    ]
    
    for benchmark in benchmarks:
        if benchmark():
            benchmark_passed += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    print(f"📊 Benchmark Results: {benchmark_passed}/{len(benchmarks)} benchmarks passed")
    
    if passed == total and benchmark_passed == len(benchmarks):
        print("🎉 ALL TESTS AND BENCHMARKS PASSED!")
        return True
    else:
        print("⚠️  Some tests or benchmarks failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)