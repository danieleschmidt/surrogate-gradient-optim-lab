"""Health check implementations for system monitoring."""

from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from typing import Any, Dict, List, Optional

import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str
    duration_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class SystemHealth:
    """Overall system health status."""
    status: HealthStatus
    checks: List[HealthCheckResult]
    timestamp: float = field(default_factory=time.time)
    version: str = "0.1.0"

    @property
    def is_healthy(self) -> bool:
        """Check if system is healthy."""
        return self.status == HealthStatus.HEALTHY

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "status": self.status.value,
            "timestamp": self.timestamp,
            "version": self.version,
            "checks": [
                {
                    "name": check.name,
                    "status": check.status.value,
                    "message": check.message,
                    "duration_ms": check.duration_ms,
                    "metadata": check.metadata,
                    "timestamp": check.timestamp,
                }
                for check in self.checks
            ],
        }


class HealthChecker:
    """Main health checker that runs various system checks."""

    def __init__(self, timeout_seconds: float = 30.0):
        """Initialize health checker.
        
        Args:
            timeout_seconds: Maximum time to wait for all checks to complete.
        """
        self.timeout_seconds = timeout_seconds
        self._checks: Dict[str, callable] = {}
        self._register_default_checks()

    def _register_default_checks(self):
        """Register default health checks."""
        self.register_check("jax_availability", self._check_jax_availability)
        self.register_check("jax_computation", self._check_jax_computation)
        self.register_check("memory_usage", self._check_memory_usage)
        self.register_check("dependency_imports", self._check_dependency_imports)

    def register_check(self, name: str, check_func: callable):
        """Register a custom health check.
        
        Args:
            name: Name of the health check.
            check_func: Function that performs the check and returns HealthCheckResult.
        """
        self._checks[name] = check_func

    def run_check(self, name: str) -> HealthCheckResult:
        """Run a specific health check.
        
        Args:
            name: Name of the check to run.
            
        Returns:
            HealthCheckResult: Result of the health check.
        """
        if name not in self._checks:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Unknown check: {name}",
                duration_ms=0.0,
            )

        start_time = time.time()
        try:
            result = self._checks[name]()
            if not isinstance(result, HealthCheckResult):
                # If check function doesn't return HealthCheckResult, create one
                result = HealthCheckResult(
                    name=name,
                    status=HealthStatus.HEALTHY,
                    message="Check completed successfully",
                    duration_ms=(time.time() - start_time) * 1000,
                )
        except Exception as e:
            logger.exception(f"Health check {name} failed")
            result = HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed: {e!s}",
                duration_ms=(time.time() - start_time) * 1000,
            )

        return result

    def run_all_checks(self, include_checks: Optional[List[str]] = None) -> SystemHealth:
        """Run all registered health checks.
        
        Args:
            include_checks: List of specific checks to run. If None, runs all checks.
            
        Returns:
            SystemHealth: Overall system health status.
        """
        checks_to_run = include_checks or list(self._checks.keys())
        results = []

        for check_name in checks_to_run:
            result = self.run_check(check_name)
            results.append(result)

        # Determine overall status
        overall_status = self._determine_overall_status(results)

        return SystemHealth(
            status=overall_status,
            checks=results,
        )

    def _determine_overall_status(self, results: List[HealthCheckResult]) -> HealthStatus:
        """Determine overall system health from individual check results."""
        if not results:
            return HealthStatus.UNHEALTHY

        statuses = [result.status for result in results]

        if all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        if any(status == HealthStatus.UNHEALTHY for status in statuses):
            return HealthStatus.UNHEALTHY
        return HealthStatus.DEGRADED

    # Default health checks
    def _check_jax_availability(self) -> HealthCheckResult:
        """Check if JAX is available and properly configured."""
        start_time = time.time()

        try:
            # Check JAX import
            import jax
            import jax.numpy as jnp

            # Check JAX devices
            devices = jax.devices()
            device_info = [{"type": str(device.device_kind), "id": device.id} for device in devices]

            # Check if JAX can perform basic operations
            x = jnp.array([1.0, 2.0, 3.0])
            y = jnp.sum(x)

            if not jnp.isfinite(y):
                raise ValueError("JAX computation returned non-finite result")

            duration_ms = (time.time() - start_time) * 1000

            return HealthCheckResult(
                name="jax_availability",
                status=HealthStatus.HEALTHY,
                message=f"JAX is available with {len(devices)} device(s)",
                duration_ms=duration_ms,
                metadata={
                    "devices": device_info,
                    "jax_version": jax.__version__,
                    "computation_result": float(y),
                }
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name="jax_availability",
                status=HealthStatus.UNHEALTHY,
                message=f"JAX check failed: {e!s}",
                duration_ms=duration_ms,
            )

    def _check_jax_computation(self) -> HealthCheckResult:
        """Check if JAX can perform gradient computations."""
        start_time = time.time()

        try:
            # Test basic gradient computation
            def test_function(x):
                return jnp.sum(x**2)

            grad_fn = jax.grad(test_function)
            x = jnp.array([1.0, 2.0, 3.0])
            gradient = grad_fn(x)

            # Expected gradient: 2*x
            expected = 2.0 * x

            if not jnp.allclose(gradient, expected, rtol=1e-6):
                raise ValueError("Gradient computation is inaccurate")

            # Test JIT compilation
            jit_grad_fn = jax.jit(grad_fn)
            jit_gradient = jit_grad_fn(x)

            if not jnp.allclose(gradient, jit_gradient):
                raise ValueError("JIT compilation produces different results")

            duration_ms = (time.time() - start_time) * 1000

            return HealthCheckResult(
                name="jax_computation",
                status=HealthStatus.HEALTHY,
                message="JAX gradient computation working correctly",
                duration_ms=duration_ms,
                metadata={
                    "gradient_test_passed": True,
                    "jit_test_passed": True,
                    "test_input": x.tolist(),
                    "computed_gradient": gradient.tolist(),
                }
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name="jax_computation",
                status=HealthStatus.UNHEALTHY,
                message=f"JAX computation check failed: {e!s}",
                duration_ms=duration_ms,
            )

    def _check_memory_usage(self) -> HealthCheckResult:
        """Check system memory usage."""
        start_time = time.time()

        try:
            import os

            import psutil

            # Get process memory info
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()

            # Get system memory info
            system_memory = psutil.virtual_memory()

            # Define thresholds
            memory_warning_threshold = 80.0  # percent
            memory_critical_threshold = 95.0  # percent

            duration_ms = (time.time() - start_time) * 1000

            if memory_percent > memory_critical_threshold:
                status = HealthStatus.UNHEALTHY
                message = f"Critical memory usage: {memory_percent:.1f}%"
            elif memory_percent > memory_warning_threshold:
                status = HealthStatus.DEGRADED
                message = f"High memory usage: {memory_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage normal: {memory_percent:.1f}%"

            return HealthCheckResult(
                name="memory_usage",
                status=status,
                message=message,
                duration_ms=duration_ms,
                metadata={
                    "process_memory_mb": memory_info.rss / 1024 / 1024,
                    "process_memory_percent": memory_percent,
                    "system_memory_total_gb": system_memory.total / 1024**3,
                    "system_memory_available_gb": system_memory.available / 1024**3,
                    "system_memory_percent": system_memory.percent,
                }
            )

        except ImportError:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name="memory_usage",
                status=HealthStatus.DEGRADED,
                message="psutil not available, cannot check memory usage",
                duration_ms=duration_ms,
            )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name="memory_usage",
                status=HealthStatus.UNHEALTHY,
                message=f"Memory check failed: {e!s}",
                duration_ms=duration_ms,
            )

    def _check_dependency_imports(self) -> HealthCheckResult:
        """Check if required dependencies can be imported."""
        start_time = time.time()

        required_packages = [
            "numpy",
            "scipy",
            "sklearn",
            "matplotlib",
            "plotly",
            "pandas",
            "tqdm",
            "pydantic",
            "typer",
            "rich",
            "loguru",
        ]

        optional_packages = [
            "psutil",
            "memory_profiler",
            "jupyter",
            "notebook",
        ]

        results = {"required": {}, "optional": {}}
        failed_required = []

        # Check required packages
        for package in required_packages:
            try:
                __import__(package)
                results["required"][package] = True
            except ImportError:
                results["required"][package] = False
                failed_required.append(package)

        # Check optional packages
        for package in optional_packages:
            try:
                __import__(package)
                results["optional"][package] = True
            except ImportError:
                results["optional"][package] = False

        duration_ms = (time.time() - start_time) * 1000

        if failed_required:
            return HealthCheckResult(
                name="dependency_imports",
                status=HealthStatus.UNHEALTHY,
                message=f"Required dependencies missing: {', '.join(failed_required)}",
                duration_ms=duration_ms,
                metadata=results,
            )
        optional_missing = sum(1 for available in results["optional"].values() if not available)
        message = "All required dependencies available"
        if optional_missing > 0:
            message += f", {optional_missing} optional dependencies missing"

        return HealthCheckResult(
            name="dependency_imports",
            status=HealthStatus.HEALTHY,
            message=message,
            duration_ms=duration_ms,
            metadata=results,
        )


# Convenience function for quick health checks
def quick_health_check() -> SystemHealth:
    """Perform a quick health check of the system."""
    checker = HealthChecker()
    return checker.run_all_checks()


# Ready endpoint check
def is_ready() -> bool:
    """Check if the system is ready to handle requests."""
    try:
        health = quick_health_check()
        return health.is_healthy
    except Exception:
        return False


# Liveness probe
def is_alive() -> bool:
    """Basic liveness probe - always returns True unless process is dead."""
    return True
