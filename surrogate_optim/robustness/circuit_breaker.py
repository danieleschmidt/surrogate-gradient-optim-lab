"""Circuit breaker pattern for robust surrogate optimization."""

from dataclasses import dataclass
from enum import Enum
import functools
import threading
import time
from typing import Any, Callable, Optional

from ..monitoring.enhanced_logging import enhanced_logger


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, calls rejected
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5
    recovery_timeout: float = 60.0  # seconds
    expected_exception: type = Exception


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self._lock = threading.Lock()

    def __call__(self, func: Callable) -> Callable:
        """Decorator to apply circuit breaker to a function."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)

        return wrapper

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    enhanced_logger.info("Circuit breaker transitioning to HALF_OPEN")
                else:
                    raise Exception("Circuit breaker is OPEN - rejecting call")

            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except self.config.expected_exception:
                self._on_failure()
                raise

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.config.recovery_timeout

    def _on_success(self) -> None:
        """Handle successful execution."""
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            enhanced_logger.info("Circuit breaker reset to CLOSED")
        self.failure_count = 0

    def _on_failure(self) -> None:
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            enhanced_logger.warning(
                f"Circuit breaker opened after {self.failure_count} failures"
            )

    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.last_failure_time = None
            enhanced_logger.info("Circuit breaker manually reset")


# Pre-configured circuit breakers for common operations
optimization_circuit_breaker = CircuitBreaker(
    CircuitBreakerConfig(
        failure_threshold=3, recovery_timeout=30.0, expected_exception=Exception
    )
)

training_circuit_breaker = CircuitBreaker(
    CircuitBreakerConfig(
        failure_threshold=2, recovery_timeout=60.0, expected_exception=Exception
    )
)

prediction_circuit_breaker = CircuitBreaker(
    CircuitBreakerConfig(
        failure_threshold=5, recovery_timeout=10.0, expected_exception=Exception
    )
)
