"""Advanced error handling and exception management system."""

from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
import functools
import inspect
import threading
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Type

import jax
from loguru import logger


class RetryStrategy(Enum):
    """Retry strategy types."""
    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIBONACCI_BACKOFF = "fibonacci_backoff"


class ErrorCategory(Enum):
    """Error categorization for handling strategies."""
    TRANSIENT = "transient"  # Temporary errors that might resolve
    PERSISTENT = "persistent"  # Errors that need intervention
    FATAL = "fatal"  # Errors that require system shutdown
    CONFIGURATION = "configuration"  # Configuration-related errors
    RESOURCE = "resource"  # Resource exhaustion errors
    NETWORK = "network"  # Network-related errors
    DATA = "data"  # Data corruption or validation errors


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    retryable_exceptions: List[Type[Exception]] = field(default_factory=list)
    non_retryable_exceptions: List[Type[Exception]] = field(default_factory=list)


@dataclass
class ErrorContext:
    """Comprehensive error context information."""
    timestamp: float
    function_name: str
    module_name: str
    line_number: int
    arguments: Dict[str, Any]
    local_variables: Dict[str, Any]
    system_state: Dict[str, Any]
    stack_trace: str
    attempt_number: int = 1
    category: Optional[ErrorCategory] = None


class ErrorClassifier:
    """Intelligent error classification system."""

    def __init__(self):
        self._classification_rules = self._initialize_classification_rules()

    def _initialize_classification_rules(self) -> Dict[str, ErrorCategory]:
        """Initialize error classification rules."""
        return {
            # Transient errors
            "ConnectionError": ErrorCategory.TRANSIENT,
            "TimeoutError": ErrorCategory.TRANSIENT,
            "TemporaryFailure": ErrorCategory.TRANSIENT,
            "ResourceTemporarilyUnavailable": ErrorCategory.TRANSIENT,

            # Persistent errors
            "ValueError": ErrorCategory.PERSISTENT,
            "TypeError": ErrorCategory.PERSISTENT,
            "AttributeError": ErrorCategory.PERSISTENT,
            "KeyError": ErrorCategory.PERSISTENT,

            # Fatal errors
            "SystemExit": ErrorCategory.FATAL,
            "KeyboardInterrupt": ErrorCategory.FATAL,
            "MemoryError": ErrorCategory.FATAL,
            "SystemError": ErrorCategory.FATAL,

            # Configuration errors
            "ImportError": ErrorCategory.CONFIGURATION,
            "ModuleNotFoundError": ErrorCategory.CONFIGURATION,
            "FileNotFoundError": ErrorCategory.CONFIGURATION,

            # Resource errors
            "OSError": ErrorCategory.RESOURCE,
            "IOError": ErrorCategory.RESOURCE,
            "PermissionError": ErrorCategory.RESOURCE,

            # Network errors
            "ConnectionError": ErrorCategory.NETWORK,
            "NetworkTimeout": ErrorCategory.NETWORK,
            "DNSError": ErrorCategory.NETWORK,

            # Data errors
            "ValidationError": ErrorCategory.DATA,
            "DataCorruption": ErrorCategory.DATA,
            "ChecksumError": ErrorCategory.DATA,
        }

    def classify_error(self, error: Exception) -> ErrorCategory:
        """Classify error into appropriate category."""
        error_name = error.__class__.__name__

        # Direct mapping
        if error_name in self._classification_rules:
            return self._classification_rules[error_name]

        # Pattern-based classification
        error_message = str(error).lower()

        if any(keyword in error_message for keyword in ["timeout", "connection", "network"]):
            return ErrorCategory.TRANSIENT
        if any(keyword in error_message for keyword in ["permission", "access", "denied"]) or any(keyword in error_message for keyword in ["memory", "space", "full"]):
            return ErrorCategory.RESOURCE
        if any(keyword in error_message for keyword in ["config", "setting", "parameter"]):
            return ErrorCategory.CONFIGURATION
        if any(keyword in error_message for keyword in ["corrupt", "invalid", "malformed"]):
            return ErrorCategory.DATA
        return ErrorCategory.PERSISTENT

    def is_retryable(self, error: Exception) -> bool:
        """Determine if error is worth retrying."""
        category = self.classify_error(error)

        # Fatal errors should never be retried
        if category == ErrorCategory.FATAL:
            return False

        # Transient errors are usually retryable
        if category == ErrorCategory.TRANSIENT:
            return True

        # Configuration errors are typically not retryable
        if category == ErrorCategory.CONFIGURATION:
            return False

        # Resource errors might be retryable after some time
        if category == ErrorCategory.RESOURCE:
            return True

        # Data errors are usually not retryable without intervention
        if category == ErrorCategory.DATA:
            return False

        # Default for persistent errors
        return False


class AdvancedErrorHandler:
    """Advanced error handling system with intelligent retry and recovery."""

    def __init__(self):
        self.classifier = ErrorClassifier()
        self._error_counts: Dict[str, int] = {}
        self._error_history: List[ErrorContext] = []
        self._retry_configs: Dict[str, RetryConfig] = {}
        self._global_retry_config = RetryConfig()
        self._locks: Dict[str, threading.RLock] = {}

        # Register default retry configurations
        self._register_default_retry_configs()

    def _register_default_retry_configs(self) -> None:
        """Register default retry configurations for common operations."""
        self._retry_configs.update({
            "optimization": RetryConfig(
                strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                max_attempts=3,
                initial_delay=1.0,
                max_delay=30.0,
                retryable_exceptions=[RuntimeError, ValueError]
            ),
            "model_training": RetryConfig(
                strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                max_attempts=2,
                initial_delay=5.0,
                max_delay=60.0,
                retryable_exceptions=[RuntimeError]
            ),
            "data_loading": RetryConfig(
                strategy=RetryStrategy.LINEAR_BACKOFF,
                max_attempts=5,
                initial_delay=0.5,
                max_delay=10.0,
                retryable_exceptions=[IOError, OSError, ConnectionError]
            ),
            "jax_compilation": RetryConfig(
                strategy=RetryStrategy.FIXED_DELAY,
                max_attempts=2,
                initial_delay=2.0,
                retryable_exceptions=[RuntimeError]
            )
        })

    def with_retry(
        self,
        operation_name: Optional[str] = None,
        config: Optional[RetryConfig] = None,
        capture_context: bool = True
    ):
        """Decorator for adding retry logic to functions."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return self._execute_with_retry(
                    func, args, kwargs, operation_name or func.__name__, config, capture_context
                )
            return wrapper
        return decorator

    def _execute_with_retry(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict,
        operation_name: str,
        config: Optional[RetryConfig],
        capture_context: bool
    ) -> Any:
        """Execute function with retry logic."""
        retry_config = config or self._retry_configs.get(operation_name, self._global_retry_config)
        last_exception = None

        for attempt in range(1, retry_config.max_attempts + 1):
            try:
                return func(*args, **kwargs)

            except Exception as e:
                last_exception = e

                # Capture error context
                if capture_context:
                    context = self._capture_error_context(func, args, kwargs, e, attempt)
                    self._record_error_context(context)

                # Check if error is retryable
                if not self._should_retry(e, attempt, retry_config):
                    logger.error(f"Operation {operation_name} failed (non-retryable): {e}")
                    raise

                # Calculate delay
                delay = self._calculate_delay(attempt, retry_config)

                logger.warning(
                    f"Operation {operation_name} failed (attempt {attempt}/{retry_config.max_attempts}): {e}. "
                    f"Retrying in {delay:.2f}s..."
                )

                # Apply delay
                time.sleep(delay)

        # All retries exhausted
        logger.error(f"Operation {operation_name} failed after {retry_config.max_attempts} attempts")
        raise last_exception

    def _should_retry(self, error: Exception, attempt: int, config: RetryConfig) -> bool:
        """Determine if error should trigger a retry."""
        if attempt >= config.max_attempts:
            return False

        # Check non-retryable exceptions
        if any(isinstance(error, exc_type) for exc_type in config.non_retryable_exceptions):
            return False

        # Check retryable exceptions
        if config.retryable_exceptions:
            if not any(isinstance(error, exc_type) for exc_type in config.retryable_exceptions):
                return False

        # Use classifier if no explicit rules
        if not config.retryable_exceptions and not config.non_retryable_exceptions:
            return self.classifier.is_retryable(error)

        return True

    def _calculate_delay(self, attempt: int, config: RetryConfig) -> float:
        """Calculate delay for retry attempt."""
        if config.strategy == RetryStrategy.FIXED_DELAY:
            delay = config.initial_delay
        elif config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = config.initial_delay * attempt
        elif config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = config.initial_delay * (config.backoff_multiplier ** (attempt - 1))
        elif config.strategy == RetryStrategy.FIBONACCI_BACKOFF:
            fib_sequence = [1, 1]
            for i in range(2, attempt):
                fib_sequence.append(fib_sequence[i-1] + fib_sequence[i-2])
            delay = config.initial_delay * fib_sequence[min(attempt - 1, len(fib_sequence) - 1)]
        else:
            delay = config.initial_delay

        # Apply maximum delay limit
        delay = min(delay, config.max_delay)

        # Add jitter to avoid thundering herd
        if config.jitter:
            import random
            jitter_factor = random.uniform(0.8, 1.2)
            delay *= jitter_factor

        return delay

    def _capture_error_context(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict,
        error: Exception,
        attempt: int
    ) -> ErrorContext:
        """Capture comprehensive error context."""
        frame = inspect.currentframe()

        try:
            # Get function frame
            while frame and frame.f_code.co_name != func.__name__:
                frame = frame.f_back

            if frame:
                local_vars = {k: self._safe_repr(v) for k, v in frame.f_locals.items()}
                line_number = frame.f_lineno
            else:
                local_vars = {}
                line_number = 0

            # Safe argument representation
            safe_args = {}
            try:
                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                safe_args = {k: self._safe_repr(v) for k, v in bound_args.arguments.items()}
            except Exception:
                safe_args = {"args": str(args)[:200], "kwargs": str(kwargs)[:200]}

            # System state
            system_state = self._capture_system_state()

            return ErrorContext(
                timestamp=time.time(),
                function_name=func.__name__,
                module_name=func.__module__,
                line_number=line_number,
                arguments=safe_args,
                local_variables=local_vars,
                system_state=system_state,
                stack_trace=traceback.format_exc(),
                attempt_number=attempt,
                category=self.classifier.classify_error(error)
            )

        finally:
            del frame

    def _safe_repr(self, obj: Any, max_length: int = 200) -> str:
        """Safe representation of objects that might fail to repr."""
        try:
            repr_str = repr(obj)
            if len(repr_str) > max_length:
                repr_str = repr_str[:max_length] + "..."
            return repr_str
        except Exception:
            return f"<{type(obj).__name__} object (repr failed)>"

    def _capture_system_state(self) -> Dict[str, Any]:
        """Capture relevant system state information."""
        try:
            import psutil

            state = {
                "memory_percent": psutil.virtual_memory().percent,
                "cpu_percent": psutil.cpu_percent(),
                "disk_percent": psutil.disk_usage("/").percent,
                "process_count": len(psutil.pids()),
                "timestamp": time.time()
            }

            # JAX-specific state
            try:
                state["jax_devices"] = len(jax.devices())
                state["jax_backend"] = jax.lib.xla_bridge.get_backend().platform
            except Exception:
                pass

            return state

        except Exception:
            return {"error": "Failed to capture system state"}

    def _record_error_context(self, context: ErrorContext) -> None:
        """Record error context in history."""
        self._error_history.append(context)

        # Maintain rolling window
        if len(self._error_history) > 1000:
            self._error_history = self._error_history[-1000:]

        # Update error counts
        error_key = f"{context.function_name}:{context.category.value if context.category else 'unknown'}"
        self._error_counts[error_key] = self._error_counts.get(error_key, 0) + 1

    @contextmanager
    def error_boundary(
        self,
        operation_name: str,
        fallback_value: Any = None,
        suppress_errors: bool = False,
        log_errors: bool = True
    ):
        """Error boundary context manager for safe execution."""
        try:
            yield
        except Exception as e:
            if log_errors:
                context = self._capture_error_context(
                    lambda: None, (), {}, e, 1
                )
                context.function_name = operation_name
                self._record_error_context(context)

                logger.error(f"Error in {operation_name}: {e}")

            if suppress_errors:
                if fallback_value is not None:
                    return fallback_value
            else:
                raise

    def circuit_breaker(
        self,
        operation_name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception
    ):
        """Circuit breaker decorator for preventing cascading failures."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return self._execute_with_circuit_breaker(
                    func, args, kwargs, operation_name,
                    failure_threshold, recovery_timeout, expected_exception
                )
            return wrapper
        return decorator

    def _execute_with_circuit_breaker(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict,
        operation_name: str,
        failure_threshold: int,
        recovery_timeout: float,
        expected_exception: Type[Exception]
    ) -> Any:
        """Execute function with circuit breaker protection."""
        # Get or create lock for this operation
        if operation_name not in self._locks:
            self._locks[operation_name] = threading.RLock()

        with self._locks[operation_name]:
            # Check circuit breaker state
            error_key = f"circuit_breaker:{operation_name}"
            failure_count = self._error_counts.get(error_key, 0)
            last_failure_time = getattr(self, f"_last_failure_{operation_name}", 0)

            # Circuit is open
            if failure_count >= failure_threshold:
                if time.time() - last_failure_time < recovery_timeout:
                    raise RuntimeError(f"Circuit breaker is OPEN for {operation_name}")
                # Try to reset circuit breaker
                logger.info(f"Attempting to reset circuit breaker for {operation_name}")

            try:
                result = func(*args, **kwargs)

                # Reset failure count on success
                if error_key in self._error_counts:
                    del self._error_counts[error_key]
                    logger.info(f"Circuit breaker reset for {operation_name}")

                return result

            except expected_exception:
                # Increment failure count
                self._error_counts[error_key] = self._error_counts.get(error_key, 0) + 1
                setattr(self, f"_last_failure_{operation_name}", time.time())

                if self._error_counts[error_key] >= failure_threshold:
                    logger.error(f"Circuit breaker OPENED for {operation_name} after {failure_threshold} failures")

                raise

    def bulk_retry(
        self,
        operations: List[Callable],
        operation_names: Optional[List[str]] = None,
        configs: Optional[List[RetryConfig]] = None,
        fail_fast: bool = False
    ) -> List[Any]:
        """Execute multiple operations with retry logic."""
        if operation_names is None:
            operation_names = [f"operation_{i}" for i in range(len(operations))]

        if configs is None:
            configs = [self._global_retry_config] * len(operations)

        results = []

        for i, (operation, name, config) in enumerate(zip(operations, operation_names, configs)):
            try:
                result = self._execute_with_retry(operation, (), {}, name, config, True)
                results.append(result)
            except Exception as e:
                if fail_fast:
                    logger.error(f"Bulk operation failed fast at {name}: {e}")
                    raise
                logger.error(f"Bulk operation {name} failed, continuing: {e}")
                results.append(None)

        return results

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        if not self._error_history:
            return {"total_errors": 0}

        # Error distribution by category
        category_counts = {}
        for context in self._error_history:
            if context.category:
                category_counts[context.category.value] = category_counts.get(context.category.value, 0) + 1

        # Error distribution by function
        function_counts = {}
        for context in self._error_history:
            function_counts[context.function_name] = function_counts.get(context.function_name, 0) + 1

        # Recent errors (last hour)
        recent_cutoff = time.time() - 3600
        recent_errors = [ctx for ctx in self._error_history if ctx.timestamp > recent_cutoff]

        return {
            "total_errors": len(self._error_history),
            "recent_errors": len(recent_errors),
            "category_distribution": category_counts,
            "function_distribution": function_counts,
            "error_counts": self._error_counts.copy(),
            "error_rate": len(recent_errors) / 3600.0  # Errors per second
        }

    def register_retry_config(self, operation_name: str, config: RetryConfig) -> None:
        """Register retry configuration for specific operation."""
        self._retry_configs[operation_name] = config
        logger.info(f"Registered retry config for {operation_name}")

    def set_global_retry_config(self, config: RetryConfig) -> None:
        """Set global retry configuration."""
        self._global_retry_config = config
        logger.info("Updated global retry configuration")


# Global error handler instance
global_error_handler = AdvancedErrorHandler()

# Convenience decorators
def with_retry(operation_name: Optional[str] = None, config: Optional[RetryConfig] = None):
    """Convenience decorator for retry functionality."""
    return global_error_handler.with_retry(operation_name, config)

def circuit_breaker(operation_name: str, failure_threshold: int = 5, recovery_timeout: float = 60.0):
    """Convenience decorator for circuit breaker functionality."""
    return global_error_handler.circuit_breaker(operation_name, failure_threshold, recovery_timeout)

def error_boundary(operation_name: str, fallback_value: Any = None, suppress_errors: bool = False):
    """Convenience context manager for error boundaries."""
    return global_error_handler.error_boundary(operation_name, fallback_value, suppress_errors)
