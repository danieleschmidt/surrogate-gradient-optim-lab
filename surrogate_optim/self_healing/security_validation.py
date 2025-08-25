"""Security validation and hardening for self-healing pipeline system."""

from dataclasses import dataclass, field
from enum import Enum
import inspect
import os
from pathlib import Path
import re
import tempfile
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Union

from loguru import logger
import numpy as np

# Security-related imports with fallbacks
try:
    import cryptography
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False
    logger.warning("Cryptography not available. Some security features disabled.")


class SecurityLevel(Enum):
    """Security enforcement levels."""
    MINIMAL = "minimal"
    STANDARD = "standard"
    STRICT = "strict"
    PARANOID = "paranoid"


class ThreatType(Enum):
    """Types of security threats."""
    CODE_INJECTION = "code_injection"
    DATA_POISONING = "data_poisoning"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    INFORMATION_DISCLOSURE = "information_disclosure"
    UNAUTHORIZED_ACCESS = "unauthorized_access"


@dataclass
class SecurityConfig:
    """Security configuration settings."""
    security_level: SecurityLevel = SecurityLevel.STANDARD
    enable_input_validation: bool = True
    enable_output_sanitization: bool = True
    enable_encryption: bool = True
    enable_audit_logging: bool = True
    max_memory_mb: int = 1024
    max_cpu_time_seconds: int = 300
    allowed_file_paths: List[str] = field(default_factory=list)
    blocked_functions: List[str] = field(default_factory=list)
    require_authentication: bool = False


@dataclass
class SecurityViolation:
    """Security violation record."""
    timestamp: float
    threat_type: ThreatType
    severity: str  # "low", "medium", "high", "critical"
    description: str
    context: Dict[str, Any]
    mitigation_applied: bool = False


class InputValidator:
    """Comprehensive input validation system."""

    def __init__(self, config: SecurityConfig):
        self.config = config
        self._dangerous_patterns = self._initialize_dangerous_patterns()

    def _initialize_dangerous_patterns(self) -> List[str]:
        """Initialize patterns for dangerous input detection."""
        return [
            # Code injection patterns
            r"__import__\s*\(",
            r"eval\s*\(",
            r"exec\s*\(",
            r"compile\s*\(",
            r"globals\s*\(\)",
            r"locals\s*\(\)",
            r"vars\s*\(",
            r"dir\s*\(",
            r"getattr\s*\(",
            r"setattr\s*\(",
            r"delattr\s*\(",
            r"hasattr\s*\(",

            # File system access patterns
            r"open\s*\(",
            r"file\s*\(",
            r"os\.",
            r"sys\.",
            r"subprocess\.",
            r"shutil\.",

            # Network access patterns
            r"urllib\.",
            r"requests\.",
            r"socket\.",
            r"http\.",

            # Pickle/serialization patterns (dangerous)
            r"pickle\.",
            r"cPickle\.",
            r"dill\.",

            # Shell command patterns
            r"system\s*\(",
            r"popen\s*\(",
            r"call\s*\(",
            r"check_output\s*\(",
        ]

    def validate_input(self, data: Any, context: str = "unknown") -> bool:
        """Validate input data for security threats."""
        try:
            # Check basic data types
            if not self._validate_data_type(data):
                self._log_security_violation(
                    ThreatType.DATA_POISONING,
                    "high",
                    f"Invalid data type in {context}",
                    {"data_type": type(data).__name__}
                )
                return False

            # Check for code injection in strings
            if isinstance(data, str):
                if not self._validate_string_content(data, context):
                    return False

            # Check numeric bounds
            if isinstance(data, (int, float, np.number)):
                if not self._validate_numeric_bounds(data, context):
                    return False

            # Check array properties
            if isinstance(data, (list, tuple, np.ndarray)):
                if not self._validate_array_properties(data, context):
                    return False

            # Recursive validation for nested structures
            if isinstance(data, dict):
                for key, value in data.items():
                    if not self.validate_input(key, f"{context}.key") or \
                       not self.validate_input(value, f"{context}.{key}"):
                        return False

            elif isinstance(data, (list, tuple)):
                for i, item in enumerate(data):
                    if not self.validate_input(item, f"{context}[{i}]"):
                        return False

            return True

        except Exception as e:
            logger.error(f"Input validation error: {e}")
            return False

    def _validate_data_type(self, data: Any) -> bool:
        """Validate data type safety."""
        allowed_types = (
            str, int, float, bool, list, tuple, dict,
            np.ndarray, np.number, type(None)
        )

        if not isinstance(data, allowed_types):
            return False

        # Check for dangerous callable objects
        if callable(data) and not isinstance(data, type):
            return False

        return True

    def _validate_string_content(self, text: str, context: str) -> bool:
        """Validate string content for dangerous patterns."""
        # Check for dangerous patterns
        for pattern in self._dangerous_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                self._log_security_violation(
                    ThreatType.CODE_INJECTION,
                    "critical",
                    f"Dangerous pattern detected in {context}: {pattern}",
                    {"pattern": pattern, "text_sample": text[:100]}
                )
                return False

        # Check for excessive length
        if len(text) > 10000:
            self._log_security_violation(
                ThreatType.RESOURCE_EXHAUSTION,
                "medium",
                f"Excessive string length in {context}",
                {"length": len(text)}
            )
            return False

        return True

    def _validate_numeric_bounds(self, value: Union[int, float, np.number], context: str) -> bool:
        """Validate numeric value bounds."""
        # Check for NaN and infinity
        if np.isnan(value) or np.isinf(value):
            self._log_security_violation(
                ThreatType.DATA_POISONING,
                "medium",
                f"Invalid numeric value in {context}",
                {"value": str(value)}
            )
            return False

        # Check for extremely large values
        if abs(float(value)) > 1e10:
            self._log_security_violation(
                ThreatType.RESOURCE_EXHAUSTION,
                "low",
                f"Extremely large numeric value in {context}",
                {"value": float(value)}
            )
            return False

        return True

    def _validate_array_properties(self, arr: Union[list, tuple, np.ndarray], context: str) -> bool:
        """Validate array properties for safety."""
        # Check size limits
        if hasattr(arr, "size"):
            size = arr.size
        else:
            size = len(arr)

        if size > 1000000:  # 1M elements
            self._log_security_violation(
                ThreatType.RESOURCE_EXHAUSTION,
                "high",
                f"Excessive array size in {context}",
                {"size": size}
            )
            return False

        # Check memory usage estimate
        if isinstance(arr, np.ndarray):
            memory_mb = arr.nbytes / 1024 / 1024
            if memory_mb > self.config.max_memory_mb:
                self._log_security_violation(
                    ThreatType.RESOURCE_EXHAUSTION,
                    "high",
                    f"Excessive memory usage in {context}",
                    {"memory_mb": memory_mb}
                )
                return False

        return True

    def _log_security_violation(self, threat_type: ThreatType, severity: str, description: str, context: Dict[str, Any]) -> None:
        """Log security violation."""
        violation = SecurityViolation(
            timestamp=time.time(),
            threat_type=threat_type,
            severity=severity,
            description=description,
            context=context
        )

        if severity in ["high", "critical"]:
            logger.error(f"SECURITY VIOLATION: {description}")
        else:
            logger.warning(f"Security concern: {description}")


class FunctionSandbox:
    """Sandbox for safe function execution."""

    def __init__(self, config: SecurityConfig):
        self.config = config
        self._execution_limits = {
            "max_memory_bytes": config.max_memory_mb * 1024 * 1024,
            "max_cpu_time": config.max_cpu_time_seconds
        }

    def execute_safely(self, func: Callable, args: tuple, kwargs: dict, context: str = "unknown") -> Any:
        """Execute function in secure sandbox."""
        # Pre-execution validation
        if not self._validate_function_safety(func):
            raise SecurityError(f"Function {func.__name__} failed security validation")

        # Validate arguments
        validator = InputValidator(self.config)
        for i, arg in enumerate(args):
            if not validator.validate_input(arg, f"{context}.arg[{i}]"):
                raise SecurityError(f"Argument {i} failed security validation")

        for key, value in kwargs.items():
            if not validator.validate_input(value, f"{context}.{key}"):
                raise SecurityError(f"Keyword argument {key} failed security validation")

        # Execute with monitoring
        start_time = time.time()
        start_memory = self._get_memory_usage()

        try:
            # Create restricted execution environment
            restricted_globals = self._create_restricted_globals()

            # Monitor execution
            result = self._monitored_execution(func, args, kwargs, restricted_globals)

            # Post-execution validation
            execution_time = time.time() - start_time
            memory_used = self._get_memory_usage() - start_memory

            if execution_time > self.config.max_cpu_time_seconds:
                raise SecurityError(f"Function execution exceeded time limit: {execution_time:.2f}s")

            if memory_used > self._execution_limits["max_memory_bytes"]:
                raise SecurityError(f"Function execution exceeded memory limit: {memory_used / 1024 / 1024:.1f}MB")

            # Validate output
            if not validator.validate_input(result, f"{context}.output"):
                raise SecurityError("Function output failed security validation")

            return result

        except Exception as e:
            if isinstance(e, SecurityError):
                raise
            logger.error(f"Sandboxed execution error: {e}")
            raise SecurityError(f"Function execution failed: {e!s}")

    def _validate_function_safety(self, func: Callable) -> bool:
        """Validate function for safety."""
        # Check if function is in blocked list
        func_name = func.__name__
        if func_name in self.config.blocked_functions:
            return False

        # Inspect function source code if available
        try:
            source = inspect.getsource(func)
            validator = InputValidator(self.config)
            return validator._validate_string_content(source, f"function.{func_name}")
        except Exception:
            # Can't inspect source, allow but log warning
            logger.warning(f"Cannot inspect source of function {func_name}")
            return True

    def _create_restricted_globals(self) -> Dict[str, Any]:
        """Create restricted global namespace."""
        # Safe built-ins only
        safe_builtins = {
            "abs", "all", "any", "bin", "bool", "chr", "dict", "enumerate",
            "filter", "float", "frozenset", "hash", "hex", "int", "len",
            "list", "map", "max", "min", "oct", "ord", "pow", "range",
            "reversed", "round", "set", "slice", "sorted", "str", "sum",
            "tuple", "type", "zip"
        }

        restricted_builtins = {}
        import builtins
        for name in safe_builtins:
            if hasattr(builtins, name):
                restricted_builtins[name] = getattr(builtins, name)

        # Add safe modules
        import math

        import jax.numpy as jnp
        import numpy as np

        restricted_globals = {
            "__builtins__": restricted_builtins,
            "math": math,
            "np": np,
            "jnp": jnp,
        }

        return restricted_globals

    def _monitored_execution(self, func: Callable, args: tuple, kwargs: dict, globals_dict: Dict[str, Any]) -> Any:
        """Execute function with monitoring."""
        # For now, execute directly (could be enhanced with actual sandboxing)
        return func(*args, **kwargs)

    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss
        except Exception:
            return 0


class SecureDataHandler:
    """Secure data handling with encryption and validation."""

    def __init__(self, config: SecurityConfig):
        self.config = config
        self._cipher = None

        if config.enable_encryption and CRYPTOGRAPHY_AVAILABLE:
            self._initialize_encryption()

    def _initialize_encryption(self) -> None:
        """Initialize encryption system."""
        # Generate or load encryption key
        key = self._get_or_create_key()
        self._cipher = Fernet(key)

    def _get_or_create_key(self) -> bytes:
        """Get or create encryption key."""
        key_file = Path(tempfile.gettempdir()) / "surrogate_optim_key"

        if key_file.exists():
            with open(key_file, "rb") as f:
                return f.read()
        else:
            # Generate new key
            key = Fernet.generate_key()
            with open(key_file, "wb") as f:
                f.write(key)
            os.chmod(key_file, 0o600)  # Restrict permissions
            return key

    def encrypt_data(self, data: bytes) -> bytes:
        """Encrypt sensitive data."""
        if not self._cipher:
            return data  # No encryption available

        try:
            return self._cipher.encrypt(data)
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            return data

    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt sensitive data."""
        if not self._cipher:
            return encrypted_data  # No encryption available

        try:
            return self._cipher.decrypt(encrypted_data)
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return encrypted_data

    def secure_serialize(self, obj: Any) -> bytes:
        """Securely serialize object."""
        import pickle

        # Validate object before serialization
        validator = InputValidator(self.config)
        if not validator.validate_input(obj, "serialization"):
            raise SecurityError("Object failed security validation for serialization")

        # Serialize
        data = pickle.dumps(obj)

        # Encrypt if enabled
        if self.config.enable_encryption:
            data = self.encrypt_data(data)

        return data

    def secure_deserialize(self, data: bytes) -> Any:
        """Securely deserialize object."""
        import pickle

        # Decrypt if needed
        if self.config.enable_encryption:
            data = self.decrypt_data(data)

        # Deserialize with safety checks
        try:
            obj = pickle.loads(data)

            # Validate deserialized object
            validator = InputValidator(self.config)
            if not validator.validate_input(obj, "deserialization"):
                raise SecurityError("Deserialized object failed security validation")

            return obj

        except Exception as e:
            logger.error(f"Deserialization failed: {e}")
            raise SecurityError(f"Failed to deserialize data: {e!s}")


class SecurityAuditor:
    """Security audit and compliance system."""

    def __init__(self, config: SecurityConfig):
        self.config = config
        self._audit_log: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def audit_operation(self, operation: str, user: str, details: Dict[str, Any]) -> None:
        """Audit security-relevant operation."""
        if not self.config.enable_audit_logging:
            return

        with self._lock:
            audit_entry = {
                "timestamp": time.time(),
                "operation": operation,
                "user": user,
                "details": details,
                "source_ip": self._get_source_ip(),
                "process_id": os.getpid()
            }

            self._audit_log.append(audit_entry)

            # Keep only recent entries
            if len(self._audit_log) > 10000:
                self._audit_log = self._audit_log[-10000:]

            # Log critical operations immediately
            if operation in ["function_execution", "data_access", "configuration_change"]:
                logger.info(f"AUDIT: {operation} by {user}")

    def _get_source_ip(self) -> str:
        """Get source IP address (simplified)."""
        return "127.0.0.1"  # Localhost for now

    def get_audit_report(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get audit report for specified time period."""
        cutoff_time = time.time() - (hours * 3600)

        with self._lock:
            return [
                entry for entry in self._audit_log
                if entry["timestamp"] > cutoff_time
            ]

    def detect_suspicious_activity(self) -> List[Dict[str, Any]]:
        """Detect suspicious activity patterns."""
        suspicious_events = []
        recent_events = self.get_audit_report(1)  # Last hour

        # Check for high-frequency operations
        operation_counts = {}
        for event in recent_events:
            key = f"{event['user']}:{event['operation']}"
            operation_counts[key] = operation_counts.get(key, 0) + 1

        for key, count in operation_counts.items():
            if count > 100:  # More than 100 operations per hour
                suspicious_events.append({
                    "type": "high_frequency_operations",
                    "details": {"key": key, "count": count},
                    "severity": "medium"
                })

        # Check for failed operations
        failed_operations = [
            event for event in recent_events
            if event["details"].get("success") is False
        ]

        if len(failed_operations) > 10:
            suspicious_events.append({
                "type": "multiple_failures",
                "details": {"count": len(failed_operations)},
                "severity": "high"
            })

        return suspicious_events


class SecurityError(Exception):
    """Security-related exception."""
    pass


class SecurityManager:
    """Main security management system."""

    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()

        # Initialize components
        self.input_validator = InputValidator(self.config)
        self.function_sandbox = FunctionSandbox(self.config)
        self.data_handler = SecureDataHandler(self.config)
        self.auditor = SecurityAuditor(self.config)

        # Security state
        self._violations: List[SecurityViolation] = []
        self._security_enabled = True

        logger.info(f"Security manager initialized with {self.config.security_level.value} level")

    def validate_and_execute(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict,
        user: str = "system",
        context: str = "unknown"
    ) -> Any:
        """Validate inputs and execute function securely."""
        # Audit the operation
        self.auditor.audit_operation(
            "function_execution",
            user,
            {"function": func.__name__, "context": context}
        )

        try:
            # Execute in sandbox
            result = self.function_sandbox.execute_safely(func, args, kwargs, context)

            # Audit success
            self.auditor.audit_operation(
                "function_execution_success",
                user,
                {"function": func.__name__, "context": context}
            )

            return result

        except SecurityError as e:
            # Record security violation
            violation = SecurityViolation(
                timestamp=time.time(),
                threat_type=ThreatType.UNAUTHORIZED_ACCESS,
                severity="high",
                description=str(e),
                context={"function": func.__name__, "user": user, "context": context}
            )
            self._violations.append(violation)

            # Audit failure
            self.auditor.audit_operation(
                "function_execution_failure",
                user,
                {"function": func.__name__, "context": context, "error": str(e)}
            )

            raise

    def secure_data_operation(self, operation: str, data: Any, user: str = "system") -> Any:
        """Perform secure data operation."""
        # Audit data access
        self.auditor.audit_operation(
            "data_access",
            user,
            {"operation": operation, "data_type": type(data).__name__}
        )

        # Validate data
        if not self.input_validator.validate_input(data, f"data_operation.{operation}"):
            raise SecurityError(f"Data validation failed for operation: {operation}")

        # Process based on operation type
        if operation == "serialize":
            return self.data_handler.secure_serialize(data)
        if operation == "deserialize":
            return self.data_handler.secure_deserialize(data)
        raise SecurityError(f"Unknown data operation: {operation}")

    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        return {
            "security_level": self.config.security_level.value,
            "security_enabled": self._security_enabled,
            "recent_violations": len([
                v for v in self._violations
                if time.time() - v.timestamp < 3600
            ]),
            "total_violations": len(self._violations),
            "suspicious_activity": self.auditor.detect_suspicious_activity(),
            "encryption_enabled": self.config.enable_encryption and CRYPTOGRAPHY_AVAILABLE,
            "audit_logging_enabled": self.config.enable_audit_logging
        }

    def emergency_shutdown(self, reason: str, user: str = "system") -> None:
        """Emergency security shutdown."""
        logger.critical(f"EMERGENCY SECURITY SHUTDOWN: {reason}")

        # Audit emergency shutdown
        self.auditor.audit_operation(
            "emergency_shutdown",
            user,
            {"reason": reason}
        )

        # Disable security operations
        self._security_enabled = False

        # Additional shutdown procedures could be added here

    def get_violation_report(self, hours: int = 24) -> List[SecurityViolation]:
        """Get security violation report."""
        cutoff_time = time.time() - (hours * 3600)
        return [
            violation for violation in self._violations
            if violation.timestamp > cutoff_time
        ]


# Global security manager instance
global_security_manager = SecurityManager()

# Security decorators
def secure_function(user: str = "system", context: str = "unknown"):
    """Decorator for secure function execution."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            return global_security_manager.validate_and_execute(
                func, args, kwargs, user, context
            )
        return wrapper
    return decorator

def validate_inputs(context: str = "unknown"):
    """Decorator for input validation only."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Validate all inputs
            for i, arg in enumerate(args):
                if not global_security_manager.input_validator.validate_input(arg, f"{context}.arg[{i}]"):
                    raise SecurityError(f"Argument {i} failed security validation")

            for key, value in kwargs.items():
                if not global_security_manager.input_validator.validate_input(value, f"{context}.{key}"):
                    raise SecurityError(f"Keyword argument {key} failed security validation")

            return func(*args, **kwargs)
        return wrapper
    return decorator
