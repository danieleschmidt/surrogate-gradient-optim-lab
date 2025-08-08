"""Security and input validation framework for surrogate optimization."""

import hashlib
import re
import time
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
from jax import Array


@dataclass
class SecurityViolation:
    """Represents a security violation or concern."""
    severity: str  # "low", "medium", "high", "critical"
    category: str  # "input_validation", "data_integrity", "resource_limits", etc.
    message: str
    details: Dict[str, Any]
    timestamp: float


class InputValidator:
    """Comprehensive input validation for surrogate optimization."""
    
    def __init__(
        self,
        max_dimensions: int = 1000,
        max_samples: int = 1000000,
        max_array_size: int = 10**8,
        allowed_dtypes: List[str] = None,
        bounds_tolerance: float = 1e-12,
    ):
        """Initialize input validator.
        
        Args:
            max_dimensions: Maximum allowed input dimensions
            max_samples: Maximum allowed number of samples
            max_array_size: Maximum allowed array size (elements)
            allowed_dtypes: List of allowed data types
            bounds_tolerance: Tolerance for bounds checking
        """
        self.max_dimensions = max_dimensions
        self.max_samples = max_samples
        self.max_array_size = max_array_size
        self.bounds_tolerance = bounds_tolerance
        
        if allowed_dtypes is None:
            self.allowed_dtypes = [
                'float32', 'float64', 'int32', 'int64',
                'complex64', 'complex128'
            ]
        else:
            self.allowed_dtypes = allowed_dtypes
        
        self.violations = []
    
    def _log_violation(self, severity: str, category: str, message: str, **details):
        """Log a security violation."""
        violation = SecurityViolation(
            severity=severity,
            category=category,
            message=message,
            details=details,
            timestamp=time.time()
        )
        self.violations.append(violation)
        return violation
    
    def validate_array(self, x: Array, name: str = "array") -> List[SecurityViolation]:
        """Validate an input array for security issues.
        
        Args:
            x: Array to validate
            name: Name of the array for error reporting
            
        Returns:
            List of security violations found
        """
        violations = []
        
        # Check array size
        if x.size > self.max_array_size:
            violations.append(self._log_violation(
                "high",
                "resource_limits",
                f"{name} exceeds maximum array size",
                size=int(x.size),
                max_size=self.max_array_size
            ))
        
        # Check dimensions
        if x.ndim > 0 and x.shape[-1] > self.max_dimensions:
            violations.append(self._log_violation(
                "medium",
                "resource_limits", 
                f"{name} exceeds maximum dimensions",
                dimensions=int(x.shape[-1]),
                max_dimensions=self.max_dimensions
            ))
        
        # Check for excessive number of samples
        if x.ndim > 1 and x.shape[0] > self.max_samples:
            violations.append(self._log_violation(
                "medium",
                "resource_limits",
                f"{name} exceeds maximum number of samples",
                samples=int(x.shape[0]),
                max_samples=self.max_samples
            ))
        
        # Check data type
        if str(x.dtype) not in self.allowed_dtypes:
            violations.append(self._log_violation(
                "medium",
                "input_validation",
                f"{name} has disallowed data type",
                dtype=str(x.dtype),
                allowed_dtypes=self.allowed_dtypes
            ))
        
        # Check for NaN/Inf values
        if not jnp.isfinite(x).all():
            nan_count = int(jnp.isnan(x).sum())
            inf_count = int(jnp.isinf(x).sum())
            violations.append(self._log_violation(
                "high",
                "data_integrity",
                f"{name} contains non-finite values",
                nan_count=nan_count,
                inf_count=inf_count
            ))
        
        # Check for extreme values that might cause numerical issues
        finite_mask = jnp.isfinite(x)
        if jnp.any(finite_mask):
            finite_values = x[finite_mask]
            max_val = float(jnp.max(jnp.abs(finite_values)))
            
            if max_val > 1e15:
                violations.append(self._log_violation(
                    "medium",
                    "numerical_stability",
                    f"{name} contains extremely large values",
                    max_absolute_value=max_val
                ))
            
            min_nonzero = float(jnp.min(jnp.abs(finite_values[finite_values != 0]))) if jnp.any(finite_values != 0) else 0
            if min_nonzero > 0 and min_nonzero < 1e-15:
                violations.append(self._log_violation(
                    "low",
                    "numerical_stability",
                    f"{name} contains extremely small values",
                    min_nonzero_value=min_nonzero
                ))
        
        return violations
    
    def validate_bounds(self, bounds: List[tuple], name: str = "bounds") -> List[SecurityViolation]:
        """Validate optimization bounds.
        
        Args:
            bounds: List of (lower, upper) bound tuples
            name: Name for error reporting
            
        Returns:
            List of security violations found
        """
        violations = []
        
        # Check number of dimensions
        if len(bounds) > self.max_dimensions:
            violations.append(self._log_violation(
                "medium",
                "resource_limits",
                f"{name} exceed maximum dimensions",
                dimensions=len(bounds),
                max_dimensions=self.max_dimensions
            ))
        
        for i, (lower, upper) in enumerate(bounds):
            # Check that bounds are finite
            if not (jnp.isfinite(lower) and jnp.isfinite(upper)):
                violations.append(self._log_violation(
                    "high",
                    "input_validation",
                    f"{name}[{i}] contains non-finite values",
                    lower=float(lower),
                    upper=float(upper)
                ))
                continue
            
            # Check that lower <= upper
            if lower > upper + self.bounds_tolerance:
                violations.append(self._log_violation(
                    "high",
                    "input_validation",
                    f"{name}[{i}] has lower bound greater than upper bound",
                    lower=float(lower),
                    upper=float(upper)
                ))
            
            # Check for excessively wide bounds
            bound_range = upper - lower
            if bound_range > 1e10:
                violations.append(self._log_violation(
                    "low",
                    "numerical_stability",
                    f"{name}[{i}] has very wide range",
                    range=float(bound_range)
                ))
        
        return violations
    
    def validate_point_in_bounds(self, x: Array, bounds: List[tuple], name: str = "point") -> List[SecurityViolation]:
        """Validate that a point is within bounds.
        
        Args:
            x: Point to validate
            bounds: List of (lower, upper) bound tuples
            name: Name for error reporting
            
        Returns:
            List of security violations found
        """
        violations = []
        
        if len(x) != len(bounds):
            violations.append(self._log_violation(
                "high",
                "input_validation",
                f"{name} dimension mismatch with bounds",
                point_dims=len(x),
                bounds_dims=len(bounds)
            ))
            return violations
        
        for i, ((lower, upper), value) in enumerate(zip(bounds, x)):
            if value < lower - self.bounds_tolerance or value > upper + self.bounds_tolerance:
                violations.append(self._log_violation(
                    "medium",
                    "input_validation",
                    f"{name}[{i}] is outside bounds",
                    value=float(value),
                    lower=float(lower),
                    upper=float(upper)
                ))
        
        return violations
    
    def validate_function_output(self, output: Union[float, Array], name: str = "function_output") -> List[SecurityViolation]:
        """Validate function output.
        
        Args:
            output: Function output to validate
            name: Name for error reporting
            
        Returns:
            List of security violations found
        """
        violations = []
        
        # Convert to array for consistent handling
        if not isinstance(output, Array):
            output = jnp.asarray(output)
        
        # Should be scalar or 1D array
        if output.ndim > 1:
            violations.append(self._log_violation(
                "high",
                "input_validation",
                f"{name} has too many dimensions",
                dimensions=output.ndim
            ))
        
        # Check for non-finite values
        if not jnp.isfinite(output).all():
            violations.append(self._log_violation(
                "high",
                "data_integrity",
                f"{name} contains non-finite values",
                value=float(output) if output.size == 1 else output.tolist()
            ))
        
        return violations
    
    def get_violation_summary(self) -> Dict[str, Any]:
        """Get summary of all violations found."""
        if not self.violations:
            return {"total_violations": 0}
        
        severity_counts = {}
        category_counts = {}
        
        for violation in self.violations:
            severity_counts[violation.severity] = severity_counts.get(violation.severity, 0) + 1
            category_counts[violation.category] = category_counts.get(violation.category, 0) + 1
        
        return {
            "total_violations": len(self.violations),
            "by_severity": severity_counts,
            "by_category": category_counts,
            "has_critical": "critical" in severity_counts,
            "has_high": "high" in severity_counts,
        }
    
    def clear_violations(self):
        """Clear all recorded violations."""
        self.violations.clear()


class SecureFunction:
    """Wrapper that adds security validation to black-box functions."""
    
    def __init__(
        self,
        func: Callable,
        validator: Optional[InputValidator] = None,
        max_evaluations: int = 10000,
        max_eval_time: float = 60.0,
        rate_limit_calls: int = 100,
        rate_limit_window: float = 60.0,
    ):
        """Initialize secure function wrapper.
        
        Args:
            func: Function to wrap
            validator: Input validator instance
            max_evaluations: Maximum number of evaluations allowed
            max_eval_time: Maximum time per evaluation (seconds)
            rate_limit_calls: Maximum calls per time window
            rate_limit_window: Rate limiting time window (seconds)
        """
        self.func = func
        self.validator = validator or InputValidator()
        self.max_evaluations = max_evaluations
        self.max_eval_time = max_eval_time
        self.rate_limit_calls = rate_limit_calls
        self.rate_limit_window = rate_limit_window
        
        # Tracking
        self.evaluation_count = 0
        self.call_history = []
        
    def _check_rate_limit(self) -> bool:
        """Check if rate limit is exceeded."""
        current_time = time.time()
        
        # Remove old entries
        cutoff_time = current_time - self.rate_limit_window
        self.call_history = [t for t in self.call_history if t > cutoff_time]
        
        # Check limit
        if len(self.call_history) >= self.rate_limit_calls:
            return False
        
        # Record this call
        self.call_history.append(current_time)
        return True
    
    def __call__(self, x: Array) -> float:
        """Secure function call with validation."""
        # Check rate limit
        if not self._check_rate_limit():
            raise RuntimeError(f"Rate limit exceeded: {self.rate_limit_calls} calls per {self.rate_limit_window}s")
        
        # Check evaluation limit
        if self.evaluation_count >= self.max_evaluations:
            raise RuntimeError(f"Maximum evaluations exceeded: {self.max_evaluations}")
        
        # Validate input
        violations = self.validator.validate_array(x, "function_input")
        
        # Check for critical violations
        critical_violations = [v for v in violations if v.severity == "critical"]
        if critical_violations:
            raise ValueError(f"Critical security violations: {[v.message for v in critical_violations]}")
        
        # High severity violations are also blocking
        high_violations = [v for v in violations if v.severity == "high"]
        if high_violations:
            raise ValueError(f"High severity security violations: {[v.message for v in high_violations]}")
        
        # Time the evaluation
        start_time = time.time()
        
        try:
            result = self.func(x)
            
            # Check evaluation time
            eval_time = time.time() - start_time
            if eval_time > self.max_eval_time:
                raise TimeoutError(f"Function evaluation timeout: {eval_time:.2f}s > {self.max_eval_time}s")
            
            # Validate output
            output_violations = self.validator.validate_function_output(result)
            high_output_violations = [v for v in output_violations if v.severity in ["critical", "high"]]
            if high_output_violations:
                raise ValueError(f"Function output security violations: {[v.message for v in high_output_violations]}")
            
            self.evaluation_count += 1
            return result
            
        except Exception as e:
            # Log the attempt
            self.evaluation_count += 1
            raise


class SecurityManager:
    """Central security manager for surrogate optimization."""
    
    def __init__(
        self,
        input_validator: Optional[InputValidator] = None,
        enable_function_wrapping: bool = True,
        enable_file_access_control: bool = True,
        allowed_file_patterns: Optional[List[str]] = None,
        log_security_events: bool = True,
    ):
        """Initialize security manager.
        
        Args:
            input_validator: Input validator instance
            enable_function_wrapping: Whether to auto-wrap functions
            enable_file_access_control: Whether to control file access
            allowed_file_patterns: List of allowed file patterns
            log_security_events: Whether to log security events
        """
        self.input_validator = input_validator or InputValidator()
        self.enable_function_wrapping = enable_function_wrapping
        self.enable_file_access_control = enable_file_access_control
        self.log_security_events = log_security_events
        
        if allowed_file_patterns is None:
            # Default safe patterns
            self.allowed_file_patterns = [
                r".*\.json$",
                r".*\.csv$", 
                r".*\.npz$",
                r".*\.pkl$",
                r".*\.h5$",
                r"benchmark_results/.*",
                r"cache/.*",
                r"data/.*",
                r"logs/.*",
            ]
        else:
            self.allowed_file_patterns = allowed_file_patterns
        
        self.security_events = []
    
    def _log_security_event(self, severity: str, event_type: str, message: str, **details):
        """Log a security event."""
        if self.log_security_events:
            event = {
                "timestamp": time.time(),
                "severity": severity,
                "type": event_type,
                "message": message,
                "details": details,
            }
            self.security_events.append(event)
    
    def validate_file_access(self, file_path: Union[str, Path]) -> bool:
        """Validate file access request.
        
        Args:
            file_path: Path to validate
            
        Returns:
            True if access is allowed, False otherwise
        """
        if not self.enable_file_access_control:
            return True
        
        file_path_str = str(file_path)
        
        # Check against allowed patterns
        for pattern in self.allowed_file_patterns:
            if re.match(pattern, file_path_str):
                return True
        
        self._log_security_event(
            "medium",
            "file_access_denied",
            f"File access denied: {file_path}",
            path=file_path_str,
            allowed_patterns=self.allowed_file_patterns
        )
        
        return False
    
    def secure_function(self, func: Callable, **kwargs) -> SecureFunction:
        """Create a secure wrapper for a function.
        
        Args:
            func: Function to wrap
            **kwargs: Additional arguments for SecureFunction
            
        Returns:
            Secure function wrapper
        """
        return SecureFunction(
            func,
            validator=self.input_validator,
            **kwargs
        )
    
    def validate_dataset_integrity(self, dataset) -> Dict[str, Any]:
        """Validate dataset integrity and security.
        
        Args:
            dataset: Dataset to validate
            
        Returns:
            Validation report
        """
        violations = []
        
        # Validate inputs
        x_violations = self.input_validator.validate_array(dataset.X, "dataset.X")
        violations.extend(x_violations)
        
        # Validate outputs
        y_violations = self.input_validator.validate_array(dataset.y, "dataset.y")
        violations.extend(y_violations)
        
        # Validate gradients if present
        if dataset.gradients is not None:
            grad_violations = self.input_validator.validate_array(dataset.gradients, "dataset.gradients")
            violations.extend(grad_violations)
        
        # Check data consistency
        if dataset.X.shape[0] != dataset.y.shape[0]:
            violations.append(SecurityViolation(
                "high",
                "data_integrity",
                "Input-output dimension mismatch",
                {"X_samples": dataset.X.shape[0], "y_samples": dataset.y.shape[0]},
                time.time()
            ))
        
        # Create summary
        severity_counts = {}
        for violation in violations:
            severity_counts[violation.severity] = severity_counts.get(violation.severity, 0) + 1
        
        return {
            "is_secure": len([v for v in violations if v.severity in ["critical", "high"]]) == 0,
            "violations": violations,
            "severity_counts": severity_counts,
            "total_violations": len(violations),
        }
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get overall security summary."""
        return {
            "total_events": len(self.security_events),
            "total_violations": len(self.input_validator.violations),
            "validation_summary": self.input_validator.get_violation_summary(),
            "recent_events": self.security_events[-10:] if self.security_events else [],
        }


# Global security manager instance
_security_manager = None

def get_security_manager(**kwargs) -> SecurityManager:
    """Get global security manager instance."""
    global _security_manager
    if _security_manager is None:
        _security_manager = SecurityManager(**kwargs)
    return _security_manager


def secure_file_operation(func):
    """Decorator to add security validation to file operations."""
    def wrapper(*args, **kwargs):
        # Find file path in arguments
        file_path = None
        for arg in args:
            if isinstance(arg, (str, Path)):
                file_path = arg
                break
        
        if file_path is None:
            for key, value in kwargs.items():
                if 'path' in key.lower() or 'file' in key.lower():
                    if isinstance(value, (str, Path)):
                        file_path = value
                        break
        
        # Validate file access
        if file_path is not None:
            security_manager = get_security_manager()
            if not security_manager.validate_file_access(file_path):
                raise PermissionError(f"File access denied: {file_path}")
        
        return func(*args, **kwargs)
    
    return wrapper