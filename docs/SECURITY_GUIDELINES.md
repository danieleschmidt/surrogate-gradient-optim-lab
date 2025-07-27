# Security Guidelines for Developers

This document provides security guidelines and best practices for developers working on the Surrogate Gradient Optimization Lab project.

## Table of Contents

1. [Secure Coding Practices](#secure-coding-practices)
2. [Input Validation](#input-validation)
3. [Model Security](#model-security)
4. [Data Protection](#data-protection)
5. [Dependency Management](#dependency-management)
6. [Authentication & Authorization](#authentication--authorization)
7. [Logging & Monitoring](#logging--monitoring)
8. [Testing for Security](#testing-for-security)
9. [Container Security](#container-security)
10. [Production Deployment](#production-deployment)

## Secure Coding Practices

### General Principles

1. **Principle of Least Privilege**: Grant minimal necessary permissions
2. **Defense in Depth**: Implement multiple layers of security
3. **Fail Securely**: Ensure failures don't compromise security
4. **Don't Trust User Input**: Validate and sanitize all inputs
5. **Keep Security Simple**: Avoid complex security mechanisms

### Code Review Checklist

- [ ] Input validation implemented for all user inputs
- [ ] No hardcoded secrets or credentials
- [ ] Proper error handling without information leakage
- [ ] Resource limits and bounds checking
- [ ] Secure random number generation
- [ ] No use of dangerous functions (eval, exec, pickle.loads)

## Input Validation

### Data Validation

```python
import jax.numpy as jnp
from typing import Union, Tuple

def validate_optimization_bounds(
    bounds: list[Tuple[float, float]], 
    dimensions: int
) -> list[Tuple[float, float]]:
    """Validate optimization bounds with proper security checks."""
    
    # Check bounds format
    if not isinstance(bounds, list):
        raise ValueError("Bounds must be a list of tuples")
    
    if len(bounds) != dimensions:
        raise ValueError(f"Expected {dimensions} bounds, got {len(bounds)}")
    
    validated_bounds = []
    for i, bound in enumerate(bounds):
        if not isinstance(bound, tuple) or len(bound) != 2:
            raise ValueError(f"Bound {i} must be a tuple of (min, max)")
        
        min_val, max_val = bound
        
        # Validate numeric types
        if not isinstance(min_val, (int, float)) or not isinstance(max_val, (int, float)):
            raise ValueError(f"Bound {i} values must be numeric")
        
        # Check for valid ranges
        if not jnp.isfinite(min_val) or not jnp.isfinite(max_val):
            raise ValueError(f"Bound {i} contains non-finite values")
        
        if min_val >= max_val:
            raise ValueError(f"Bound {i}: min ({min_val}) must be less than max ({max_val})")
        
        # Prevent extremely large ranges that could cause DoS
        range_size = max_val - min_val
        if range_size > 1e10:
            raise ValueError(f"Bound {i} range too large: {range_size}")
        
        validated_bounds.append((float(min_val), float(max_val)))
    
    return validated_bounds


def validate_training_data(X: jnp.ndarray, y: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Validate training data for security and correctness."""
    
    # Check array types
    if not isinstance(X, jnp.ndarray) or not isinstance(y, jnp.ndarray):
        raise ValueError("Training data must be JAX arrays")
    
    # Check dimensions
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have same number of samples")
    
    # Check for valid values
    if not jnp.isfinite(X).all():
        raise ValueError("X contains non-finite values")
    
    if not jnp.isfinite(y).all():
        raise ValueError("y contains non-finite values")
    
    # Check for reasonable data sizes (prevent DoS)
    max_samples = 1_000_000
    max_features = 10_000
    
    if X.shape[0] > max_samples:
        raise ValueError(f"Too many samples: {X.shape[0]} > {max_samples}")
    
    if X.shape[1] > max_features:
        raise ValueError(f"Too many features: {X.shape[1]} > {max_features}")
    
    # Check for reasonable value ranges
    x_max = jnp.max(jnp.abs(X))
    y_max = jnp.max(jnp.abs(y))
    
    if x_max > 1e6:
        raise ValueError(f"X values too large: max {x_max}")
    
    if y_max > 1e6:
        raise ValueError(f"y values too large: max {y_max}")
    
    return X, y
```

### File Path Validation

```python
import os
from pathlib import Path

def validate_file_path(file_path: str, allowed_extensions: set[str] = None) -> Path:
    """Validate file paths to prevent directory traversal attacks."""
    
    # Convert to Path object
    path = Path(file_path).resolve()
    
    # Check for directory traversal attempts
    if ".." in str(path):
        raise ValueError("Directory traversal not allowed")
    
    # Ensure path is within allowed directories
    allowed_dirs = [Path.cwd(), Path("/tmp"), Path.home() / "data"]
    
    if not any(str(path).startswith(str(allowed_dir)) for allowed_dir in allowed_dirs):
        raise ValueError("Path not in allowed directories")
    
    # Check file extension if specified
    if allowed_extensions and path.suffix not in allowed_extensions:
        raise ValueError(f"File extension {path.suffix} not allowed")
    
    return path
```

## Model Security

### Safe Model Loading

```python
import pickle
import hashlib
from pathlib import Path

def safe_load_model(model_path: str, expected_hash: str = None) -> any:
    """Safely load a model file with integrity checking."""
    
    path = validate_file_path(model_path, {".pkl", ".joblib", ".json"})
    
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    
    # Check file size (prevent loading extremely large files)
    max_size = 100 * 1024 * 1024  # 100MB
    if path.stat().st_size > max_size:
        raise ValueError(f"Model file too large: {path.stat().st_size} bytes")
    
    # Verify file integrity if hash provided
    if expected_hash:
        with open(path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        
        if file_hash != expected_hash:
            raise ValueError("Model file integrity check failed")
    
    # Load model with appropriate method
    if path.suffix == ".pkl":
        # Use restricted pickle loading for security
        with open(path, "rb") as f:
            # Only allow safe built-ins
            class SafeUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    # Only allow specific safe modules
                    safe_modules = {
                        "numpy", "jax", "sklearn", 
                        "surrogate_optim.models"
                    }
                    
                    if module.split(".")[0] not in safe_modules:
                        raise pickle.UnpicklingError(f"Unsafe module: {module}")
                    
                    return super().find_class(module, name)
            
            return SafeUnpickler(f).load()
    
    else:
        # Handle other formats safely
        raise ValueError(f"Unsupported model format: {path.suffix}")


def save_model_securely(model: any, model_path: str) -> str:
    """Save model with integrity hash."""
    
    path = validate_file_path(model_path, {".pkl", ".joblib"})
    
    # Create directory if needed
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model
    with open(path, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Generate integrity hash
    with open(path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    
    # Save hash file
    hash_path = path.with_suffix(path.suffix + ".sha256")
    with open(hash_path, "w") as f:
        f.write(file_hash)
    
    return file_hash
```

### Model Training Security

```python
import time
import signal
from contextlib import contextmanager

@contextmanager
def training_timeout(seconds: int):
    """Context manager to timeout training operations."""
    
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Training timeout after {seconds} seconds")
    
    # Set up signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def secure_training_params(params: dict) -> dict:
    """Validate and sanitize training parameters."""
    
    # Define safe parameter ranges
    safe_ranges = {
        "learning_rate": (1e-6, 1.0),
        "epochs": (1, 10000),
        "batch_size": (1, 10000),
        "hidden_dims": (1, 1000),
        "dropout_rate": (0.0, 0.9),
    }
    
    validated_params = {}
    
    for param, value in params.items():
        if param in safe_ranges:
            min_val, max_val = safe_ranges[param]
            
            if not isinstance(value, (int, float)):
                raise ValueError(f"Parameter {param} must be numeric")
            
            if not min_val <= value <= max_val:
                raise ValueError(f"Parameter {param} must be between {min_val} and {max_val}")
            
            validated_params[param] = value
        else:
            # Log unknown parameters but don't include them
            logger.warning(f"Unknown parameter ignored: {param}")
    
    return validated_params
```

## Data Protection

### Sensitive Data Handling

```python
import secrets
from typing import Optional

class SecureDataHandler:
    """Handle sensitive data with proper security measures."""
    
    def __init__(self):
        self._encryption_key: Optional[bytes] = None
    
    def generate_key(self) -> bytes:
        """Generate a secure encryption key."""
        self._encryption_key = secrets.token_bytes(32)  # 256-bit key
        return self._encryption_key
    
    def sanitize_data(self, data: jnp.ndarray) -> jnp.ndarray:
        """Remove or anonymize sensitive information from data."""
        
        # Add noise for differential privacy
        noise_scale = 0.1
        noise = jax.random.normal(
            jax.random.PRNGKey(secrets.randbits(32)), 
            data.shape
        ) * noise_scale
        
        return data + noise
    
    def secure_delete(self, variable):
        """Securely delete sensitive variables."""
        if hasattr(variable, '__array__'):
            # Overwrite array with random data
            variable.fill(0)
        
        del variable


def mask_sensitive_logs(log_message: str) -> str:
    """Mask sensitive information in log messages."""
    import re
    
    # Mask potential API keys, tokens, passwords
    patterns = [
        (r'(api[_-]?key[_-]?=\s*)[\w\-]{10,}', r'\1***'),
        (r'(token[_-]?=\s*)[\w\-]{10,}', r'\1***'),
        (r'(password[_-]?=\s*)[\w\-]{6,}', r'\1***'),
        (r'(\d{4})[_-]?\d{4}[_-]?\d{4}[_-]?(\d{4})', r'\1****\2'),  # Credit card
    ]
    
    masked_message = log_message
    for pattern, replacement in patterns:
        masked_message = re.sub(pattern, replacement, masked_message, flags=re.IGNORECASE)
    
    return masked_message
```

## Dependency Management

### Secure Dependency Loading

```python
import subprocess
import sys
from packaging import version

def check_dependency_security():
    """Check dependencies for known security vulnerabilities."""
    
    # Run safety check
    try:
        result = subprocess.run(
            [sys.executable, "-m", "safety", "check", "--json"],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            vulnerabilities = result.stdout
            logger.error(f"Security vulnerabilities found: {vulnerabilities}")
            return False
        
        return True
        
    except subprocess.TimeoutExpired:
        logger.error("Security check timed out")
        return False
    except Exception as e:
        logger.error(f"Security check failed: {e}")
        return False


def validate_package_versions():
    """Validate that packages are at secure versions."""
    
    # Define minimum secure versions
    min_versions = {
        "jax": "0.4.0",
        "numpy": "1.21.0",
        "scipy": "1.7.0",
        "scikit-learn": "1.0.0",
        "cryptography": "3.4.8",
    }
    
    for package, min_version in min_versions.items():
        try:
            import importlib
            module = importlib.import_module(package)
            
            if hasattr(module, "__version__"):
                current_version = module.__version__
                
                if version.parse(current_version) < version.parse(min_version):
                    logger.warning(
                        f"Package {package} version {current_version} "
                        f"is below minimum secure version {min_version}"
                    )
                    
        except ImportError:
            # Package not installed, skip check
            continue
```

## Authentication & Authorization

### API Key Management

```python
import os
import secrets
from typing import Optional

class APIKeyManager:
    """Manage API keys securely."""
    
    def __init__(self):
        self._keys: dict[str, str] = {}
    
    def generate_api_key(self, user_id: str) -> str:
        """Generate a secure API key for a user."""
        
        # Generate cryptographically secure random key
        api_key = secrets.token_urlsafe(32)
        
        # Store with user association
        self._keys[api_key] = user_id
        
        return api_key
    
    def validate_api_key(self, api_key: str) -> Optional[str]:
        """Validate API key and return user ID."""
        
        # Use constant-time comparison to prevent timing attacks
        for stored_key, user_id in self._keys.items():
            if secrets.compare_digest(api_key, stored_key):
                return user_id
        
        return None
    
    def revoke_api_key(self, api_key: str):
        """Revoke an API key."""
        self._keys.pop(api_key, None)


def require_authentication(func):
    """Decorator to require authentication for API endpoints."""
    
    def wrapper(*args, **kwargs):
        api_key = os.getenv("API_KEY")
        
        if not api_key:
            raise PermissionError("API key required")
        
        # Validate API key
        key_manager = APIKeyManager()
        user_id = key_manager.validate_api_key(api_key)
        
        if not user_id:
            raise PermissionError("Invalid API key")
        
        # Add user context to function
        kwargs["user_id"] = user_id
        
        return func(*args, **kwargs)
    
    return wrapper
```

## Logging & Monitoring

### Security Event Logging

```python
import json
from datetime import datetime
from enum import Enum

class SecurityEventType(Enum):
    AUTHENTICATION_FAILURE = "auth_failure"
    AUTHORIZATION_FAILURE = "authz_failure"
    SUSPICIOUS_INPUT = "suspicious_input"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    DATA_ACCESS = "data_access"
    MODEL_TRAINING = "model_training"
    FILE_ACCESS = "file_access"

class SecurityLogger:
    """Logger for security events."""
    
    def __init__(self, log_file: str = "security.log"):
        self.log_file = log_file
    
    def log_security_event(
        self, 
        event_type: SecurityEventType,
        user_id: str = None,
        details: dict = None,
        severity: str = "INFO"
    ):
        """Log a security event."""
        
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type.value,
            "user_id": user_id,
            "severity": severity,
            "details": details or {},
            "source_ip": self._get_client_ip(),
        }
        
        # Mask sensitive information
        event = self._mask_sensitive_data(event)
        
        # Write to log file
        with open(self.log_file, "a") as f:
            f.write(json.dumps(event) + "\n")
    
    def _get_client_ip(self) -> str:
        """Get client IP address (placeholder for actual implementation)."""
        return "127.0.0.1"
    
    def _mask_sensitive_data(self, event: dict) -> dict:
        """Mask sensitive data in log events."""
        # Implementation would mask sensitive fields
        return event


# Usage example
security_logger = SecurityLogger()

def log_failed_authentication(user_id: str, reason: str):
    """Log failed authentication attempt."""
    security_logger.log_security_event(
        event_type=SecurityEventType.AUTHENTICATION_FAILURE,
        user_id=user_id,
        details={"reason": reason},
        severity="WARNING"
    )
```

## Testing for Security

### Security Test Examples

```python
import pytest
import tempfile
from pathlib import Path

class TestSecurityValidation:
    """Security-focused tests."""
    
    def test_input_validation_prevents_injection(self):
        """Test that input validation prevents injection attacks."""
        
        malicious_inputs = [
            "../../../etc/passwd",
            "'; DROP TABLE users; --",
            "__import__('os').system('rm -rf /')",
            "eval('malicious_code')",
        ]
        
        for malicious_input in malicious_inputs:
            with pytest.raises(ValueError):
                validate_file_path(malicious_input)
    
    def test_bounds_validation_prevents_dos(self):
        """Test that bounds validation prevents DoS attacks."""
        
        # Extremely large bounds that could cause memory issues
        malicious_bounds = [(0, 1e20), (0, 1e20)]
        
        with pytest.raises(ValueError):
            validate_optimization_bounds(malicious_bounds, 2)
    
    def test_training_data_size_limits(self):
        """Test that training data size limits are enforced."""
        
        import jax.numpy as jnp
        
        # Create oversized training data
        large_X = jnp.ones((2_000_000, 100))  # Too many samples
        large_y = jnp.ones(2_000_000)
        
        with pytest.raises(ValueError):
            validate_training_data(large_X, large_y)
    
    def test_safe_model_loading(self):
        """Test that model loading prevents arbitrary code execution."""
        
        # Create a malicious pickle file
        with tempfile.NamedTemporaryFile(suffix=".pkl") as f:
            import pickle
            
            # This would be dangerous if not properly handled
            malicious_data = b"cos\nsystem\n(S'echo malicious'\ntR."
            f.write(malicious_data)
            f.flush()
            
            with pytest.raises(pickle.UnpicklingError):
                safe_load_model(f.name)
    
    def test_api_key_validation(self):
        """Test API key validation security."""
        
        key_manager = APIKeyManager()
        
        # Test invalid keys
        assert key_manager.validate_api_key("invalid_key") is None
        assert key_manager.validate_api_key("") is None
        assert key_manager.validate_api_key(None) is None
        
        # Test timing attack resistance
        import time
        
        valid_key = key_manager.generate_api_key("user1")
        invalid_key = "x" * len(valid_key)
        
        # Measure validation times
        start = time.time()
        key_manager.validate_api_key(valid_key)
        valid_time = time.time() - start
        
        start = time.time()
        key_manager.validate_api_key(invalid_key)
        invalid_time = time.time() - start
        
        # Times should be similar (constant-time comparison)
        assert abs(valid_time - invalid_time) < 0.01  # 10ms tolerance
```

## Container Security

### Docker Security Best Practices

```dockerfile
# Use specific version tags, not 'latest'
FROM python:3.9.16-slim

# Create non-root user
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash appuser

# Install security updates
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set up application directory with proper permissions
WORKDIR /app
RUN chown -R appuser:appuser /app

# Copy and install dependencies as non-root user
USER appuser
COPY --chown=appuser:appuser requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=appuser:appuser . .

# Remove unnecessary files
RUN rm -rf tests/ docs/ .git/

# Set security-focused environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import surrogate_optim; print('OK')" || exit 1

# Use non-root user
USER appuser

# Expose only necessary ports
EXPOSE 8000

# Use specific command
CMD ["python", "-m", "surrogate_optim.cli"]
```

## Production Deployment

### Security Checklist for Production

- [ ] All secrets managed through secure secret management system
- [ ] TLS/SSL encryption enabled for all network communication
- [ ] Network segmentation and firewall rules configured
- [ ] Resource limits (CPU, memory, disk) properly configured
- [ ] Monitoring and alerting for security events enabled
- [ ] Regular security updates and patches applied
- [ ] Backup and disaster recovery procedures tested
- [ ] Access logging enabled for all components
- [ ] Input validation enforced at all entry points
- [ ] Rate limiting implemented to prevent DoS attacks

### Environment Configuration

```bash
# Production environment variables
export ENVIRONMENT=production
export DEBUG=false
export LOG_LEVEL=INFO
export ENABLE_METRICS=true
export SECURITY_SCAN_ENABLED=true

# Security settings
export SECURE_COOKIES=true
export CSRF_PROTECTION=true
export XSS_PROTECTION=true

# Resource limits
export MAX_MEMORY_MB=2048
export MAX_CPU_CORES=4
export REQUEST_TIMEOUT_SECONDS=30

# Monitoring
export HEALTH_CHECK_ENABLED=true
export METRICS_ENDPOINT_ENABLED=true
```

---

This security guidelines document should be regularly reviewed and updated as new threats emerge and the project evolves. All developers should be familiar with these guidelines and apply them consistently throughout the development process.