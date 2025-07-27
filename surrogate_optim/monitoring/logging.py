"""Logging configuration and utilities."""

import logging
import logging.config
import sys
import os
import time
from typing import Dict, Any, Optional
from pathlib import Path
import json

from loguru import logger as loguru_logger


def setup_logging(
    level: str = "INFO",
    format_type: str = "standard",
    log_file: Optional[str] = None,
    json_logs: bool = False,
    correlation_id: bool = True,
) -> logging.Logger:
    """Setup logging configuration for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Format type (standard, detailed, minimal)
        log_file: Optional log file path
        json_logs: Whether to use JSON format for logs
        correlation_id: Whether to include correlation IDs
        
    Returns:
        logging.Logger: Configured logger instance
    """
    
    # Clear any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Configure logging level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    if json_logs:
        # Use structured JSON logging
        config = _get_json_logging_config(numeric_level, log_file)
    else:
        # Use standard text logging
        config = _get_standard_logging_config(numeric_level, format_type, log_file)
    
    logging.config.dictConfig(config)
    
    # Configure loguru for enhanced logging features
    _setup_loguru(level, log_file, json_logs)
    
    return logging.getLogger("surrogate_optim")


def _get_standard_logging_config(
    level: int, 
    format_type: str, 
    log_file: Optional[str] = None
) -> Dict[str, Any]:
    """Get standard logging configuration."""
    
    formats = {
        "minimal": "%(levelname)s: %(message)s",
        "standard": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "detailed": (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "%(filename)s:%(lineno)d - %(funcName)s - %(message)s"
        ),
    }
    
    format_string = formats.get(format_type, formats["standard"])
    
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": format_string,
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": level,
                "formatter": "standard",
                "stream": sys.stdout,
            },
        },
        "root": {
            "level": level,
            "handlers": ["console"],
        },
        "loggers": {
            "surrogate_optim": {
                "level": level,
                "handlers": ["console"],
                "propagate": False,
            },
            "jax": {
                "level": "WARNING",  # Reduce JAX verbosity
            },
            "matplotlib": {
                "level": "WARNING",  # Reduce matplotlib verbosity
            },
        },
    }
    
    # Add file handler if log file is specified
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": level,
            "formatter": "standard",
            "filename": log_file,
            "maxBytes": 10 * 1024 * 1024,  # 10MB
            "backupCount": 5,
        }
        
        config["root"]["handlers"].append("file")
        config["loggers"]["surrogate_optim"]["handlers"].append("file")
    
    return config


def _get_json_logging_config(
    level: int, 
    log_file: Optional[str] = None
) -> Dict[str, Any]:
    """Get JSON logging configuration."""
    
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "class": "pythonjsonlogger.jsonlogger.JsonFormatter",
                "format": (
                    "%(asctime)s %(name)s %(levelname)s "
                    "%(filename)s %(lineno)d %(funcName)s %(message)s"
                ),
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": level,
                "formatter": "json",
                "stream": sys.stdout,
            },
        },
        "root": {
            "level": level,
            "handlers": ["console"],
        },
        "loggers": {
            "surrogate_optim": {
                "level": level,
                "handlers": ["console"],
                "propagate": False,
            },
        },
    }
    
    # Add file handler if log file is specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": level,
            "formatter": "json",
            "filename": log_file,
            "maxBytes": 10 * 1024 * 1024,  # 10MB
            "backupCount": 5,
        }
        
        config["root"]["handlers"].append("file")
        config["loggers"]["surrogate_optim"]["handlers"].append("file")
    
    return config


def _setup_loguru(
    level: str, 
    log_file: Optional[str] = None, 
    json_logs: bool = False
):
    """Setup loguru for enhanced logging."""
    
    # Remove default handler
    loguru_logger.remove()
    
    # Console handler
    if json_logs:
        console_format = (
            '{"time": "{time:YYYY-MM-DD HH:mm:ss.SSS}", '
            '"level": "{level}", '
            '"logger": "{name}", '
            '"file": "{file.name}", '
            '"line": {line}, '
            '"function": "{function}", '
            '"message": "{message}"}'
        )
    else:
        console_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )
    
    loguru_logger.add(
        sys.stdout,
        format=console_format,
        level=level.upper(),
        colorize=not json_logs,
    )
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        if json_logs:
            file_format = console_format  # Same JSON format
        else:
            file_format = (
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
                "{level: <8} | "
                "{name}:{function}:{line} - "
                "{message}"
            )
        
        loguru_logger.add(
            log_file,
            format=file_format,
            level=level.upper(),
            rotation="10 MB",
            retention="1 week",
            compression="gz",
        )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name.
    
    Args:
        name: Logger name
        
    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(f"surrogate_optim.{name}")


class StructuredLogger:
    """Enhanced logger with structured logging capabilities."""
    
    def __init__(self, name: str):
        self.logger = get_logger(name)
        self.correlation_id = None
    
    def set_correlation_id(self, correlation_id: str):
        """Set correlation ID for this logger."""
        self.correlation_id = correlation_id
    
    def _log_with_context(self, level: int, message: str, **kwargs):
        """Log message with additional context."""
        extra = kwargs.copy()
        
        if self.correlation_id:
            extra["correlation_id"] = self.correlation_id
        
        # Add caller information
        import inspect
        frame = inspect.currentframe().f_back.f_back
        extra["caller_filename"] = frame.f_code.co_filename
        extra["caller_lineno"] = frame.f_lineno
        extra["caller_function"] = frame.f_code.co_name
        
        self.logger.log(level, message, extra=extra)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log_with_context(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log_with_context(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._log_with_context(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self._log_with_context(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self._log_with_context(logging.CRITICAL, message, **kwargs)
    
    def exception(self, message: str, **kwargs):
        """Log exception with traceback."""
        kwargs["exc_info"] = True
        self._log_with_context(logging.ERROR, message, **kwargs)


class PerformanceLogger:
    """Logger for performance metrics and timing."""
    
    def __init__(self, name: str):
        self.logger = get_logger(f"{name}.performance")
    
    def log_timing(self, operation: str, duration: float, **metadata):
        """Log timing information."""
        self.logger.info(
            f"Operation '{operation}' completed",
            extra={
                "operation": operation,
                "duration_seconds": duration,
                "duration_ms": duration * 1000,
                **metadata,
            }
        )
    
    def log_memory_usage(self, operation: str, memory_mb: float, **metadata):
        """Log memory usage information."""
        self.logger.info(
            f"Memory usage for '{operation}': {memory_mb:.2f} MB",
            extra={
                "operation": operation,
                "memory_mb": memory_mb,
                **metadata,
            }
        )
    
    def log_model_metrics(
        self, 
        model_name: str, 
        loss: float, 
        accuracy: Optional[float] = None,
        **metrics
    ):
        """Log model training/evaluation metrics."""
        extra = {
            "model_name": model_name,
            "loss": loss,
            **metrics,
        }
        
        if accuracy is not None:
            extra["accuracy"] = accuracy
        
        self.logger.info(
            f"Model '{model_name}' metrics: loss={loss:.6f}",
            extra=extra,
        )


class AuditLogger:
    """Logger for audit trails and security events."""
    
    def __init__(self, name: str):
        self.logger = get_logger(f"{name}.audit")
    
    def log_model_training_start(self, model_type: str, dataset_size: int, **params):
        """Log start of model training."""
        self.logger.info(
            f"Started training {model_type} model",
            extra={
                "event_type": "model_training_start",
                "model_type": model_type,
                "dataset_size": dataset_size,
                "parameters": params,
                "timestamp": time.time(),
            }
        )
    
    def log_model_training_complete(
        self, 
        model_type: str, 
        duration: float, 
        final_loss: float
    ):
        """Log completion of model training."""
        self.logger.info(
            f"Completed training {model_type} model",
            extra={
                "event_type": "model_training_complete",
                "model_type": model_type,
                "duration_seconds": duration,
                "final_loss": final_loss,
                "timestamp": time.time(),
            }
        )
    
    def log_optimization_run(
        self, 
        optimizer_type: str, 
        function_evals: int, 
        final_value: float
    ):
        """Log optimization run."""
        self.logger.info(
            f"Optimization run completed",
            extra={
                "event_type": "optimization_complete",
                "optimizer_type": optimizer_type,
                "function_evaluations": function_evals,
                "final_objective_value": final_value,
                "timestamp": time.time(),
            }
        )


# Context manager for operation logging
class log_operation:
    """Context manager for logging operations with timing."""
    
    def __init__(
        self, 
        logger: logging.Logger, 
        operation: str, 
        level: int = logging.INFO,
        **metadata
    ):
        self.logger = logger
        self.operation = operation
        self.level = level
        self.metadata = metadata
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.log(
            self.level,
            f"Starting operation: {self.operation}",
            extra={"operation": self.operation, **self.metadata}
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        if exc_type is None:
            self.logger.log(
                self.level,
                f"Completed operation: {self.operation} ({duration:.3f}s)",
                extra={
                    "operation": self.operation,
                    "duration_seconds": duration,
                    "success": True,
                    **self.metadata,
                }
            )
        else:
            self.logger.error(
                f"Failed operation: {self.operation} ({duration:.3f}s)",
                extra={
                    "operation": self.operation,
                    "duration_seconds": duration,
                    "success": False,
                    "error_type": exc_type.__name__,
                    "error_message": str(exc_val),
                    **self.metadata,
                },
                exc_info=True,
            )


# Initialize default logging
_default_logger = None

def get_default_logger() -> logging.Logger:
    """Get the default application logger."""
    global _default_logger
    if _default_logger is None:
        # Setup with environment variables or defaults
        level = os.getenv("LOG_LEVEL", "INFO")
        format_type = os.getenv("LOG_FORMAT", "standard")
        log_file = os.getenv("LOG_FILE")
        json_logs = os.getenv("JSON_LOGS", "false").lower() == "true"
        
        _default_logger = setup_logging(
            level=level,
            format_type=format_type,
            log_file=log_file,
            json_logs=json_logs,
        )
    
    return _default_logger