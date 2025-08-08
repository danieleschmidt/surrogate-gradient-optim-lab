"""Enhanced logging and monitoring system for surrogate optimization."""

import logging
import sys
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union
from contextlib import contextmanager
from dataclasses import dataclass, asdict

import jax.numpy as jnp
from jax import Array


@dataclass
class OptimizationEvent:
    """Event data structure for optimization logging."""
    timestamp: str
    event_type: str
    component: str
    level: str
    message: str
    metadata: Dict[str, Any]
    duration: Optional[float] = None
    memory_usage: Optional[float] = None


class StructuredLogger:
    """Structured logger for surrogate optimization components."""
    
    def __init__(
        self,
        name: str,
        log_file: Optional[Union[str, Path]] = None,
        log_level: str = "INFO",
        structured_output: bool = True,
        include_performance_metrics: bool = True,
    ):
        """Initialize structured logger.
        
        Args:
            name: Logger name
            log_file: Optional log file path
            log_level: Logging level
            structured_output: Whether to use structured JSON output
            include_performance_metrics: Whether to track performance metrics
        """
        self.name = name
        self.structured_output = structured_output
        self.include_performance_metrics = include_performance_metrics
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatters
        if structured_output:
            formatter = JsonFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # Event storage for analysis
        self.events = []
    
    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        if not self.include_performance_metrics:
            return {}
        
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            
            return {
                "memory_rss_mb": memory_info.rss / 1024 / 1024,
                "memory_vms_mb": memory_info.vms / 1024 / 1024,
                "cpu_percent": process.cpu_percent(),
                "num_threads": process.num_threads(),
            }
        except ImportError:
            return {}
        except Exception:
            return {}
    
    def _log_event(
        self,
        level: str,
        message: str,
        event_type: str,
        component: str,
        metadata: Optional[Dict] = None,
        duration: Optional[float] = None,
    ):
        """Log a structured event."""
        metadata = metadata or {}
        
        # Add performance metrics
        if self.include_performance_metrics:
            metadata.update(self._get_performance_metrics())
        
        # Create event
        event = OptimizationEvent(
            timestamp=datetime.now().isoformat(),
            event_type=event_type,
            component=component,
            level=level,
            message=message,
            metadata=metadata,
            duration=duration,
        )
        
        # Store event
        self.events.append(event)
        
        # Log event
        if self.structured_output:
            log_data = asdict(event)
            getattr(self.logger, level.lower())(json.dumps(log_data))
        else:
            extra_info = f" [{event_type}:{component}]"
            if duration:
                extra_info += f" (took {duration:.3f}s)"
            getattr(self.logger, level.lower())(message + extra_info)
    
    def info(self, message: str, event_type: str = "info", component: str = "general", **metadata):
        """Log info level event."""
        self._log_event("INFO", message, event_type, component, metadata)
    
    def warning(self, message: str, event_type: str = "warning", component: str = "general", **metadata):
        """Log warning level event."""
        self._log_event("WARNING", message, event_type, component, metadata)
    
    def error(self, message: str, event_type: str = "error", component: str = "general", **metadata):
        """Log error level event."""
        self._log_event("ERROR", message, event_type, component, metadata)
    
    def debug(self, message: str, event_type: str = "debug", component: str = "general", **metadata):
        """Log debug level event."""
        self._log_event("DEBUG", message, event_type, component, metadata)
    
    @contextmanager
    def timed_operation(
        self,
        operation_name: str,
        component: str = "general",
        log_start: bool = True,
        log_end: bool = True,
        **metadata
    ):
        """Context manager for timing operations."""
        if log_start:
            self.info(f"Starting {operation_name}", "operation_start", component, **metadata)
        
        start_time = time.time()
        try:
            yield self
        finally:
            duration = time.time() - start_time
            if log_end:
                self.info(
                    f"Completed {operation_name}",
                    "operation_end",
                    component,
                    duration=duration,
                    **metadata
                )
    
    def log_dataset_info(self, dataset, component: str = "data"):
        """Log information about a dataset."""
        metadata = {
            "n_samples": dataset.n_samples,
            "n_dimensions": dataset.n_dims,
            "has_gradients": dataset.gradients is not None,
            "output_range": float(jnp.max(dataset.y) - jnp.min(dataset.y)),
            "output_mean": float(jnp.mean(dataset.y)),
            "output_std": float(jnp.std(dataset.y)),
        }
        
        self.info(f"Dataset loaded with {dataset.n_samples} samples", "dataset_info", component, **metadata)
    
    def log_model_info(self, model, component: str = "model"):
        """Log information about a surrogate model."""
        metadata = {
            "model_type": model.__class__.__name__,
            "model_module": model.__class__.__module__,
        }
        
        # Add model-specific information if available
        if hasattr(model, 'get_params'):
            try:
                model_params = model.get_params()
                metadata.update(model_params)
            except:
                pass
        
        self.info(f"Model initialized: {model.__class__.__name__}", "model_init", component, **metadata)
    
    def log_optimization_result(self, result, component: str = "optimization"):
        """Log optimization result."""
        metadata = {
            "success": result.success,
            "function_value": float(result.fun),
            "n_iterations": result.nit,
            "n_evaluations": result.nfev,
            "optimum_found": result.x.tolist(),
            "message": result.message,
        }
        
        if hasattr(result, 'convergence_history') and result.convergence_history:
            metadata["final_convergence_value"] = float(result.convergence_history[-1])
            metadata["convergence_improvement"] = float(
                result.convergence_history[0] - result.convergence_history[-1]
            )
        
        event_type = "optimization_success" if result.success else "optimization_failure"
        level = "INFO" if result.success else "WARNING"
        
        message = f"Optimization {'succeeded' if result.success else 'failed'}: f = {result.fun:.6f}"
        self._log_event(level, message, event_type, component, metadata)
    
    def log_validation_metrics(self, metrics: Dict[str, float], component: str = "validation"):
        """Log model validation metrics."""
        formatted_metrics = {k: float(v) for k, v in metrics.items()}
        
        self.info(
            f"Model validation completed: {len(metrics)} metrics",
            "validation_complete",
            component,
            **formatted_metrics
        )
    
    def get_events_summary(self) -> Dict[str, Any]:
        """Get summary of logged events."""
        if not self.events:
            return {"total_events": 0}
        
        # Count by type and level
        type_counts = {}
        level_counts = {}
        component_counts = {}
        
        total_duration = 0
        operations_with_duration = 0
        
        for event in self.events:
            type_counts[event.event_type] = type_counts.get(event.event_type, 0) + 1
            level_counts[event.level] = level_counts.get(event.level, 0) + 1
            component_counts[event.component] = component_counts.get(event.component, 0) + 1
            
            if event.duration:
                total_duration += event.duration
                operations_with_duration += 1
        
        return {
            "total_events": len(self.events),
            "event_types": type_counts,
            "levels": level_counts,
            "components": component_counts,
            "total_duration": total_duration,
            "average_operation_duration": total_duration / operations_with_duration if operations_with_duration > 0 else 0,
            "first_event": self.events[0].timestamp,
            "last_event": self.events[-1].timestamp,
        }
    
    def export_events(self, file_path: Union[str, Path], format: str = "json"):
        """Export events to file."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            events_data = [asdict(event) for event in self.events]
            with open(file_path, 'w') as f:
                json.dump(events_data, f, indent=2)
        
        elif format == "csv":
            import csv
            
            if self.events:
                with open(file_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=asdict(self.events[0]).keys())
                    writer.writeheader()
                    for event in self.events:
                        writer.writerow(asdict(event))


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        """Format log record as JSON."""
        try:
            # If message is already JSON, parse it
            log_data = json.loads(record.getMessage())
        except (json.JSONDecodeError, ValueError):
            # Create structured log entry
            log_data = {
                "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                "logger": record.name,
                "level": record.levelname,
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno,
            }
            
            # Add exception info if present
            if record.exc_info:
                log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)


class OptimizationMonitor:
    """Monitor and track optimization progress in real-time."""
    
    def __init__(self, logger: StructuredLogger):
        """Initialize optimization monitor.
        
        Args:
            logger: Structured logger instance
        """
        self.logger = logger
        self.current_optimization = None
        self.optimization_history = []
    
    def start_optimization(
        self,
        function_name: str,
        surrogate_type: str,
        optimizer_type: str,
        initial_point: Array,
        bounds: Optional[list] = None,
        **metadata
    ):
        """Start monitoring an optimization run."""
        self.current_optimization = {
            "function_name": function_name,
            "surrogate_type": surrogate_type,
            "optimizer_type": optimizer_type,
            "initial_point": initial_point.tolist(),
            "bounds": bounds,
            "start_time": time.time(),
            "iterations": 0,
            "best_value": float('inf'),
            "convergence_history": [],
            "metadata": metadata,
        }
        
        self.logger.info(
            f"Starting optimization: {function_name} with {surrogate_type} + {optimizer_type}",
            "optimization_start",
            "monitor",
            **{k: v for k, v in self.current_optimization.items() if k != "convergence_history"}
        )
    
    def log_iteration(self, iteration: int, current_value: float, current_point: Array):
        """Log an optimization iteration."""
        if self.current_optimization is None:
            return
        
        self.current_optimization["iterations"] = iteration
        self.current_optimization["convergence_history"].append(current_value)
        
        if current_value < self.current_optimization["best_value"]:
            self.current_optimization["best_value"] = current_value
            self.current_optimization["best_point"] = current_point.tolist()
            
            self.logger.info(
                f"New best value: {current_value:.6f} at iteration {iteration}",
                "optimization_improvement",
                "monitor",
                iteration=iteration,
                value=current_value,
                point=current_point.tolist()
            )
    
    def end_optimization(self, result):
        """End optimization monitoring."""
        if self.current_optimization is None:
            return
        
        duration = time.time() - self.current_optimization["start_time"]
        
        final_summary = {
            **self.current_optimization,
            "duration": duration,
            "final_result": {
                "success": result.success,
                "final_value": float(result.fun),
                "final_point": result.x.tolist(),
                "message": result.message,
            }
        }
        
        self.optimization_history.append(final_summary)
        
        self.logger.info(
            f"Optimization completed in {duration:.2f}s",
            "optimization_complete",
            "monitor",
            **{k: v for k, v in final_summary.items() if k != "convergence_history"}
        )
        
        self.current_optimization = None
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of all completed optimizations."""
        if not self.optimization_history:
            return {"total_optimizations": 0}
        
        total_optimizations = len(self.optimization_history)
        successful_optimizations = sum(1 for opt in self.optimization_history if opt["final_result"]["success"])
        total_duration = sum(opt["duration"] for opt in self.optimization_history)
        
        # Performance by method
        method_performance = {}
        for opt in self.optimization_history:
            method = f"{opt['surrogate_type']}+{opt['optimizer_type']}"
            if method not in method_performance:
                method_performance[method] = {"count": 0, "success_rate": 0, "avg_duration": 0}
            
            method_performance[method]["count"] += 1
            if opt["final_result"]["success"]:
                method_performance[method]["success_rate"] += 1
            method_performance[method]["avg_duration"] += opt["duration"]
        
        # Calculate averages
        for method_stats in method_performance.values():
            method_stats["success_rate"] /= method_stats["count"]
            method_stats["avg_duration"] /= method_stats["count"]
        
        return {
            "total_optimizations": total_optimizations,
            "successful_optimizations": successful_optimizations,
            "success_rate": successful_optimizations / total_optimizations,
            "total_duration": total_duration,
            "average_duration": total_duration / total_optimizations,
            "method_performance": method_performance,
        }


# Global logger instances
_loggers = {}

def get_logger(name: str, **kwargs) -> StructuredLogger:
    """Get or create a structured logger instance."""
    if name not in _loggers:
        _loggers[name] = StructuredLogger(name, **kwargs)
    return _loggers[name]


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    structured: bool = True,
    performance_metrics: bool = True,
):
    """Setup global logging configuration."""
    # Configure default loggers
    loggers = [
        "surrogate_optim.core",
        "surrogate_optim.models",
        "surrogate_optim.optimizers",
        "surrogate_optim.data",
        "surrogate_optim.benchmarks",
    ]
    
    for logger_name in loggers:
        get_logger(
            logger_name,
            log_level=log_level,
            log_file=log_file,
            structured_output=structured,
            include_performance_metrics=performance_metrics,
        )