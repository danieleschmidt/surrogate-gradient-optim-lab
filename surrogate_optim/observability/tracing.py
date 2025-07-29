"""OpenTelemetry tracing integration for surrogate optimization."""

import functools
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Union

try:
    from opentelemetry import trace
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.trace.status import Status, StatusCode
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None
    Status = None
    StatusCode = None

logger = logging.getLogger(__name__)


class SpanStatus(Enum):
    """Span status enumeration."""
    OK = "OK"
    ERROR = "ERROR"
    CANCELLED = "CANCELLED"


@dataclass
class TracingConfig:
    """Configuration for tracing setup."""
    service_name: str = "surrogate-optim"
    service_version: str = "0.1.0"
    environment: str = "development"
    jaeger_endpoint: Optional[str] = None
    otlp_endpoint: Optional[str] = None
    sample_rate: float = 1.0
    max_tag_value_length: int = 4096


class TracingManager:
    """Manages OpenTelemetry tracing configuration."""
    
    def __init__(self, config: TracingConfig):
        self.config = config
        self._tracer_provider = None
        self._tracer = None
        self._initialized = False
    
    def initialize(self):
        """Initialize tracing with configured exporters."""
        if not OTEL_AVAILABLE:
            logger.warning("OpenTelemetry not available, tracing disabled")
            return
        
        if self._initialized:
            return
        
        # Create resource with service information
        resource = Resource.create({
            "service.name": self.config.service_name,
            "service.version": self.config.service_version,
            "deployment.environment": self.config.environment,
        })
        
        # Create tracer provider
        self._tracer_provider = TracerProvider(resource=resource)
        
        # Configure exporters
        if self.config.jaeger_endpoint:
            jaeger_exporter = JaegerExporter(
                agent_host_name="localhost",
                agent_port=14268,
                collector_endpoint=self.config.jaeger_endpoint,
            )
            self._tracer_provider.add_span_processor(
                BatchSpanProcessor(jaeger_exporter)
            )
        
        if self.config.otlp_endpoint:
            otlp_exporter = OTLPSpanExporter(
                endpoint=self.config.otlp_endpoint,
                insecure=True,
            )
            self._tracer_provider.add_span_processor(
                BatchSpanProcessor(otlp_exporter)
            )
        
        # Set global tracer provider
        trace.set_tracer_provider(self._tracer_provider)
        self._tracer = trace.get_tracer(
            __name__,
            version=self.config.service_version,
        )
        
        self._initialized = True
        logger.info(f"Tracing initialized for service: {self.config.service_name}")
    
    def get_tracer(self):
        """Get the configured tracer."""
        if not self._initialized:
            self.initialize()
        return self._tracer
    
    def shutdown(self):
        """Shutdown tracing and flush pending spans."""
        if self._tracer_provider:
            self._tracer_provider.shutdown()


# Global tracing manager
_tracing_manager = None


def configure_tracing(config: TracingConfig) -> TracingManager:
    """Configure global tracing with given configuration."""
    global _tracing_manager
    _tracing_manager = TracingManager(config)
    _tracing_manager.initialize()
    return _tracing_manager


def get_tracer():
    """Get the global tracer instance."""
    if not OTEL_AVAILABLE:
        return None
    
    global _tracing_manager
    if _tracing_manager is None:
        # Initialize with default config
        _tracing_manager = TracingManager(TracingConfig())
        _tracing_manager.initialize()
    
    return _tracing_manager.get_tracer()


@contextmanager 
def create_span(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
    kind: Optional[str] = None
):
    """Create a new span context manager."""
    tracer = get_tracer()
    if not tracer:
        yield None
        return
    
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for key, value in attributes.items():
                set_span_attribute(span, key, value)
        
        if kind:
            # Set span kind if supported
            pass
        
        try:
            yield span
        except Exception as e:
            set_span_status(span, SpanStatus.ERROR, str(e))
            raise


def set_span_attribute(span, key: str, value: Any):
    """Set an attribute on a span."""
    if span and hasattr(span, 'set_attribute'):
        # Convert value to string if needed
        if not isinstance(value, (str, int, float, bool)):
            value = str(value)
        span.set_attribute(key, value)


def set_span_status(span, status: SpanStatus, description: str = ""):
    """Set span status."""
    if not span or not hasattr(span, 'set_status'):
        return
    
    if not OTEL_AVAILABLE:
        return
    
    status_map = {
        SpanStatus.OK: StatusCode.OK,
        SpanStatus.ERROR: StatusCode.ERROR, 
        SpanStatus.CANCELLED: StatusCode.ERROR,
    }
    
    otel_status = status_map.get(status, StatusCode.OK)
    span.set_status(Status(otel_status, description))


def trace_function(
    span_name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
    record_exception: bool = True
):
    """Decorator to trace function execution."""
    def decorator(func):
        name = span_name or f"{func.__module__}.{func.__qualname__}"
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_attributes = {
                "function.name": func.__name__,
                "function.module": func.__module__,
            }
            
            if attributes:
                func_attributes.update(attributes) 
            
            with create_span(name, func_attributes) as span:
                try:
                    result = func(*args, **kwargs)
                    set_span_status(span, SpanStatus.OK)
                    return result
                except Exception as e:
                    if record_exception and span:
                        # Record exception details
                        set_span_attribute(span, "exception.type", type(e).__name__)
                        set_span_attribute(span, "exception.message", str(e))
                    
                    set_span_status(span, SpanStatus.ERROR, str(e))
                    raise
        
        return wrapper
    return decorator


# Pre-configured decorators for common operations
def trace_training(span_name: Optional[str] = None):
    """Decorator specifically for training operations."""
    return trace_function(
        span_name=span_name,
        attributes={"operation.type": "training"}
    )


def trace_prediction(span_name: Optional[str] = None):
    """Decorator specifically for prediction operations."""
    return trace_function(
        span_name=span_name, 
        attributes={"operation.type": "prediction"}
    )


def trace_optimization(span_name: Optional[str] = None):
    """Decorator specifically for optimization operations.""" 
    return trace_function(
        span_name=span_name,
        attributes={"operation.type": "optimization"}
    )


class TraceContext:
    """Helper class to manage trace context."""
    
    @staticmethod
    def get_current_span():
        """Get current active span."""
        if not OTEL_AVAILABLE:
            return None
        return trace.get_current_span()
    
    @staticmethod
    def get_trace_id() -> Optional[str]:
        """Get current trace ID."""
        span = TraceContext.get_current_span()
        if span and hasattr(span, 'get_span_context'):
            ctx = span.get_span_context()
            if hasattr(ctx, 'trace_id'):
                return format(ctx.trace_id, '032x')
        return None
    
    @staticmethod
    def get_span_id() -> Optional[str]:
        """Get current span ID."""
        span = TraceContext.get_current_span()
        if span and hasattr(span, 'get_span_context'):
            ctx = span.get_span_context()
            if hasattr(ctx, 'span_id'):
                return format(ctx.span_id, '016x')
        return None
    
    @staticmethod
    def inject_context() -> Dict[str, str]:
        """Inject current trace context into headers."""
        if not OTEL_AVAILABLE:
            return {}
        
        from opentelemetry.propagate import inject
        
        headers = {}
        inject(headers)
        return headers
    
    @staticmethod
    def extract_context(headers: Dict[str, str]):
        """Extract trace context from headers."""
        if not OTEL_AVAILABLE:
            return
        
        from opentelemetry.propagate import extract
        extract(headers)


# Sampling utilities
class AdaptiveSampler:
    """Adaptive sampling based on error rates and performance."""
    
    def __init__(self, base_rate: float = 0.1, error_boost: float = 1.0):
        self.base_rate = base_rate
        self.error_boost = error_boost
        self._error_count = 0
        self._total_count = 0
    
    def should_sample(self, has_error: bool = False) -> bool:
        """Determine if this trace should be sampled."""
        self._total_count += 1
        if has_error:
            self._error_count += 1
        
        # Boost sampling rate when errors are occurring
        error_rate = self._error_count / max(self._total_count, 1)
        current_rate = min(self.base_rate + (error_rate * self.error_boost), 1.0)
        
        import random
        return random.random() < current_rate
    
    def reset_stats(self):
        """Reset error tracking statistics."""
        self._error_count = 0
        self._total_count = 0


# Example usage patterns
def setup_production_tracing():
    """Setup tracing for production environment."""
    config = TracingConfig(
        service_name="surrogate-optim-prod",
        environment="production",
        jaeger_endpoint="http://jaeger-collector:14268/api/traces",
        sample_rate=0.1,
    )
    configure_tracing(config)


def setup_development_tracing():
    """Setup tracing for development environment.""" 
    config = TracingConfig(
        service_name="surrogate-optim-dev",
        environment="development",
        jaeger_endpoint="http://localhost:14268/api/traces",
        sample_rate=1.0,
    )
    configure_tracing(config)