# Monitoring & Observability

This directory contains documentation for monitoring and observability setup for the Surrogate Gradient Optimization Lab.

## Overview

The monitoring stack includes:

- **Health Checks**: System health monitoring with liveness and readiness probes
- **Metrics Collection**: Prometheus-compatible metrics for performance monitoring
- **Logging**: Structured logging with multiple output formats
- **Tracing**: OpenTelemetry integration for distributed tracing
- **Alerting**: Configurable alert rules and notification channels

## Components

### Health Checks
- JAX availability and computation tests
- Memory usage monitoring
- Dependency validation
- System resource checks

### Metrics
- Training duration and performance metrics
- Optimization iteration counters
- Memory and CPU usage gauges
- Model accuracy tracking

### Configuration
See `observability.yml` in the root directory for complete monitoring configuration.

## Setup

1. **Basic monitoring** (enabled by default):
   ```python
   from surrogate_optim.health import quick_health_check
   health = quick_health_check()
   ```

2. **Prometheus metrics**:
   ```bash
   # Start with Prometheus metrics endpoint
   python -m surrogate_optim.monitoring.server
   # Metrics available at http://localhost:9090/metrics
   ```

3. **Full observability stack** (Docker):
   ```bash
   # Start monitoring services
   docker-compose --profile monitoring up -d
   ```

## Endpoints

- `GET /health/live` - Liveness probe
- `GET /health/ready` - Readiness probe  
- `GET /health/detailed` - Detailed health status
- `GET /metrics` - Prometheus metrics
- `GET /metrics/json` - JSON metrics export