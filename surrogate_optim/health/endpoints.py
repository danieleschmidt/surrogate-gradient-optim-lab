"""HTTP endpoints for health checks and monitoring."""

import json
import time
from typing import Dict, Any
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import logging

from .checks import HealthChecker, HealthStatus, quick_health_check, is_ready, is_alive

logger = logging.getLogger(__name__)


class HealthHandler(BaseHTTPRequestHandler):
    """HTTP handler for health check endpoints."""
    
    def __init__(self, *args, health_checker: HealthChecker = None, **kwargs):
        self.health_checker = health_checker or HealthChecker()
        super().__init__(*args, **kwargs)
    
    def log_message(self, format, *args):
        """Override to use logger instead of stderr."""
        logger.info(f"{self.address_string()} - {format % args}")
    
    def do_GET(self):
        """Handle GET requests."""
        if self.path == "/health":
            self._handle_health()
        elif self.path == "/health/ready" or self.path == "/ready":
            self._handle_ready()
        elif self.path == "/health/live" or self.path == "/live":
            self._handle_live()
        elif self.path == "/health/detailed":
            self._handle_detailed_health()
        elif self.path == "/metrics":
            self._handle_metrics()
        elif self.path == "/version":
            self._handle_version()
        else:
            self._handle_not_found()
    
    def _handle_health(self):
        """Handle basic health check endpoint."""
        try:
            health = quick_health_check()
            
            if health.is_healthy:
                status_code = 200
            else:
                status_code = 503
            
            response = {
                "status": health.status.value,
                "timestamp": health.timestamp,
                "version": health.version,
            }
            
            self._send_json_response(status_code, response)
            
        except Exception as e:
            logger.exception("Health check failed")
            self._send_json_response(500, {
                "status": "error",
                "message": f"Health check failed: {str(e)}",
                "timestamp": time.time(),
            })
    
    def _handle_ready(self):
        """Handle readiness probe endpoint."""
        try:
            ready = is_ready()
            status_code = 200 if ready else 503
            
            response = {
                "ready": ready,
                "timestamp": time.time(),
            }
            
            self._send_json_response(status_code, response)
            
        except Exception as e:
            logger.exception("Readiness check failed")
            self._send_json_response(500, {
                "ready": False,
                "error": str(e),
                "timestamp": time.time(),
            })
    
    def _handle_live(self):
        """Handle liveness probe endpoint."""
        try:
            alive = is_alive()
            status_code = 200 if alive else 503
            
            response = {
                "alive": alive,
                "timestamp": time.time(),
            }
            
            self._send_json_response(status_code, response)
            
        except Exception as e:
            logger.exception("Liveness check failed")
            self._send_json_response(500, {
                "alive": False,
                "error": str(e),
                "timestamp": time.time(),
            })
    
    def _handle_detailed_health(self):
        """Handle detailed health check endpoint."""
        try:
            health = self.health_checker.run_all_checks()
            
            status_code = 200 if health.is_healthy else 503
            
            self._send_json_response(status_code, health.to_dict())
            
        except Exception as e:
            logger.exception("Detailed health check failed")
            self._send_json_response(500, {
                "status": "error",
                "message": f"Detailed health check failed: {str(e)}",
                "timestamp": time.time(),
            })
    
    def _handle_metrics(self):
        """Handle metrics endpoint (Prometheus format)."""
        try:
            health = self.health_checker.run_all_checks()
            
            # Generate Prometheus-style metrics
            metrics = []
            
            # Overall health status
            status_value = 1 if health.status == HealthStatus.HEALTHY else 0
            metrics.append(f'surrogate_optim_health_status {status_value}')
            
            # Individual check metrics
            for check in health.checks:
                check_name = check.name.replace('-', '_')
                check_value = 1 if check.status == HealthStatus.HEALTHY else 0
                metrics.append(f'surrogate_optim_health_check{{name="{check.name}"}} {check_value}')
                metrics.append(f'surrogate_optim_health_check_duration_ms{{name="{check.name}"}} {check.duration_ms}')
            
            # System metrics if available
            try:
                import psutil
                import os
                
                process = psutil.Process(os.getpid())
                memory_percent = process.memory_percent()
                cpu_percent = process.cpu_percent()
                
                metrics.append(f'surrogate_optim_memory_usage_percent {memory_percent}')
                metrics.append(f'surrogate_optim_cpu_usage_percent {cpu_percent}')
                
            except ImportError:
                pass
            
            metrics_text = '\n'.join(metrics) + '\n'
            
            self._send_response(200, metrics_text, content_type='text/plain; version=0.0.4; charset=utf-8')
            
        except Exception as e:
            logger.exception("Metrics endpoint failed")
            self._send_response(500, f"# Error generating metrics: {str(e)}\n", 
                              content_type='text/plain')
    
    def _handle_version(self):
        """Handle version endpoint."""
        try:
            from surrogate_optim import __version__
            
            response = {
                "version": __version__,
                "timestamp": time.time(),
            }
            
            self._send_json_response(200, response)
            
        except Exception as e:
            logger.exception("Version endpoint failed")
            self._send_json_response(500, {
                "error": f"Failed to get version: {str(e)}",
                "timestamp": time.time(),
            })
    
    def _handle_not_found(self):
        """Handle 404 Not Found."""
        response = {
            "error": "Not Found",
            "path": self.path,
            "available_endpoints": [
                "/health",
                "/health/ready",
                "/health/live", 
                "/health/detailed",
                "/metrics",
                "/version",
            ],
            "timestamp": time.time(),
        }
        
        self._send_json_response(404, response)
    
    def _send_json_response(self, status_code: int, data: Dict[str, Any]):
        """Send JSON response."""
        json_data = json.dumps(data, indent=2)
        self._send_response(status_code, json_data, content_type='application/json')
    
    def _send_response(self, status_code: int, body: str, content_type: str = 'application/json'):
        """Send HTTP response."""
        self.send_response(status_code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body.encode('utf-8'))))
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        self.end_headers()
        self.wfile.write(body.encode('utf-8'))


class HealthServer:
    """HTTP server for health check endpoints."""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8080, health_checker: HealthChecker = None):
        """Initialize health server.
        
        Args:
            host: Host to bind to.
            port: Port to bind to.
            health_checker: Health checker instance to use.
        """
        self.host = host
        self.port = port
        self.health_checker = health_checker or HealthChecker()
        self.server = None
        self.thread = None
    
    def start(self):
        """Start the health server in a separate thread."""
        if self.server is not None:
            logger.warning("Health server already running")
            return
        
        # Create handler class with health checker
        def handler_factory(*args, **kwargs):
            return HealthHandler(*args, health_checker=self.health_checker, **kwargs)
        
        self.server = HTTPServer((self.host, self.port), handler_factory)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        
        logger.info(f"Health server started on {self.host}:{self.port}")
    
    def stop(self):
        """Stop the health server."""
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
        
        if self.thread is not None:
            self.thread.join(timeout=5.0)
            self.thread = None
        
        logger.info("Health server stopped")
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


def create_health_app(health_checker: HealthChecker = None) -> HealthServer:
    """Create a health check server application.
    
    Args:
        health_checker: Health checker instance to use.
        
    Returns:
        HealthServer: Configured health server.
    """
    return HealthServer(health_checker=health_checker)


# CLI interface for health server
def main():
    """Main entry point for health server CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Surrogate Optim Health Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Start health server
    server = HealthServer(host=args.host, port=args.port)
    
    try:
        server.start()
        logger.info(f"Health server running on http://{args.host}:{args.port}")
        logger.info("Available endpoints:")
        logger.info("  /health - Basic health check")
        logger.info("  /health/ready - Readiness probe")
        logger.info("  /health/live - Liveness probe")
        logger.info("  /health/detailed - Detailed health information")
        logger.info("  /metrics - Prometheus metrics")
        logger.info("  /version - Version information")
        
        # Keep main thread alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Shutting down health server...")
    finally:
        server.stop()


if __name__ == "__main__":
    main()