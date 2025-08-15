"""Real-time monitoring dashboard for self-healing pipeline status."""

import time
import json
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import asyncio
from pathlib import Path

import numpy as np
from loguru import logger

# Web framework imports with fallbacks
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    from fastapi.requests import Request
    from fastapi.responses import HTMLResponse
    import uvicorn
    WEB_FRAMEWORK_AVAILABLE = True
except ImportError:
    WEB_FRAMEWORK_AVAILABLE = False
    logger.warning("Web framework not available. Dashboard will use file-based output.")

from .pipeline_monitor import PipelineMonitor, PipelineHealth, HealthStatus
from .recovery_engine import RecoveryEngine, RecoveryResult
from .health_diagnostics import HealthDiagnostics, DiagnosticReport
from .robust_monitoring import RobustMonitor


@dataclass
class DashboardMetrics:
    """Dashboard metrics snapshot."""
    timestamp: float
    system_health: Optional[PipelineHealth]
    diagnostic_report: Optional[DiagnosticReport]
    recovery_history: List[RecoveryResult]
    performance_metrics: Dict[str, Any]
    error_statistics: Dict[str, Any]
    resource_usage: Dict[str, float]


class MonitoringDashboard:
    """Real-time monitoring dashboard for self-healing pipeline."""
    
    def __init__(
        self,
        monitor: Optional[RobustMonitor] = None,
        update_interval: float = 5.0,
        port: int = 8080,
        enable_web_interface: bool = True
    ):
        self.monitor = monitor
        self.update_interval = update_interval
        self.port = port
        self.enable_web_interface = enable_web_interface and WEB_FRAMEWORK_AVAILABLE
        
        # Data storage
        self._metrics_history: List[DashboardMetrics] = []
        self._active_websockets: List[WebSocket] = []
        
        # Dashboard state
        self._dashboard_active = False
        self._update_thread: Optional[threading.Thread] = None
        
        # Initialize web app if available
        if self.enable_web_interface:
            self.app = self._create_web_app()
        else:
            self.app = None
            logger.info("Web interface disabled, using file-based dashboard")
            
    def _create_web_app(self) -> FastAPI:
        """Create FastAPI web application."""
        app = FastAPI(title="Self-Healing Pipeline Dashboard", version="1.0.0")
        
        # Static files and templates
        templates_dir = Path(__file__).parent / "templates"
        static_dir = Path(__file__).parent / "static"
        
        # Create directories if they don't exist
        templates_dir.mkdir(exist_ok=True)
        static_dir.mkdir(exist_ok=True)
        
        # Create basic HTML template if it doesn't exist
        self._create_dashboard_template(templates_dir)
        
        if static_dir.exists():
            app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
            
        templates = Jinja2Templates(directory=str(templates_dir))
        
        @app.get("/", response_class=HTMLResponse)
        async def dashboard_home(request: Request):
            return templates.TemplateResponse("dashboard.html", {"request": request})
            
        @app.get("/api/health")
        async def get_health_status():
            return self._get_current_status()
            
        @app.get("/api/metrics")
        async def get_metrics():
            return self._get_metrics_summary()
            
        @app.get("/api/history")
        async def get_history(limit: int = 100):
            return self._get_metrics_history(limit)
            
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self._handle_websocket(websocket)
            
        return app
        
    def _create_dashboard_template(self, templates_dir: Path) -> None:
        """Create basic dashboard HTML template."""
        template_path = templates_dir / "dashboard.html"
        
        if template_path.exists():
            return
            
        html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Self-Healing Pipeline Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .card { background: white; border-radius: 8px; padding: 20px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .status-healthy { color: #28a745; }
        .status-warning { color: #ffc107; }
        .status-critical { color: #dc3545; }
        .status-failed { color: #6c757d; }
        .metric { display: inline-block; margin: 10px; padding: 10px; background: #f8f9fa; border-radius: 4px; }
        .chart { height: 300px; background: #f8f9fa; border-radius: 4px; margin: 10px 0; padding: 20px; }
        #status { font-size: 24px; font-weight: bold; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .error { color: #dc3545; }
        .success { color: #28a745; }
        .timestamp { color: #6c757d; font-size: 0.9em; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Self-Healing Pipeline Dashboard</h1>
        
        <div class="card">
            <h2>System Status</h2>
            <div id="status">Loading...</div>
            <div id="lastUpdate" class="timestamp"></div>
        </div>
        
        <div class="grid">
            <div class="card">
                <h3>Health Metrics</h3>
                <div id="healthMetrics">Loading...</div>
            </div>
            
            <div class="card">
                <h3>Performance</h3>
                <div id="performanceMetrics">Loading...</div>
            </div>
            
            <div class="card">
                <h3>Error Statistics</h3>
                <div id="errorStats">Loading...</div>
            </div>
            
            <div class="card">
                <h3>Recovery Actions</h3>
                <div id="recoveryActions">Loading...</div>
            </div>
        </div>
        
        <div class="card">
            <h3>System Resources</h3>
            <div class="chart" id="resourceChart">Resource usage chart will appear here</div>
        </div>
        
        <div class="card">
            <h3>Health Trend</h3>
            <div class="chart" id="healthChart">Health trend chart will appear here</div>
        </div>
    </div>
    
    <script>
        let socket = new WebSocket(`ws://${window.location.host}/ws`);
        
        socket.onmessage = function(event) {
            const data = JSON.parse(event.data);
            updateDashboard(data);
        };
        
        socket.onclose = function(event) {
            console.log('WebSocket connection closed, attempting to reconnect...');
            setTimeout(() => {
                socket = new WebSocket(`ws://${window.location.host}/ws`);
            }, 5000);
        };
        
        function updateDashboard(data) {
            // Update status
            const statusElement = document.getElementById('status');
            if (data.system_health) {
                const status = data.system_health.overall_status;
                statusElement.textContent = status.toUpperCase();
                statusElement.className = `status-${status}`;
            }
            
            document.getElementById('lastUpdate').textContent = 
                `Last updated: ${new Date(data.timestamp * 1000).toLocaleString()}`;
            
            // Update health metrics
            updateHealthMetrics(data.system_health);
            
            // Update performance metrics
            updatePerformanceMetrics(data.performance_metrics);
            
            // Update error statistics
            updateErrorStats(data.error_statistics);
            
            // Update recovery actions
            updateRecoveryActions(data.recovery_history);
        }
        
        function updateHealthMetrics(health) {
            const container = document.getElementById('healthMetrics');
            if (!health || !health.metrics) {
                container.innerHTML = '<p>No health data available</p>';
                return;
            }
            
            let html = '';
            for (const [name, metric] of Object.entries(health.metrics)) {
                const statusClass = `status-${metric.status}`;
                html += `
                    <div class="metric">
                        <strong>${name}:</strong> 
                        <span class="${statusClass}">${metric.value.toFixed(3)}</span>
                    </div>
                `;
            }
            container.innerHTML = html;
        }
        
        function updatePerformanceMetrics(metrics) {
            const container = document.getElementById('performanceMetrics');
            if (!metrics) {
                container.innerHTML = '<p>No performance data available</p>';
                return;
            }
            
            let html = '';
            for (const [name, value] of Object.entries(metrics)) {
                html += `
                    <div class="metric">
                        <strong>${name}:</strong> ${JSON.stringify(value)}
                    </div>
                `;
            }
            container.innerHTML = html;
        }
        
        function updateErrorStats(stats) {
            const container = document.getElementById('errorStats');
            if (!stats) {
                container.innerHTML = '<p>No error data available</p>';
                return;
            }
            
            let html = `
                <div class="metric">
                    <strong>Total Errors:</strong> ${stats.total_errors || 0}
                </div>
                <div class="metric">
                    <strong>Recent Errors:</strong> ${stats.recent_errors || 0}
                </div>
                <div class="metric">
                    <strong>Error Rate:</strong> ${(stats.error_rate || 0).toFixed(4)}/s
                </div>
            `;
            container.innerHTML = html;
        }
        
        function updateRecoveryActions(actions) {
            const container = document.getElementById('recoveryActions');
            if (!actions || actions.length === 0) {
                container.innerHTML = '<p>No recent recovery actions</p>';
                return;
            }
            
            let html = '';
            actions.slice(-5).forEach(action => {
                const statusClass = action.success ? 'success' : 'error';
                html += `
                    <div class="metric">
                        <span class="${statusClass}">
                            ${action.action}: ${action.success ? 'SUCCESS' : 'FAILED'}
                        </span>
                        <br>
                        <small>${action.message}</small>
                    </div>
                `;
            });
            container.innerHTML = html;
        }
        
        // Initial data load
        fetch('/api/health')
            .then(response => response.json())
            .then(data => updateDashboard(data))
            .catch(error => console.error('Error loading initial data:', error));
    </script>
</body>
</html>
        """
        
        with open(template_path, 'w') as f:
            f.write(html_content)
            
    async def _handle_websocket(self, websocket: WebSocket) -> None:
        """Handle WebSocket connection for real-time updates."""
        await websocket.accept()
        self._active_websockets.append(websocket)
        
        try:
            while True:
                # Send current status
                status = self._get_current_status()
                await websocket.send_text(json.dumps(status, default=str))
                
                # Wait for next update
                await asyncio.sleep(self.update_interval)
                
        except WebSocketDisconnect:
            self._active_websockets.remove(websocket)
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            if websocket in self._active_websockets:
                self._active_websockets.remove(websocket)
                
    def start_dashboard(self) -> None:
        """Start the monitoring dashboard."""
        self._dashboard_active = True
        
        # Start metrics collection thread
        self._update_thread = threading.Thread(target=self._metrics_update_loop, daemon=True)
        self._update_thread.start()
        
        if self.enable_web_interface and self.app:
            # Start web server in separate thread
            web_thread = threading.Thread(
                target=self._run_web_server,
                daemon=True
            )
            web_thread.start()
            logger.info(f"Dashboard web interface started at http://localhost:{self.port}")
        else:
            # Start file-based dashboard
            logger.info("Dashboard started in file-based mode")
            
    def _run_web_server(self) -> None:
        """Run the web server."""
        try:
            uvicorn.run(self.app, host="0.0.0.0", port=self.port, log_level="warning")
        except Exception as e:
            logger.error(f"Web server error: {e}")
            
    def stop_dashboard(self) -> None:
        """Stop the monitoring dashboard."""
        self._dashboard_active = False
        
        if self._update_thread:
            self._update_thread.join(timeout=5.0)
            
        # Close WebSocket connections
        for websocket in self._active_websockets:
            try:
                asyncio.create_task(websocket.close())
            except Exception:
                pass
                
        logger.info("Dashboard stopped")
        
    def _metrics_update_loop(self) -> None:
        """Main metrics collection and update loop."""
        while self._dashboard_active:
            try:
                # Collect current metrics
                metrics = self._collect_dashboard_metrics()
                self._metrics_history.append(metrics)
                
                # Maintain rolling window
                if len(self._metrics_history) > 1000:
                    self._metrics_history = self._metrics_history[-1000:]
                    
                # Export to file if web interface is not available
                if not self.enable_web_interface:
                    self._export_metrics_to_file(metrics)
                    
                # Send updates to WebSocket clients
                if self.enable_web_interface:
                    asyncio.run(self._broadcast_metrics_update(metrics))
                    
            except Exception as e:
                logger.error(f"Metrics update error: {e}")
                
            time.sleep(self.update_interval)
            
    def _collect_dashboard_metrics(self) -> DashboardMetrics:
        """Collect all dashboard metrics."""
        system_health = None
        diagnostic_report = None
        recovery_history = []
        performance_metrics = {}
        error_statistics = {}
        
        if self.monitor:
            # Get system health
            system_health = self.monitor.pipeline_monitor.get_current_health()
            
            # Get diagnostic report
            if self.monitor.diagnostics and system_health:
                try:
                    diagnostic_report = self.monitor.diagnostics.analyze_current_health(system_health)
                except Exception as e:
                    logger.warning(f"Failed to generate diagnostic report: {e}")
                    
            # Get recovery history
            recovery_history = self.monitor.recovery_engine.get_recovery_history(20)
            
            # Get performance metrics
            performance_metrics = self.monitor._get_performance_summary()
            
            # Get error statistics
            if hasattr(self.monitor, '_error_handler'):
                error_statistics = self.monitor._error_handler.get_error_statistics()
                
        # Resource usage
        resource_usage = self._get_resource_usage()
        
        return DashboardMetrics(
            timestamp=time.time(),
            system_health=system_health,
            diagnostic_report=diagnostic_report,
            recovery_history=recovery_history,
            performance_metrics=performance_metrics,
            error_statistics=error_statistics,
            resource_usage=resource_usage
        )
        
    def _get_resource_usage(self) -> Dict[str, float]:
        """Get current resource usage."""
        try:
            import psutil
            
            return {
                "memory_percent": psutil.virtual_memory().percent,
                "cpu_percent": psutil.cpu_percent(),
                "disk_percent": psutil.disk_usage('/').percent,
                "swap_percent": psutil.swap_memory().percent,
                "process_count": len(psutil.pids())
            }
        except Exception as e:
            logger.warning(f"Failed to get resource usage: {e}")
            return {}
            
    async def _broadcast_metrics_update(self, metrics: DashboardMetrics) -> None:
        """Broadcast metrics update to all WebSocket clients."""
        if not self._active_websockets:
            return
            
        message = json.dumps(asdict(metrics), default=str)
        
        # Send to all connected clients
        disconnected = []
        for websocket in self._active_websockets:
            try:
                await websocket.send_text(message)
            except Exception:
                disconnected.append(websocket)
                
        # Remove disconnected clients
        for websocket in disconnected:
            self._active_websockets.remove(websocket)
            
    def _export_metrics_to_file(self, metrics: DashboardMetrics) -> None:
        """Export metrics to JSON file for file-based dashboard."""
        try:
            output_file = Path("dashboard_metrics.json")
            
            # Create summary data
            summary = {
                "timestamp": metrics.timestamp,
                "last_update": datetime.fromtimestamp(metrics.timestamp).isoformat(),
                "system_health": asdict(metrics.system_health) if metrics.system_health else None,
                "diagnostic_summary": {
                    "anomaly_count": len(metrics.diagnostic_report.anomalies) if metrics.diagnostic_report else 0,
                    "health_score": metrics.diagnostic_report.overall_health_score if metrics.diagnostic_report else 0.0,
                    "recommendations": metrics.diagnostic_report.recommendations if metrics.diagnostic_report else []
                },
                "recovery_summary": {
                    "recent_actions": len(metrics.recovery_history),
                    "success_rate": sum(1 for r in metrics.recovery_history if r.success) / max(1, len(metrics.recovery_history))
                },
                "resource_usage": metrics.resource_usage,
                "error_statistics": metrics.error_statistics
            }
            
            with open(output_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
                
            # Also create a simple HTML file
            self._create_simple_html_report(summary)
            
        except Exception as e:
            logger.error(f"Failed to export metrics to file: {e}")
            
    def _create_simple_html_report(self, summary: Dict[str, Any]) -> None:
        """Create simple HTML report for file-based dashboard."""
        try:
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Pipeline Health Report</title>
    <meta charset="utf-8">
    <meta http-equiv="refresh" content="{int(self.update_interval)}">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .status-healthy {{ color: green; }}
        .status-warning {{ color: orange; }}
        .status-critical {{ color: red; }}
        .status-failed {{ color: darkred; }}
        .metric {{ margin: 10px 0; padding: 10px; background: #f5f5f5; border-radius: 4px; }}
        h1, h2 {{ color: #333; }}
    </style>
</head>
<body>
    <h1>Self-Healing Pipeline Status</h1>
    <p><strong>Last Updated:</strong> {summary['last_update']}</p>
    
    <h2>System Health</h2>
    """
            
            if summary['system_health']:
                status = summary['system_health']['overall_status']
                html_content += f'<p class="status-{status}"><strong>Status:</strong> {status.upper()}</p>'
                
                if summary['system_health']['metrics']:
                    html_content += '<h3>Metrics:</h3>'
                    for name, metric in summary['system_health']['metrics'].items():
                        status_class = f"status-{metric['status']}"
                        html_content += f'<div class="metric"><strong>{name}:</strong> <span class="{status_class}">{metric["value"]:.3f}</span></div>'
            else:
                html_content += '<p>No health data available</p>'
                
            html_content += f"""
    <h2>Resource Usage</h2>
    """
            
            for name, value in summary['resource_usage'].items():
                html_content += f'<div class="metric"><strong>{name}:</strong> {value:.1f}%</div>'
                
            html_content += f"""
    <h2>Error Statistics</h2>
    <div class="metric"><strong>Total Errors:</strong> {summary['error_statistics'].get('total_errors', 0)}</div>
    <div class="metric"><strong>Recent Errors:</strong> {summary['error_statistics'].get('recent_errors', 0)}</div>
    <div class="metric"><strong>Error Rate:</strong> {summary['error_statistics'].get('error_rate', 0):.4f}/s</div>
    
    <h2>Recovery Actions</h2>
    <div class="metric"><strong>Recent Actions:</strong> {summary['recovery_summary']['recent_actions']}</div>
    <div class="metric"><strong>Success Rate:</strong> {summary['recovery_summary']['success_rate']:.2%}</div>
    
    </body>
    </html>
            """
            
            with open("dashboard_report.html", 'w') as f:
                f.write(html_content)
                
        except Exception as e:
            logger.error(f"Failed to create HTML report: {e}")
            
    def _get_current_status(self) -> Dict[str, Any]:
        """Get current status for API endpoints."""
        if self._metrics_history:
            latest = self._metrics_history[-1]
            return asdict(latest)
        else:
            return {"timestamp": time.time(), "status": "no_data"}
            
    def _get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary for API endpoints."""
        if not self._metrics_history:
            return {}
            
        # Calculate trends and summaries
        recent_metrics = self._metrics_history[-10:]  # Last 10 data points
        
        health_scores = [m.diagnostic_report.overall_health_score for m in recent_metrics 
                        if m.diagnostic_report]
        
        return {
            "avg_health_score": np.mean(health_scores) if health_scores else 0.0,
            "health_trend": self._calculate_trend(health_scores),
            "total_data_points": len(self._metrics_history),
            "time_range": {
                "start": self._metrics_history[0].timestamp if self._metrics_history else 0,
                "end": self._metrics_history[-1].timestamp if self._metrics_history else 0
            }
        }
        
    def _get_metrics_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get metrics history for API endpoints."""
        recent = self._metrics_history[-limit:]
        return [asdict(m) for m in recent]
        
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a series of values."""
        if len(values) < 2:
            return "stable"
            
        # Simple linear trend
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        slope = coeffs[0]
        
        if slope > 0.01:
            return "improving"
        elif slope < -0.01:
            return "degrading"
        else:
            return "stable"