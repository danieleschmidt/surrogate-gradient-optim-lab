#!/usr/bin/env python3
"""
Metrics dashboard for real-time monitoring of repository health and performance.
Provides a web-based interface for viewing collected metrics and quality gates.
"""

import json
import time
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import argparse

try:
    from flask import Flask, render_template, jsonify, request
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    Flask = None
    FLASK_AVAILABLE = False

from .quality_gates import QualityGateRunner
from ..scripts.metrics_collector import MetricsCollector


@dataclass
class MetricPoint:
    """A single metric data point."""
    timestamp: str
    value: float
    status: str = "ok"
    threshold: Optional[float] = None


@dataclass
class DashboardData:
    """Complete dashboard data structure."""
    last_updated: str
    system_status: str
    quality_gates: List[Dict[str, Any]]
    metrics_history: Dict[str, List[MetricPoint]]
    recent_activity: List[Dict[str, Any]]
    health_score: float


class MetricsDashboard:
    """Real-time metrics dashboard for repository monitoring."""
    
    def __init__(self, repo_path: str = ".", port: int = 8080, debug: bool = False):
        self.repo_path = Path(repo_path)
        self.port = port
        self.debug = debug
        self.metrics_collector = MetricsCollector(str(self.repo_path))
        self.quality_runner = QualityGateRunner(str(self.repo_path))
        
        # Data storage
        self.metrics_history: Dict[str, List[MetricPoint]] = {}
        self.recent_activity: List[Dict[str, Any]] = []
        self.last_collection_time = None
        
        # Background thread for data collection
        self.collection_thread = None
        self.stop_collection = threading.Event()
        
        if FLASK_AVAILABLE:
            self.app = Flask(__name__)
            CORS(self.app)
            self._setup_routes()
        else:
            print("Flask not available. Install with: pip install flask flask-cors")
            self.app = None
    
    def _setup_routes(self):
        """Setup Flask routes for the dashboard."""
        
        @self.app.route('/')
        def dashboard():
            """Main dashboard page."""
            return self._render_dashboard()
        
        @self.app.route('/api/data')
        def api_data():
            """API endpoint for dashboard data."""
            return jsonify(asdict(self._get_dashboard_data()))
        
        @self.app.route('/api/metrics/<metric_name>')
        def api_metric(metric_name):
            """API endpoint for specific metric history."""
            history = self.metrics_history.get(metric_name, [])
            return jsonify([asdict(point) for point in history])
        
        @self.app.route('/api/quality-gates')
        def api_quality_gates():
            """API endpoint for quality gate status."""
            results = self.quality_runner.run_all()
            return jsonify([asdict(result) for result in results])
        
        @self.app.route('/api/refresh', methods=['POST'])
        def api_refresh():
            """Force refresh of metrics data."""
            self._collect_metrics()
            return jsonify({"status": "refreshed", "timestamp": datetime.now(timezone.utc).isoformat()})
    
    def _render_dashboard(self) -> str:
        """Render the main dashboard HTML."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Surrogate Optimization Lab - Metrics Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; }
        .header { background: #2c3e50; color: white; padding: 1rem 2rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .header h1 { font-size: 1.5rem; font-weight: 600; }
        .subtitle { opacity: 0.8; font-size: 0.9rem; margin-top: 0.25rem; }
        .container { max-width: 1200px; margin: 2rem auto; padding: 0 1rem; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5rem; }
        .card { background: white; border-radius: 8px; padding: 1.5rem; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        .card h2 { color: #2c3e50; margin-bottom: 1rem; font-size: 1.2rem; }
        .status-good { color: #27ae60; font-weight: 600; }
        .status-warning { color: #f39c12; font-weight: 600; }
        .status-error { color: #e74c3c; font-weight: 600; }
        .metric-item { display: flex; justify-content: space-between; align-items: center; padding: 0.5rem 0; border-bottom: 1px solid #ecf0f1; }
        .metric-item:last-child { border-bottom: none; }
        .metric-value { font-weight: 600; font-size: 1.1rem; }
        .refresh-btn { background: #3498db; color: white; border: none; padding: 0.5rem 1rem; border-radius: 4px; cursor: pointer; }
        .refresh-btn:hover { background: #2980b9; }
        .loading { text-align: center; color: #7f8c8d; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 Surrogate Optimization Lab</h1>
        <div class="subtitle">Real-time Repository Metrics Dashboard</div>
    </div>
    
    <div class="container">
        <div style="text-align: center; margin-bottom: 2rem;">
            <button class="refresh-btn" onclick="refreshData()">🔄 Refresh Data</button>
            <span id="last-updated" style="margin-left: 1rem; color: #7f8c8d;"></span>
        </div>
        
        <div class="grid">
            <div class="card">
                <h2>📊 System Health</h2>
                <div id="system-health" class="loading">Loading...</div>
            </div>
            
            <div class="card">
                <h2>🚦 Quality Gates</h2>
                <div id="quality-gates" class="loading">Loading...</div>
            </div>
            
            <div class="card">
                <h2>📈 Code Metrics</h2>
                <div id="code-metrics" class="loading">Loading...</div>
            </div>
            
            <div class="card">
                <h2>⚡ Performance</h2>
                <div id="performance-metrics" class="loading">Loading...</div>
            </div>
            
            <div class="card">
                <h2>🔒 Security</h2>
                <div id="security-metrics" class="loading">Loading...</div>
            </div>
            
            <div class="card">
                <h2>🤝 Collaboration</h2>
                <div id="collaboration-metrics" class="loading">Loading...</div>
            </div>
        </div>
    </div>
    
    <script>
        async function loadDashboardData() {
            try {
                const response = await fetch('/api/data');
                const data = await response.json();
                updateDashboard(data);
            } catch (error) {
                console.error('Failed to load dashboard data:', error);
            }
        }
        
        function updateDashboard(data) {
            document.getElementById('last-updated').textContent = 
                `Last updated: ${new Date(data.last_updated).toLocaleString()}`;
            
            // System Health
            const healthScore = data.health_score;
            const healthStatus = healthScore >= 90 ? 'good' : healthScore >= 70 ? 'warning' : 'error';
            document.getElementById('system-health').innerHTML = `
                <div class="metric-item">
                    <span>Overall Health Score</span>
                    <span class="metric-value status-${healthStatus}">${healthScore.toFixed(1)}%</span>
                </div>
                <div class="metric-item">
                    <span>System Status</span>
                    <span class="metric-value status-${data.system_status === 'healthy' ? 'good' : 'warning'}">${data.system_status.toUpperCase()}</span>
                </div>
            `;
            
            // Quality Gates
            const gatesHtml = data.quality_gates.map(gate => `
                <div class="metric-item">
                    <span>${gate.name}</span>
                    <span class="metric-value status-${gate.status === 'passed' ? 'good' : 'error'}">${gate.status.toUpperCase()}</span>
                </div>
            `).join('');
            document.getElementById('quality-gates').innerHTML = gatesHtml;
            
            // Code Metrics
            const codeMetrics = data.metrics_history;
            document.getElementById('code-metrics').innerHTML = `
                <div class="metric-item">
                    <span>Test Coverage</span>
                    <span class="metric-value">${getLatestValue(codeMetrics, 'test_coverage', 0).toFixed(1)}%</span>
                </div>
                <div class="metric-item">
                    <span>Lines of Code</span>
                    <span class="metric-value">${getLatestValue(codeMetrics, 'lines_of_code', 0).toLocaleString()}</span>
                </div>
                <div class="metric-item">
                    <span>Dependencies</span>
                    <span class="metric-value">${getLatestValue(codeMetrics, 'dependencies', 0)}</span>
                </div>
            `;
            
            // Performance Metrics
            document.getElementById('performance-metrics').innerHTML = `
                <div class="metric-item">
                    <span>Build Time</span>
                    <span class="metric-value">${getLatestValue(codeMetrics, 'build_time', 0).toFixed(1)}min</span>
                </div>
                <div class="metric-item">
                    <span>Test Execution</span>
                    <span class="metric-value">${getLatestValue(codeMetrics, 'test_time', 0).toFixed(1)}s</span>
                </div>
            `;
            
            // Security Metrics
            document.getElementById('security-metrics').innerHTML = `
                <div class="metric-item">
                    <span>Vulnerabilities</span>
                    <span class="metric-value status-${getLatestValue(codeMetrics, 'vulnerabilities', 0) === 0 ? 'good' : 'error'}">${getLatestValue(codeMetrics, 'vulnerabilities', 0)}</span>
                </div>
                <div class="metric-item">
                    <span>Security Score</span>
                    <span class="metric-value status-good">${getLatestValue(codeMetrics, 'security_score', 100).toFixed(1)}%</span>
                </div>
            `;
            
            // Collaboration Metrics
            document.getElementById('collaboration-metrics').innerHTML = `
                <div class="metric-item">
                    <span>Contributors</span>
                    <span class="metric-value">${getLatestValue(codeMetrics, 'contributors', 0)}</span>
                </div>
                <div class="metric-item">
                    <span>Recent Commits</span>
                    <span class="metric-value">${getLatestValue(codeMetrics, 'recent_commits', 0)}</span>
                </div>
            `;
        }
        
        function getLatestValue(metrics, key, defaultValue = 0) {
            const metric = metrics[key];
            return metric && metric.length > 0 ? metric[metric.length - 1].value : defaultValue;
        }
        
        async function refreshData() {
            document.querySelectorAll('.card > div:not(h2)').forEach(el => {
                el.innerHTML = '<div class="loading">Refreshing...</div>';
            });
            
            try {
                await fetch('/api/refresh', { method: 'POST' });
                await loadDashboardData();
            } catch (error) {
                console.error('Failed to refresh data:', error);
            }
        }
        
        // Load initial data
        loadDashboardData();
        
        // Auto-refresh every 5 minutes
        setInterval(loadDashboardData, 5 * 60 * 1000);
    </script>
</body>
</html>
        """
    
    def _get_dashboard_data(self) -> DashboardData:
        """Get current dashboard data."""
        # Run quality gates
        quality_results = self.quality_runner.run_all()
        quality_gates = [
            {
                "name": result.gate_name,
                "status": "passed" if result.passed else "failed",
                "message": result.message,
                "details": result.details
            }
            for result in quality_results
        ]
        
        # Calculate health score
        passed_gates = sum(1 for result in quality_results if result.passed)
        total_gates = len(quality_results)
        health_score = (passed_gates / total_gates * 100) if total_gates > 0 else 0
        
        # Determine system status
        system_status = "healthy" if health_score >= 80 else "degraded" if health_score >= 60 else "unhealthy"
        
        return DashboardData(
            last_updated=datetime.now(timezone.utc).isoformat(),
            system_status=system_status,
            quality_gates=quality_gates,
            metrics_history=self._get_metrics_history(),
            recent_activity=self.recent_activity,
            health_score=health_score
        )
    
    def _get_metrics_history(self) -> Dict[str, List[MetricPoint]]:
        """Get historical metrics data."""
        # Convert stored metrics to the format expected by dashboard
        formatted_history = {}
        for metric_name, points in self.metrics_history.items():
            formatted_history[metric_name] = [asdict(point) for point in points]
        return formatted_history
    
    def _collect_metrics(self):
        """Collect current metrics and update history."""
        try:
            # Collect code metrics
            code_metrics = self.metrics_collector.collect_code_metrics()
            git_metrics = self.metrics_collector.collect_git_metrics()
            performance_metrics = self.metrics_collector.collect_performance_metrics()
            
            timestamp = datetime.now(timezone.utc).isoformat()
            
            # Update metrics history
            all_metrics = {**code_metrics, **git_metrics, **performance_metrics}
            for metric_name, value in all_metrics.items():
                if isinstance(value, (int, float)):
                    if metric_name not in self.metrics_history:
                        self.metrics_history[metric_name] = []
                    
                    # Keep only last 100 points
                    if len(self.metrics_history[metric_name]) >= 100:
                        self.metrics_history[metric_name].pop(0)
                    
                    self.metrics_history[metric_name].append(
                        MetricPoint(timestamp=timestamp, value=float(value))
                    )
            
            # Add activity log
            self.recent_activity.insert(0, {
                "timestamp": timestamp,
                "type": "metrics_collection",
                "message": f"Collected {len(all_metrics)} metrics",
                "details": {"metric_count": len(all_metrics)}
            })
            
            # Keep only last 50 activities
            if len(self.recent_activity) > 50:
                self.recent_activity = self.recent_activity[:50]
            
            self.last_collection_time = datetime.now(timezone.utc)
            
        except Exception as e:
            print(f"Error collecting metrics: {e}")
    
    def start_background_collection(self, interval_seconds: int = 300):
        """Start background metrics collection."""
        def collect_loop():
            while not self.stop_collection.is_set():
                self._collect_metrics()
                self.stop_collection.wait(interval_seconds)
        
        self.collection_thread = threading.Thread(target=collect_loop, daemon=True)
        self.collection_thread.start()
        print(f"Started background metrics collection (interval: {interval_seconds}s)")
    
    def run(self, background_collection: bool = True, collection_interval: int = 300):
        """Run the metrics dashboard."""
        if not FLASK_AVAILABLE:
            print("Flask not available. Please install with: pip install flask flask-cors")
            return
        
        # Initial metrics collection
        self._collect_metrics()
        
        # Start background collection if requested
        if background_collection:
            self.start_background_collection(collection_interval)
        
        print(f"🚀 Metrics Dashboard starting on http://localhost:{self.port}")
        print("📊 Real-time repository monitoring available")
        print("🔄 Auto-refresh enabled (5 minutes)")
        
        try:
            self.app.run(host='0.0.0.0', port=self.port, debug=self.debug)
        except KeyboardInterrupt:
            print("\n🛑 Dashboard stopped by user")
        finally:
            if self.collection_thread:
                self.stop_collection.set()
                self.collection_thread.join(timeout=5)


def main():
    """CLI entry point for metrics dashboard."""
    parser = argparse.ArgumentParser(description="Launch the metrics dashboard")
    parser.add_argument("--port", type=int, default=8080, help="Port to run dashboard on")
    parser.add_argument("--repo-path", default=".", help="Path to repository")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--no-background", action="store_true", help="Disable background collection")
    parser.add_argument("--interval", type=int, default=300, help="Collection interval in seconds")
    
    args = parser.parse_args()
    
    dashboard = MetricsDashboard(
        repo_path=args.repo_path,
        port=args.port,
        debug=args.debug
    )
    
    dashboard.run(
        background_collection=not args.no_background,
        collection_interval=args.interval
    )


if __name__ == "__main__":
    main()