#!/usr/bin/env python3
"""
Automated metrics collection script for repository health monitoring.
Collects various metrics including code quality, performance, and collaboration data.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
import argparse


class MetricsCollector:
    """Collects and reports repository metrics."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.metrics_config = self._load_metrics_config()
        
    def _load_metrics_config(self) -> Dict[str, Any]:
        """Load metrics configuration from project-metrics.json."""
        config_path = self.repo_path / ".github" / "project-metrics.json"
        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)
        return {}
    
    def collect_code_metrics(self) -> Dict[str, Any]:
        """Collect code quality metrics."""
        metrics = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "lines_of_code": self._count_lines_of_code(),
            "test_coverage": self._get_test_coverage(),
            "complexity": self._get_code_complexity(),
            "dependencies": self._count_dependencies(),
        }
        return metrics
    
    def collect_git_metrics(self) -> Dict[str, Any]:
        """Collect Git repository metrics."""
        try:
            # Get commit count
            commit_count = subprocess.run(
                ["git", "rev-list", "--count", "HEAD"],
                capture_output=True, text=True, cwd=self.repo_path
            ).stdout.strip()
            
            # Get contributor count
            contributors = subprocess.run(
                ["git", "shortlog", "-sn", "--all"],
                capture_output=True, text=True, cwd=self.repo_path
            ).stdout.strip().split('\n')
            
            # Get recent activity
            recent_commits = subprocess.run(
                ["git", "log", "--since='30 days ago'", "--oneline"],
                capture_output=True, text=True, cwd=self.repo_path
            ).stdout.strip().split('\n')
            
            return {
                "total_commits": int(commit_count) if commit_count else 0,
                "unique_contributors": len([c for c in contributors if c.strip()]),
                "commits_last_30_days": len([c for c in recent_commits if c.strip()]),
                "current_branch": self._get_current_branch(),
            }
        except Exception as e:
            print(f"Warning: Could not collect Git metrics: {e}")
            return {"error": str(e)}
    
    def collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect performance-related metrics."""
        metrics = {
            "build_time": self._measure_build_time(),
            "test_execution_time": self._measure_test_time(),
            "repo_size_mb": self._get_repo_size(),
        }
        
        # Run benchmarks if available
        if (self.repo_path / "tests" / "benchmarks").exists():
            metrics["benchmark_results"] = self._run_benchmarks()
            
        return metrics
    
    def collect_dependency_metrics(self) -> Dict[str, Any]:
        """Collect dependency-related metrics."""
        metrics = {
            "outdated_dependencies": self._check_outdated_deps(),
            "security_vulnerabilities": self._check_security_issues(),
        }
        return metrics
    
    def _count_lines_of_code(self) -> Dict[str, int]:
        """Count lines of code by file type."""
        try:
            result = subprocess.run(
                ["find", ".", "-name", "*.py", "-exec", "wc", "-l", "{}", "+"],
                capture_output=True, text=True, cwd=self.repo_path
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                total = 0
                for line in lines:
                    if 'total' in line.lower():
                        total = int(line.split()[0])
                        break
                return {"python": total}
            return {"python": 0}
        except Exception:
            return {"python": 0}
    
    def _get_test_coverage(self) -> Optional[float]:
        """Get test coverage percentage."""
        try:
            # Try to get coverage from pytest-cov
            result = subprocess.run(
                ["python3", "-m", "pytest", "--cov=surrogate_optim", "--cov-report=term-missing", "--quiet"],
                capture_output=True, text=True, cwd=self.repo_path
            )
            
            # Parse coverage from output
            for line in result.stdout.split('\n'):
                if 'TOTAL' in line and '%' in line:
                    coverage = line.split()[-1].replace('%', '')
                    return float(coverage)
            return None
        except Exception:
            return None
    
    def _get_code_complexity(self) -> Dict[str, Any]:
        """Get code complexity metrics."""
        try:
            # Use radon for complexity analysis
            result = subprocess.run(
                ["python3", "-m", "radon", "cc", "surrogate_optim", "--json"],
                capture_output=True, text=True, cwd=self.repo_path
            )
            
            if result.returncode == 0:
                complexity_data = json.loads(result.stdout)
                # Calculate average complexity
                total_complexity = 0
                function_count = 0
                
                for file_data in complexity_data.values():
                    for item in file_data:
                        if item['type'] in ['function', 'method']:
                            total_complexity += item['complexity']
                            function_count += 1
                
                avg_complexity = total_complexity / function_count if function_count > 0 else 0
                return {
                    "average_complexity": round(avg_complexity, 2),
                    "total_functions": function_count,
                    "high_complexity_functions": sum(
                        1 for file_data in complexity_data.values()
                        for item in file_data
                        if item['type'] in ['function', 'method'] and item['complexity'] > 10
                    )
                }
        except Exception:
            pass
        
        return {"average_complexity": 0, "total_functions": 0, "high_complexity_functions": 0}
    
    def _count_dependencies(self) -> Dict[str, int]:
        """Count project dependencies."""
        try:
            pyproject_path = self.repo_path / "pyproject.toml"
            if pyproject_path.exists():
                with open(pyproject_path) as f:
                    content = f.read()
                    # Simple count of dependencies in pyproject.toml
                    deps = content.count('=')  # Rough approximation
                    return {"total": deps}
            return {"total": 0}
        except Exception:
            return {"total": 0}
    
    def _get_current_branch(self) -> str:
        """Get current Git branch."""
        try:
            result = subprocess.run(
                ["git", "branch", "--show-current"],
                capture_output=True, text=True, cwd=self.repo_path
            )
            return result.stdout.strip()
        except Exception:
            return "unknown"
    
    def _measure_build_time(self) -> Optional[float]:
        """Measure build time in seconds."""
        try:
            import time
            start_time = time.time()
            result = subprocess.run(
                ["python3", "-m", "build"],
                capture_output=True, cwd=self.repo_path
            )
            end_time = time.time()
            
            if result.returncode == 0:
                return round(end_time - start_time, 2)
            return None
        except Exception:
            return None
    
    def _measure_test_time(self) -> Optional[float]:
        """Measure test execution time in seconds."""
        try:
            import time
            start_time = time.time()
            result = subprocess.run(
                ["python3", "-m", "pytest", "--quiet"],
                capture_output=True, cwd=self.repo_path
            )
            end_time = time.time()
            
            if result.returncode == 0:
                return round(end_time - start_time, 2)
            return None
        except Exception:
            return None
    
    def _get_repo_size(self) -> float:
        """Get repository size in MB."""
        try:
            result = subprocess.run(
                ["du", "-sh", "."],
                capture_output=True, text=True, cwd=self.repo_path
            )
            size_str = result.stdout.split()[0]
            
            # Convert to MB
            if 'K' in size_str:
                return round(float(size_str.replace('K', '')) / 1024, 2)
            elif 'M' in size_str:
                return round(float(size_str.replace('M', '')), 2)
            elif 'G' in size_str:
                return round(float(size_str.replace('G', '')) * 1024, 2)
            return 0.0
        except Exception:
            return 0.0
    
    def _run_benchmarks(self) -> Dict[str, Any]:
        """Run performance benchmarks."""
        try:
            result = subprocess.run(
                ["python3", "-m", "pytest", "tests/benchmarks/", "--quiet"],
                capture_output=True, text=True, cwd=self.repo_path
            )
            
            return {
                "status": "passed" if result.returncode == 0 else "failed",
                "output": result.stdout if result.returncode == 0 else result.stderr
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _check_outdated_deps(self) -> int:
        """Check for outdated dependencies."""
        try:
            result = subprocess.run(
                ["pip", "list", "--outdated", "--format=json"],
                capture_output=True, text=True
            )
            
            if result.returncode == 0:
                outdated = json.loads(result.stdout)
                return len(outdated)
            return 0
        except Exception:
            return 0
    
    def _check_security_issues(self) -> Dict[str, Any]:
        """Check for security vulnerabilities."""
        try:
            result = subprocess.run(
                ["python3", "-m", "pip_audit", "--format=json"],
                capture_output=True, text=True
            )
            
            if result.returncode == 0:
                audit_data = json.loads(result.stdout)
                return {
                    "vulnerabilities_found": len(audit_data.get("vulnerabilities", [])),
                    "status": "clean" if len(audit_data.get("vulnerabilities", [])) == 0 else "issues_found"
                }
        except Exception:
            pass
        
        return {"vulnerabilities_found": 0, "status": "unknown"}
    
    def generate_report(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive metrics report."""
        report = {
            "metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "collector_version": "1.0.0",
                "repository": self.metrics_config.get("repository", {})
            },
            "metrics": {
                "code_quality": self.collect_code_metrics(),
                "git_activity": self.collect_git_metrics(),
                "performance": self.collect_performance_metrics(),
                "dependencies": self.collect_dependency_metrics(),
            }
        }
        
        # Add threshold analysis
        report["analysis"] = self._analyze_thresholds(report["metrics"])
        
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Metrics report saved to: {output_path}")
        
        return report
    
    def _analyze_thresholds(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze metrics against configured thresholds."""
        thresholds = self.metrics_config.get("metrics", {})
        analysis = {"warnings": [], "violations": [], "status": "healthy"}
        
        # Check code coverage
        coverage = metrics["code_quality"].get("test_coverage")
        if coverage is not None:
            threshold = thresholds.get("code_quality", {}).get("coverage_threshold", 90)
            if coverage < threshold:
                analysis["violations"].append(f"Test coverage {coverage}% below threshold {threshold}%")
        
        # Check complexity
        complexity = metrics["code_quality"]["complexity"]["average_complexity"]
        complexity_threshold = thresholds.get("code_quality", {}).get("complexity_threshold", 10)
        if complexity > complexity_threshold:
            analysis["warnings"].append(f"Average complexity {complexity} above threshold {complexity_threshold}")
        
        # Set overall status
        if analysis["violations"]:
            analysis["status"] = "critical"
        elif analysis["warnings"]:
            analysis["status"] = "warning"
        
        return analysis


def main():
    """Main entry point for metrics collection."""
    parser = argparse.ArgumentParser(description="Collect repository metrics")
    parser.add_argument("--output", "-o", help="Output file for metrics report")
    parser.add_argument("--repo-path", default=".", help="Path to repository")
    parser.add_argument("--format", choices=["json", "summary"], default="json", help="Output format")
    
    args = parser.parse_args()
    
    collector = MetricsCollector(args.repo_path)
    
    try:
        report = collector.generate_report(args.output)
        
        if args.format == "summary":
            print("\n📊 Repository Metrics Summary")
            print("=" * 40)
            
            # Code metrics
            code_metrics = report["metrics"]["code_quality"]
            print(f"Lines of Code: {code_metrics['lines_of_code']['python']}")
            if code_metrics['test_coverage'] is not None:
                print(f"Test Coverage: {code_metrics['test_coverage']}%")
            print(f"Average Complexity: {code_metrics['complexity']['average_complexity']}")
            
            # Git metrics
            git_metrics = report["metrics"]["git_activity"]
            print(f"Total Commits: {git_metrics.get('total_commits', 'N/A')}")
            print(f"Contributors: {git_metrics.get('unique_contributors', 'N/A')}")
            print(f"Commits (30 days): {git_metrics.get('commits_last_30_days', 'N/A')}")
            
            # Analysis
            analysis = report["analysis"]
            print(f"\nStatus: {analysis['status'].upper()}")
            if analysis["violations"]:
                print("❌ Violations:")
                for violation in analysis["violations"]:
                    print(f"  • {violation}")
            if analysis["warnings"]:
                print("⚠️  Warnings:")
                for warning in analysis["warnings"]:
                    print(f"  • {warning}")
        else:
            print(json.dumps(report, indent=2))
        
    except Exception as e:
        print(f"Error collecting metrics: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()