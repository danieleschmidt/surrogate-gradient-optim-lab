#!/usr/bin/env python3
"""
Repository health monitoring script.
Performs comprehensive health checks on the repository and its components.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse


class HealthChecker:
    """Performs health checks on repository components."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.checks = []
        self.results = {"healthy": [], "warnings": [], "critical": []}
        
    def add_check(self, name: str, check_func, critical: bool = False):
        """Add a health check."""
        self.checks.append({
            "name": name,
            "func": check_func,
            "critical": critical
        })
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all registered health checks."""
        print("🔍 Running repository health checks...")
        
        for check in self.checks:
            try:
                result = check["func"]()
                
                check_result = {
                    "name": check["name"],
                    "status": result.get("status", "unknown"),
                    "message": result.get("message", ""),
                    "details": result.get("details", {}),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                if result.get("status") == "healthy":
                    self.results["healthy"].append(check_result)
                elif result.get("status") == "warning":
                    self.results["warnings"].append(check_result)
                else:
                    if check["critical"]:
                        self.results["critical"].append(check_result)
                    else:
                        self.results["warnings"].append(check_result)
                        
            except Exception as e:
                error_result = {
                    "name": check["name"],
                    "status": "error",
                    "message": f"Check failed: {str(e)}",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                self.results["critical"].append(error_result)
        
        return self.results
    
    def check_git_repository(self) -> Dict[str, Any]:
        """Check Git repository health."""
        try:
            # Check if we're in a git repo
            result = subprocess.run(
                ["git", "status"],
                capture_output=True, text=True, cwd=self.repo_path
            )
            
            if result.returncode != 0:
                return {
                    "status": "critical",
                    "message": "Not a valid Git repository"
                }
            
            # Check for uncommitted changes
            if "nothing to commit" not in result.stdout:
                return {
                    "status": "warning",
                    "message": "Repository has uncommitted changes",
                    "details": {"git_status": result.stdout}
                }
            
            return {
                "status": "healthy",
                "message": "Git repository is clean"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to check Git status: {str(e)}"
            }
    
    def check_python_environment(self) -> Dict[str, Any]:
        """Check Python environment health."""
        try:
            # Check Python version
            python_version = sys.version
            
            # Check if we can import main package
            try:
                import surrogate_optim
                package_version = getattr(surrogate_optim, "__version__", "unknown")
            except ImportError:
                return {
                    "status": "critical",
                    "message": "Cannot import surrogate_optim package"
                }
            
            # Check dependencies
            try:
                result = subprocess.run(
                    ["pip", "check"],
                    capture_output=True, text=True
                )
                
                if result.returncode != 0:
                    return {
                        "status": "warning",
                        "message": "Dependency conflicts detected",
                        "details": {"pip_check": result.stdout}
                    }
            except Exception:
                pass
            
            return {
                "status": "healthy",
                "message": "Python environment is healthy",
                "details": {
                    "python_version": python_version,
                    "package_version": package_version
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to check Python environment: {str(e)}"
            }
    
    def check_dependencies(self) -> Dict[str, Any]:
        """Check dependency health."""
        try:
            # Check for security vulnerabilities
            try:
                result = subprocess.run(
                    ["python3", "-m", "pip_audit", "--format=json"],
                    capture_output=True, text=True
                )
                
                if result.returncode == 0:
                    audit_data = json.loads(result.stdout)
                    vulnerabilities = audit_data.get("vulnerabilities", [])
                    
                    if vulnerabilities:
                        return {
                            "status": "critical",
                            "message": f"Found {len(vulnerabilities)} security vulnerabilities",
                            "details": {"vulnerabilities": vulnerabilities}
                        }
            except Exception:
                pass
            
            # Check for outdated packages
            try:
                result = subprocess.run(
                    ["pip", "list", "--outdated", "--format=json"],
                    capture_output=True, text=True
                )
                
                if result.returncode == 0:
                    outdated = json.loads(result.stdout)
                    if len(outdated) > 10:  # Arbitrary threshold
                        return {
                            "status": "warning",
                            "message": f"Many outdated dependencies ({len(outdated)})",
                            "details": {"outdated_count": len(outdated)}
                        }
            except Exception:
                pass
            
            return {
                "status": "healthy",
                "message": "Dependencies are up to date and secure"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to check dependencies: {str(e)}"
            }
    
    def check_tests(self) -> Dict[str, Any]:
        """Check test suite health."""
        try:
            # Run tests
            result = subprocess.run(
                ["python3", "-m", "pytest", "--tb=no", "-q"],
                capture_output=True, text=True, cwd=self.repo_path
            )
            
            if result.returncode != 0:
                return {
                    "status": "critical",
                    "message": "Test suite is failing",
                    "details": {"test_output": result.stdout}
                }
            
            # Check test coverage
            coverage_result = subprocess.run(
                ["python3", "-m", "pytest", "--cov=surrogate_optim", "--cov-report=term-missing", "--tb=no", "-q"],
                capture_output=True, text=True, cwd=self.repo_path
            )
            
            coverage_percent = None
            if coverage_result.returncode == 0:
                for line in coverage_result.stdout.split('\n'):
                    if 'TOTAL' in line and '%' in line:
                        coverage_percent = float(line.split()[-1].replace('%', ''))
                        break
            
            if coverage_percent is not None and coverage_percent < 80:
                return {
                    "status": "warning",
                    "message": f"Test coverage is low: {coverage_percent}%",
                    "details": {"coverage": coverage_percent}
                }
            
            return {
                "status": "healthy",
                "message": "All tests passing",
                "details": {"coverage": coverage_percent}
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to run tests: {str(e)}"
            }
    
    def check_documentation(self) -> Dict[str, Any]:
        """Check documentation health."""
        try:
            required_docs = [
                "README.md",
                "CONTRIBUTING.md",
                "CODE_OF_CONDUCT.md",
                "SECURITY.md",
                "LICENSE",
                "docs/ROADMAP.md"
            ]
            
            missing_docs = []
            for doc in required_docs:
                if not (self.repo_path / doc).exists():
                    missing_docs.append(doc)
            
            if missing_docs:
                return {
                    "status": "warning",
                    "message": f"Missing documentation files: {', '.join(missing_docs)}",
                    "details": {"missing": missing_docs}
                }
            
            # Check if README is substantial
            readme_path = self.repo_path / "README.md"
            if readme_path.exists():
                readme_size = readme_path.stat().st_size
                if readme_size < 5000:  # Less than 5KB
                    return {
                        "status": "warning",
                        "message": "README.md appears to be minimal",
                        "details": {"readme_size_bytes": readme_size}
                    }
            
            return {
                "status": "healthy",
                "message": "Documentation is comprehensive"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to check documentation: {str(e)}"
            }
    
    def check_build_system(self) -> Dict[str, Any]:
        """Check build system health."""
        try:
            # Check for build configuration
            build_files = [
                "pyproject.toml",
                "setup.py",
                "Dockerfile",
                "Makefile"
            ]
            
            existing_build_files = [f for f in build_files if (self.repo_path / f).exists()]
            
            if not existing_build_files:
                return {
                    "status": "critical",
                    "message": "No build configuration files found"
                }
            
            # Try to build the package
            try:
                result = subprocess.run(
                    ["python3", "-m", "build", "--wheel"],
                    capture_output=True, text=True, cwd=self.repo_path
                )
                
                if result.returncode != 0:
                    return {
                        "status": "warning",
                        "message": "Package build failed",
                        "details": {"build_error": result.stderr}
                    }
            except Exception:
                # Build tool might not be installed
                pass
            
            return {
                "status": "healthy",
                "message": "Build system is configured",
                "details": {"build_files": existing_build_files}
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to check build system: {str(e)}"
            }
    
    def check_security_configuration(self) -> Dict[str, Any]:
        """Check security configuration."""
        try:
            security_files = [
                "SECURITY.md",
                ".github/dependabot.yml",
                "bandit.yml"
            ]
            
            existing_security_files = [f for f in security_files if (self.repo_path / f).exists()]
            
            if len(existing_security_files) < 2:
                return {
                    "status": "warning",
                    "message": "Limited security configuration",
                    "details": {"existing_files": existing_security_files}
                }
            
            # Check for secrets in code (basic check)
            try:
                result = subprocess.run(
                    ["git", "log", "--all", "--grep=password", "--grep=secret", "--grep=token", "--grep=key"],
                    capture_output=True, text=True, cwd=self.repo_path
                )
                
                if result.stdout.strip():
                    return {
                        "status": "warning",
                        "message": "Potential secrets mentioned in commit history"
                    }
            except Exception:
                pass
            
            return {
                "status": "healthy",
                "message": "Security configuration is adequate",
                "details": {"security_files": existing_security_files}
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to check security configuration: {str(e)}"
            }
    
    def check_performance(self) -> Dict[str, Any]:
        """Check performance benchmarks."""
        try:
            # Check if benchmarks exist
            benchmark_path = self.repo_path / "tests" / "benchmarks"
            if not benchmark_path.exists():
                return {
                    "status": "warning",
                    "message": "No performance benchmarks found"
                }
            
            # Try to run benchmarks
            try:
                result = subprocess.run(
                    ["python3", "-m", "pytest", "tests/benchmarks/", "--tb=no", "-q"],
                    capture_output=True, text=True, cwd=self.repo_path,
                    timeout=60  # 1 minute timeout
                )
                
                if result.returncode != 0:
                    return {
                        "status": "warning",
                        "message": "Performance benchmarks are failing"
                    }
                
                return {
                    "status": "healthy",
                    "message": "Performance benchmarks are running"
                }
                
            except subprocess.TimeoutExpired:
                return {
                    "status": "warning",
                    "message": "Performance benchmarks timed out"
                }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to check performance: {str(e)}"
            }


def main():
    """Main entry point for health checking."""
    parser = argparse.ArgumentParser(description="Check repository health")
    parser.add_argument("--repo-path", default=".", help="Path to repository")
    parser.add_argument("--output", "-o", help="Output file for health report")
    parser.add_argument("--format", choices=["json", "summary"], default="summary", help="Output format")
    
    args = parser.parse_args()
    
    checker = HealthChecker(args.repo_path)
    
    # Register all health checks
    checker.add_check("Git Repository", checker.check_git_repository, critical=True)
    checker.add_check("Python Environment", checker.check_python_environment, critical=True)
    checker.add_check("Dependencies", checker.check_dependencies)
    checker.add_check("Test Suite", checker.check_tests, critical=True)
    checker.add_check("Documentation", checker.check_documentation)
    checker.add_check("Build System", checker.check_build_system)
    checker.add_check("Security Configuration", checker.check_security_configuration)
    checker.add_check("Performance Benchmarks", checker.check_performance)
    
    try:
        results = checker.run_all_checks()
        
        # Generate report
        report = {
            "metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "repository_path": str(Path(args.repo_path).absolute()),
                "checker_version": "1.0.0"
            },
            "summary": {
                "total_checks": len(checker.checks),
                "healthy": len(results["healthy"]),
                "warnings": len(results["warnings"]),
                "critical": len(results["critical"]),
                "overall_status": "healthy" if not results["critical"] and len(results["warnings"]) < 3 else "warning" if not results["critical"] else "critical"
            },
            "results": results
        }
        
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Health report saved to: {output_path}")
        
        if args.format == "summary":
            print("\n🏥 Repository Health Summary")
            print("=" * 40)
            
            summary = report["summary"]
            status_emoji = {"healthy": "✅", "warning": "⚠️", "critical": "❌"}
            print(f"Overall Status: {status_emoji.get(summary['overall_status'], '❓')} {summary['overall_status'].upper()}")
            print(f"Total Checks: {summary['total_checks']}")
            print(f"Healthy: {summary['healthy']}")
            print(f"Warnings: {summary['warnings']}")
            print(f"Critical: {summary['critical']}")
            
            # Show details
            if results["critical"]:
                print("\n❌ Critical Issues:")
                for issue in results["critical"]:
                    print(f"  • {issue['name']}: {issue['message']}")
            
            if results["warnings"]:
                print("\n⚠️  Warnings:")
                for warning in results["warnings"]:
                    print(f"  • {warning['name']}: {warning['message']}")
            
            if results["healthy"] and not results["critical"] and not results["warnings"]:
                print("\n🎉 All health checks passed!")
        else:
            print(json.dumps(report, indent=2))
        
        # Exit with appropriate code
        if results["critical"]:
            sys.exit(2)
        elif results["warnings"]:
            sys.exit(1)
        else:
            sys.exit(0)
        
    except Exception as e:
        print(f"Error running health checks: {e}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()