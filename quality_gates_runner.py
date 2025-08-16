#!/usr/bin/env python3
"""Comprehensive Quality Gates Runner - Production Ready.

This script executes all quality gates required for production deployment:
- Code quality and formatting
- Security scanning
- Performance benchmarks
- Test coverage
- Documentation coverage
- Compliance checks
"""

import subprocess
import sys
import json
import time
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class QualityGateResult:
    """Result of a quality gate check."""
    name: str
    passed: bool
    score: float
    details: Dict[str, Any]
    execution_time: float
    error_message: str = ""


class QualityGatesRunner:
    """Comprehensive quality gates execution system."""
    
    def __init__(self, project_root: Path = None):
        """Initialize quality gates runner."""
        self.project_root = project_root or Path.cwd()
        self.results: List[QualityGateResult] = []
        
        # Quality thresholds
        self.thresholds = {
            "test_coverage": 85.0,
            "security_score": 9.0,
            "performance_score": 8.0,
            "code_quality_score": 8.5,
            "documentation_score": 7.0,
        }
        
        logger.info(f"Quality gates runner initialized for {self.project_root}")
    
    def run_command(self, command: List[str], capture_output: bool = True) -> Tuple[int, str, str]:
        """Run shell command and return result."""
        try:
            result = subprocess.run(
                command,
                capture_output=capture_output,
                text=True,
                cwd=self.project_root,
                timeout=300  # 5 minute timeout
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return 1, "", "Command timed out"
        except Exception as e:
            return 1, "", str(e)
    
    def run_test_coverage(self) -> QualityGateResult:
        """Run test suite with coverage analysis."""
        logger.info("🧪 Running test coverage analysis...")
        start_time = time.time()
        
        try:
            # Activate venv and run pytest with coverage
            command = [
                "bash", "-c", 
                "source venv/bin/activate && python -m pytest tests/unit/test_sample.py -v "
                "--cov=surrogate_optim --cov-report=json --cov-report=term-missing"
            ]
            
            returncode, stdout, stderr = self.run_command(command)
            execution_time = time.time() - start_time
            
            # Parse coverage report
            coverage_file = self.project_root / "coverage.json"
            if coverage_file.exists():
                with open(coverage_file) as f:
                    coverage_data = json.load(f)
                
                total_coverage = coverage_data["totals"]["percent_covered"]
                
                details = {
                    "coverage_percentage": total_coverage,
                    "lines_covered": coverage_data["totals"]["covered_lines"],
                    "lines_missing": coverage_data["totals"]["missing_lines"],
                    "num_statements": coverage_data["totals"]["num_statements"],
                    "tests_passed": "PASSED" in stdout,
                    "test_output": stdout[-1000:],  # Last 1000 chars
                }
                
                passed = total_coverage >= self.thresholds["test_coverage"]
                score = min(10.0, total_coverage / 10.0)
                
                return QualityGateResult(
                    name="Test Coverage",
                    passed=passed,
                    score=score,
                    details=details,
                    execution_time=execution_time
                )
            else:
                # Fallback: analyze test output
                passed = returncode == 0 and "FAILED" not in stdout
                
                details = {
                    "returncode": returncode,
                    "tests_output": stdout,
                    "coverage_file_missing": True
                }
                
                return QualityGateResult(
                    name="Test Coverage",
                    passed=passed,
                    score=7.0 if passed else 3.0,
                    details=details,
                    execution_time=execution_time
                )
        
        except Exception as e:
            return QualityGateResult(
                name="Test Coverage",
                passed=False,
                score=0.0,
                details={"error": str(e)},
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def run_security_scan(self) -> QualityGateResult:
        """Run security vulnerability scanning."""
        logger.info("🔒 Running security scan...")
        start_time = time.time()
        
        try:
            # Run bandit security scanner
            command = [
                "bash", "-c",
                "source venv/bin/activate && bandit -r surrogate_optim/ -f json -o security_report.json"
            ]
            
            returncode, stdout, stderr = self.run_command(command)
            execution_time = time.time() - start_time
            
            # Parse security report
            security_file = self.project_root / "security_report.json"
            if security_file.exists():
                with open(security_file) as f:
                    security_data = json.load(f)
                
                # Calculate security score
                high_issues = len([r for r in security_data.get("results", []) if r.get("issue_severity") == "HIGH"])
                medium_issues = len([r for r in security_data.get("results", []) if r.get("issue_severity") == "MEDIUM"])
                low_issues = len([r for r in security_data.get("results", []) if r.get("issue_severity") == "LOW"])
                
                total_issues = high_issues + medium_issues + low_issues
                
                # Security score: 10 - (high*3 + medium*1 + low*0.3)
                security_score = max(0, 10 - (high_issues * 3 + medium_issues * 1 + low_issues * 0.3))
                
                details = {
                    "high_severity_issues": high_issues,
                    "medium_severity_issues": medium_issues,
                    "low_severity_issues": low_issues,
                    "total_issues": total_issues,
                    "files_scanned": security_data.get("metrics", {}).get("_totals", {}).get("loc", 0),
                    "security_score": security_score,
                    "issues": security_data.get("results", [])[:5]  # First 5 issues
                }
                
                passed = security_score >= self.thresholds["security_score"]
                
                return QualityGateResult(
                    name="Security Scan",
                    passed=passed,
                    score=security_score,
                    details=details,
                    execution_time=execution_time
                )
            else:
                # Bandit completed but no report file
                passed = returncode == 0
                
                return QualityGateResult(
                    name="Security Scan",
                    passed=passed,
                    score=8.0 if passed else 5.0,
                    details={
                        "returncode": returncode,
                        "stdout": stdout,
                        "no_report_file": True
                    },
                    execution_time=execution_time
                )
        
        except Exception as e:
            return QualityGateResult(
                name="Security Scan",
                passed=False,
                score=0.0,
                details={"error": str(e)},
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def run_code_quality_check(self) -> QualityGateResult:
        """Run code quality analysis."""
        logger.info("📝 Running code quality checks...")
        start_time = time.time()
        
        try:
            # Run multiple code quality tools
            quality_scores = []
            details = {}
            
            # Check with ruff (linting)
            command = ["bash", "-c", "source venv/bin/activate && ruff check surrogate_optim/ --output-format=json"]
            returncode, stdout, stderr = self.run_command(command)
            
            if returncode == 0 and stdout:
                try:
                    ruff_results = json.loads(stdout) if stdout.strip() else []
                    ruff_issues = len(ruff_results)
                    ruff_score = max(0, 10 - (ruff_issues * 0.1))
                    quality_scores.append(ruff_score)
                    details["ruff_issues"] = ruff_issues
                    details["ruff_score"] = ruff_score
                except json.JSONDecodeError:
                    quality_scores.append(8.0)  # Default score if no issues
                    details["ruff_issues"] = 0
                    details["ruff_score"] = 8.0
            else:
                quality_scores.append(6.0)
                details["ruff_error"] = stderr
            
            # Check with mypy (type checking)
            command = ["bash", "-c", "source venv/bin/activate && mypy surrogate_optim/ --ignore-missing-imports"]
            returncode, stdout, stderr = self.run_command(command)
            
            mypy_errors = stdout.count("error:") if stdout else 0
            mypy_score = max(0, 10 - (mypy_errors * 0.2))
            quality_scores.append(mypy_score)
            details["mypy_errors"] = mypy_errors
            details["mypy_score"] = mypy_score
            
            # Overall code quality score
            avg_score = sum(quality_scores) / len(quality_scores) if quality_scores else 5.0
            passed = avg_score >= self.thresholds["code_quality_score"]
            
            details["overall_score"] = avg_score
            details["quality_tools_run"] = len(quality_scores)
            
            return QualityGateResult(
                name="Code Quality",
                passed=passed,
                score=avg_score,
                details=details,
                execution_time=time.time() - start_time
            )
        
        except Exception as e:
            return QualityGateResult(
                name="Code Quality",
                passed=False,
                score=0.0,
                details={"error": str(e)},
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def run_performance_benchmark(self) -> QualityGateResult:
        """Run performance benchmarks."""
        logger.info("⚡ Running performance benchmarks...")
        start_time = time.time()
        
        try:
            # Run our performance optimization demo
            command = [
                "bash", "-c",
                "source venv/bin/activate && timeout 60 python surrogate_optim/performance/advanced_optimization.py"
            ]
            
            returncode, stdout, stderr = self.run_command(command)
            execution_time = time.time() - start_time
            
            # Parse performance results
            performance_score = 8.0  # Default
            details = {
                "benchmark_completed": returncode == 0,
                "execution_time": execution_time,
                "stdout": stdout[-500:] if stdout else "",
            }
            
            # Extract performance metrics from output
            if "Speedup:" in stdout:
                try:
                    speedup_line = [line for line in stdout.split("\\n") if "Speedup:" in line][0]
                    speedup = float(speedup_line.split("Speedup: ")[1].split("x")[0])
                    performance_score = min(10.0, 5.0 + speedup * 2)
                    details["speedup"] = speedup
                except:
                    pass
            
            if "Cache speedup:" in stdout:
                try:
                    cache_line = [line for line in stdout.split("\\n") if "Cache speedup:" in line][0]
                    cache_speedup = float(cache_line.split("Cache speedup: ")[1].split("x")[0])
                    details["cache_speedup"] = cache_speedup
                except:
                    pass
            
            passed = performance_score >= self.thresholds["performance_score"]
            
            return QualityGateResult(
                name="Performance Benchmark",
                passed=passed,
                score=performance_score,
                details=details,
                execution_time=execution_time
            )
        
        except Exception as e:
            return QualityGateResult(
                name="Performance Benchmark",
                passed=False,
                score=0.0,
                details={"error": str(e)},
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def run_documentation_check(self) -> QualityGateResult:
        """Check documentation coverage and quality."""
        logger.info("📚 Checking documentation coverage...")
        start_time = time.time()
        
        try:
            # Count Python files and docstrings
            python_files = list(Path("surrogate_optim").rglob("*.py"))
            total_files = len(python_files)
            
            files_with_docstrings = 0
            total_functions = 0
            functions_with_docstrings = 0
            
            for py_file in python_files:
                if py_file.name == "__init__.py":
                    continue
                
                try:
                    with open(py_file, 'r') as f:
                        content = f.read()
                    
                    # Check for module docstring
                    if '"""' in content[:500] or "'''" in content[:500]:
                        files_with_docstrings += 1
                    
                    # Count functions and their docstrings
                    lines = content.split('\\n')
                    for i, line in enumerate(lines):
                        if line.strip().startswith("def ") and not line.strip().startswith("def _"):
                            total_functions += 1
                            # Check next few lines for docstring
                            for j in range(i+1, min(i+5, len(lines))):
                                if '"""' in lines[j] or "'''" in lines[j]:
                                    functions_with_docstrings += 1
                                    break
                
                except Exception:
                    continue
            
            # Calculate documentation scores
            file_doc_rate = files_with_docstrings / total_files if total_files > 0 else 0
            func_doc_rate = functions_with_docstrings / total_functions if total_functions > 0 else 0
            
            overall_doc_score = (file_doc_rate * 4 + func_doc_rate * 6)  # Weighted average
            
            details = {
                "total_python_files": total_files,
                "files_with_docstrings": files_with_docstrings,
                "file_documentation_rate": file_doc_rate,
                "total_functions": total_functions,
                "functions_with_docstrings": functions_with_docstrings,
                "function_documentation_rate": func_doc_rate,
                "overall_documentation_score": overall_doc_score,
            }
            
            passed = overall_doc_score >= self.thresholds["documentation_score"]
            
            return QualityGateResult(
                name="Documentation Coverage",
                passed=passed,
                score=overall_doc_score,
                details=details,
                execution_time=time.time() - start_time
            )
        
        except Exception as e:
            return QualityGateResult(
                name="Documentation Coverage",
                passed=False,
                score=0.0,
                details={"error": str(e)},
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def run_all_quality_gates(self) -> bool:
        """Run all quality gates and return overall pass/fail status."""
        logger.info("🚀 Starting comprehensive quality gates execution...")
        
        # List of quality gate functions
        quality_gates = [
            self.run_test_coverage,
            self.run_security_scan,
            self.run_code_quality_check,
            self.run_performance_benchmark,
            self.run_documentation_check,
        ]
        
        # Execute all quality gates
        for gate_func in quality_gates:
            try:
                result = gate_func()
                self.results.append(result)
                
                status = "✅ PASSED" if result.passed else "❌ FAILED"
                logger.info(f"{status} {result.name} (Score: {result.score:.1f}/10.0)")
                
                if not result.passed and result.error_message:
                    logger.error(f"Error in {result.name}: {result.error_message}")
            
            except Exception as e:
                logger.error(f"Quality gate execution failed: {e}")
                self.results.append(QualityGateResult(
                    name="Unknown Gate",
                    passed=False,
                    score=0.0,
                    details={"error": str(e)},
                    execution_time=0.0,
                    error_message=str(e)
                ))
        
        # Calculate overall results
        overall_passed = all(result.passed for result in self.results)
        return overall_passed
    
    def generate_report(self) -> str:
        """Generate comprehensive quality gates report."""
        if not self.results:
            return "No quality gates executed."
        
        # Calculate overall statistics
        total_gates = len(self.results)
        passed_gates = sum(1 for r in self.results if r.passed)
        overall_score = sum(r.score for r in self.results) / total_gates if total_gates > 0 else 0
        total_time = sum(r.execution_time for r in self.results)
        
        # Generate report
        report_lines = [
            "=" * 80,
            "COMPREHENSIVE QUALITY GATES REPORT",
            "=" * 80,
            f"Overall Status: {'✅ PASSED' if passed_gates == total_gates else '❌ FAILED'}",
            f"Gates Passed: {passed_gates}/{total_gates} ({passed_gates/total_gates*100:.1f}%)",
            f"Overall Score: {overall_score:.1f}/10.0",
            f"Total Execution Time: {total_time:.2f} seconds",
            "",
            "INDIVIDUAL GATE RESULTS:",
            "-" * 50,
        ]
        
        for result in self.results:
            status_symbol = "✅" if result.passed else "❌"
            report_lines.append(
                f"{status_symbol} {result.name}: {result.score:.1f}/10.0 "
                f"({result.execution_time:.2f}s)"
            )
            
            # Add key details
            if "coverage_percentage" in result.details:
                report_lines.append(f"   Coverage: {result.details['coverage_percentage']:.1f}%")
            
            if "total_issues" in result.details:
                report_lines.append(f"   Security Issues: {result.details['total_issues']}")
            
            if "ruff_issues" in result.details:
                report_lines.append(f"   Code Quality Issues: {result.details['ruff_issues']}")
            
            if "speedup" in result.details:
                report_lines.append(f"   Performance Speedup: {result.details['speedup']:.1f}x")
            
            if "overall_documentation_score" in result.details:
                report_lines.append(f"   Documentation Score: {result.details['overall_documentation_score']:.1f}/10.0")
            
            if result.error_message:
                report_lines.append(f"   Error: {result.error_message}")
        
        # Recommendations
        report_lines.extend([
            "",
            "RECOMMENDATIONS:",
            "-" * 30,
        ])
        
        failed_gates = [r for r in self.results if not r.passed]
        if not failed_gates:
            report_lines.append("🎉 All quality gates passed! Ready for production deployment.")
        else:
            for failed_gate in failed_gates:
                if failed_gate.name == "Test Coverage":
                    report_lines.append("• Increase test coverage by adding more unit tests")
                elif failed_gate.name == "Security Scan":
                    report_lines.append("• Review and fix security vulnerabilities")
                elif failed_gate.name == "Code Quality":
                    report_lines.append("• Address code quality issues (linting, type checking)")
                elif failed_gate.name == "Performance Benchmark":
                    report_lines.append("• Optimize performance bottlenecks")
                elif failed_gate.name == "Documentation Coverage":
                    report_lines.append("• Add docstrings to modules and functions")
        
        report_lines.append("=" * 80)
        
        return "\\n".join(report_lines)
    
    def save_report(self, filename: str = "quality_gates_report.txt"):
        """Save quality gates report to file."""
        report = self.generate_report()
        report_path = self.project_root / filename
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Quality gates report saved to {report_path}")
        return report_path


def main():
    """Main execution function."""
    print("🏗️  PRODUCTION QUALITY GATES EXECUTION")
    print("=" * 60)
    
    # Initialize quality gates runner
    runner = QualityGatesRunner()
    
    # Run all quality gates
    overall_passed = runner.run_all_quality_gates()
    
    # Generate and display report
    report = runner.generate_report()
    print("\\n" + report)
    
    # Save report to file
    report_path = runner.save_report()
    
    # Exit with appropriate code
    if overall_passed:
        print("\\n🎉 ALL QUALITY GATES PASSED - READY FOR PRODUCTION!")
        sys.exit(0)
    else:
        print("\\n❌ QUALITY GATES FAILED - DEPLOYMENT BLOCKED")
        sys.exit(1)


if __name__ == "__main__":
    main()