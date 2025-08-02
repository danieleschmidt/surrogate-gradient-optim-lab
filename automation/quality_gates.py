"""
Quality gates automation for ensuring code quality standards.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import argparse


class QualityGate:
    """Base class for quality gates."""
    
    def __init__(self, name: str, threshold: Optional[Any] = None):
        self.name = name
        self.threshold = threshold
    
    def check(self) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Check if quality gate passes.
        
        Returns:
            Tuple of (passed, message, details)
        """
        raise NotImplementedError
    
    def __str__(self):
        return f"QualityGate({self.name})"


class CodeCoverageGate(QualityGate):
    """Quality gate for code coverage."""
    
    def __init__(self, threshold: float = 80.0, repo_path: str = "."):
        super().__init__("Code Coverage", threshold)
        self.repo_path = Path(repo_path)
    
    def check(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Check code coverage against threshold."""
        try:
            result = subprocess.run(
                ["python3", "-m", "pytest", "--cov=surrogate_optim", "--cov-report=json", "--tb=no", "-q"],
                capture_output=True, text=True, cwd=self.repo_path
            )
            
            coverage_file = self.repo_path / "coverage.json"
            if coverage_file.exists():
                with open(coverage_file) as f:
                    coverage_data = json.load(f)
                
                total_coverage = coverage_data["totals"]["percent_covered"]
                
                passed = total_coverage >= self.threshold
                message = f"Coverage: {total_coverage:.1f}% (threshold: {self.threshold}%)"
                
                return passed, message, {
                    "coverage": total_coverage,
                    "threshold": self.threshold,
                    "lines_covered": coverage_data["totals"]["covered_lines"],
                    "total_lines": coverage_data["totals"]["num_statements"]
                }
            else:
                return False, "Coverage report not generated", {}
                
        except Exception as e:
            return False, f"Failed to check coverage: {str(e)}", {}


class TestPassingGate(QualityGate):
    """Quality gate for test passing."""
    
    def __init__(self, repo_path: str = "."):
        super().__init__("Test Passing")
        self.repo_path = Path(repo_path)
    
    def check(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Check if all tests pass."""
        try:
            result = subprocess.run(
                ["python3", "-m", "pytest", "--tb=no", "-v"],
                capture_output=True, text=True, cwd=self.repo_path
            )
            
            passed = result.returncode == 0
            
            # Parse test results
            output_lines = result.stdout.split('\n')
            test_results = {"passed": 0, "failed": 0, "skipped": 0}
            
            for line in output_lines:
                if " PASSED " in line:
                    test_results["passed"] += 1
                elif " FAILED " in line:
                    test_results["failed"] += 1
                elif " SKIPPED " in line:
                    test_results["skipped"] += 1
            
            total_tests = sum(test_results.values())
            message = f"Tests: {test_results['passed']}/{total_tests} passed"
            
            if not passed:
                message += f", {test_results['failed']} failed"
            
            return passed, message, test_results
            
        except Exception as e:
            return False, f"Failed to run tests: {str(e)}", {}


class LintingGate(QualityGate):
    """Quality gate for linting."""
    
    def __init__(self, max_violations: int = 0, repo_path: str = "."):
        super().__init__("Linting", max_violations)
        self.repo_path = Path(repo_path)
    
    def check(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Check linting violations."""
        try:
            # Run ruff
            result = subprocess.run(
                ["python3", "-m", "ruff", "check", "surrogate_optim", "--format=json"],
                capture_output=True, text=True, cwd=self.repo_path
            )
            
            violations = []
            if result.stdout.strip():
                violations = json.loads(result.stdout)
            
            violation_count = len(violations)
            passed = violation_count <= self.threshold
            
            message = f"Linting: {violation_count} violations (max: {self.threshold})"
            
            return passed, message, {
                "violation_count": violation_count,
                "max_violations": self.threshold,
                "violations": violations[:10]  # Limit details
            }
            
        except Exception as e:
            return False, f"Failed to run linting: {str(e)}", {}


class SecurityGate(QualityGate):
    """Quality gate for security issues."""
    
    def __init__(self, max_vulnerabilities: int = 0, repo_path: str = "."):
        super().__init__("Security", max_vulnerabilities)
        self.repo_path = Path(repo_path)
    
    def check(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Check for security vulnerabilities."""
        try:
            # Run bandit for security issues
            result = subprocess.run(
                ["python3", "-m", "bandit", "-r", "surrogate_optim", "-f", "json"],
                capture_output=True, text=True, cwd=self.repo_path
            )
            
            issues = []
            if result.stdout.strip():
                try:
                    bandit_data = json.loads(result.stdout)
                    issues = bandit_data.get("results", [])
                except json.JSONDecodeError:
                    pass
            
            # Check for dependency vulnerabilities
            vuln_count = 0
            try:
                pip_audit_result = subprocess.run(
                    ["python3", "-m", "pip_audit", "--format=json"],
                    capture_output=True, text=True
                )
                
                if pip_audit_result.returncode == 0:
                    audit_data = json.loads(pip_audit_result.stdout)
                    vuln_count = len(audit_data.get("vulnerabilities", []))
            except Exception:
                pass
            
            total_issues = len(issues) + vuln_count
            passed = total_issues <= self.threshold
            
            message = f"Security: {total_issues} issues (max: {self.threshold})"
            
            return passed, message, {
                "total_issues": total_issues,
                "bandit_issues": len(issues),
                "dependency_vulnerabilities": vuln_count,
                "max_issues": self.threshold
            }
            
        except Exception as e:
            return False, f"Failed to run security checks: {str(e)}", {}


class ComplexityGate(QualityGate):
    """Quality gate for code complexity."""
    
    def __init__(self, max_complexity: float = 10.0, repo_path: str = "."):
        super().__init__("Code Complexity", max_complexity)
        self.repo_path = Path(repo_path)
    
    def check(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Check code complexity."""
        try:
            result = subprocess.run(
                ["python3", "-m", "radon", "cc", "surrogate_optim", "--json"],
                capture_output=True, text=True, cwd=self.repo_path
            )
            
            if result.returncode == 0:
                complexity_data = json.loads(result.stdout)
                
                # Calculate average complexity
                total_complexity = 0
                function_count = 0
                high_complexity_functions = 0
                
                for file_data in complexity_data.values():
                    for item in file_data:
                        if item['type'] in ['function', 'method']:
                            complexity = item['complexity']
                            total_complexity += complexity
                            function_count += 1
                            
                            if complexity > self.threshold:
                                high_complexity_functions += 1
                
                avg_complexity = total_complexity / function_count if function_count > 0 else 0
                passed = avg_complexity <= self.threshold
                
                message = f"Complexity: {avg_complexity:.1f} avg (max: {self.threshold})"
                
                return passed, message, {
                    "average_complexity": avg_complexity,
                    "max_complexity": self.threshold,
                    "total_functions": function_count,
                    "high_complexity_functions": high_complexity_functions
                }
            else:
                return False, "Failed to analyze complexity", {}
                
        except Exception as e:
            return False, f"Failed to check complexity: {str(e)}", {}


class PerformanceGate(QualityGate):
    """Quality gate for performance benchmarks."""
    
    def __init__(self, max_regression_percent: float = 10.0, repo_path: str = "."):
        super().__init__("Performance", max_regression_percent)
        self.repo_path = Path(repo_path)
    
    def check(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Check performance benchmarks."""
        try:
            # Check if benchmarks exist
            benchmark_path = self.repo_path / "tests" / "benchmarks"
            if not benchmark_path.exists():
                return True, "No benchmarks to check", {"benchmarks_exist": False}
            
            # Run benchmarks
            result = subprocess.run(
                ["python3", "-m", "pytest", "tests/benchmarks/", "--tb=no", "-v"],
                capture_output=True, text=True, cwd=self.repo_path,
                timeout=120  # 2 minute timeout
            )
            
            passed = result.returncode == 0
            message = "Benchmarks passed" if passed else "Benchmarks failed"
            
            # Parse benchmark results (simplified)
            benchmark_count = result.stdout.count("PASSED") + result.stdout.count("FAILED")
            
            return passed, message, {
                "benchmarks_passed": passed,
                "benchmark_count": benchmark_count,
                "max_regression": self.threshold
            }
            
        except subprocess.TimeoutExpired:
            return False, "Benchmarks timed out", {"timeout": True}
        except Exception as e:
            return False, f"Failed to run benchmarks: {str(e)}", {}


class QualityGateRunner:
    """Runs multiple quality gates and reports results."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = repo_path
        self.gates = []
    
    def add_gate(self, gate: QualityGate):
        """Add a quality gate."""
        self.gates.append(gate)
    
    def run_all(self) -> Dict[str, Any]:
        """Run all quality gates."""
        results = {
            "passed": [],
            "failed": [],
            "summary": {
                "total": len(self.gates),
                "passed": 0,
                "failed": 0,
                "overall_passed": True
            }
        }
        
        print(f"🚦 Running {len(self.gates)} quality gates...")
        
        for gate in self.gates:
            print(f"  Checking {gate.name}...", end=" ")
            
            try:
                passed, message, details = gate.check()
                
                gate_result = {
                    "name": gate.name,
                    "passed": passed,
                    "message": message,
                    "details": details
                }
                
                if passed:
                    results["passed"].append(gate_result)
                    results["summary"]["passed"] += 1
                    print("✅ PASS")
                else:
                    results["failed"].append(gate_result)
                    results["summary"]["failed"] += 1
                    results["summary"]["overall_passed"] = False
                    print("❌ FAIL")
                    
            except Exception as e:
                gate_result = {
                    "name": gate.name,
                    "passed": False,
                    "message": f"Error: {str(e)}",
                    "details": {}
                }
                results["failed"].append(gate_result)
                results["summary"]["failed"] += 1
                results["summary"]["overall_passed"] = False
                print("💥 ERROR")
        
        return results


def create_default_gates(repo_path: str = ".") -> List[QualityGate]:
    """Create default set of quality gates."""
    return [
        TestPassingGate(repo_path),
        CodeCoverageGate(80.0, repo_path),
        LintingGate(0, repo_path),
        SecurityGate(0, repo_path),
        ComplexityGate(10.0, repo_path),
        PerformanceGate(10.0, repo_path),
    ]


def main():
    """Main entry point for quality gate checking."""
    parser = argparse.ArgumentParser(description="Run quality gates")
    parser.add_argument("--repo-path", default=".", help="Path to repository")
    parser.add_argument("--output", "-o", help="Output file for results")
    parser.add_argument("--strict", action="store_true", help="Fail on any gate failure")
    parser.add_argument("--gates", nargs="+", 
                       choices=["tests", "coverage", "linting", "security", "complexity", "performance"],
                       help="Specific gates to run")
    
    args = parser.parse_args()
    
    runner = QualityGateRunner(args.repo_path)
    
    # Add gates based on arguments
    all_gates = {
        "tests": TestPassingGate(args.repo_path),
        "coverage": CodeCoverageGate(80.0, args.repo_path),
        "linting": LintingGate(0, args.repo_path),
        "security": SecurityGate(0, args.repo_path),
        "complexity": ComplexityGate(10.0, args.repo_path),
        "performance": PerformanceGate(10.0, args.repo_path),
    }
    
    if args.gates:
        for gate_name in args.gates:
            runner.add_gate(all_gates[gate_name])
    else:
        for gate in create_default_gates(args.repo_path):
            runner.add_gate(gate)
    
    try:
        results = runner.run_all()
        
        # Add metadata
        report = {
            "metadata": {
                "repository_path": str(Path(args.repo_path).absolute()),
                "runner_version": "1.0.0",
                "strict_mode": args.strict
            },
            "results": results
        }
        
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\nQuality gate report saved to: {output_path}")
        
        # Print summary
        summary = results["summary"]
        print(f"\n🚦 Quality Gates Summary")
        print("=" * 30)
        print(f"Total Gates: {summary['total']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        
        if results["failed"]:
            print("\n❌ Failed Gates:")
            for failure in results["failed"]:
                print(f"  • {failure['name']}: {failure['message']}")
        
        if summary["overall_passed"]:
            print("\n🎉 All quality gates passed!")
            sys.exit(0)
        else:
            print("\n💥 Some quality gates failed!")
            sys.exit(1 if args.strict else 0)
        
    except Exception as e:
        print(f"Error running quality gates: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()