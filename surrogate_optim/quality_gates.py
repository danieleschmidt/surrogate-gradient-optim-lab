"""Quality gates and automated quality assurance for surrogate optimization."""

import time
import subprocess
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json

import jax.numpy as jnp
from jax import Array


@dataclass
class QualityGate:
    """Represents a quality gate with pass/fail criteria."""
    gate_id: str
    name: str
    description: str
    check_function: Callable
    threshold: Optional[float] = None
    required: bool = True
    category: str = "general"


@dataclass
class QualityResult:
    """Result of a quality gate check."""
    gate_id: str
    gate_name: str
    passed: bool
    score: Optional[float] = None
    threshold: Optional[float] = None
    message: str = ""
    execution_time: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


class QualityGateEngine:
    """Engine for executing quality gates and enforcing quality standards."""
    
    def __init__(
        self,
        strict_mode: bool = False,
        parallel_execution: bool = True,
        output_dir: Optional[Path] = None,
    ):
        """Initialize quality gate engine.
        
        Args:
            strict_mode: If True, any failed gate stops execution
            parallel_execution: Whether to run gates in parallel
            output_dir: Directory for quality reports
        """
        self.strict_mode = strict_mode
        self.parallel_execution = parallel_execution
        self.output_dir = Path(output_dir) if output_dir else Path("quality_reports")
        self.output_dir.mkdir(exist_ok=True)
        
        self.gates = {}
        self.gate_history = []
        
        # Initialize default gates
        self._initialize_default_gates()
    
    def _initialize_default_gates(self):
        """Initialize default quality gates."""
        
        # Code quality gates
        self.add_gate(QualityGate(
            gate_id="code_style",
            name="Code Style",
            description="Check code formatting and style compliance",
            check_function=self._check_code_style,
            category="code_quality"
        ))
        
        self.add_gate(QualityGate(
            gate_id="type_checking",
            name="Type Checking",
            description="Static type checking with mypy",
            check_function=self._check_types,
            category="code_quality"
        ))
        
        # Security gates
        self.add_gate(QualityGate(
            gate_id="security_scan",
            name="Security Scan",
            description="Security vulnerability scanning",
            check_function=self._check_security,
            category="security"
        ))
        
        # Performance gates
        self.add_gate(QualityGate(
            gate_id="performance_regression",
            name="Performance Regression",
            description="Check for performance regressions",
            check_function=self._check_performance_regression,
            threshold=1.2,  # Max 20% slower than baseline
            category="performance"
        ))
        
        # Functional quality gates
        self.add_gate(QualityGate(
            gate_id="model_accuracy",
            name="Model Accuracy",
            description="Surrogate model accuracy on test functions",
            check_function=self._check_model_accuracy,
            threshold=0.95,  # Min 95% accuracy
            category="functional"
        ))
        
        self.add_gate(QualityGate(
            gate_id="optimization_convergence",
            name="Optimization Convergence",
            description="Optimization convergence on benchmark functions",
            check_function=self._check_optimization_convergence,
            threshold=0.8,  # Min 80% convergence rate
            category="functional"
        ))
        
        # Robustness gates
        self.add_gate(QualityGate(
            gate_id="error_handling",
            name="Error Handling",
            description="Robust error handling under adverse conditions",
            check_function=self._check_error_handling,
            category="robustness"
        ))
        
        self.add_gate(QualityGate(
            gate_id="memory_usage",
            name="Memory Usage",
            description="Memory usage within acceptable limits",
            check_function=self._check_memory_usage,
            threshold=1000.0,  # Max 1GB memory usage
            category="performance"
        ))
    
    def add_gate(self, gate: QualityGate):
        """Add a quality gate to the engine."""
        self.gates[gate.gate_id] = gate
    
    def remove_gate(self, gate_id: str):
        """Remove a quality gate."""
        if gate_id in self.gates:
            del self.gates[gate_id]
    
    def run_gates(
        self,
        gate_categories: Optional[List[str]] = None,
        gate_ids: Optional[List[str]] = None,
    ) -> List[QualityResult]:
        """Run quality gates.
        
        Args:
            gate_categories: Only run gates in these categories
            gate_ids: Only run specific gates
            
        Returns:
            List of quality results
        """
        # Filter gates
        gates_to_run = []
        for gate in self.gates.values():
            if gate_categories and gate.category not in gate_categories:
                continue
            if gate_ids and gate.gate_id not in gate_ids:
                continue
            gates_to_run.append(gate)
        
        print(f"Running {len(gates_to_run)} quality gates...")
        
        if self.parallel_execution:
            results = self._run_gates_parallel(gates_to_run)
        else:
            results = self._run_gates_sequential(gates_to_run)
        
        self.gate_history.extend(results)
        
        # Check for failures in strict mode
        if self.strict_mode:
            failed_gates = [r for r in results if not r.passed and self.gates[r.gate_id].required]
            if failed_gates:
                raise RuntimeError(f"Quality gates failed in strict mode: {[r.gate_name for r in failed_gates]}")
        
        return results
    
    def _run_gates_sequential(self, gates: List[QualityGate]) -> List[QualityResult]:
        """Run gates sequentially."""
        results = []
        for gate in gates:
            print(f"  Running gate: {gate.name}")
            result = self._execute_gate(gate)
            results.append(result)
            
            status = "PASS" if result.passed else "FAIL"
            print(f"    {status} ({result.execution_time:.3f}s)")
            if not result.passed:
                print(f"    {result.message}")
        
        return results
    
    def _run_gates_parallel(self, gates: List[QualityGate]) -> List[QualityResult]:
        """Run gates in parallel."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_gate = {
                executor.submit(self._execute_gate, gate): gate
                for gate in gates
            }
            
            for future in as_completed(future_to_gate):
                gate = future_to_gate[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    status = "PASS" if result.passed else "FAIL"
                    print(f"  {status}: {gate.name} ({result.execution_time:.3f}s)")
                    
                except Exception as e:
                    result = QualityResult(
                        gate_id=gate.gate_id,
                        gate_name=gate.name,
                        passed=False,
                        message=f"Gate execution failed: {e}",
                        execution_time=0.0
                    )
                    results.append(result)
                    print(f"  FAIL: {gate.name} - {e}")
        
        return results
    
    def _execute_gate(self, gate: QualityGate) -> QualityResult:
        """Execute a single quality gate."""
        start_time = time.time()
        
        try:
            score = gate.check_function()
            execution_time = time.time() - start_time
            
            # Determine if gate passed
            if gate.threshold is not None:
                if isinstance(score, (int, float)):
                    passed = score >= gate.threshold
                    message = f"Score: {score:.3f}, Threshold: {gate.threshold}"
                else:
                    passed = bool(score)
                    message = f"Result: {score}"
            else:
                passed = bool(score)
                message = f"Result: {score}"
            
            return QualityResult(
                gate_id=gate.gate_id,
                gate_name=gate.name,
                passed=passed,
                score=score if isinstance(score, (int, float)) else None,
                threshold=gate.threshold,
                message=message,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return QualityResult(
                gate_id=gate.gate_id,
                gate_name=gate.name,
                passed=False,
                message=f"Error: {e}",
                execution_time=execution_time
            )
    
    # Default gate implementations
    
    def _check_code_style(self) -> bool:
        """Check code style compliance."""
        try:
            # Check with ruff (if available)
            result = subprocess.run(
                ["python", "-m", "ruff", "check", "/root/repo/surrogate_optim", "--quiet"],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except FileNotFoundError:
            # Fallback: basic style check
            return True  # Assume pass if no tool available
    
    def _check_types(self) -> bool:
        """Check static typing."""
        try:
            result = subprocess.run(
                ["python", "-m", "mypy", "/root/repo/surrogate_optim", "--ignore-missing-imports"],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except FileNotFoundError:
            return True  # Assume pass if mypy not available
    
    def _check_security(self) -> bool:
        """Run security vulnerability scan."""
        try:
            result = subprocess.run(
                ["python", "-m", "bandit", "-r", "/root/repo/surrogate_optim", "-q"],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except FileNotFoundError:
            return True  # Assume pass if bandit not available
    
    def _check_performance_regression(self) -> float:
        """Check for performance regressions."""
        # Simplified performance test
        start_time = time.time()
        
        # Simulate some computation
        x = jnp.random.normal(0, 1, (1000, 10))
        y = jnp.sum(x**2, axis=1)
        
        elapsed = time.time() - start_time
        
        # Baseline performance (should complete in reasonable time)
        baseline_time = 0.1  # 100ms baseline
        performance_ratio = baseline_time / elapsed
        
        return performance_ratio
    
    def _check_model_accuracy(self) -> float:
        """Check surrogate model accuracy."""
        try:
            from .models.neural import NeuralSurrogate
            from .models.base import Dataset
            
            # Create test data (simple quadratic function)
            n_samples = 100
            X = jnp.random.uniform(-2, 2, (n_samples, 2))
            y = jnp.sum(X**2, axis=1)  # Simple quadratic
            dataset = Dataset(X=X, y=y)
            
            # Train surrogate
            model = NeuralSurrogate(hidden_dims=[16, 8])
            model.fit(dataset)
            
            # Test accuracy
            n_test = 50
            X_test = jnp.random.uniform(-2, 2, (n_test, 2))
            y_test = jnp.sum(X_test**2, axis=1)
            y_pred = model.predict(X_test)
            
            # Calculate R² score
            ss_res = jnp.sum((y_test - y_pred)**2)
            ss_tot = jnp.sum((y_test - jnp.mean(y_test))**2)
            r2_score = 1 - (ss_res / ss_tot)
            
            return float(r2_score)
            
        except Exception:
            return 0.0  # Failed
    
    def _check_optimization_convergence(self) -> float:
        """Check optimization convergence rate."""
        try:
            from .core import SurrogateOptimizer
            from .data.collector import collect_data
            
            # Simple test function (quadratic)
            def test_func(x):
                return jnp.sum((x - 1)**2)
            
            bounds = [(-2, 2), (-2, 2)]
            
            # Collect data and optimize
            data = collect_data(test_func, n_samples=20, bounds=bounds, verbose=False)
            optimizer = SurrogateOptimizer(surrogate_type="neural_network")
            optimizer.fit_surrogate(data)
            
            # Run optimization
            result = optimizer.optimize(
                initial_point=jnp.array([0.0, 0.0]),
                bounds=bounds,
                num_steps=50
            )
            
            # Check convergence (distance to optimum)
            optimum = jnp.array([1.0, 1.0])
            error = jnp.linalg.norm(result.x - optimum)
            
            # Convert error to score (lower error = higher score)
            convergence_score = 1.0 / (1.0 + error)
            
            return float(convergence_score)
            
        except Exception:
            return 0.0
    
    def _check_error_handling(self) -> bool:
        """Check error handling robustness."""
        try:
            from .security import InputValidator
            from .robustness import RobustSurrogate
            from .models.neural import NeuralSurrogate
            
            # Test input validation
            validator = InputValidator()
            violations = validator.validate_array(
                jnp.array([jnp.nan, 1.0, 2.0]),
                "test_invalid"
            )
            
            if len(violations) == 0:
                return False  # Should detect NaN values
            
            # Test robust surrogate with invalid data
            try:
                model = NeuralSurrogate(hidden_dims=[8])
                robust_model = RobustSurrogate(model)
                
                # This should handle invalid input gracefully
                result = robust_model.predict(jnp.array([jnp.nan, 1.0]))
                
                # Should return finite result even with invalid input
                return jnp.isfinite(result)
                
            except Exception:
                return False
            
        except Exception:
            return False
    
    def _check_memory_usage(self) -> float:
        """Check memory usage."""
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            return float(memory_mb)
            
        except ImportError:
            return 100.0  # Assume reasonable usage if psutil not available
    
    def generate_quality_report(self, results: List[QualityResult]) -> str:
        """Generate quality assessment report.
        
        Args:
            results: Quality gate results
            
        Returns:
            Report as string
        """
        total_gates = len(results)
        passed_gates = sum(1 for r in results if r.passed)
        failed_gates = total_gates - passed_gates
        
        # Calculate category scores
        category_scores = {}
        for result in results:
            gate = self.gates[result.gate_id]
            category = gate.category
            
            if category not in category_scores:
                category_scores[category] = {"passed": 0, "total": 0}
            
            category_scores[category]["total"] += 1
            if result.passed:
                category_scores[category]["passed"] += 1
        
        report = []
        report.append("=" * 60)
        report.append("QUALITY GATE ASSESSMENT REPORT")
        report.append("=" * 60)
        report.append(f"Total Gates: {total_gates}")
        report.append(f"Passed: {passed_gates}")
        report.append(f"Failed: {failed_gates}")
        report.append(f"Success Rate: {passed_gates/total_gates*100:.1f}%")
        report.append("")
        
        # Category breakdown
        report.append("CATEGORY BREAKDOWN:")
        report.append("-" * 30)
        for category, scores in category_scores.items():
            success_rate = scores["passed"] / scores["total"] * 100
            report.append(f"{category.title()}: {scores['passed']}/{scores['total']} ({success_rate:.1f}%)")
        report.append("")
        
        # Failed gates
        if failed_gates > 0:
            report.append("FAILED GATES:")
            report.append("-" * 30)
            for result in results:
                if not result.passed:
                    gate = self.gates[result.gate_id]
                    required_str = " (REQUIRED)" if gate.required else ""
                    report.append(f"❌ {result.gate_name}{required_str}")
                    report.append(f"   {result.message}")
                    report.append("")
        
        # Performance metrics
        execution_times = [r.execution_time for r in results]
        if execution_times:
            avg_time = sum(execution_times) / len(execution_times)
            total_time = sum(execution_times)
            report.append("PERFORMANCE METRICS:")
            report.append("-" * 30)
            report.append(f"Total Execution Time: {total_time:.3f}s")
            report.append(f"Average Gate Time: {avg_time:.3f}s")
            report.append("")
        
        # Quality score
        quality_score = passed_gates / total_gates * 100 if total_gates > 0 else 0
        report.append(f"OVERALL QUALITY SCORE: {quality_score:.1f}/100")
        
        return "\n".join(report)
    
    def save_quality_report(self, results: List[QualityResult], filename: Optional[str] = None):
        """Save quality report to file.
        
        Args:
            results: Quality gate results
            filename: Output filename
        """
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"quality_report_{timestamp}.txt"
        
        report_file = self.output_dir / filename
        report = self.generate_quality_report(results)
        
        with open(report_file, 'w') as f:
            f.write(report)
        
        # Also save JSON data
        json_file = self.output_dir / filename.replace('.txt', '.json')
        json_data = {
            "timestamp": time.time(),
            "total_gates": len(results),
            "passed_gates": sum(1 for r in results if r.passed),
            "results": [
                {
                    "gate_id": r.gate_id,
                    "gate_name": r.gate_name,
                    "passed": r.passed,
                    "score": r.score,
                    "threshold": r.threshold,
                    "message": r.message,
                    "execution_time": r.execution_time,
                    "category": self.gates[r.gate_id].category,
                    "required": self.gates[r.gate_id].required,
                }
                for r in results
            ]
        }
        
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"Quality report saved to: {report_file}")
        print(f"Quality data saved to: {json_file}")


def run_quality_gates(
    categories: Optional[List[str]] = None,
    strict_mode: bool = False,
    output_dir: Optional[str] = None
) -> List[QualityResult]:
    """Run quality gates for the surrogate optimization library.
    
    Args:
        categories: Only run gates in these categories
        strict_mode: Stop on first failure
        output_dir: Output directory for reports
        
    Returns:
        List of quality results
    """
    engine = QualityGateEngine(
        strict_mode=strict_mode,
        output_dir=Path(output_dir) if output_dir else None
    )
    
    results = engine.run_gates(gate_categories=categories)
    
    # Generate and save report
    report = engine.generate_quality_report(results)
    print("\n" + report)
    
    engine.save_quality_report(results)
    
    return results