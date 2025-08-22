"""AI-Enhanced Quality Gates for intelligent validation and testing.

This module implements next-generation quality gates that use AI and machine learning
to provide intelligent validation, automated test generation, and adaptive quality metrics.
"""

import time
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
from abc import ABC, abstractmethod
import logging
import ast
import re

import jax
import jax.numpy as jnp
from jax import Array
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from scipy import stats


class QualityGateStatus(Enum):
    """Status of quality gate evaluation."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    PENDING = "pending"
    SKIPPED = "skipped"


class TestType(Enum):
    """Types of tests that can be generated."""
    UNIT_TEST = "unit_test"
    INTEGRATION_TEST = "integration_test"
    PERFORMANCE_TEST = "performance_test"
    SECURITY_TEST = "security_test"
    REGRESSION_TEST = "regression_test"
    PROPERTY_TEST = "property_test"


@dataclass
class QualityMetric:
    """Represents a quality metric measurement."""
    name: str
    value: float
    threshold: float
    status: QualityGateStatus
    confidence: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CodeQualityAnalysis:
    """Results of code quality analysis."""
    complexity_score: float
    maintainability_index: float
    test_coverage: float
    documentation_coverage: float
    security_score: float
    performance_score: float
    code_smells: List[str]
    potential_bugs: List[str]
    suggestions: List[str]


@dataclass
class GeneratedTest:
    """Represents an automatically generated test."""
    test_type: TestType
    function_name: str
    test_code: str
    description: str
    confidence: float
    expected_coverage_increase: float


class AIQualityAnalyzer(ABC):
    """Abstract base class for AI-powered quality analyzers."""
    
    @abstractmethod
    def analyze(self, code: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze code quality using AI techniques."""
        pass
    
    @abstractmethod
    def get_confidence(self) -> float:
        """Get confidence in the analysis."""
        pass


class ComplexityAnalyzer(AIQualityAnalyzer):
    """AI-powered complexity analysis."""
    
    def __init__(self):
        """Initialize complexity analyzer."""
        self.complexity_threshold = 10  # McCabe complexity
        self.confidence = 0.9  # High confidence for static analysis
        
    def analyze(self, code: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze code complexity.
        
        Args:
            code: Source code to analyze
            context: Additional context information
            
        Returns:
            Complexity analysis results
        """
        try:
            # Parse code into AST
            tree = ast.parse(code)
            
            # Calculate complexity metrics
            complexity_scores = []
            function_complexities = {}
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    complexity = self._calculate_cyclomatic_complexity(node)
                    complexity_scores.append(complexity)
                    function_complexities[node.name] = complexity
            
            # Overall metrics
            avg_complexity = np.mean(complexity_scores) if complexity_scores else 0
            max_complexity = max(complexity_scores) if complexity_scores else 0
            
            # Identify complex functions
            complex_functions = [
                name for name, complexity in function_complexities.items()
                if complexity > self.complexity_threshold
            ]
            
            return {
                "average_complexity": avg_complexity,
                "max_complexity": max_complexity,
                "complex_functions": complex_functions,
                "function_complexities": function_complexities,
                "needs_refactoring": len(complex_functions) > 0
            }
            
        except SyntaxError as e:
            return {
                "error": f"Syntax error in code: {e}",
                "average_complexity": float('inf'),
                "max_complexity": float('inf'),
                "complex_functions": [],
                "function_complexities": {},
                "needs_refactoring": True
            }
    
    def _calculate_cyclomatic_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity for a function.
        
        Args:
            node: AST node representing the function
            
        Returns:
            Cyclomatic complexity score
        """
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            # Decision points increase complexity
            if isinstance(child, (ast.If, ast.While, ast.For, ast.With)):
                complexity += 1
            elif isinstance(child, ast.Try):
                complexity += len(child.handlers) + (1 if child.orelse else 0)
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(child, ast.comprehension):
                complexity += 1
        
        return complexity
    
    def get_confidence(self) -> float:
        """Get confidence in complexity analysis."""
        return self.confidence


class SecurityAnalyzer(AIQualityAnalyzer):
    """AI-powered security analysis."""
    
    def __init__(self):
        """Initialize security analyzer."""
        # Common security patterns to detect
        self.security_patterns = {
            r'eval\s*\(': "Use of eval() function",
            r'exec\s*\(': "Use of exec() function",
            r'input\s*\(': "Use of input() function (potential injection)",
            r'os\.system\s*\(': "Use of os.system() (command injection risk)",
            r'subprocess\.call\s*\(.*shell\s*=\s*True': "Shell injection risk",
            r'pickle\.loads?\s*\(': "Unsafe deserialization",
            r'yaml\.load\s*\(': "Unsafe YAML loading",
            r'md5\s*\(': "Use of weak MD5 hash",
            r'sha1\s*\(': "Use of weak SHA1 hash",
            r'random\.random\s*\(': "Use of weak random number generator",
        }
        
        self.confidence_base = 0.8
        
    def analyze(self, code: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze code for security issues.
        
        Args:
            code: Source code to analyze
            context: Additional context information
            
        Returns:
            Security analysis results
        """
        security_issues = []
        confidence_scores = []
        
        # Pattern-based detection
        for pattern, description in self.security_patterns.items():
            matches = re.finditer(pattern, code, re.IGNORECASE)
            for match in matches:
                line_num = code[:match.start()].count('\n') + 1
                security_issues.append({
                    "type": "security_pattern",
                    "description": description,
                    "line": line_num,
                    "severity": "high",
                    "code_snippet": match.group()
                })
                confidence_scores.append(0.9)  # High confidence for pattern matches
        
        # AST-based analysis
        try:
            tree = ast.parse(code)
            ast_issues = self._analyze_ast_security(tree)
            security_issues.extend(ast_issues)
            confidence_scores.extend([0.7] * len(ast_issues))  # Medium confidence for AST analysis
            
        except SyntaxError:
            security_issues.append({
                "type": "syntax_error",
                "description": "Code contains syntax errors",
                "line": 0,
                "severity": "medium",
                "code_snippet": ""
            })
            confidence_scores.append(0.8)
        
        # Calculate overall security score
        if not security_issues:
            security_score = 10.0  # Perfect score
        else:
            high_severity = sum(1 for issue in security_issues if issue["severity"] == "high")
            medium_severity = sum(1 for issue in security_issues if issue["severity"] == "medium")
            low_severity = sum(1 for issue in security_issues if issue["severity"] == "low")
            
            # Weighted penalty system
            penalty = high_severity * 3.0 + medium_severity * 1.5 + low_severity * 0.5
            security_score = max(0.0, 10.0 - penalty)
        
        return {
            "security_score": security_score,
            "security_issues": security_issues,
            "issues_by_severity": {
                "high": [i for i in security_issues if i["severity"] == "high"],
                "medium": [i for i in security_issues if i["severity"] == "medium"],
                "low": [i for i in security_issues if i["severity"] == "low"]
            },
            "recommendations": self._generate_security_recommendations(security_issues)
        }
    
    def _analyze_ast_security(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Analyze AST for security issues.
        
        Args:
            tree: Parsed AST
            
        Returns:
            List of security issues
        """
        issues = []
        
        for node in ast.walk(tree):
            # Check for hardcoded passwords/secrets
            if isinstance(node, ast.Str):
                if self._looks_like_secret(node.s):
                    issues.append({
                        "type": "hardcoded_secret",
                        "description": "Potential hardcoded secret or password",
                        "line": getattr(node, 'lineno', 0),
                        "severity": "high",
                        "code_snippet": node.s[:20] + "..." if len(node.s) > 20 else node.s
                    })
            
            # Check for dangerous imports
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in ['os', 'subprocess', 'sys']:
                        issues.append({
                            "type": "dangerous_import",
                            "description": f"Import of potentially dangerous module: {alias.name}",
                            "line": getattr(node, 'lineno', 0),
                            "severity": "medium",
                            "code_snippet": f"import {alias.name}"
                        })
        
        return issues
    
    def _looks_like_secret(self, text: str) -> bool:
        """Check if text looks like a secret or password.
        
        Args:
            text: Text to analyze
            
        Returns:
            True if text looks like a secret
        """
        if len(text) < 8:  # Too short to be a meaningful secret
            return False
        
        # Patterns that suggest secrets
        secret_indicators = [
            'password', 'passwd', 'pwd', 'secret', 'key', 'token',
            'api_key', 'auth', 'credential', 'private'
        ]
        
        text_lower = text.lower()
        
        # Check for secret keywords
        has_secret_keyword = any(indicator in text_lower for indicator in secret_indicators)
        
        # Check for high entropy (random-looking strings)
        entropy = self._calculate_entropy(text)
        high_entropy = entropy > 4.0  # Threshold for randomness
        
        # Check for common patterns
        has_special_chars = bool(re.search(r'[!@#$%^&*()_+\-=\[\]{};:"\\|,.<>?]', text))
        has_mixed_case = text != text.lower() and text != text.upper()
        
        return has_secret_keyword or (high_entropy and has_special_chars and has_mixed_case)
    
    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Entropy value
        """
        if not text:
            return 0.0
        
        # Count character frequencies
        char_counts = defaultdict(int)
        for char in text:
            char_counts[char] += 1
        
        # Calculate entropy
        text_len = len(text)
        entropy = 0.0
        
        for count in char_counts.values():
            probability = count / text_len
            if probability > 0:
                entropy -= probability * np.log2(probability)
        
        return entropy
    
    def _generate_security_recommendations(self, issues: List[Dict[str, Any]]) -> List[str]:
        """Generate security recommendations based on found issues.
        
        Args:
            issues: List of security issues
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Count issue types
        issue_types = defaultdict(int)
        for issue in issues:
            issue_types[issue["type"]] += 1
        
        # Generate specific recommendations
        if issue_types["security_pattern"] > 0:
            recommendations.append("Replace dangerous functions with safer alternatives")
            recommendations.append("Use parameterized queries to prevent injection attacks")
        
        if issue_types["hardcoded_secret"] > 0:
            recommendations.append("Move secrets to environment variables or secure vaults")
            recommendations.append("Use configuration management for sensitive data")
        
        if issue_types["dangerous_import"] > 0:
            recommendations.append("Minimize use of system-level modules")
            recommendations.append("Validate and sanitize all external inputs")
        
        # General recommendations
        recommendations.extend([
            "Enable static analysis tools in CI/CD pipeline",
            "Perform regular security audits and penetration testing",
            "Keep dependencies up to date",
            "Follow secure coding guidelines"
        ])
        
        return list(set(recommendations))  # Remove duplicates
    
    def get_confidence(self) -> float:
        """Get confidence in security analysis."""
        return self.confidence_base


class TestGenerator:
    """AI-powered test case generator."""
    
    def __init__(self):
        """Initialize test generator."""
        self.logger = logging.getLogger(__name__)
        
    def generate_tests(
        self, 
        code: str, 
        existing_tests: Optional[str] = None,
        target_coverage: float = 0.8
    ) -> List[GeneratedTest]:
        """Generate test cases for given code.
        
        Args:
            code: Source code to generate tests for
            existing_tests: Existing test code (optional)
            target_coverage: Target code coverage to achieve
            
        Returns:
            List of generated tests
        """
        generated_tests = []
        
        try:
            # Parse code to extract functions
            tree = ast.parse(code)
            functions = self._extract_functions(tree)
            
            # Generate tests for each function
            for func_info in functions:
                # Generate different types of tests
                unit_tests = self._generate_unit_tests(func_info)
                property_tests = self._generate_property_tests(func_info)
                edge_case_tests = self._generate_edge_case_tests(func_info)
                
                generated_tests.extend(unit_tests)
                generated_tests.extend(property_tests)
                generated_tests.extend(edge_case_tests)
            
            # Generate integration tests
            integration_tests = self._generate_integration_tests(functions)
            generated_tests.extend(integration_tests)
            
            # Prioritize tests by expected coverage increase
            generated_tests.sort(key=lambda t: t.expected_coverage_increase, reverse=True)
            
            return generated_tests
            
        except Exception as e:
            self.logger.error(f"Test generation failed: {e}")
            return []
    
    def _extract_functions(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract function information from AST.
        
        Args:
            tree: Parsed AST
            
        Returns:
            List of function information dictionaries
        """
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_info = {
                    "name": node.name,
                    "args": [arg.arg for arg in node.args.args],
                    "docstring": ast.get_docstring(node),
                    "return_annotation": getattr(node.returns, 'id', None) if node.returns else None,
                    "decorators": [d.id if isinstance(d, ast.Name) else str(d) for d in node.decorator_list],
                    "complexity": self._estimate_complexity(node),
                    "line_number": node.lineno,
                    "body_preview": self._get_function_body_preview(node)
                }
                functions.append(func_info)
        
        return functions
    
    def _estimate_complexity(self, node: ast.FunctionDef) -> int:
        """Estimate function complexity for test prioritization.
        
        Args:
            node: Function AST node
            
        Returns:
            Complexity estimate
        """
        complexity = 1
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.Try)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def _get_function_body_preview(self, node: ast.FunctionDef) -> str:
        """Get a preview of function body for context.
        
        Args:
            node: Function AST node
            
        Returns:
            Function body preview
        """
        try:
            # Get first few statements
            preview_lines = []
            for stmt in node.body[:3]:  # First 3 statements
                if isinstance(stmt, ast.Return):
                    preview_lines.append("return ...")
                elif isinstance(stmt, ast.Assign):
                    preview_lines.append("assignment")
                elif isinstance(stmt, ast.If):
                    preview_lines.append("if statement")
                elif isinstance(stmt, ast.For):
                    preview_lines.append("for loop")
                elif isinstance(stmt, ast.While):
                    preview_lines.append("while loop")
                else:
                    preview_lines.append("statement")
            
            return ", ".join(preview_lines)
            
        except Exception:
            return "unknown"
    
    def _generate_unit_tests(self, func_info: Dict[str, Any]) -> List[GeneratedTest]:
        """Generate unit tests for a function.
        
        Args:
            func_info: Function information
            
        Returns:
            List of generated unit tests
        """
        tests = []
        func_name = func_info["name"]
        args = func_info["args"]
        
        # Basic functionality test
        test_code = self._create_basic_test(func_name, args)
        tests.append(GeneratedTest(
            test_type=TestType.UNIT_TEST,
            function_name=func_name,
            test_code=test_code,
            description=f"Basic functionality test for {func_name}",
            confidence=0.7,
            expected_coverage_increase=0.2
        ))
        
        # Return value test
        if func_info["return_annotation"]:
            test_code = self._create_return_type_test(func_name, args, func_info["return_annotation"])
            tests.append(GeneratedTest(
                test_type=TestType.UNIT_TEST,
                function_name=func_name,
                test_code=test_code,
                description=f"Return type test for {func_name}",
                confidence=0.8,
                expected_coverage_increase=0.15
            ))
        
        return tests
    
    def _generate_property_tests(self, func_info: Dict[str, Any]) -> List[GeneratedTest]:
        """Generate property-based tests for a function.
        
        Args:
            func_info: Function information
            
        Returns:
            List of generated property tests
        """
        tests = []
        func_name = func_info["name"]
        
        # Generate property test for deterministic functions
        test_code = self._create_property_test(func_name, func_info["args"])
        
        tests.append(GeneratedTest(
            test_type=TestType.PROPERTY_TEST,
            function_name=func_name,
            test_code=test_code,
            description=f"Property-based test for {func_name}",
            confidence=0.6,
            expected_coverage_increase=0.25
        ))
        
        return tests
    
    def _generate_edge_case_tests(self, func_info: Dict[str, Any]) -> List[GeneratedTest]:
        """Generate edge case tests for a function.
        
        Args:
            func_info: Function information
            
        Returns:
            List of generated edge case tests
        """
        tests = []
        func_name = func_info["name"]
        args = func_info["args"]
        
        # Edge case test
        test_code = self._create_edge_case_test(func_name, args)
        
        tests.append(GeneratedTest(
            test_type=TestType.UNIT_TEST,
            function_name=func_name,
            test_code=test_code,
            description=f"Edge case test for {func_name}",
            confidence=0.8,
            expected_coverage_increase=0.3
        ))
        
        return tests
    
    def _generate_integration_tests(self, functions: List[Dict[str, Any]]) -> List[GeneratedTest]:
        """Generate integration tests for multiple functions.
        
        Args:
            functions: List of function information
            
        Returns:
            List of generated integration tests
        """
        tests = []
        
        if len(functions) >= 2:
            # Create integration test combining multiple functions
            func_names = [f["name"] for f in functions[:2]]  # Use first two functions
            
            test_code = self._create_integration_test(func_names)
            
            tests.append(GeneratedTest(
                test_type=TestType.INTEGRATION_TEST,
                function_name="integration",
                test_code=test_code,
                description=f"Integration test for {', '.join(func_names)}",
                confidence=0.5,
                expected_coverage_increase=0.4
            ))
        
        return tests
    
    def _create_basic_test(self, func_name: str, args: List[str]) -> str:
        """Create basic test code.
        
        Args:
            func_name: Name of function to test
            args: Function arguments
            
        Returns:
            Generated test code
        """
        # Create sample arguments
        sample_args = []
        for arg in args:
            if 'array' in arg.lower() or 'x' in arg.lower():
                sample_args.append("jnp.array([1.0, 2.0, 3.0])")
            elif 'int' in arg.lower() or 'n' in arg.lower():
                sample_args.append("10")
            elif 'float' in arg.lower():
                sample_args.append("1.5")
            elif 'str' in arg.lower():
                sample_args.append("'test_string'")
            else:
                sample_args.append("None")
        
        args_str = ", ".join(sample_args)
        
        return f"""
def test_{func_name}_basic():
    \"\"\"Basic functionality test for {func_name}.\"\"\"
    result = {func_name}({args_str})
    assert result is not None
    # TODO: Add specific assertions based on expected behavior
"""
    
    def _create_return_type_test(self, func_name: str, args: List[str], return_type: str) -> str:
        """Create return type test code.
        
        Args:
            func_name: Name of function to test
            args: Function arguments
            return_type: Expected return type
            
        Returns:
            Generated test code
        """
        sample_args = ["1.0"] * len(args)  # Simple sample args
        args_str = ", ".join(sample_args)
        
        return f"""
def test_{func_name}_return_type():
    \"\"\"Test return type for {func_name}.\"\"\"
    result = {func_name}({args_str})
    assert isinstance(result, {return_type})
"""
    
    def _create_property_test(self, func_name: str, args: List[str]) -> str:
        """Create property-based test code.
        
        Args:
            func_name: Name of function to test
            args: Function arguments
            
        Returns:
            Generated test code
        """
        return f"""
@pytest.mark.property
def test_{func_name}_properties():
    \"\"\"Property-based test for {func_name}.\"\"\"
    # Test deterministic property
    args1 = {[f"test_arg_{i}" for i in range(len(args))]}
    result1 = {func_name}(*args1)
    result2 = {func_name}(*args1)
    assert jnp.allclose(result1, result2), "Function should be deterministic"
    
    # Test with different inputs
    for _ in range(10):
        # Generate random test inputs
        random_args = {[f"np.random.random()" for _ in range(len(args))]}
        result = {func_name}(*random_args)
        # Add property assertions here
        assert not jnp.any(jnp.isnan(result)), "Result should not contain NaN"
"""
    
    def _create_edge_case_test(self, func_name: str, args: List[str]) -> str:
        """Create edge case test code.
        
        Args:
            func_name: Name of function to test
            args: Function arguments
            
        Returns:
            Generated test code
        """
        return f"""
def test_{func_name}_edge_cases():
    \"\"\"Edge case tests for {func_name}.\"\"\"
    # Test with empty/zero inputs
    try:
        empty_args = {["jnp.array([])" if "array" in arg else "0" for arg in args]}
        result = {func_name}(*empty_args)
        # Should handle edge cases gracefully
    except (ValueError, IndexError):
        pass  # Expected for some edge cases
    
    # Test with extreme values
    try:
        extreme_args = {["jnp.array([1e10, -1e10])" if "array" in arg else "1e10" for arg in args]}
        result = {func_name}(*extreme_args)
        assert jnp.all(jnp.isfinite(result)), "Result should be finite for extreme inputs"
    except (OverflowError, ValueError):
        pass  # May be expected for extreme values
"""
    
    def _create_integration_test(self, func_names: List[str]) -> str:
        """Create integration test code.
        
        Args:
            func_names: Names of functions to test together
            
        Returns:
            Generated test code
        """
        return f"""
def test_integration_{('_'.join(func_names))}():
    \"\"\"Integration test for {', '.join(func_names)}.\"\"\"
    # Test function composition/interaction
    test_input = jnp.array([1.0, 2.0, 3.0])
    
    # Chain function calls
    intermediate = {func_names[0]}(test_input)
    final_result = {func_names[1]}(intermediate)
    
    # Verify integration works correctly
    assert final_result is not None
    assert jnp.all(jnp.isfinite(final_result))
    
    # Test bidirectional compatibility if applicable
    # TODO: Add specific integration assertions
"""


class AIQualityGates:
    """AI-enhanced quality gates system.
    
    This system uses AI and machine learning techniques to provide intelligent
    quality validation with adaptive thresholds and automated improvements.
    """
    
    def __init__(
        self,
        adaptive_thresholds: bool = True,
        auto_test_generation: bool = True,
        anomaly_detection: bool = True
    ):
        """Initialize AI quality gates.
        
        Args:
            adaptive_thresholds: Whether to use adaptive quality thresholds
            auto_test_generation: Whether to automatically generate tests
            anomaly_detection: Whether to use anomaly detection for quality metrics
        """
        self.adaptive_thresholds = adaptive_thresholds
        self.auto_test_generation = auto_test_generation
        self.anomaly_detection = anomaly_detection
        
        # AI analyzers
        self.complexity_analyzer = ComplexityAnalyzer()
        self.security_analyzer = SecurityAnalyzer()
        self.test_generator = TestGenerator()
        
        # Quality metrics history
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Adaptive threshold models
        self.threshold_models: Dict[str, Any] = {}
        self.anomaly_detectors: Dict[str, IsolationForest] = {}
        
        # Quality trends
        self.quality_trends: Dict[str, List[float]] = defaultdict(list)
        
        # Generated tests cache
        self.generated_tests_cache: Dict[str, List[GeneratedTest]] = {}
        
        # Performance tracking
        self.gate_performance_history: List[Dict[str, Any]] = []
        
        # Logger
        self.logger = logging.getLogger(__name__)
    
    def evaluate_quality_gates(
        self,
        code: str,
        test_results: Optional[Dict[str, Any]] = None,
        coverage_data: Optional[Dict[str, Any]] = None,
        performance_data: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, QualityMetric]:
        """Evaluate all quality gates using AI-enhanced analysis.
        
        Args:
            code: Source code to analyze
            test_results: Test execution results
            coverage_data: Code coverage data
            performance_data: Performance metrics
            context: Additional context information
            
        Returns:
            Dictionary of quality metrics with AI-enhanced evaluation
        """
        context = context or {}
        metrics = {}
        
        # Code complexity analysis
        complexity_metric = self._evaluate_complexity_gate(code, context)
        metrics["complexity"] = complexity_metric
        
        # Security analysis
        security_metric = self._evaluate_security_gate(code, context)
        metrics["security"] = security_metric
        
        # Test coverage analysis (if data available)
        if coverage_data:
            coverage_metric = self._evaluate_coverage_gate(coverage_data, context)
            metrics["coverage"] = coverage_metric
        
        # Performance analysis (if data available)
        if performance_data:
            performance_metric = self._evaluate_performance_gate(performance_data, context)
            metrics["performance"] = performance_metric
        
        # Documentation analysis
        documentation_metric = self._evaluate_documentation_gate(code, context)
        metrics["documentation"] = documentation_metric
        
        # Maintainability analysis
        maintainability_metric = self._evaluate_maintainability_gate(code, context)
        metrics["maintainability"] = maintainability_metric
        
        # Update metrics history
        self._update_metrics_history(metrics)
        
        # Update adaptive thresholds
        if self.adaptive_thresholds:
            self._update_adaptive_thresholds(metrics)
        
        # Generate tests if enabled
        if self.auto_test_generation:
            self._generate_missing_tests(code, metrics, context)
        
        # Record gate evaluation
        self._record_gate_performance(metrics, context)
        
        return metrics
    
    def _evaluate_complexity_gate(self, code: str, context: Dict[str, Any]) -> QualityMetric:
        """Evaluate complexity quality gate.
        
        Args:
            code: Source code to analyze
            context: Analysis context
            
        Returns:
            Complexity quality metric
        """
        analysis = self.complexity_analyzer.analyze(code, context)
        
        avg_complexity = analysis.get("average_complexity", 0)
        max_complexity = analysis.get("max_complexity", 0)
        
        # Adaptive threshold or default
        threshold = self._get_adaptive_threshold("complexity", default=10.0)
        
        # Calculate score (lower complexity is better)
        if max_complexity <= threshold:
            score = 10.0 - (avg_complexity / threshold) * 3.0  # Scale to 0-10
            status = QualityGateStatus.PASSED
        elif max_complexity <= threshold * 1.5:
            score = 7.0 - (avg_complexity / threshold) * 2.0
            status = QualityGateStatus.WARNING
        else:
            score = max(0.0, 5.0 - (avg_complexity / threshold))
            status = QualityGateStatus.FAILED
        
        return QualityMetric(
            name="complexity",
            value=avg_complexity,
            threshold=threshold,
            status=status,
            confidence=self.complexity_analyzer.get_confidence(),
            timestamp=time.time(),
            metadata={
                "max_complexity": max_complexity,
                "complex_functions": analysis.get("complex_functions", []),
                "needs_refactoring": analysis.get("needs_refactoring", False),
                "score": score
            }
        )
    
    def _evaluate_security_gate(self, code: str, context: Dict[str, Any]) -> QualityMetric:
        """Evaluate security quality gate.
        
        Args:
            code: Source code to analyze
            context: Analysis context
            
        Returns:
            Security quality metric
        """
        analysis = self.security_analyzer.analyze(code, context)
        
        security_score = analysis.get("security_score", 0.0)
        security_issues = analysis.get("security_issues", [])
        
        # Adaptive threshold or default
        threshold = self._get_adaptive_threshold("security", default=7.0)
        
        # Determine status
        if security_score >= threshold:
            status = QualityGateStatus.PASSED
        elif security_score >= threshold * 0.7:
            status = QualityGateStatus.WARNING
        else:
            status = QualityGateStatus.FAILED
        
        return QualityMetric(
            name="security",
            value=security_score,
            threshold=threshold,
            status=status,
            confidence=self.security_analyzer.get_confidence(),
            timestamp=time.time(),
            metadata={
                "security_issues": len(security_issues),
                "high_severity_issues": len(analysis.get("issues_by_severity", {}).get("high", [])),
                "recommendations": analysis.get("recommendations", []),
                "issues_detail": security_issues
            }
        )
    
    def _evaluate_coverage_gate(self, coverage_data: Dict[str, Any], context: Dict[str, Any]) -> QualityMetric:
        """Evaluate test coverage quality gate.
        
        Args:
            coverage_data: Coverage data
            context: Analysis context
            
        Returns:
            Coverage quality metric
        """
        line_coverage = coverage_data.get("line_coverage", 0.0)
        branch_coverage = coverage_data.get("branch_coverage", 0.0)
        
        # Combined coverage score
        coverage_score = 0.6 * line_coverage + 0.4 * branch_coverage
        
        # Adaptive threshold or default
        threshold = self._get_adaptive_threshold("coverage", default=0.8)
        
        # Determine status
        if coverage_score >= threshold:
            status = QualityGateStatus.PASSED
        elif coverage_score >= threshold * 0.8:
            status = QualityGateStatus.WARNING
        else:
            status = QualityGateStatus.FAILED
        
        return QualityMetric(
            name="coverage",
            value=coverage_score,
            threshold=threshold,
            status=status,
            confidence=0.9,  # High confidence for coverage metrics
            timestamp=time.time(),
            metadata={
                "line_coverage": line_coverage,
                "branch_coverage": branch_coverage,
                "uncovered_lines": coverage_data.get("uncovered_lines", []),
                "missing_branches": coverage_data.get("missing_branches", [])
            }
        )
    
    def _evaluate_performance_gate(self, performance_data: Dict[str, Any], context: Dict[str, Any]) -> QualityMetric:
        """Evaluate performance quality gate.
        
        Args:
            performance_data: Performance metrics
            context: Analysis context
            
        Returns:
            Performance quality metric
        """
        execution_time = performance_data.get("execution_time", 0.0)
        memory_usage = performance_data.get("memory_usage", 0.0)
        cpu_usage = performance_data.get("cpu_usage", 0.0)
        
        # Normalize metrics (lower is better for time and resource usage)
        time_score = max(0, 10 - execution_time)  # Assume 10s is max acceptable
        memory_score = max(0, 10 - memory_usage / 1000)  # Normalize memory in MB
        cpu_score = max(0, 10 - cpu_usage * 10)  # CPU as percentage
        
        # Combined performance score
        performance_score = (time_score + memory_score + cpu_score) / 3.0
        
        # Adaptive threshold or default
        threshold = self._get_adaptive_threshold("performance", default=7.0)
        
        # Determine status
        if performance_score >= threshold:
            status = QualityGateStatus.PASSED
        elif performance_score >= threshold * 0.8:
            status = QualityGateStatus.WARNING
        else:
            status = QualityGateStatus.FAILED
        
        return QualityMetric(
            name="performance",
            value=performance_score,
            threshold=threshold,
            status=status,
            confidence=0.8,
            timestamp=time.time(),
            metadata={
                "execution_time": execution_time,
                "memory_usage": memory_usage,
                "cpu_usage": cpu_usage,
                "time_score": time_score,
                "memory_score": memory_score,
                "cpu_score": cpu_score
            }
        )
    
    def _evaluate_documentation_gate(self, code: str, context: Dict[str, Any]) -> QualityMetric:
        """Evaluate documentation quality gate.
        
        Args:
            code: Source code to analyze
            context: Analysis context
            
        Returns:
            Documentation quality metric
        """
        try:
            tree = ast.parse(code)
            
            # Count functions and their documentation
            total_functions = 0
            documented_functions = 0
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    total_functions += 1
                    docstring = ast.get_docstring(node)
                    if docstring and len(docstring.strip()) > 10:
                        documented_functions += 1
            
            # Calculate documentation coverage
            if total_functions > 0:
                doc_coverage = documented_functions / total_functions
            else:
                doc_coverage = 1.0  # No functions to document
            
            # Score based on coverage
            doc_score = doc_coverage * 10.0
            
            # Adaptive threshold or default
            threshold = self._get_adaptive_threshold("documentation", default=0.7)
            
            # Determine status
            if doc_coverage >= threshold:
                status = QualityGateStatus.PASSED
            elif doc_coverage >= threshold * 0.8:
                status = QualityGateStatus.WARNING
            else:
                status = QualityGateStatus.FAILED
            
            return QualityMetric(
                name="documentation",
                value=doc_coverage,
                threshold=threshold,
                status=status,
                confidence=0.9,
                timestamp=time.time(),
                metadata={
                    "total_functions": total_functions,
                    "documented_functions": documented_functions,
                    "documentation_score": doc_score,
                    "missing_documentation": total_functions - documented_functions
                }
            )
            
        except SyntaxError:
            return QualityMetric(
                name="documentation",
                value=0.0,
                threshold=0.7,
                status=QualityGateStatus.FAILED,
                confidence=0.5,
                timestamp=time.time(),
                metadata={"error": "Syntax error in code"}
            )
    
    def _evaluate_maintainability_gate(self, code: str, context: Dict[str, Any]) -> QualityMetric:
        """Evaluate maintainability quality gate.
        
        Args:
            code: Source code to analyze
            context: Analysis context
            
        Returns:
            Maintainability quality metric
        """
        try:
            # Basic maintainability metrics
            lines = code.split('\n')
            total_lines = len([line for line in lines if line.strip()])
            comment_lines = len([line for line in lines if line.strip().startswith('#')])
            
            # Comment ratio
            comment_ratio = comment_lines / max(1, total_lines)
            
            # Average line length
            line_lengths = [len(line) for line in lines if line.strip()]
            avg_line_length = np.mean(line_lengths) if line_lengths else 0
            
            # Function count and average size
            tree = ast.parse(code)
            function_sizes = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Count lines in function
                    func_lines = []
                    if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                        func_lines = lines[node.lineno-1:node.end_lineno]
                    function_sizes.append(len(func_lines))
            
            avg_function_size = np.mean(function_sizes) if function_sizes else 0
            
            # Calculate maintainability score
            comment_score = min(10.0, comment_ratio * 50)  # Good comment ratio is ~20%
            line_length_score = max(0, 10 - (avg_line_length - 80) / 10) if avg_line_length > 80 else 10
            function_size_score = max(0, 10 - (avg_function_size - 20) / 5) if avg_function_size > 20 else 10
            
            maintainability_score = (comment_score + line_length_score + function_size_score) / 3.0
            
            # Adaptive threshold or default
            threshold = self._get_adaptive_threshold("maintainability", default=7.0)
            
            # Determine status
            if maintainability_score >= threshold:
                status = QualityGateStatus.PASSED
            elif maintainability_score >= threshold * 0.8:
                status = QualityGateStatus.WARNING
            else:
                status = QualityGateStatus.FAILED
            
            return QualityMetric(
                name="maintainability",
                value=maintainability_score,
                threshold=threshold,
                status=status,
                confidence=0.7,
                timestamp=time.time(),
                metadata={
                    "comment_ratio": comment_ratio,
                    "avg_line_length": avg_line_length,
                    "avg_function_size": avg_function_size,
                    "total_functions": len(function_sizes),
                    "comment_score": comment_score,
                    "line_length_score": line_length_score,
                    "function_size_score": function_size_score
                }
            )
            
        except Exception as e:
            return QualityMetric(
                name="maintainability",
                value=0.0,
                threshold=7.0,
                status=QualityGateStatus.FAILED,
                confidence=0.5,
                timestamp=time.time(),
                metadata={"error": str(e)}
            )
    
    def _get_adaptive_threshold(self, metric_name: str, default: float) -> float:
        """Get adaptive threshold for a metric.
        
        Args:
            metric_name: Name of the metric
            default: Default threshold value
            
        Returns:
            Adaptive threshold value
        """
        if not self.adaptive_thresholds or metric_name not in self.metrics_history:
            return default
        
        # Get historical values
        historical_values = list(self.metrics_history[metric_name])
        
        if len(historical_values) < 10:  # Need minimum history
            return default
        
        # Calculate adaptive threshold based on historical performance
        values = [h.value for h in historical_values if h.status == QualityGateStatus.PASSED]
        
        if not values:
            return default
        
        # Use percentile-based adaptive threshold
        if metric_name in ["complexity"]:
            # Lower is better - use 75th percentile
            adaptive_threshold = np.percentile(values, 75)
        else:
            # Higher is better - use 25th percentile
            adaptive_threshold = np.percentile(values, 25)
        
        # Smooth adaptation (move slowly towards new threshold)
        if metric_name in self.threshold_models:
            current_threshold = self.threshold_models[metric_name]
            adaptive_threshold = 0.9 * current_threshold + 0.1 * adaptive_threshold
        
        self.threshold_models[metric_name] = adaptive_threshold
        
        return adaptive_threshold
    
    def _update_metrics_history(self, metrics: Dict[str, QualityMetric]) -> None:
        """Update metrics history for adaptive learning.
        
        Args:
            metrics: Current quality metrics
        """
        for metric_name, metric in metrics.items():
            self.metrics_history[metric_name].append(metric)
            
            # Update quality trends
            self.quality_trends[metric_name].append(metric.value)
            if len(self.quality_trends[metric_name]) > 50:  # Keep last 50 values
                self.quality_trends[metric_name] = self.quality_trends[metric_name][-50:]
    
    def _update_adaptive_thresholds(self, metrics: Dict[str, QualityMetric]) -> None:
        """Update adaptive thresholds based on recent performance.
        
        Args:
            metrics: Current quality metrics
        """
        for metric_name, metric in metrics.items():
            # Update anomaly detection model
            if self.anomaly_detection:
                self._update_anomaly_detector(metric_name, metric.value)
    
    def _update_anomaly_detector(self, metric_name: str, value: float) -> None:
        """Update anomaly detection model for a metric.
        
        Args:
            metric_name: Name of the metric
            value: Current metric value
        """
        if metric_name not in self.anomaly_detectors:
            self.anomaly_detectors[metric_name] = IsolationForest(contamination=0.1, random_state=42)
        
        # Collect recent values for training
        if metric_name in self.quality_trends and len(self.quality_trends[metric_name]) >= 20:
            values = np.array(self.quality_trends[metric_name]).reshape(-1, 1)
            
            try:
                self.anomaly_detectors[metric_name].fit(values)
            except Exception as e:
                self.logger.warning(f"Anomaly detector update failed for {metric_name}: {e}")
    
    def _generate_missing_tests(
        self, 
        code: str, 
        metrics: Dict[str, QualityMetric], 
        context: Dict[str, Any]
    ) -> None:
        """Generate missing tests based on coverage analysis.
        
        Args:
            code: Source code
            metrics: Quality metrics
            context: Analysis context
        """
        # Check if test generation is needed
        coverage_metric = metrics.get("coverage")
        
        if (coverage_metric and 
            coverage_metric.status in [QualityGateStatus.FAILED, QualityGateStatus.WARNING]):
            
            # Generate tests for low coverage areas
            code_hash = hash(code)
            
            if code_hash not in self.generated_tests_cache:
                generated_tests = self.test_generator.generate_tests(code)
                self.generated_tests_cache[code_hash] = generated_tests
                
                self.logger.info(f"Generated {len(generated_tests)} tests for improved coverage")
    
    def _record_gate_performance(self, metrics: Dict[str, QualityMetric], context: Dict[str, Any]) -> None:
        """Record quality gate performance for analysis.
        
        Args:
            metrics: Quality metrics
            context: Analysis context
        """
        performance_record = {
            "timestamp": time.time(),
            "metrics_count": len(metrics),
            "passed_gates": sum(1 for m in metrics.values() if m.status == QualityGateStatus.PASSED),
            "failed_gates": sum(1 for m in metrics.values() if m.status == QualityGateStatus.FAILED),
            "warning_gates": sum(1 for m in metrics.values() if m.status == QualityGateStatus.WARNING),
            "average_confidence": np.mean([m.confidence for m in metrics.values()]),
            "context": context
        }
        
        self.gate_performance_history.append(performance_record)
        
        # Keep recent history
        if len(self.gate_performance_history) > 100:
            self.gate_performance_history = self.gate_performance_history[-100:]
    
    def get_quality_insights(self) -> Dict[str, Any]:
        """Get AI-powered insights about code quality trends.
        
        Returns:
            Dictionary containing quality insights and recommendations
        """
        insights = {
            "overall_trends": {},
            "anomalies_detected": {},
            "recommendations": [],
            "generated_tests_summary": {},
            "adaptive_thresholds": self.threshold_models.copy(),
        }
        
        # Analyze trends for each metric
        for metric_name, trend_data in self.quality_trends.items():
            if len(trend_data) >= 10:
                # Calculate trend slope
                x = np.arange(len(trend_data))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, trend_data)
                
                insights["overall_trends"][metric_name] = {
                    "slope": slope,
                    "direction": "improving" if slope > 0.01 else "declining" if slope < -0.01 else "stable",
                    "r_squared": r_value ** 2,
                    "significance": "significant" if p_value < 0.05 else "not_significant",
                    "recent_average": np.mean(trend_data[-5:]),
                    "overall_average": np.mean(trend_data)
                }
        
        # Detect anomalies
        for metric_name, detector in self.anomaly_detectors.items():
            if metric_name in self.quality_trends:
                recent_values = np.array(self.quality_trends[metric_name][-10:]).reshape(-1, 1)
                
                try:
                    anomaly_scores = detector.decision_function(recent_values)
                    anomalies = detector.predict(recent_values)
                    
                    insights["anomalies_detected"][metric_name] = {
                        "anomaly_count": sum(1 for a in anomalies if a == -1),
                        "anomaly_scores": anomaly_scores.tolist(),
                        "has_recent_anomaly": anomalies[-1] == -1 if len(anomalies) > 0 else False
                    }
                    
                except Exception as e:
                    self.logger.warning(f"Anomaly detection failed for {metric_name}: {e}")
        
        # Generate recommendations
        insights["recommendations"] = self._generate_quality_recommendations(insights)
        
        # Summarize generated tests
        total_tests = sum(len(tests) for tests in self.generated_tests_cache.values())
        test_types = defaultdict(int)
        
        for tests in self.generated_tests_cache.values():
            for test in tests:
                test_types[test.test_type.value] += 1
        
        insights["generated_tests_summary"] = {
            "total_generated": total_tests,
            "by_type": dict(test_types),
            "cache_entries": len(self.generated_tests_cache)
        }
        
        return insights
    
    def _generate_quality_recommendations(self, insights: Dict[str, Any]) -> List[str]:
        """Generate quality improvement recommendations based on insights.
        
        Args:
            insights: Quality insights data
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Trend-based recommendations
        for metric_name, trend in insights.get("overall_trends", {}).items():
            if trend["direction"] == "declining":
                if metric_name == "complexity":
                    recommendations.append(f"Code complexity is increasing. Consider refactoring complex functions.")
                elif metric_name == "security":
                    recommendations.append(f"Security score is declining. Review recent security changes.")
                elif metric_name == "coverage":
                    recommendations.append(f"Test coverage is decreasing. Add tests for new functionality.")
                elif metric_name == "documentation":
                    recommendations.append(f"Documentation coverage is declining. Add documentation for new functions.")
        
        # Anomaly-based recommendations
        for metric_name, anomaly_info in insights.get("anomalies_detected", {}).items():
            if anomaly_info.get("has_recent_anomaly", False):
                recommendations.append(f"Anomaly detected in {metric_name} metrics. Investigate recent changes.")
        
        # Test generation recommendations
        test_summary = insights.get("generated_tests_summary", {})
        if test_summary.get("total_generated", 0) > 0:
            recommendations.append(f"Consider implementing {test_summary['total_generated']} AI-generated tests to improve coverage.")
        
        # Adaptive threshold recommendations
        adaptive_thresholds = insights.get("adaptive_thresholds", {})
        if adaptive_thresholds:
            recommendations.append("Quality thresholds have been automatically adapted based on historical performance.")
        
        # General recommendations
        recommendations.extend([
            "Enable continuous quality monitoring for early issue detection.",
            "Consider implementing automated refactoring for code smells.",
            "Use AI-generated tests to improve coverage in critical areas.",
            "Review security patterns regularly with updated threat intelligence."
        ])
        
        return recommendations
    
    def get_ai_quality_statistics(self) -> Dict[str, Any]:
        """Get comprehensive AI quality gate statistics.
        
        Returns:
            Dictionary containing AI quality statistics
        """
        # Calculate gate performance statistics
        if self.gate_performance_history:
            recent_performance = self.gate_performance_history[-10:]  # Last 10 evaluations
            
            avg_passed = np.mean([p["passed_gates"] for p in recent_performance])
            avg_failed = np.mean([p["failed_gates"] for p in recent_performance])
            avg_confidence = np.mean([p["average_confidence"] for p in recent_performance])
        else:
            avg_passed = avg_failed = avg_confidence = 0.0
        
        return {
            "ai_features_enabled": {
                "adaptive_thresholds": self.adaptive_thresholds,
                "auto_test_generation": self.auto_test_generation,
                "anomaly_detection": self.anomaly_detection,
            },
            "gate_performance": {
                "total_evaluations": len(self.gate_performance_history),
                "average_passed_gates": avg_passed,
                "average_failed_gates": avg_failed,
                "average_confidence": avg_confidence,
            },
            "adaptive_learning": {
                "metrics_tracked": len(self.metrics_history),
                "adaptive_thresholds_count": len(self.threshold_models),
                "anomaly_detectors_count": len(self.anomaly_detectors),
            },
            "test_generation": {
                "total_tests_generated": sum(len(tests) for tests in self.generated_tests_cache.values()),
                "code_files_analyzed": len(self.generated_tests_cache),
                "test_types_available": len(TestType),
            },
            "quality_trends": {
                metric: len(trend) for metric, trend in self.quality_trends.items()
            },
            "analyzers_confidence": {
                "complexity_analyzer": self.complexity_analyzer.get_confidence(),
                "security_analyzer": self.security_analyzer.get_confidence(),
            }
        }