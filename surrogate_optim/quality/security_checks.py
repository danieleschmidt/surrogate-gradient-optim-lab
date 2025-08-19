"""Security scanning and validation utilities."""

import ast
import inspect
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..monitoring.logging import get_logger


class SecurityScanner:
    """Scanner for security vulnerabilities in surrogate optimization workflows."""
    
    def __init__(self):
        """Initialize security scanner."""
        self.logger = get_logger()
        self.security_issues = []
    
    def scan_function(self, function: callable) -> Dict[str, Any]:
        """Scan a function for security issues.
        
        Args:
            function: Function to scan
            
        Returns:
            Security scan results
        """
        issues = []
        
        try:
            # Get function source code
            source = inspect.getsource(function)
            
            # Check for dangerous patterns
            dangerous_patterns = [
                (r'exec\s*\(', "Use of exec() detected - code injection risk"),
                (r'eval\s*\(', "Use of eval() detected - code injection risk"),
                (r'__import__\s*\(', "Dynamic imports detected - potential security risk"),
                (r'open\s*\([^)]*["\']w', "File write operations detected - potential data modification"),
                (r'subprocess\.', "Subprocess calls detected - command injection risk"),
                (r'os\.system', "OS system calls detected - command injection risk"),
                (r'shell=True', "Shell execution enabled - command injection risk"),
            ]
            
            for pattern, message in dangerous_patterns:
                if re.search(pattern, source, re.IGNORECASE):
                    issues.append({
                        "type": "dangerous_function",
                        "severity": "high",
                        "message": message,
                        "pattern": pattern,
                    })
            
            # Check for hardcoded secrets (basic patterns)
            secret_patterns = [
                (r'password\s*=\s*["\'][^"\']+["\']', "Hardcoded password detected"),
                (r'api_key\s*=\s*["\'][^"\']+["\']', "Hardcoded API key detected"),
                (r'secret\s*=\s*["\'][^"\']+["\']', "Hardcoded secret detected"),
                (r'token\s*=\s*["\'][^"\']+["\']', "Hardcoded token detected"),
            ]
            
            for pattern, message in secret_patterns:
                if re.search(pattern, source, re.IGNORECASE):
                    issues.append({
                        "type": "hardcoded_secret",
                        "severity": "medium", 
                        "message": message,
                        "pattern": pattern,
                    })
                    
        except Exception as e:
            issues.append({
                "type": "scan_error",
                "severity": "low",
                "message": f"Could not analyze function: {e}",
            })
        
        return {
            "function_name": getattr(function, '__name__', 'unknown'),
            "issues": issues,
            "severity_counts": self._count_by_severity(issues),
        }
    
    def scan_data_sources(self, data_sources: List[Any]) -> Dict[str, Any]:
        """Scan data sources for security issues.
        
        Args:
            data_sources: List of data sources to scan
            
        Returns:
            Security scan results
        """
        issues = []
        
        for i, source in enumerate(data_sources):
            source_issues = []
            
            # Check for pickle usage (deserialization vulnerability)
            if hasattr(source, '__reduce__') or hasattr(source, '__reduce_ex__'):
                source_issues.append({
                    "type": "pickle_usage",
                    "severity": "high",
                    "message": "Pickle serialization detected - deserialization vulnerability risk",
                })
            
            # Check for network data sources
            if isinstance(source, str):
                if source.startswith(('http://', 'https://', 'ftp://')):
                    source_issues.append({
                        "type": "network_source",
                        "severity": "medium",
                        "message": "Network data source detected - verify SSL/TLS usage",
                        "source": source,
                    })
                
                # Check for world-writable paths
                if source.startswith('/tmp/') or source.startswith('/var/tmp/'):
                    source_issues.append({
                        "type": "insecure_path",
                        "severity": "medium", 
                        "message": "Temporary directory usage - potential race condition",
                        "source": source,
                    })
            
            if source_issues:
                issues.extend([{**issue, "source_index": i} for issue in source_issues])
        
        return {
            "sources_scanned": len(data_sources),
            "issues": issues,
            "severity_counts": self._count_by_severity(issues),
        }
    
    def scan_model_serialization(self, model_data: Any) -> Dict[str, Any]:
        """Scan model serialization for security issues.
        
        Args:
            model_data: Model data or serialized content
            
        Returns:
            Security scan results
        """
        issues = []
        
        # Check for pickle usage in model serialization
        if isinstance(model_data, bytes):
            # Check if it looks like pickle data
            if model_data.startswith(b'\x80\x03') or model_data.startswith(b'\x80\x04'):
                issues.append({
                    "type": "pickle_serialization",
                    "severity": "high",
                    "message": "Pickle serialization detected in model data",
                })
        
        # Check for custom __reduce__ methods
        if hasattr(model_data, '__reduce__'):
            issues.append({
                "type": "custom_reduce",
                "severity": "medium",
                "message": "Custom __reduce__ method found - review serialization safety",
            })
        
        return {
            "issues": issues,
            "severity_counts": self._count_by_severity(issues),
        }
    
    def scan_input_validation(self, validation_config: Dict[str, Any]) -> Dict[str, Any]:
        """Scan input validation configuration.
        
        Args:
            validation_config: Input validation configuration
            
        Returns:
            Security scan results
        """
        issues = []
        
        # Check for missing input validation
        required_validations = [
            "bounds_validation",
            "type_validation", 
            "range_validation",
            "sanitization",
        ]
        
        for validation in required_validations:
            if not validation_config.get(validation, False):
                issues.append({
                    "type": "missing_validation",
                    "severity": "medium",
                    "message": f"Missing {validation} - input validation vulnerability",
                    "missing_validation": validation,
                })
        
        # Check for overly permissive settings
        if validation_config.get("allow_arbitrary_functions", False):
            issues.append({
                "type": "permissive_setting",
                "severity": "high",
                "message": "Arbitrary function execution allowed - code injection risk",
            })
        
        if validation_config.get("disable_bounds_checking", False):
            issues.append({
                "type": "permissive_setting", 
                "severity": "medium",
                "message": "Bounds checking disabled - potential DoS via resource exhaustion",
            })
        
        return {
            "issues": issues,
            "severity_counts": self._count_by_severity(issues),
        }
    
    def _count_by_severity(self, issues: List[Dict]) -> Dict[str, int]:
        """Count issues by severity level."""
        counts = {"high": 0, "medium": 0, "low": 0}
        
        for issue in issues:
            severity = issue.get("severity", "low")
            if severity in counts:
                counts[severity] += 1
        
        return counts
    
    def comprehensive_scan(
        self,
        functions: Optional[List[callable]] = None,
        data_sources: Optional[List[Any]] = None,
        model_data: Optional[Any] = None,
        validation_config: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Run comprehensive security scan.
        
        Args:
            functions: Functions to scan
            data_sources: Data sources to scan
            model_data: Model data to scan
            validation_config: Input validation config to scan
            
        Returns:
            Comprehensive security scan results
        """
        scan_results = {
            "scan_timestamp": str(pd.Timestamp.now()),
            "components_scanned": {},
            "total_issues": 0,
            "severity_summary": {"high": 0, "medium": 0, "low": 0},
            "overall_risk_level": "low",
        }
        
        # Scan functions
        if functions:
            function_results = []
            for func in functions:
                result = self.scan_function(func)
                function_results.append(result)
                
                # Update totals
                for severity, count in result["severity_counts"].items():
                    scan_results["severity_summary"][severity] += count
            
            scan_results["components_scanned"]["functions"] = function_results
        
        # Scan data sources
        if data_sources:
            data_result = self.scan_data_sources(data_sources)
            scan_results["components_scanned"]["data_sources"] = data_result
            
            for severity, count in data_result["severity_counts"].items():
                scan_results["severity_summary"][severity] += count
        
        # Scan model data
        if model_data:
            model_result = self.scan_model_serialization(model_data)
            scan_results["components_scanned"]["model_data"] = model_result
            
            for severity, count in model_result["severity_counts"].items():
                scan_results["severity_summary"][severity] += count
        
        # Scan validation config
        if validation_config:
            validation_result = self.scan_input_validation(validation_config)
            scan_results["components_scanned"]["validation"] = validation_result
            
            for severity, count in validation_result["severity_counts"].items():
                scan_results["severity_summary"][severity] += count
        
        # Calculate totals and risk level
        total_issues = sum(scan_results["severity_summary"].values())
        scan_results["total_issues"] = total_issues
        
        if scan_results["severity_summary"]["high"] > 0:
            scan_results["overall_risk_level"] = "high"
        elif scan_results["severity_summary"]["medium"] > 0:
            scan_results["overall_risk_level"] = "medium"
        else:
            scan_results["overall_risk_level"] = "low"
        
        return scan_results
    
    def generate_security_report(self, scan_results: Dict[str, Any]) -> str:
        """Generate security scan report.
        
        Args:
            scan_results: Results from comprehensive_scan
            
        Returns:
            Formatted security report
        """
        report = []
        report.append("SECURITY SCAN REPORT")
        report.append("=" * 50)
        
        # Summary
        total_issues = scan_results["total_issues"]
        risk_level = scan_results["overall_risk_level"].upper()
        severity_summary = scan_results["severity_summary"]
        
        report.append(f"\nOVERALL RISK LEVEL: {risk_level}")
        report.append(f"Total Issues Found: {total_issues}")
        report.append(f"High Severity: {severity_summary['high']}")
        report.append(f"Medium Severity: {severity_summary['medium']}")
        report.append(f"Low Severity: {severity_summary['low']}")
        
        # Detailed results
        components = scan_results["components_scanned"]
        
        if "functions" in components:
            report.append("\nFUNCTION SECURITY SCAN:")
            report.append("-" * 30)
            
            for func_result in components["functions"]:
                func_name = func_result["function_name"]
                issues = func_result["issues"]
                
                if issues:
                    report.append(f"\n{func_name}: {len(issues)} issues")
                    for issue in issues:
                        report.append(f"  [{issue['severity'].upper()}] {issue['message']}")
                else:
                    report.append(f"\n{func_name}: No issues found")
        
        if "data_sources" in components:
            data_result = components["data_sources"]
            report.append(f"\nDATA SOURCE SECURITY SCAN:")
            report.append("-" * 30)
            report.append(f"Sources scanned: {data_result['sources_scanned']}")
            
            if data_result["issues"]:
                for issue in data_result["issues"]:
                    source_idx = issue.get("source_index", "unknown")
                    report.append(f"  Source {source_idx} [{issue['severity'].upper()}]: {issue['message']}")
            else:
                report.append("  No issues found")
        
        if "model_data" in components:
            model_result = components["model_data"]
            report.append(f"\nMODEL DATA SECURITY SCAN:")
            report.append("-" * 30)
            
            if model_result["issues"]:
                for issue in model_result["issues"]:
                    report.append(f"  [{issue['severity'].upper()}] {issue['message']}")
            else:
                report.append("  No issues found")
        
        if "validation" in components:
            validation_result = components["validation"]
            report.append(f"\nVALIDATION SECURITY SCAN:")
            report.append("-" * 30)
            
            if validation_result["issues"]:
                for issue in validation_result["issues"]:
                    report.append(f"  [{issue['severity'].upper()}] {issue['message']}")
            else:
                report.append("  No issues found")
        
        # Recommendations
        report.append("\nSECURITY RECOMMENDATIONS:")
        report.append("-" * 30)
        
        if scan_results["severity_summary"]["high"] > 0:
            report.append("- CRITICAL: Address high-severity issues immediately")
            report.append("- Review and sanitize all user inputs")
            report.append("- Avoid dynamic code execution (eval, exec)")
            report.append("- Use safe serialization methods instead of pickle")
        
        if scan_results["severity_summary"]["medium"] > 0:
            report.append("- Implement proper input validation")
            report.append("- Use HTTPS for network communications")
            report.append("- Avoid world-writable directories")
            report.append("- Review custom serialization methods")
        
        report.append("- Regularly update dependencies")
        report.append("- Implement logging and monitoring")
        report.append("- Use principle of least privilege")
        report.append("- Conduct regular security audits")
        
        return "\n".join(report)


# Fix the pandas import issue
try:
    import pandas as pd
except ImportError:
    # Create a simple timestamp substitute
    from datetime import datetime
    class pd:
        class Timestamp:
            @staticmethod
            def now():
                return datetime.now().isoformat()


def scan_surrogate_security(
    surrogate_optimizer,
    data_sources: Optional[List] = None,
    validation_config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Convenience function to scan surrogate optimizer security.
    
    Args:
        surrogate_optimizer: Surrogate optimizer to scan
        data_sources: Optional data sources
        validation_config: Optional validation configuration
        
    Returns:
        Security scan results
    """
    scanner = SecurityScanner()
    
    # Extract functions to scan
    functions = []
    if hasattr(surrogate_optimizer, 'surrogate'):
        if hasattr(surrogate_optimizer.surrogate, 'predict'):
            functions.append(surrogate_optimizer.surrogate.predict)
        if hasattr(surrogate_optimizer.surrogate, 'gradient'):
            functions.append(surrogate_optimizer.surrogate.gradient)
    
    return scanner.comprehensive_scan(
        functions=functions,
        data_sources=data_sources,
        validation_config=validation_config,
    )


class SecurityValidator:
    """Enhanced security validator for surrogate optimization workflows."""
    
    def __init__(self):
        """Initialize security validator."""
        self.function_call_count = 0
        self.max_function_calls = 1000
        
    def validate_bounds(self, bounds):
        """Validate optimization bounds for security."""
        for i, (lower, upper) in enumerate(bounds):
            if upper - lower > 1000:
                raise ValueError(f"Bounds too wide for dimension {i}: potential resource exhaustion")
            if abs(lower) > 1e6 or abs(upper) > 1e6:
                raise ValueError(f"Bounds too extreme for dimension {i}: potential numerical issues")
    
    def monitor_function_calls(self):
        """Context manager to monitor function calls."""
        from contextlib import contextmanager
        
        @contextmanager
        def monitor():
            initial_count = self.function_call_count
            try:
                yield
            finally:
                calls_made = self.function_call_count - initial_count
                if calls_made > self.max_function_calls:
                    raise ValueError(f"Too many function calls: {calls_made} > {self.max_function_calls}")
        
        return monitor()
    
    def timeout_context(self, timeout_seconds):
        """Context manager for operation timeouts."""
        import signal
        from contextlib import contextmanager
        
        @contextmanager  
        def timeout():
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Operation timed out after {timeout_seconds} seconds")
            
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
            
            try:
                yield
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        
        return timeout()