"""Comprehensive validation and security for Generation 2 robustness."""

from typing import Any, Callable, Dict, List

import jax.numpy as jnp

from ..models.base import Dataset
from ..monitoring.enhanced_logging import enhanced_logger
from ..quality.security_checks import SecurityScanner


class RobustValidator:
    """Comprehensive validator for robust surrogate optimization."""

    def __init__(self):
        self.security_scanner = SecurityScanner()
        self.validation_history: List[Dict[str, Any]] = []

    def validate_complete_workflow(
        self,
        objective_function: Callable,
        data: Dataset,
        surrogate_config: Dict[str, Any],
        optimizer_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Perform comprehensive validation of optimization workflow."""
        enhanced_logger.info("Starting comprehensive workflow validation")

        validation_results = {
            "security_scan": {},
            "data_validation": {},
            "config_validation": {},
            "function_validation": {},
            "overall_status": "unknown",
            "warnings": [],
            "errors": [],
        }

        try:
            # 1. Security validation
            enhanced_logger.debug("Running security scan")
            security_results = self.security_scanner.scan_function(objective_function)
            validation_results["security_scan"] = security_results

            if security_results.get("issues"):
                for issue in security_results["issues"]:
                    if issue.get("severity") == "high":
                        validation_results["errors"].append(
                            f"Security: {issue.get('message')}"
                        )
                    else:
                        validation_results["warnings"].append(
                            f"Security: {issue.get('message')}"
                        )

            # 2. Data validation
            enhanced_logger.debug("Validating training data")
            data_issues = self._validate_dataset_comprehensive(data)
            validation_results["data_validation"] = data_issues

            # 3. Configuration validation
            enhanced_logger.debug("Validating configurations")
            config_issues = self._validate_configurations(
                surrogate_config, optimizer_config
            )
            validation_results["config_validation"] = config_issues

            # 4. Function validation
            enhanced_logger.debug("Validating objective function")
            function_issues = self._validate_objective_function(
                objective_function, data
            )
            validation_results["function_validation"] = function_issues

            # Determine overall status
            if validation_results["errors"]:
                validation_results["overall_status"] = "failed"
            elif validation_results["warnings"]:
                validation_results["overall_status"] = "warning"
            else:
                validation_results["overall_status"] = "passed"

            # Log summary
            enhanced_logger.info(
                f"Validation complete. Status: {validation_results['overall_status']}"
            )
            if validation_results["warnings"]:
                enhanced_logger.warning(
                    f"Found {len(validation_results['warnings'])} warnings"
                )
            if validation_results["errors"]:
                enhanced_logger.error(
                    f"Found {len(validation_results['errors'])} errors"
                )

        except Exception as e:
            enhanced_logger.error(f"Validation failed with exception: {e}")
            validation_results["errors"].append(f"Validation exception: {e!s}")
            validation_results["overall_status"] = "failed"

        # Store in history
        self.validation_history.append(validation_results)

        return validation_results

    def _validate_dataset_comprehensive(self, data: Dataset) -> Dict[str, Any]:
        """Comprehensive dataset validation."""
        issues = {"errors": [], "warnings": [], "info": []}

        # Basic structure validation
        if data.n_samples == 0:
            issues["errors"].append("Dataset is empty")
            return issues

        # Data quality checks
        if jnp.any(jnp.isnan(data.X)):
            issues["errors"].append("Input features contain NaN values")

        if jnp.any(jnp.isnan(data.y)):
            issues["errors"].append("Target values contain NaN values")

        if jnp.any(jnp.isinf(data.X)):
            issues["errors"].append("Input features contain infinite values")

        if jnp.any(jnp.isinf(data.y)):
            issues["errors"].append("Target values contain infinite values")

        # Distribution checks
        if data.n_samples < 20:
            issues["warnings"].append(
                "Very small dataset - consider collecting more data"
            )

        if jnp.std(data.y) < 1e-10:
            issues["warnings"].append(
                "Target values have very low variance - may be constant"
            )

        # Dimensionality checks
        if data.n_dims > 50:
            issues["warnings"].append(
                "High-dimensional data may be challenging to optimize"
            )

        # Correlation analysis
        if data.n_dims > 1:
            correlation_matrix = jnp.corrcoef(data.X.T)
            high_corr = jnp.abs(correlation_matrix) > 0.95
            high_corr = high_corr & ~jnp.eye(data.n_dims, dtype=bool)
            if jnp.any(high_corr):
                issues["warnings"].append(
                    "High correlation detected between input features"
                )

        issues["info"].append(
            f"Dataset: {data.n_samples} samples, {data.n_dims} dimensions"
        )
        issues["info"].append(
            f"Target range: [{float(jnp.min(data.y)):.4f}, {float(jnp.max(data.y)):.4f}]"
        )

        return issues

    def _validate_configurations(
        self, surrogate_config: Dict, optimizer_config: Dict
    ) -> Dict[str, Any]:
        """Validate surrogate and optimizer configurations."""
        issues = {"errors": [], "warnings": [], "info": []}

        # Surrogate config validation
        if "hidden_dims" in surrogate_config:
            dims = surrogate_config["hidden_dims"]
            if isinstance(dims, list) and max(dims) > 1000:
                issues["warnings"].append("Very large network architecture may be slow")

        # Optimizer config validation
        if "learning_rate" in optimizer_config:
            lr = optimizer_config["learning_rate"]
            if lr > 1.0:
                issues["warnings"].append("High learning rate may cause instability")
            elif lr < 1e-6:
                issues["warnings"].append(
                    "Very low learning rate may cause slow convergence"
                )

        if "max_iterations" in optimizer_config:
            max_iter = optimizer_config["max_iterations"]
            if max_iter > 10000:
                issues["warnings"].append(
                    "Very high iteration count may take long time"
                )

        issues["info"].append(f"Surrogate config: {len(surrogate_config)} parameters")
        issues["info"].append(f"Optimizer config: {len(optimizer_config)} parameters")

        return issues

    def _validate_objective_function(
        self, func: Callable, data: Dataset
    ) -> Dict[str, Any]:
        """Validate objective function behavior."""
        issues = {"errors": [], "warnings": [], "info": []}

        try:
            # Test function on a sample point
            test_point = data.X[0] if data.n_samples > 0 else jnp.zeros(2)
            result = func(test_point)

            # Check return type
            if not isinstance(result, (float, int, jnp.ndarray)):
                issues["errors"].append("Function must return numeric value")

            # Check for NaN/inf in function output
            if jnp.isnan(result) or jnp.isinf(result):
                issues["warnings"].append("Function returns NaN/inf for test input")

            # Check function signature
            import inspect

            sig = inspect.signature(func)
            if len(sig.parameters) != 1:
                issues["warnings"].append("Function should take exactly one parameter")

            issues["info"].append("Objective function validation passed")

        except Exception as e:
            issues["errors"].append(f"Function evaluation failed: {e!s}")

        return issues


# Global robust validator instance
robust_validator = RobustValidator()
