"""Self-Healing Pipeline Guard System.

Autonomous pipeline monitoring and recovery for surrogate optimization workflows.
"""

from .compliance_framework import ComplianceFramework, LegalBasis, ProcessingPurpose
from .error_handling import AdvancedErrorHandler, circuit_breaker, with_retry
from .global_deployment import ComplianceStandard, GlobalDeploymentManager, Region
from .health_diagnostics import HealthDiagnostics
from .monitoring_dashboard import MonitoringDashboard
from .performance_optimization import PerformanceOptimizer, optimize_performance
from .pipeline_monitor import PipelineMonitor
from .recovery_engine import RecoveryEngine
from .robust_monitoring import MonitoringConfig, RobustMonitor
from .scalable_architecture import ScalableOptimizer, ScalingConfig
from .security_validation import SecurityConfig, SecurityManager
from .self_healing_optimizer import SelfHealingOptimizer

__all__ = [
    "AdvancedErrorHandler",
    "ComplianceFramework",
    "ComplianceStandard",
    "GlobalDeploymentManager",
    "HealthDiagnostics",
    "LegalBasis",
    "MonitoringConfig",
    "MonitoringDashboard",
    "PerformanceOptimizer",
    "PipelineMonitor",
    "ProcessingPurpose",
    "RecoveryEngine",
    "Region",
    "RobustMonitor",
    "ScalableOptimizer",
    "ScalingConfig",
    "SecurityConfig",
    "SecurityManager",
    "SelfHealingOptimizer",
    "circuit_breaker",
    "optimize_performance",
    "with_retry"
]
