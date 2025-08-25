"""Edge Computing Module for distributed surrogate optimization.

This module enables deployment and execution of surrogate optimization
on edge computing infrastructure for low-latency, distributed processing.
"""

from .distributed_coordinator import DistributedEdgeCoordinator
from .edge_deployment import EdgeDeploymentManager
from .edge_runtime import EdgeOptimizationRuntime
from .lightweight_models import LightweightSurrogateFactory

__all__ = [
    "DistributedEdgeCoordinator",
    "EdgeDeploymentManager",
    "EdgeOptimizationRuntime",
    "LightweightSurrogateFactory",
]
