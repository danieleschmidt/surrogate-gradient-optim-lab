"""Edge Computing Module for distributed surrogate optimization.

This module enables deployment and execution of surrogate optimization
on edge computing infrastructure for low-latency, distributed processing.
"""

from .edge_runtime import EdgeOptimizationRuntime
from .distributed_coordinator import DistributedEdgeCoordinator
from .lightweight_models import LightweightSurrogateFactory
from .edge_deployment import EdgeDeploymentManager

__all__ = [
    "EdgeOptimizationRuntime",
    "DistributedEdgeCoordinator", 
    "LightweightSurrogateFactory",
    "EdgeDeploymentManager",
]