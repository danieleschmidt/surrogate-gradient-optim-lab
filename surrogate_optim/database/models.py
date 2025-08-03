"""Database models for surrogate optimization."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum

import numpy as np


class ExperimentStatus(Enum):
    """Experiment status enumeration."""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SurrogateType(Enum):
    """Surrogate model type enumeration."""
    NEURAL_NETWORK = "neural_network"
    GAUSSIAN_PROCESS = "gaussian_process"
    RANDOM_FOREST = "random_forest"
    HYBRID = "hybrid"


@dataclass
class ExperimentModel:
    """Experiment database model."""
    id: Optional[int] = None
    name: str = ""
    description: str = ""
    status: ExperimentStatus = ExperimentStatus.CREATED
    config: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Results summary
    best_value: Optional[float] = None
    best_point: Optional[np.ndarray] = None
    n_evaluations: int = 0
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        from .connection import serialize_json
        
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "config": serialize_json(self.config),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "best_value": self.best_value,
            "best_point": serialize_json(self.best_point) if self.best_point is not None else None,
            "n_evaluations": self.n_evaluations,
            "tags": serialize_json(self.tags),
            "metadata": serialize_json(self.metadata),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentModel":
        """Create from dictionary from database."""
        from .connection import deserialize_json
        
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            status=ExperimentStatus(data["status"]),
            config=deserialize_json(data["config"]) if data["config"] else {},
            created_at=datetime.fromisoformat(data["created_at"]) if data["created_at"] else None,
            updated_at=datetime.fromisoformat(data["updated_at"]) if data["updated_at"] else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data["completed_at"] else None,
            best_value=data["best_value"],
            best_point=deserialize_json(data["best_point"]) if data["best_point"] else None,
            n_evaluations=data["n_evaluations"],
            tags=deserialize_json(data["tags"]) if data["tags"] else [],
            metadata=deserialize_json(data["metadata"]) if data["metadata"] else {},
        )


@dataclass
class DatasetModel:
    """Dataset database model."""
    id: Optional[int] = None
    experiment_id: Optional[int] = None
    name: str = ""
    description: str = ""
    
    # Data arrays
    X: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    gradients: Optional[np.ndarray] = None
    
    # Dataset metadata
    n_samples: int = 0
    n_dims: int = 0
    bounds: Optional[List[List[float]]] = None
    sampling_strategy: str = ""
    
    # Timestamps
    created_at: Optional[datetime] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        from .connection import serialize_json
        
        return {
            "id": self.id,
            "experiment_id": self.experiment_id,
            "name": self.name,
            "description": self.description,
            "X": serialize_json(self.X) if self.X is not None else None,
            "y": serialize_json(self.y) if self.y is not None else None,
            "gradients": serialize_json(self.gradients) if self.gradients is not None else None,
            "n_samples": self.n_samples,
            "n_dims": self.n_dims,
            "bounds": serialize_json(self.bounds) if self.bounds else None,
            "sampling_strategy": self.sampling_strategy,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "metadata": serialize_json(self.metadata),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetModel":
        """Create from dictionary from database."""
        from .connection import deserialize_json
        
        return cls(
            id=data["id"],
            experiment_id=data["experiment_id"],
            name=data["name"],
            description=data["description"],
            X=deserialize_json(data["X"]) if data["X"] else None,
            y=deserialize_json(data["y"]) if data["y"] else None,
            gradients=deserialize_json(data["gradients"]) if data["gradients"] else None,
            n_samples=data["n_samples"],
            n_dims=data["n_dims"],
            bounds=deserialize_json(data["bounds"]) if data["bounds"] else None,
            sampling_strategy=data["sampling_strategy"],
            created_at=datetime.fromisoformat(data["created_at"]) if data["created_at"] else None,
            metadata=deserialize_json(data["metadata"]) if data["metadata"] else {},
        )


@dataclass
class SurrogateModel:
    """Surrogate model database model."""
    id: Optional[int] = None
    experiment_id: Optional[int] = None
    dataset_id: Optional[int] = None
    name: str = ""
    surrogate_type: SurrogateType = SurrogateType.NEURAL_NETWORK
    
    # Model configuration
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Training results
    training_loss: Optional[float] = None
    validation_score: Optional[float] = None
    training_time: Optional[float] = None
    
    # Model artifacts (serialized)
    model_data: Optional[bytes] = None
    
    # Timestamps
    created_at: Optional[datetime] = None
    trained_at: Optional[datetime] = None
    
    # Performance metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        from .connection import serialize_json
        
        return {
            "id": self.id,
            "experiment_id": self.experiment_id,
            "dataset_id": self.dataset_id,
            "name": self.name,
            "surrogate_type": self.surrogate_type.value,
            "config": serialize_json(self.config),
            "training_loss": self.training_loss,
            "validation_score": self.validation_score,
            "training_time": self.training_time,
            "model_data": self.model_data,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "trained_at": self.trained_at.isoformat() if self.trained_at else None,
            "metrics": serialize_json(self.metrics),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SurrogateModel":
        """Create from dictionary from database."""
        from .connection import deserialize_json
        
        return cls(
            id=data["id"],
            experiment_id=data["experiment_id"],
            dataset_id=data["dataset_id"],
            name=data["name"],
            surrogate_type=SurrogateType(data["surrogate_type"]),
            config=deserialize_json(data["config"]) if data["config"] else {},
            training_loss=data["training_loss"],
            validation_score=data["validation_score"],
            training_time=data["training_time"],
            model_data=data["model_data"],
            created_at=datetime.fromisoformat(data["created_at"]) if data["created_at"] else None,
            trained_at=datetime.fromisoformat(data["trained_at"]) if data["trained_at"] else None,
            metrics=deserialize_json(data["metrics"]) if data["metrics"] else {},
        )


@dataclass
class OptimizationResult:
    """Optimization result database model."""
    id: Optional[int] = None
    experiment_id: Optional[int] = None
    surrogate_id: Optional[int] = None
    
    # Optimization configuration
    method: str = ""
    initial_point: Optional[np.ndarray] = None
    bounds: Optional[List[List[float]]] = None
    
    # Results
    optimal_point: Optional[np.ndarray] = None
    optimal_value: Optional[float] = None
    success: bool = False
    message: str = ""
    
    # Performance metrics
    n_iterations: int = 0
    n_function_evaluations: int = 0
    n_gradient_evaluations: int = 0
    optimization_time: Optional[float] = None
    
    # Trajectory (optional)
    trajectory: Optional[np.ndarray] = None
    
    # Timestamps
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        from .connection import serialize_json
        
        return {
            "id": self.id,
            "experiment_id": self.experiment_id,
            "surrogate_id": self.surrogate_id,
            "method": self.method,
            "initial_point": serialize_json(self.initial_point) if self.initial_point is not None else None,
            "bounds": serialize_json(self.bounds) if self.bounds else None,
            "optimal_point": serialize_json(self.optimal_point) if self.optimal_point is not None else None,
            "optimal_value": self.optimal_value,
            "success": self.success,
            "message": self.message,
            "n_iterations": self.n_iterations,
            "n_function_evaluations": self.n_function_evaluations,
            "n_gradient_evaluations": self.n_gradient_evaluations,
            "optimization_time": self.optimization_time,
            "trajectory": serialize_json(self.trajectory) if self.trajectory is not None else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metadata": serialize_json(self.metadata),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimizationResult":
        """Create from dictionary from database."""
        from .connection import deserialize_json
        
        return cls(
            id=data["id"],
            experiment_id=data["experiment_id"],
            surrogate_id=data["surrogate_id"],
            method=data["method"],
            initial_point=deserialize_json(data["initial_point"]) if data["initial_point"] else None,
            bounds=deserialize_json(data["bounds"]) if data["bounds"] else None,
            optimal_point=deserialize_json(data["optimal_point"]) if data["optimal_point"] else None,
            optimal_value=data["optimal_value"],
            success=data["success"],
            message=data["message"],
            n_iterations=data["n_iterations"],
            n_function_evaluations=data["n_function_evaluations"],
            n_gradient_evaluations=data["n_gradient_evaluations"],
            optimization_time=data["optimization_time"],
            trajectory=deserialize_json(data["trajectory"]) if data["trajectory"] else None,
            started_at=datetime.fromisoformat(data["started_at"]) if data["started_at"] else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data["completed_at"] else None,
            metadata=deserialize_json(data["metadata"]) if data["metadata"] else {},
        )