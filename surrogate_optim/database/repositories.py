"""Data access layer repositories."""

from datetime import datetime
from typing import List, Optional, Dict, Any

from .connection import get_db_manager
from .models import ExperimentModel, DatasetModel, SurrogateModel, OptimizationResult, ExperimentStatus


class BaseRepository:
    """Base repository with common operations."""
    
    def __init__(self):
        """Initialize repository."""
        self.db = get_db_manager()


class ExperimentRepository(BaseRepository):
    """Repository for experiment operations."""
    
    def create(self, experiment: ExperimentModel) -> int:
        """Create new experiment."""
        experiment.created_at = datetime.now()
        experiment.updated_at = experiment.created_at
        
        data = experiment.to_dict()
        
        query = """
        INSERT INTO experiments (
            name, description, status, config, created_at, updated_at,
            best_value, best_point, n_evaluations, tags, metadata
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (
            data["name"], data["description"], data["status"], data["config"],
            data["created_at"], data["updated_at"], data["best_value"],
            data["best_point"], data["n_evaluations"], data["tags"], data["metadata"]
        )
        
        experiment_id = self.db.execute_insert(query, params)
        experiment.id = experiment_id
        return experiment_id
    
    def get_by_id(self, experiment_id: int) -> Optional[ExperimentModel]:
        """Get experiment by ID."""
        query = "SELECT * FROM experiments WHERE id = ?"
        results = self.db.execute_query(query, (experiment_id,))
        
        if results:
            return ExperimentModel.from_dict(dict(results[0]))
        return None
    
    def get_by_name(self, name: str) -> Optional[ExperimentModel]:
        """Get experiment by name."""
        query = "SELECT * FROM experiments WHERE name = ?"
        results = self.db.execute_query(query, (name,))
        
        if results:
            return ExperimentModel.from_dict(dict(results[0]))
        return None
    
    def list_all(self, status: Optional[ExperimentStatus] = None) -> List[ExperimentModel]:
        """List all experiments, optionally filtered by status."""
        if status:
            query = "SELECT * FROM experiments WHERE status = ? ORDER BY created_at DESC"
            results = self.db.execute_query(query, (status.value,))
        else:
            query = "SELECT * FROM experiments ORDER BY created_at DESC"
            results = self.db.execute_query(query)
        
        return [ExperimentModel.from_dict(dict(row)) for row in results]
    
    def update(self, experiment: ExperimentModel) -> bool:
        """Update experiment."""
        experiment.updated_at = datetime.now()
        data = experiment.to_dict()
        
        query = """
        UPDATE experiments SET
            name = ?, description = ?, status = ?, config = ?, updated_at = ?,
            completed_at = ?, best_value = ?, best_point = ?, n_evaluations = ?,
            tags = ?, metadata = ?
        WHERE id = ?
        """
        
        params = (
            data["name"], data["description"], data["status"], data["config"],
            data["updated_at"], data["completed_at"], data["best_value"],
            data["best_point"], data["n_evaluations"], data["tags"],
            data["metadata"], experiment.id
        )
        
        rows_affected = self.db.execute_update(query, params)
        return rows_affected > 0
    
    def delete(self, experiment_id: int) -> bool:
        """Delete experiment and all related data."""
        # Delete in reverse dependency order
        delete_queries = [
            "DELETE FROM optimization_results WHERE experiment_id = ?",
            "DELETE FROM surrogates WHERE experiment_id = ?",
            "DELETE FROM datasets WHERE experiment_id = ?",
            "DELETE FROM experiments WHERE id = ?",
        ]
        
        total_affected = 0
        for query in delete_queries:
            rows_affected = self.db.execute_update(query, (experiment_id,))
            total_affected += rows_affected
        
        return total_affected > 0
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for experiments."""
        queries = {
            "total_experiments": "SELECT COUNT(*) as count FROM experiments",
            "completed_experiments": "SELECT COUNT(*) as count FROM experiments WHERE status = 'completed'",
            "running_experiments": "SELECT COUNT(*) as count FROM experiments WHERE status = 'running'",
            "failed_experiments": "SELECT COUNT(*) as count FROM experiments WHERE status = 'failed'",
            "avg_evaluations": "SELECT AVG(n_evaluations) as avg FROM experiments WHERE n_evaluations > 0",
        }
        
        stats = {}
        for key, query in queries.items():
            result = self.db.execute_query(query)
            if result:
                value = result[0]["count"] if "count" in result[0] else result[0]["avg"]
                stats[key] = value if value is not None else 0
            else:
                stats[key] = 0
        
        return stats


class DatasetRepository(BaseRepository):
    """Repository for dataset operations."""
    
    def create(self, dataset: DatasetModel) -> int:
        """Create new dataset."""
        dataset.created_at = datetime.now()
        data = dataset.to_dict()
        
        query = """
        INSERT INTO datasets (
            experiment_id, name, description, X, y, gradients, n_samples, n_dims,
            bounds, sampling_strategy, created_at, metadata
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (
            data["experiment_id"], data["name"], data["description"],
            data["X"], data["y"], data["gradients"], data["n_samples"],
            data["n_dims"], data["bounds"], data["sampling_strategy"],
            data["created_at"], data["metadata"]
        )
        
        dataset_id = self.db.execute_insert(query, params)
        dataset.id = dataset_id
        return dataset_id
    
    def get_by_id(self, dataset_id: int) -> Optional[DatasetModel]:
        """Get dataset by ID."""
        query = "SELECT * FROM datasets WHERE id = ?"
        results = self.db.execute_query(query, (dataset_id,))
        
        if results:
            return DatasetModel.from_dict(dict(results[0]))
        return None
    
    def list_by_experiment(self, experiment_id: int) -> List[DatasetModel]:
        """List datasets for an experiment."""
        query = "SELECT * FROM datasets WHERE experiment_id = ? ORDER BY created_at DESC"
        results = self.db.execute_query(query, (experiment_id,))
        
        return [DatasetModel.from_dict(dict(row)) for row in results]
    
    def update(self, dataset: DatasetModel) -> bool:
        """Update dataset."""
        data = dataset.to_dict()
        
        query = """
        UPDATE datasets SET
            name = ?, description = ?, X = ?, y = ?, gradients = ?,
            n_samples = ?, n_dims = ?, bounds = ?, sampling_strategy = ?, metadata = ?
        WHERE id = ?
        """
        
        params = (
            data["name"], data["description"], data["X"], data["y"],
            data["gradients"], data["n_samples"], data["n_dims"],
            data["bounds"], data["sampling_strategy"], data["metadata"],
            dataset.id
        )
        
        rows_affected = self.db.execute_update(query, params)
        return rows_affected > 0
    
    def delete(self, dataset_id: int) -> bool:
        """Delete dataset and related surrogates."""
        # Delete surrogates first (they reference datasets)
        self.db.execute_update("DELETE FROM surrogates WHERE dataset_id = ?", (dataset_id,))
        
        # Delete dataset
        rows_affected = self.db.execute_update("DELETE FROM datasets WHERE id = ?", (dataset_id,))
        return rows_affected > 0


class SurrogateRepository(BaseRepository):
    """Repository for surrogate model operations."""
    
    def create(self, surrogate: SurrogateModel) -> int:
        """Create new surrogate model."""
        surrogate.created_at = datetime.now()
        data = surrogate.to_dict()
        
        query = """
        INSERT INTO surrogates (
            experiment_id, dataset_id, name, surrogate_type, config,
            training_loss, validation_score, training_time, model_data,
            created_at, trained_at, metrics
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (
            data["experiment_id"], data["dataset_id"], data["name"],
            data["surrogate_type"], data["config"], data["training_loss"],
            data["validation_score"], data["training_time"], data["model_data"],
            data["created_at"], data["trained_at"], data["metrics"]
        )
        
        surrogate_id = self.db.execute_insert(query, params)
        surrogate.id = surrogate_id
        return surrogate_id
    
    def get_by_id(self, surrogate_id: int) -> Optional[SurrogateModel]:
        """Get surrogate by ID."""
        query = "SELECT * FROM surrogates WHERE id = ?"
        results = self.db.execute_query(query, (surrogate_id,))
        
        if results:
            return SurrogateModel.from_dict(dict(results[0]))
        return None
    
    def list_by_experiment(self, experiment_id: int) -> List[SurrogateModel]:
        """List surrogates for an experiment."""
        query = "SELECT * FROM surrogates WHERE experiment_id = ? ORDER BY created_at DESC"
        results = self.db.execute_query(query, (experiment_id,))
        
        return [SurrogateModel.from_dict(dict(row)) for row in results]
    
    def list_by_dataset(self, dataset_id: int) -> List[SurrogateModel]:
        """List surrogates for a dataset."""
        query = "SELECT * FROM surrogates WHERE dataset_id = ? ORDER BY created_at DESC"
        results = self.db.execute_query(query, (dataset_id,))
        
        return [SurrogateModel.from_dict(dict(row)) for row in results]
    
    def update(self, surrogate: SurrogateModel) -> bool:
        """Update surrogate model."""
        data = surrogate.to_dict()
        
        query = """
        UPDATE surrogates SET
            name = ?, surrogate_type = ?, config = ?, training_loss = ?,
            validation_score = ?, training_time = ?, model_data = ?,
            trained_at = ?, metrics = ?
        WHERE id = ?
        """
        
        params = (
            data["name"], data["surrogate_type"], data["config"],
            data["training_loss"], data["validation_score"], data["training_time"],
            data["model_data"], data["trained_at"], data["metrics"], surrogate.id
        )
        
        rows_affected = self.db.execute_update(query, params)
        return rows_affected > 0
    
    def delete(self, surrogate_id: int) -> bool:
        """Delete surrogate and related optimization results."""
        # Delete optimization results first
        self.db.execute_update("DELETE FROM optimization_results WHERE surrogate_id = ?", (surrogate_id,))
        
        # Delete surrogate
        rows_affected = self.db.execute_update("DELETE FROM surrogates WHERE id = ?", (surrogate_id,))
        return rows_affected > 0


class OptimizationResultRepository(BaseRepository):
    """Repository for optimization result operations."""
    
    def create(self, result: OptimizationResult) -> int:
        """Create new optimization result."""
        data = result.to_dict()
        
        query = """
        INSERT INTO optimization_results (
            experiment_id, surrogate_id, method, initial_point, bounds,
            optimal_point, optimal_value, success, message, n_iterations,
            n_function_evaluations, n_gradient_evaluations, optimization_time,
            trajectory, started_at, completed_at, metadata
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (
            data["experiment_id"], data["surrogate_id"], data["method"],
            data["initial_point"], data["bounds"], data["optimal_point"],
            data["optimal_value"], data["success"], data["message"],
            data["n_iterations"], data["n_function_evaluations"],
            data["n_gradient_evaluations"], data["optimization_time"],
            data["trajectory"], data["started_at"], data["completed_at"],
            data["metadata"]
        )
        
        result_id = self.db.execute_insert(query, params)
        result.id = result_id
        return result_id
    
    def get_by_id(self, result_id: int) -> Optional[OptimizationResult]:
        """Get optimization result by ID."""
        query = "SELECT * FROM optimization_results WHERE id = ?"
        results = self.db.execute_query(query, (result_id,))
        
        if results:
            return OptimizationResult.from_dict(dict(results[0]))
        return None
    
    def list_by_experiment(self, experiment_id: int) -> List[OptimizationResult]:
        """List optimization results for an experiment."""
        query = "SELECT * FROM optimization_results WHERE experiment_id = ? ORDER BY completed_at DESC"
        results = self.db.execute_query(query, (experiment_id,))
        
        return [OptimizationResult.from_dict(dict(row)) for row in results]
    
    def list_by_surrogate(self, surrogate_id: int) -> List[OptimizationResult]:
        """List optimization results for a surrogate."""
        query = "SELECT * FROM optimization_results WHERE surrogate_id = ? ORDER BY completed_at DESC"
        results = self.db.execute_query(query, (surrogate_id,))
        
        return [OptimizationResult.from_dict(dict(row)) for row in results]
    
    def get_best_result(self, experiment_id: int) -> Optional[OptimizationResult]:
        """Get best optimization result for an experiment."""
        query = """
        SELECT * FROM optimization_results 
        WHERE experiment_id = ? AND success = 1 
        ORDER BY optimal_value DESC 
        LIMIT 1
        """
        results = self.db.execute_query(query, (experiment_id,))
        
        if results:
            return OptimizationResult.from_dict(dict(results[0]))
        return None
    
    def update(self, result: OptimizationResult) -> bool:
        """Update optimization result."""
        data = result.to_dict()
        
        query = """
        UPDATE optimization_results SET
            method = ?, initial_point = ?, bounds = ?, optimal_point = ?,
            optimal_value = ?, success = ?, message = ?, n_iterations = ?,
            n_function_evaluations = ?, n_gradient_evaluations = ?,
            optimization_time = ?, trajectory = ?, completed_at = ?, metadata = ?
        WHERE id = ?
        """
        
        params = (
            data["method"], data["initial_point"], data["bounds"],
            data["optimal_point"], data["optimal_value"], data["success"],
            data["message"], data["n_iterations"], data["n_function_evaluations"],
            data["n_gradient_evaluations"], data["optimization_time"],
            data["trajectory"], data["completed_at"], data["metadata"], result.id
        )
        
        rows_affected = self.db.execute_update(query, params)
        return rows_affected > 0
    
    def delete(self, result_id: int) -> bool:
        """Delete optimization result."""
        rows_affected = self.db.execute_update("DELETE FROM optimization_results WHERE id = ?", (result_id,))
        return rows_affected > 0