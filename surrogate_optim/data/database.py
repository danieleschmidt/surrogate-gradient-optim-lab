"""Database utilities for persistent storage of optimization data."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import jax.numpy as jnp
from jax import Array

from ..models.base import Dataset
from ..optimizers.base import OptimizationResult


class OptimizationDatabase:
    """SQLite database for storing optimization experiments and results."""
    
    def __init__(self, db_path: str = "optimization_experiments.db"):
        """Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        
        # Initialize database schema
        self._init_schema()
    
    def _init_schema(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            # Experiments table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    description TEXT,
                    function_name TEXT,
                    surrogate_type TEXT,
                    optimizer_type TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT  -- JSON string
                )
            """)
            
            # Datasets table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS datasets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER,
                    name TEXT NOT NULL,
                    n_samples INTEGER,
                    n_dims INTEGER,
                    has_gradients BOOLEAN,
                    sampling_method TEXT,
                    bounds TEXT,  -- JSON string
                    data_file_path TEXT,  -- Path to actual data file
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (id)
                )
            """)
            
            # Optimization results table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS optimization_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER,
                    dataset_id INTEGER,
                    optimal_point TEXT,  -- JSON array
                    optimal_value REAL,
                    success BOOLEAN,
                    message TEXT,
                    n_iterations INTEGER,
                    n_function_evaluations INTEGER,
                    convergence_history TEXT,  -- JSON array
                    optimization_time REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,  -- JSON string
                    FOREIGN KEY (experiment_id) REFERENCES experiments (id),
                    FOREIGN KEY (dataset_id) REFERENCES datasets (id)
                )
            """)
            
            # Function evaluations table (for caching and analysis)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS function_evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER,
                    input_point TEXT,  -- JSON array
                    output_value REAL,
                    gradient TEXT,  -- JSON array, nullable
                    evaluation_time REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (id)
                )
            """)
            
            # Model parameters table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_parameters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER,
                    parameter_name TEXT,
                    parameter_value TEXT,  -- JSON string
                    parameter_type TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (id)
                )
            """)
            
            # Create indexes for better performance
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_experiments_name 
                ON experiments (name)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_datasets_experiment 
                ON datasets (experiment_id)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_results_experiment 
                ON optimization_results (experiment_id)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_evaluations_experiment 
                ON function_evaluations (experiment_id)
            """)
    
    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get database connection with automatic cleanup."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def create_experiment(
        self,
        name: str,
        description: str = "",
        function_name: str = "",
        surrogate_type: str = "",
        optimizer_type: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Create a new experiment.
        
        Args:
            name: Unique experiment name
            description: Experiment description
            function_name: Name of the black-box function
            surrogate_type: Type of surrogate model
            optimizer_type: Type of optimizer
            metadata: Additional metadata
            
        Returns:
            Experiment ID
        """
        with self._get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO experiments 
                (name, description, function_name, surrogate_type, optimizer_type, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                name,
                description,
                function_name,
                surrogate_type,
                optimizer_type,
                json.dumps(metadata or {})
            ))
            
            return cursor.lastrowid
    
    def get_experiment(self, experiment_id: int) -> Optional[Dict[str, Any]]:
        """Get experiment by ID.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Experiment data or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM experiments WHERE id = ?
            """, (experiment_id,))
            
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
    
    def get_experiment_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get experiment by name.
        
        Args:
            name: Experiment name
            
        Returns:
            Experiment data or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM experiments WHERE name = ?
            """, (name,))
            
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments.
        
        Returns:
            List of experiment data
        """
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM experiments ORDER BY created_at DESC
            """)
            
            return [dict(row) for row in cursor.fetchall()]
    
    def save_dataset(
        self,
        experiment_id: int,
        dataset: Dataset,
        name: str,
        data_dir: str = "data",
    ) -> int:
        """Save dataset to database and file system.
        
        Args:
            experiment_id: Experiment ID
            dataset: Dataset to save
            name: Dataset name
            data_dir: Directory to save data files
            
        Returns:
            Dataset ID
        """
        # Save dataset to file system
        data_path = Path(data_dir)
        data_path.mkdir(exist_ok=True)
        
        data_file = data_path / f"dataset_{name}_{experiment_id}.npz"
        jnp.savez_compressed(
            data_file,
            X=dataset.X,
            y=dataset.y,
            gradients=dataset.gradients if dataset.gradients is not None else jnp.array([])
        )
        
        # Save metadata to database
        with self._get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO datasets 
                (experiment_id, name, n_samples, n_dims, has_gradients, 
                 sampling_method, bounds, data_file_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                experiment_id,
                name,
                dataset.n_samples,
                dataset.n_dims,
                dataset.gradients is not None,
                dataset.metadata.get("sampling_method", "unknown"),
                json.dumps(dataset.metadata.get("bounds", [])),
                str(data_file)
            ))
            
            return cursor.lastrowid
    
    def load_dataset(self, dataset_id: int) -> Optional[Dataset]:
        """Load dataset from database and file system.
        
        Args:
            dataset_id: Dataset ID
            
        Returns:
            Dataset or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM datasets WHERE id = ?
            """, (dataset_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            # Load data from file
            data_file = Path(row["data_file_path"])
            if not data_file.exists():
                raise FileNotFoundError(f"Dataset file not found: {data_file}")
            
            data = jnp.load(data_file)
            X = data["X"]
            y = data["y"]
            gradients = data["gradients"] if data["gradients"].size > 0 else None
            
            # Reconstruct metadata
            metadata = {
                "sampling_method": row["sampling_method"],
                "bounds": json.loads(row["bounds"]),
                "n_samples": row["n_samples"],
                "n_dims": row["n_dims"],
                "has_gradients": bool(row["has_gradients"]),
            }
            
            return Dataset(X=X, y=y, gradients=gradients, metadata=metadata)
    
    def save_optimization_result(
        self,
        experiment_id: int,
        dataset_id: int,
        result: OptimizationResult,
        optimization_time: float = 0.0,
    ) -> int:
        """Save optimization result to database.
        
        Args:
            experiment_id: Experiment ID
            dataset_id: Dataset ID used for optimization
            result: Optimization result
            optimization_time: Time taken for optimization
            
        Returns:
            Result ID
        """
        with self._get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO optimization_results 
                (experiment_id, dataset_id, optimal_point, optimal_value, 
                 success, message, n_iterations, n_function_evaluations,
                 convergence_history, optimization_time, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                experiment_id,
                dataset_id,
                json.dumps(result.x.tolist()),
                float(result.fun),
                result.success,
                result.message,
                result.nit,
                result.nfev,
                json.dumps(result.convergence_history or []),
                optimization_time,
                json.dumps(result.metadata or {})
            ))
            
            return cursor.lastrowid
    
    def get_optimization_results(
        self,
        experiment_id: int
    ) -> List[Dict[str, Any]]:
        """Get all optimization results for an experiment.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            List of optimization results
        """
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM optimization_results 
                WHERE experiment_id = ?
                ORDER BY created_at DESC
            """, (experiment_id,))
            
            results = []
            for row in cursor.fetchall():
                result_dict = dict(row)
                # Parse JSON fields
                result_dict["optimal_point"] = json.loads(result_dict["optimal_point"])
                result_dict["convergence_history"] = json.loads(result_dict["convergence_history"])
                result_dict["metadata"] = json.loads(result_dict["metadata"])
                results.append(result_dict)
            
            return results
    
    def log_function_evaluation(
        self,
        experiment_id: int,
        input_point: Array,
        output_value: float,
        gradient: Optional[Array] = None,
        evaluation_time: float = 0.0,
    ):
        """Log a function evaluation.
        
        Args:
            experiment_id: Experiment ID
            input_point: Input point
            output_value: Function output
            gradient: Optional gradient
            evaluation_time: Time taken for evaluation
        """
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO function_evaluations 
                (experiment_id, input_point, output_value, gradient, evaluation_time)
                VALUES (?, ?, ?, ?, ?)
            """, (
                experiment_id,
                json.dumps(input_point.tolist()),
                float(output_value),
                json.dumps(gradient.tolist()) if gradient is not None else None,
                evaluation_time
            ))
    
    def get_experiment_statistics(self, experiment_id: int) -> Dict[str, Any]:
        """Get statistics for an experiment.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Dictionary with experiment statistics
        """
        with self._get_connection() as conn:
            # Get basic experiment info
            exp_cursor = conn.execute("""
                SELECT * FROM experiments WHERE id = ?
            """, (experiment_id,))
            experiment = exp_cursor.fetchone()
            
            if not experiment:
                return {}
            
            # Count datasets
            dataset_cursor = conn.execute("""
                SELECT COUNT(*) as count FROM datasets WHERE experiment_id = ?
            """, (experiment_id,))
            n_datasets = dataset_cursor.fetchone()["count"]
            
            # Count results
            result_cursor = conn.execute("""
                SELECT COUNT(*) as count FROM optimization_results WHERE experiment_id = ?
            """, (experiment_id,))
            n_results = result_cursor.fetchone()["count"]
            
            # Count function evaluations
            eval_cursor = conn.execute("""
                SELECT COUNT(*) as count FROM function_evaluations WHERE experiment_id = ?
            """, (experiment_id,))
            n_evaluations = eval_cursor.fetchone()["count"]
            
            # Get best result
            best_cursor = conn.execute("""
                SELECT MIN(optimal_value) as best_value 
                FROM optimization_results 
                WHERE experiment_id = ? AND success = 1
            """, (experiment_id,))
            best_result = best_cursor.fetchone()
            best_value = best_result["best_value"] if best_result["best_value"] else None
            
            return {
                "experiment_id": experiment_id,
                "experiment_name": experiment["name"],
                "created_at": experiment["created_at"],
                "n_datasets": n_datasets,
                "n_optimization_runs": n_results,
                "n_function_evaluations": n_evaluations,
                "best_value": best_value,
                "surrogate_type": experiment["surrogate_type"],
                "optimizer_type": experiment["optimizer_type"],
            }
    
    def delete_experiment(self, experiment_id: int):
        """Delete an experiment and all associated data.
        
        Args:
            experiment_id: Experiment ID
        """
        with self._get_connection() as conn:
            # Delete in reverse order of dependencies
            conn.execute("DELETE FROM function_evaluations WHERE experiment_id = ?", (experiment_id,))
            conn.execute("DELETE FROM model_parameters WHERE experiment_id = ?", (experiment_id,))
            conn.execute("DELETE FROM optimization_results WHERE experiment_id = ?", (experiment_id,))
            
            # Get dataset files to delete
            cursor = conn.execute("SELECT data_file_path FROM datasets WHERE experiment_id = ?", (experiment_id,))
            data_files = [row["data_file_path"] for row in cursor.fetchall()]
            
            conn.execute("DELETE FROM datasets WHERE experiment_id = ?", (experiment_id,))
            conn.execute("DELETE FROM experiments WHERE id = ?", (experiment_id,))
            
            # Delete data files
            for data_file in data_files:
                try:
                    Path(data_file).unlink()
                except FileNotFoundError:
                    pass  # File already deleted
    
    def export_experiment(self, experiment_id: int, output_path: str):
        """Export experiment data to JSON file.
        
        Args:
            experiment_id: Experiment ID
            output_path: Output file path
        """
        # Get all experiment data
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        # Get datasets
        with self._get_connection() as conn:
            dataset_cursor = conn.execute("""
                SELECT * FROM datasets WHERE experiment_id = ?
            """, (experiment_id,))
            datasets = [dict(row) for row in dataset_cursor.fetchall()]
        
        # Get results
        results = self.get_optimization_results(experiment_id)
        
        # Get statistics
        stats = self.get_experiment_statistics(experiment_id)
        
        # Create export data
        export_data = {
            "experiment": experiment,
            "datasets": datasets,
            "results": results,
            "statistics": stats,
            "exported_at": datetime.now().isoformat(),
        }
        
        # Save to file
        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2)


# Global database instance
_default_db = None


def get_default_database() -> OptimizationDatabase:
    """Get default database instance."""
    global _default_db
    if _default_db is None:
        _default_db = OptimizationDatabase()
    return _default_db