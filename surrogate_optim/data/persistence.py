"""Data persistence utilities for surrogate models and optimization results."""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import jax.numpy as jnp
from jax import Array

from ..models.base import Dataset
from ..optimizers.base import OptimizationResult


class DataPersistence:
    """Handle saving and loading of datasets and optimization results."""
    
    def __init__(self, base_path: Union[str, Path] = "data"):
        """Initialize data persistence manager.
        
        Args:
            base_path: Base directory for data storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.base_path / "datasets").mkdir(exist_ok=True)
        (self.base_path / "results").mkdir(exist_ok=True)
        (self.base_path / "models").mkdir(exist_ok=True)
    
    def save_dataset(
        self,
        dataset: Dataset,
        name: str,
        format: str = "npz",
        overwrite: bool = False,
    ) -> Path:
        """Save dataset to disk.
        
        Args:
            dataset: Dataset to save
            name: Name for the saved dataset
            format: File format ('npz', 'json', 'pickle')
            overwrite: Whether to overwrite existing files
            
        Returns:
            Path to saved file
        """
        if format == "npz":
            filepath = self.base_path / "datasets" / f"{name}.npz"
        elif format == "json":
            filepath = self.base_path / "datasets" / f"{name}.json"
        elif format == "pickle":
            filepath = self.base_path / "datasets" / f"{name}.pkl"
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        if filepath.exists() and not overwrite:
            raise FileExistsError(f"File {filepath} already exists. Use overwrite=True.")
        
        if format == "npz":
            # NumPy compressed format - most efficient for arrays
            save_dict = {
                "X": dataset.X,
                "y": dataset.y,
            }
            if dataset.gradients is not None:
                save_dict["gradients"] = dataset.gradients
            
            jnp.savez_compressed(filepath, **save_dict)
            
            # Save metadata separately
            metadata_path = filepath.with_suffix(".json")
            with open(metadata_path, "w") as f:
                json.dump(dataset.metadata, f, indent=2)
        
        elif format == "json":
            # JSON format - human readable but less efficient
            data_dict = {
                "X": dataset.X.tolist(),
                "y": dataset.y.tolist(),
                "gradients": dataset.gradients.tolist() if dataset.gradients is not None else None,
                "metadata": dataset.metadata,
            }
            
            with open(filepath, "w") as f:
                json.dump(data_dict, f, indent=2)
        
        elif format == "pickle":
            # Pickle format - most flexible but Python-specific
            with open(filepath, "wb") as f:
                pickle.dump(dataset, f)
        
        return filepath
    
    def load_dataset(self, name: str, format: str = "auto") -> Dataset:
        """Load dataset from disk.
        
        Args:
            name: Name of the dataset to load
            format: File format ('npz', 'json', 'pickle', 'auto')
            
        Returns:
            Loaded dataset
        """
        if format == "auto":
            # Try to detect format from existing files
            for fmt in ["npz", "json", "pickle"]:
                if self._dataset_exists(name, fmt):
                    format = fmt
                    break
            else:
                raise FileNotFoundError(f"No dataset found with name '{name}'")
        
        if format == "npz":
            filepath = self.base_path / "datasets" / f"{name}.npz"
            metadata_path = filepath.with_suffix(".json")
            
            if not filepath.exists():
                raise FileNotFoundError(f"Dataset file not found: {filepath}")
            
            # Load arrays
            data = jnp.load(filepath)
            X = data["X"]
            y = data["y"]
            gradients = data.get("gradients")
            
            # Load metadata
            metadata = {}
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
            
            return Dataset(X=X, y=y, gradients=gradients, metadata=metadata)
        
        elif format == "json":
            filepath = self.base_path / "datasets" / f"{name}.json"
            
            if not filepath.exists():
                raise FileNotFoundError(f"Dataset file not found: {filepath}")
            
            with open(filepath, "r") as f:
                data_dict = json.load(f)
            
            X = jnp.array(data_dict["X"])
            y = jnp.array(data_dict["y"])
            gradients = jnp.array(data_dict["gradients"]) if data_dict["gradients"] else None
            metadata = data_dict.get("metadata", {})
            
            return Dataset(X=X, y=y, gradients=gradients, metadata=metadata)
        
        elif format == "pickle":
            filepath = self.base_path / "datasets" / f"{name}.pkl"
            
            if not filepath.exists():
                raise FileNotFoundError(f"Dataset file not found: {filepath}")
            
            with open(filepath, "rb") as f:
                return pickle.load(f)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _dataset_exists(self, name: str, format: str) -> bool:
        """Check if dataset exists in given format."""
        extensions = {"npz": ".npz", "json": ".json", "pickle": ".pkl"}
        filepath = self.base_path / "datasets" / f"{name}{extensions[format]}"
        return filepath.exists()
    
    def list_datasets(self) -> List[str]:
        """List all available datasets.
        
        Returns:
            List of dataset names
        """
        datasets = set()
        dataset_dir = self.base_path / "datasets"
        
        for filepath in dataset_dir.glob("*"):
            if filepath.suffix in [".npz", ".json", ".pkl"]:
                datasets.add(filepath.stem)
        
        return sorted(list(datasets))
    
    def save_optimization_result(
        self,
        result: OptimizationResult,
        name: str,
        include_trajectory: bool = True,
    ) -> Path:
        """Save optimization result to disk.
        
        Args:
            result: Optimization result to save
            name: Name for the saved result
            include_trajectory: Whether to include optimization trajectory
            
        Returns:
            Path to saved file
        """
        filepath = self.base_path / "results" / f"{name}.json"
        
        # Convert result to serializable format
        result_dict = {
            "x": result.x.tolist(),
            "fun": float(result.fun),
            "success": result.success,
            "message": result.message,
            "nit": result.nit,
            "nfev": result.nfev,
            "metadata": result.metadata,
        }
        
        if include_trajectory and result.trajectory:
            result_dict["trajectory"] = [x.tolist() for x in result.trajectory]
        
        if result.convergence_history:
            result_dict["convergence_history"] = [float(x) for x in result.convergence_history]
        
        with open(filepath, "w") as f:
            json.dump(result_dict, f, indent=2)
        
        return filepath
    
    def load_optimization_result(self, name: str) -> OptimizationResult:
        """Load optimization result from disk.
        
        Args:
            name: Name of the result to load
            
        Returns:
            Loaded optimization result
        """
        filepath = self.base_path / "results" / f"{name}.json"
        
        if not filepath.exists():
            raise FileNotFoundError(f"Result file not found: {filepath}")
        
        with open(filepath, "r") as f:
            result_dict = json.load(f)
        
        # Convert back to OptimizationResult
        trajectory = None
        if "trajectory" in result_dict:
            trajectory = [jnp.array(x) for x in result_dict["trajectory"]]
        
        return OptimizationResult(
            x=jnp.array(result_dict["x"]),
            fun=result_dict["fun"],
            success=result_dict["success"],
            message=result_dict["message"],
            nit=result_dict["nit"],
            nfev=result_dict["nfev"],
            trajectory=trajectory,
            convergence_history=result_dict.get("convergence_history"),
            metadata=result_dict.get("metadata", {}),
        )
    
    def list_results(self) -> List[str]:
        """List all available optimization results.
        
        Returns:
            List of result names
        """
        results_dir = self.base_path / "results"
        return [f.stem for f in results_dir.glob("*.json")]
    
    def save_model(
        self,
        model: Any,
        name: str,
        format: str = "pickle",
    ) -> Path:
        """Save trained model to disk.
        
        Args:
            model: Trained model to save
            name: Name for the saved model
            format: File format ('pickle' only for now)
            
        Returns:
            Path to saved file
        """
        if format != "pickle":
            raise ValueError("Only pickle format is currently supported for models")
        
        filepath = self.base_path / "models" / f"{name}.pkl"
        
        with open(filepath, "wb") as f:
            pickle.dump(model, f)
        
        return filepath
    
    def load_model(self, name: str) -> Any:
        """Load trained model from disk.
        
        Args:
            name: Name of the model to load
            
        Returns:
            Loaded model
        """
        filepath = self.base_path / "models" / f"{name}.pkl"
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, "rb") as f:
            return pickle.load(f)
    
    def list_models(self) -> List[str]:
        """List all available saved models.
        
        Returns:
            List of model names
        """
        models_dir = self.base_path / "models"
        return [f.stem for f in models_dir.glob("*.pkl")]
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Get information about stored data.
        
        Returns:
            Dictionary with storage statistics
        """
        info = {
            "base_path": str(self.base_path),
            "datasets": {
                "count": len(self.list_datasets()),
                "names": self.list_datasets(),
            },
            "results": {
                "count": len(self.list_results()),
                "names": self.list_results(),
            },
            "models": {
                "count": len(self.list_models()),
                "names": self.list_models(),
            }
        }
        
        # Calculate total storage size
        total_size = 0
        for filepath in self.base_path.rglob("*"):
            if filepath.is_file():
                total_size += filepath.stat().st_size
        
        info["total_size_mb"] = total_size / (1024 * 1024)
        
        return info
    
    def cleanup(self, older_than_days: int = 30) -> Dict[str, int]:
        """Clean up old files.
        
        Args:
            older_than_days: Remove files older than this many days
            
        Returns:
            Dictionary with cleanup statistics
        """
        import time
        
        cutoff_time = time.time() - (older_than_days * 24 * 3600)
        
        removed_counts = {"datasets": 0, "results": 0, "models": 0}
        
        for category in ["datasets", "results", "models"]:
            category_dir = self.base_path / category
            
            for filepath in category_dir.iterdir():
                if filepath.is_file() and filepath.stat().st_mtime < cutoff_time:
                    filepath.unlink()
                    removed_counts[category] += 1
        
        return removed_counts


# Global persistence instance
_default_persistence = None


def get_default_persistence() -> DataPersistence:
    """Get default data persistence instance."""
    global _default_persistence
    if _default_persistence is None:
        _default_persistence = DataPersistence()
    return _default_persistence


# Convenience functions
def save_dataset(dataset: Dataset, name: str, **kwargs) -> Path:
    """Save dataset using default persistence."""
    return get_default_persistence().save_dataset(dataset, name, **kwargs)


def load_dataset(name: str, **kwargs) -> Dataset:
    """Load dataset using default persistence."""
    return get_default_persistence().load_dataset(name, **kwargs)


def save_result(result: OptimizationResult, name: str, **kwargs) -> Path:
    """Save optimization result using default persistence."""
    return get_default_persistence().save_optimization_result(result, name, **kwargs)


def load_result(name: str) -> OptimizationResult:
    """Load optimization result using default persistence."""
    return get_default_persistence().load_optimization_result(name)