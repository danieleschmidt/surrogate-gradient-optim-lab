"""Database connection management."""

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Generator
import threading
from dataclasses import dataclass

import numpy as np


@dataclass
class DatabaseConfig:
    """Database configuration."""
    url: str = "sqlite:///./surrogate_optim.db"
    pool_size: int = 10
    max_overflow: int = 20
    echo: bool = False
    
    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Create config from environment variables."""
        return cls(
            url=os.getenv("DATABASE_URL", "sqlite:///./surrogate_optim.db"),
            pool_size=int(os.getenv("DB_POOL_SIZE", "10")),
            max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "20")),
            echo=os.getenv("DB_ECHO", "false").lower() == "true",
        )


class DatabaseManager:
    """Database connection and session management."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern for database manager."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize database manager."""
        if not hasattr(self, 'initialized'):
            self.config = DatabaseConfig.from_env()
            self.connection = None
            self._setup_sqlite()
            self.initialized = True
    
    def _setup_sqlite(self):
        """Setup SQLite database."""
        if self.config.url.startswith("sqlite"):
            # Extract path from URL
            db_path = self.config.url.replace("sqlite:///", "")
            db_dir = Path(db_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)
            
            # Create connection
            self.connection = sqlite3.connect(db_path, check_same_thread=False)
            self.connection.row_factory = sqlite3.Row
            
            # Enable foreign keys
            self.connection.execute("PRAGMA foreign_keys = ON")
            
            # Enable WAL mode for better concurrency
            self.connection.execute("PRAGMA journal_mode = WAL")
            
            # Set reasonable timeout
            self.connection.execute("PRAGMA busy_timeout = 30000")
    
    @contextmanager
    def get_session(self) -> Generator[sqlite3.Connection, None, None]:
        """Get database session context manager."""
        if self.connection is None:
            raise RuntimeError("Database not initialized")
        
        try:
            yield self.connection
        except Exception as e:
            self.connection.rollback()
            raise e
    
    def execute_script(self, script: str) -> None:
        """Execute SQL script."""
        with self.get_session() as session:
            session.executescript(script)
            session.commit()
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> list:
        """Execute query and return results."""
        with self.get_session() as session:
            cursor = session.execute(query, params or ())
            return cursor.fetchall()
    
    def execute_insert(self, query: str, params: Optional[tuple] = None) -> int:
        """Execute insert and return last row ID."""
        with self.get_session() as session:
            cursor = session.execute(query, params or ())
            session.commit()
            return cursor.lastrowid
    
    def execute_update(self, query: str, params: Optional[tuple] = None) -> int:
        """Execute update and return affected rows."""
        with self.get_session() as session:
            cursor = session.execute(query, params or ())
            session.commit()
            return cursor.rowcount
    
    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None


# Global instance
_db_manager = None


def get_db_manager() -> DatabaseManager:
    """Get database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


def get_connection():
    """Get database connection context manager."""
    return get_db_manager().get_session()


def serialize_array(arr: np.ndarray) -> bytes:
    """Serialize numpy array to bytes."""
    return arr.tobytes()


def deserialize_array(data: bytes, shape: tuple, dtype: str = "float64") -> np.ndarray:
    """Deserialize bytes to numpy array."""
    return np.frombuffer(data, dtype=dtype).reshape(shape)


def serialize_json(obj) -> str:
    """Serialize object to JSON string."""
    import json
    
    def json_serializer(obj):
        if isinstance(obj, np.ndarray):
            return {
                "__type__": "ndarray",
                "data": obj.tolist(),
                "shape": obj.shape,
                "dtype": str(obj.dtype),
            }
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    return json.dumps(obj, default=json_serializer)


def deserialize_json(json_str: str):
    """Deserialize JSON string to object."""
    import json
    
    def json_deserializer(obj):
        if isinstance(obj, dict) and obj.get("__type__") == "ndarray":
            return np.array(obj["data"], dtype=obj["dtype"]).reshape(obj["shape"])
        return obj
    
    data = json.loads(json_str)
    return _convert_json_arrays(data)


def _convert_json_arrays(obj):
    """Recursively convert JSON arrays back to numpy arrays."""
    if isinstance(obj, dict):
        if obj.get("__type__") == "ndarray":
            return np.array(obj["data"], dtype=obj["dtype"]).reshape(obj["shape"])
        return {k: _convert_json_arrays(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_json_arrays(item) for item in obj]
    return obj