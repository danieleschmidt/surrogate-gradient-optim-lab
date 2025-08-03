"""Database migrations and schema management."""

from .connection import get_db_manager


def create_tables():
    """Create all database tables."""
    db = get_db_manager()
    
    # Experiments table
    experiments_table = """
    CREATE TABLE IF NOT EXISTS experiments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        description TEXT,
        status TEXT NOT NULL DEFAULT 'created',
        config TEXT,
        created_at TEXT,
        updated_at TEXT,
        completed_at TEXT,
        best_value REAL,
        best_point TEXT,
        n_evaluations INTEGER DEFAULT 0,
        tags TEXT,
        metadata TEXT,
        UNIQUE(name)
    );
    """
    
    # Datasets table
    datasets_table = """
    CREATE TABLE IF NOT EXISTS datasets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        experiment_id INTEGER,
        name TEXT NOT NULL,
        description TEXT,
        X TEXT,
        y TEXT,
        gradients TEXT,
        n_samples INTEGER NOT NULL,
        n_dims INTEGER NOT NULL,
        bounds TEXT,
        sampling_strategy TEXT,
        created_at TEXT,
        metadata TEXT,
        FOREIGN KEY (experiment_id) REFERENCES experiments (id)
    );
    """
    
    # Surrogates table
    surrogates_table = """
    CREATE TABLE IF NOT EXISTS surrogates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        experiment_id INTEGER,
        dataset_id INTEGER,
        name TEXT NOT NULL,
        surrogate_type TEXT NOT NULL,
        config TEXT,
        training_loss REAL,
        validation_score REAL,
        training_time REAL,
        model_data BLOB,
        created_at TEXT,
        trained_at TEXT,
        metrics TEXT,
        FOREIGN KEY (experiment_id) REFERENCES experiments (id),
        FOREIGN KEY (dataset_id) REFERENCES datasets (id)
    );
    """
    
    # Optimization results table
    optimization_results_table = """
    CREATE TABLE IF NOT EXISTS optimization_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        experiment_id INTEGER,
        surrogate_id INTEGER,
        method TEXT NOT NULL,
        initial_point TEXT,
        bounds TEXT,
        optimal_point TEXT,
        optimal_value REAL,
        success BOOLEAN,
        message TEXT,
        n_iterations INTEGER DEFAULT 0,
        n_function_evaluations INTEGER DEFAULT 0,
        n_gradient_evaluations INTEGER DEFAULT 0,
        optimization_time REAL,
        trajectory TEXT,
        started_at TEXT,
        completed_at TEXT,
        metadata TEXT,
        FOREIGN KEY (experiment_id) REFERENCES experiments (id),
        FOREIGN KEY (surrogate_id) REFERENCES surrogates (id)
    );
    """
    
    # Create indices for better performance
    indices = [
        "CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);",
        "CREATE INDEX IF NOT EXISTS idx_experiments_created_at ON experiments(created_at);",
        "CREATE INDEX IF NOT EXISTS idx_datasets_experiment_id ON datasets(experiment_id);",
        "CREATE INDEX IF NOT EXISTS idx_surrogates_experiment_id ON surrogates(experiment_id);",
        "CREATE INDEX IF NOT EXISTS idx_surrogates_dataset_id ON surrogates(dataset_id);",
        "CREATE INDEX IF NOT EXISTS idx_optimization_results_experiment_id ON optimization_results(experiment_id);",
        "CREATE INDEX IF NOT EXISTS idx_optimization_results_surrogate_id ON optimization_results(surrogate_id);",
    ]
    
    # Execute all creation statements
    db.execute_script(experiments_table)
    db.execute_script(datasets_table)
    db.execute_script(surrogates_table)
    db.execute_script(optimization_results_table)
    
    for index in indices:
        db.execute_script(index)
    
    print("Database tables created successfully.")


def run_migrations():
    """Run all pending migrations."""
    # For now, just create tables
    # In a real application, you would track migration versions
    create_tables()
    

def drop_all_tables():
    """Drop all tables (for testing/development)."""
    db = get_db_manager()
    
    drop_statements = [
        "DROP TABLE IF EXISTS optimization_results;",
        "DROP TABLE IF EXISTS surrogates;",
        "DROP TABLE IF EXISTS datasets;",
        "DROP TABLE IF EXISTS experiments;",
    ]
    
    for statement in drop_statements:
        db.execute_script(statement)
    
    print("All tables dropped.")


def reset_database():
    """Reset database by dropping and recreating tables."""
    drop_all_tables()
    create_tables()
    print("Database reset completed.")


def get_table_info():
    """Get information about database tables."""
    db = get_db_manager()
    
    tables = ["experiments", "datasets", "surrogates", "optimization_results"]
    info = {}
    
    for table in tables:
        try:
            # Get table schema
            schema_query = f"PRAGMA table_info({table});"
            schema = db.execute_query(schema_query)
            
            # Get row count
            count_query = f"SELECT COUNT(*) as count FROM {table};"
            count_result = db.execute_query(count_query)
            count = count_result[0]["count"] if count_result else 0
            
            info[table] = {
                "schema": schema,
                "row_count": count,
            }
        except Exception as e:
            info[table] = {"error": str(e)}
    
    return info


def backup_database(backup_path: str):
    """Backup database to file."""
    import shutil
    from pathlib import Path
    
    db = get_db_manager()
    
    # Get current database path
    if db.config.url.startswith("sqlite"):
        db_path = db.config.url.replace("sqlite:///", "")
        
        # Create backup directory
        backup_dir = Path(backup_path).parent
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy database file
        shutil.copy2(db_path, backup_path)
        print(f"Database backed up to: {backup_path}")
    else:
        raise NotImplementedError("Backup only supported for SQLite databases")


def restore_database(backup_path: str):
    """Restore database from backup file."""
    import shutil
    from pathlib import Path
    
    db = get_db_manager()
    
    if db.config.url.startswith("sqlite"):
        db_path = db.config.url.replace("sqlite:///", "")
        
        if not Path(backup_path).exists():
            raise FileNotFoundError(f"Backup file not found: {backup_path}")
        
        # Close current connection
        db.close()
        
        # Copy backup file
        shutil.copy2(backup_path, db_path)
        
        # Reinitialize connection
        db._setup_sqlite()
        
        print(f"Database restored from: {backup_path}")
    else:
        raise NotImplementedError("Restore only supported for SQLite databases")


if __name__ == "__main__":
    """Command-line interface for migrations."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python migrations.py [create|migrate|reset|info|backup|restore]")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "create":
        create_tables()
    elif command == "migrate":
        run_migrations()
    elif command == "reset":
        reset_database()
    elif command == "info":
        info = get_table_info()
        for table, details in info.items():
            print(f"\nTable: {table}")
            if "error" in details:
                print(f"  Error: {details['error']}")
            else:
                print(f"  Rows: {details['row_count']}")
                print(f"  Columns: {len(details['schema'])}")
    elif command == "backup":
        if len(sys.argv) < 3:
            print("Usage: python migrations.py backup <backup_path>")
            sys.exit(1)
        backup_database(sys.argv[2])
    elif command == "restore":
        if len(sys.argv) < 3:
            print("Usage: python migrations.py restore <backup_path>")
            sys.exit(1)
        restore_database(sys.argv[2])
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)