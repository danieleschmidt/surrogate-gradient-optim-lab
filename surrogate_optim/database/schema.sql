-- Database schema for Surrogate Gradient Optimization Lab
-- SQLite Schema Definition

-- Enable foreign key constraints
PRAGMA foreign_keys = ON;

-- Experiments table
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    status TEXT NOT NULL DEFAULT 'created' CHECK (status IN ('created', 'running', 'completed', 'failed', 'cancelled')),
    config TEXT, -- JSON configuration
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT,
    best_value REAL,
    best_point TEXT, -- JSON serialized numpy array
    n_evaluations INTEGER DEFAULT 0,
    tags TEXT, -- JSON array of strings
    metadata TEXT -- JSON metadata
);

-- Datasets table
CREATE TABLE IF NOT EXISTS datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER,
    name TEXT NOT NULL,
    description TEXT,
    X TEXT NOT NULL, -- JSON serialized input data
    y TEXT NOT NULL, -- JSON serialized target data
    gradients TEXT, -- JSON serialized gradient data (optional)
    n_samples INTEGER NOT NULL,
    n_dims INTEGER NOT NULL,
    bounds TEXT, -- JSON serialized bounds
    sampling_strategy TEXT,
    created_at TEXT NOT NULL,
    metadata TEXT, -- JSON metadata
    FOREIGN KEY (experiment_id) REFERENCES experiments (id) ON DELETE CASCADE
);

-- Surrogate models table
CREATE TABLE IF NOT EXISTS surrogates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER,
    dataset_id INTEGER,
    name TEXT NOT NULL,
    surrogate_type TEXT NOT NULL CHECK (surrogate_type IN ('neural_network', 'gaussian_process', 'random_forest', 'hybrid')),
    config TEXT, -- JSON configuration
    training_loss REAL,
    validation_score REAL,
    training_time REAL,
    model_data BLOB, -- Serialized model parameters
    created_at TEXT NOT NULL,
    trained_at TEXT,
    metrics TEXT, -- JSON performance metrics
    FOREIGN KEY (experiment_id) REFERENCES experiments (id) ON DELETE CASCADE,
    FOREIGN KEY (dataset_id) REFERENCES datasets (id) ON DELETE CASCADE
);

-- Optimization results table
CREATE TABLE IF NOT EXISTS optimization_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER,
    surrogate_id INTEGER,
    method TEXT NOT NULL,
    initial_point TEXT, -- JSON serialized initial point
    bounds TEXT, -- JSON serialized bounds
    optimal_point TEXT, -- JSON serialized optimal point
    optimal_value REAL,
    success BOOLEAN NOT NULL DEFAULT 0,
    message TEXT,
    n_iterations INTEGER DEFAULT 0,
    n_function_evaluations INTEGER DEFAULT 0,
    n_gradient_evaluations INTEGER DEFAULT 0,
    optimization_time REAL,
    trajectory TEXT, -- JSON serialized optimization trajectory
    started_at TEXT,
    completed_at TEXT,
    metadata TEXT, -- JSON metadata
    FOREIGN KEY (experiment_id) REFERENCES experiments (id) ON DELETE CASCADE,
    FOREIGN KEY (surrogate_id) REFERENCES surrogates (id) ON DELETE CASCADE
);

-- Performance indices
CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);
CREATE INDEX IF NOT EXISTS idx_experiments_created_at ON experiments(created_at);
CREATE INDEX IF NOT EXISTS idx_experiments_name ON experiments(name);

CREATE INDEX IF NOT EXISTS idx_datasets_experiment_id ON datasets(experiment_id);
CREATE INDEX IF NOT EXISTS idx_datasets_created_at ON datasets(created_at);
CREATE INDEX IF NOT EXISTS idx_datasets_n_samples ON datasets(n_samples);

CREATE INDEX IF NOT EXISTS idx_surrogates_experiment_id ON surrogates(experiment_id);
CREATE INDEX IF NOT EXISTS idx_surrogates_dataset_id ON surrogates(dataset_id);
CREATE INDEX IF NOT EXISTS idx_surrogates_type ON surrogates(surrogate_type);
CREATE INDEX IF NOT EXISTS idx_surrogates_created_at ON surrogates(created_at);

CREATE INDEX IF NOT EXISTS idx_optimization_results_experiment_id ON optimization_results(experiment_id);
CREATE INDEX IF NOT EXISTS idx_optimization_results_surrogate_id ON optimization_results(surrogate_id);
CREATE INDEX IF NOT EXISTS idx_optimization_results_success ON optimization_results(success);
CREATE INDEX IF NOT EXISTS idx_optimization_results_completed_at ON optimization_results(completed_at);
CREATE INDEX IF NOT EXISTS idx_optimization_results_optimal_value ON optimization_results(optimal_value);

-- Views for common queries
CREATE VIEW IF NOT EXISTS experiment_summary AS
SELECT 
    e.id,
    e.name,
    e.description,
    e.status,
    e.created_at,
    e.completed_at,
    e.best_value,
    e.n_evaluations,
    COUNT(DISTINCT d.id) as n_datasets,
    COUNT(DISTINCT s.id) as n_surrogates,
    COUNT(DISTINCT or.id) as n_optimization_runs,
    COUNT(CASE WHEN or.success = 1 THEN 1 END) as n_successful_runs
FROM experiments e
LEFT JOIN datasets d ON e.id = d.experiment_id
LEFT JOIN surrogates s ON e.id = s.experiment_id
LEFT JOIN optimization_results or ON e.id = or.experiment_id
GROUP BY e.id;

CREATE VIEW IF NOT EXISTS surrogate_performance AS
SELECT 
    s.id,
    s.name,
    s.surrogate_type,
    s.training_loss,
    s.validation_score,
    s.training_time,
    s.created_at,
    s.trained_at,
    COUNT(or.id) as n_optimization_runs,
    COUNT(CASE WHEN or.success = 1 THEN 1 END) as n_successful_runs,
    AVG(or.optimal_value) as avg_optimal_value,
    MAX(or.optimal_value) as best_optimal_value,
    AVG(or.optimization_time) as avg_optimization_time
FROM surrogates s
LEFT JOIN optimization_results or ON s.id = or.surrogate_id
GROUP BY s.id;

-- Triggers for maintaining data consistency
CREATE TRIGGER IF NOT EXISTS update_experiment_timestamp
    AFTER UPDATE ON experiments
    FOR EACH ROW
BEGIN
    UPDATE experiments SET updated_at = datetime('now') WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS update_experiment_best_value
    AFTER INSERT ON optimization_results
    FOR EACH ROW
    WHEN NEW.success = 1 AND NEW.optimal_value IS NOT NULL
BEGIN
    UPDATE experiments 
    SET 
        best_value = NEW.optimal_value,
        n_evaluations = n_evaluations + NEW.n_function_evaluations
    WHERE id = NEW.experiment_id 
    AND (best_value IS NULL OR NEW.optimal_value > best_value);
END;

-- Function to calculate dataset statistics (via application layer)
-- SQLite doesn't support user-defined functions, so these would be calculated in Python

-- Cleanup old data (to be run periodically)
-- DELETE FROM experiments WHERE status = 'completed' AND created_at < datetime('now', '-30 days');

-- Vacuum command to reclaim space (run manually or via maintenance script)
-- VACUUM;

-- Analyze command to update query planner statistics (run periodically)
-- ANALYZE;