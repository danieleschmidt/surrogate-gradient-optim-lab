# 🚀 Production Deployment Guide

## Surrogate Gradient Optimization Lab - Research-Grade Deployment

This guide provides comprehensive instructions for deploying the Surrogate Gradient Optimization Lab in production research environments.

## 📋 Prerequisites

### System Requirements
- **Python**: 3.9+ (3.10+ recommended for research workloads)
- **Memory**: Minimum 8GB RAM, 32GB+ recommended for large-scale research
- **CPU**: Multi-core processor, 8+ cores recommended
- **GPU** (Optional): NVIDIA GPU with CUDA support for accelerated computation
- **Storage**: 10GB+ for datasets and model artifacts

### Dependencies
```bash
# Core dependencies
pip install jax[cpu]>=0.4.0  # or jax[cuda] for GPU
pip install numpy>=1.21.0
pip install scipy>=1.7.0
pip install scikit-learn>=1.0.0
pip install matplotlib>=3.5.0
pip install pandas>=1.3.0
pip install pydantic>=2.0.0

# Development dependencies (optional)
pip install pytest>=7.0.0
pip install black>=23.0.0
pip install mypy>=1.0.0
```

## 🔧 Installation Options

### Option 1: Development Installation
```bash
git clone https://github.com/terragon-labs/surrogate-gradient-optim-lab.git
cd surrogate-gradient-optim-lab
pip install -e ".[all]"
```

### Option 2: Production Installation
```bash
pip install surrogate-gradient-optim-lab
```

### Option 3: Docker Deployment
```bash
docker pull terragonlabs/surrogate-optim:latest
docker run -it --gpus all -v /data:/workspace/data terragonlabs/surrogate-optim:latest
```

## ⚡ Performance Configuration

### High-Performance Computing (HPC)
```python
from surrogate_optim.performance.research_parallel import create_research_config

# Configure for HPC cluster
config = create_research_config(
    optimization_target="speed",
    available_memory_gb=64,
    enable_gpu=True,
    research_scale="xlarge"
)
```

### Memory-Constrained Environments
```python
config = create_research_config(
    optimization_target="memory", 
    available_memory_gb=8,
    enable_gpu=False,
    research_scale="small"
)
```

### Balanced Research Workloads
```python
config = create_research_config(
    optimization_target="balanced",
    available_memory_gb=32,
    enable_gpu=True,
    research_scale="large"
)
```

## 🔬 Research Execution Pipeline

### Autonomous Research Execution
```python
from surrogate_optim.research.research_execution_engine import (
    ResearchConfiguration, execute_autonomous_research
)

# Configure research parameters
config = ResearchConfiguration()
config.statistical_config['n_trials'] = 10
config.benchmark_functions = ['sphere_2d', 'rosenbrock_2d', 'rastrigin_2d']

# Execute full research pipeline
results = execute_autonomous_research(config, "research_output")
```

### Individual Research Components
```python
from surrogate_optim.research.novel_algorithms import (
    PhysicsInformedSurrogate,
    AdaptiveAcquisitionOptimizer,
    MultiObjectiveSurrogateOptimizer
)

# Physics-informed optimization
physics_surrogate = PhysicsInformedSurrogate(
    hidden_dims=[64, 64, 32],
    physics_weight=0.1
)

# Adaptive acquisition
adaptive_opt = AdaptiveAcquisitionOptimizer(
    adaptation_rate=0.1,
    exploration_schedule="decay"
)
```

## 📊 Monitoring and Observability

### Resource Monitoring
```python
from surrogate_optim.performance.research_parallel import ResourceMonitor

monitor = ResourceMonitor(config)
monitor.start_monitoring()

# Your optimization code here

metrics = monitor.stop_monitoring()
print(f"Throughput: {metrics.throughput:.1f} ops/sec")
print(f"Memory usage: {metrics.memory_mb:.1f} MB")
```

### Logging Configuration
```python
import logging
from surrogate_optim.monitoring.logging import get_logger

# Configure research-grade logging
logger = get_logger()
logger.setLevel(logging.INFO)

# Add file handler for production
handler = logging.FileHandler('surrogate_optim.log')
handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
logger.addHandler(handler)
```

## 🛡️ Security and Validation

### Input Validation
```python
from surrogate_optim.validation.input_validation import RobustValidator

validator = RobustValidator(strict_mode=True)

# Validate bounds
bounds_result = validator.validate_bounds(bounds, n_dims)
if not bounds_result.is_valid:
    print(f"Validation errors: {bounds_result.errors}")

# Validate dataset
dataset_result = validator.validate_dataset(dataset)
if not dataset_result.is_valid:
    print(f"Dataset issues: {dataset_result.warnings}")
```

### Security Recommendations
- **Data Isolation**: Use separate environments for different research projects
- **Access Control**: Implement proper authentication for multi-user deployments
- **Audit Logging**: Enable comprehensive logging for research reproducibility
- **Resource Limits**: Set memory and CPU limits to prevent resource exhaustion

## 🔄 CI/CD Integration

### GitHub Actions Workflow
```yaml
name: Surrogate Optimization CI
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
      - name: Run tests
        run: |
          pytest tests/ --cov=surrogate_optim
      - name: Validate research components
        run: |
          python -m surrogate_optim.research.research_execution_engine --validate
```

### Quality Gates
```bash
# Syntax validation
python -m py_compile surrogate_optim/**/*.py

# Type checking
mypy surrogate_optim/

# Code formatting
black --check surrogate_optim/
isort --check-only surrogate_optim/

# Security scanning
bandit -r surrogate_optim/

# Dependency vulnerability scanning
safety check
```

## 🌐 Deployment Architectures

### Single-Node Research Station
```
┌─────────────────┐
│ Research Station │
│ - 32GB RAM      │
│ - 8 CPU cores   │
│ - GPU (optional)│
│ - Local storage │
└─────────────────┘
```

### Distributed HPC Cluster
```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Head Node    │    │ Compute Node │    │ Compute Node │
│ - Scheduler  │────│ - Worker     │────│ - Worker     │
│ - Monitoring │    │ - GPU        │    │ - GPU        │
└──────────────┘    └──────────────┘    └──────────────┘
         │
┌──────────────┐
│ Shared       │
│ Storage      │
│ - Datasets   │
│ - Results    │
└──────────────┘
```

### Cloud Deployment (AWS/Azure/GCP)
```python
# Example Kubernetes deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: surrogate-optim
spec:
  replicas: 3
  selector:
    matchLabels:
      app: surrogate-optim
  template:
    metadata:
      labels:
        app: surrogate-optim
    spec:
      containers:
      - name: surrogate-optim
        image: terragonlabs/surrogate-optim:latest
        resources:
          requests:
            memory: "8Gi"
            cpu: "2"
          limits:
            memory: "32Gi"
            cpu: "8"
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
```

## 🔧 Configuration Management

### Environment Variables
```bash
# Performance settings
export SURROGATE_OPTIM_WORKERS=8
export SURROGATE_OPTIM_MEMORY_LIMIT=32GB
export SURROGATE_OPTIM_GPU_ENABLED=true

# Research settings
export SURROGATE_OPTIM_CACHE_DIR="/data/cache"
export SURROGATE_OPTIM_OUTPUT_DIR="/data/results"
export SURROGATE_OPTIM_LOG_LEVEL=INFO

# JAX settings
export JAX_ENABLE_X64=true
export JAX_PLATFORM_NAME=gpu  # or cpu
```

### Configuration File (surrogate_optim.yaml)
```yaml
performance:
  max_workers: 8
  memory_limit_gb: 32
  enable_gpu: true
  optimization_level: "balanced"

research:
  statistical_trials: 10
  significance_level: 0.05
  benchmark_functions:
    - "sphere_2d"
    - "rosenbrock_2d" 
    - "rastrigin_2d"

output:
  cache_results: true
  generate_plots: true
  save_raw_results: true

logging:
  level: "INFO"
  file: "surrogate_optim.log"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

## 📈 Scaling Guidelines

### Memory Scaling
- **Small datasets** (< 1K samples): 4-8GB RAM
- **Medium datasets** (1K-10K samples): 16-32GB RAM  
- **Large datasets** (10K-100K samples): 64-128GB RAM
- **Research ensembles**: 128GB+ RAM recommended

### CPU Scaling
- **Development**: 2-4 cores
- **Production research**: 8-16 cores
- **High-throughput**: 32+ cores
- **Distributed**: Scale horizontally with cluster

### GPU Acceleration
- **Single GPU**: RTX 3080/4080, A100, V100
- **Multi-GPU**: Scale across multiple devices
- **Memory**: 16GB+ VRAM for large models
- **Compute**: FP32/FP16 precision as needed

## 🐛 Troubleshooting

### Common Issues

#### Out of Memory Errors
```python
# Reduce batch size
config.chunk_size = 50

# Enable memory-efficient processing  
config.optimization_level = "memory"

# Use memory monitoring
monitor = ResourceMonitor(config)
```

#### GPU Issues
```bash
# Check GPU availability
python -c "import jax; print(jax.devices())"

# Force CPU execution
export JAX_PLATFORM_NAME=cpu
```

#### Performance Issues
```python
# Enable JIT compilation
from jax import jit

# Use vectorized operations
from jax import vmap

# Profile performance
from surrogate_optim.performance.research_parallel import ResourceMonitor
```

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable JAX debugging
import jax
jax.config.update("jax_debug_nans", True)
jax.config.update("jax_log_compiles", True)
```

## 📞 Support and Maintenance

### Health Checks
```python
from surrogate_optim.health.checks import run_health_checks

# Run comprehensive health check
health_status = run_health_checks()
if not health_status.all_passed:
    print("Health check failures:", health_status.failures)
```

### Monitoring Endpoints
- Health: `GET /health`
- Metrics: `GET /metrics` 
- Status: `GET /status`
- Version: `GET /version`

### Backup and Recovery
```bash
# Backup research data
tar -czf research_backup_$(date +%Y%m%d).tar.gz \
    research_output/ \
    cache/ \
    logs/

# Backup configuration
cp surrogate_optim.yaml config_backup_$(date +%Y%m%d).yaml
```

## 🎯 Performance Benchmarks

### Expected Performance (Single Node)
- **Function evaluations**: 1K-10K/sec (CPU), 10K-100K/sec (GPU)
- **Surrogate training**: 1-60 seconds (depending on size)
- **Optimization convergence**: 10-1000 iterations
- **Memory usage**: 100MB-10GB (depending on dataset)

### Scaling Metrics
- **Linear scaling**: Up to 8-16 cores
- **Memory efficiency**: 80%+ utilization
- **GPU utilization**: 70%+ for large workloads
- **Cache hit ratio**: 60%+ in research workflows

---

## 🏁 Deployment Checklist

- [ ] System requirements met
- [ ] Dependencies installed  
- [ ] Configuration validated
- [ ] Security measures implemented
- [ ] Monitoring configured
- [ ] Health checks passing
- [ ] Performance benchmarks validated
- [ ] Backup procedures established
- [ ] Documentation reviewed
- [ ] Team training completed

For additional support, consult the [API Reference](docs/api-reference.md) and [User Guide](docs/user-guide.md).

**Happy Optimizing! 🚀**