# Multi-stage Dockerfile for Surrogate Gradient Optimization Lab

# =============================================================================
# Base Stage - Common dependencies
# =============================================================================
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Create non-root user
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash appuser

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libblas-dev \
    liblapack-dev \
    libhdf5-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Dependencies Stage - Install Python dependencies
# =============================================================================
FROM base as dependencies

# Copy dependency files
COPY pyproject.toml /tmp/
COPY requirements*.txt /tmp/ 2>/dev/null || true

# Install base dependencies
RUN pip install --upgrade pip setuptools wheel

# Install Python dependencies
WORKDIR /tmp
RUN pip install . || pip install \
    jax[cpu] \
    numpy \
    scipy \
    scikit-learn \
    matplotlib \
    plotly \
    pandas \
    tqdm \
    pydantic \
    typer \
    rich \
    loguru \
    python-dotenv

# =============================================================================
# Development Stage - For development and testing
# =============================================================================
FROM dependencies as development

# Install development dependencies
RUN pip install \
    pytest \
    pytest-cov \
    pytest-xdist \
    pytest-mock \
    pytest-benchmark \
    black \
    isort \
    flake8 \
    mypy \
    ruff \
    pre-commit \
    bandit \
    safety \
    coverage \
    sphinx \
    sphinx-rtd-theme \
    jupyterlab \
    notebook

# Set up development environment
WORKDIR /workspace
RUN chown -R appuser:appuser /workspace
USER appuser

# Expose ports for development servers
EXPOSE 8000 8080 8888 6006

# Default command for development
CMD ["/bin/bash"]

# =============================================================================
# Production Stage - Optimized for production deployment
# =============================================================================
FROM base as production

# Copy installed packages from dependencies stage
COPY --from=dependencies /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Create application directory
WORKDIR /app
RUN chown -R appuser:appuser /app

# Copy application code
COPY --chown=appuser:appuser . /app/

# Remove development files
RUN rm -rf tests/ docs/ examples/ .git/ .github/ \
    .pytest_cache/ .mypy_cache/ .ruff_cache/ \
    *.egg-info/ build/ dist/

# Install the package
RUN pip install --no-deps -e .

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import surrogate_optim; print('OK')" || exit 1

# Expose application port
EXPOSE 8000

# Production command
CMD ["python", "-m", "surrogate_optim.cli", "--help"]

# =============================================================================
# GPU Stage - For GPU-accelerated workloads
# =============================================================================
FROM nvidia/cuda:12.9.1-devel-ubuntu20.04 as gpu

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install Python and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.9 \
    python3.9-dev \
    python3.9-distutils \
    python3-pip \
    build-essential \
    curl \
    git \
    libblas-dev \
    liblapack-dev \
    libhdf5-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3.9 /usr/bin/python

# Create non-root user
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash appuser

# Install Python dependencies with GPU support
RUN pip install --upgrade pip setuptools wheel
RUN pip install "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Copy dependency files and install
COPY pyproject.toml /tmp/
WORKDIR /tmp
RUN pip install .[gpu] || pip install \
    numpy \
    scipy \
    scikit-learn \
    matplotlib \
    plotly \
    pandas \
    tqdm \
    pydantic \
    typer \
    rich \
    loguru \
    python-dotenv

# Set up application
WORKDIR /app
RUN chown -R appuser:appuser /app

# Copy application code
COPY --chown=appuser:appuser . /app/
RUN pip install --no-deps -e .

# Switch to non-root user
USER appuser

# GPU health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import jax; print(f'JAX devices: {jax.devices()}'); import surrogate_optim; print('OK')" || exit 1

# Expose application port
EXPOSE 8000

# GPU command
CMD ["python", "-m", "surrogate_optim.cli", "--help"]

# =============================================================================
# Metadata
# =============================================================================
LABEL maintainer="Terragon Labs <team@terragon-labs.com>"
LABEL description="Surrogate Gradient Optimization Lab - Toolkit for offline black-box optimization"
LABEL version="0.1.0"
LABEL org.opencontainers.image.source="https://github.com/terragon-labs/surrogate-gradient-optim-lab"
LABEL org.opencontainers.image.documentation="https://docs.terragon-labs.com/surrogate-optim"
LABEL org.opencontainers.image.licenses="MIT"