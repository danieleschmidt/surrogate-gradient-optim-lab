# Development container for Surrogate Gradient Optimization Lab
FROM mcr.microsoft.com/devcontainers/python:1-3.9-bullseye

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        build-essential \
        curl \
        git \
        graphviz \
        graphviz-dev \
        libblas-dev \
        liblapack-dev \
        libhdf5-dev \
        pkg-config \
        software-properties-common \
        vim \
        wget \
        zip \
        unzip \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install Python development tools
RUN pip install --upgrade pip setuptools wheel

# Install common ML/scientific computing packages
RUN pip install \
    numpy \
    scipy \
    matplotlib \
    pandas \
    scikit-learn \
    jupyter \
    jupyterlab \
    notebook \
    ipywidgets \
    plotly \
    seaborn

# Install JAX with CPU support (GPU support would require CUDA)
RUN pip install jax jaxlib

# Install development and testing tools
RUN pip install \
    pytest \
    pytest-cov \
    pytest-xdist \
    pytest-mock \
    black \
    isort \
    flake8 \
    mypy \
    ruff \
    pre-commit \
    sphinx \
    sphinx-rtd-theme \
    sphinx-autodoc-typehints \
    coverage \
    bandit \
    safety

# Install additional useful tools
RUN pip install \
    rich \
    typer \
    click \
    tqdm \
    python-dotenv \
    pydantic \
    loguru

# Create workspace directory
RUN mkdir -p /workspace
WORKDIR /workspace

# Set up git configuration for the container
RUN git config --global --add safe.directory /workspace

# Create a non-root user
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Configure shell for better development experience
RUN echo 'alias ll="ls -la"' >> /home/$USERNAME/.bashrc \
    && echo 'alias la="ls -la"' >> /home/$USERNAME/.bashrc \
    && echo 'alias l="ls -CF"' >> /home/$USERNAME/.bashrc \
    && echo 'export PYTHONPATH="/workspace:$PYTHONPATH"' >> /home/$USERNAME/.bashrc

# Expose ports for development servers
EXPOSE 8000 8080 8888 6006

# Set the default command
CMD ["/bin/bash"]