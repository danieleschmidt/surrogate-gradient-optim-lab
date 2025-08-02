#!/bin/bash

# Post-create script for Surrogate Gradient Optimization Lab development environment
# This script runs after the devcontainer is created to set up the development environment

echo "🚀 Setting up Surrogate Gradient Optimization Lab development environment..."

# Update package lists
echo "📦 Updating package lists..."
sudo apt-get update

# Install system dependencies
echo "🔧 Installing system dependencies..."
sudo apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    graphviz \
    graphviz-dev \
    pkg-config \
    libhdf5-dev \
    libblas-dev \
    liblapack-dev \
    gfortran

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install --upgrade pip setuptools wheel

# Install the package in development mode with all extras
echo "📚 Installing project in development mode..."
pip install -e ".[dev,docs,notebook,cuda]" 2>/dev/null || pip install -e ".[dev,docs,notebook]"

# Install pre-commit hooks
echo "🪝 Setting up pre-commit hooks..."
pre-commit install --install-hooks

# Install additional development tools
echo "🛠️ Installing additional development tools..."
pip install \
    jupyterlab \
    jupyter-book \
    plotly \
    dash \
    streamlit \
    ipywidgets \
    voila

# Set up Jupyter Lab extensions
echo "🔬 Setting up Jupyter Lab..."
jupyter lab --generate-config
echo "c.ServerApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_lab_config.py
echo "c.ServerApp.allow_root = True" >> ~/.jupyter/jupyter_lab_config.py
echo "c.ServerApp.open_browser = False" >> ~/.jupyter/jupyter_lab_config.py

# Create useful aliases
echo "🔗 Setting up aliases..."
cat >> ~/.bashrc << 'EOF'

# Surrogate Optimization Lab aliases
alias sgol-test='python -m pytest tests/ -v'
alias sgol-coverage='python -m pytest tests/ --cov=surrogate_optim --cov-report=html'
alias sgol-lint='ruff check . && mypy surrogate_optim/'
alias sgol-format='black . && isort .'
alias sgol-docs='jupyter-book build docs/'
alias sgol-clean='find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; find . -name "*.pyc" -delete'
alias sgol-benchmark='python -m pytest tests/benchmarks/ -v --benchmark-only'
alias sgol-lab='jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root'
alias sgol-dash='python scripts/launch_dashboard.py'

# Git shortcuts
alias gs='git status'
alias ga='git add'
alias gc='git commit'
alias gp='git push'
alias gl='git log --oneline -10'
alias gd='git diff'

# Development shortcuts
alias ll='ls -la'
alias la='ls -A'
alias l='ls -CF'

EOF

# Create development directories
echo "📁 Creating development directories..."
mkdir -p ~/.local/share/jupyter/kernels
mkdir -p ~/notebooks
mkdir -p ~/experiments
mkdir -p ~/data

# Set up git configuration (if not already configured)
if ! git config --global user.name > /dev/null 2>&1; then
    echo "⚙️ Git user not configured. Please run:"
    echo "git config --global user.name 'Your Name'"
    echo "git config --global user.email 'your.email@example.com'"
fi

# Create sample notebook
echo "📓 Creating sample notebook..."
cat > ~/notebooks/getting_started.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Getting Started with Surrogate Gradient Optimization Lab\n",
    "\n",
    "Welcome to the development environment for the Surrogate Gradient Optimization Lab!\n",
    "\n",
    "## Quick Start\n",
    "\n",
    "This notebook will help you get started with the basic functionality."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import the main modules\n",
    "import jax.numpy as jnp\n",
    "from surrogate_optim import SurrogateOptimizer\n",
    "\n",
    "print(\"🎉 Surrogate Optimization Lab is ready!\")\n",
    "print(\"📚 Check out the examples in the repository for more details.\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.11.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

# Set up environment variables
echo "🌍 Setting up environment variables..."
cat >> ~/.bashrc << 'EOF'

# Surrogate Optimization Lab environment
export SGOL_HOME="/workspaces/surrogate-gradient-optim-lab"
export PYTHONPATH="${SGOL_HOME}:${PYTHONPATH}"
export JUPYTER_CONFIG_DIR="~/.jupyter"

EOF

# Install JAX with appropriate backend
echo "🧮 Installing JAX..."
pip install --upgrade "jax[cpu]" 2>/dev/null || echo "⚠️ JAX installation failed, will use CPU-only version"

# Run initial tests to verify setup
echo "🧪 Running initial tests..."
if python -c "import surrogate_optim; print('✅ Package import successful')"; then
    echo "✅ Development environment setup complete!"
else
    echo "⚠️ Package import failed. Please check the installation."
fi

# Print success message
echo ""
echo "🎉 Surrogate Gradient Optimization Lab development environment is ready!"
echo ""
echo "📋 Available commands:"
echo "  sgol-test      - Run tests"
echo "  sgol-coverage  - Run tests with coverage"
echo "  sgol-lint      - Run linting"
echo "  sgol-format    - Format code"
echo "  sgol-docs      - Build documentation"
echo "  sgol-benchmark - Run benchmarks"
echo "  sgol-lab       - Launch Jupyter Lab"
echo "  sgol-clean     - Clean cache files"
echo ""
echo "📚 To get started:"
echo "  1. Open ~/notebooks/getting_started.ipynb"
echo "  2. Run 'sgol-test' to verify everything works"
echo "  3. Run 'sgol-lab' to start Jupyter Lab"
echo ""
echo "Happy optimizing! 🚀"