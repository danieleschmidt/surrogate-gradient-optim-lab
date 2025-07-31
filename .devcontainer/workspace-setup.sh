#!/bin/bash
# Development environment setup script for devcontainer

set -e

echo "🚀 Setting up development environment..."

# Install additional system dependencies
sudo apt-get update
sudo apt-get install -y \
    graphviz \
    graphviz-dev \
    pkg-config \
    build-essential

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -e ".[dev,docs,notebook]"

# Setup pre-commit hooks
echo "🪝 Setting up pre-commit hooks..."
pre-commit install
pre-commit install --hook-type commit-msg

# Setup git configuration (if not already set)
if [ -z "$(git config --global user.name)" ]; then
    echo "⚙️  Setting up git configuration..."
    git config --global user.name "Developer"
    git config --global user.email "developer@example.com"
fi

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data outputs logs cache checkpoints models results experiments

# Install additional ML tools
echo "🧠 Installing additional ML tools..."
pip install \
    jupyter-lab-git \
    jupyterlab-code-formatter \
    jupyterlab-lsp

# Setup Jupyter extensions
echo "🔧 Configuring Jupyter..."
jupyter lab build

echo "✅ Development environment setup complete!"
echo "🎯 You can now run: make test, make lint, make docs"
echo "📊 Start Jupyter Lab with: jupyter lab --ip=0.0.0.0 --port=8888"