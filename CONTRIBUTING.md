# Contributing to Surrogate Gradient Optimization Lab

Thank you for your interest in contributing to the Surrogate Gradient Optimization Lab! This document provides guidelines for contributing to the project.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [How to Contribute](#how-to-contribute)
4. [Development Process](#development-process)
5. [Pull Request Guidelines](#pull-request-guidelines)
6. [Issue Guidelines](#issue-guidelines)
7. [Community](#community)

## Code of Conduct

This project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to team@terragon-labs.com.

## Getting Started

### Prerequisites

Before contributing, please ensure you have:

- Python 3.9 or higher
- Git installed and configured
- Familiarity with JAX and NumPy
- Basic understanding of optimization and machine learning

### Development Setup

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/surrogate-gradient-optim-lab.git
   cd surrogate-gradient-optim-lab
   ```
3. **Set up development environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev,docs,benchmark,notebook]"
   pre-commit install
   ```
4. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/terragon-labs/surrogate-gradient-optim-lab.git
   ```

### Verify Your Setup

```bash
# Run tests to ensure everything works
pytest tests/unit/test_sample.py -v

# Check code quality
make lint

# Verify imports
python -c "import surrogate_optim; print('Setup successful!')"
```

## How to Contribute

### Types of Contributions

We welcome various types of contributions:

- **Bug reports and fixes**
- **Feature requests and implementations**
- **Documentation improvements**
- **Performance optimizations**
- **Test coverage improvements**
- **Example notebooks and tutorials**
- **Code reviews**

### Finding Work

- Check our [issue tracker](https://github.com/terragon-labs/surrogate-gradient-optim-lab/issues)
- Look for issues labeled `good first issue` or `help wanted`
- Review our [project roadmap](docs/ROADMAP.md)
- Propose new features by opening an issue

## Development Process

### 1. Planning Your Contribution

Before starting work:

1. **Check existing issues** to avoid duplicate work
2. **Open an issue** to discuss significant changes
3. **Get feedback** from maintainers on your approach
4. **Break down large features** into smaller, manageable pieces

### 2. Working on Your Contribution

1. **Create a feature branch**:
   ```bash
   git checkout main
   git pull upstream main
   git checkout -b feature/your-feature-name
   ```

2. **Write your code** following our [development guidelines](docs/DEVELOPMENT.md)

3. **Add tests** for new functionality:
   ```bash
   # Unit tests
   pytest tests/unit/test_your_feature.py
   
   # Integration tests if needed
   pytest tests/integration/test_your_integration.py
   ```

4. **Update documentation** as needed:
   - Docstrings for new functions/classes
   - API documentation updates
   - README or guide updates
   - Example notebooks

5. **Run quality checks**:
   ```bash
   # All quality checks
   make lint
   
   # Tests with coverage
   pytest --cov=surrogate_optim
   
   # Security checks
   make security
   ```

### 3. Code Standards

#### Code Quality

- Follow PEP 8 style guidelines
- Use type hints for all public functions
- Write clear, descriptive variable names
- Add docstrings for all public APIs
- Keep functions focused and modular

#### Example Code Style

```python
from typing import Optional, Tuple, Dict, Any
import jax
import jax.numpy as jnp


def train_neural_surrogate(
    X: jax.Array,
    y: jax.Array,
    hidden_dims: list[int] = [64, 64],
    learning_rate: float = 0.001,
    max_epochs: int = 1000,
    validation_split: float = 0.2,
    early_stopping: bool = True,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Train a neural network surrogate model.

    Args:
        X: Input features with shape (n_samples, n_features)
        y: Target values with shape (n_samples,)
        hidden_dims: Hidden layer dimensions
        learning_rate: Learning rate for optimization
        max_epochs: Maximum number of training epochs
        validation_split: Fraction of data used for validation
        early_stopping: Whether to use early stopping
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary containing:
            - model: Trained model parameters
            - history: Training history
            - metrics: Final training metrics

    Raises:
        ValueError: If input data is invalid
        RuntimeError: If training fails to converge

    Example:
        >>> import jax.numpy as jnp
        >>> X = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        >>> y = jnp.array([5.0, 7.0])
        >>> result = train_neural_surrogate(X, y)
        >>> model = result['model']
    """
    # Validate inputs
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have same number of samples")
    
    # Implementation here...
    pass
```

#### Testing Standards

```python
import pytest
import jax.numpy as jnp
from surrogate_optim.models import NeuralSurrogate


class TestNeuralSurrogate:
    """Test neural network surrogate model."""

    def test_initialization(self):
        """Test model initialization with various parameters."""
        model = NeuralSurrogate(hidden_dims=[32, 32])
        assert model.hidden_dims == [32, 32]

    def test_training_basic(self, sample_data_2d):
        """Test basic training functionality."""
        X, y = sample_data_2d["X"], sample_data_2d["y"]
        
        model = NeuralSurrogate()
        model.fit(X, y)
        
        # Check that model can make predictions
        predictions = model.predict(X[:5])
        assert predictions.shape == (5,)
        assert jnp.isfinite(predictions).all()

    @pytest.mark.parametrize("hidden_dims", [
        [16], [32, 32], [64, 32, 16]
    ])
    def test_different_architectures(self, sample_data_2d, hidden_dims):
        """Test training with different network architectures."""
        X, y = sample_data_2d["X"], sample_data_2d["y"]
        
        model = NeuralSurrogate(hidden_dims=hidden_dims)
        model.fit(X, y)
        
        predictions = model.predict(X[:5])
        assert predictions.shape == (5,)

    def test_gradient_computation(self, sample_data_2d):
        """Test gradient computation accuracy."""
        X, y = sample_data_2d["X"], sample_data_2d["y"]
        
        model = NeuralSurrogate()
        model.fit(X, y)
        
        point = X[0]
        gradient = model.gradient(point)
        
        assert gradient.shape == point.shape
        assert jnp.isfinite(gradient).all()

    def test_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        model = NeuralSurrogate()
        
        with pytest.raises(ValueError):
            # Wrong dimensions
            model.fit(jnp.array([1, 2, 3]), jnp.array([1, 2]))
        
        with pytest.raises(ValueError):
            # Non-finite values
            X = jnp.array([[1.0, jnp.inf], [2.0, 3.0]])
            y = jnp.array([1.0, 2.0])
            model.fit(X, y)
```

## Pull Request Guidelines

### Before Submitting

- [ ] All tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation is updated
- [ ] Commit messages follow conventions
- [ ] No merge conflicts with main branch

### Pull Request Process

1. **Update your branch**:
   ```bash
   git checkout main
   git pull upstream main
   git checkout your-feature-branch
   git rebase main
   ```

2. **Push your changes**:
   ```bash
   git push origin your-feature-branch
   ```

3. **Create pull request** on GitHub with:
   - Clear title describing the change
   - Detailed description of what was changed and why
   - Reference to related issues
   - Screenshots for UI changes
   - Testing instructions

4. **Respond to feedback** promptly and make requested changes

### Pull Request Template

Your pull request should include:

```markdown
## Description
Brief description of the changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## How Has This Been Tested?
Describe the tests you ran to verify your changes.

## Checklist:
- [ ] My code follows the style guidelines of this project
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
```

## Issue Guidelines

### Reporting Bugs

When reporting bugs, please include:

1. **Environment information**:
   - Operating system
   - Python version
   - Package version
   - JAX version

2. **Reproduction steps**:
   ```python
   # Minimal code example that reproduces the issue
   import surrogate_optim
   
   # Your code here
   ```

3. **Expected vs actual behavior**
4. **Error messages and stack traces**
5. **Additional context** (screenshots, logs, etc.)

### Feature Requests

For feature requests, please provide:

1. **Problem description**: What problem does this solve?
2. **Proposed solution**: How should it work?
3. **Use cases**: Who would benefit and how?
4. **Alternatives considered**: Other approaches you've thought about
5. **Implementation ideas**: If you have technical suggestions

### Questions and Support

For questions:

1. **Check documentation** first
2. **Search existing issues** for similar questions
3. **Provide context** about what you're trying to achieve
4. **Include code examples** when relevant

## Community

### Communication Channels

- **GitHub Issues**: Bug reports, feature requests, questions
- **GitHub Discussions**: General discussions, ideas, Q&A
- **Email**: team@terragon-labs.com for private matters

### Recognition

We recognize contributors in several ways:

- **Contributors section** in README
- **Release notes** mention significant contributions
- **Author credits** on papers and publications
- **Contributor badges** on GitHub

### Becoming a Maintainer

Regular contributors may be invited to become maintainers. Maintainers:

- Review and merge pull requests
- Triage issues and provide support
- Guide project direction and priorities
- Mentor new contributors

## Getting Help

If you need help:

1. **Read the documentation** in the `docs/` directory
2. **Check existing issues** for similar problems
3. **Ask questions** by opening a new issue
4. **Join community discussions** on GitHub Discussions

## Thank You!

Your contributions make this project better for everyone. Whether you're fixing a typo, adding a feature, or helping other users, every contribution is valuable and appreciated.

Happy coding! 🚀