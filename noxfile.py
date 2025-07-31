"""Nox configuration for cross-environment testing and automation."""

import nox

# Define Python versions to test
PYTHON_VERSIONS = ["3.9", "3.10", "3.11", "3.12"]

# Set default sessions
nox.options.sessions = ["tests", "lint", "type_check"]


@nox.session(python=PYTHON_VERSIONS)
def tests(session):
    """Run the test suite with coverage."""
    session.install(".[dev]")
    session.run(
        "pytest",
        "--cov=surrogate_optim",
        "--cov-report=term-missing",
        "--cov-report=xml",
        "--cov-fail-under=85",
        *session.posargs,
    )


@nox.session(python="3.11")
def lint(session):
    """Run code linting with ruff."""
    session.install(".[dev]")
    session.run("ruff", "check", "surrogate_optim", "tests")
    session.run("ruff", "format", "--check", "surrogate_optim", "tests")


@nox.session(python="3.11")
def type_check(session):
    """Run type checking with mypy."""
    session.install(".[dev]")
    session.run("mypy", "surrogate_optim")


@nox.session(python="3.11")
def security(session):
    """Run security checks."""
    session.install(".[dev]")
    session.run("bandit", "-r", "surrogate_optim")
    session.run("safety", "check")


@nox.session(python="3.11")
def docs(session):
    """Build documentation."""
    session.install(".[docs]")
    session.run("sphinx-build", "-b", "html", "docs", "docs/_build/html")


@nox.session(python="3.11")
def benchmarks(session):
    """Run performance benchmarks."""
    session.install(".[benchmark]")
    session.run("pytest", "tests/benchmarks", "-v", "--benchmark-only")


@nox.session(python=PYTHON_VERSIONS)
def install_test(session):
    """Test package installation."""
    session.install(".")
    session.run("python", "-c", "import surrogate_optim; print(surrogate_optim.__version__)")


@nox.session(python="3.11")
def clean(session):
    """Clean build artifacts."""
    session.run("rm", "-rf", "build", "dist", "*.egg-info", external=True)
    session.run("rm", "-rf", ".pytest_cache", ".coverage", "htmlcov", external=True)
    session.run("rm", "-rf", ".mypy_cache", ".ruff_cache", external=True)