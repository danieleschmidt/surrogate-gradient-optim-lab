# Workflow Requirements Documentation

## Overview

This directory contains workflow requirements and templates for the Surrogate Gradient Optimization Lab project's CI/CD implementation.

## Required GitHub Actions Workflows

The following workflows need to be manually created in `.github/workflows/` due to permission limitations:

### 1. CI Workflow (`ci.yml`)
- **Triggers**: Pull requests, pushes to main
- **Jobs**: Lint, test, security scan, type check
- **Required checks**: All tests must pass before merge

### 2. Security Scanning (`security.yml`)
- **Triggers**: Schedule (weekly), pull requests
- **Tools**: Bandit, Safety, CodeQL
- **Reports**: SARIF upload to GitHub Security tab

### 3. Release Automation (`release.yml`)
- **Triggers**: Tags matching `v*`
- **Actions**: Build, test, create GitHub release
- **Artifacts**: PyPI upload, documentation deployment

### 4. Dependency Updates (`dependabot.yml`)
- **Schedule**: Weekly dependency checks
- **Scope**: Python packages, GitHub Actions
- **Auto-merge**: Security patches only

## Workflow Templates

Example workflow templates are documented in [SETUP_REQUIRED.md](../SETUP_REQUIRED.md).

## Manual Setup Required

1. Copy workflow templates from documentation
2. Create `.github/workflows/` directory
3. Configure repository branch protection rules
4. Enable required status checks

## References

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Python CI/CD Best Practices](https://docs.python.org/3/tutorial/packaging.html)
- [Security Scanning Guide](https://github.com/features/security)

## Status

- ✅ Documentation complete
- ⚠️ Workflows require manual creation (permission limitations)
- ⚠️ Branch protection requires admin access