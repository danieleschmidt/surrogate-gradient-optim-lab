# CI/CD Setup Guide

This document provides comprehensive instructions for setting up GitHub Actions workflows for the Surrogate Gradient Optimization Lab project.

## Overview

The CI/CD pipeline consists of several workflows that ensure code quality, security, and automated deployment:

1. **Pull Request Validation** - Runs on every PR
2. **Main Branch CI** - Runs on pushes to main
3. **Release Automation** - Triggered by version tags
4. **Security Scanning** - Scheduled security checks
5. **Dependency Updates** - Automated dependency management

## Required GitHub Secrets

Before setting up workflows, configure these secrets in your repository settings:

### Core Secrets
- `PYPI_API_TOKEN` - PyPI token for package publishing
- `DOCKER_USERNAME` - Docker Hub username
- `DOCKER_PASSWORD` - Docker Hub password or token
- `CODECOV_TOKEN` - Codecov token for coverage reporting

### Optional Secrets (for advanced features)
- `SLACK_WEBHOOK_URL` - Slack notifications
- `DISCORD_WEBHOOK_URL` - Discord notifications
- `WANDB_API_KEY` - Weights & Biases integration
- `SONAR_TOKEN` - SonarCloud code quality analysis

## Workflow Configurations

### 1. Pull Request Validation (`.github/workflows/pr-validation.yml`)

```yaml
name: Pull Request Validation

on:
  pull_request:
    branches: [main, develop]
    types: [opened, synchronize, reopened]

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  code-quality:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11', '3.12']
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
        cache: 'pip'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev,test]"
    
    - name: Run linting
      run: |
        ruff check surrogate_optim/ tests/
        black --check surrogate_optim/ tests/
        isort --check-only surrogate_optim/ tests/
    
    - name: Run type checking
      run: mypy surrogate_optim/
    
    - name: Run security checks
      run: |
        bandit -r surrogate_optim/ -x tests/
        safety check
    
    - name: Run tests
      run: |
        pytest tests/ -v \
          --cov=surrogate_optim \
          --cov-report=xml \
          --cov-report=term-missing \
          --cov-fail-under=80
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        token: ${{ secrets.CODECOV_TOKEN }}
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella

  docker-build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Build Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        push: false
        tags: surrogate-optim:test
        cache-from: type=gha
        cache-to: type=gha,mode=max

  performance-tests:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
        cache: 'pip'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev,benchmark]"
    
    - name: Run benchmark tests
      run: |
        pytest tests/benchmarks/ -v \
          --benchmark-only \
          --benchmark-json=benchmark_results.json
    
    - name: Store benchmark results
      uses: benchmark-action/github-action-benchmark@v1
      with:
        tool: 'pytest'
        output-file-path: benchmark_results.json
        github-token: ${{ secrets.GITHUB_TOKEN }}
        auto-push: false
```

### 2. Main Branch CI (`.github/workflows/main-ci.yml`)

```yaml
name: Main Branch CI

on:
  push:
    branches: [main]
  workflow_dispatch:

jobs:
  test-and-build:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11', '3.12']
    
    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
        cache: 'pip'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev,test]"
    
    - name: Run full test suite
      run: |
        pytest tests/ -v \
          --cov=surrogate_optim \
          --cov-report=xml \
          --cov-report=html \
          --junitxml=pytest.xml \
          --cov-fail-under=80
    
    - name: Upload coverage reports
      uses: codecov/codecov-action@v3
      with:
        token: ${{ secrets.CODECOV_TOKEN }}
        file: ./coverage.xml
        flags: unittests
    
    - name: Build package
      run: |
        python -m build
    
    - name: Upload build artifacts
      uses: actions/upload-artifact@v3
      with:
        name: dist-${{ matrix.python-version }}
        path: dist/

  docker-build-and-push:
    runs-on: ubuntu-latest
    needs: test-and-build
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Login to Docker Hub
      uses: docker/login-action@v3
      with:
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_PASSWORD }}
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        push: true
        tags: |
          ${{ secrets.DOCKER_USERNAME }}/surrogate-optim:latest
          ${{ secrets.DOCKER_USERNAME }}/surrogate-optim:${{ github.sha }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  security-scan:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        scan-type: 'fs'
        scan-ref: '.'
        format: 'sarif'
        output: 'trivy-results.sarif'
    
    - name: Upload Trivy scan results to GitHub Security tab
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-results.sarif'
```

### 3. Release Automation (`.github/workflows/release.yml`)

```yaml
name: Release

on:
  push:
    tags:
      - 'v*.*.*'

jobs:
  create-release:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
        cache: 'pip'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
        pip install build twine
    
    - name: Run tests
      run: |
        pytest tests/ -v --cov=surrogate_optim --cov-fail-under=80
    
    - name: Build package
      run: |
        python -m build
    
    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: |
        twine upload dist/*
    
    - name: Build Docker images
      run: |
        docker build -t surrogate-optim:${{ github.ref_name }} .
        docker build --target gpu -t surrogate-optim:${{ github.ref_name }}-gpu .
    
    - name: Push Docker images
      run: |
        echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
        docker tag surrogate-optim:${{ github.ref_name }} ${{ secrets.DOCKER_USERNAME }}/surrogate-optim:${{ github.ref_name }}
        docker tag surrogate-optim:${{ github.ref_name }} ${{ secrets.DOCKER_USERNAME }}/surrogate-optim:latest
        docker push ${{ secrets.DOCKER_USERNAME }}/surrogate-optim:${{ github.ref_name }}
        docker push ${{ secrets.DOCKER_USERNAME }}/surrogate-optim:latest
    
    - name: Create GitHub Release
      uses: actions/create-release@v1
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      with:
        tag_name: ${{ github.ref }}
        release_name: Release ${{ github.ref }}
        body: |
          ## Changes
          
          Automated release for version ${{ github.ref_name }}
          
          ## Installation
          
          ```bash
          pip install surrogate-gradient-optim-lab==${{ github.ref_name }}
          ```
          
          ## Docker
          
          ```bash
          docker pull ${{ secrets.DOCKER_USERNAME }}/surrogate-optim:${{ github.ref_name }}
          ```
        draft: false
        prerelease: false
```

### 4. Security Scanning (`.github/workflows/security.yml`)

```yaml
name: Security Scanning

on:
  schedule:
    - cron: '0 2 * * 1'  # Weekly on Monday at 2 AM
  workflow_dispatch:

jobs:
  codeql-analysis:
    runs-on: ubuntu-latest
    permissions:
      actions: read
      contents: read
      security-events: write
    
    strategy:
      fail-fast: false
      matrix:
        language: ['python']
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Initialize CodeQL
      uses: github/codeql-action/init@v2
      with:
        languages: ${{ matrix.language }}
    
    - name: Autobuild
      uses: github/codeql-action/autobuild@v2
    
    - name: Perform CodeQL Analysis
      uses: github/codeql-action/analyze@v2

  dependency-scan:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install safety bandit semgrep
    
    - name: Run safety check
      run: safety check --json --output safety-report.json || true
    
    - name: Run bandit security linter
      run: bandit -r surrogate_optim/ -f json -o bandit-report.json || true
    
    - name: Run semgrep
      run: semgrep --config=auto --json --output=semgrep-report.json surrogate_optim/ || true
    
    - name: Upload security reports
      uses: actions/upload-artifact@v3
      with:
        name: security-reports
        path: |
          safety-report.json
          bandit-report.json
          semgrep-report.json

  docker-security-scan:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Build Docker image
      run: docker build -t surrogate-optim:security-scan .
    
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: 'surrogate-optim:security-scan'
        format: 'table'
        exit-code: '1'
        ignore-unfixed: true
        vuln-type: 'os,library'
        severity: 'CRITICAL,HIGH'
```

### 5. Dependency Updates (`.github/workflows/dependency-update.yml`)

```yaml
name: Dependency Updates

on:
  schedule:
    - cron: '0 4 * * 1'  # Weekly on Monday at 4 AM
  workflow_dispatch:

jobs:
  update-dependencies:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
      with:
        token: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install pip-tools
    
    - name: Update dependencies
      run: |
        pip-compile pyproject.toml --upgrade --output-file requirements.txt
        pip-compile pyproject.toml --extra dev --upgrade --output-file requirements-dev.txt
    
    - name: Create Pull Request
      uses: peter-evans/create-pull-request@v5
      with:
        token: ${{ secrets.GITHUB_TOKEN }}
        commit-message: 'chore: update dependencies'
        title: 'chore: automated dependency updates'
        body: |
          Automated dependency updates generated by GitHub Actions.
          
          Please review the changes and ensure all tests pass before merging.
        branch: chore/dependency-updates
        delete-branch: true
```

## Branch Protection Rules

Configure these branch protection rules for the `main` branch:

1. **Require pull request reviews before merging**
   - Required number of reviewers: 1
   - Dismiss stale reviews when new commits are pushed
   - Require review from CODEOWNERS

2. **Require status checks to pass before merging**
   - Require branches to be up to date before merging
   - Required status checks:
     - `code-quality (3.9)`
     - `code-quality (3.10)`
     - `code-quality (3.11)`
     - `code-quality (3.12)`
     - `docker-build`

3. **Require conversation resolution before merging**

4. **Restrict pushes that create files**
   - Restrict pushes to the `main` branch

## Monitoring and Notifications

### Slack Integration

Add Slack webhook notifications to your workflows:

```yaml
- name: Notify Slack on failure
  if: failure()
  uses: 8398a7/action-slack@v3
  with:
    status: failure
    webhook_url: ${{ secrets.SLACK_WEBHOOK_URL }}
    text: "❌ CI failed for ${{ github.repository }} on branch ${{ github.ref }}"
```

### Discord Integration

Add Discord webhook notifications:

```yaml
- name: Notify Discord on success
  if: success()
  uses: sarisia/actions-status-discord@v1
  with:
    webhook: ${{ secrets.DISCORD_WEBHOOK_URL }}
    title: "✅ CI passed for ${{ github.repository }}"
    description: "All checks passed on branch ${{ github.ref }}"
```

## Performance Monitoring

### Benchmark Tracking

Set up continuous benchmarking with GitHub Actions Benchmark:

```yaml
- name: Store benchmark results
  uses: benchmark-action/github-action-benchmark@v1
  with:
    tool: 'pytest'
    output-file-path: benchmark_results.json
    github-token: ${{ secrets.GITHUB_TOKEN }}
    auto-push: true
    comment-on-alert: true
    alert-threshold: '200%'
    fail-on-alert: true
```

### Code Quality Metrics

Integrate with SonarCloud for detailed code quality analysis:

```yaml
- name: SonarCloud Scan
  uses: SonarSource/sonarcloud-github-action@master
  env:
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
    SONAR_TOKEN: ${{ secrets.SONAR_TOKEN }}
```

## Troubleshooting

### Common Issues

1. **Test failures on specific Python versions**
   - Check for version-specific dependencies
   - Review compatibility matrices

2. **Docker build failures**
   - Verify Dockerfile syntax
   - Check for missing dependencies

3. **Security scan failures**
   - Review vulnerability reports
   - Update dependencies with known vulnerabilities

4. **Coverage threshold failures**
   - Add missing tests
   - Review coverage exclusions

### Debug Mode

Enable debug logging in workflows:

```yaml
env:
  ACTIONS_STEP_DEBUG: true
  ACTIONS_RUNNER_DEBUG: true
```

## Implementation Checklist

- [ ] Create all workflow files in `.github/workflows/`
- [ ] Configure required secrets in repository settings
- [ ] Set up branch protection rules
- [ ] Configure status checks
- [ ] Test workflows with a small change
- [ ] Set up monitoring and notifications
- [ ] Document any custom requirements
- [ ] Train team on workflow usage

## Next Steps

After implementing the CI/CD pipeline:

1. Monitor workflow performance and adjust as needed
2. Add additional security scanning tools
3. Implement blue-green deployment strategies
4. Set up staging environment automation
5. Configure automated rollback procedures