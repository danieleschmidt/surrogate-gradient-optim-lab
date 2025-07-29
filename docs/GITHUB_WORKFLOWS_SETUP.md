# GitHub Actions Workflows Setup Guide

## Overview

This document provides comprehensive GitHub Actions workflow configurations for the Surrogate Gradient Optimization Lab project. Due to GitHub App permission limitations, these workflows must be manually created in the `.github/workflows/` directory.

## Required Repository Secrets

Before implementing workflows, configure these secrets in your repository settings (Settings → Secrets and variables → Actions):

### Core Secrets
- `CODECOV_TOKEN` - Codecov token for coverage reporting
- `DOCKER_USERNAME` - Docker Hub username
- `DOCKER_PASSWORD` - Docker Hub password or token
- `PYPI_API_TOKEN` - PyPI token for package publishing

### Optional Secrets (for notifications)
- `SLACK_WEBHOOK_URL` - Slack webhook for notifications
- `DISCORD_WEBHOOK_URL` - Discord webhook for notifications

## Workflow Files to Create

### 1. Main CI Pipeline (`.github/workflows/ci.yml`)

```yaml
name: Continuous Integration

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]
  workflow_dispatch:

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

env:
  PYTHON_VERSION: "3.9"
  CACHE_VERSION: v1

jobs:
  lint-and-format:
    name: Lint and Format
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ env.PYTHON_VERSION }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'
          cache-dependency-path: pyproject.toml

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"

      - name: Run pre-commit hooks
        uses: pre-commit/action@v3.0.0
        with:
          extra_args: --all-files

  security-scan:
    name: Security Scan
    runs-on: ubuntu-latest
    permissions:
      security-events: write
      contents: read
      actions: read
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ env.PYTHON_VERSION }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install bandit[toml] safety semgrep

      - name: Run Bandit security scan
        run: |
          bandit -r surrogate_optim/ -f sarif -o bandit-results.sarif
        continue-on-error: true

      - name: Run Safety dependency scan
        run: |
          safety check --json --output safety-results.json
        continue-on-error: true

      - name: Run Semgrep scan
        run: |
          semgrep --config=auto --json --output=semgrep-results.json surrogate_optim/
        continue-on-error: true

      - name: Upload security scan results
        uses: github/codeql-action/upload-sarif@v2
        if: always()
        with:
          sarif_file: bandit-results.sarif

  test:
    name: Test Suite
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ["3.9", "3.10", "3.11", "3.12"]
        exclude:
          - os: windows-latest
            python-version: "3.12"
          - os: macos-latest
            python-version: "3.12"

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

      - name: Run unit tests
        run: |
          pytest tests/unit/ -v \
            --cov=surrogate_optim \
            --cov-report=xml \
            --cov-report=term-missing \
            --cov-fail-under=80 \
            --junitxml=pytest-results.xml

      - name: Run integration tests
        run: |
          pytest tests/integration/ -v \
            --cov=surrogate_optim \
            --cov-append \
            --cov-report=xml \
            --junitxml=pytest-integration.xml

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        if: matrix.os == 'ubuntu-latest' && matrix.python-version == '3.9'
        with:
          token: ${{ secrets.CODECOV_TOKEN }}
          file: ./coverage.xml
          flags: unittests
          name: codecov-umbrella

      - name: Upload test results
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: test-results-${{ matrix.os }}-${{ matrix.python-version }}
          path: |
            pytest-results.xml
            pytest-integration.xml
            coverage.xml

  benchmark:
    name: Performance Benchmarks
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ env.PYTHON_VERSION }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev,benchmark]"

      - name: Run benchmark tests
        run: |
          pytest tests/benchmarks/ -v \
            --benchmark-only \
            --benchmark-json=benchmark-results.json \
            --benchmark-sort=mean

      - name: Store benchmark results
        uses: benchmark-action/github-action-benchmark@v1
        with:
          tool: 'pytest'
          output-file-path: benchmark-results.json
          github-token: ${{ secrets.GITHUB_TOKEN }}
          auto-push: true
          comment-on-alert: true
          alert-threshold: '150%'
          fail-on-alert: true

  docker:
    name: Docker Build
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Login to Docker Hub
        if: github.event_name == 'push' && github.ref == 'refs/heads/main'
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: |
            ${{ secrets.DOCKER_USERNAME }}/surrogate-optim
          tags: |
            type=ref,event=branch
            type=ref,event=pr
            type=sha,prefix={{branch}}-
            type=raw,value=latest,enable={{is_default_branch}}

      - name: Build and push Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          target: production
          push: ${{ github.event_name == 'push' && github.ref == 'refs/heads/main' }}
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

      - name: Build GPU image
        if: github.event_name == 'push' && github.ref == 'refs/heads/main'
        uses: docker/build-push-action@v5
        with:
          context: .
          target: gpu
          push: true
          tags: ${{ secrets.DOCKER_USERNAME }}/surrogate-optim:gpu-latest
          cache-from: type=gha
          cache-to: type=gha,mode=max

  docs:
    name: Documentation
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ env.PYTHON_VERSION }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[docs]"

      - name: Build documentation
        run: |
          cd docs && make html

      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        if: success()
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./docs/_build/html

  notify:
    name: Notify Results
    runs-on: ubuntu-latest
    needs: [lint-and-format, security-scan, test, docker]
    if: always() && (github.event_name == 'push' || github.event_name == 'pull_request')
    steps:
      - name: Determine overall status
        id: status
        run: |
          if [[ "${{ needs.lint-and-format.result }}" == "success" && \
                "${{ needs.security-scan.result }}" == "success" && \
                "${{ needs.test.result }}" == "success" && \
                "${{ needs.docker.result }}" == "success" ]]; then
            echo "status=success" >> $GITHUB_OUTPUT
            echo "color=good" >> $GITHUB_OUTPUT
          else
            echo "status=failure" >> $GITHUB_OUTPUT
            echo "color=danger" >> $GITHUB_OUTPUT
          fi

      - name: Notify Slack on failure
        if: steps.status.outputs.status == 'failure' && github.event_name == 'push'
        uses: 8398a7/action-slack@v3
        with:
          status: failure
          webhook_url: ${{ secrets.SLACK_WEBHOOK_URL }}
          text: "❌ CI failed for ${{ github.repository }} on branch ${{ github.ref_name }}"
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}

      - name: Comment PR on failure
        if: steps.status.outputs.status == 'failure' && github.event_name == 'pull_request'
        uses: actions/github-script@v6
        with:
          script: |
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: '❌ CI checks failed. Please review the logs and fix any issues before merging.'
            })
```

### 2. Security Scanning Workflow (`.github/workflows/security-scan.yml`)

```yaml
name: Security Scanning

on:
  schedule:
    - cron: '0 2 * * 1'  # Weekly on Monday at 2 AM UTC
  workflow_dispatch:
  push:
    branches: [ main ]
    paths:
      - '**/*.py'
      - 'pyproject.toml'
      - 'requirements*.txt'
      - '.github/workflows/security-scan.yml'

permissions:
  contents: read
  security-events: write
  actions: read

jobs:
  codeql-analysis:
    name: CodeQL Analysis
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        language: [ 'python' ]

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Initialize CodeQL
        uses: github/codeql-action/init@v2
        with:
          languages: ${{ matrix.language }}
          queries: +security-and-quality

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
          cache: 'pip'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e .

      - name: Perform CodeQL Analysis
        uses: github/codeql-action/analyze@v2
        with:
          category: "/language:${{matrix.language}}"

  dependency-scan:
    name: Dependency Security Scan
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
          pip install safety bandit[toml] pip-audit cyclonedx-bom

      - name: Generate SBOM
        run: |
          pip install -e .
          cyclonedx-py -o sbom.json

      - name: Run Safety check
        run: |
          safety check --json --output safety-report.json
        continue-on-error: true

      - name: Run pip-audit
        run: |
          pip-audit --format=json --output=pip-audit-report.json
        continue-on-error: true

      - name: Run Bandit security scan
        run: |
          bandit -r surrogate_optim/ -f json -o bandit-report.json
        continue-on-error: true

      - name: Upload SBOM
        uses: actions/upload-artifact@v3
        with:
          name: sbom
          path: sbom.json

      - name: Upload security reports
        uses: actions/upload-artifact@v3
        with:
          name: security-reports
          path: |
            safety-report.json
            pip-audit-report.json
            bandit-report.json

  container-scan:
    name: Container Security Scan
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Build Docker image
        run: |
          docker build -t surrogate-optim:security-scan .

      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: 'surrogate-optim:security-scan'
          format: 'sarif'
          output: 'trivy-results.sarif'

      - name: Upload Trivy scan results to GitHub Security tab
        uses: github/codeql-action/upload-sarif@v2
        if: always()
        with:
          sarif_file: 'trivy-results.sarif'

      - name: Run Trivy for critical and high CVEs
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: 'surrogate-optim:security-scan'
          format: 'table'
          exit-code: '1'
          ignore-unfixed: true
          vuln-type: 'os,library'
          severity: 'CRITICAL,HIGH'

  secrets-scan:
    name: Secrets Detection
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install detect-secrets
        run: |
          pip install detect-secrets

      - name: Run secrets scan
        run: |
          detect-secrets scan --all-files --baseline .secrets.baseline --exclude-files '\.git/.*'

      - name: Verify secrets baseline
        run: |
          detect-secrets audit .secrets.baseline

  semgrep-scan:
    name: Semgrep Security Analysis
    runs-on: ubuntu-latest
    container:
      image: returntocorp/semgrep
    steps:
      - uses: actions/checkout@v4

      - name: Run Semgrep
        run: |
          semgrep --config=auto --sarif --output=semgrep.sarif

      - name: Upload Semgrep results
        uses: github/codeql-action/upload-sarif@v2
        if: always()
        with:
          sarif_file: semgrep.sarif

  license-compliance:
    name: License Compliance Check
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install pip-licenses
        run: |
          pip install pip-licenses

      - name: Install project dependencies
        run: |
          pip install -e .

      - name: Check licenses
        run: |
          pip-licenses --format=json --output-file=licenses.json
          pip-licenses --format=csv --output-file=licenses.csv

      - name: Verify license compatibility
        run: |
          python -c "
import json
with open('licenses.json') as f:
    licenses = json.load(f)

prohibited = ['GPL-3.0', 'AGPL-3.0', 'LGPL-3.0']
issues = []

for pkg in licenses:
    if pkg['License'] in prohibited:
        issues.append(f\"{pkg['Name']}: {pkg['License']}\")

if issues:
    print('❌ License compliance issues found:')
    for issue in issues:
        print(f'  - {issue}')
    exit(1)
else:
    print('✅ All licenses are compatible')
"

      - name: Upload license reports
        uses: actions/upload-artifact@v3
        with:
          name: license-reports
          path: |
            licenses.json
            licenses.csv

  security-report:
    name: Generate Security Report
    runs-on: ubuntu-latest
    needs: [codeql-analysis, dependency-scan, container-scan, secrets-scan, semgrep-scan, license-compliance]
    if: always()
    steps:
      - uses: actions/checkout@v4

      - name: Download all security artifacts
        uses: actions/download-artifact@v3

      - name: Generate security summary
        run: |
          cat > security-summary.md << 'EOF'
          # Security Scan Summary
          
          **Scan Date:** $(date -u +"%Y-%m-%d %H:%M:%S UTC")
          **Repository:** ${{ github.repository }}
          **Branch:** ${{ github.ref_name }}
          **Commit:** ${{ github.sha }}
          
          ## Scan Results
          
          | Scan Type | Status | Details |
          |-----------|--------|---------|
          | CodeQL Analysis | ${{ needs.codeql-analysis.result }} | Static code analysis |
          | Dependency Scan | ${{ needs.dependency-scan.result }} | Vulnerability scanning |
          | Container Scan | ${{ needs.container-scan.result }} | Docker image analysis |
          | Secrets Detection | ${{ needs.secrets-scan.result }} | Secret leak detection |
          | Semgrep Analysis | ${{ needs.semgrep-scan.result }} | Security rule engine |
          | License Compliance | ${{ needs.license-compliance.result }} | License verification |
          
          ## Next Steps
          
          - Review any failed scans in the GitHub Security tab
          - Update dependencies with known vulnerabilities
          - Address any policy violations
          - Regenerate SBOM for compliance tracking
          
          EOF

      - name: Upload security summary
        uses: actions/upload-artifact@v3
        with:
          name: security-summary
          path: security-summary.md

      - name: Notify on critical findings
        if: needs.container-scan.result == 'failure'
        uses: 8398a7/action-slack@v3
        with:
          status: failure
          webhook_url: ${{ secrets.SLACK_WEBHOOK_URL }}
          text: "🚨 Critical security vulnerabilities found in ${{ github.repository }}"
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
```

### 3. Dependency Updates Workflow (`.github/workflows/dependency-update.yml`)

```yaml
name: Automated Dependency Updates

on:
  schedule:
    - cron: '0 4 * * 1'  # Weekly on Monday at 4 AM UTC
  workflow_dispatch:

permissions:
  contents: write
  pull-requests: write

jobs:
  update-python-dependencies:
    name: Update Python Dependencies
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          token: ${{ secrets.GITHUB_TOKEN }}

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
          cache: 'pip'

      - name: Install pip-tools
        run: |
          python -m pip install --upgrade pip
          pip install pip-tools

      - name: Update dependencies
        run: |
          # Generate updated requirements files
          pip-compile pyproject.toml --upgrade --output-file requirements.txt
          pip-compile pyproject.toml --extra dev --upgrade --output-file requirements-dev.txt
          pip-compile pyproject.toml --extra test --upgrade --output-file requirements-test.txt
          pip-compile pyproject.toml --extra docs --upgrade --output-file requirements-docs.txt

      - name: Check for security vulnerabilities
        run: |
          pip install safety
          safety check --json --output safety-report.json || true

      - name: Generate dependency report
        run: |
          cat > dependency-update-report.md << 'EOF'
          # Dependency Update Report
          
          **Update Date:** $(date -u +"%Y-%m-%d")
          **Python Version:** 3.9+
          
          ## Updated Dependencies
          
          This automated update includes the latest versions of all dependencies while maintaining compatibility.
          
          ## Security Check
          
          EOF
          
          if [ -f safety-report.json ]; then
            echo "Security scan completed. Review safety-report.json for any vulnerabilities." >> dependency-update-report.md
          fi
          
          echo "" >> dependency-update-report.md
          echo "## Testing" >> dependency-update-report.md
          echo "" >> dependency-update-report.md
          echo "- [ ] All tests pass" >> dependency-update-report.md
          echo "- [ ] No breaking changes detected" >> dependency-update-report.md
          echo "- [ ] Security vulnerabilities addressed" >> dependency-update-report.md

      - name: Create Pull Request
        uses: peter-evans/create-pull-request@v5
        with:
          token: ${{ secrets.GITHUB_TOKEN }}
          commit-message: 'chore: update Python dependencies'
          title: 'chore: automated dependency updates'
          body-path: dependency-update-report.md
          branch: chore/dependency-updates
          delete-branch: true
          labels: |
            dependencies
            automated
            chore
          assignees: ${{ github.repository_owner }}
          reviewers: ${{ github.repository_owner }}

  update-github-actions:
    name: Update GitHub Actions
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          token: ${{ secrets.GITHUB_TOKEN }}

      - name: Update GitHub Actions versions
        uses: nicknovitski/action-updater@v1
        with:
          patterns: |
            .github/workflows/*.yml
            .github/workflows/*.yaml

      - name: Create Pull Request for Actions Updates
        uses: peter-evans/create-pull-request@v5
        with:
          token: ${{ secrets.GITHUB_TOKEN }}
          commit-message: 'ci: update GitHub Actions versions'
          title: 'ci: automated GitHub Actions updates'
          body: |
            Automated update of GitHub Actions to their latest versions.
            
            ## Changes
            - Updated action versions in workflow files
            - Maintained compatibility with existing configurations
            
            ## Testing
            - [ ] All workflows execute successfully
            - [ ] No breaking changes in action APIs
          branch: ci/actions-updates
          delete-branch: true
          labels: |
            ci
            github-actions
            automated

  security-audit:
    name: Security Audit
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install audit tools
        run: |
          pip install safety bandit pip-audit

      - name: Run comprehensive security audit
        run: |
          # Safety check
          safety check --json --output safety-audit.json || true
          
          # Pip audit
          pip-audit --format=json --output=pip-audit-results.json || true
          
          # Bandit security scan
          bandit -r surrogate_optim/ -f json -o bandit-audit.json || true

      - name: Generate security report
        run: |
          cat > security-audit-report.md << 'EOF'
          # Security Audit Report
          
          **Audit Date:** $(date -u +"%Y-%m-%d %H:%M:%S UTC")
          **Repository:** ${{ github.repository }}
          
          ## Audit Tools Used
          - Safety: Python package vulnerability scanner
          - Pip-audit: Python package auditing tool
          - Bandit: Python security linter
          
          ## Results
          
          Review the attached JSON reports for detailed findings:
          - `safety-audit.json`: Known vulnerabilities in dependencies
          - `pip-audit-results.json`: Comprehensive package audit
          - `bandit-audit.json`: Static security analysis
          
          ## Recommendations
          
          1. Update any packages with known vulnerabilities
          2. Review and address any security warnings
          3. Consider implementing additional security measures
          
          EOF

      - name: Upload audit results
        uses: actions/upload-artifact@v3
        with:
          name: security-audit-results
          path: |
            safety-audit.json
            pip-audit-results.json
            bandit-audit.json
            security-audit-report.md

      - name: Create security issue if vulnerabilities found
        uses: actions/github-script@v6
        with:
          script: |
            const fs = require('fs');
            
            let hasVulnerabilities = false;
            let issueBody = '# Security Vulnerabilities Detected\n\n';
            
            // Check safety results
            try {
              const safetyData = JSON.parse(fs.readFileSync('safety-audit.json', 'utf8'));
              if (safetyData.vulnerabilities && safetyData.vulnerabilities.length > 0) {
                hasVulnerabilities = true;
                issueBody += '## Safety Scan Results\n\n';
                issueBody += `Found ${safetyData.vulnerabilities.length} vulnerabilities in dependencies.\n\n`;
              }
            } catch (e) {
              console.log('Safety results not available');
            }
            
            // Check pip-audit results
            try {
              const pipAuditData = JSON.parse(fs.readFileSync('pip-audit-results.json', 'utf8'));
              if (pipAuditData.vulnerabilities && pipAuditData.vulnerabilities.length > 0) {
                hasVulnerabilities = true;
                issueBody += '## Pip-Audit Results\n\n';
                issueBody += `Found ${pipAuditData.vulnerabilities.length} vulnerabilities in packages.\n\n`;
              }
            } catch (e) {
              console.log('Pip-audit results not available');
            }
            
            if (hasVulnerabilities) {
              issueBody += '## Action Required\n\n';
              issueBody += '- Review detailed vulnerability reports in the workflow artifacts\n';
              issueBody += '- Update vulnerable packages to secure versions\n';
              issueBody += '- Test thoroughly after updates\n';
              issueBody += '- Close this issue once vulnerabilities are resolved\n\n';
              issueBody += `**Workflow Run:** ${context.payload.repository.html_url}/actions/runs/${context.runId}`;
              
              await github.rest.issues.create({
                owner: context.repo.owner,
                repo: context.repo.repo,
                title: '🚨 Security Vulnerabilities Detected',
                body: issueBody,
                labels: ['security', 'vulnerability', 'automated']
              });
            }
```

## Implementation Steps

1. **Create the workflow directory structure:**
   ```bash
   mkdir -p .github/workflows
   ```

2. **Copy the workflow content:**
   - Copy each workflow configuration above into separate files in `.github/workflows/`
   - Ensure file names match exactly: `ci.yml`, `security-scan.yml`, `dependency-update.yml`

3. **Configure repository secrets:**
   - Go to Settings → Secrets and variables → Actions
   - Add the required secrets listed above

4. **Enable GitHub Actions:**
   - Go to Settings → Actions → General
   - Enable "Allow all actions and reusable workflows"

5. **Set up branch protection:**
   - Go to Settings → Branches
   - Add protection rule for `main` branch
   - Require status checks: `lint-and-format`, `security-scan`, `test`
   - Require branches to be up to date before merging

## Additional Configuration

### Dependabot Setup

Create `.github/dependabot.yml`:

```yaml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    reviewers:
      - "your-username"
    assignees:
      - "your-username"
    commit-message:
      prefix: "chore"
      include: "scope"
  
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
    reviewers:
      - "your-username"
    assignees:
      - "your-username"
    commit-message:
      prefix: "ci"
```

### CodeQL Configuration

Create `.github/codeql/codeql-config.yml`:

```yaml
name: "CodeQL Config"

queries:
  - uses: security-and-quality

paths-ignore:
  - docs/
  - tests/fixtures/
  - examples/

paths:
  - surrogate_optim/
```

## Monitoring and Maintenance

1. **Regular Review:**
   - Check workflow runs weekly
   - Review security scan results
   - Monitor performance benchmarks

2. **Update Cycles:**
   - Dependencies updated automatically weekly
   - GitHub Actions updated automatically weekly
   - Security scans run automatically on push and weekly

3. **Failure Response:**
   - Security issues create automatic GitHub issues
   - Failed workflows trigger Slack notifications
   - PR comments provide immediate feedback

This comprehensive setup provides enterprise-grade CI/CD with security scanning, automated maintenance, and comprehensive monitoring for the Surrogate Gradient Optimization Lab project.