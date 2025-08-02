# Manual Workflow Setup Instructions

## Overview

This document provides step-by-step instructions for manually setting up GitHub Actions workflows from the provided templates.

## ⚠️ Important Notice

Due to GitHub App permission limitations, workflow files cannot be created automatically. The templates in `docs/workflows-templates/` must be manually copied to `.github/workflows/`.

## 🔧 Setup Steps

### 1. Create Workflow Directory

```bash
mkdir -p .github/workflows/
```

### 2. Copy Workflow Templates

```bash
# Copy all workflow templates
cp docs/workflows-templates/ci.yml .github/workflows/
cp docs/workflows-templates/security.yml .github/workflows/
cp docs/workflows-templates/release.yml .github/workflows/

# Optional: Copy additional workflows
cp docs/workflows-templates/dependency-update.yml .github/workflows/ 2>/dev/null || true
```

### 3. Configure Repository Secrets

Navigate to `Settings > Secrets and variables > Actions` and add:

#### Required Secrets
- `PYPI_API_TOKEN` - For PyPI package publishing
- `GITHUB_TOKEN` - Automatically provided by GitHub

#### Optional Secrets
- `CODECOV_TOKEN` - For code coverage reporting
- `SLACK_WEBHOOK_URL` - For team notifications
- `GITLEAKS_LICENSE` - For enhanced secret scanning

### 4. Set Repository Permissions

In `Settings > Actions > General`, ensure:

- ✅ Workflow permissions: "Read and write permissions"
- ✅ Allow GitHub Actions to create and approve pull requests
- ✅ Allow all actions and reusable workflows

### 5. Configure Branch Protection

In `Settings > Branches`, add protection rules for `main`:

- ✅ Require a pull request before merging
- ✅ Require status checks to pass before merging
  - Select: `Code Quality & Security`
  - Select: `Test Suite`
  - Select: `Security Scan`
- ✅ Require branches to be up to date before merging
- ✅ Require linear history

### 6. Set Up Environments

In `Settings > Environments`, create:

#### Staging Environment
- Name: `staging`
- Protection rules: No restrictions
- Environment secrets: staging-specific configuration

#### Production Environment
- Name: `production`
- Protection rules: Required reviewers (repository administrators)
- Environment secrets: production configuration

### 7. Commit and Push Workflows

```bash
git add .github/workflows/
git commit -m "feat: add enterprise CI/CD workflows

- Add comprehensive CI/CD pipeline with quality gates
- Add multi-layered security scanning workflow
- Add automated release and deployment workflow

🚀 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"

git push origin main
```

## 🧪 Testing the Setup

### 1. Test CI Pipeline
```bash
# Create a test branch and PR
git checkout -b test/workflow-setup
echo "# Test" >> TEST.md
git add TEST.md
git commit -m "test: verify workflow setup"
git push origin test/workflow-setup

# Create PR and verify all checks pass
```

### 2. Test Security Scanning
- Check the Actions tab for successful security scans
- Review security findings in the Security tab

### 3. Test Release Process
```bash
# Create a test release tag
git tag v0.1.1
git push origin v0.1.1

# Monitor release workflow in Actions tab
```

## 🔍 Verification Checklist

After setup, verify:

- [ ] CI workflow runs on push/PR
- [ ] All quality gates pass
- [ ] Security scans complete successfully
- [ ] Test suite executes properly
- [ ] Documentation builds and deploys
- [ ] Container images build successfully
- [ ] Release process works end-to-end

## 🚨 Troubleshooting

### Common Issues

1. **Permission Denied Errors**
   - Ensure repository permissions are correctly set
   - Verify secrets are properly configured

2. **Workflow Not Triggering**
   - Check branch protection rules
   - Verify workflow file syntax with GitHub Actions validator

3. **Build Failures**
   - Review logs in Actions tab
   - Check dependency versions and compatibility

4. **Security Scan Issues**
   - Review security findings
   - Update dependencies if vulnerabilities found

### Getting Help

- Check [GitHub Actions Documentation](https://docs.github.com/en/actions)
- Review workflow logs for specific errors
- Consult repository maintainers for organization-specific issues

## 📊 Expected Outcomes

After successful setup:

- **Automated Quality Gates**: Every PR gets quality, security, and test validation
- **Secure Release Process**: Tagged releases automatically build, test, and deploy
- **Security Monitoring**: Continuous vulnerability scanning and compliance checking
- **Documentation**: Automated documentation building and deployment
- **Notifications**: Team alerts for build status and security findings

## 🏆 Success Metrics

Your setup is successful when:

- All workflow badges show green status
- Security dashboard shows no critical vulnerabilities
- Documentation site is accessible and up-to-date
- Release packages are published to PyPI and container registry
- Team receives proper notifications

---

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>