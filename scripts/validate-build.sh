#!/bin/bash

# Build validation script for Surrogate Gradient Optimization Lab
# Validates that all build components are properly configured

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

log_info() {
    echo "[INFO] $*"
}

log_success() {
    echo "[SUCCESS] $*"
}

log_error() {
    echo "[ERROR] $*" >&2
}

check_file() {
    local file="$1"
    local description="$2"
    
    if [[ -f "$file" ]]; then
        log_success "$description: $file"
        return 0
    else
        log_error "$description missing: $file"
        return 1
    fi
}

check_build_files() {
    log_info "Checking build configuration files..."
    
    local errors=0
    
    check_file "$PROJECT_ROOT/Dockerfile" "Dockerfile" || ((errors++))
    check_file "$PROJECT_ROOT/docker-compose.yml" "Docker Compose" || ((errors++))
    check_file "$PROJECT_ROOT/.dockerignore" "Docker ignore" || ((errors++))
    check_file "$PROJECT_ROOT/Makefile" "Makefile" || ((errors++))
    check_file "$PROJECT_ROOT/pyproject.toml" "Python project config" || ((errors++))
    check_file "$PROJECT_ROOT/scripts/build.sh" "Build script" || ((errors++))
    check_file "$PROJECT_ROOT/scripts/release.sh" "Release script" || ((errors++))
    
    return $errors
}

check_docker_targets() {
    log_info "Checking Docker targets..."
    
    local dockerfile="$PROJECT_ROOT/Dockerfile"
    local targets=("base" "dependencies" "development" "production" "gpu")
    local errors=0
    
    for target in "${targets[@]}"; do
        if grep -q "FROM .* as $target" "$dockerfile"; then
            log_success "Docker target: $target"
        else
            log_error "Docker target missing: $target"
            ((errors++))
        fi
    done
    
    return $errors
}

check_makefile_targets() {
    log_info "Checking Makefile targets..."
    
    local makefile="$PROJECT_ROOT/Makefile"
    local targets=("build" "test" "lint" "docker-build" "docker-run" "install-dev")
    local errors=0
    
    for target in "${targets[@]}"; do
        if grep -q "^$target:" "$makefile"; then
            log_success "Makefile target: $target"
        else
            log_error "Makefile target missing: $target"
            ((errors++))
        fi
    done
    
    return $errors
}

check_docker_compose_services() {
    log_info "Checking Docker Compose services..."
    
    local compose_file="$PROJECT_ROOT/docker-compose.yml"
    local services=("surrogate-optim-dev" "surrogate-optim-prod" "surrogate-optim-gpu")
    local errors=0
    
    for service in "${services[@]}"; do
        if grep -q "  $service:" "$compose_file"; then
            log_success "Docker Compose service: $service"
        else
            log_error "Docker Compose service missing: $service"
            ((errors++))
        fi
    done
    
    return $errors
}

validate_python_config() {
    log_info "Validating Python configuration..."
    
    cd "$PROJECT_ROOT"
    
    # Check if pyproject.toml is valid
    if python -c "import toml; toml.load('pyproject.toml')" 2>/dev/null; then
        log_success "pyproject.toml is valid"
    else
        log_error "pyproject.toml is invalid"
        return 1
    fi
    
    # Check if version can be extracted
    local version
    if version=$(python -c "import toml; print(toml.load('pyproject.toml')['project']['version'])" 2>/dev/null); then
        log_success "Version extracted: $version"
    else
        log_error "Could not extract version from pyproject.toml"
        return 1
    fi
    
    return 0
}

main() {
    log_info "Starting build validation for Surrogate Gradient Optimization Lab..."
    echo ""
    
    local total_errors=0
    
    # Run all checks
    check_build_files || ((total_errors += $?))
    echo ""
    
    check_docker_targets || ((total_errors += $?))
    echo ""
    
    check_makefile_targets || ((total_errors += $?))
    echo ""
    
    check_docker_compose_services || ((total_errors += $?))
    echo ""
    
    validate_python_config || ((total_errors += $?))
    echo ""
    
    # Summary
    if [[ $total_errors -eq 0 ]]; then
        log_success "All build validation checks passed! ✅"
        echo ""
        log_info "Build system is properly configured and ready for use."
        echo ""
        log_info "Quick start commands:"
        echo "  make install-dev    # Set up development environment"
        echo "  make test          # Run tests"
        echo "  make docker-build  # Build Docker image"
        echo "  make up-dev        # Start development environment"
        exit 0
    else
        log_error "Build validation failed with $total_errors error(s) ❌"
        echo ""
        log_error "Please fix the above issues before proceeding."
        exit 1
    fi
}

main "$@"