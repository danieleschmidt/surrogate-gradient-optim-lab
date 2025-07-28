#!/bin/bash

# Release script for Surrogate Gradient Optimization Lab
# Handles version bumping, tagging, and release automation

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PROJECT_NAME="surrogate-gradient-optim-lab"

# Default values
RELEASE_TYPE="patch"
DRY_RUN=false
SKIP_TESTS=false
SKIP_DOCKER=false
SKIP_PUBLISH=false
FORCE=false

# =============================================================================
# Functions
# =============================================================================

show_help() {
    cat << EOF
Release Script for Surrogate Gradient Optimization Lab

Usage: $0 [OPTIONS] [RELEASE_TYPE]

Release Types:
    patch      Increment patch version (0.1.0 -> 0.1.1) [default]
    minor      Increment minor version (0.1.0 -> 0.2.0)
    major      Increment major version (0.1.0 -> 1.0.0)
    VERSION    Specific version number (e.g., 1.2.3)

Options:
    --dry-run          Show what would be done without making changes
    --skip-tests       Skip running tests before release
    --skip-docker      Skip building Docker images
    --skip-publish     Skip publishing to PyPI and Docker registry
    --force            Force release even if working directory is dirty
    -h, --help         Show this help message

Examples:
    $0                        # Patch release (0.1.0 -> 0.1.1)
    $0 minor                  # Minor release (0.1.0 -> 0.2.0)
    $0 1.5.0                  # Specific version
    $0 --dry-run major        # Preview major release
    $0 --skip-tests patch     # Quick patch without tests

Workflow:
    1. Check working directory is clean
    2. Run tests (unless --skip-tests)
    3. Update version in pyproject.toml
    4. Update CHANGELOG.md
    5. Create git commit and tag
    6. Build Docker images (unless --skip-docker)
    7. Publish to PyPI and Docker registry (unless --skip-publish)
    8. Push changes and tags to remote

EOF
}

log_info() {
    echo "[INFO] $*"
}

log_warn() {
    echo "[WARN] $*" >&2
}

log_error() {
    echo "[ERROR] $*" >&2
}

log_dry_run() {
    echo "[DRY RUN] $*"
}

get_current_version() {
    cd "$PROJECT_ROOT"
    python -c "import toml; print(toml.load('pyproject.toml')['project']['version'])" 2>/dev/null || echo "0.1.0"
}

increment_version() {
    local current="$1"
    local release_type="$2"
    
    # Parse current version
    local major minor patch
    IFS='.' read -r major minor patch <<< "$current"
    
    case "$release_type" in
        major)
            echo "$((major + 1)).0.0"
            ;;
        minor)
            echo "$major.$((minor + 1)).0"
            ;;
        patch)
            echo "$major.$minor.$((patch + 1))"
            ;;
        *)
            # Assume it's a specific version number
            if [[ "$release_type" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
                echo "$release_type"
            else
                log_error "Invalid version format: $release_type"
                return 1
            fi
            ;;
    esac
}

check_working_directory() {
    if [[ "$FORCE" == "false" ]]; then
        log_info "Checking working directory status..."
        
        cd "$PROJECT_ROOT"
        
        if ! git diff-index --quiet HEAD --; then
            log_error "Working directory is not clean. Commit or stash changes first."
            log_error "Use --force to override this check."
            return 1
        fi
        
        if [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
            log_error "Untracked files found. Add or ignore them first."
            log_error "Use --force to override this check."
            return 1
        fi
        
        log_info "Working directory is clean."
    fi
}

run_tests() {
    if [[ "$SKIP_TESTS" == "false" ]]; then
        log_info "Running tests..."
        
        cd "$PROJECT_ROOT"
        
        if [[ "$DRY_RUN" == "true" ]]; then
            log_dry_run "Would run: make test"
        else
            make test || {
                log_error "Tests failed. Fix issues before releasing."
                return 1
            }
        fi
        
        log_info "Tests passed."
    else
        log_warn "Skipping tests."
    fi
}

update_version() {
    local new_version="$1"
    
    log_info "Updating version to $new_version..."
    
    cd "$PROJECT_ROOT"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_dry_run "Would update version in pyproject.toml to $new_version"
    else
        # Update pyproject.toml
        python -c "
import toml
with open('pyproject.toml', 'r') as f:
    data = toml.load(f)
data['project']['version'] = '$new_version'
with open('pyproject.toml', 'w') as f:
    toml.dump(data, f)
        "
        
        log_info "Updated pyproject.toml with version $new_version"
    fi
}

update_changelog() {
    local new_version="$1"
    local date="$(date +%Y-%m-%d)"
    
    log_info "Updating CHANGELOG.md..."
    
    cd "$PROJECT_ROOT"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_dry_run "Would update CHANGELOG.md with version $new_version"
        return
    fi
    
    if [[ -f "CHANGELOG.md" ]]; then
        # Create temporary file with new entry
        local temp_file="$(mktemp)"
        {
            # Keep header until first version entry
            sed -n '1,/^## \[/p' CHANGELOG.md | head -n -1
            
            # Add new version entry
            echo "## [$new_version] - $date"
            echo ""
            echo "### Added"
            echo "- Release version $new_version"
            echo ""
            
            # Add rest of file
            sed -n '/^## \[/,$p' CHANGELOG.md
        } > "$temp_file"
        
        mv "$temp_file" CHANGELOG.md
        log_info "Updated CHANGELOG.md"
    else
        log_warn "CHANGELOG.md not found, skipping changelog update"
    fi
}

create_git_commit_and_tag() {
    local new_version="$1"
    
    log_info "Creating git commit and tag..."
    
    cd "$PROJECT_ROOT"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_dry_run "Would create commit and tag v$new_version"
        return
    fi
    
    # Add changed files
    git add pyproject.toml CHANGELOG.md
    
    # Create commit
    git commit -m "chore: release version $new_version

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"
    
    # Create tag
    git tag -a "v$new_version" -m "Release version $new_version"
    
    log_info "Created commit and tag v$new_version"
}

build_docker_images() {
    local new_version="$1"
    
    if [[ "$SKIP_DOCKER" == "false" ]]; then
        log_info "Building Docker images..."
        
        if [[ "$DRY_RUN" == "true" ]]; then
            log_dry_run "Would build Docker images for version $new_version"
            return
        fi
        
        # Build production image
        "$SCRIPT_DIR/build.sh" --target production --version "$new_version" --clean
        
        # Build development image
        "$SCRIPT_DIR/build.sh" --target development --version "$new_version"
        
        log_info "Docker images built successfully"
    else
        log_warn "Skipping Docker image builds"
    fi
}

publish_release() {
    local new_version="$1"
    
    if [[ "$SKIP_PUBLISH" == "false" ]]; then
        log_info "Publishing release..."
        
        cd "$PROJECT_ROOT"
        
        if [[ "$DRY_RUN" == "true" ]]; then
            log_dry_run "Would publish to PyPI and Docker registry"
            log_dry_run "Would push git changes and tags"
            return
        fi
        
        # Build Python package
        log_info "Building Python package..."
        python -m build
        
        # Publish to PyPI (requires authentication)
        log_info "Publishing to PyPI..."
        if command -v twine &> /dev/null; then
            twine upload dist/*
            log_info "Published to PyPI"
        else
            log_warn "twine not found, skipping PyPI upload"
        fi
        
        # Push Docker images
        log_info "Pushing Docker images..."
        "$SCRIPT_DIR/build.sh" --target production --version "$new_version" --push
        "$SCRIPT_DIR/build.sh" --target development --version "$new_version" --push
        
        # Push git changes and tags
        log_info "Pushing git changes and tags..."
        git push origin main
        git push origin "v$new_version"
        
        log_info "Release published successfully"
    else
        log_warn "Skipping publication (--skip-publish)"
    fi
}

show_release_summary() {
    local current_version="$1"
    local new_version="$2"
    
    log_info "Release Summary:"
    echo "  Project: $PROJECT_NAME"
    echo "  Current Version: $current_version"
    echo "  New Version: $new_version"
    echo "  Release Type: $RELEASE_TYPE"
    echo "  Dry Run: $DRY_RUN"
    echo "  Skip Tests: $SKIP_TESTS"
    echo "  Skip Docker: $SKIP_DOCKER"
    echo "  Skip Publish: $SKIP_PUBLISH"
    echo ""
}

# =============================================================================
# Main Script
# =============================================================================

main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --skip-docker)
                SKIP_DOCKER=true
                shift
                ;;
            --skip-publish)
                SKIP_PUBLISH=true
                shift
                ;;
            --force)
                FORCE=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            major|minor|patch)
                RELEASE_TYPE="$1"
                shift
                ;;
            [0-9]*.[0-9]*.[0-9]*)
                RELEASE_TYPE="$1"
                shift
                ;;
            *)
                log_error "Unknown option or invalid version: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Get current version
    local current_version
    current_version="$(get_current_version)"
    
    # Calculate new version
    local new_version
    new_version="$(increment_version "$current_version" "$RELEASE_TYPE")"
    
    # Show release summary
    show_release_summary "$current_version" "$new_version"
    
    # Confirm release if not dry run
    if [[ "$DRY_RUN" == "false" ]]; then
        echo -n "Proceed with release? (y/N): "
        read -r confirmation
        if [[ "$confirmation" != "y" && "$confirmation" != "Y" ]]; then
            log_info "Release cancelled."
            exit 0
        fi
    fi
    
    # Execute release steps
    check_working_directory
    run_tests
    update_version "$new_version"
    update_changelog "$new_version"
    create_git_commit_and_tag "$new_version"
    build_docker_images "$new_version"
    publish_release "$new_version"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "Dry run completed. No changes were made."
    else
        log_info "Release $new_version completed successfully!"
        echo ""
        log_info "Next steps:"
        echo "  1. Verify the release on GitHub"
        echo "  2. Check PyPI for package availability"
        echo "  3. Test Docker images from registry"
        echo "  4. Update documentation if needed"
    fi
}

# Execute main function
main "$@"