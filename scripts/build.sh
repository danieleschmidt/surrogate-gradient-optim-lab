#!/bin/bash

# Build script for Surrogate Gradient Optimization Lab
# Provides automated building, testing, and packaging

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PROJECT_NAME="surrogate-gradient-optim-lab"
IMAGE_NAME="surrogate-optim"
REGISTRY="ghcr.io/terragon-labs"

# Default values
TARGET="production"
VERSION="$(cd "$PROJECT_ROOT" && python -c "import toml; print(toml.load('pyproject.toml')['project']['version'])" 2>/dev/null || echo "0.1.0")"
TAG="$VERSION"
PUSH=false
CLEAN=false
NO_CACHE=false
RUN_TESTS=false
PLATFORM="linux/amd64"

# =============================================================================
# Functions
# =============================================================================

show_help() {
    cat << EOF
Build Script for Surrogate Gradient Optimization Lab

Usage: $0 [OPTIONS]

Options:
    -t, --target TARGET     Build target (development|production|gpu) [default: production]
    -v, --version VERSION   Version tag [default: auto-detect from pyproject.toml]
    --tag TAG              Custom tag name [default: same as version]
    -p, --push             Push image to registry after building
    -c, --clean            Clean build cache and intermediate images
    --no-cache             Build without using cache
    --test                 Run tests in container after building
    --platform PLATFORM   Target platform [default: linux/amd64]
    --registry REGISTRY    Docker registry [default: ghcr.io/terragon-labs]
    -h, --help             Show this help message

Examples:
    $0                                    # Build production image
    $0 --target development              # Build development image
    $0 --target gpu --push               # Build and push GPU image
    $0 --version 1.2.3 --push            # Build with custom version
    $0 --clean --no-cache --test         # Clean build with tests
    $0 --platform linux/arm64            # Build for ARM64

Targets:
    development    Development image with all dev tools
    production     Optimized production image
    gpu           GPU-enabled image with CUDA support

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

check_requirements() {
    log_info "Checking requirements..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is required but not installed."
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running or accessible."
        exit 1
    fi
    
    if [[ "$TARGET" == "gpu" ]] && ! docker info | grep -q nvidia; then
        log_warn "GPU target selected but NVIDIA Docker runtime not detected."
        log_warn "GPU functionality may not work properly."
    fi
    
    log_info "Requirements check passed."
}

clean_docker() {
    if [[ "$CLEAN" == "true"]]; then
        log_info "Cleaning Docker build cache and intermediate images..."
        
        # Remove dangling images
        docker image prune -f || true
        
        # Remove build cache
        docker builder prune -f || true
        
        # Remove project-specific images
        docker images --filter="reference=$IMAGE_NAME" --filter="reference=$REGISTRY/$IMAGE_NAME" -q | xargs -r docker rmi -f || true
        
        log_info "Docker cleanup completed."
    fi
}

build_image() {
    local target="$1"
    local tag="$2"
    local full_tag="$REGISTRY/$IMAGE_NAME:$tag"
    
    log_info "Building image: $full_tag (target: $target)"
    
    # Build arguments
    local build_args=()
    build_args+=("--target" "$target")
    build_args+=("--tag" "$IMAGE_NAME:$tag")
    build_args+=("--tag" "$IMAGE_NAME:latest-$target")
    build_args+=("--tag" "$full_tag")
    build_args+=("--tag" "$REGISTRY/$IMAGE_NAME:latest-$target")
    build_args+=("--platform" "$PLATFORM")
    
    # Add metadata labels
    build_args+=("--label" "org.opencontainers.image.version=$VERSION")
    build_args+=("--label" "org.opencontainers.image.created=$(date -u +%Y-%m-%dT%H:%M:%SZ)")
    build_args+=("--label" "org.opencontainers.image.revision=$(git rev-parse HEAD 2>/dev/null || echo 'unknown')")
    build_args+=("--label" "org.opencontainers.image.url=https://github.com/terragon-labs/surrogate-gradient-optim-lab")
    
    if [[ "$NO_CACHE" == "true" ]]; then
        build_args+=("--no-cache")
    fi
    
    # Build the image
    cd "$PROJECT_ROOT"
    docker build "${build_args[@]}" .
    
    log_info "Successfully built: $full_tag"
}

run_tests() {
    if [[ "$RUN_TESTS" == "true" ]]; then
        local test_tag="$IMAGE_NAME:$TAG"
        
        log_info "Running tests in container: $test_tag"
        
        # Run tests in the built container
        docker run --rm \
            --volume "$PROJECT_ROOT:/workspace:ro" \
            --workdir /workspace \
            "$test_tag" \
            bash -c "
                if command -v pytest &> /dev/null; then
                    echo 'Running pytest...'
                    pytest tests/ -v --tb=short
                else
                    echo 'pytest not available, running basic import test...'
                    python -c 'import surrogate_optim; print(\"Import test passed\")'
                fi
            "
        
        log_info "Tests completed successfully."
    fi
}

push_image() {
    if [[ "$PUSH" == "true" ]]; then
        local full_tag="$REGISTRY/$IMAGE_NAME:$TAG"
        local latest_tag="$REGISTRY/$IMAGE_NAME:latest-$TARGET"
        
        log_info "Pushing images to registry..."
        
        # Push versioned tag
        docker push "$full_tag"
        log_info "Pushed: $full_tag"
        
        # Push latest tag for target
        docker push "$latest_tag"
        log_info "Pushed: $latest_tag"
        
        # Push latest tag if this is production
        if [[ "$TARGET" == "production" ]]; then
            docker tag "$full_tag" "$REGISTRY/$IMAGE_NAME:latest"
            docker push "$REGISTRY/$IMAGE_NAME:latest"
            log_info "Pushed: $REGISTRY/$IMAGE_NAME:latest"
        fi
    fi
}

show_build_info() {
    log_info "Build Information:"
    echo "  Project: $PROJECT_NAME"
    echo "  Target: $TARGET"
    echo "  Version: $VERSION"
    echo "  Tag: $TAG"
    echo "  Platform: $PLATFORM"
    echo "  Registry: $REGISTRY"
    echo "  Push: $PUSH"
    echo "  Clean: $CLEAN"
    echo "  No Cache: $NO_CACHE"
    echo "  Run Tests: $RUN_TESTS"
    echo ""
}

# =============================================================================
# Main Script
# =============================================================================

main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -t|--target)
                TARGET="$2"
                shift 2
                ;;
            -v|--version)
                VERSION="$2"
                TAG="$2"
                shift 2
                ;;
            --tag)
                TAG="$2"
                shift 2
                ;;
            -p|--push)
                PUSH=true
                shift
                ;;
            -c|--clean)
                CLEAN=true
                shift
                ;;
            --no-cache)
                NO_CACHE=true
                shift
                ;;
            --test)
                RUN_TESTS=true
                shift
                ;;
            --platform)
                PLATFORM="$2"
                shift 2
                ;;
            --registry)
                REGISTRY="$2"
                shift 2
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Validate target
    case "$TARGET" in
        development|production|gpu)
            ;;
        *)
            log_error "Invalid target: $TARGET. Must be one of: development, production, gpu"
            exit 1
            ;;
    esac
    
    # Show build information
    show_build_info
    
    # Execute build steps
    check_requirements
    clean_docker
    build_image "$TARGET" "$TAG"
    run_tests
    push_image
    
    log_info "Build completed successfully!"
    
    # Show usage information
    echo ""
    log_info "Usage examples:"
    echo "  docker run --rm -it $IMAGE_NAME:$TAG"
    echo "  docker-compose up surrogate-optim-$TARGET"
    
    if [[ "$TARGET" == "development" ]]; then
        echo "  docker run --rm -it -v \$(pwd):/workspace $IMAGE_NAME:$TAG jupyter lab --ip=0.0.0.0"
    fi
}

# Execute main function
main "$@"