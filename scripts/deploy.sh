#!/bin/bash

# 🚀 Production Deployment Script for Surrogate Optimization Lab
# Usage: ./scripts/deploy.sh [environment] [options]

set -e

# Configuration
PROJECT_NAME="surrogate-optim"
VERSION=${VERSION:-"latest"}
ENVIRONMENT=${1:-"production"}
DRY_RUN=${DRY_RUN:-false}
BACKUP_ENABLED=${BACKUP_ENABLED:-true}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is required but not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is required but not installed"
        exit 1
    fi
    
    # Check available disk space (at least 10GB)
    AVAILABLE_SPACE=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ "$AVAILABLE_SPACE" -lt 10 ]; then
        log_error "Insufficient disk space. At least 10GB required, found ${AVAILABLE_SPACE}GB"
        exit 1
    fi
    
    # Check memory (at least 8GB)
    TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$TOTAL_MEM" -lt 8 ]; then
        log_warn "Low memory detected (${TOTAL_MEM}GB). At least 8GB recommended"
    fi
    
    log_info "Prerequisites check passed"
}

# Backup current deployment
backup_deployment() {
    if [ "$BACKUP_ENABLED" = "true" ] && [ -f "docker-compose.yml" ]; then
        BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
        log_info "Creating backup in $BACKUP_DIR"
        
        mkdir -p "$BACKUP_DIR"
        
        # Backup configuration
        cp docker-compose.yml "$BACKUP_DIR/"
        cp -r data "$BACKUP_DIR/" 2>/dev/null || true
        cp -r logs "$BACKUP_DIR/" 2>/dev/null || true
        
        # Backup database (if exists)
        if docker ps --format "table {{.Names}}" | grep -q redis; then
            log_info "Backing up Redis data"
            docker exec surrogate-optim-redis redis-cli BGSAVE
            docker cp surrogate-optim-redis:/data/dump.rdb "$BACKUP_DIR/"
        fi
        
        log_info "Backup completed in $BACKUP_DIR"
    fi
}

# Setup environment-specific configuration
setup_environment() {
    log_info "Setting up environment: $ENVIRONMENT"
    
    case $ENVIRONMENT in
        "development")
            COMPOSE_FILE="docker-compose.yml"
            DOCKERFILE="Dockerfile"
            ;;
        "staging")
            COMPOSE_FILE="docker-compose.staging.yml"
            DOCKERFILE="Dockerfile.production"
            ;;
        "production")
            COMPOSE_FILE="docker-compose.production.yml" 
            DOCKERFILE="Dockerfile.production"
            ;;
        "research")
            COMPOSE_FILE="docker-compose.research.yml"
            DOCKERFILE="Dockerfile.production"
            ;;
        *)
            log_error "Unknown environment: $ENVIRONMENT"
            log_info "Available environments: development, staging, production, research"
            exit 1
            ;;
    esac
    
    if [ ! -f "$COMPOSE_FILE" ]; then
        log_error "Compose file not found: $COMPOSE_FILE"
        exit 1
    fi
    
    export COMPOSE_FILE
    export DOCKERFILE
}

# Create required directories
setup_directories() {
    log_info "Setting up directories"
    
    mkdir -p data/{datasets,models,experiments}
    mkdir -p logs/{api,workers,research}
    mkdir -p cache/{functions,models,results}
    mkdir -p results/{experiments,benchmarks,reports}
    mkdir -p monitoring/{prometheus,grafana}
    mkdir -p backups
    
    # Set permissions
    chmod 755 data logs cache results
    
    log_info "Directories created successfully"
}

# Generate configuration files
generate_config() {
    log_info "Generating configuration files"
    
    # Generate monitoring configuration
    cat > monitoring/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "rules/*.yml"

scrape_configs:
  - job_name: 'surrogate-optim'
    static_configs:
      - targets: ['surrogate-optim-api:8000']
    scrape_interval: 30s
    metrics_path: /metrics
    
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
EOF

    # Generate Nginx configuration
    mkdir -p nginx
    cat > nginx/nginx.conf << 'EOF'
events {
    worker_connections 1024;
}

http {
    upstream surrogate_optim {
        server surrogate-optim-api:8000;
    }
    
    server {
        listen 80;
        server_name _;
        
        location /health {
            proxy_pass http://surrogate_optim/health;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }
        
        location /api/ {
            proxy_pass http://surrogate_optim/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Increase timeout for long-running optimizations
            proxy_read_timeout 300s;
            proxy_connect_timeout 75s;
            proxy_send_timeout 300s;
        }
        
        location /static/ {
            alias /var/www/static/;
            expires 1y;
            add_header Cache-Control "public, immutable";
        }
    }
}
EOF

    # Generate environment file
    cat > .env << EOF
# Environment Configuration
ENVIRONMENT=$ENVIRONMENT
VERSION=$VERSION

# Performance Settings
SURROGATE_OPTIM_WORKERS=8
SURROGATE_OPTIM_MEMORY_LIMIT=16GB
SURROGATE_OPTIM_GPU_ENABLED=true
SURROGATE_OPTIM_LOG_LEVEL=INFO

# JAX Configuration
JAX_ENABLE_X64=true
JAX_PLATFORM_NAME=gpu

# Security
JWT_SECRET=$(openssl rand -base64 32)
API_KEY=$(openssl rand -hex 32)

# Database
REDIS_URL=redis://redis:6379/0
REDIS_PASSWORD=$(openssl rand -base64 16)

# Monitoring
PROMETHEUS_URL=http://prometheus:9090
GRAFANA_ADMIN_PASSWORD=$(openssl rand -base64 16)
EOF

    log_info "Configuration files generated"
}

# Build Docker images
build_images() {
    log_info "Building Docker images for $ENVIRONMENT"
    
    if [ "$DRY_RUN" = "true" ]; then
        log_info "[DRY RUN] Would build images with:"
        log_info "  - docker-compose -f $COMPOSE_FILE build"
        return
    fi
    
    # Build images
    docker-compose -f "$COMPOSE_FILE" build --parallel
    
    # Tag images with version
    docker tag "${PROJECT_NAME}_surrogate-optim-api:latest" "${PROJECT_NAME}_surrogate-optim-api:$VERSION"
    
    log_info "Images built successfully"
}

# Deploy services
deploy_services() {
    log_info "Deploying services for $ENVIRONMENT"
    
    if [ "$DRY_RUN" = "true" ]; then
        log_info "[DRY RUN] Would deploy with:"
        log_info "  - docker-compose -f $COMPOSE_FILE up -d"
        return
    fi
    
    # Stop existing services
    log_info "Stopping existing services..."
    docker-compose -f "$COMPOSE_FILE" down --remove-orphans || true
    
    # Start new services
    log_info "Starting new services..."
    docker-compose -f "$COMPOSE_FILE" up -d
    
    log_info "Services deployed successfully"
}

# Wait for services to be healthy
wait_for_health() {
    log_info "Waiting for services to become healthy..."
    
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f http://localhost:8000/health >/dev/null 2>&1; then
            log_info "Health check passed"
            return 0
        fi
        
        log_debug "Health check attempt $attempt/$max_attempts failed, retrying in 10s..."
        sleep 10
        attempt=$((attempt + 1))
    done
    
    log_error "Health check failed after $max_attempts attempts"
    return 1
}

# Run smoke tests
run_smoke_tests() {
    log_info "Running smoke tests..."
    
    # Test API endpoints
    if curl -f http://localhost:8000/health >/dev/null 2>&1; then
        log_info "✓ Health endpoint accessible"
    else
        log_error "✗ Health endpoint failed"
        return 1
    fi
    
    # Test metrics endpoint
    if curl -f http://localhost:8000/metrics >/dev/null 2>&1; then
        log_info "✓ Metrics endpoint accessible"
    else
        log_warn "✗ Metrics endpoint not accessible"
    fi
    
    # Test Prometheus
    if curl -f http://localhost:9090/-/healthy >/dev/null 2>&1; then
        log_info "✓ Prometheus healthy"
    else
        log_warn "✗ Prometheus not healthy"
    fi
    
    # Test Grafana
    if curl -f http://localhost:3000/api/health >/dev/null 2>&1; then
        log_info "✓ Grafana healthy"
    else
        log_warn "✗ Grafana not healthy"
    fi
    
    log_info "Smoke tests completed"
}

# Show deployment status
show_status() {
    log_info "Deployment Status:"
    echo "==================="
    docker-compose -f "$COMPOSE_FILE" ps
    echo
    
    log_info "Service URLs:"
    echo "- API: http://localhost:8000"
    echo "- Health: http://localhost:8000/health"
    echo "- Metrics: http://localhost:8000/metrics"
    echo "- Prometheus: http://localhost:9090"
    echo "- Grafana: http://localhost:3000"
    echo "- Kibana: http://localhost:5601"
    echo
    
    log_info "Log commands:"
    echo "- API logs: docker-compose -f $COMPOSE_FILE logs -f surrogate-optim-api"
    echo "- All logs: docker-compose -f $COMPOSE_FILE logs -f"
    echo
}

# Cleanup function
cleanup() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        log_error "Deployment failed with exit code $exit_code"
        
        log_info "Rolling back..."
        docker-compose -f "$COMPOSE_FILE" down || true
        
        # Restore from backup if available
        if [ "$BACKUP_ENABLED" = "true" ] && [ -d "backups" ]; then
            LATEST_BACKUP=$(ls -t backups/ | head -1)
            if [ -n "$LATEST_BACKUP" ]; then
                log_info "Restoring from backup: $LATEST_BACKUP"
                cp "backups/$LATEST_BACKUP/docker-compose.yml" . || true
                docker-compose up -d || true
            fi
        fi
    fi
}

# Print usage
usage() {
    cat << EOF
Usage: $0 [environment] [options]

Environments:
  development  - Development setup with hot reload
  staging      - Staging environment for testing
  production   - Production environment (default)
  research     - Research setup with GPU support

Options:
  --dry-run           Show what would be done without executing
  --no-backup         Skip backup creation
  --skip-health       Skip health checks
  --skip-tests        Skip smoke tests
  --help              Show this help message

Environment Variables:
  VERSION             Docker image version (default: latest)
  DRY_RUN             Enable dry run mode (default: false)
  BACKUP_ENABLED      Enable backup creation (default: true)

Examples:
  ./scripts/deploy.sh production
  ./scripts/deploy.sh research --dry-run
  VERSION=v1.2.3 ./scripts/deploy.sh staging

EOF
}

# Main deployment function
main() {
    # Parse arguments
    SKIP_HEALTH=false
    SKIP_TESTS=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --no-backup)
                BACKUP_ENABLED=false
                shift
                ;;
            --skip-health)
                SKIP_HEALTH=true
                shift
                ;;
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --help|-h)
                usage
                exit 0
                ;;
            --*)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
            *)
                if [ -z "$ENVIRONMENT" ] || [ "$ENVIRONMENT" = "production" ]; then
                    ENVIRONMENT=$1
                fi
                shift
                ;;
        esac
    done
    
    # Set up cleanup trap
    trap cleanup EXIT
    
    log_info "🚀 Starting deployment for environment: $ENVIRONMENT"
    log_info "Version: $VERSION"
    log_info "Dry run: $DRY_RUN"
    
    # Execute deployment steps
    check_prerequisites
    setup_environment
    backup_deployment
    setup_directories
    generate_config
    build_images
    deploy_services
    
    if [ "$SKIP_HEALTH" = "false" ] && [ "$DRY_RUN" = "false" ]; then
        wait_for_health
    fi
    
    if [ "$SKIP_TESTS" = "false" ] && [ "$DRY_RUN" = "false" ]; then
        run_smoke_tests
    fi
    
    show_status
    
    log_info "🎉 Deployment completed successfully!"
    log_info "Environment: $ENVIRONMENT"
    log_info "Version: $VERSION"
}

# Execute main function if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi