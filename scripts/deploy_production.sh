#!/bin/bash

# Production Deployment Script for Trading Intelligence Agent System
# Usage: ./scripts/deploy_production.sh [environment]

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENVIRONMENT="${1:-production}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        error "This script should not be run as root"
        exit 1
    fi
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check available disk space (minimum 10GB)
    available_space=$(df -BG "$PROJECT_DIR" | awk 'NR==2 {print $4}' | sed 's/G//')
    if [[ $available_space -lt 10 ]]; then
        error "Insufficient disk space. At least 10GB required."
        exit 1
    fi
    
    # Check available memory (minimum 8GB)
    available_memory=$(free -g | awk 'NR==2 {print $7}')
    if [[ $available_memory -lt 8 ]]; then
        warning "Low available memory detected. At least 8GB recommended for optimal performance."
    fi
    
    success "Prerequisites check passed"
}

# Setup environment variables
setup_environment() {
    log "Setting up environment for $ENVIRONMENT..."
    
    cd "$PROJECT_DIR"
    
    # Create .env file if it doesn't exist
    if [[ ! -f .env ]]; then
        log "Creating .env file from template..."
        cat > .env << EOF
# Trading System Environment Configuration
TRADING_ENV=$ENVIRONMENT

# Database
POSTGRES_PASSWORD=$(openssl rand -base64 32)
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=trading_system
POSTGRES_USER=trader

# Redis
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_DB=0

# Kafka
KAFKA_BROKERS=kafka:9092

# API Keys (Add your own)
NEWS_API_KEY=your_news_api_key_here
TWITTER_BEARER_TOKEN=your_twitter_bearer_token_here
BLOOMBERG_API_KEY=your_bloomberg_api_key_here
REFINITIV_API_KEY=your_refinitiv_api_key_here
IEX_CLOUD_TOKEN=your_iex_cloud_token_here
ALPHA_VANTAGE_KEY=your_alpha_vantage_key_here

# Monitoring
GRAFANA_PASSWORD=$(openssl rand -base64 16)
DATADOG_API_KEY=your_datadog_api_key_here

# Security
JWT_SECRET_KEY=$(openssl rand -base64 64)

# Logging
LOG_LEVEL=INFO
SENTRY_DSN=your_sentry_dsn_here

# Cloud Configuration (if using cloud deployment)
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_aws_access_key_here
AWS_SECRET_ACCESS_KEY=your_aws_secret_key_here

# Slack notifications (optional)
SLACK_WEBHOOK_URL=your_slack_webhook_url_here
EOF
        warning "Please update the .env file with your actual API keys and configuration"
        success ".env file created"
    else
        log ".env file already exists"
    fi
    
    # Create directories
    mkdir -p logs/{agents,system,monitoring}
    mkdir -p data/{raw,processed,models}
    mkdir -p monitoring/{prometheus,grafana/{dashboards,datasources}}
    
    success "Environment setup completed"
}

# Setup monitoring configuration
setup_monitoring() {
    log "Setting up monitoring configuration..."
    
    # Prometheus configuration
    cat > monitoring/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

scrape_configs:
  - job_name: 'trading-system'
    static_configs:
      - targets: ['localhost:8000', 'localhost:8501']
    metrics_path: /metrics
    scrape_interval: 30s

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']

  - job_name: 'kafka'
    static_configs:
      - targets: ['kafka:9092']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
EOF

    # Grafana datasource configuration
    mkdir -p monitoring/grafana/datasources
    cat > monitoring/grafana/datasources/prometheus.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF

    # Grafana dashboard configuration
    mkdir -p monitoring/grafana/dashboards
    cat > monitoring/grafana/dashboards/trading-system.json << EOF
{
  "dashboard": {
    "id": null,
    "title": "Trading System Overview",
    "uid": "trading-system",
    "version": 1,
    "schemaVersion": 27,
    "panels": [
      {
        "title": "Agent Health Status",
        "type": "stat",
        "targets": [
          {
            "expr": "up{job=\"trading-system\"}",
            "legendFormat": "{{instance}}"
          }
        ]
      },
      {
        "title": "News Processing Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(news_items_processed_total[5m])",
            "legendFormat": "News Items/sec"
          }
        ]
      },
      {
        "title": "Trading Signals Generated",
        "type": "graph", 
        "targets": [
          {
            "expr": "trading_signals_generated_total",
            "legendFormat": "Signals"
          }
        ]
      }
    ]
  }
}
EOF

    # Filebeat configuration
    cat > monitoring/filebeat.yml << EOF
filebeat.inputs:
- type: log
  enabled: true
  paths:
    - /usr/share/filebeat/logs/**/*.log
  fields:
    service: trading-system
  fields_under_root: true

output.elasticsearch:
  hosts: ["elasticsearch:9200"]

setup.kibana:
  host: "kibana:5601"

logging.level: info
EOF

    success "Monitoring configuration completed"
}

# Pull Docker images
pull_images() {
    log "Pulling Docker images..."
    
    cd "$PROJECT_DIR"
    docker-compose pull
    
    success "Docker images pulled successfully"
}

# Build application images
build_images() {
    log "Building application images..."
    
    cd "$PROJECT_DIR"
    docker-compose build --parallel
    
    success "Application images built successfully"
}

# Start infrastructure services first
start_infrastructure() {
    log "Starting infrastructure services..."
    
    cd "$PROJECT_DIR"
    
    # Start core infrastructure
    docker-compose up -d redis postgres zookeeper kafka
    
    # Wait for services to be ready
    log "Waiting for infrastructure services to be ready..."
    
    # Wait for Redis
    log "Waiting for Redis..."
    until docker-compose exec -T redis redis-cli ping 2>/dev/null; do
        sleep 2
    done
    
    # Wait for PostgreSQL
    log "Waiting for PostgreSQL..."
    until docker-compose exec -T postgres pg_isready -U trader -d trading_system 2>/dev/null; do
        sleep 2
    done
    
    # Wait for Kafka
    log "Waiting for Kafka..."
    until docker-compose exec -T kafka kafka-broker-api-versions --bootstrap-server localhost:9092 2>/dev/null; do
        sleep 5
    done
    
    success "Infrastructure services are ready"
}

# Start monitoring services
start_monitoring() {
    log "Starting monitoring services..."
    
    cd "$PROJECT_DIR"
    
    # Start monitoring stack
    docker-compose up -d prometheus grafana elasticsearch kibana
    
    # Wait for services
    log "Waiting for monitoring services to be ready..."
    sleep 30
    
    success "Monitoring services started"
}

# Start application services
start_application() {
    log "Starting application services..."
    
    cd "$PROJECT_DIR"
    
    # Start data pipeline first
    docker-compose up -d data-pipeline
    sleep 10
    
    # Start analysis agents
    docker-compose up -d news-intelligence market-intelligence fundamental-analysis technical-analysis
    sleep 15
    
    # Start management agents
    docker-compose up -d risk-management portfolio-management
    sleep 10
    
    # Start orchestrator
    docker-compose up -d orchestrator
    sleep 10
    
    # Start web services
    docker-compose up -d dashboard api
    
    success "Application services started"
}

# Health check
health_check() {
    log "Performing health checks..."
    
    local failed_checks=0
    
    # Check Redis
    if ! docker-compose exec -T redis redis-cli ping >/dev/null 2>&1; then
        error "Redis health check failed"
        ((failed_checks++))
    else
        success "Redis is healthy"
    fi
    
    # Check PostgreSQL
    if ! docker-compose exec -T postgres pg_isready -U trader -d trading_system >/dev/null 2>&1; then
        error "PostgreSQL health check failed"
        ((failed_checks++))
    else
        success "PostgreSQL is healthy"
    fi
    
    # Check Kafka
    if ! docker-compose exec -T kafka kafka-broker-api-versions --bootstrap-server localhost:9092 >/dev/null 2>&1; then
        error "Kafka health check failed"
        ((failed_checks++))
    else
        success "Kafka is healthy"
    fi
    
    # Check API endpoint
    sleep 5
    if ! curl -f http://localhost:8000/health >/dev/null 2>&1; then
        error "API health check failed"
        ((failed_checks++))
    else
        success "API is healthy"
    fi
    
    # Check Dashboard
    if ! curl -f http://localhost:8501 >/dev/null 2>&1; then
        error "Dashboard health check failed"
        ((failed_checks++))
    else
        success "Dashboard is healthy"
    fi
    
    if [[ $failed_checks -eq 0 ]]; then
        success "All health checks passed"
        return 0
    else
        error "$failed_checks health checks failed"
        return 1
    fi
}

# Display deployment summary
deployment_summary() {
    log "Deployment Summary"
    echo "===================="
    echo "Environment: $ENVIRONMENT"
    echo "Project Directory: $PROJECT_DIR"
    echo ""
    echo "Services:"
    echo "- Dashboard: http://localhost:8501"
    echo "- API: http://localhost:8000"
    echo "- Grafana: http://localhost:3000 (admin/admin)"
    echo "- Prometheus: http://localhost:9090"
    echo "- Kibana: http://localhost:5601"
    echo ""
    echo "Commands:"
    echo "- View logs: docker-compose logs -f [service]"
    echo "- Stop services: docker-compose down"
    echo "- Scale service: docker-compose up -d --scale [service]=N"
    echo ""
    success "Deployment completed successfully!"
}

# Cleanup function
cleanup() {
    log "Cleaning up..."
    cd "$PROJECT_DIR"
    docker-compose down --remove-orphans
}

# Trap cleanup on script exit
trap cleanup EXIT

# Main deployment function
main() {
    log "Starting deployment of Trading Intelligence Agent System"
    log "Environment: $ENVIRONMENT"
    
    check_root
    check_prerequisites
    setup_environment
    setup_monitoring
    pull_images
    build_images
    start_infrastructure
    start_monitoring
    start_application
    
    if health_check; then
        deployment_summary
    else
        error "Deployment failed health checks"
        exit 1
    fi
}

# Run main function
main "$@" 