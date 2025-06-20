#!/bin/bash

# Enhanced Signal Monitoring Deployment Script
# This script deploys the improved monitoring system with incremental processing,
# stateful checkpointing, and Prometheus integration.

set -e

echo "🚀 Deploying Enhanced Trading System Monitoring..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install it and try again."
    exit 1
fi

# Create necessary directories
print_status "Creating monitoring directories..."
mkdir -p monitoring/checkpoints
mkdir -p monitoring/grafana/dashboards
mkdir -p monitoring/grafana/datasources
mkdir -p logs/monitoring

# Set permissions for monitoring directories
chmod 755 monitoring/checkpoints
chmod 755 logs/monitoring

# Create environment file if it doesn't exist
if [ ! -f .env ]; then
    print_status "Creating .env file with default values..."
    cat > .env << EOF
# Trading System Environment Variables
TRADING_ENV=production
POSTGRES_PASSWORD=secure_password_change_me
GRAFANA_PASSWORD=admin
REDIS_URL=redis://redis:6379/0
PROMETHEUS_GATEWAY=prometheus-pushgateway:9091

# Alerting Configuration
SMTP_PASSWORD=your_smtp_password
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK
PAGERDUTY_ROUTING_KEY=your_pagerduty_routing_key

# API Keys (set these for production)
NEWS_API_KEY=your_news_api_key
BLOOMBERG_API_KEY=your_bloomberg_api_key
REFINITIV_API_KEY=your_refinitiv_api_key
EOF
    print_warning "Please update .env file with your actual credentials before running in production!"
fi

# Build the monitoring container
print_status "Building monitoring container..."
docker build -f monitoring/Dockerfile.monitoring -t trading-system/signal-monitor:latest .

# Deploy the monitoring stack
print_status "Deploying monitoring stack with Docker Compose..."
docker-compose up -d prometheus prometheus-pushgateway alertmanager node-exporter grafana

# Wait for services to be ready
print_status "Waiting for services to be ready..."
sleep 30

# Check if services are healthy
print_status "Checking service health..."
services=("redis" "prometheus" "prometheus-pushgateway" "alertmanager" "grafana")
for service in "${services[@]}"; do
    if docker-compose ps "$service" | grep -q "Up"; then
        print_status "✅ $service is running"
    else
        print_error "❌ $service is not running"
        exit 1
    fi
done

# Deploy the signal monitor
print_status "Deploying signal monitoring service..."
docker-compose up -d signal-monitor

# Wait for signal monitor to start
sleep 10

# Check signal monitor status
if docker-compose ps signal-monitor | grep -q "Up"; then
    print_status "✅ Signal monitor is running"
else
    print_error "❌ Signal monitor failed to start"
    docker-compose logs signal-monitor
    exit 1
fi

# Create Grafana datasource configuration
print_status "Creating Grafana datasource configuration..."
cat > monitoring/grafana/datasources/prometheus.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
EOF

# Create a basic dashboard configuration
print_status "Creating Grafana dashboard configuration..."
cat > monitoring/grafana/dashboards/dashboard.yml << EOF
apiVersion: 1

providers:
  - name: 'default'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /etc/grafana/provisioning/dashboards
EOF

# Test the monitoring system
print_status "Testing monitoring system..."

# Check if Prometheus is scraping metrics
prometheus_url="http://localhost:9090"
if curl -s "$prometheus_url/api/v1/targets" | grep -q "prometheus-pushgateway"; then
    print_status "✅ Prometheus is scraping pushgateway"
else
    print_warning "⚠️  Prometheus may not be scraping pushgateway yet"
fi

# Check alertmanager
alertmanager_url="http://localhost:9093"
if curl -s "$alertmanager_url/api/v1/status" | grep -q "success"; then
    print_status "✅ Alertmanager is responding"
else
    print_warning "⚠️  Alertmanager may not be ready yet"
fi

# Deployment summary
echo ""
echo "🎉 Enhanced Monitoring System Deployed Successfully!"
echo "==============================================="
echo ""
echo "📊 Access Points:"
echo "  - Grafana Dashboard: http://localhost:3000 (admin/admin)"
echo "  - Prometheus: http://localhost:9090"
echo "  - Alertmanager: http://localhost:9093"
echo "  - Pushgateway: http://localhost:9091"
echo ""
echo "🔧 Monitoring Features:"
echo "  ✅ Incremental signal processing with XREAD"
echo "  ✅ Stateful checkpointing"
echo "  ✅ Prometheus metrics integration"
echo "  ✅ Alertmanager notifications"
echo "  ✅ Containerized deployment"
echo "  ✅ Auto-restart on failure"
echo ""
echo "📋 Next Steps:"
echo "  1. Update .env file with production credentials"
echo "  2. Configure alerting channels (Slack, PagerDuty, Email)"
echo "  3. Set up Grafana dashboards for visualization"
echo "  4. Test alert thresholds and notifications"
echo "  5. Monitor logs: docker-compose logs -f signal-monitor"
echo ""
echo "🚨 Health Check:"
echo "  Run: docker-compose ps"
echo "  Logs: docker-compose logs [service-name]"
echo ""

# Show running services
print_status "Currently running services:"
docker-compose ps

# Show recent logs from signal monitor
print_status "Recent signal monitor logs:"
docker-compose logs --tail=10 signal-monitor

echo ""
print_status "Deployment complete! Monitor the logs and verify everything is working correctly." 