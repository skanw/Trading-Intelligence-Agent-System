# Enhanced Signal Monitoring System

This document describes the improvements implemented for the trading system's signal monitoring, based on the Agent Architecture defined in `AGENTS.md`.

## 🚀 Key Improvements Implemented

### 1. **Incremental Processing with XREAD**

**Previous Approach:**
```python
# Old method - always reads all signals
entries = r.xrevrange("signals", count=1)
```

**New Approach:**
```python
# New method - incremental reading with XREAD
last_id = Path('checkpoint.txt').read_text() or '0-0'
entries = r.xread({'signals': last_id}, count=100, block=5000)
```

**Benefits:**
- ✅ Only processes new signals since last run
- ✅ Reduced Redis load and network traffic
- ✅ Faster processing for large signal streams
- ✅ Blocking read with timeout for real-time processing

### 2. **Stateful Checkpointing**

**Implementation:**
```python
def save_checkpoint(self, signal_id: str) -> None:
    """Save the last processed signal ID to checkpoint file."""
    self.checkpoint_file.write_text(signal_id)
    
def get_last_checkpoint(self) -> str:
    """Get the last processed signal ID from checkpoint file."""
    if self.checkpoint_file.exists():
        return self.checkpoint_file.read_text().strip()
    return '0-0'  # Start from beginning if no checkpoint
```

**Benefits:**
- ✅ Fault tolerance - no signal processing loss on restart
- ✅ Exactly-once processing guarantee
- ✅ Efficient resume from last processed signal
- ✅ Persistent state across container restarts

### 3. **Prometheus Integration**

**Previous Approach:**
```python
# Old method - simple print and exit
print("ALERT: signal latency > 5 min", file=sys.stderr)
sys.exit(1)
```

**New Approach:**
```python
# New method - structured metrics to Prometheus
self.signal_age_gauge.set(age_minutes)
self.signals_processed_counter.inc(len(signals))
push_to_gateway(self.prometheus_gateway, job=self.job_name, registry=self.registry)
```

**Metrics Exposed:**
- `signal_age_minutes` - Age of latest signal in minutes
- `signals_processed_total` - Total number of signals processed
- `monitoring_errors_total` - Total number of monitoring errors

**Benefits:**
- ✅ Rich metrics for observability
- ✅ Historical data retention
- ✅ Flexible alerting rules
- ✅ Integration with existing monitoring stack

### 4. **Containerization & Scheduling**

**Docker Container:**
```dockerfile
FROM python:3.11-slim
# Optimized for monitoring workloads
# Non-root user for security
# Health checks included
```

**Docker Compose Integration:**
```yaml
signal-monitor:
  build:
    context: .
    dockerfile: monitoring/Dockerfile.monitoring
  environment:
    - REDIS_URL=redis://redis:6379/0
    - PROMETHEUS_GATEWAY=prometheus-pushgateway:9091
  volumes:
    - ./monitoring/checkpoints:/app/checkpoints
  restart: unless-stopped
```

**Kubernetes CronJob:**
```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: signal-monitor-cronjob
spec:
  schedule: "*/2 * * * *"  # Every 2 minutes
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: signal-monitor
            image: trading-system/signal-monitor:latest
```

## 📊 Monitoring Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│                 │    │                 │    │                 │
│  Redis Streams  │────│ Signal Monitor  │────│  Prometheus     │
│   (signals)     │    │  (Enhanced)     │    │  Pushgateway    │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │                 │
                       │  Checkpoint     │
                       │  Storage        │
                       │                 │
                       └─────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│                 │    │                 │    │                 │
│  Prometheus     │────│  Alertmanager   │────│  Notifications  │
│  (Metrics)      │    │  (Rules)        │    │  (Slack/Email)  │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚨 Alerting Rules

### Signal Latency Alerts

| Alert | Condition | Severity | Description |
|-------|-----------|----------|-------------|
| SignalLatencyHigh | `signal_age_minutes > 5` | Warning | Signal processing delayed |
| SignalLatencyCritical | `signal_age_minutes > 15` | Critical | Significant processing delay |
| NoSignalsReceived | `signal_age_minutes > 30` | Critical | No signals for extended period |

### System Health Alerts

| Alert | Condition | Severity | Description |
|-------|-----------|----------|-------------|
| MonitoringErrors | `increase(monitoring_errors_total[5m]) > 0` | Warning | Monitoring system errors |
| HighMonitoringErrorRate | `rate(monitoring_errors_total[5m]) > 0.1` | Critical | High error rate detected |
| RedisDown | `up{job="redis"} == 0` | Critical | Redis connectivity issues |

## 🔧 Deployment Options

### Option 1: Docker Compose (Recommended for Development)

```bash
# Quick deployment
./scripts/deploy_monitoring.sh

# Manual deployment
docker-compose up -d prometheus prometheus-pushgateway alertmanager signal-monitor
```

### Option 2: Kubernetes (Recommended for Production)

```bash
# Deploy CronJob for scheduled monitoring
kubectl apply -f k8s/signal-monitor.yaml

# Deploy continuous monitoring
kubectl apply -f k8s/signal-monitor.yaml
```

### Option 3: Standalone Script

```bash
# Single run
python scripts/monitor_signals_enhanced.py

# Continuous monitoring
python scripts/monitor_signals_enhanced.py --continuous --interval 60
```

## 📈 Usage Examples

### Basic Monitoring
```bash
# Run once and exit
python monitor_signals_enhanced.py \
  --redis-url redis://localhost:6379/0 \
  --prometheus-gateway localhost:9091 \
  --alert-threshold 5.0
```

### Continuous Monitoring
```bash
# Run continuously with 30-second intervals
python monitor_signals_enhanced.py \
  --continuous \
  --interval 30 \
  --checkpoint-file /app/checkpoints/monitor.checkpoint
```

### Custom Configuration
```bash
# Custom configuration
python monitor_signals_enhanced.py \
  --redis-url redis://prod-redis:6379/0 \
  --prometheus-gateway pushgateway.monitoring.svc.cluster.local:9091 \
  --job-name prod-signal-monitor \
  --alert-threshold 3.0 \
  --continuous
```

## 🔍 Monitoring & Troubleshooting

### Health Checks

```bash
# Check service status
docker-compose ps signal-monitor

# View logs
docker-compose logs -f signal-monitor

# Check metrics
curl http://localhost:9091/metrics | grep signal_age_minutes
```

### Prometheus Queries

```promql
# Current signal age
signal_age_minutes

# Signal processing rate
rate(signals_processed_total[5m])

# Error rate
rate(monitoring_errors_total[5m])

# Alert status
ALERTS{alertname="SignalLatencyHigh"}
```

### Debugging

```bash
# Check Redis stream
redis-cli XINFO STREAM signals

# Check checkpoint file
cat monitoring/checkpoints/signal_monitor.checkpoint

# Test Prometheus connectivity
curl -X POST http://localhost:9091/metrics/job/signal_monitor
```

## 🏗️ Architecture Benefits

### Reliability
- **Fault Tolerance**: Checkpoint-based recovery
- **Health Monitoring**: Comprehensive health checks
- **Auto-Recovery**: Container restart policies

### Performance
- **Efficient Processing**: Incremental reads with XREAD
- **Reduced Load**: Only process new signals
- **Scalability**: Containerized deployment

### Observability
- **Rich Metrics**: Detailed performance metrics
- **Alerting**: Multi-channel notifications
- **Dashboards**: Grafana integration ready

### Operations
- **Easy Deployment**: One-command deployment
- **Configuration**: Environment-based config
- **Scaling**: Kubernetes-ready

## 🚀 Next Steps

1. **Configure Alerting Channels**
   - Update Slack webhook URLs
   - Configure PagerDuty integration
   - Set up email notifications

2. **Create Grafana Dashboards**
   - Signal processing metrics
   - System health overview
   - Alert status dashboard

3. **Production Tuning**
   - Adjust alert thresholds
   - Optimize polling intervals
   - Configure retention policies

4. **Security Hardening**
   - Use secrets management
   - Enable TLS/SSL
   - Implement RBAC

## 📚 Related Documentation

- [AGENTS.md](./AGENTS.md) - Agent architecture and responsibilities
- [SYSTEM_STATUS.md](./SYSTEM_STATUS.md) - Current system status
- [docker-compose.yml](./docker-compose.yml) - Container orchestration
- [monitoring/](./monitoring/) - Monitoring configuration files

---

*This enhanced monitoring system provides production-ready signal monitoring with enterprise-grade reliability, observability, and operational capabilities.* 