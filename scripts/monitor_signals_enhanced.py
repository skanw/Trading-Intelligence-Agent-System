#!/usr/bin/env python3
"""
Enhanced signal monitoring with incremental processing, checkpointing, and Prometheus integration.
"""
import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from redis import Redis
import pandas as pd
from datetime import datetime, timedelta
from prometheus_client import CollectorRegistry, Gauge, Counter, push_to_gateway
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SignalMonitor:
    """Enhanced signal monitoring with incremental processing and alerting."""
    
    def __init__(self, 
                 redis_url: str = "redis://localhost:6379/0",
                 checkpoint_file: str = "checkpoint.txt",
                 prometheus_gateway: str = "localhost:9091",
                 job_name: str = "signal_monitor"):
        
        self.redis = Redis.from_url(redis_url, decode_responses=True)
        self.checkpoint_file = Path(checkpoint_file)
        self.prometheus_gateway = prometheus_gateway
        self.job_name = job_name
        
        # Prometheus metrics
        self.registry = CollectorRegistry()
        self.signal_age_gauge = Gauge(
            'signal_age_minutes', 
            'Age of latest signal in minutes',
            registry=self.registry
        )
        self.signals_processed_counter = Counter(
            'signals_processed_total',
            'Total number of signals processed',
            registry=self.registry
        )
        self.monitoring_errors_counter = Counter(
            'monitoring_errors_total',
            'Total number of monitoring errors',
            registry=self.registry
        )
        
    def get_last_checkpoint(self) -> str:
        """Get the last processed signal ID from checkpoint file."""
        try:
            if self.checkpoint_file.exists():
                last_id = self.checkpoint_file.read_text().strip()
                logger.info(f"Loaded checkpoint: {last_id}")
                return last_id
            else:
                logger.info("No checkpoint found, starting from beginning")
                return '0-0'
        except Exception as e:
            logger.error(f"Error reading checkpoint: {e}")
            return '0-0'
    
    def save_checkpoint(self, signal_id: str) -> None:
        """Save the last processed signal ID to checkpoint file."""
        try:
            self.checkpoint_file.write_text(signal_id)
            logger.debug(f"Saved checkpoint: {signal_id}")
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
    
    def read_incremental_signals(self, count: int = 100, block_ms: int = 5000) -> list:
        """Read new signals incrementally using XREAD."""
        last_id = self.get_last_checkpoint()
        
        try:
            # Use XREAD to get only new entries since last checkpoint
            entries = self.redis.xread({'signals': last_id}, count=count, block=block_ms)
            
            if not entries:
                logger.debug("No new signals found")
                return []
            
            # Extract signals from Redis response
            signals = []
            for stream_name, stream_entries in entries:
                for entry_id, entry_data in stream_entries:
                    signals.append((entry_id, entry_data))
            
            logger.info(f"Read {len(signals)} new signals")
            return signals
            
        except Exception as e:
            logger.error(f"Error reading signals: {e}")
            self.monitoring_errors_counter.inc()
            return []
    
    def analyze_signal_latency(self, signals: list) -> Dict[str, Any]:
        """Analyze signal latency and return metrics."""
        if not signals:
            # If no new signals, check the latest signal age
            return self._check_latest_signal_age()
        
        latest_signal_id, latest_data = signals[-1]
        
        try:
            # Parse the published timestamp
            pub_ts = pd.to_datetime(latest_data["published_at"])
            now = pd.Timestamp.utcnow()
            age_minutes = (now - pub_ts).total_seconds() / 60
            
            # Update metrics
            self.signal_age_gauge.set(age_minutes)
            self.signals_processed_counter.inc(len(signals))
            
            # Save checkpoint
            self.save_checkpoint(latest_signal_id)
            
            return {
                'latest_signal_age_minutes': age_minutes,
                'signals_processed': len(signals),
                'latest_signal_id': latest_signal_id,
                'latest_signal_time': pub_ts.isoformat(),
                'current_time': now.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing signal latency: {e}")
            self.monitoring_errors_counter.inc()
            return {'error': str(e)}
    
    def _check_latest_signal_age(self) -> Dict[str, Any]:
        """Check the age of the latest signal when no new signals are found."""
        try:
            # Get the most recent signal
            last = self.redis.xrevrange("signals", count=1)
            
            if not last:
                logger.warning("No signals found in Redis stream")
                # Set a high value to trigger alerts
                self.signal_age_gauge.set(999)
                return {'error': 'No signals in Redis stream', 'latest_signal_age_minutes': 999}
            
            signal_id, data = last[0]
            pub_ts = pd.to_datetime(data["published_at"])
            now = pd.Timestamp.utcnow()
            age_minutes = (now - pub_ts).total_seconds() / 60
            
            self.signal_age_gauge.set(age_minutes)
            
            return {
                'latest_signal_age_minutes': age_minutes,
                'signals_processed': 0,
                'latest_signal_id': signal_id,
                'latest_signal_time': pub_ts.isoformat(),
                'current_time': now.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error checking latest signal: {e}")
            self.monitoring_errors_counter.inc()
            self.signal_age_gauge.set(999)
            return {'error': str(e), 'latest_signal_age_minutes': 999}
    
    def push_metrics_to_prometheus(self) -> None:
        """Push metrics to Prometheus pushgateway."""
        try:
            push_to_gateway(
                self.prometheus_gateway, 
                job=self.job_name, 
                registry=self.registry
            )
            logger.debug("Metrics pushed to Prometheus")
        except Exception as e:
            logger.error(f"Error pushing metrics to Prometheus: {e}")
    
    def run_monitoring_cycle(self, alert_threshold_minutes: float = 5.0) -> bool:
        """Run a single monitoring cycle and return success status."""
        logger.info("Starting monitoring cycle")
        
        # Read new signals incrementally
        signals = self.read_incremental_signals()
        
        # Analyze signal latency
        metrics = self.analyze_signal_latency(signals)
        
        # Push metrics to Prometheus
        self.push_metrics_to_prometheus()
        
        # Log results
        if 'error' in metrics:
            logger.error(f"Monitoring error: {metrics['error']}")
            return False
        
        age_minutes = metrics['latest_signal_age_minutes']
        logger.info(f"Latest signal age: {age_minutes:.1f} minutes")
        
        # Check alert threshold (for local alerting if needed)
        if age_minutes > alert_threshold_minutes:
            logger.warning(f"Signal latency exceeded threshold: {age_minutes:.1f} > {alert_threshold_minutes} minutes")
            return False
        
        logger.info("Monitoring cycle completed successfully")
        return True

def main():
    """Main monitoring function with command line arguments."""
    parser = argparse.ArgumentParser(description='Enhanced signal monitoring')
    parser.add_argument('--redis-url', default=os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
                        help='Redis connection URL')
    parser.add_argument('--checkpoint-file', default='checkpoint.txt',
                        help='File to store processing checkpoint')
    parser.add_argument('--prometheus-gateway', default=os.getenv('PROMETHEUS_GATEWAY', 'localhost:9091'),
                        help='Prometheus pushgateway address')
    parser.add_argument('--job-name', default='signal_monitor',
                        help='Prometheus job name')
    parser.add_argument('--alert-threshold', type=float, default=5.0,
                        help='Alert threshold in minutes')
    parser.add_argument('--continuous', action='store_true',
                        help='Run continuously instead of single cycle')
    parser.add_argument('--interval', type=int, default=60,
                        help='Interval between monitoring cycles in seconds (for continuous mode)')
    
    args = parser.parse_args()
    
    # Initialize monitor
    monitor = SignalMonitor(
        redis_url=args.redis_url,
        checkpoint_file=args.checkpoint_file,
        prometheus_gateway=args.prometheus_gateway,
        job_name=args.job_name
    )
    
    if args.continuous:
        logger.info(f"Starting continuous monitoring with {args.interval}s interval")
        while True:
            try:
                success = monitor.run_monitoring_cycle(args.alert_threshold)
                if not success:
                    logger.warning("Monitoring cycle had issues, continuing...")
                
                time.sleep(args.interval)
                
            except KeyboardInterrupt:
                logger.info("Received interrupt signal, shutting down...")
                break
            except Exception as e:
                logger.error(f"Unexpected error in monitoring loop: {e}")
                time.sleep(args.interval)
    else:
        # Single run mode
        success = monitor.run_monitoring_cycle(args.alert_threshold)
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 