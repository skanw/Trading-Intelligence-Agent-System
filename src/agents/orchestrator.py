"""
Orchestration Agent - Central Coordination and Workflow Management
"""

import asyncio
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import heapq
import logging

from .base_agent import BaseAgent, AgentMessage, AgentSignal
from ..config import config


@dataclass
class WorkflowTask:
    """Workflow task definition"""
    task_id: str
    task_type: str
    target_agent: str
    message_type: str
    data: Dict[str, Any]
    priority: int
    dependencies: List[str]
    scheduled_time: datetime
    timeout: int  # seconds
    retry_count: int
    max_retries: int
    status: str  # 'PENDING', 'RUNNING', 'COMPLETED', 'FAILED'
    created_time: datetime
    started_time: Optional[datetime]
    completed_time: Optional[datetime]
    result: Optional[Dict[str, Any]]
    error: Optional[str]


@dataclass
class SignalConflict:
    """Signal conflict between agents"""
    conflict_id: str
    symbol: str
    conflicting_signals: List[Dict[str, Any]]
    resolution: Optional[str]
    confidence: float
    timestamp: datetime


@dataclass
class SystemStatus:
    """Overall system status"""
    timestamp: datetime
    active_agents: Dict[str, Dict[str, Any]]
    message_queue_size: int
    workflow_tasks_pending: int
    workflow_tasks_running: int
    signal_conflicts: int
    system_health: str  # 'HEALTHY', 'DEGRADED', 'CRITICAL'
    performance_metrics: Dict[str, float]
    alerts: List[Dict[str, Any]]


@dataclass
class IntegratedReport:
    """Integrated investment committee report"""
    timestamp: datetime
    market_regime: Dict[str, Any]
    top_signals: List[Dict[str, Any]]
    risk_summary: Dict[str, Any]
    portfolio_recommendations: List[Dict[str, Any]]
    execution_summary: Dict[str, Any]
    key_risks: List[str]
    action_items: List[str]
    next_review: datetime


class OrchestrationAgent(BaseAgent):
    """
    Orchestration Agent for coordinating the entire trading system
    """
    
    def __init__(self, agent_id: str = 'orchestrator-agent', config_override: Optional[Dict] = None):
        super().__init__(agent_id, config_override)
        
        # Agent registry
        self.registered_agents = {
            'market-intel-agent': {'status': 'UNKNOWN', 'last_heartbeat': None, 'health': 'UNKNOWN'},
            'news-intel-agent': {'status': 'UNKNOWN', 'last_heartbeat': None, 'health': 'UNKNOWN'},
            'fundamental-agent': {'status': 'UNKNOWN', 'last_heartbeat': None, 'health': 'UNKNOWN'},
            'technical-agent': {'status': 'UNKNOWN', 'last_heartbeat': None, 'health': 'UNKNOWN'},
            'risk-mgmt-agent': {'status': 'UNKNOWN', 'last_heartbeat': None, 'health': 'UNKNOWN'},
            'execution-agent': {'status': 'UNKNOWN', 'last_heartbeat': None, 'health': 'UNKNOWN'},
            'portfolio-mgmt-agent': {'status': 'UNKNOWN', 'last_heartbeat': None, 'health': 'UNKNOWN'}
        }
        
        # Message prioritization
        self.message_priority_queue = []
        self.message_queue_lock = asyncio.Lock()
        
        # Workflow management
        self.workflow_tasks: Dict[str, WorkflowTask] = {}
        self.task_dependencies: Dict[str, Set[str]] = defaultdict(set)
        
        # Signal aggregation
        self.active_signals: Dict[str, List[Dict[str, Any]]] = defaultdict(list)  # By symbol
        self.signal_history: deque = deque(maxlen=1000)
        self.signal_conflicts: List[SignalConflict] = []
        
        # System monitoring
        self.system_alerts: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, float] = {}
        
        # Coordination parameters
        self.heartbeat_timeout = 300  # 5 minutes
        self.signal_timeout = 3600  # 1 hour
        self.max_concurrent_workflows = 10
        
        # Report generation
        self.last_integrated_report = None
        self.report_generation_interval = 1800  # 30 minutes
        
    async def agent_initialize(self):
        """Initialize orchestration agent"""
        try:
            # Start agent health monitoring
            asyncio.create_task(self._monitor_agent_health())
            
            # Start workflow processor
            asyncio.create_task(self._process_workflows())
            
            # Start signal aggregation
            asyncio.create_task(self._aggregate_signals())
            
            # Start report generation
            asyncio.create_task(self._generate_integrated_reports())
            
            # Send initialization messages to all agents
            await self._initialize_agent_communication()
            
            self.logger.info("Orchestration Agent initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Orchestration Agent: {e}")
            raise
    
    async def execute(self):
        """Main execution logic - coordinate system operations"""
        try:
            # Process high-priority messages
            await self._process_priority_messages()
            
            # Check system health
            system_status = await self._assess_system_health()
            
            # Handle critical situations
            if system_status.system_health == 'CRITICAL':
                await self._handle_critical_situation(system_status)
            
            # Clean up old data
            await self._cleanup_old_data()
            
            # Update metrics
            await self._update_performance_metrics()
            
            self.logger.info(f"Orchestration completed - System: {system_status.system_health}")
            
        except Exception as e:
            self.logger.error(f"Error in orchestration execution: {e}")
            self.metrics['errors'] += 1
    
    async def _monitor_agent_health(self):
        """Monitor health of all registered agents"""
        while True:
            try:
                current_time = datetime.now(timezone.utc)
                
                for agent_id, status in self.registered_agents.items():
                    if status['last_heartbeat']:
                        time_since_heartbeat = (current_time - status['last_heartbeat']).total_seconds()
                        
                        if time_since_heartbeat > self.heartbeat_timeout:
                            if status['health'] != 'UNHEALTHY':
                                status['health'] = 'UNHEALTHY'
                                await self._handle_unhealthy_agent(agent_id)
                        elif time_since_heartbeat > self.heartbeat_timeout / 2:
                            status['health'] = 'DEGRADED'
                        else:
                            status['health'] = 'HEALTHY'
                    
                # Send health check requests
                for agent_id in self.registered_agents.keys():
                    try:
                        await self.send_message(
                            agent_id,
                            'health_check',
                            {'timestamp': current_time.isoformat()},
                            priority=5
                        )
                    except Exception as e:
                        self.logger.warning(f"Failed to send health check to {agent_id}: {e}")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in agent health monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _handle_unhealthy_agent(self, agent_id: str):
        """Handle unhealthy agent situation"""
        try:
            self.logger.warning(f"Agent {agent_id} is unhealthy")
            
            # Create system alert
            alert = {
                'type': 'AGENT_UNHEALTHY',
                'agent_id': agent_id,
                'severity': 'HIGH',
                'message': f"Agent {agent_id} has not responded to health checks",
                'timestamp': datetime.now(timezone.utc),
                'recommended_action': 'Investigate agent status and restart if necessary'
            }
            
            self.system_alerts.append(alert)
            
            # Notify other agents if this is a critical agent
            critical_agents = ['risk-mgmt-agent', 'execution-agent', 'portfolio-mgmt-agent']
            if agent_id in critical_agents:
                for other_agent in self.registered_agents.keys():
                    if other_agent != agent_id:
                        await self.send_message(
                            other_agent,
                            'critical_agent_down',
                            {'failed_agent': agent_id, 'alert': alert},
                            priority=1
                        )
            
        except Exception as e:
            self.logger.error(f"Error handling unhealthy agent {agent_id}: {e}")
    
    async def _process_priority_messages(self):
        """Process high-priority messages from the queue"""
        try:
            async with self.message_queue_lock:
                processed_count = 0
                max_process = 20  # Process up to 20 messages per cycle
                
                while self.message_priority_queue and processed_count < max_process:
                    # Get highest priority message
                    priority, timestamp, message_data = heapq.heappop(self.message_priority_queue)
                    
                    # Process the message
                    await self._process_coordinated_message(message_data)
                    processed_count += 1
                    
            if processed_count > 0:
                self.logger.info(f"Processed {processed_count} priority messages")
                
        except Exception as e:
            self.logger.error(f"Error processing priority messages: {e}")
    
    async def _process_coordinated_message(self, message_data: Dict[str, Any]):
        """Process a message that requires coordination"""
        try:
            message_type = message_data.get('type')
            
            if message_type == 'signal_conflict':
                await self._resolve_signal_conflict(message_data)
            elif message_type == 'risk_alert':
                await self._handle_risk_alert(message_data)
            elif message_type == 'market_alert':
                await self._handle_market_alert(message_data)
            elif message_type == 'execution_notification':
                await self._handle_execution_notification(message_data)
            elif message_type == 'workflow_request':
                await self._handle_workflow_request(message_data)
            
        except Exception as e:
            self.logger.error(f"Error processing coordinated message: {e}")
    
    async def _resolve_signal_conflict(self, conflict_data: Dict[str, Any]):
        """Resolve conflicts between trading signals"""
        try:
            symbol = conflict_data.get('symbol')
            signals = conflict_data.get('signals', [])
            
            if not symbol or not signals:
                return
            
            # Analyze conflicting signals
            buy_signals = [s for s in signals if s.get('signal_type') == 'BUY']
            sell_signals = [s for s in signals if s.get('signal_type') == 'SELL']
            
            # Weight signals by confidence and strength
            buy_weight = sum(s.get('confidence', 0.5) * s.get('strength', 0.5) for s in buy_signals)
            sell_weight = sum(s.get('confidence', 0.5) * s.get('strength', 0.5) for s in sell_signals)
            
            # Determine resolution
            if buy_weight > sell_weight * 1.2:  # 20% threshold
                resolution = 'BUY'
                confidence = buy_weight / (buy_weight + sell_weight)
            elif sell_weight > buy_weight * 1.2:
                resolution = 'SELL'
                confidence = sell_weight / (buy_weight + sell_weight)
            else:
                resolution = 'HOLD'
                confidence = 0.5
            
            # Create conflict record
            conflict = SignalConflict(
                conflict_id=f"conflict_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                symbol=symbol,
                conflicting_signals=signals,
                resolution=resolution,
                confidence=confidence,
                timestamp=datetime.now(timezone.utc)
            )
            
            self.signal_conflicts.append(conflict)
            
            # Send resolution to portfolio management
            await self.send_message(
                'portfolio-mgmt-agent',
                'signal_conflict_resolution',
                {
                    'symbol': symbol,
                    'resolution': resolution,
                    'confidence': confidence,
                    'conflicting_signals': signals
                },
                priority=2
            )
            
            self.logger.info(f"Resolved signal conflict for {symbol}: {resolution} (confidence: {confidence:.2f})")
            
        except Exception as e:
            self.logger.error(f"Error resolving signal conflict: {e}")
    
    async def _handle_risk_alert(self, alert_data: Dict[str, Any]):
        """Handle risk management alerts"""
        try:
            alert_type = alert_data.get('alert_type')
            severity = alert_data.get('severity')
            
            # Escalate critical risk alerts
            if severity in ['CRITICAL', 'HIGH']:
                # Notify all relevant agents
                for agent_id in ['portfolio-mgmt-agent', 'execution-agent']:
                    await self.send_message(
                        agent_id,
                        'critical_risk_alert',
                        alert_data,
                        priority=1
                    )
                
                # If it's a position concentration issue, suggest rebalancing
                if alert_type == 'POSITION_CONCENTRATION':
                    await self.send_message(
                        'portfolio-mgmt-agent',
                        'rebalancing_request',
                        {
                            'reason': 'Risk limit breach',
                            'alert': alert_data
                        },
                        priority=2
                    )
            
            # Log alert for reporting
            self.system_alerts.append({
                'timestamp': datetime.now(timezone.utc),
                'type': 'RISK_ALERT',
                'data': alert_data,
                'handled': True
            })
            
        except Exception as e:
            self.logger.error(f"Error handling risk alert: {e}")
    
    async def _handle_market_alert(self, alert_data: Dict[str, Any]):
        """Handle market intelligence alerts"""
        try:
            alert_type = alert_data.get('type')
            severity = alert_data.get('severity')
            
            # For regime changes, notify all agents
            if alert_type == 'REGIME_CHANGE':
                regime_data = alert_data.get('data', {})
                
                # Broadcast to all agents
                for agent_id in self.registered_agents.keys():
                    if agent_id != 'market-intel-agent':  # Don't send back to source
                        await self.send_message(
                            agent_id,
                            'market_regime_update',
                            regime_data,
                            priority=2
                        )
            
            # For high volatility, suggest defensive positioning
            elif alert_type == 'HIGH_VOLATILITY':
                await self.send_message(
                    'portfolio-mgmt-agent',
                    'volatility_adjustment_request',
                    alert_data,
                    priority=3
                )
                
                await self.send_message(
                    'risk-mgmt-agent',
                    'increase_monitoring',
                    alert_data,
                    priority=3
                )
            
        except Exception as e:
            self.logger.error(f"Error handling market alert: {e}")
    
    async def _handle_execution_notification(self, execution_data: Dict[str, Any]):
        """Handle trade execution notifications"""
        try:
            order = execution_data.get('order', {})
            fill = execution_data.get('fill', {})
            
            symbol = order.get('symbol')
            if symbol:
                # Update position tracking
                await self.send_message(
                    'risk-mgmt-agent',
                    'update_position',
                    {
                        'symbol': symbol,
                        'fill': fill,
                        'order': order
                    },
                    priority=3
                )
                
                # Notify fundamental and technical agents for potential re-analysis
                if fill.get('quantity', 0) > 1000:  # Large trade
                    for agent_id in ['fundamental-agent', 'technical-agent']:
                        await self.send_message(
                            agent_id,
                            'large_trade_notification',
                            {
                                'symbol': symbol,
                                'quantity': fill.get('quantity'),
                                'price': fill.get('price')
                            },
                            priority=5
                        )
            
        except Exception as e:
            self.logger.error(f"Error handling execution notification: {e}")
    
    async def _handle_workflow_request(self, workflow_data: Dict[str, Any]):
        """Handle workflow execution requests"""
        try:
            workflow_type = workflow_data.get('workflow_type')
            
            if workflow_type == 'SYMBOL_ANALYSIS':
                await self._execute_symbol_analysis_workflow(workflow_data)
            elif workflow_type == 'PORTFOLIO_REBALANCE':
                await self._execute_rebalancing_workflow(workflow_data)
            elif workflow_type == 'RISK_ASSESSMENT':
                await self._execute_risk_assessment_workflow(workflow_data)
            
        except Exception as e:
            self.logger.error(f"Error handling workflow request: {e}")
    
    async def _execute_symbol_analysis_workflow(self, workflow_data: Dict[str, Any]):
        """Execute comprehensive symbol analysis workflow"""
        try:
            symbol = workflow_data.get('symbol')
            if not symbol:
                return
            
            # Create workflow tasks
            tasks = [
                ('fundamental-agent', 'request_fundamental_analysis'),
                ('technical-agent', 'request_technical_analysis'),
                ('news-intel-agent', 'request_news_analysis')
            ]
            
            # Execute tasks in parallel
            task_results = {}
            for agent_id, message_type in tasks:
                try:
                    result = await asyncio.wait_for(
                        self._send_and_wait_for_response(
                            agent_id, 
                            message_type, 
                            {'symbol': symbol}
                        ),
                        timeout=30
                    )
                    task_results[agent_id] = result
                except asyncio.TimeoutError:
                    self.logger.warning(f"Timeout waiting for {agent_id} analysis of {symbol}")
                    task_results[agent_id] = None
            
            # Aggregate results
            analysis_summary = {
                'symbol': symbol,
                'timestamp': datetime.now(timezone.utc),
                'fundamental': task_results.get('fundamental-agent'),
                'technical': task_results.get('technical-agent'),
                'news': task_results.get('news-intel-agent')
            }
            
            # Send to portfolio management
            await self.send_message(
                'portfolio-mgmt-agent',
                'integrated_analysis',
                analysis_summary,
                priority=3
            )
            
        except Exception as e:
            self.logger.error(f"Error executing symbol analysis workflow: {e}")
    
    async def _send_and_wait_for_response(self, agent_id: str, message_type: str, 
                                        data: Dict[str, Any], timeout: int = 30) -> Optional[Dict[str, Any]]:
        """Send message and wait for response"""
        try:
            correlation_id = str(asyncio.current_task())
            
            await self.send_message(agent_id, message_type, data, correlation_id=correlation_id)
            
            # Wait for response (simplified - in production would use proper async waiting)
            start_time = datetime.now()
            while (datetime.now() - start_time).seconds < timeout:
                # Check for response in cache or message queue
                response = self.get_cached_data(f'response_{correlation_id}')
                if response:
                    return response
                await asyncio.sleep(0.1)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error waiting for response from {agent_id}: {e}")
            return None
    
    async def _process_workflows(self):
        """Process workflow tasks"""
        while True:
            try:
                # Process pending workflows
                current_time = datetime.now(timezone.utc)
                completed_tasks = []
                
                for task in self.workflow_tasks:
                    if task.status == 'PENDING' and task.scheduled_time <= current_time:
                        # Check if dependencies are met
                        dependencies_met = all(
                            any(t.task_id == dep_id and t.status == 'COMPLETED' for t in self.workflow_tasks)
                            for dep_id in task.dependencies
                        )
                        
                        if dependencies_met:
                            # Execute task
                            task.status = 'RUNNING'
                            task.started_time = current_time
                            
                            try:
                                await self.send_message(
                                    task.target_agent,
                                    task.message_type,
                                    task.data,
                                    priority=task.priority
                                )
                                
                                task.status = 'COMPLETED'
                                task.completed_time = current_time
                                completed_tasks.append(task)
                                
                            except Exception as e:
                                task.retry_count += 1
                                if task.retry_count >= task.max_retries:
                                    task.status = 'FAILED'
                                    task.error = str(e)
                                else:
                                    task.status = 'PENDING'
                                    task.scheduled_time = current_time + timedelta(seconds=30)  # Retry in 30 seconds
                
                # Clean up completed tasks older than 1 hour
                cutoff_time = current_time - timedelta(hours=1)
                self.workflow_tasks = [
                    t for t in self.workflow_tasks 
                    if not (t.status in ['COMPLETED', 'FAILED'] and 
                           t.completed_time and t.completed_time < cutoff_time)
                ]
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in workflow processing: {e}")
                await asyncio.sleep(10)

    async def _aggregate_signals(self):
        """Aggregate and analyze signals from all agents"""
        while True:
            try:
                current_time = datetime.now(timezone.utc)
                
                # Clean up old signals
                for symbol in list(self.active_signals.keys()):
                    self.active_signals[symbol] = [
                        signal for signal in self.active_signals[symbol]
                        if (current_time - datetime.fromisoformat(signal.get('timestamp', current_time.isoformat()))).seconds < self.signal_timeout
                    ]
                    
                    if not self.active_signals[symbol]:
                        del self.active_signals[symbol]
                
                # Check for signal conflicts
                for symbol, signals in self.active_signals.items():
                    if len(signals) > 1:
                        signal_types = set(signal.get('signal_type') for signal in signals)
                        if len(signal_types) > 1 and 'HOLD' not in signal_types:
                            # We have conflicting signals
                            await self._process_coordinated_message({
                                'type': 'signal_conflict',
                                'symbol': symbol,
                                'signals': signals
                            })
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in signal aggregation: {e}")
                await asyncio.sleep(30)
    
    async def _generate_integrated_reports(self):
        """Generate integrated investment committee reports"""
        while True:
            try:
                await asyncio.sleep(self.report_generation_interval)
                
                # Gather data from all agents
                market_regime = self.get_cached_data('market_regime') or {}
                portfolio_risk = self.get_cached_data('portfolio_risk') or {}
                execution_report = self.get_cached_data('execution_report') or {}
                
                # Get top signals
                top_signals = []
                for symbol, signals in self.active_signals.items():
                    if signals:
                        best_signal = max(signals, key=lambda s: s.get('confidence', 0) * s.get('strength', 0))
                        top_signals.append({
                            'symbol': symbol,
                            'signal': best_signal
                        })
                
                top_signals = sorted(top_signals, 
                                   key=lambda x: x['signal'].get('confidence', 0) * x['signal'].get('strength', 0), 
                                   reverse=True)[:10]
                
                # Generate recommendations
                portfolio_recommendations = await self._generate_portfolio_recommendations(
                    market_regime, portfolio_risk, top_signals
                )
                
                # Create integrated report
                report = IntegratedReport(
                    timestamp=datetime.now(timezone.utc),
                    market_regime=market_regime,
                    top_signals=top_signals,
                    risk_summary=self._summarize_risk_metrics(portfolio_risk),
                    portfolio_recommendations=portfolio_recommendations,
                    execution_summary=self._summarize_execution_metrics(execution_report),
                    key_risks=self._identify_key_risks(),
                    action_items=self._generate_action_items(),
                    next_review=datetime.now(timezone.utc) + timedelta(minutes=self.report_generation_interval//60)
                )
                
                self.last_integrated_report = report
                
                # Cache the report
                self.cache_data('integrated_report', asdict(report), expiry=self.report_generation_interval)
                
                self.logger.info("Generated integrated investment committee report")
                
            except Exception as e:
                self.logger.error(f"Error generating integrated reports: {e}")
    
    async def _generate_portfolio_recommendations(self, market_regime: Dict, portfolio_risk: Dict, 
                                                top_signals: List[Dict]) -> List[Dict[str, Any]]:
        """Generate portfolio recommendations based on current conditions"""
        recommendations = []
        
        try:
            # Market regime based recommendations
            regime_type = market_regime.get('regime_type', 'NEUTRAL')
            
            if regime_type == 'RISK_OFF':
                recommendations.append({
                    'type': 'DEFENSIVE_POSITIONING',
                    'description': 'Consider increasing defensive positions and reducing risk exposure',
                    'rationale': 'Market regime indicates risk-off environment',
                    'priority': 'HIGH'
                })
            elif regime_type == 'RISK_ON':
                recommendations.append({
                    'type': 'GROWTH_POSITIONING',
                    'description': 'Consider increasing growth positions and cyclical exposure',
                    'rationale': 'Market regime indicates risk-on environment',
                    'priority': 'MEDIUM'
                })
            
            # Risk-based recommendations
            concentration_risk = portfolio_risk.get('concentration_risk', 0)
            if concentration_risk > 0.15:  # 15%
                recommendations.append({
                    'type': 'DIVERSIFICATION',
                    'description': f'Reduce concentration risk (currently {concentration_risk:.1%})',
                    'rationale': 'Single position concentration exceeds comfort level',
                    'priority': 'HIGH'
                })
            
            # Signal-based recommendations
            strong_buy_signals = [s for s in top_signals if s['signal'].get('signal_type') == 'BUY' 
                                and s['signal'].get('strength', 0) > 0.7]
            
            if strong_buy_signals:
                recommendations.append({
                    'type': 'BUY_OPPORTUNITIES',
                    'description': f'Consider {len(strong_buy_signals)} high-conviction buy opportunities',
                    'rationale': 'Strong fundamental and/or technical signals',
                    'priority': 'MEDIUM',
                    'symbols': [s['symbol'] for s in strong_buy_signals[:5]]
                })
            
        except Exception as e:
            self.logger.error(f"Error generating portfolio recommendations: {e}")
        
        return recommendations
    
    def _summarize_risk_metrics(self, portfolio_risk: Dict) -> Dict[str, Any]:
        """Summarize key risk metrics"""
        return {
            'portfolio_var': portfolio_risk.get('portfolio_var_1d', 0),
            'concentration_risk': portfolio_risk.get('concentration_risk', 0),
            'leverage': portfolio_risk.get('leverage', 0),
            'limit_breaches': len(portfolio_risk.get('limit_breaches', [])),
            'overall_health': 'HEALTHY' if len(portfolio_risk.get('limit_breaches', [])) == 0 else 'ATTENTION_NEEDED'
        }
    
    def _summarize_execution_metrics(self, execution_report: Dict) -> Dict[str, Any]:
        """Summarize key execution metrics"""
        return {
            'total_orders': execution_report.get('total_orders', 0),
            'total_volume': execution_report.get('total_volume', 0),
            'avg_slippage': execution_report.get('avg_slippage', 0),
            'execution_quality': 'GOOD' if execution_report.get('avg_slippage', 0) < 5 else 'NEEDS_IMPROVEMENT'
        }
    
    def _identify_key_risks(self) -> List[str]:
        """Identify key risks from system alerts"""
        risks = []
        
        recent_alerts = [alert for alert in self.system_alerts 
                        if (datetime.now(timezone.utc) - alert['timestamp']).seconds < 3600]
        
        for alert in recent_alerts:
            if alert.get('type') == 'RISK_ALERT':
                risks.append(f"Risk Alert: {alert['data'].get('alert_type', 'Unknown')}")
            elif alert.get('type') == 'AGENT_UNHEALTHY':
                risks.append(f"System Risk: {alert['agent_id']} is unhealthy")
        
        # Add generic risks
        if not risks:
            risks.append("Market volatility")
            risks.append("Concentration risk")
        
        return list(set(risks))  # Remove duplicates
    
    def _generate_action_items(self) -> List[str]:
        """Generate action items based on current state"""
        actions = []
        
        # Check for unhealthy agents
        unhealthy_agents = [agent_id for agent_id, status in self.registered_agents.items() 
                          if status['health'] == 'UNHEALTHY']
        
        if unhealthy_agents:
            actions.append(f"Investigate and restart unhealthy agents: {', '.join(unhealthy_agents)}")
        
        # Check for unresolved conflicts
        recent_conflicts = [c for c in self.signal_conflicts 
                          if (datetime.now(timezone.utc) - c.timestamp).seconds < 3600]
        
        if len(recent_conflicts) > 5:
            actions.append("Review signal generation logic - high number of conflicts")
        
        # Default actions
        if not actions:
            actions.append("Continue normal operations")
            actions.append("Monitor market conditions")
        
        return actions
    
    async def _assess_system_health(self) -> SystemStatus:
        """Assess overall system health"""
        try:
            current_time = datetime.now(timezone.utc)
            
            # Count healthy agents
            healthy_agents = sum(1 for status in self.registered_agents.values() 
                               if status['health'] == 'HEALTHY')
            total_agents = len(self.registered_agents)
            
            # Determine system health
            if healthy_agents == total_agents:
                system_health = 'HEALTHY'
            elif healthy_agents >= total_agents * 0.8:  # 80%
                system_health = 'DEGRADED'
            else:
                system_health = 'CRITICAL'
            
            # Get performance metrics
            recent_alerts = [alert for alert in self.system_alerts 
                           if (current_time - alert['timestamp']).seconds < 3600]
            
            return SystemStatus(
                timestamp=current_time,
                active_agents={agent_id: status for agent_id, status in self.registered_agents.items()},
                message_queue_size=len(self.message_priority_queue),
                workflow_tasks_pending=len([t for t in self.workflow_tasks.values() if t.status == 'PENDING']),
                workflow_tasks_running=len([t for t in self.workflow_tasks.values() if t.status == 'RUNNING']),
                signal_conflicts=len(self.signal_conflicts),
                system_health=system_health,
                performance_metrics=self.performance_metrics,
                alerts=recent_alerts
            )
            
        except Exception as e:
            self.logger.error(f"Error assessing system health: {e}")
            return SystemStatus(
                timestamp=datetime.now(timezone.utc),
                active_agents={},
                message_queue_size=0,
                workflow_tasks_pending=0,
                workflow_tasks_running=0,
                signal_conflicts=0,
                system_health='UNKNOWN',
                performance_metrics={},
                alerts=[]
            )
    
    async def _handle_critical_situation(self, system_status: SystemStatus):
        """Handle critical system situations"""
        try:
            self.logger.critical(f"System in critical state: {system_status.system_health}")
            
            # Stop non-essential operations
            await self._emergency_mode()
            
            # Notify external systems
            await self._send_critical_alert(system_status)
            
        except Exception as e:
            self.logger.error(f"Error handling critical situation: {e}")
    
    async def _emergency_mode(self):
        """Enter emergency mode - stop non-essential operations"""
        try:
            # Cancel non-critical workflows
            for task in self.workflow_tasks.values():
                if task.status in ['PENDING', 'RUNNING'] and task.priority > 3:
                    task.status = 'CANCELLED'
            
            # Send emergency stop to execution agent
            await self.send_message(
                'execution-agent',
                'emergency_stop',
                {'reason': 'Critical system state'},
                priority=1
            )
            
            self.logger.warning("Entered emergency mode")
            
        except Exception as e:
            self.logger.error(f"Error entering emergency mode: {e}")
    
    async def _send_critical_alert(self, system_status: SystemStatus):
        """Send critical system alert to external monitoring"""
        try:
            alert = {
                'type': 'CRITICAL_SYSTEM_ALERT',
                'timestamp': system_status.timestamp.isoformat(),
                'system_health': system_status.system_health,
                'unhealthy_agents': [agent_id for agent_id, status in system_status.active_agents.items() 
                                   if status['health'] != 'HEALTHY'],
                'message': 'Trading system in critical state - immediate attention required'
            }
            
            # In production, this would send to external monitoring systems
            self.logger.critical(f"CRITICAL ALERT: {alert}")
            
        except Exception as e:
            self.logger.error(f"Error sending critical alert: {e}")
    
    async def _cleanup_old_data(self):
        """Clean up old data and maintain system performance"""
        try:
            current_time = datetime.now(timezone.utc)
            
            # Clean up old alerts
            self.system_alerts = [alert for alert in self.system_alerts 
                                if (current_time - alert['timestamp']).seconds < 86400]  # 24 hours
            
            # Clean up old signal conflicts
            self.signal_conflicts = [conflict for conflict in self.signal_conflicts 
                                   if (current_time - conflict.timestamp).seconds < 86400]
            
            # Clean up completed workflow tasks
            completed_tasks = [task_id for task_id, task in self.workflow_tasks.items() 
                             if task.status in ['COMPLETED', 'FAILED'] and 
                             task.completed_time and 
                             (current_time - task.completed_time).seconds > 3600]  # 1 hour
            
            for task_id in completed_tasks:
                del self.workflow_tasks[task_id]
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")
    
    async def _update_performance_metrics(self):
        """Update system performance metrics"""
        try:
            # Calculate message processing rate
            self.performance_metrics['message_queue_size'] = len(self.message_priority_queue)
            
            # Calculate agent health percentage
            healthy_count = sum(1 for status in self.registered_agents.values() 
                              if status['health'] == 'HEALTHY')
            self.performance_metrics['agent_health_percentage'] = (healthy_count / len(self.registered_agents)) * 100
            
            # Calculate signal conflict rate
            recent_conflicts = [c for c in self.signal_conflicts 
                              if (datetime.now(timezone.utc) - c.timestamp).seconds < 3600]
            self.performance_metrics['signal_conflicts_per_hour'] = len(recent_conflicts)
            
            # Cache metrics
            self.cache_data('orchestrator_metrics', self.performance_metrics, expiry=300)
            
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
    
    async def _initialize_agent_communication(self):
        """Initialize communication with all registered agents"""
        try:
            init_message = {
                'orchestrator_id': self.agent_id,
                'initialization_time': datetime.now(timezone.utc).isoformat(),
                'system_config': {
                    'heartbeat_interval': 60,
                    'signal_timeout': self.signal_timeout,
                    'priority_levels': [1, 2, 3, 4, 5]
                }
            }
            
            for agent_id in self.registered_agents.keys():
                await self.send_message(
                    agent_id,
                    'orchestrator_init',
                    init_message,
                    priority=2
                )
                
        except Exception as e:
            self.logger.error(f"Error initializing agent communication: {e}")
    
    async def handle_message(self, message: AgentMessage):
        """Handle incoming messages from other agents"""
        try:
            if message.message_type == 'heartbeat':
                # Update agent health
                if message.agent_id in self.registered_agents:
                    self.registered_agents[message.agent_id]['last_heartbeat'] = datetime.now(timezone.utc)
                    self.registered_agents[message.agent_id]['status'] = 'ACTIVE'
            
            elif message.message_type == 'signal_generated':
                # Add to active signals
                symbol = message.data.get('symbol')
                if symbol:
                    signal_data = message.data.copy()
                    signal_data['source_agent'] = message.agent_id
                    signal_data['timestamp'] = datetime.now(timezone.utc).isoformat()
                    
                    self.active_signals[symbol].append(signal_data)
                    self.signal_history.append(signal_data)
            
            elif message.message_type in ['risk_alert', 'market_alert', 'execution_notification']:
                # Add to priority queue for coordination
                priority = message.priority or 3
                timestamp = datetime.now()
                
                async with self.message_queue_lock:
                    heapq.heappush(self.message_priority_queue, 
                                 (priority, timestamp, {
                                     'type': message.message_type,
                                     'data': message.data,
                                     'source_agent': message.agent_id
                                 }))
            
            elif message.message_type == 'request_integrated_report':
                # Provide integrated report
                if self.last_integrated_report:
                    await self.send_message(
                        message.agent_id,
                        'integrated_report_response',
                        asdict(self.last_integrated_report),
                        correlation_id=message.correlation_id
                    )
            
            elif message.message_type == 'request_system_status':
                # Provide system status
                system_status = await self._assess_system_health()
                await self.send_message(
                    message.agent_id,
                    'system_status_response',
                    asdict(system_status),
                    correlation_id=message.correlation_id
                )
                
        except Exception as e:
            self.logger.error(f"Error handling message from {message.agent_id}: {e}")
    
    async def agent_cleanup(self):
        """Cleanup orchestration agent resources"""
        try:
            # Send shutdown notification to all agents
            for agent_id in self.registered_agents.keys():
                try:
                    await self.send_message(
                        agent_id,
                        'system_shutdown',
                        {'reason': 'Orchestrator cleanup'},
                        priority=1
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to notify {agent_id} of shutdown: {e}")
            
            # Clear all data structures
            self.active_signals.clear()
            self.workflow_tasks.clear()
            self.signal_conflicts.clear()
            self.system_alerts.clear()
            
            self.logger.info("Orchestration Agent cleaned up successfully")
            
        except Exception as e:
            self.logger.error(f"Error during orchestrator cleanup: {e}") 