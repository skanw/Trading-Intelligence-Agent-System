"""
Risk Management Agent - Portfolio Risk Monitoring and Position Sizing
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

from .base_agent import BaseAgent, AgentMessage, AgentSignal
from ..config import config


@dataclass
class PositionRisk:
    """Individual position risk metrics"""
    symbol: str
    position_size: float  # Dollar amount
    weight: float  # Portfolio weight
    volatility: float  # Annualized volatility
    var_1d: float  # 1-day Value at Risk
    var_5d: float  # 5-day Value at Risk
    beta: float  # Market beta
    correlation_spy: float  # Correlation with SPY
    max_drawdown: float
    sharpe_ratio: float
    stop_loss_level: Optional[float]
    risk_score: float  # 0-100 risk score
    timestamp: datetime


@dataclass
class PortfolioRisk:
    """Overall portfolio risk metrics"""
    total_value: float
    total_positions: int
    cash_position: float
    leverage: float
    portfolio_var_1d: float
    portfolio_var_5d: float
    portfolio_volatility: float
    portfolio_beta: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    concentration_risk: float  # Largest position weight
    sector_concentration: Dict[str, float]
    correlation_matrix: Dict[str, Dict[str, float]]
    risk_limits: Dict[str, Dict[str, float]]
    limit_breaches: List[Dict[str, Any]]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RiskAlert:
    """Risk management alert"""
    alert_type: str
    severity: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    message: str
    symbol: Optional[str]
    current_value: float
    limit_value: float
    recommended_action: str
    timestamp: datetime


@dataclass
class PositionSize:
    """Position sizing recommendation"""
    symbol: str
    recommended_size: float  # Dollar amount
    max_size: float  # Maximum allowed size
    sizing_method: str  # 'KELLY', 'VOLATILITY', 'EQUAL_WEIGHT', 'RISK_PARITY'
    risk_adjusted: bool
    confidence: float
    rationale: str
    timestamp: datetime


class RiskManagementAgent(BaseAgent):
    """
    Risk Management Agent for portfolio risk monitoring and position sizing
    """
    
    def __init__(self, agent_id: str = 'risk-mgmt-agent', config_override: Optional[Dict] = None):
        super().__init__(agent_id, config_override)
        
        # Risk parameters
        self.max_portfolio_var = 0.02  # 2% daily VaR
        self.max_position_weight = 0.10  # 10% max position
        self.max_sector_weight = 0.25  # 25% max sector
        self.max_leverage = 1.0  # No leverage by default
        self.min_cash_position = 0.05  # 5% minimum cash
        self.max_correlation = 0.7  # Maximum position correlation
        
        # Override with config if available
        if hasattr(self.config, 'get') and self.config:
            self.max_portfolio_var = self.config.get('max_portfolio_var', self.max_portfolio_var)
            self.max_position_weight = self.config.get('max_position_weight', self.max_position_weight)
            self.max_sector_weight = self.config.get('max_sector_weight', self.max_sector_weight)
            self.max_leverage = self.config.get('max_leverage', self.max_leverage)
            self.min_cash_position = self.config.get('min_cash_position', self.min_cash_position)
            self.max_correlation = self.config.get('max_correlation', self.max_correlation)
        
        # Portfolio tracking
        self.current_positions = {}
        self.position_history = {}
        self.correlation_matrix = {}
        self.portfolio_value = 1000000.0  # Default $1M portfolio
        
        # Risk limits configuration
        self.risk_limits = {
            'position_limits': {
                'max_weight': self.max_position_weight,
                'max_var_contribution': 0.05,  # 5% max VaR contribution
                'max_volatility': 0.50  # 50% max annualized volatility
            },
            'portfolio_limits': {
                'max_var_1d': self.max_portfolio_var,
                'max_leverage': self.max_leverage,
                'min_cash': self.min_cash_position,
                'max_drawdown': 0.15  # 15% maximum drawdown
            },
            'concentration_limits': {
                'max_sector_weight': self.max_sector_weight,
                'max_single_position': self.max_position_weight,
                'max_correlation': self.max_correlation
            }
        }
        
        # Position sizing methods
        self.sizing_methods = ['KELLY', 'VOLATILITY', 'EQUAL_WEIGHT', 'RISK_PARITY']
        
        # Risk monitoring
        self.risk_alerts = []
        self.breach_history = []
        
    async def agent_initialize(self):
        """Initialize risk management agent"""
        try:
            # Load current portfolio positions (would come from portfolio DB)
            await self._load_current_positions()
            
            # Initialize risk limits
            await self._initialize_risk_limits()
            
            self.logger.info("Risk Management Agent initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Risk Management Agent: {e}")
            raise
    
    async def execute(self):
        """Main execution logic - monitor portfolio risk"""
        try:
            # Update current positions from portfolio
            await self._update_positions()
            
            # Calculate portfolio risk metrics
            portfolio_risk = await self._calculate_portfolio_risk()
            
            # Check risk limits and generate alerts
            alerts = await self._check_risk_limits(portfolio_risk)
            
            # Send risk alerts
            for alert in alerts:
                await self._send_risk_alert(alert)
            
            # Cache risk data
            self._cache_risk_data(portfolio_risk)
            
            # Generate risk report
            await self._generate_risk_report(portfolio_risk)
            
            self.logger.info(f"Risk monitoring completed - {len(alerts)} alerts generated")
            
        except Exception as e:
            self.logger.error(f"Error in risk management execution: {e}")
            self.metrics['errors'] += 1
    
    async def _load_current_positions(self):
        """Load current portfolio positions"""
        try:
            # In production, this would load from portfolio database
            # For demo, using sample positions
            
            self.current_positions = {
                'AAPL': {
                    'shares': 100,
                    'avg_price': 150.0,
                    'current_price': 155.0,
                    'market_value': 15500,
                    'sector': 'Technology',
                    'entry_date': datetime.now() - timedelta(days=30)
                },
                'MSFT': {
                    'shares': 50,
                    'avg_price': 300.0,
                    'current_price': 310.0,
                    'market_value': 15500,
                    'sector': 'Technology',
                    'entry_date': datetime.now() - timedelta(days=20)
                },
                'SPY': {
                    'shares': 200,
                    'avg_price': 400.0,
                    'current_price': 410.0,
                    'market_value': 82000,
                    'sector': 'Diversified',
                    'entry_date': datetime.now() - timedelta(days=60)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error loading current positions: {e}")
    
    async def _update_positions(self):
        """Update position prices and values"""
        try:
            # This would normally fetch current prices from market data
            # For demo, we'll use cached or simulate small price movements
            
            for symbol in self.current_positions:
                position = self.current_positions[symbol]
                # Simulate small price movement
                current_price = position['current_price'] * (1 + np.random.normal(0, 0.01))
                position['current_price'] = current_price
                position['market_value'] = position['shares'] * current_price
                position['unrealized_pnl'] = (current_price - position['avg_price']) * position['shares']
                position['unrealized_pnl_pct'] = (current_price - position['avg_price']) / position['avg_price']
                
        except Exception as e:
            self.logger.error(f"Error updating positions: {e}")
    
    async def _calculate_portfolio_risk(self) -> PortfolioRisk:
        """Calculate comprehensive portfolio risk metrics"""
        try:
            # Calculate total portfolio value
            total_value = sum(pos['market_value'] for pos in self.current_positions.values())
            cash_position = max(0, self.portfolio_value - total_value)
            
            # Position-level risk calculations
            position_risks = {}
            for symbol, position in self.current_positions.items():
                position_risk = await self._calculate_position_risk(symbol, position, total_value)
                position_risks[symbol] = position_risk
            
            # Portfolio-level calculations
            portfolio_var_1d = self._calculate_portfolio_var(position_risks, 1)
            portfolio_var_5d = self._calculate_portfolio_var(position_risks, 5)
            
            # Portfolio volatility (simplified)
            portfolio_volatility = np.sqrt(sum(
                (pos['market_value'] / total_value) ** 2 * (risk.volatility ** 2)
                for symbol, pos in self.current_positions.items()
                for risk in [position_risks[symbol]]
            ))
            
            # Portfolio beta
            portfolio_beta = sum(
                (pos['market_value'] / total_value) * position_risks[symbol].beta
                for symbol, pos in self.current_positions.items()
            )
            
            # Concentration metrics
            largest_position = max(pos['market_value'] for pos in self.current_positions.values())
            concentration_risk = largest_position / total_value
            
            # Sector concentration
            sector_exposure = {}
            for position in self.current_positions.values():
                sector = position.get('sector', 'Unknown')
                sector_exposure[sector] = sector_exposure.get(sector, 0) + position['market_value']
            
            sector_concentration = {
                sector: value / total_value 
                for sector, value in sector_exposure.items()
            }
            
            # Risk limit checks
            limit_breaches = self._check_limit_breaches(
                total_value, concentration_risk, sector_concentration, 
                portfolio_var_1d, cash_position / self.portfolio_value
            )
            
            return PortfolioRisk(
                total_value=total_value,
                total_positions=len(self.current_positions),
                cash_position=cash_position,
                leverage=total_value / self.portfolio_value,
                portfolio_var_1d=portfolio_var_1d,
                portfolio_var_5d=portfolio_var_5d,
                portfolio_volatility=portfolio_volatility,
                portfolio_beta=portfolio_beta,
                max_drawdown=0.0,  # Would calculate from historical data
                sharpe_ratio=0.0,  # Would calculate from returns
                sortino_ratio=0.0,  # Would calculate from returns
                concentration_risk=concentration_risk,
                sector_concentration=sector_concentration,
                correlation_matrix={},  # Would calculate from price data
                risk_limits=self.risk_limits,
                limit_breaches=limit_breaches,
                timestamp=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio risk: {e}")
            return self._get_default_portfolio_risk()
    
    async def _calculate_position_risk(self, symbol: str, position: Dict, total_value: float) -> PositionRisk:
        """Calculate risk metrics for individual position"""
        try:
            market_value = position['market_value']
            weight = market_value / total_value
            
            # Get volatility from technical analysis agent or calculate
            volatility = await self._get_position_volatility(symbol)
            
            # Calculate VaR (simplified normal distribution assumption)
            var_1d = market_value * volatility / np.sqrt(252) * 1.65  # 95% confidence
            var_5d = market_value * volatility / np.sqrt(252/5) * 1.65
            
            # Get beta (simplified)
            beta = await self._get_position_beta(symbol)
            
            # Risk score (0-100)
            risk_score = min(100, weight * 100 + volatility * 50 + abs(beta - 1) * 25)
            
            return PositionRisk(
                symbol=symbol,
                position_size=market_value,
                weight=weight,
                volatility=volatility,
                var_1d=var_1d,
                var_5d=var_5d,
                beta=beta,
                correlation_spy=0.5,  # Simplified
                max_drawdown=0.0,  # Would calculate from historical data
                sharpe_ratio=0.0,  # Would calculate from returns
                stop_loss_level=None,  # Would set based on technical analysis
                risk_score=risk_score,
                timestamp=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating position risk for {symbol}: {e}")
            return PositionRisk(
                symbol=symbol,
                position_size=0.0,
                weight=0.0,
                volatility=0.2,
                var_1d=0.0,
                var_5d=0.0,
                beta=1.0,
                correlation_spy=0.5,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                stop_loss_level=None,
                risk_score=50.0,
                timestamp=datetime.now(timezone.utc)
            )
    
    async def _get_position_volatility(self, symbol: str) -> float:
        """Get position volatility from technical analysis or estimate"""
        try:
            # Try to get from technical analysis agent
            risk_data = self.get_cached_data(f'risk_metrics_{symbol}')
            if risk_data and 'volatility' in risk_data:
                return risk_data['volatility']
            
            # Default volatility estimates by asset type
            volatility_estimates = {
                'SPY': 0.16,
                'AAPL': 0.25,
                'MSFT': 0.22,
                'GOOGL': 0.28,
                'AMZN': 0.30,
                'TSLA': 0.50
            }
            
            return volatility_estimates.get(symbol, 0.25)  # Default 25%
            
        except Exception as e:
            self.logger.error(f"Error getting volatility for {symbol}: {e}")
            return 0.25
    
    async def _get_position_beta(self, symbol: str) -> float:
        """Get position beta from technical analysis or estimate"""
        try:
            # Try to get from technical analysis agent
            risk_data = self.get_cached_data(f'risk_metrics_{symbol}')
            if risk_data and 'beta' in risk_data:
                return risk_data['beta']
            
            # Default beta estimates
            beta_estimates = {
                'SPY': 1.0,
                'AAPL': 1.2,
                'MSFT': 0.9,
                'GOOGL': 1.1,
                'AMZN': 1.3,
                'TSLA': 2.0
            }
            
            return beta_estimates.get(symbol, 1.0)  # Default beta of 1.0
            
        except Exception as e:
            self.logger.error(f"Error getting beta for {symbol}: {e}")
            return 1.0
    
    def _calculate_portfolio_var(self, position_risks: Dict[str, PositionRisk], days: int) -> float:
        """Calculate portfolio-level Value at Risk"""
        try:
            # Simplified VaR calculation (assumes zero correlation)
            # In practice, would use full covariance matrix
            
            individual_vars = [risk.var_1d if days == 1 else risk.var_5d for risk in position_risks.values()]
            
            # Portfolio VaR (simplified as square root of sum of squares)
            portfolio_var = np.sqrt(sum(var ** 2 for var in individual_vars))
            
            return portfolio_var
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio VaR: {e}")
            return 0.0
    
    def _check_limit_breaches(self, total_value: float, concentration_risk: float, 
                            sector_concentration: Dict[str, float], portfolio_var: float, 
                            cash_ratio: float) -> List[Dict[str, Any]]:
        """Check for risk limit breaches"""
        breaches = []
        
        try:
            # Position concentration breach
            if concentration_risk > self.risk_limits['concentration_limits']['max_single_position']:
                breaches.append({
                    'type': 'POSITION_CONCENTRATION',
                    'current': concentration_risk,
                    'limit': self.risk_limits['concentration_limits']['max_single_position'],
                    'severity': 'HIGH'
                })
            
            # Sector concentration breach
            for sector, weight in sector_concentration.items():
                if weight > self.risk_limits['concentration_limits']['max_sector_weight']:
                    breaches.append({
                        'type': 'SECTOR_CONCENTRATION',
                        'sector': sector,
                        'current': weight,
                        'limit': self.risk_limits['concentration_limits']['max_sector_weight'],
                        'severity': 'MEDIUM'
                    })
            
            # Portfolio VaR breach
            if portfolio_var > self.risk_limits['portfolio_limits']['max_var_1d'] * total_value:
                breaches.append({
                    'type': 'PORTFOLIO_VAR',
                    'current': portfolio_var / total_value,
                    'limit': self.risk_limits['portfolio_limits']['max_var_1d'],
                    'severity': 'HIGH'
                })
            
            # Cash position breach
            if cash_ratio < self.risk_limits['portfolio_limits']['min_cash']:
                breaches.append({
                    'type': 'CASH_POSITION',
                    'current': cash_ratio,
                    'limit': self.risk_limits['portfolio_limits']['min_cash'],
                    'severity': 'MEDIUM'
                })
            
        except Exception as e:
            self.logger.error(f"Error checking limit breaches: {e}")
        
        return breaches
    
    async def _check_risk_limits(self, portfolio_risk: PortfolioRisk) -> List[RiskAlert]:
        """Check risk limits and generate alerts"""
        alerts = []
        
        try:
            for breach in portfolio_risk.limit_breaches:
                alert = RiskAlert(
                    alert_type=breach['type'],
                    severity=breach['severity'],
                    message=self._generate_breach_message(breach),
                    symbol=breach.get('symbol'),
                    current_value=breach['current'],
                    limit_value=breach['limit'],
                    recommended_action=self._get_recommended_action(breach),
                    timestamp=datetime.now(timezone.utc)
                )
                alerts.append(alert)
            
            # Add additional risk checks
            if portfolio_risk.portfolio_var_1d > portfolio_risk.total_value * 0.03:  # 3% VaR warning
                alerts.append(RiskAlert(
                    alert_type='HIGH_PORTFOLIO_RISK',
                    severity='MEDIUM',
                    message=f"Portfolio VaR is {portfolio_risk.portfolio_var_1d/portfolio_risk.total_value:.1%}",
                    symbol=None,
                    current_value=portfolio_risk.portfolio_var_1d/portfolio_risk.total_value,
                    limit_value=0.02,
                    recommended_action="Consider reducing position sizes or adding hedges",
                    timestamp=datetime.now(timezone.utc)
                ))
            
        except Exception as e:
            self.logger.error(f"Error checking risk limits: {e}")
        
        return alerts
    
    def _generate_breach_message(self, breach: Dict[str, Any]) -> str:
        """Generate human-readable breach message"""
        breach_type = breach['type']
        current = breach['current']
        limit = breach['limit']
        
        messages = {
            'POSITION_CONCENTRATION': f"Single position concentration {current:.1%} exceeds limit of {limit:.1%}",
            'SECTOR_CONCENTRATION': f"Sector {breach.get('sector', 'Unknown')} concentration {current:.1%} exceeds limit of {limit:.1%}",
            'PORTFOLIO_VAR': f"Portfolio VaR {current:.1%} exceeds limit of {limit:.1%}",
            'CASH_POSITION': f"Cash position {current:.1%} below minimum of {limit:.1%}"
        }
        
        return messages.get(breach_type, f"Risk limit breach: {breach_type}")
    
    def _get_recommended_action(self, breach: Dict[str, Any]) -> str:
        """Get recommended action for breach"""
        actions = {
            'POSITION_CONCENTRATION': "Reduce largest position or add diversifying positions",
            'SECTOR_CONCENTRATION': f"Reduce exposure to {breach.get('sector', 'sector')} or add other sectors",
            'PORTFOLIO_VAR': "Reduce position sizes, add hedges, or increase cash",
            'CASH_POSITION': "Sell positions to increase cash reserves"
        }
        
        return actions.get(breach['type'], "Review risk exposure and take appropriate action")
    
    async def _send_risk_alert(self, alert: RiskAlert):
        """Send risk alert to relevant agents"""
        try:
            # Send to portfolio management
            await self.send_message(
                'portfolio-mgmt-agent',
                'risk_alert',
                asdict(alert),
                priority=1 if alert.severity == 'CRITICAL' else 2
            )
            
            # Send to orchestrator
            await self.send_message(
                'orchestrator-agent',
                'risk_alert',
                asdict(alert),
                priority=1 if alert.severity == 'CRITICAL' else 2
            )
            
            # Send to execution agent for critical alerts
            if alert.severity in ['CRITICAL', 'HIGH']:
                await self.send_message(
                    'execution-agent',
                    'risk_alert',
                    asdict(alert),
                    priority=1
                )
                
        except Exception as e:
            self.logger.error(f"Error sending risk alert: {e}")
    
    def _cache_risk_data(self, portfolio_risk: PortfolioRisk):
        """Cache risk data for other agents"""
        try:
            # Cache portfolio risk
            self.cache_data('portfolio_risk', portfolio_risk.to_dict(), expiry=300)  # 5 minutes
            
            # Cache key metrics
            self.cache_data('portfolio_var', portfolio_risk.portfolio_var_1d, expiry=300)
            self.cache_data('concentration_risk', portfolio_risk.concentration_risk, expiry=600)
            self.cache_data('sector_concentration', portfolio_risk.sector_concentration, expiry=600)
            
        except Exception as e:
            self.logger.error(f"Error caching risk data: {e}")
    
    async def _generate_risk_report(self, portfolio_risk: PortfolioRisk):
        """Generate comprehensive risk report"""
        try:
            report = {
                'timestamp': portfolio_risk.timestamp.isoformat(),
                'portfolio_summary': {
                    'total_value': portfolio_risk.total_value,
                    'positions': portfolio_risk.total_positions,
                    'cash': portfolio_risk.cash_position,
                    'leverage': portfolio_risk.leverage
                },
                'risk_metrics': {
                    'var_1d': portfolio_risk.portfolio_var_1d,
                    'var_5d': portfolio_risk.portfolio_var_5d,
                    'volatility': portfolio_risk.portfolio_volatility,
                    'beta': portfolio_risk.portfolio_beta
                },
                'concentration': {
                    'largest_position': portfolio_risk.concentration_risk,
                    'sector_breakdown': portfolio_risk.sector_concentration
                },
                'limit_breaches': portfolio_risk.limit_breaches
            }
            
            # Cache the report
            self.cache_data('risk_report', report, expiry=1800)  # 30 minutes
            
        except Exception as e:
            self.logger.error(f"Error generating risk report: {e}")
    
    async def calculate_position_size(self, symbol: str, signal_strength: float, 
                                   current_price: float, method: str = 'VOLATILITY') -> PositionSize:
        """Calculate optimal position size"""
        try:
            volatility = await self._get_position_volatility(symbol)
            
            if method == 'KELLY':
                # Kelly Criterion (simplified)
                win_rate = 0.55  # Would come from historical analysis
                avg_win = 0.08   # Would come from historical analysis
                avg_loss = 0.05  # Would come from historical analysis
                
                kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
                recommended_size = self.portfolio_value * kelly_fraction * signal_strength
                
            elif method == 'VOLATILITY':
                # Volatility-based position sizing
                target_risk = 0.01  # 1% risk per position
                position_risk = volatility / np.sqrt(252)  # Daily volatility
                recommended_size = (self.portfolio_value * target_risk) / position_risk
                
            elif method == 'EQUAL_WEIGHT':
                # Equal weight
                target_positions = 20  # Target 20 positions
                recommended_size = self.portfolio_value / target_positions
                
            else:  # RISK_PARITY
                # Risk parity (simplified)
                target_risk_contribution = 0.05  # 5% risk contribution
                recommended_size = (self.portfolio_value * target_risk_contribution) / volatility
            
            # Apply position limits
            max_size = self.portfolio_value * self.max_position_weight
            recommended_size = min(recommended_size, max_size)
            
            # Adjust for signal strength
            recommended_size *= signal_strength
            
            return PositionSize(
                symbol=symbol,
                recommended_size=recommended_size,
                max_size=max_size,
                sizing_method=method,
                risk_adjusted=True,
                confidence=0.8,
                rationale=f"Based on {method} method with {volatility:.1%} volatility",
                timestamp=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating position size for {symbol}: {e}")
            return PositionSize(
                symbol=symbol,
                recommended_size=0.0,
                max_size=0.0,
                sizing_method=method,
                risk_adjusted=False,
                confidence=0.0,
                rationale="Error in calculation",
                timestamp=datetime.now(timezone.utc)
            )
    
    def _get_default_portfolio_risk(self) -> PortfolioRisk:
        """Get default portfolio risk when calculation fails"""
        return PortfolioRisk(
            total_value=0.0,
            total_positions=0,
            cash_position=0.0,
            leverage=0.0,
            portfolio_var_1d=0.0,
            portfolio_var_5d=0.0,
            portfolio_volatility=0.0,
            portfolio_beta=1.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            concentration_risk=0.0,
            sector_concentration={},
            correlation_matrix={},
            risk_limits=self.risk_limits,
            limit_breaches=[],
            timestamp=datetime.now(timezone.utc)
        )
    
    async def _initialize_risk_limits(self):
        """Initialize risk limits from configuration"""
        try:
            # Load custom risk limits if available
            custom_limits = config.get('risk_limits', {})
            if custom_limits:
                self.risk_limits.update(custom_limits)
                
        except Exception as e:
            self.logger.error(f"Error initializing risk limits: {e}")
    
    async def handle_message(self, message: AgentMessage):
        """Handle incoming messages from other agents"""
        try:
            if message.message_type == 'position_size_request':
                symbol = message.data.get('symbol')
                signal_strength = message.data.get('signal_strength', 0.5)
                current_price = message.data.get('current_price', 0.0)
                method = message.data.get('method', 'VOLATILITY')
                
                if symbol:
                    position_size = await self.calculate_position_size(
                        symbol, signal_strength, current_price, method
                    )
                    
                    await self.send_message(
                        message.agent_id,
                        'position_size_response',
                        asdict(position_size),
                        correlation_id=message.correlation_id
                    )
            
            elif message.message_type == 'risk_check_request':
                portfolio_risk = self.get_cached_data('portfolio_risk')
                if portfolio_risk:
                    await self.send_message(
                        message.agent_id,
                        'risk_check_response',
                        portfolio_risk,
                        correlation_id=message.correlation_id
                    )
            
            elif message.message_type == 'update_position':
                # Update position in risk monitoring
                symbol = message.data.get('symbol')
                if symbol:
                    # Would update position tracking here
                    self.logger.info(f"Updated position tracking for {symbol}")
                    
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
    
    async def agent_cleanup(self):
        """Cleanup agent resources"""
        self.current_positions.clear()
        self.position_history.clear()
        self.logger.info("Risk Management Agent cleaned up successfully") 