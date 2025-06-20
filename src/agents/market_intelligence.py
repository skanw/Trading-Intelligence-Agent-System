"""
Market Intelligence Agent - Global Market Context and Regime Analysis
"""

import asyncio
import aiohttp
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from .base_agent import BaseAgent, AgentMessage, AgentSignal
from ..config import config


@dataclass
class MarketRegime:
    """Market regime classification"""
    regime_type: str  # 'RISK_ON', 'RISK_OFF', 'NEUTRAL', 'VOLATILE'
    confidence: float  # 0.0 to 1.0
    indicators: Dict[str, float]
    risk_level: str  # 'LOW', 'MEDIUM', 'HIGH'
    volatility_regime: str  # 'LOW_VOL', 'NORMAL_VOL', 'HIGH_VOL'
    correlation_regime: str  # 'DIVERSIFIED', 'CORRELATED', 'CRISIS'
    timestamp: datetime


@dataclass
class MarketSnapshot:
    """Current market state snapshot"""
    timestamp: datetime
    indices: Dict[str, float]
    volatility: Dict[str, float]
    currencies: Dict[str, float]
    commodities: Dict[str, float]
    bonds: Dict[str, float]
    sector_performance: Dict[str, float]
    market_regime: MarketRegime
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EconomicIndicator:
    """Economic data point"""
    name: str
    value: float
    previous_value: Optional[float]
    change: Optional[float]
    change_percent: Optional[float]
    release_date: datetime
    next_release: Optional[datetime]
    importance: str  # 'HIGH', 'MEDIUM', 'LOW'
    market_impact: str


class MarketIntelligenceAgent(BaseAgent):
    """
    Market Intelligence Agent for global market context and regime analysis
    """
    
    def __init__(self, agent_id: str = 'market-intel-agent', config_override: Optional[Dict] = None):
        super().__init__(agent_id, config_override)
        
        # Market data tracking
        self.market_indices = {
            # US Markets
            '^GSPC': 'S&P 500',
            '^DJI': 'Dow Jones',
            '^IXIC': 'NASDAQ',
            '^RUT': 'Russell 2000',
            
            # International
            '^FTSE': 'FTSE 100',
            '^GDAXI': 'DAX',
            '^N225': 'Nikkei 225',
            '^HSI': 'Hang Seng',
            '^AXJO': 'ASX 200',
            
            # Emerging Markets
            'EEM': 'Emerging Markets ETF',
            'FXI': 'China Large Cap ETF'
        }
        
        self.volatility_indices = {
            '^VIX': 'VIX',
            '^VIX9D': 'VIX9D',
            '^MOVE': 'MOVE Index'
        }
        
        self.currencies = {
            'DX-Y.NYB': 'US Dollar Index',
            'EURUSD=X': 'EUR/USD',
            'GBPUSD=X': 'GBP/USD',
            'USDJPY=X': 'USD/JPY',
            'AUDUSD=X': 'AUD/USD'
        }
        
        self.commodities = {
            'GC=F': 'Gold',
            'CL=F': 'Crude Oil',
            '^TNX': '10Y Treasury',
            'BTC-USD': 'Bitcoin'
        }
        
        self.sector_etfs = {
            'XLK': 'Technology',
            'XLF': 'Financials',
            'XLE': 'Energy',
            'XLV': 'Healthcare',
            'XLI': 'Industrials',
            'XLP': 'Consumer Staples',
            'XLY': 'Consumer Discretionary',
            'XLU': 'Utilities',
            'XLRE': 'Real Estate',
            'XLB': 'Materials',
            'XLC': 'Communication Services'
        }
        
        # Analysis parameters
        self.lookback_days = 30
        self.correlation_window = 20
        self.volatility_window = 20
        
        # Historical data storage
        self.price_history = {}
        self.correlation_matrix = None
        self.current_regime = None
        
        # Economic calendar
        self.economic_events = []
        
    async def agent_initialize(self):
        """Initialize market intelligence agent"""
        try:
            # Load initial market data
            await self._load_historical_data()
            
            # Initialize economic calendar
            await self._load_economic_calendar()
            
            self.logger.info("Market Intelligence Agent initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Market Intelligence Agent: {e}")
            raise
    
    async def execute(self):
        """Main execution logic - analyze market conditions"""
        try:
            # Fetch current market data
            market_snapshot = await self._fetch_market_snapshot()
            
            # Analyze market regime
            regime = await self._analyze_market_regime(market_snapshot)
            market_snapshot.market_regime = regime
            
            # Update correlations
            await self._update_correlation_analysis()
            
            # Generate market alerts
            alerts = await self._generate_market_alerts(market_snapshot)
            
            # Send alerts to other agents
            for alert in alerts:
                await self._send_market_alert(alert)
            
            # Cache market data
            self._cache_market_data(market_snapshot)
            
            # Update economic calendar
            await self._check_economic_events()
            
            self.logger.info(f"Market analysis completed - Regime: {regime.regime_type}")
            
        except Exception as e:
            self.logger.error(f"Error in market intelligence execution: {e}")
            self.metrics['errors'] += 1
    
    async def _fetch_market_snapshot(self) -> MarketSnapshot:
        """Fetch current market data across all asset classes"""
        snapshot_data = {
            'indices': {},
            'volatility': {},
            'currencies': {},
            'commodities': {},
            'sector_performance': {}
        }
        
        try:
            # Fetch indices
            for symbol, name in self.market_indices.items():
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="2d")
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
                        prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                        change_pct = ((current_price - prev_price) / prev_price) * 100
                        snapshot_data['indices'][name] = {
                            'price': current_price,
                            'change_pct': change_pct
                        }
                except Exception as e:
                    self.logger.warning(f"Failed to fetch {symbol}: {e}")
            
            # Fetch volatility indices
            for symbol, name in self.volatility_indices.items():
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="2d")
                    if not hist.empty:
                        current_vol = hist['Close'].iloc[-1]
                        snapshot_data['volatility'][name] = current_vol
                except Exception as e:
                    self.logger.warning(f"Failed to fetch volatility {symbol}: {e}")
            
            # Fetch currencies
            for symbol, name in self.currencies.items():
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="2d")
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
                        prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                        change_pct = ((current_price - prev_price) / prev_price) * 100
                        snapshot_data['currencies'][name] = {
                            'price': current_price,
                            'change_pct': change_pct
                        }
                except Exception as e:
                    self.logger.warning(f"Failed to fetch currency {symbol}: {e}")
            
            # Fetch commodities
            for symbol, name in self.commodities.items():
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="2d")
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
                        prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                        change_pct = ((current_price - prev_price) / prev_price) * 100
                        snapshot_data['commodities'][name] = {
                            'price': current_price,
                            'change_pct': change_pct
                        }
                except Exception as e:
                    self.logger.warning(f"Failed to fetch commodity {symbol}: {e}")
            
            # Fetch sector performance
            for symbol, name in self.sector_etfs.items():
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="5d")
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
                        week_ago_price = hist['Close'].iloc[0]
                        change_pct = ((current_price - week_ago_price) / week_ago_price) * 100
                        snapshot_data['sector_performance'][name] = change_pct
                except Exception as e:
                    self.logger.warning(f"Failed to fetch sector {symbol}: {e}")
            
        except Exception as e:
            self.logger.error(f"Error fetching market snapshot: {e}")
        
        return MarketSnapshot(
            timestamp=datetime.now(timezone.utc),
            indices=snapshot_data['indices'],
            volatility=snapshot_data['volatility'],
            currencies=snapshot_data['currencies'],
            commodities=snapshot_data['commodities'],
            bonds={},  # Would add bond data here
            sector_performance=snapshot_data['sector_performance'],
            market_regime=None  # Will be set by analyze_market_regime
        )
    
    async def _analyze_market_regime(self, snapshot: MarketSnapshot) -> MarketRegime:
        """Analyze current market regime based on multiple indicators"""
        try:
            indicators = {}
            
            # VIX analysis
            vix_level = snapshot.volatility.get('VIX', 20)
            indicators['vix'] = vix_level
            
            # Market breadth (using index performance)
            sp500_change = snapshot.indices.get('S&P 500', {}).get('change_pct', 0)
            nasdaq_change = snapshot.indices.get('NASDAQ', {}).get('change_pct', 0)
            russell_change = snapshot.indices.get('Russell 2000', {}).get('change_pct', 0)
            
            # Calculate average market performance
            market_performance = np.mean([sp500_change, nasdaq_change, russell_change])
            indicators['market_performance'] = market_performance
            
            # Currency strength (USD index)
            usd_strength = snapshot.currencies.get('US Dollar Index', {}).get('change_pct', 0)
            indicators['usd_strength'] = usd_strength
            
            # Safe haven flows (Gold performance)
            gold_performance = snapshot.commodities.get('Gold', {}).get('change_pct', 0)
            indicators['gold_performance'] = gold_performance
            
            # Sector rotation analysis
            sectors = snapshot.sector_performance
            if sectors:
                cyclical_sectors = ['Technology', 'Financials', 'Energy', 'Industrials', 'Consumer Discretionary']
                defensive_sectors = ['Utilities', 'Consumer Staples', 'Healthcare']
                
                cyclical_performance = np.mean([sectors.get(sector, 0) for sector in cyclical_sectors if sector in sectors])
                defensive_performance = np.mean([sectors.get(sector, 0) for sector in defensive_sectors if sector in sectors])
                
                rotation_signal = cyclical_performance - defensive_performance
                indicators['sector_rotation'] = rotation_signal
            
            # Determine regime
            regime_type = self._classify_regime(indicators)
            confidence = self._calculate_regime_confidence(indicators)
            
            # Risk level assessment
            risk_level = 'LOW'
            if vix_level > 25 or abs(market_performance) > 2:
                risk_level = 'HIGH'
            elif vix_level > 20 or abs(market_performance) > 1:
                risk_level = 'MEDIUM'
            
            # Volatility regime
            volatility_regime = 'NORMAL_VOL'
            if vix_level > 30:
                volatility_regime = 'HIGH_VOL'
            elif vix_level < 15:
                volatility_regime = 'LOW_VOL'
            
            # Correlation regime (simplified)
            correlation_regime = 'DIVERSIFIED'
            if vix_level > 25:
                correlation_regime = 'CRISIS'
            elif abs(market_performance) > 1.5:
                correlation_regime = 'CORRELATED'
            
            return MarketRegime(
                regime_type=regime_type,
                confidence=confidence,
                indicators=indicators,
                risk_level=risk_level,
                volatility_regime=volatility_regime,
                correlation_regime=correlation_regime,
                timestamp=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing market regime: {e}")
            return MarketRegime(
                regime_type='NEUTRAL',
                confidence=0.5,
                indicators={},
                risk_level='MEDIUM',
                volatility_regime='NORMAL_VOL',
                correlation_regime='DIVERSIFIED',
                timestamp=datetime.now(timezone.utc)
            )
    
    def _classify_regime(self, indicators: Dict[str, float]) -> str:
        """Classify market regime based on indicators"""
        vix = indicators.get('vix', 20)
        market_perf = indicators.get('market_performance', 0)
        rotation = indicators.get('sector_rotation', 0)
        gold_perf = indicators.get('gold_performance', 0)
        
        # Risk-off conditions
        if vix > 25 or market_perf < -1.5 or gold_perf > 1:
            return 'RISK_OFF'
        
        # Risk-on conditions
        if vix < 18 and market_perf > 1 and rotation > 0.5:
            return 'RISK_ON'
        
        # Volatile conditions
        if vix > 30 or abs(market_perf) > 2:
            return 'VOLATILE'
        
        return 'NEUTRAL'
    
    def _calculate_regime_confidence(self, indicators: Dict[str, float]) -> float:
        """Calculate confidence in regime classification"""
        confidence = 0.5  # Base confidence
        
        # Strong VIX signals increase confidence
        vix = indicators.get('vix', 20)
        if vix > 30 or vix < 15:
            confidence += 0.2
        
        # Consistent market performance
        market_perf = abs(indicators.get('market_performance', 0))
        if market_perf > 1.5:
            confidence += 0.2
        
        # Clear sector rotation
        rotation = abs(indicators.get('sector_rotation', 0))
        if rotation > 1:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    async def _update_correlation_analysis(self):
        """Update correlation matrix for major assets"""
        try:
            # Get recent price data for correlation analysis
            symbols = list(self.market_indices.keys())[:5]  # Top 5 indices
            
            price_data = {}
            for symbol in symbols:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=f"{self.correlation_window}d")
                if not hist.empty:
                    price_data[symbol] = hist['Close'].pct_change().dropna()
            
            if len(price_data) > 1:
                # Create DataFrame and calculate correlation
                df = pd.DataFrame(price_data)
                self.correlation_matrix = df.corr()
                
                # Cache correlation data
                self.cache_data('correlation_matrix', self.correlation_matrix.to_dict())
                
        except Exception as e:
            self.logger.error(f"Error updating correlation analysis: {e}")
    
    async def _generate_market_alerts(self, snapshot: MarketSnapshot) -> List[Dict[str, Any]]:
        """Generate market-based alerts"""
        alerts = []
        
        try:
            regime = snapshot.market_regime
            
            # Regime change alert
            if self.current_regime and self.current_regime != regime.regime_type:
                alerts.append({
                    'type': 'REGIME_CHANGE',
                    'message': f"Market regime changed from {self.current_regime} to {regime.regime_type}",
                    'severity': 'HIGH',
                    'data': regime.__dict__
                })
            
            # High volatility alert
            vix_level = snapshot.volatility.get('VIX', 20)
            if vix_level > 30:
                alerts.append({
                    'type': 'HIGH_VOLATILITY',
                    'message': f"VIX spiked to {vix_level:.1f}",
                    'severity': 'HIGH',
                    'data': {'vix': vix_level}
                })
            
            # Market stress alert
            sp500_change = snapshot.indices.get('S&P 500', {}).get('change_pct', 0)
            if abs(sp500_change) > 2:
                alerts.append({
                    'type': 'MARKET_STRESS',
                    'message': f"S&P 500 moved {sp500_change:.1f}% today",
                    'severity': 'MEDIUM',
                    'data': {'sp500_change': sp500_change}
                })
            
            # Update current regime
            self.current_regime = regime.regime_type
            
        except Exception as e:
            self.logger.error(f"Error generating market alerts: {e}")
        
        return alerts
    
    async def _send_market_alert(self, alert: Dict[str, Any]):
        """Send market alert to relevant agents"""
        try:
            # Send to risk management
            await self.send_message(
                'risk-mgmt-agent',
                'market_alert',
                alert,
                priority=2 if alert['severity'] == 'HIGH' else 5
            )
            
            # Send to portfolio management
            await self.send_message(
                'portfolio-mgmt-agent',
                'market_alert',
                alert,
                priority=2 if alert['severity'] == 'HIGH' else 5
            )
            
            # Send to orchestrator
            await self.send_message(
                'orchestrator-agent',
                'market_alert',
                alert,
                priority=2 if alert['severity'] == 'HIGH' else 5
            )
            
        except Exception as e:
            self.logger.error(f"Error sending market alert: {e}")
    
    def _cache_market_data(self, snapshot: MarketSnapshot):
        """Cache market data for other agents"""
        try:
            # Cache current snapshot
            self.cache_data('market_snapshot', snapshot.to_dict(), expiry=300)  # 5 minutes
            
            # Cache regime data
            self.cache_data('market_regime', asdict(snapshot.market_regime), expiry=1800)  # 30 minutes
            
            # Cache sector performance
            self.cache_data('sector_performance', snapshot.sector_performance, expiry=900)  # 15 minutes
            
            # Cache volatility data
            self.cache_data('volatility_indices', snapshot.volatility, expiry=300)  # 5 minutes
            
        except Exception as e:
            self.logger.error(f"Error caching market data: {e}")
    
    async def _load_historical_data(self):
        """Load historical data for analysis"""
        try:
            # Load price history for major indices
            for symbol in list(self.market_indices.keys())[:5]:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=f"{self.lookback_days}d")
                if not hist.empty:
                    self.price_history[symbol] = hist
                    
        except Exception as e:
            self.logger.error(f"Error loading historical data: {e}")
    
    async def _load_economic_calendar(self):
        """Load economic calendar events (simplified)"""
        try:
            # In production, this would connect to economic calendar API
            # For now, we'll use a simplified approach
            
            current_date = datetime.now()
            
            # Sample economic events
            self.economic_events = [
                {
                    'name': 'Federal Reserve Meeting',
                    'date': current_date + timedelta(days=7),
                    'importance': 'HIGH',
                    'impact': 'Interest rates and monetary policy'
                },
                {
                    'name': 'Non-Farm Payrolls',
                    'date': current_date + timedelta(days=3),
                    'importance': 'HIGH',
                    'impact': 'Employment data'
                },
                {
                    'name': 'CPI Release',
                    'date': current_date + timedelta(days=5),
                    'importance': 'HIGH',
                    'impact': 'Inflation data'
                }
            ]
            
        except Exception as e:
            self.logger.error(f"Error loading economic calendar: {e}")
    
    async def _check_economic_events(self):
        """Check for upcoming economic events"""
        try:
            now = datetime.now()
            upcoming_events = []
            
            for event in self.economic_events:
                if event['date'] > now and (event['date'] - now).days <= 2:
                    upcoming_events.append(event)
            
            if upcoming_events:
                for event in upcoming_events:
                    await self.send_message(
                        'orchestrator-agent',
                        'economic_event_alert',
                        event,
                        priority=3
                    )
                    
        except Exception as e:
            self.logger.error(f"Error checking economic events: {e}")
    
    async def handle_message(self, message: AgentMessage):
        """Handle incoming messages from other agents"""
        try:
            if message.message_type == 'request_market_regime':
                # Provide current market regime
                if self.current_regime:
                    cached_regime = self.get_cached_data('market_regime')
                    if cached_regime:
                        await self.send_message(
                            message.agent_id,
                            'market_regime_response',
                            cached_regime,
                            correlation_id=message.correlation_id
                        )
            
            elif message.message_type == 'request_sector_performance':
                # Provide sector performance data
                sector_data = self.get_cached_data('sector_performance')
                if sector_data:
                    await self.send_message(
                        message.agent_id,
                        'sector_performance_response',
                        sector_data,
                        correlation_id=message.correlation_id
                    )
            
            elif message.message_type == 'request_volatility_data':
                # Provide volatility indices
                vol_data = self.get_cached_data('volatility_indices')
                if vol_data:
                    await self.send_message(
                        message.agent_id,
                        'volatility_data_response',
                        vol_data,
                        correlation_id=message.correlation_id
                    )
                    
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
    
    async def agent_cleanup(self):
        """Cleanup agent resources"""
        self.logger.info("Market Intelligence Agent cleaned up successfully") 