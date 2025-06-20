"""
Technical Analysis Agent - Multi-timeframe TA with momentum and patterns
"""

import asyncio
import numpy as np
import pandas as pd
import talib
import yfinance as yf
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

from .base_agent import BaseAgent, AgentMessage, AgentSignal
from ..config import config


@dataclass
class TechnicalSignal:
    """Technical analysis signal"""
    symbol: str
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    strength: float  # 0.0 to 1.0
    timeframe: str  # '1m', '5m', '15m', '1h', '4h', '1d', '1w'
    price: float
    target_price: Optional[float]
    stop_loss: Optional[float]
    confidence: float
    indicators: Dict[str, float]
    pattern: Optional[str]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SupportResistance:
    """Support and resistance levels"""
    symbol: str
    support_levels: List[float]
    resistance_levels: List[float]
    pivot_point: float
    fibonacci_levels: Dict[str, float]
    volume_profile: Dict[str, float]
    timestamp: datetime


@dataclass
class TechnicalAnalysis:
    """Complete technical analysis for a symbol"""
    symbol: str
    current_price: float
    trend: Dict[str, str]  # By timeframe
    momentum: Dict[str, float]
    volume_analysis: Dict[str, Any]
    support_resistance: SupportResistance
    signals: List[TechnicalSignal]
    risk_metrics: Dict[str, float]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TechnicalAnalysisAgent(BaseAgent):
    """
    Technical Analysis Agent for multi-timeframe technical analysis
    """
    
    def __init__(self, agent_id: str = 'technical-agent', config_override: Optional[Dict] = None):
        super().__init__(agent_id, config_override)
        
        # Technical analysis configuration
        self.timeframes = ['1d', '4h', '1h', '15m', '5m']
        default_symbols = {
            # Major indices
            'SPY': 'SPDR S&P 500 ETF',
            'QQQ': 'Invesco QQQ Trust',
            'IWM': 'iShares Russell 2000 ETF',
            'DIA': 'SPDR Dow Jones Industrial Average ETF',
            
            # Mega cap stocks
            'AAPL': 'Apple Inc.',
            'MSFT': 'Microsoft Corporation',
            'GOOGL': 'Alphabet Inc.',
            'AMZN': 'Amazon.com Inc.',
            'TSLA': 'Tesla Inc.',
            'NVDA': 'NVIDIA Corporation',
            'META': 'Meta Platforms Inc.',
            'BRK-B': 'Berkshire Hathaway Inc.',
            'JNJ': 'Johnson & Johnson',
            'V': 'Visa Inc.',
            
            # Sector ETFs
            'XLK': 'Technology Select Sector SPDR Fund',
            'XLF': 'Financial Select Sector SPDR Fund',
            'XLE': 'Energy Select Sector SPDR Fund',
            'XLV': 'Health Care Select Sector SPDR Fund',
            'XLI': 'Industrial Select Sector SPDR Fund'
        }
        
        # Use config override or default symbols
        self.coverage_symbols = self.config.get('coverage_symbols', default_symbols) if hasattr(self.config, 'get') and self.config else default_symbols
        
        # Technical indicators configuration
        self.ma_periods = [9, 21, 50, 100, 200]
        self.rsi_period = 14
        self.macd_params = (12, 26, 9)
        self.bb_period = 20
        self.bb_std = 2
        
        # Pattern recognition
        self.candlestick_patterns = [
            'CDL2CROWS', 'CDL3BLACKCROWS', 'CDL3INSIDE', 'CDL3LINESTRIKE',
            'CDL3OUTSIDE', 'CDL3STARSINSOUTH', 'CDL3WHITESOLDIERS', 'CDLABANDONED',
            'CDLBELTHOLD', 'CDLBREAKAWAY', 'CDLCLOSINGMARUBOZU', 'CDLCONCEALBABYSWALL',
            'CDLCOUNTERATTACK', 'CDLDARKCLOUDCOVER', 'CDLDOJI', 'CDLDOJISTAR',
            'CDLDRAGONFLYDOJI', 'CDLENGULFING', 'CDLEVENINGDOJISTAR', 'CDLEVENINGSTAR',
            'CDLGAPSIDESIDEWHITE', 'CDLGRAVESTONEDOJI', 'CDLHAMMER', 'CDLHANGINGMAN',
            'CDLHARAMI', 'CDLHARAMICROSS', 'CDLHIGHWAVE', 'CDLHIKKAKE',
            'CDLHIKKAKEMOD', 'CDLHOMINGPIGEON', 'CDLIDENTICAL3CROWS', 'CDLINNECK',
            'CDLINVERTEDHAMMER', 'CDLKICKING', 'CDLKICKINGBYLENGTH', 'CDLLADDERBOTTOM',
            'CDLLONGLEGGEDDOJI', 'CDLLONGLINE', 'CDLMARUBOZU', 'CDLMATCHINGLOW',
            'CDLMATHOLD', 'CDLMORNINGDOJISTAR', 'CDLMORNINGSTAR', 'CDLONNECK',
            'CDLPIERCING', 'CDLRICKSHAWMAN', 'CDLRISEFALL3METHODS', 'CDLSEPARATINGLINES',
            'CDLSHOOTINGSTAR', 'CDLSHORTLINE', 'CDLSPINNINGTOP', 'CDLSTALLEDPATTERN',
            'CDLSTICKSANDWICH', 'CDLTAKURI', 'CDLTASUKIGAP', 'CDLTHRUSTING',
            'CDLTRISTAR', 'CDLUNIQUE3RIVER', 'CDLUPSIDEGAP2CROWS', 'CDLXSIDEGAP3METHODS'
        ]
        
        # Analysis cache
        self.analysis_cache = {}
        self.price_data_cache = {}
        
    async def agent_initialize(self):
        """Initialize technical analysis agent"""
        try:
            # Initialize price data for covered symbols
            await self._initialize_price_data()
            
            self.logger.info("Technical Analysis Agent initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Technical Analysis Agent: {e}")
            raise
    
    async def execute(self):
        """Main execution logic - perform technical analysis"""
        try:
            # Analyze all covered symbols
            for symbol in self.coverage_symbols.keys():
                try:
                    analysis = await self._analyze_symbol(symbol)
                    if analysis:
                        # Cache analysis
                        self.analysis_cache[symbol] = analysis
                        
                        # Send signals to other agents
                        await self._send_technical_signals(analysis)
                        
                        # Cache technical data
                        self._cache_technical_data(symbol, analysis)
                        
                except Exception as e:
                    self.logger.error(f"Error analyzing {symbol}: {e}")
                    continue
            
            self.logger.info(f"Technical analysis completed for {len(self.coverage_symbols)} symbols")
            
        except Exception as e:
            self.logger.error(f"Error in technical analysis execution: {e}")
            self.metrics['errors'] += 1
    
    async def _analyze_symbol(self, symbol: str) -> Optional[TechnicalAnalysis]:
        """Perform comprehensive technical analysis for a symbol"""
        try:
            # Get price data
            price_data = await self._get_price_data(symbol)
            if price_data is None or price_data.empty:
                return None
            
            current_price = price_data['Close'].iloc[-1]
            
            # Trend analysis across timeframes
            trend_analysis = self._analyze_trends(price_data)
            
            # Momentum indicators
            momentum_analysis = self._analyze_momentum(price_data)
            
            # Volume analysis
            volume_analysis = self._analyze_volume(price_data)
            
            # Support and resistance
            support_resistance = self._calculate_support_resistance(symbol, price_data)
            
            # Generate signals
            signals = self._generate_technical_signals(symbol, price_data, current_price)
            
            # Risk metrics
            risk_metrics = self._calculate_risk_metrics(price_data)
            
            return TechnicalAnalysis(
                symbol=symbol,
                current_price=current_price,
                trend=trend_analysis,
                momentum=momentum_analysis,
                volume_analysis=volume_analysis,
                support_resistance=support_resistance,
                signals=signals,
                risk_metrics=risk_metrics,
                timestamp=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            self.logger.error(f"Error in technical analysis for {symbol}: {e}")
            return None
    
    async def _get_price_data(self, symbol: str, period: str = '3mo') -> Optional[pd.DataFrame]:
        """Get price data for technical analysis"""
        try:
            # Check cache first
            cache_key = f"{symbol}_{period}"
            if cache_key in self.price_data_cache:
                cached_data, cache_time = self.price_data_cache[cache_key]
                if (datetime.now() - cache_time).seconds < 300:  # 5 minutes
                    return cached_data
            
            # Fetch fresh data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval='1d')
            
            if not data.empty:
                # Cache the data
                self.price_data_cache[cache_key] = (data, datetime.now())
                return data
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching price data for {symbol}: {e}")
            return None
    
    def _analyze_trends(self, data: pd.DataFrame) -> Dict[str, str]:
        """Analyze trend direction across different timeframes"""
        try:
            trends = {}
            
            # Short-term trend (9 vs 21 MA)
            ma9 = talib.SMA(data['Close'], timeperiod=9)
            ma21 = talib.SMA(data['Close'], timeperiod=21)
            
            if ma9.iloc[-1] > ma21.iloc[-1]:
                trends['short_term'] = 'BULLISH'
            elif ma9.iloc[-1] < ma21.iloc[-1]:
                trends['short_term'] = 'BEARISH'
            else:
                trends['short_term'] = 'NEUTRAL'
            
            # Medium-term trend (21 vs 50 MA)
            ma50 = talib.SMA(data['Close'], timeperiod=50)
            
            if len(ma50.dropna()) > 0:
                if ma21.iloc[-1] > ma50.iloc[-1]:
                    trends['medium_term'] = 'BULLISH'
                elif ma21.iloc[-1] < ma50.iloc[-1]:
                    trends['medium_term'] = 'BEARISH'
                else:
                    trends['medium_term'] = 'NEUTRAL'
            else:
                trends['medium_term'] = 'NEUTRAL'
            
            # Long-term trend (50 vs 200 MA)
            ma200 = talib.SMA(data['Close'], timeperiod=200)
            
            if len(ma200.dropna()) > 0:
                if ma50.iloc[-1] > ma200.iloc[-1]:
                    trends['long_term'] = 'BULLISH'
                elif ma50.iloc[-1] < ma200.iloc[-1]:
                    trends['long_term'] = 'BEARISH'
                else:
                    trends['long_term'] = 'NEUTRAL'
            else:
                trends['long_term'] = 'NEUTRAL'
            
            return trends
            
        except Exception as e:
            self.logger.error(f"Error analyzing trends: {e}")
            return {'short_term': 'NEUTRAL', 'medium_term': 'NEUTRAL', 'long_term': 'NEUTRAL'}
    
    def _analyze_momentum(self, data: pd.DataFrame) -> Dict[str, float]:
        """Analyze momentum indicators"""
        try:
            momentum = {}
            
            # RSI
            rsi = talib.RSI(data['Close'], timeperiod=self.rsi_period)
            momentum['rsi'] = float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(
                data['Close'], 
                fastperiod=self.macd_params[0],
                slowperiod=self.macd_params[1],
                signalperiod=self.macd_params[2]
            )
            momentum['macd'] = float(macd.iloc[-1]) if not np.isnan(macd.iloc[-1]) else 0.0
            momentum['macd_signal'] = float(macd_signal.iloc[-1]) if not np.isnan(macd_signal.iloc[-1]) else 0.0
            momentum['macd_histogram'] = float(macd_hist.iloc[-1]) if not np.isnan(macd_hist.iloc[-1]) else 0.0
            
            # Stochastic
            slowk, slowd = talib.STOCH(data['High'], data['Low'], data['Close'])
            momentum['stoch_k'] = float(slowk.iloc[-1]) if not np.isnan(slowk.iloc[-1]) else 50.0
            momentum['stoch_d'] = float(slowd.iloc[-1]) if not np.isnan(slowd.iloc[-1]) else 50.0
            
            # Williams %R
            willr = talib.WILLR(data['High'], data['Low'], data['Close'])
            momentum['williams_r'] = float(willr.iloc[-1]) if not np.isnan(willr.iloc[-1]) else -50.0
            
            # CCI
            cci = talib.CCI(data['High'], data['Low'], data['Close'])
            momentum['cci'] = float(cci.iloc[-1]) if not np.isnan(cci.iloc[-1]) else 0.0
            
            return momentum
            
        except Exception as e:
            self.logger.error(f"Error analyzing momentum: {e}")
            return {'rsi': 50.0, 'macd': 0.0, 'macd_signal': 0.0, 'macd_histogram': 0.0, 
                   'stoch_k': 50.0, 'stoch_d': 50.0, 'williams_r': -50.0, 'cci': 0.0}
    
    def _analyze_volume(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume patterns"""
        try:
            volume_analysis = {}
            
            # Average volume
            avg_volume = data['Volume'].rolling(window=20).mean()
            current_volume = data['Volume'].iloc[-1]
            volume_analysis['current_volume'] = float(current_volume)
            volume_analysis['avg_volume_20d'] = float(avg_volume.iloc[-1])
            volume_analysis['volume_ratio'] = float(current_volume / avg_volume.iloc[-1]) if avg_volume.iloc[-1] > 0 else 1.0
            
            # On-Balance Volume
            obv = talib.OBV(data['Close'], data['Volume'])
            volume_analysis['obv'] = float(obv.iloc[-1])
            volume_analysis['obv_trend'] = 'RISING' if obv.iloc[-1] > obv.iloc[-5] else 'FALLING'
            
            # Volume Price Trend
            close_change = data['Close'].pct_change()
            vpt = (close_change * data['Volume']).cumsum()
            volume_analysis['vpt'] = float(vpt.iloc[-1])
            
            # Accumulation/Distribution Line
            ad_line = talib.AD(data['High'], data['Low'], data['Close'], data['Volume'])
            volume_analysis['ad_line'] = float(ad_line.iloc[-1])
            
            return volume_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing volume: {e}")
            return {'current_volume': 0, 'avg_volume_20d': 0, 'volume_ratio': 1.0, 
                   'obv': 0, 'obv_trend': 'NEUTRAL', 'vpt': 0, 'ad_line': 0}
    
    def _calculate_support_resistance(self, symbol: str, data: pd.DataFrame) -> SupportResistance:
        """Calculate support and resistance levels"""
        try:
            # Simple pivot points calculation
            high = data['High'].iloc[-1]
            low = data['Low'].iloc[-1]
            close = data['Close'].iloc[-1]
            
            pivot = (high + low + close) / 3
            
            # Support levels
            s1 = 2 * pivot - high
            s2 = pivot - (high - low)
            
            # Resistance levels
            r1 = 2 * pivot - low
            r2 = pivot + (high - low)
            
            # Fibonacci retracements
            price_range = high - low
            fib_levels = {
                '23.6%': high - (price_range * 0.236),
                '38.2%': high - (price_range * 0.382),
                '50.0%': high - (price_range * 0.500),
                '61.8%': high - (price_range * 0.618),
                '78.6%': high - (price_range * 0.786)
            }
            
            # Volume profile (simplified)
            volume_profile = {}
            price_levels = np.linspace(low, high, 10)
            for i, level in enumerate(price_levels):
                volume_profile[f'level_{i}'] = float(level)
            
            return SupportResistance(
                symbol=symbol,
                support_levels=[s2, s1],
                resistance_levels=[r1, r2],
                pivot_point=pivot,
                fibonacci_levels=fib_levels,
                volume_profile=volume_profile,
                timestamp=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating support/resistance: {e}")
            return SupportResistance(
                symbol=symbol,
                support_levels=[],
                resistance_levels=[],
                pivot_point=0.0,
                fibonacci_levels={},
                volume_profile={},
                timestamp=datetime.now(timezone.utc)
            )
    
    def _generate_technical_signals(self, symbol: str, data: pd.DataFrame, current_price: float) -> List[TechnicalSignal]:
        """Generate technical trading signals"""
        signals = []
        
        try:
            # RSI signals
            rsi = talib.RSI(data['Close'], timeperiod=self.rsi_period)
            current_rsi = rsi.iloc[-1]
            
            if current_rsi < 30:
                signals.append(TechnicalSignal(
                    symbol=symbol,
                    signal_type='BUY',
                    strength=0.7,
                    timeframe='1d',
                    price=current_price,
                    target_price=current_price * 1.05,
                    stop_loss=current_price * 0.95,
                    confidence=0.65,
                    indicators={'rsi': current_rsi},
                    pattern='RSI_OVERSOLD',
                    timestamp=datetime.now(timezone.utc)
                ))
            elif current_rsi > 70:
                signals.append(TechnicalSignal(
                    symbol=symbol,
                    signal_type='SELL',
                    strength=0.7,
                    timeframe='1d',
                    price=current_price,
                    target_price=current_price * 0.95,
                    stop_loss=current_price * 1.05,
                    confidence=0.65,
                    indicators={'rsi': current_rsi},
                    pattern='RSI_OVERBOUGHT',
                    timestamp=datetime.now(timezone.utc)
                ))
            
            # MACD signals
            macd, macd_signal, macd_hist = talib.MACD(data['Close'])
            
            if len(macd_hist) > 1:
                current_hist = macd_hist.iloc[-1]
                prev_hist = macd_hist.iloc[-2]
                
                if prev_hist < 0 and current_hist > 0:  # MACD bullish crossover
                    signals.append(TechnicalSignal(
                        symbol=symbol,
                        signal_type='BUY',
                        strength=0.8,
                        timeframe='1d',
                        price=current_price,
                        target_price=current_price * 1.08,
                        stop_loss=current_price * 0.92,
                        confidence=0.75,
                        indicators={'macd': float(macd.iloc[-1]), 'macd_signal': float(macd_signal.iloc[-1])},
                        pattern='MACD_BULLISH_CROSSOVER',
                        timestamp=datetime.now(timezone.utc)
                    ))
                elif prev_hist > 0 and current_hist < 0:  # MACD bearish crossover
                    signals.append(TechnicalSignal(
                        symbol=symbol,
                        signal_type='SELL',
                        strength=0.8,
                        timeframe='1d',
                        price=current_price,
                        target_price=current_price * 0.92,
                        stop_loss=current_price * 1.08,
                        confidence=0.75,
                        indicators={'macd': float(macd.iloc[-1]), 'macd_signal': float(macd_signal.iloc[-1])},
                        pattern='MACD_BEARISH_CROSSOVER',
                        timestamp=datetime.now(timezone.utc)
                    ))
            
            # Moving average signals
            ma20 = talib.SMA(data['Close'], timeperiod=20)
            ma50 = talib.SMA(data['Close'], timeperiod=50)
            
            if len(ma20) > 1 and len(ma50) > 1:
                if ma20.iloc[-1] > ma50.iloc[-1] and ma20.iloc[-2] <= ma50.iloc[-2]:
                    signals.append(TechnicalSignal(
                        symbol=symbol,
                        signal_type='BUY',
                        strength=0.6,
                        timeframe='1d',
                        price=current_price,
                        target_price=current_price * 1.06,
                        stop_loss=current_price * 0.94,
                        confidence=0.60,
                        indicators={'ma20': float(ma20.iloc[-1]), 'ma50': float(ma50.iloc[-1])},
                        pattern='MA_GOLDEN_CROSS',
                        timestamp=datetime.now(timezone.utc)
                    ))
                elif ma20.iloc[-1] < ma50.iloc[-1] and ma20.iloc[-2] >= ma50.iloc[-2]:
                    signals.append(TechnicalSignal(
                        symbol=symbol,
                        signal_type='SELL',
                        strength=0.6,
                        timeframe='1d',
                        price=current_price,
                        target_price=current_price * 0.94,
                        stop_loss=current_price * 1.06,
                        confidence=0.60,
                        indicators={'ma20': float(ma20.iloc[-1]), 'ma50': float(ma50.iloc[-1])},
                        pattern='MA_DEATH_CROSS',
                        timestamp=datetime.now(timezone.utc)
                    ))
            
            # Bollinger Bands signals
            bb_upper, bb_middle, bb_lower = talib.BBANDS(data['Close'], timeperiod=20, nbdevup=2, nbdevdn=2)
            
            if current_price <= bb_lower.iloc[-1]:
                signals.append(TechnicalSignal(
                    symbol=symbol,
                    signal_type='BUY',
                    strength=0.5,
                    timeframe='1d',
                    price=current_price,
                    target_price=float(bb_middle.iloc[-1]),
                    stop_loss=current_price * 0.95,
                    confidence=0.55,
                    indicators={'bb_position': 'LOWER_BAND'},
                    pattern='BB_OVERSOLD',
                    timestamp=datetime.now(timezone.utc)
                ))
            elif current_price >= bb_upper.iloc[-1]:
                signals.append(TechnicalSignal(
                    symbol=symbol,
                    signal_type='SELL',
                    strength=0.5,
                    timeframe='1d',
                    price=current_price,
                    target_price=float(bb_middle.iloc[-1]),
                    stop_loss=current_price * 1.05,
                    confidence=0.55,
                    indicators={'bb_position': 'UPPER_BAND'},
                    pattern='BB_OVERBOUGHT',
                    timestamp=datetime.now(timezone.utc)
                ))
            
        except Exception as e:
            self.logger.error(f"Error generating technical signals for {symbol}: {e}")
        
        return signals
    
    def _calculate_risk_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate risk metrics from price data"""
        try:
            returns = data['Close'].pct_change().dropna()
            
            # Volatility (annualized)
            volatility = returns.std() * np.sqrt(252)
            
            # Value at Risk (5% VaR)
            var_5 = np.percentile(returns, 5)
            
            # Maximum drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Beta (simplified, using SPY as benchmark)
            # Note: This would be more accurate with actual benchmark data
            beta = 1.0  # Placeholder
            
            # Sharpe ratio (using risk-free rate of 2%)
            risk_free_rate = 0.02 / 252  # Daily risk-free rate
            excess_returns = returns - risk_free_rate
            sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            
            return {
                'volatility': float(volatility),
                'var_5_percent': float(var_5),
                'max_drawdown': float(max_drawdown),
                'beta': float(beta),
                'sharpe_ratio': float(sharpe_ratio)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {e}")
            return {
                'volatility': 0.0,
                'var_5_percent': 0.0,
                'max_drawdown': 0.0,
                'beta': 1.0,
                'sharpe_ratio': 0.0
            }
    
    async def _send_technical_signals(self, analysis: TechnicalAnalysis):
        """Send technical signals to other agents"""
        try:
            for signal in analysis.signals:
                if signal.confidence > 0.6:  # Only send high-confidence signals
                    # Send to portfolio management
                    await self.send_message(
                        'portfolio-mgmt-agent',
                        'technical_signal',
                        signal.to_dict(),
                        priority=3
                    )
                    
                    # Send to execution agent for high-strength signals
                    if signal.strength > 0.7:
                        await self.send_message(
                            'execution-agent',
                            'execution_signal',
                            signal.to_dict(),
                            priority=2
                        )
            
        except Exception as e:
            self.logger.error(f"Error sending technical signals: {e}")
    
    def _cache_technical_data(self, symbol: str, analysis: TechnicalAnalysis):
        """Cache technical analysis data"""
        try:
            # Cache full analysis
            self.cache_data(f'technical_analysis_{symbol}', analysis.to_dict(), expiry=900)  # 15 minutes
            
            # Cache key metrics
            self.cache_data(f'support_resistance_{symbol}', asdict(analysis.support_resistance), expiry=1800)
            self.cache_data(f'momentum_{symbol}', analysis.momentum, expiry=300)
            self.cache_data(f'trend_{symbol}', analysis.trend, expiry=600)
            
        except Exception as e:
            self.logger.error(f"Error caching technical data: {e}")
    
    async def _initialize_price_data(self):
        """Initialize price data for covered symbols"""
        try:
            for symbol in list(self.coverage_symbols.keys())[:5]:  # Start with first 5
                await self._get_price_data(symbol)
                await asyncio.sleep(0.1)  # Rate limiting
                
        except Exception as e:
            self.logger.error(f"Error initializing price data: {e}")
    
    async def handle_message(self, message: AgentMessage):
        """Handle incoming messages from other agents"""
        try:
            if message.message_type == 'request_technical_analysis':
                symbol = message.data.get('symbol')
                if symbol and symbol in self.analysis_cache:
                    await self.send_message(
                        message.agent_id,
                        'technical_analysis_response',
                        self.analysis_cache[symbol].to_dict(),
                        correlation_id=message.correlation_id
                    )
            
            elif message.message_type == 'request_support_resistance':
                symbol = message.data.get('symbol')
                cached_data = self.get_cached_data(f'support_resistance_{symbol}')
                if cached_data:
                    await self.send_message(
                        message.agent_id,
                        'support_resistance_response',
                        cached_data,
                        correlation_id=message.correlation_id
                    )
            
            elif message.message_type == 'add_symbol_coverage':
                symbol = message.data.get('symbol')
                name = message.data.get('name', symbol)
                if symbol:
                    self.coverage_symbols[symbol] = name
                    self.logger.info(f"Added {symbol} to technical analysis coverage")
                    
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
    
    async def agent_cleanup(self):
        """Cleanup agent resources"""
        self.analysis_cache.clear()
        self.price_data_cache.clear()
        self.logger.info("Technical Analysis Agent cleaned up successfully") 