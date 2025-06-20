"""
Fundamental Analysis Agent - Deep Company Analysis and Valuation
"""

import asyncio
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

from .base_agent import BaseAgent, AgentMessage, AgentSignal
from ..config import config


@dataclass
class FinancialMetrics:
    """Financial metrics for a company"""
    symbol: str
    revenue_growth: float
    gross_margin: float
    operating_margin: float
    net_margin: float
    roe: float  # Return on Equity
    roa: float  # Return on Assets
    debt_to_equity: float
    current_ratio: float
    quick_ratio: float
    free_cash_flow: float
    fcf_yield: float
    revenue_ttm: float
    earnings_ttm: float
    book_value_per_share: float
    timestamp: datetime


@dataclass
class ValuationMetrics:
    """Valuation metrics for a company"""
    symbol: str
    pe_ratio: float
    peg_ratio: float
    pb_ratio: float
    ps_ratio: float
    ev_ebitda: float
    enterprise_value: float
    market_cap: float
    fair_value_estimate: float
    upside_downside: float  # % upside/downside to fair value
    valuation_grade: str  # 'UNDERVALUED', 'FAIR', 'OVERVALUED'
    timestamp: datetime


@dataclass
class EarningsAnalysis:
    """Earnings and estimates analysis"""
    symbol: str
    next_earnings_date: Optional[datetime]
    eps_estimate_current: float
    eps_estimate_next: float
    eps_growth_current: float
    eps_growth_next: float
    revenue_estimate_current: float
    revenue_estimate_next: float
    estimate_revisions_up: int
    estimate_revisions_down: int
    surprise_history: List[float]  # Last 4 quarters
    guidance_direction: str  # 'RAISED', 'LOWERED', 'MAINTAINED'
    timestamp: datetime


@dataclass
class FundamentalSignal:
    """Fundamental analysis signal"""
    symbol: str
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    strength: float  # 0.0 to 1.0
    confidence: float
    rationale: str
    target_price: Optional[float]
    time_horizon: str  # 'SHORT', 'MEDIUM', 'LONG'
    key_factors: List[str]
    risks: List[str]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FundamentalAnalysis:
    """Complete fundamental analysis"""
    symbol: str
    company_name: str
    sector: str
    industry: str
    financial_metrics: FinancialMetrics
    valuation_metrics: ValuationMetrics
    earnings_analysis: EarningsAnalysis
    signals: List[FundamentalSignal]
    overall_score: float  # 0-100 composite score
    grade: str  # 'A', 'B', 'C', 'D', 'F'
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FundamentalAnalysisAgent(BaseAgent):
    """
    Fundamental Analysis Agent for deep company analysis
    """
    
    def __init__(self, agent_id: str = 'fundamental-agent', config_override: Optional[Dict] = None):
        super().__init__(agent_id, config_override)
        
        # Coverage universe
        default_symbols = {
            # Large Cap Technology
            'AAPL': 'Apple Inc.',
            'MSFT': 'Microsoft Corporation', 
            'GOOGL': 'Alphabet Inc.',
            'AMZN': 'Amazon.com Inc.',
            'META': 'Meta Platforms Inc.',
            'NVDA': 'NVIDIA Corporation',
            'TSLA': 'Tesla Inc.',
            
            # Large Cap Healthcare
            'JNJ': 'Johnson & Johnson',
            'PFE': 'Pfizer Inc.',
            'UNH': 'UnitedHealth Group',
            'ABBV': 'AbbVie Inc.',
            
            # Large Cap Financials
            'JPM': 'JPMorgan Chase & Co.',
            'BAC': 'Bank of America Corp',
            'WFC': 'Wells Fargo & Company',
            'GS': 'Goldman Sachs Group Inc.',
            
            # Large Cap Consumer
            'KO': 'Coca-Cola Company',
            'PG': 'Procter & Gamble Co.',
            'WMT': 'Walmart Inc.',
            'HD': 'Home Depot Inc.',
            
            # Large Cap Industrial
            'BA': 'Boeing Company',
            'CAT': 'Caterpillar Inc.',
            'GE': 'General Electric Company',
            'MMM': '3M Company'
        }
        
        # Use config override or default symbols
        self.coverage_symbols = self.config.get('fundamental_coverage', default_symbols) if hasattr(self.config, 'get') and self.config else default_symbols
        
        # Analysis parameters
        self.financial_health_weights = {
            'profitability': 0.25,
            'growth': 0.25,
            'financial_strength': 0.25,
            'valuation': 0.25
        }
        
        # Sector benchmarks (simplified)
        self.sector_benchmarks = {
            'Technology': {'pe': 25, 'roe': 0.20, 'margin': 0.25},
            'Healthcare': {'pe': 18, 'roe': 0.15, 'margin': 0.18},
            'Financials': {'pe': 12, 'roe': 0.12, 'margin': 0.30},
            'Consumer': {'pe': 20, 'roe': 0.18, 'margin': 0.15},
            'Industrial': {'pe': 16, 'roe': 0.14, 'margin': 0.12}
        }
        
        # Analysis cache
        self.analysis_cache = {}
        self.financial_data_cache = {}
        
    async def agent_initialize(self):
        """Initialize fundamental analysis agent"""
        try:
            # Load initial financial data
            await self._load_financial_data()
            
            self.logger.info("Fundamental Analysis Agent initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Fundamental Analysis Agent: {e}")
            raise
    
    async def execute(self):
        """Main execution logic - perform fundamental analysis"""
        try:
            # Analyze all covered symbols
            for symbol in self.coverage_symbols.keys():
                try:
                    analysis = await self._analyze_symbol(symbol)
                    if analysis:
                        # Cache analysis
                        self.analysis_cache[symbol] = analysis
                        
                        # Send signals to other agents
                        await self._send_fundamental_signals(analysis)
                        
                        # Cache fundamental data
                        self._cache_fundamental_data(symbol, analysis)
                        
                except Exception as e:
                    self.logger.error(f"Error analyzing {symbol}: {e}")
                    continue
            
            self.logger.info(f"Fundamental analysis completed for {len(self.coverage_symbols)} symbols")
            
        except Exception as e:
            self.logger.error(f"Error in fundamental analysis execution: {e}")
            self.metrics['errors'] += 1
    
    async def _analyze_symbol(self, symbol: str) -> Optional[FundamentalAnalysis]:
        """Perform comprehensive fundamental analysis for a symbol"""
        try:
            # Get company info
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            company_name = info.get('longName', symbol)
            sector = info.get('sector', 'Unknown')
            industry = info.get('industry', 'Unknown')
            
            # Calculate financial metrics
            financial_metrics = await self._calculate_financial_metrics(symbol, ticker, info)
            
            # Calculate valuation metrics
            valuation_metrics = await self._calculate_valuation_metrics(symbol, ticker, info)
            
            # Analyze earnings
            earnings_analysis = await self._analyze_earnings(symbol, ticker, info)
            
            # Generate signals
            signals = await self._generate_fundamental_signals(
                symbol, financial_metrics, valuation_metrics, earnings_analysis
            )
            
            # Calculate overall score and grade
            overall_score, grade = self._calculate_overall_score(
                financial_metrics, valuation_metrics, earnings_analysis
            )
            
            return FundamentalAnalysis(
                symbol=symbol,
                company_name=company_name,
                sector=sector,
                industry=industry,
                financial_metrics=financial_metrics,
                valuation_metrics=valuation_metrics,
                earnings_analysis=earnings_analysis,
                signals=signals,
                overall_score=overall_score,
                grade=grade,
                timestamp=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            self.logger.error(f"Error in fundamental analysis for {symbol}: {e}")
            return None
    
    async def _calculate_financial_metrics(self, symbol: str, ticker, info: Dict) -> FinancialMetrics:
        """Calculate key financial metrics"""
        try:
            # Get financial statements
            financials = ticker.financials
            balance_sheet = ticker.balance_sheet
            cashflow = ticker.cashflow
            
            # Revenue and growth
            if not financials.empty and 'Total Revenue' in financials.index:
                revenues = financials.loc['Total Revenue'].dropna()
                revenue_ttm = revenues.iloc[0] if len(revenues) > 0 else 0.0
                revenue_growth = ((revenues.iloc[0] - revenues.iloc[1]) / revenues.iloc[1]) if len(revenues) > 1 else 0.0
            else:
                revenue_ttm = info.get('totalRevenue', 0.0)
                revenue_growth = info.get('revenueGrowth', 0.0) or 0.0
            
            # Margins
            gross_margin = info.get('grossMargins', 0.0) or 0.0
            operating_margin = info.get('operatingMargins', 0.0) or 0.0
            net_margin = info.get('profitMargins', 0.0) or 0.0
            
            # Returns
            roe = info.get('returnOnEquity', 0.0) or 0.0
            roa = info.get('returnOnAssets', 0.0) or 0.0
            
            # Financial strength
            debt_to_equity = info.get('debtToEquity', 0.0) or 0.0
            current_ratio = info.get('currentRatio', 0.0) or 0.0
            quick_ratio = info.get('quickRatio', 0.0) or 0.0
            
            # Cash flow
            free_cash_flow = info.get('freeCashflow', 0.0) or 0.0
            market_cap = info.get('marketCap', 1.0) or 1.0
            fcf_yield = free_cash_flow / market_cap if market_cap > 0 else 0.0
            
            # Earnings
            earnings_ttm = info.get('trailingEps', 0.0) or 0.0
            earnings_ttm *= info.get('sharesOutstanding', 1.0) or 1.0
            
            # Book value
            book_value_per_share = info.get('bookValue', 0.0) or 0.0
            
            return FinancialMetrics(
                symbol=symbol,
                revenue_growth=float(revenue_growth),
                gross_margin=float(gross_margin),
                operating_margin=float(operating_margin),
                net_margin=float(net_margin),
                roe=float(roe),
                roa=float(roa),
                debt_to_equity=float(debt_to_equity / 100) if debt_to_equity > 10 else float(debt_to_equity),
                current_ratio=float(current_ratio),
                quick_ratio=float(quick_ratio),
                free_cash_flow=float(free_cash_flow),
                fcf_yield=float(fcf_yield),
                revenue_ttm=float(revenue_ttm),
                earnings_ttm=float(earnings_ttm),
                book_value_per_share=float(book_value_per_share),
                timestamp=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating financial metrics for {symbol}: {e}")
            return self._get_default_financial_metrics(symbol)
    
    async def _calculate_valuation_metrics(self, symbol: str, ticker, info: Dict) -> ValuationMetrics:
        """Calculate valuation metrics"""
        try:
            # Basic valuation ratios
            pe_ratio = info.get('trailingPE', 0.0) or 0.0
            peg_ratio = info.get('pegRatio', 0.0) or 0.0
            pb_ratio = info.get('priceToBook', 0.0) or 0.0
            ps_ratio = info.get('priceToSalesTrailing12Months', 0.0) or 0.0
            ev_ebitda = info.get('enterpriseToEbitda', 0.0) or 0.0
            
            # Market metrics
            enterprise_value = info.get('enterpriseValue', 0.0) or 0.0
            market_cap = info.get('marketCap', 0.0) or 0.0
            
            # Fair value estimation (simplified DCF approach)
            fair_value_estimate = await self._estimate_fair_value(symbol, info)
            current_price = info.get('currentPrice', 0.0) or info.get('regularMarketPrice', 0.0) or 0.0
            
            upside_downside = ((fair_value_estimate - current_price) / current_price) if current_price > 0 else 0.0
            
            # Valuation grade
            if upside_downside > 0.20:
                valuation_grade = 'UNDERVALUED'
            elif upside_downside < -0.20:
                valuation_grade = 'OVERVALUED'
            else:
                valuation_grade = 'FAIR'
            
            return ValuationMetrics(
                symbol=symbol,
                pe_ratio=float(pe_ratio),
                peg_ratio=float(peg_ratio),
                pb_ratio=float(pb_ratio),
                ps_ratio=float(ps_ratio),
                ev_ebitda=float(ev_ebitda),
                enterprise_value=float(enterprise_value),
                market_cap=float(market_cap),
                fair_value_estimate=float(fair_value_estimate),
                upside_downside=float(upside_downside),
                valuation_grade=valuation_grade,
                timestamp=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating valuation metrics for {symbol}: {e}")
            return self._get_default_valuation_metrics(symbol)
    
    async def _analyze_earnings(self, symbol: str, ticker, info: Dict) -> EarningsAnalysis:
        """Analyze earnings and estimates"""
        try:
            # Earnings calendar
            try:
                calendar = ticker.calendar
                next_earnings_date = calendar.index[0] if not calendar.empty else None
            except:
                next_earnings_date = None
            
            # Estimates (from yfinance info)
            eps_estimate_current = info.get('forwardEps', 0.0) or 0.0
            eps_growth_current = info.get('earningsGrowth', 0.0) or 0.0
            
            # Revenue estimates
            revenue_estimate_current = info.get('revenueGrowth', 0.0) or 0.0
            
            # Simplified estimates for next year (would use analyst data in production)
            eps_estimate_next = eps_estimate_current * (1 + eps_growth_current)
            eps_growth_next = eps_growth_current * 0.8  # Assume some deceleration
            revenue_estimate_next = revenue_estimate_current * 0.9
            
            # Earnings surprises (simplified)
            try:
                earnings_history = ticker.earnings_history
                surprise_history = []
                if not earnings_history.empty and 'Surprise(%)' in earnings_history.columns:
                    surprise_history = earnings_history['Surprise(%)'].tail(4).tolist()
            except:
                surprise_history = [0.0, 0.0, 0.0, 0.0]
            
            # Estimate revisions (simplified)
            estimate_revisions_up = 0
            estimate_revisions_down = 0
            
            # Guidance direction (simplified)
            guidance_direction = 'MAINTAINED'
            
            return EarningsAnalysis(
                symbol=symbol,
                next_earnings_date=next_earnings_date,
                eps_estimate_current=float(eps_estimate_current),
                eps_estimate_next=float(eps_estimate_next),
                eps_growth_current=float(eps_growth_current),
                eps_growth_next=float(eps_growth_next),
                revenue_estimate_current=float(revenue_estimate_current),
                revenue_estimate_next=float(revenue_estimate_next),
                estimate_revisions_up=estimate_revisions_up,
                estimate_revisions_down=estimate_revisions_down,
                surprise_history=surprise_history,
                guidance_direction=guidance_direction,
                timestamp=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing earnings for {symbol}: {e}")
            return self._get_default_earnings_analysis(symbol)
    
    async def _estimate_fair_value(self, symbol: str, info: Dict) -> float:
        """Estimate fair value using simplified DCF model"""
        try:
            # Get key inputs
            free_cash_flow = info.get('freeCashflow', 0.0) or 0.0
            revenue_growth = info.get('revenueGrowth', 0.05) or 0.05
            shares_outstanding = info.get('sharesOutstanding', 1.0) or 1.0
            
            if free_cash_flow <= 0 or shares_outstanding <= 0:
                # Fallback to P/E based valuation
                eps = info.get('trailingEps', 0.0) or 0.0
                sector_pe = self._get_sector_benchmark(info.get('sector', ''), 'pe')
                return eps * sector_pe
            
            # Simplified DCF
            # Assume growth declines linearly over 5 years to 3%
            terminal_growth = 0.03
            discount_rate = 0.10  # 10% WACC assumption
            
            # Project cash flows
            cash_flows = []
            current_fcf = free_cash_flow
            
            for year in range(1, 6):  # 5 year projection
                growth_rate = revenue_growth * (1 - (year-1) * 0.2)  # Declining growth
                growth_rate = max(growth_rate, terminal_growth)
                current_fcf *= (1 + growth_rate)
                pv_factor = (1 + discount_rate) ** year
                cash_flows.append(current_fcf / pv_factor)
            
            # Terminal value
            terminal_fcf = current_fcf * (1 + terminal_growth)
            terminal_value = terminal_fcf / (discount_rate - terminal_growth)
            terminal_pv = terminal_value / ((1 + discount_rate) ** 5)
            
            # Total value
            total_value = sum(cash_flows) + terminal_pv
            fair_value_per_share = total_value / shares_outstanding
            
            return fair_value_per_share
            
        except Exception as e:
            self.logger.error(f"Error estimating fair value for {symbol}: {e}")
            return 0.0
    
    def _get_sector_benchmark(self, sector: str, metric: str) -> float:
        """Get sector benchmark for comparison"""
        benchmarks = self.sector_benchmarks.get(sector, self.sector_benchmarks['Technology'])
        return benchmarks.get(metric, 20.0)  # Default P/E of 20
    
    async def _generate_fundamental_signals(self, symbol: str, financial_metrics: FinancialMetrics,
                                          valuation_metrics: ValuationMetrics, 
                                          earnings_analysis: EarningsAnalysis) -> List[FundamentalSignal]:
        """Generate fundamental trading signals"""
        signals = []
        
        try:
            # Valuation signal
            if valuation_metrics.upside_downside > 0.25:  # 25%+ upside
                signals.append(FundamentalSignal(
                    symbol=symbol,
                    signal_type='BUY',
                    strength=0.8,
                    confidence=0.75,
                    rationale=f"Undervalued by {valuation_metrics.upside_downside:.1%}",
                    target_price=valuation_metrics.fair_value_estimate,
                    time_horizon='LONG',
                    key_factors=['Valuation', 'Fair Value'],
                    risks=['Market Risk', 'Execution Risk'],
                    timestamp=datetime.now(timezone.utc)
                ))
            elif valuation_metrics.upside_downside < -0.25:  # 25%+ overvalued
                signals.append(FundamentalSignal(
                    symbol=symbol,
                    signal_type='SELL',
                    strength=0.7,
                    confidence=0.70,
                    rationale=f"Overvalued by {abs(valuation_metrics.upside_downside):.1%}",
                    target_price=valuation_metrics.fair_value_estimate,
                    time_horizon='MEDIUM',
                    key_factors=['Valuation', 'Mean Reversion'],
                    risks=['Growth Surprise', 'Multiple Expansion'],
                    timestamp=datetime.now(timezone.utc)
                ))
            
            # Growth signal
            if financial_metrics.revenue_growth > 0.20 and earnings_analysis.eps_growth_current > 0.15:
                signals.append(FundamentalSignal(
                    symbol=symbol,
                    signal_type='BUY',
                    strength=0.6,
                    confidence=0.65,
                    rationale=f"Strong growth: Revenue {financial_metrics.revenue_growth:.1%}, EPS {earnings_analysis.eps_growth_current:.1%}",
                    target_price=None,
                    time_horizon='MEDIUM',
                    key_factors=['Revenue Growth', 'Earnings Growth'],
                    risks=['Growth Deceleration', 'Competition'],
                    timestamp=datetime.now(timezone.utc)
                ))
            
            # Quality signal
            if (financial_metrics.roe > 0.15 and financial_metrics.operating_margin > 0.15 and 
                financial_metrics.debt_to_equity < 0.5):
                signals.append(FundamentalSignal(
                    symbol=symbol,
                    signal_type='BUY',
                    strength=0.5,
                    confidence=0.60,
                    rationale=f"High quality: ROE {financial_metrics.roe:.1%}, Strong balance sheet",
                    target_price=None,
                    time_horizon='LONG',
                    key_factors=['Profitability', 'Financial Strength'],
                    risks=['Industry Disruption', 'Economic Downturn'],
                    timestamp=datetime.now(timezone.utc)
                ))
            
            # Earnings momentum signal
            avg_surprise = np.mean(earnings_analysis.surprise_history) if earnings_analysis.surprise_history else 0.0
            if avg_surprise > 5.0:  # Consistent positive surprises
                signals.append(FundamentalSignal(
                    symbol=symbol,
                    signal_type='BUY',
                    strength=0.4,
                    confidence=0.55,
                    rationale=f"Consistent earnings beats: Avg surprise {avg_surprise:.1f}%",
                    target_price=None,
                    time_horizon='SHORT',
                    key_factors=['Earnings Momentum', 'Management Execution'],
                    risks=['High Expectations', 'Guidance Miss'],
                    timestamp=datetime.now(timezone.utc)
                ))
            
        except Exception as e:
            self.logger.error(f"Error generating fundamental signals for {symbol}: {e}")
        
        return signals
    
    def _calculate_overall_score(self, financial_metrics: FinancialMetrics,
                               valuation_metrics: ValuationMetrics,
                               earnings_analysis: EarningsAnalysis) -> Tuple[float, str]:
        """Calculate overall fundamental score and grade"""
        try:
            scores = {}
            
            # Profitability score (0-25)
            profit_score = 0.0
            profit_score += min(financial_metrics.roe * 100, 10)  # Up to 10 points for ROE
            profit_score += min(financial_metrics.operating_margin * 50, 10)  # Up to 10 points for margins
            profit_score += min(financial_metrics.fcf_yield * 100, 5)  # Up to 5 points for FCF yield
            scores['profitability'] = profit_score
            
            # Growth score (0-25)
            growth_score = 0.0
            growth_score += min(financial_metrics.revenue_growth * 50, 15)  # Up to 15 points for revenue growth
            growth_score += min(earnings_analysis.eps_growth_current * 50, 10)  # Up to 10 points for EPS growth
            scores['growth'] = growth_score
            
            # Financial strength score (0-25)
            strength_score = 15.0  # Base score
            if financial_metrics.debt_to_equity > 1.0:
                strength_score -= 5
            if financial_metrics.current_ratio < 1.0:
                strength_score -= 5
            if financial_metrics.free_cash_flow < 0:
                strength_score -= 10
            scores['financial_strength'] = max(strength_score, 0)
            
            # Valuation score (0-25)
            valuation_score = 12.5  # Neutral score
            if valuation_metrics.upside_downside > 0.30:
                valuation_score = 25
            elif valuation_metrics.upside_downside > 0.10:
                valuation_score = 20
            elif valuation_metrics.upside_downside < -0.30:
                valuation_score = 0
            elif valuation_metrics.upside_downside < -0.10:
                valuation_score = 5
            scores['valuation'] = valuation_score
            
            # Weighted overall score
            overall_score = sum(
                scores[category] * self.financial_health_weights[category]
                for category in scores
            )
            
            # Grade assignment
            if overall_score >= 80:
                grade = 'A'
            elif overall_score >= 70:
                grade = 'B'
            elif overall_score >= 60:
                grade = 'C'
            elif overall_score >= 50:
                grade = 'D'
            else:
                grade = 'F'
            
            return overall_score, grade
            
        except Exception as e:
            self.logger.error(f"Error calculating overall score: {e}")
            return 50.0, 'C'
    
    async def _send_fundamental_signals(self, analysis: FundamentalAnalysis):
        """Send fundamental signals to other agents"""
        try:
            for signal in analysis.signals:
                if signal.confidence > 0.6:  # Only send high-confidence signals
                    # Send to portfolio management
                    await self.send_message(
                        'portfolio-mgmt-agent',
                        'fundamental_signal',
                        signal.to_dict(),
                        priority=4  # Lower priority than technical signals
                    )
                    
                    # Send high-strength signals to risk management
                    if signal.strength > 0.7:
                        await self.send_message(
                            'risk-mgmt-agent',
                            'fundamental_signal',
                            signal.to_dict(),
                            priority=4
                        )
            
        except Exception as e:
            self.logger.error(f"Error sending fundamental signals: {e}")
    
    def _cache_fundamental_data(self, symbol: str, analysis: FundamentalAnalysis):
        """Cache fundamental analysis data"""
        try:
            # Cache full analysis
            self.cache_data(f'fundamental_analysis_{symbol}', analysis.to_dict(), expiry=3600)  # 1 hour
            
            # Cache key metrics
            self.cache_data(f'financial_metrics_{symbol}', asdict(analysis.financial_metrics), expiry=3600)
            self.cache_data(f'valuation_metrics_{symbol}', asdict(analysis.valuation_metrics), expiry=1800)
            self.cache_data(f'fundamental_score_{symbol}', analysis.overall_score, expiry=3600)
            
        except Exception as e:
            self.logger.error(f"Error caching fundamental data: {e}")
    
    def _get_default_financial_metrics(self, symbol: str) -> FinancialMetrics:
        """Get default financial metrics when calculation fails"""
        return FinancialMetrics(
            symbol=symbol,
            revenue_growth=0.0,
            gross_margin=0.0,
            operating_margin=0.0,
            net_margin=0.0,
            roe=0.0,
            roa=0.0,
            debt_to_equity=0.0,
            current_ratio=1.0,
            quick_ratio=1.0,
            free_cash_flow=0.0,
            fcf_yield=0.0,
            revenue_ttm=0.0,
            earnings_ttm=0.0,
            book_value_per_share=0.0,
            timestamp=datetime.now(timezone.utc)
        )
    
    def _get_default_valuation_metrics(self, symbol: str) -> ValuationMetrics:
        """Get default valuation metrics when calculation fails"""
        return ValuationMetrics(
            symbol=symbol,
            pe_ratio=0.0,
            peg_ratio=0.0,
            pb_ratio=0.0,
            ps_ratio=0.0,
            ev_ebitda=0.0,
            enterprise_value=0.0,
            market_cap=0.0,
            fair_value_estimate=0.0,
            upside_downside=0.0,
            valuation_grade='FAIR',
            timestamp=datetime.now(timezone.utc)
        )
    
    def _get_default_earnings_analysis(self, symbol: str) -> EarningsAnalysis:
        """Get default earnings analysis when calculation fails"""
        return EarningsAnalysis(
            symbol=symbol,
            next_earnings_date=None,
            eps_estimate_current=0.0,
            eps_estimate_next=0.0,
            eps_growth_current=0.0,
            eps_growth_next=0.0,
            revenue_estimate_current=0.0,
            revenue_estimate_next=0.0,
            estimate_revisions_up=0,
            estimate_revisions_down=0,
            surprise_history=[0.0, 0.0, 0.0, 0.0],
            guidance_direction='MAINTAINED',
            timestamp=datetime.now(timezone.utc)
        )
    
    async def _load_financial_data(self):
        """Load initial financial data for covered symbols"""
        try:
            for symbol in list(self.coverage_symbols.keys())[:5]:  # Start with first 5
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    self.financial_data_cache[symbol] = info
                    await asyncio.sleep(0.2)  # Rate limiting
                except Exception as e:
                    self.logger.warning(f"Failed to load data for {symbol}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error loading financial data: {e}")
    
    async def handle_message(self, message: AgentMessage):
        """Handle incoming messages from other agents"""
        try:
            if message.message_type == 'request_fundamental_analysis':
                symbol = message.data.get('symbol')
                if symbol and symbol in self.analysis_cache:
                    await self.send_message(
                        message.agent_id,
                        'fundamental_analysis_response',
                        self.analysis_cache[symbol].to_dict(),
                        correlation_id=message.correlation_id
                    )
            
            elif message.message_type == 'request_valuation':
                symbol = message.data.get('symbol')
                cached_data = self.get_cached_data(f'valuation_metrics_{symbol}')
                if cached_data:
                    await self.send_message(
                        message.agent_id,
                        'valuation_response',
                        cached_data,
                        correlation_id=message.correlation_id
                    )
            
            elif message.message_type == 'add_symbol_coverage':
                symbol = message.data.get('symbol')
                name = message.data.get('name', symbol)
                if symbol:
                    self.coverage_symbols[symbol] = name
                    self.logger.info(f"Added {symbol} to fundamental analysis coverage")
                    
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
    
    async def agent_cleanup(self):
        """Cleanup agent resources"""
        self.analysis_cache.clear()
        self.financial_data_cache.clear()
        self.logger.info("Fundamental Analysis Agent cleaned up successfully") 