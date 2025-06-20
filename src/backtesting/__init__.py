"""
TIAS Backtesting Framework

Production-grade backtesting suite for trading signal validation and strategy optimization.
"""

from .data_loader import DataLoader, PriceData, SignalData
from .backtest_engine import BacktestEngine, BacktestResult
from .performance_metrics import PerformanceCalculator, PerformanceMetrics
from .portfolio import Portfolio, Position, Trade

__all__ = [
    'DataLoader',
    'PriceData',
    'SignalData',
    'BacktestEngine',
    'BacktestResult',
    'PerformanceCalculator',
    'PerformanceMetrics',
    'Portfolio',
    'Position',
    'Trade'
]

__version__ = '1.0.0' 