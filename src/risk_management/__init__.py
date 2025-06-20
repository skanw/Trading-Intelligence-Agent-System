"""
TIAS Risk Management Framework

Production-grade risk management system for position sizing, stop-loss, and portfolio risk controls.
"""

from .risk_manager import RiskManager, PositionSizer, RiskMetrics
from .position_sizing import FixedFractionalSizer, VolatilityBasedSizer, KellySizer
from .risk_controls import StopLossManager, TakeProfitManager, DrawdownController

__all__ = [
    'RiskManager',
    'PositionSizer', 
    'RiskMetrics',
    'FixedFractionalSizer',
    'VolatilityBasedSizer',
    'KellySizer',
    'StopLossManager',
    'TakeProfitManager',
    'DrawdownController'
]

__version__ = '1.0.0' 