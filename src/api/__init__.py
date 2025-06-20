"""
TIAS API Framework

FastAPI-based REST API for trading system monitoring and control.
"""

from .trading_api import TradingAPI
from .dashboard_api import DashboardAPI

__all__ = [
    'TradingAPI',
    'DashboardAPI'
]

__version__ = '1.0.0' 